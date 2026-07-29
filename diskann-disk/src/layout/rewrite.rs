/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    fs::{self, File},
    path::Path,
    time::Instant,
};

use memmap2::{Mmap, MmapMut};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::data_model::{GraphHeader, GraphLayoutVersion};

use super::sidecar::{
    binding_digest, default_sidecar_path, digest_fingerprint, expected_index_len, io_error,
    read_index_header, validate_mapping, write_sidecar, LayoutError, Result, SidecarContents,
    StructuralFields,
};

const SIDECAR_FIXED_SIZE: u64 = 160;
const ALGORITHM: &str = "deterministic greedy outgoing-edge packing; lowest-ID ties; \
                         deterministic weighted inter-block ordering from the medoid";

/// Resource estimate for a graph-aware rewrite.
#[derive(Debug, Clone, Serialize)]
pub struct LayoutEstimate {
    pub output_bytes: u64,
    pub sidecar_bytes: u64,
    pub temporary_bytes: u64,
    pub algorithm_resident_bytes: u64,
    pub mapping_bytes: u64,
    pub assignment_bytes: u64,
    pub score_bytes: u64,
    pub bitset_bytes: u64,
    pub block_vector_bytes: u64,
    pub file_passes: u64,
    pub bytes_processed: u64,
}

/// Graph edge locality for one physical layout.
#[derive(Debug, Clone, Serialize)]
pub struct LayoutQuality {
    pub same_block_edge_fraction: f64,
    pub graph_edge_block_distance_mean: f64,
    pub graph_edge_block_distance_p50: u64,
    pub graph_edge_block_distance_p95: u64,
}

/// Wall-clock phase durations in milliseconds.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PhaseTimings {
    pub validation_ms: u64,
    pub packing_ms: u64,
    pub block_ordering_ms: u64,
    pub quality_ms: u64,
    pub output_write_ms: u64,
    pub sidecar_write_ms: u64,
    pub total_ms: u64,
}

/// Serializable rewrite result and locality report.
#[derive(Debug, Clone, Serialize)]
pub struct LayoutReport {
    pub source_index: String,
    pub output_index: String,
    pub sidecar: String,
    pub source_size: u64,
    pub output_size: u64,
    pub num_points: u64,
    pub num_edges: u64,
    pub num_blocks: u64,
    pub block_size: u64,
    pub node_length: u64,
    pub capacity: u64,
    pub occupancy: f64,
    pub mapping_bytes: u64,
    pub deterministic_algorithm: String,
    pub before: LayoutQuality,
    pub after: LayoutQuality,
    pub estimate: LayoutEstimate,
    pub timings: PhaseTimings,
}

struct MappedIndex {
    mmap: Mmap,
    header: GraphHeader,
    fields: StructuralFields,
    vector_bytes: usize,
    record_stride: u64,
}

impl MappedIndex {
    fn open(path: &Path) -> Result<Self> {
        let (header, index_len) = read_index_header(path)?;
        let version = header.layout_version();
        if version == &GraphHeader::GRAPH_AWARE_LAYOUT_VERSION {
            return Err(LayoutError::Invalid(
                "source index is already graph layout 1.1".into(),
            ));
        }
        if version != &GraphLayoutVersion::new(0, 0)
            && version != &GraphHeader::CURRENT_LAYOUT_VERSION
        {
            return Err(LayoutError::Invalid(format!(
                "unsupported source graph layout version {version}"
            )));
        }
        if header.metadata().layout_fingerprint().is_some() {
            return Err(LayoutError::Invalid(
                "legacy source index has a non-zero layout fingerprint".into(),
            ));
        }
        let fields = StructuralFields::from_header(&header, index_len)?;
        validate_fields(&header, &fields)?;

        let file = File::open(path).map_err(|error| io_error(path, error))?;
        // SAFETY: The file is opened read-only and the mapping is never mutated.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|error| io_error(path, error))?;
        let record_stride = if fields.nodes_per_block == 0 {
            fields
                .node_len
                .div_ceil(fields.block_size)
                .checked_mul(fields.block_size)
                .ok_or_else(|| LayoutError::Invalid("record stride overflows u64".into()))?
        } else {
            fields.node_len
        };
        let mut index = Self {
            mmap,
            header,
            fields,
            vector_bytes: 0,
            record_stride,
        };
        index.vector_bytes = index.infer_vector_bytes()?;
        Ok(index)
    }

    fn record_range(&self, slot: u32) -> Result<std::ops::Range<usize>> {
        let slot = slot as u64;
        let start =
            if self.fields.nodes_per_block > 0 {
                let block = slot / self.fields.nodes_per_block;
                let in_block = slot % self.fields.nodes_per_block;
                self.fields
                    .block_size
                    .checked_add(block.checked_mul(self.fields.block_size).ok_or_else(|| {
                        LayoutError::Invalid("record offset overflows u64".into())
                    })?)
                    .and_then(|offset| {
                        in_block
                            .checked_mul(self.fields.node_len)
                            .and_then(|within| offset.checked_add(within))
                    })
            } else {
                slot.checked_mul(self.record_stride)
                    .and_then(|offset| self.fields.block_size.checked_add(offset))
            }
            .ok_or_else(|| LayoutError::Invalid("record offset overflows u64".into()))?;
        let end = start
            .checked_add(self.fields.node_len)
            .ok_or_else(|| LayoutError::Invalid("record end overflows u64".into()))?;
        if end > self.fields.index_len {
            return Err(LayoutError::Invalid(format!(
                "record slot {slot} extends beyond the index"
            )));
        }
        Ok(start as usize..end as usize)
    }

    fn infer_vector_bytes(&self) -> Result<usize> {
        for element_size in [4usize, 2, 1, 8] {
            let Some(vector_bytes) = self.header.metadata().dims.checked_mul(element_size) else {
                continue;
            };
            if self.validate_adjacency_offset(vector_bytes).is_ok() {
                return Ok(vector_bytes);
            }
        }
        Err(LayoutError::Invalid(
            "cannot infer vector element width from valid adjacency records".into(),
        ))
    }

    fn validate_adjacency_offset(&self, vector_bytes: usize) -> Result<()> {
        let usable = (self.fields.node_len as usize)
            .checked_sub(self.header.metadata().associated_data_length)
            .ok_or_else(|| LayoutError::Invalid("associated data exceeds node length".into()))?;
        let adjacency_bytes = usable
            .checked_sub(vector_bytes)
            .ok_or_else(|| LayoutError::Invalid("vector data exceeds node record".into()))?;
        if adjacency_bytes < 4 || !adjacency_bytes.is_multiple_of(4) {
            return Err(LayoutError::Invalid(
                "node adjacency region is not u32-aligned".into(),
            ));
        }
        let max_neighbors = adjacency_bytes / 4 - 1;
        for logical in 0..self.fields.num_points as u32 {
            let record = &self.mmap[self.record_range(logical)?];
            let count = read_u32(record, vector_bytes)? as usize;
            if count > max_neighbors {
                return Err(LayoutError::Invalid(format!(
                    "node {logical} neighbor count {count} exceeds capacity {max_neighbors}"
                )));
            }
            for neighbor_index in 0..count {
                let neighbor = read_u32(record, vector_bytes + 4 + neighbor_index * 4)?;
                if neighbor as u64 >= self.fields.num_points {
                    return Err(LayoutError::Invalid(format!(
                        "node {logical} has out-of-range neighbor {neighbor}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn neighbors(&self, logical: u32) -> Result<Neighbors<'_>> {
        let record = &self.mmap[self.record_range(logical)?];
        let count = read_u32(record, self.vector_bytes)? as usize;
        Ok(Neighbors {
            bytes: record,
            offset: self.vector_bytes + 4,
            count,
            current: 0,
        })
    }
}

struct Neighbors<'a> {
    bytes: &'a [u8],
    offset: usize,
    count: usize,
    current: usize,
}

impl Iterator for Neighbors<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.count {
            return None;
        }
        let value = read_u32(self.bytes, self.offset + self.current * 4).ok()?;
        self.current += 1;
        Some(value)
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| LayoutError::Invalid("adjacency record is truncated".into()))?;
    Ok(u32::from_le_bytes(
        value.try_into().expect("four-byte adjacency value"),
    ))
}

fn validate_fields(header: &GraphHeader, fields: &StructuralFields) -> Result<()> {
    if fields.num_points == 0 || fields.num_points > u32::MAX as u64 {
        return Err(LayoutError::Invalid(
            "point count must be in 1..=u32::MAX".into(),
        ));
    }
    if fields.medoid >= fields.num_points {
        return Err(LayoutError::Invalid("medoid is out of range".into()));
    }
    if fields.block_size < (8 + GraphHeader::get_size()) as u64 {
        return Err(LayoutError::Invalid(
            "block is too small for the graph header".into(),
        ));
    }
    let expected = expected_index_len(fields)?;
    if expected != fields.index_len || header.metadata().disk_index_file_size != fields.index_len {
        return Err(LayoutError::Invalid(format!(
            "index length {} does not match header geometry {expected}",
            fields.index_len
        )));
    }
    if header.metadata().associated_data_length as u64 >= fields.node_len {
        return Err(LayoutError::Invalid(
            "associated data leaves no room for graph data".into(),
        ));
    }
    Ok(())
}

/// Estimate output and resident-memory requirements without creating outputs.
pub fn estimate_graph_aware_layout(source_index: impl AsRef<Path>) -> Result<LayoutEstimate> {
    let source_index = source_index.as_ref();
    let (header, index_len) = read_index_header(source_index)?;
    if header.layout_version() == &GraphHeader::GRAPH_AWARE_LAYOUT_VERSION {
        return Err(LayoutError::Invalid(
            "source index is already graph layout 1.1".into(),
        ));
    }
    if header.layout_version() != &GraphLayoutVersion::new(0, 0)
        && header.layout_version() != &GraphHeader::CURRENT_LAYOUT_VERSION
    {
        return Err(LayoutError::Invalid(format!(
            "unsupported source graph layout version {}",
            header.layout_version()
        )));
    }
    if header.metadata().layout_fingerprint().is_some() {
        return Err(LayoutError::Invalid(
            "legacy source index has a non-zero layout fingerprint".into(),
        ));
    }
    let fields = StructuralFields::from_header(&header, index_len)?;
    validate_fields(&header, &fields)?;
    estimate(&fields)
}

fn estimate(fields: &StructuralFields) -> Result<LayoutEstimate> {
    let points = fields.num_points;
    let blocks = if fields.nodes_per_block > 0 {
        points.div_ceil(fields.nodes_per_block)
    } else {
        points
    };
    let physical_blocks = if fields.nodes_per_block > 0 {
        blocks
    } else {
        points
            .checked_mul(fields.node_len.div_ceil(fields.block_size))
            .ok_or_else(|| LayoutError::Invalid("physical-block estimate overflows u64".into()))?
    };
    let mapping_bytes = points
        .checked_mul(4)
        .ok_or_else(|| LayoutError::Invalid("mapping estimate overflows u64".into()))?;
    let assignment_bytes = points;
    let score_bytes = mapping_bytes;
    let bitset_bytes = points.div_ceil(8);
    let usize_bytes = size_of::<usize>() as u64;
    let per_node_block_bytes = points
        .checked_mul(4 + usize_bytes)
        .ok_or_else(|| LayoutError::Invalid("block-vector estimate overflows u64".into()))?;
    let per_block_bytes = 24 + 2 * usize_bytes + 5;
    let block_vector_bytes =
        per_node_block_bytes
            .checked_add(blocks.checked_mul(per_block_bytes).ok_or_else(|| {
                LayoutError::Invalid("block-vector estimate overflows u64".into())
            })?)
            .and_then(|value| {
                physical_blocks
                    .checked_mul(8)
                    .and_then(|histogram| value.checked_add(histogram))
            })
            .ok_or_else(|| LayoutError::Invalid("block-vector estimate overflows u64".into()))?;
    let algorithm_resident_bytes = mapping_bytes
        .checked_add(assignment_bytes)
        .and_then(|value| value.checked_add(score_bytes))
        .and_then(|value| value.checked_add(bitset_bytes))
        .and_then(|value| value.checked_add(block_vector_bytes))
        .ok_or_else(|| LayoutError::Invalid("resident-memory estimate overflows u64".into()))?;
    let sidecar_bytes = SIDECAR_FIXED_SIZE
        .checked_add(mapping_bytes)
        .ok_or_else(|| LayoutError::Invalid("sidecar estimate overflows u64".into()))?;
    let file_passes = 7;
    let bytes_processed = fields
        .index_len
        .checked_mul(file_passes)
        .and_then(|value| value.checked_add(sidecar_bytes))
        .ok_or_else(|| LayoutError::Invalid("processed-byte estimate overflows u64".into()))?;
    Ok(LayoutEstimate {
        output_bytes: fields.index_len,
        sidecar_bytes,
        temporary_bytes: 0,
        algorithm_resident_bytes,
        mapping_bytes,
        assignment_bytes,
        score_bytes,
        bitset_bytes,
        block_vector_bytes,
        file_passes,
        bytes_processed,
    })
}

fn pack_nodes(index: &MappedIndex, capacity: usize) -> Result<Vec<Vec<u32>>> {
    let count = index.fields.num_points as usize;
    let mut assigned = vec![false; count];
    let mut scores = vec![0u32; count];
    let mut next_lowest = 0usize;
    let mut blocks = Vec::with_capacity(count.div_ceil(capacity));
    let mut assigned_count = 0usize;

    while assigned_count < count {
        while assigned[next_lowest] {
            next_lowest += 1;
        }
        let starter = next_lowest as u32;
        let mut block = Vec::with_capacity(capacity);
        let mut frontier = BinaryHeap::<(u32, Reverse<u32>)>::new();
        let mut touched = Vec::new();
        let mut candidate = starter;

        while block.len() < capacity && assigned_count < count {
            if assigned[candidate as usize] {
                return Err(LayoutError::Invalid(
                    "packing selected an assigned node".into(),
                ));
            }
            assigned[candidate as usize] = true;
            assigned_count += 1;
            block.push(candidate);
            while next_lowest < count && assigned[next_lowest] {
                next_lowest += 1;
            }
            for neighbor in index.neighbors(candidate)? {
                let neighbor = neighbor as usize;
                if assigned[neighbor] {
                    continue;
                }
                if scores[neighbor] == 0 {
                    touched.push(neighbor);
                }
                scores[neighbor] = scores[neighbor].saturating_add(1);
                frontier.push((scores[neighbor], Reverse(neighbor as u32)));
            }
            candidate = loop {
                match frontier.pop() {
                    Some((score, Reverse(node)))
                        if !assigned[node as usize] && scores[node as usize] == score =>
                    {
                        break node;
                    }
                    Some(_) => continue,
                    None => {
                        if assigned_count == count || block.len() == capacity {
                            break 0;
                        }
                        break next_lowest as u32;
                    }
                }
            };
        }
        for node in touched {
            scores[node] = 0;
        }
        blocks.push(block);
    }
    Ok(blocks)
}

fn order_blocks(index: &MappedIndex, blocks: &[Vec<u32>]) -> Result<Vec<usize>> {
    let mut node_block = vec![0usize; index.fields.num_points as usize];
    for (block_id, block) in blocks.iter().enumerate() {
        for &node in block {
            node_block[node as usize] = block_id;
        }
    }
    let mut placed = vec![false; blocks.len()];
    let mut order = Vec::with_capacity(blocks.len());
    let mut current = node_block[index.fields.medoid as usize];
    let mut lowest = 0usize;
    let mut weights = vec![0u32; blocks.len()];
    let mut touched = Vec::new();

    while order.len() < blocks.len() {
        if placed[current] {
            return Err(LayoutError::Invalid(
                "block ordering selected a placed block".into(),
            ));
        }
        placed[current] = true;
        order.push(current);
        while lowest < blocks.len() && placed[lowest] {
            lowest += 1;
        }
        if order.len() == blocks.len() {
            break;
        }
        for &node in &blocks[current] {
            for neighbor in index.neighbors(node)? {
                let target = node_block[neighbor as usize];
                if !placed[target] && target != current {
                    if weights[target] == 0 {
                        touched.push(target);
                    }
                    weights[target] = weights[target].saturating_add(1);
                }
            }
        }
        let next = touched
            .iter()
            .copied()
            .max_by(|&left, &right| {
                weights[left]
                    .cmp(&weights[right])
                    .then_with(|| right.cmp(&left))
            })
            .unwrap_or(lowest);
        for &block in &touched {
            weights[block] = 0;
        }
        touched.clear();
        current = next;
    }
    Ok(order)
}

fn make_mapping(blocks: &[Vec<u32>], order: &[usize], count: u64) -> Result<Vec<u32>> {
    let mut mapping = vec![0u32; count as usize];
    let mut physical = 0u32;
    for &block in order {
        for &logical in &blocks[block] {
            mapping[logical as usize] = physical;
            physical += 1;
        }
    }
    validate_mapping(&mapping, count)?;
    Ok(mapping)
}

fn quality(index: &MappedIndex, mapping: &[u32], capacity: u64) -> Result<(LayoutQuality, u64)> {
    let mut same_block = 0u64;
    let mut edge_count = 0u64;
    let mut distance_sum = 0u128;
    let blocks_per_node = index.fields.node_len.div_ceil(index.fields.block_size);
    let physical_blocks = if index.fields.nodes_per_block > 0 {
        index.fields.num_points.div_ceil(capacity)
    } else {
        index.fields.num_points * blocks_per_node
    };
    let mut distance_histogram = vec![0u64; physical_blocks as usize];
    for logical in 0..index.fields.num_points as u32 {
        let source_block = if index.fields.nodes_per_block > 0 {
            mapping[logical as usize] as u64 / capacity
        } else {
            mapping[logical as usize] as u64 * blocks_per_node
        };
        for neighbor in index.neighbors(logical)? {
            let target_block = if index.fields.nodes_per_block > 0 {
                mapping[neighbor as usize] as u64 / capacity
            } else {
                mapping[neighbor as usize] as u64 * blocks_per_node
            };
            let distance = source_block.abs_diff(target_block);
            same_block += u64::from(distance == 0);
            edge_count += 1;
            distance_sum += distance as u128;
            distance_histogram[distance as usize] += 1;
        }
    }
    let percentile = |percent: usize| {
        if edge_count == 0 {
            0
        } else {
            let rank = (edge_count as usize * percent).div_ceil(100);
            let mut cumulative = 0usize;
            distance_histogram
                .iter()
                .position(|count| {
                    cumulative += *count as usize;
                    cumulative >= rank
                })
                .unwrap_or(0) as u64
        }
    };
    Ok((
        LayoutQuality {
            same_block_edge_fraction: if edge_count == 0 {
                0.0
            } else {
                same_block as f64 / edge_count as f64
            },
            graph_edge_block_distance_mean: if edge_count == 0 {
                0.0
            } else {
                distance_sum as f64 / edge_count as f64
            },
            graph_edge_block_distance_p50: percentile(50),
            graph_edge_block_distance_p95: percentile(95),
        },
        edge_count,
    ))
}

fn source_digest(mmap: &[u8]) -> [u8; 32] {
    Sha256::digest(mmap).into()
}

/// Rewrite to graph layout 1.1 using the default `<output>.layout` sidecar.
pub fn rewrite_graph_aware_layout(
    source_index: impl AsRef<Path>,
    output_index: impl AsRef<Path>,
) -> Result<LayoutReport> {
    let sidecar = default_sidecar_path(output_index.as_ref());
    rewrite_graph_aware_layout_to(source_index, output_index, sidecar)
}

/// Rewrite to graph layout 1.1 using an explicit sidecar path.
pub fn rewrite_graph_aware_layout_to(
    source_index: impl AsRef<Path>,
    output_index: impl AsRef<Path>,
    sidecar_path: impl AsRef<Path>,
) -> Result<LayoutReport> {
    let started = Instant::now();
    let source_index = source_index.as_ref();
    let output_index = output_index.as_ref();
    let sidecar_path = sidecar_path.as_ref();
    for path in [output_index, sidecar_path] {
        if path.exists() {
            return Err(LayoutError::AlreadyExists(path.to_path_buf()));
        }
    }
    if source_index == output_index || source_index == sidecar_path || output_index == sidecar_path
    {
        return Err(LayoutError::Invalid(
            "source, output, and sidecar paths must be distinct".into(),
        ));
    }

    let phase = Instant::now();
    let index = MappedIndex::open(source_index)?;
    let digest = source_digest(&index.mmap);
    let estimate = estimate(&index.fields)?;
    let mut timings = PhaseTimings {
        validation_ms: elapsed_ms(phase),
        ..PhaseTimings::default()
    };
    let capacity = index.fields.nodes_per_block.max(1);

    let phase = Instant::now();
    let blocks = pack_nodes(&index, capacity as usize)?;
    timings.packing_ms = elapsed_ms(phase);

    let phase = Instant::now();
    let block_order = order_blocks(&index, &blocks)?;
    let mapping = make_mapping(&blocks, &block_order, index.fields.num_points)?;
    timings.block_ordering_ms = elapsed_ms(phase);

    let phase = Instant::now();
    let identity: Vec<u32> = (0..index.fields.num_points as u32).collect();
    let (before, num_edges) = quality(&index, &identity, capacity)?;
    let (after, _) = quality(&index, &mapping, capacity)?;
    timings.quality_ms = elapsed_ms(phase);

    let binding = binding_digest(&digest, &index.fields, &mapping);
    let fingerprint = digest_fingerprint(&binding)?;
    let result = write_outputs(
        &index,
        output_index,
        sidecar_path,
        &mapping,
        &digest,
        &binding,
        fingerprint,
        &mut timings,
    );
    if let Err(error) = result {
        let _ = fs::remove_file(output_index);
        let _ = fs::remove_file(sidecar_path);
        return Err(error);
    }

    let num_blocks = if index.fields.nodes_per_block > 0 {
        index.fields.num_points.div_ceil(capacity)
    } else {
        index
            .fields
            .num_points
            .checked_mul(index.fields.node_len.div_ceil(index.fields.block_size))
            .ok_or_else(|| LayoutError::Invalid("report block count overflows u64".into()))?
    };
    let occupancy = if index.fields.nodes_per_block > 0 {
        index.fields.num_points as f64 / num_blocks as f64 / capacity as f64
    } else {
        index.fields.num_points as f64 * index.fields.node_len as f64
            / num_blocks as f64
            / index.fields.block_size as f64
    };
    timings.total_ms = elapsed_ms(started);
    Ok(LayoutReport {
        source_index: source_index.display().to_string(),
        output_index: output_index.display().to_string(),
        sidecar: sidecar_path.display().to_string(),
        source_size: index.fields.index_len,
        output_size: fs::metadata(output_index)
            .map_err(|error| io_error(output_index, error))?
            .len(),
        num_points: index.fields.num_points,
        num_edges,
        num_blocks,
        block_size: index.fields.block_size,
        node_length: index.fields.node_len,
        capacity,
        occupancy,
        mapping_bytes: index.fields.num_points * 4,
        deterministic_algorithm: ALGORITHM.into(),
        before,
        after,
        estimate,
        timings,
    })
}

#[allow(clippy::too_many_arguments)]
fn write_outputs(
    index: &MappedIndex,
    output_index: &Path,
    sidecar_path: &Path,
    mapping: &[u32],
    source_digest: &[u8; 32],
    binding: &[u8; 32],
    fingerprint: crate::data_model::LayoutFingerprint,
    timings: &mut PhaseTimings,
) -> Result<()> {
    let phase = Instant::now();
    let output = File::options()
        .read(true)
        .write(true)
        .create_new(true)
        .open(output_index)
        .map_err(|error| io_error(output_index, error))?;
    output
        .set_len(index.fields.index_len)
        .map_err(|error| io_error(output_index, error))?;
    // SAFETY: This function exclusively owns the newly created output file and mapping.
    let mut output_map =
        unsafe { MmapMut::map_mut(&output) }.map_err(|error| io_error(output_index, error))?;
    output_map.fill(0);
    output_map[..index.fields.block_size as usize]
        .copy_from_slice(&index.mmap[..index.fields.block_size as usize]);
    for (logical, &physical) in mapping.iter().enumerate() {
        let source = index.record_range(logical as u32)?;
        let target = index.record_range(physical)?;
        output_map[target].copy_from_slice(&index.mmap[source]);
    }
    let mut metadata = index.header.metadata().clone();
    metadata.set_layout_fingerprint(fingerprint);
    let header = GraphHeader::new(
        metadata,
        index.fields.block_size,
        GraphHeader::GRAPH_AWARE_LAYOUT_VERSION,
    );
    let header_bytes = header
        .to_bytes()
        .map_err(|error| LayoutError::Invalid(format!("cannot serialize graph header: {error}")))?;
    output_map[8..8 + GraphHeader::get_size()].copy_from_slice(&header_bytes);
    output_map
        .flush()
        .map_err(|error| io_error(output_index, error))?;
    drop(output_map);
    drop(output);
    timings.output_write_ms = elapsed_ms(phase);

    let phase = Instant::now();
    write_sidecar(
        sidecar_path,
        SidecarContents {
            fields: &index.fields,
            source_digest,
            binding_digest: binding,
            fingerprint,
            mapping,
        },
    )?;
    super::sidecar::load_physical_layout(output_index, Some(sidecar_path))?;
    timings.sidecar_write_ms = elapsed_ms(phase);
    Ok(())
}

fn elapsed_ms(start: Instant) -> u64 {
    start.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::io::{Seek, SeekFrom, Write};

    use tempfile::tempdir;

    use super::*;
    use crate::{
        data_model::{GraphMetadata, LayoutFingerprint},
        layout::load_physical_layout,
    };

    const BLOCK_SIZE: u64 = 256;
    const NODE_LEN: u64 = 32;

    fn fixture(path: &Path, adjacency: &[&[u32]]) {
        fixture_with_geometry(path, adjacency, BLOCK_SIZE, NODE_LEN, BLOCK_SIZE / NODE_LEN);
    }

    fn fixture_with_geometry(
        path: &Path,
        adjacency: &[&[u32]],
        block_size: u64,
        node_len: u64,
        nodes_per_block: u64,
    ) {
        let points = adjacency.len() as u64;
        let data_blocks = if nodes_per_block > 0 {
            points.div_ceil(nodes_per_block)
        } else {
            points * node_len.div_ceil(block_size)
        };
        let len = (data_blocks + 1) * block_size;
        let metadata = GraphMetadata::new(points, 2, 0, node_len, nodes_per_block, 0, 0, len, 0);
        let header = GraphHeader::new(metadata, block_size, GraphHeader::CURRENT_LAYOUT_VERSION);
        let mut file = File::create(path).unwrap();
        file.set_len(len).unwrap();
        file.seek(SeekFrom::Start(8)).unwrap();
        file.write_all(&header.to_bytes().unwrap()).unwrap();
        for (logical, neighbors) in adjacency.iter().enumerate() {
            let offset = if nodes_per_block > 0 {
                let block = logical as u64 / nodes_per_block;
                let within = logical as u64 % nodes_per_block;
                block_size + block * block_size + within * node_len
            } else {
                block_size + logical as u64 * node_len.div_ceil(block_size) * block_size
            };
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(&(logical as f32).to_le_bytes()).unwrap();
            file.write_all(&(-(logical as f32)).to_le_bytes()).unwrap();
            file.write_all(&(neighbors.len() as u32).to_le_bytes())
                .unwrap();
            for &neighbor in *neighbors {
                file.write_all(&neighbor.to_le_bytes()).unwrap();
            }
        }
    }

    #[test]
    fn packing_and_mapping_are_deterministic_bijections() {
        let directory = tempdir().unwrap();
        let source = directory.path().join("source.index");
        fixture(&source, &[&[1, 2], &[2, 3], &[0], &[4], &[5], &[3], &[0]]);
        let index = MappedIndex::open(&source).unwrap();
        let first_blocks = pack_nodes(&index, 3).unwrap();
        let second_blocks = pack_nodes(&index, 3).unwrap();
        assert_eq!(first_blocks, second_blocks);
        assert_eq!(first_blocks[0], vec![0, 1, 2]);
        let order = order_blocks(&index, &first_blocks).unwrap();
        let mapping = make_mapping(&first_blocks, &order, 7).unwrap();
        validate_mapping(&mapping, 7).unwrap();
    }

    #[test]
    fn block_ordering_uses_weights_ties_and_lowest_restart() {
        let directory = tempdir().unwrap();
        let source = directory.path().join("source.index");
        fixture(&source, &[&[3, 2], &[3], &[], &[2, 5], &[], &[]]);
        let index = MappedIndex::open(&source).unwrap();
        let blocks = vec![vec![0, 1], vec![2], vec![3], vec![4], vec![5]];

        assert_eq!(order_blocks(&index, &blocks).unwrap(), vec![0, 2, 1, 3, 4]);
    }

    #[test]
    fn rewrite_preserves_every_record_byte_and_sets_fingerprint() {
        let directory = tempdir().unwrap();
        let source = directory.path().join("source.index");
        let output = directory.path().join("output.index");
        fixture(&source, &[&[2], &[0, 2], &[1], &[2], &[0]]);

        rewrite_graph_aware_layout(&source, &output).unwrap();
        let layout = load_physical_layout(&output, None).unwrap().unwrap();
        let source_index = MappedIndex::open(&source).unwrap();
        let output_bytes = fs::read(&output).unwrap();
        for logical in 0..5u32 {
            let source_range = source_index.record_range(logical).unwrap();
            let output_range = source_index
                .record_range(layout.physical_slot(logical).unwrap())
                .unwrap();
            assert_eq!(
                &source_index.mmap[source_range],
                &output_bytes[output_range]
            );
        }
        let (header, _) = read_index_header(&output).unwrap();
        assert_eq!(
            header.layout_version(),
            &GraphHeader::GRAPH_AWARE_LAYOUT_VERSION
        );
        assert!(header.metadata().layout_fingerprint().is_some());
    }

    #[test]
    fn quality_calculation_uses_physical_blocks() {
        let directory = tempdir().unwrap();
        let source = directory.path().join("source.index");
        fixture(&source, &[&[1], &[0], &[3], &[2]]);
        let index = MappedIndex::open(&source).unwrap();
        let (quality, edges) = quality(&index, &[0, 1, 2, 3], 2).unwrap();
        assert_eq!(edges, 4);
        assert_eq!(quality.same_block_edge_fraction, 1.0);
        assert_eq!(quality.graph_edge_block_distance_mean, 0.0);
    }

    #[test]
    fn rewrite_rejects_graph_aware_source() {
        let directory = tempdir().unwrap();
        let source = directory.path().join("source.index");
        fixture(&source, &[&[]]);
        let (header, _) = read_index_header(&source).unwrap();
        let mut metadata = header.metadata().clone();
        metadata.set_layout_fingerprint(LayoutFingerprint::new(7).unwrap());
        let header = GraphHeader::new(
            metadata,
            BLOCK_SIZE,
            GraphHeader::GRAPH_AWARE_LAYOUT_VERSION,
        );
        let mut file = File::options().write(true).open(&source).unwrap();
        file.seek(SeekFrom::Start(8)).unwrap();
        file.write_all(&header.to_bytes().unwrap()).unwrap();

        let output = directory.path().join("output.index");
        assert!(rewrite_graph_aware_layout(&source, &output).is_err());
        assert!(!output.exists());
    }

    #[test]
    fn malformed_and_wrong_sidecars_are_rejected() {
        let directory = tempdir().unwrap();
        let source_a = directory.path().join("source-a.index");
        let source_b = directory.path().join("source-b.index");
        let output_a = directory.path().join("output-a.index");
        let output_b = directory.path().join("output-b.index");
        fixture(&source_a, &[&[1], &[2], &[0]]);
        fixture(&source_b, &[&[2], &[0], &[1]]);
        rewrite_graph_aware_layout(&source_a, &output_a).unwrap();
        rewrite_graph_aware_layout(&source_b, &output_b).unwrap();

        let sidecar_a = default_sidecar_path(&output_a);
        let sidecar_b = default_sidecar_path(&output_b);
        assert!(load_physical_layout(&output_a, Some(&sidecar_b)).is_err());

        let mut bytes = fs::read(&sidecar_a).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        fs::write(&sidecar_a, bytes).unwrap();
        assert!(load_physical_layout(&output_a, Some(&sidecar_a)).is_err());
    }

    #[test]
    fn rewrite_supports_multi_block_records() {
        let directory = tempdir().unwrap();
        let source = directory.path().join("source.index");
        let output = directory.path().join("output.index");
        fixture_with_geometry(&source, &[&[1], &[2], &[0]], 128, 160, 0);

        rewrite_graph_aware_layout(&source, &output).unwrap();
        let layout = load_physical_layout(&output, None).unwrap().unwrap();
        let source_index = MappedIndex::open(&source).unwrap();
        let output_bytes = fs::read(output).unwrap();
        for logical in 0..3 {
            let source_range = source_index.record_range(logical).unwrap();
            let output_range = source_index
                .record_range(layout.physical_slot(logical).unwrap())
                .unwrap();
            assert_eq!(
                &source_index.mmap[source_range],
                &output_bytes[output_range]
            );
        }
    }
}
