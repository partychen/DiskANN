/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Query-aware entry points for disk graph search.

use std::{
    collections::HashSet,
    io::{Cursor, Read, Write},
    num::NonZeroUsize,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use diskann::{error::IntoANNResult, utils::VectorRepr, ANNError, ANNResult};
use diskann_providers::{
    storage::{PQStorage, StorageReadProvider, StorageWriteProvider},
    utils::{gen_random_slice, load_metadata_from_file, RayonThreadPoolRef, VectorDataIterator},
};
use diskann_vector::{distance::Metric, DistanceFunction};
use rand::Rng;

use crate::{
    data_model::GraphHeader,
    utils::{compute_closest_centers, k_means_clustering, spherical_k_means_clustering},
};

const ROUTING_TABLE_MAGIC: &[u8; 8] = b"DANNRTE1";
const REPRESENTATIVE_BATCH_SIZE: usize = 1_200;
const DISK_INDEX_BINARY_PREFIX_SIZE: usize = 2 * std::mem::size_of::<u32>();
const DEFAULT_DISK_BLOCK_SIZE: u64 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RoutingCandidateIds {
    num_points: usize,
    stride: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LegacyReorderLayout {
    start_block: u64,
    vectors_per_block: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DiskBlockLayout {
    num_points: u64,
    routing_dimension: Option<usize>,
    node_len: u64,
    nodes_per_block: u64,
    block_size: u64,
    declared_file_size: u64,
    legacy_reorder: Option<LegacyReorderLayout>,
}

impl RoutingCandidateIds {
    fn all(num_points: usize) -> Self {
        Self {
            num_points,
            stride: 1,
        }
    }

    fn len(self) -> usize {
        self.num_points.div_ceil(self.stride)
    }

    fn iter(self) -> impl Iterator<Item = usize> {
        (0..self.num_points).step_by(self.stride)
    }
}

/// In-memory vectors and graph IDs used to choose query-specific graph entry points.
#[derive(Debug, Clone)]
pub struct RoutingTable {
    ids: Vec<u32>,
    vectors: Vec<f32>,
    dimension: usize,
}

impl RoutingTable {
    /// Construct a routing table from real graph node IDs and vectors.
    pub fn new(ids: Vec<u32>, vectors: Vec<f32>, dimension: usize) -> ANNResult<Self> {
        if ids.is_empty() {
            return Err(ANNError::log_index_config_error(
                "routing_entries".into(),
                "routing table must contain at least one entry".into(),
            ));
        }
        if dimension == 0 || vectors.len() != ids.len() * dimension {
            return Err(ANNError::log_index_config_error(
                "routing_dimension".into(),
                format!(
                    "routing vectors contain {} elements for {} entries with dimension {}",
                    vectors.len(),
                    ids.len(),
                    dimension
                ),
            ));
        }

        Ok(Self {
            ids,
            vectors,
            dimension,
        })
    }

    /// Number of routing entries.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the routing table is empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Stored vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Real graph node IDs represented by this table.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Select up to `count` unique graph IDs nearest to `query`.
    pub fn select(
        &self,
        query: &[f32],
        metric: Metric,
        count: NonZeroUsize,
    ) -> ANNResult<Vec<u32>> {
        if query.len() != self.dimension {
            return Err(ANNError::log_index_error(format_args!(
                "query dimension {} does not match routing dimension {}",
                query.len(),
                self.dimension
            )));
        }

        let distance = f32::distance(metric, Some(self.dimension));
        let mut ranked = self
            .ids
            .iter()
            .copied()
            .zip(self.vectors.chunks_exact(self.dimension))
            .map(|(id, vector)| (distance.evaluate_similarity(query, vector), id))
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));

        let mut selected = Vec::with_capacity(count.get().min(ranked.len()));
        let mut seen = HashSet::with_capacity(selected.capacity());
        for (_, id) in ranked {
            if seen.insert(id) {
                selected.push(id);
                if selected.len() == count.get() {
                    break;
                }
            }
        }
        Ok(selected)
    }

    /// Load a routing table sidecar.
    pub fn load<P>(path: &str, storage_provider: &P) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        let mut reader = storage_provider.open_reader(path)?;
        let mut magic = [0u8; ROUTING_TABLE_MAGIC.len()];
        reader.read_exact(&mut magic)?;
        if &magic != ROUTING_TABLE_MAGIC {
            return Err(ANNError::log_invalid_file_format(format!(
                "'{}' is not a supported routing table",
                path
            )));
        }

        let num_entries = reader.read_u32::<LittleEndian>()? as usize;
        let dimension = reader.read_u32::<LittleEndian>()? as usize;
        let element_size = reader.read_u32::<LittleEndian>()? as usize;
        if element_size != std::mem::size_of::<f32>() {
            return Err(ANNError::log_invalid_file_format(format!(
                "routing table element size {} does not match requested type size {}",
                element_size,
                std::mem::size_of::<f32>()
            )));
        }

        let vector_size = dimension
            .checked_mul(element_size)
            .ok_or_else(|| ANNError::log_invalid_file_format("routing table size overflow"))?;
        let entry_size = std::mem::size_of::<u32>()
            .checked_add(vector_size)
            .ok_or_else(|| ANNError::log_invalid_file_format("routing table size overflow"))?;
        let entries_size = num_entries
            .checked_mul(entry_size)
            .ok_or_else(|| ANNError::log_invalid_file_format("routing table size overflow"))?;
        let expected_size = ROUTING_TABLE_MAGIC
            .len()
            .checked_add(3 * std::mem::size_of::<u32>())
            .and_then(|header_size| header_size.checked_add(entries_size))
            .ok_or_else(|| ANNError::log_invalid_file_format("routing table size overflow"))?;
        if storage_provider.get_length(path)? != expected_size as u64 {
            return Err(ANNError::log_invalid_file_format(format!(
                "routing table '{}' has an invalid length",
                path
            )));
        }

        let mut ids = Vec::with_capacity(num_entries);
        let mut vectors = Vec::with_capacity(num_entries * dimension);
        let mut vector = vec![0.0f32; dimension];
        for _ in 0..num_entries {
            ids.push(reader.read_u32::<LittleEndian>()?);
            reader.read_exact(bytemuck::cast_slice_mut(&mut vector))?;
            vectors.extend_from_slice(&vector);
        }

        Self::new(ids, vectors, dimension)
    }

    /// Save this routing table as a sidecar.
    pub fn save<P>(&self, path: &str, storage_provider: &P) -> ANNResult<()>
    where
        P: StorageWriteProvider,
    {
        let mut writer = storage_provider.create_for_write(path)?;
        writer.write_all(ROUTING_TABLE_MAGIC)?;
        writer.write_u32::<LittleEndian>(self.len().try_into().map_err(|_| {
            ANNError::log_index_error("routing table contains more than u32::MAX entries")
        })?)?;
        writer.write_u32::<LittleEndian>(self.dimension.try_into().map_err(|_| {
            ANNError::log_index_error("routing table dimension exceeds u32::MAX")
        })?)?;
        writer.write_u32::<LittleEndian>(std::mem::size_of::<f32>() as u32)?;
        for (id, vector) in self
            .ids
            .iter()
            .zip(self.vectors.chunks_exact(self.dimension))
        {
            writer.write_u32::<LittleEndian>(*id)?;
            writer.write_all(bytemuck::cast_slice(vector))?;
        }
        writer.flush()?;
        Ok(())
    }
}

/// Train L2 k-means centers and map each center to its nearest real dataset node.
#[allow(clippy::too_many_arguments)]
pub fn generate_routing_table<T, P>(
    dataset_path: &str,
    output_path: &str,
    num_centers: NonZeroUsize,
    sampling_rate: f64,
    max_kmeans_reps: NonZeroUsize,
    storage_provider: &P,
    rng: &mut impl Rng,
    pool: RayonThreadPoolRef<'_>,
) -> ANNResult<RoutingTable>
where
    T: VectorRepr,
    P: StorageReadProvider + StorageWriteProvider,
{
    if !(0.0 < sampling_rate && sampling_rate <= 1.0) {
        return Err(ANNError::log_index_config_error(
            "routing_sampling_rate".into(),
            format!("sampling rate must be in (0, 1], got {}", sampling_rate),
        ));
    }

    let (training_data, num_training_points, full_dimension) =
        gen_random_slice::<T, P>(dataset_path, sampling_rate, storage_provider, rng)?;
    if num_training_points < num_centers.get() {
        return Err(ANNError::log_index_config_error(
            "routing_num_centers".into(),
            format!(
                "{} sampled training points are insufficient for {} centers",
                num_training_points,
                num_centers.get()
            ),
        ));
    }

    let mut centers = vec![0.0; num_centers.get() * full_dimension];
    k_means_clustering(
        &training_data,
        num_training_points,
        full_dimension,
        &mut centers,
        num_centers.get(),
        max_kmeans_reps.get(),
        rng,
        &mut false,
        pool,
    )?;

    let mut dataset = VectorDataIterator::<P, T>::new(dataset_path, None, storage_provider)?;
    let num_points = dataset.get_num_points();
    let mut best_distances = vec![f32::INFINITY; num_centers.get()];
    let mut representative_ids = vec![0; num_centers.get()];
    let mut representative_vectors = vec![0.0; num_centers.get() * full_dimension];
    let mut processed = 0usize;

    while let Some(batch) = dataset.next_n(REPRESENTATIVE_BATCH_SIZE) {
        let batch_size = batch.len();
        let mut batch_f32 = vec![0.0; batch_size * full_dimension];
        for (row, (vector, ())) in batch.iter().enumerate() {
            T::as_f32_into(
                vector,
                &mut batch_f32[row * full_dimension..(row + 1) * full_dimension],
            )
            .into_ann_result()?;
        }

        let mut closest_centers = vec![0; batch_size];
        compute_closest_centers(
            &batch_f32,
            batch_size,
            full_dimension,
            &centers,
            num_centers.get(),
            1,
            &mut closest_centers,
            None,
            None,
            pool,
        )?;

        for (row, (_, center_id)) in batch.iter().zip(closest_centers).enumerate() {
            let center_id = center_id as usize;
            let point = &batch_f32[row * full_dimension..(row + 1) * full_dimension];
            let center = &centers[center_id * full_dimension..(center_id + 1) * full_dimension];
            let distance = point
                .iter()
                .zip(center)
                .map(|(left, right)| (left - right) * (left - right))
                .sum();
            if distance < best_distances[center_id] {
                best_distances[center_id] = distance;
                representative_ids[center_id] = (processed + row).try_into().map_err(|_| {
                    ANNError::log_index_error("dataset contains more than u32::MAX points")
                })?;
                representative_vectors
                    [center_id * full_dimension..(center_id + 1) * full_dimension]
                    .copy_from_slice(point);
            }
        }
        processed += batch_size;
    }

    if processed != num_points {
        return Err(ANNError::log_invalid_file_format(format!(
            "read {} of {} vectors while generating routing entries",
            processed, num_points
        )));
    }

    let table = RoutingTable::new(representative_ids, representative_vectors, full_dimension)?;
    table.save(output_path, storage_provider)?;
    Ok(table)
}

/// L2-normalize a vector in place. No-op for zero vectors.
fn normalize_in_place(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Generate a routing table by clustering in the index's Search-PQ space.
///
/// Unlike [`generate_routing_table`] (L2 k-means over full-precision base vectors), this:
/// - reconstructs each base vector from the index's Search-PQ codes — the same approximation
///   search actually operates on;
/// - uses ordinary k-means for L2 or spherical k-means for inner product;
/// - selects each region's representative with the same metric used for clustering.
///
/// Stored representative vectors are the reconstructed (PQ-space) vectors, so query-time
/// [`RoutingTable::select`] scores entries in the same space search estimates distances in.
#[allow(clippy::too_many_arguments)]
pub fn generate_routing_table_pq<P>(
    pq_pivots_path: &str,
    pq_compressed_path: &str,
    output_path: &str,
    metric: Metric,
    num_centers: NonZeroUsize,
    sampling_rate: f64,
    max_kmeans_reps: NonZeroUsize,
    storage_provider: &P,
    rng: &mut impl Rng,
    pool: RayonThreadPoolRef<'_>,
) -> ANNResult<RoutingTable>
where
    P: StorageReadProvider + StorageWriteProvider,
{
    generate_routing_table_pq_from_candidates(
        pq_pivots_path,
        pq_compressed_path,
        output_path,
        metric,
        num_centers,
        sampling_rate,
        max_kmeans_reps,
        None,
        storage_provider,
        rng,
        pool,
    )
}

/// Generate a Search-PQ routing table using only the first node in each physical disk block.
///
/// The disk graph is not reordered. This variant reads its block layout from `disk_index_path`,
/// trains k-means on block-first node IDs, and restricts routing representatives to those IDs.
#[allow(clippy::too_many_arguments)]
pub fn generate_routing_table_pq_block_first<P>(
    pq_pivots_path: &str,
    pq_compressed_path: &str,
    disk_index_path: &str,
    output_path: &str,
    metric: Metric,
    num_centers: NonZeroUsize,
    sampling_rate: f64,
    max_kmeans_reps: NonZeroUsize,
    storage_provider: &P,
    rng: &mut impl Rng,
    pool: RayonThreadPoolRef<'_>,
) -> ANNResult<RoutingTable>
where
    P: StorageReadProvider + StorageWriteProvider,
{
    generate_routing_table_pq_from_candidates(
        pq_pivots_path,
        pq_compressed_path,
        output_path,
        metric,
        num_centers,
        sampling_rate,
        max_kmeans_reps,
        Some(disk_index_path),
        storage_provider,
        rng,
        pool,
    )
}

#[allow(clippy::too_many_arguments)]
fn generate_routing_table_pq_from_candidates<P>(
    pq_pivots_path: &str,
    pq_compressed_path: &str,
    output_path: &str,
    metric: Metric,
    num_centers: NonZeroUsize,
    sampling_rate: f64,
    max_kmeans_reps: NonZeroUsize,
    block_first_disk_index_path: Option<&str>,
    storage_provider: &P,
    rng: &mut impl Rng,
    pool: RayonThreadPoolRef<'_>,
) -> ANNResult<RoutingTable>
where
    P: StorageReadProvider + StorageWriteProvider,
{
    if !(0.0 < sampling_rate && sampling_rate <= 1.0) {
        return Err(ANNError::log_index_config_error(
            "routing_sampling_rate".into(),
            format!("sampling rate must be in (0, 1], got {}", sampling_rate),
        ));
    }

    // Load the index quantizer (pivots + compressed codes) to reconstruct the approximate
    // vectors that search operates on.
    let pq_storage = PQStorage::new(pq_pivots_path, pq_compressed_path, None);
    let pq_table = pq_storage.load_pq_pivots_bin(pq_pivots_path, 0, storage_provider)?;
    let metadata = load_metadata_from_file(storage_provider, pq_compressed_path)?;
    let num_points = metadata.npoints();
    let num_chunks = pq_table.get_num_chunks();
    let dim = pq_table.get_dim();
    let compressed = PQStorage::load_pq_compressed_vectors_bin(
        pq_compressed_path,
        num_points,
        num_chunks,
        storage_provider,
    )?;

    let candidate_ids = block_first_disk_index_path
        .map(|path| block_first_candidate_ids(path, num_points, dim, storage_provider))
        .transpose()?
        .unwrap_or_else(|| RoutingCandidateIds::all(num_points));
    let num_candidates = candidate_ids.len();

    if num_candidates < num_centers.get() {
        return Err(ANNError::log_index_config_error(
            "routing_num_centers".into(),
            format!(
                "{num_candidates} eligible points are insufficient for {} centers",
                num_centers.get()
            ),
        ));
    }

    let reconstruct = |id: usize, out: &mut [f32]| -> ANNResult<()> {
        let code = compressed.get_row(id).ok_or_else(|| {
            ANNError::log_index_error("vector id out of range in compressed data")
        })?;
        pq_table.inflate_vector_into(code, out);
        Ok(())
    };

    if !matches!(metric, Metric::L2 | Metric::InnerProduct) {
        return Err(ANNError::log_index_config_error(
            "routing_metric".into(),
            format!(
                "PQ-space routing supports only l2 and innerproduct, got {}",
                metric
            ),
        ));
    }

    // Training set: reconstruct a random sample. MIPS uses unit-normalized vectors for
    // spherical k-means; L2 preserves the reconstructed magnitudes.
    let num_train = (((num_candidates as f64) * sampling_rate).round() as usize)
        .max(num_centers.get())
        .min(num_candidates);
    let mut sampled = candidate_ids.iter().collect::<Vec<_>>();
    for i in 0..num_train {
        let j = rng.random_range(i..num_candidates);
        sampled.swap(i, j);
    }
    sampled.truncate(num_train);

    let mut training = vec![0.0f32; num_train * dim];
    let mut buf = vec![0.0f32; dim];
    for (row, &id) in sampled.iter().enumerate() {
        reconstruct(id, &mut buf)?;
        if metric == Metric::InnerProduct {
            normalize_in_place(&mut buf);
        }
        training[row * dim..(row + 1) * dim].copy_from_slice(&buf);
    }

    let mut centers = vec![0.0f32; num_centers.get() * dim];
    match metric {
        Metric::L2 => {
            k_means_clustering(
                &training,
                num_train,
                dim,
                &mut centers,
                num_centers.get(),
                max_kmeans_reps.get(),
                rng,
                &mut false,
                pool,
            )?;
        }
        Metric::InnerProduct => {
            spherical_k_means_clustering(
                &training,
                num_train,
                dim,
                &mut centers,
                num_centers.get(),
                max_kmeans_reps.get(),
                rng,
                &mut false,
                pool,
            )?;
        }
        _ => unreachable!("metric validated above"),
    }

    // Assign every base point and keep the closest real node to each center.
    let initial_score = match metric {
        Metric::L2 => f32::INFINITY,
        Metric::InnerProduct => f32::NEG_INFINITY,
        _ => unreachable!("metric validated above"),
    };
    let mut best_score = vec![initial_score; num_centers.get()];
    let mut rep_ids = vec![0u32; num_centers.get()];
    let mut rep_vectors = vec![0.0f32; num_centers.get() * dim];
    let mut assign_representative = |id: usize| -> ANNResult<()> {
        reconstruct(id, &mut buf)?;
        let mut best_c = 0usize;
        let mut best_value = initial_score;
        for c in 0..num_centers.get() {
            let center = &centers[c * dim..(c + 1) * dim];
            let value = match metric {
                Metric::L2 => buf.iter().zip(center).map(|(x, y)| (x - y) * (x - y)).sum(),
                Metric::InnerProduct => buf.iter().zip(center).map(|(x, y)| x * y).sum(),
                _ => unreachable!("metric validated above"),
            };
            let is_better = match metric {
                Metric::L2 => value < best_value,
                Metric::InnerProduct => value > best_value,
                _ => unreachable!("metric validated above"),
            };
            if is_better {
                best_value = value;
                best_c = c;
            }
        }
        let is_better_representative = match metric {
            Metric::L2 => best_value < best_score[best_c],
            Metric::InnerProduct => best_value > best_score[best_c],
            _ => unreachable!("metric validated above"),
        };
        if is_better_representative {
            best_score[best_c] = best_value;
            rep_ids[best_c] = id.try_into().map_err(|_| {
                ANNError::log_index_error("dataset contains more than u32::MAX points")
            })?;
            rep_vectors[best_c * dim..(best_c + 1) * dim].copy_from_slice(&buf);
        }
        Ok(())
    };
    for id in candidate_ids.iter() {
        assign_representative(id)?;
    }

    // Drop empty regions (rare for small k), keeping only populated representatives.
    let mut ids = Vec::with_capacity(num_centers.get());
    let mut vectors = Vec::with_capacity(num_centers.get() * dim);
    for c in 0..num_centers.get() {
        if best_score[c].is_finite() {
            ids.push(rep_ids[c]);
            vectors.extend_from_slice(&rep_vectors[c * dim..(c + 1) * dim]);
        }
    }

    let table = RoutingTable::new(ids, vectors, dim)?;
    table.save(output_path, storage_provider)?;
    Ok(table)
}

fn block_first_candidate_ids<P>(
    disk_index_path: &str,
    expected_num_points: usize,
    expected_dimension: usize,
    storage_provider: &P,
) -> ANNResult<RoutingCandidateIds>
where
    P: StorageReadProvider,
{
    let minimum_file_size = DISK_INDEX_BINARY_PREFIX_SIZE
        .checked_add(GraphHeader::get_size())
        .ok_or_else(|| ANNError::log_invalid_file_format("disk header size overflow"))?;
    let actual_file_size = storage_provider.get_length(disk_index_path)?;
    if actual_file_size < minimum_file_size as u64 {
        return Err(ANNError::log_invalid_file_format(format!(
            "disk index '{disk_index_path}' is too small to contain a graph header"
        )));
    }

    let mut reader = storage_provider.open_reader(disk_index_path)?;
    let mut header_bytes = vec![0u8; minimum_file_size];
    reader.read_exact(&mut header_bytes)?;
    let layout = parse_disk_block_layout(&header_bytes[DISK_INDEX_BINARY_PREFIX_SIZE..])?;
    let num_points = usize::try_from(layout.num_points)
        .map_err(|_| ANNError::log_invalid_file_format("disk point count exceeds usize"))?;
    if num_points != expected_num_points {
        return Err(ANNError::log_index_error(format!(
            "disk index contains {num_points} points but PQ codes contain {expected_num_points}"
        )));
    }
    if let Some(dimension) = layout.routing_dimension {
        if dimension != expected_dimension {
            return Err(ANNError::log_index_error(format!(
                "disk index dimension is {dimension} but PQ pivots use dimension {expected_dimension}"
            )));
        }
    }
    if num_points == 0 {
        return Err(ANNError::log_invalid_file_format(
            "disk index contains no points",
        ));
    }
    if layout.routing_dimension == Some(0) {
        return Err(ANNError::log_invalid_file_format(
            "disk index dimension is zero",
        ));
    }
    if layout.block_size < minimum_file_size as u64 {
        return Err(ANNError::log_invalid_file_format(format!(
            "disk block size {} is too small for the {minimum_file_size}-byte graph header",
            layout.block_size
        )));
    }

    if layout.node_len == 0 {
        return Err(ANNError::log_invalid_file_format(
            "disk index node length is zero",
        ));
    }

    let nodes_per_block = layout.nodes_per_block;
    let stride = if nodes_per_block == 0 {
        if layout.node_len <= layout.block_size {
            return Err(ANNError::log_invalid_file_format(format!(
                "disk index reports multi-block nodes but node length {} fits in block size {}",
                layout.node_len, layout.block_size
            )));
        }
        1
    } else {
        let expected_nodes_per_block = layout.block_size / layout.node_len;
        if expected_nodes_per_block != nodes_per_block {
            return Err(ANNError::log_invalid_file_format(format!(
                "disk index reports {nodes_per_block} nodes per block, expected {expected_nodes_per_block} from block size {} and node length {}",
                layout.block_size, layout.node_len
            )));
        }
        usize::try_from(nodes_per_block)
            .map_err(|_| ANNError::log_invalid_file_format("block stride exceeds usize"))?
    };

    let data_blocks = if nodes_per_block > 0 {
        layout.num_points.div_ceil(nodes_per_block)
    } else {
        let blocks_per_node = layout.node_len.div_ceil(layout.block_size);
        layout
            .num_points
            .checked_mul(blocks_per_node)
            .ok_or_else(|| ANNError::log_invalid_file_format("disk block count overflow"))?
    };
    let graph_span = data_blocks
        .checked_add(1)
        .and_then(|blocks| blocks.checked_mul(layout.block_size))
        .ok_or_else(|| ANNError::log_invalid_file_format("disk index size overflow"))?;
    let minimum_layout_span = match layout.legacy_reorder {
        Some(reorder) => {
            let expected_start_block = data_blocks
                .checked_add(1)
                .ok_or_else(|| ANNError::log_invalid_file_format("disk block count overflow"))?;
            if reorder.start_block != expected_start_block {
                return Err(ANNError::log_invalid_file_format(format!(
                    "legacy reorder data starts at block {}, expected {expected_start_block}",
                    reorder.start_block
                )));
            }
            let reorder_dimension = layout.routing_dimension.ok_or_else(|| {
                ANNError::log_invalid_file_format(
                    "legacy reorder metadata is missing its vector dimension",
                )
            })?;
            let vector_bytes = u64::try_from(reorder_dimension)
                .ok()
                .and_then(|dimension| dimension.checked_mul(std::mem::size_of::<f32>() as u64))
                .ok_or_else(|| {
                    ANNError::log_invalid_file_format("legacy reorder vector size overflow")
                })?;
            let expected_vectors_per_block = layout.block_size / vector_bytes;
            if expected_vectors_per_block == 0
                || reorder.vectors_per_block != expected_vectors_per_block
            {
                return Err(ANNError::log_invalid_file_format(format!(
                    "legacy reorder metadata reports {} vectors per block, expected {expected_vectors_per_block}",
                    reorder.vectors_per_block
                )));
            }
            let reorder_blocks = layout.num_points.div_ceil(reorder.vectors_per_block);
            reorder
                .start_block
                .checked_add(reorder_blocks)
                .and_then(|blocks| blocks.checked_mul(layout.block_size))
                .ok_or_else(|| ANNError::log_invalid_file_format("disk index size overflow"))?
        }
        None => graph_span,
    };
    if layout.declared_file_size < minimum_layout_span
        || actual_file_size < layout.declared_file_size
    {
        return Err(ANNError::log_invalid_file_format(format!(
            "disk index size is inconsistent: header {}, actual file {actual_file_size}, minimum layout span {minimum_layout_span}",
            layout.declared_file_size
        )));
    }

    Ok(RoutingCandidateIds { num_points, stride })
}

fn parse_disk_block_layout(header_bytes: &[u8]) -> ANNResult<DiskBlockLayout> {
    if header_bytes.len() < GraphHeader::get_size() {
        return Err(ANNError::log_invalid_file_format(
            "disk graph header is truncated",
        ));
    }

    let mut cursor = Cursor::new(header_bytes);
    let num_points = cursor.read_u64::<LittleEndian>()?;
    let graph_dimension = cursor.read_u64::<LittleEndian>()?;
    let _medoid = cursor.read_u64::<LittleEndian>()?;
    let node_len = cursor.read_u64::<LittleEndian>()?;
    let nodes_per_block = cursor.read_u64::<LittleEndian>()?;
    let _num_frozen = cursor.read_u64::<LittleEndian>()?;
    let _frozen_location = cursor.read_u64::<LittleEndian>()?;
    let append_reorder_data = cursor.read_u64::<LittleEndian>()?;

    if append_reorder_data == 0 {
        let header = GraphHeader::try_from(header_bytes)?;
        let version = header.layout_version();
        let is_legacy = version.major_version() == 0 && version.minor_version() == 0;
        if !is_legacy && version != &GraphHeader::CURRENT_LAYOUT_VERSION {
            return Err(ANNError::log_invalid_file_format(format!(
                "unsupported graph layout version {version}"
            )));
        }
        let block_size = if is_legacy {
            DEFAULT_DISK_BLOCK_SIZE
        } else {
            header.block_size()
        };
        return Ok(DiskBlockLayout {
            num_points,
            routing_dimension: if is_legacy {
                None
            } else {
                Some(usize::try_from(graph_dimension).map_err(|_| {
                    ANNError::log_invalid_file_format("disk dimension exceeds usize")
                })?)
            },
            node_len,
            nodes_per_block,
            block_size,
            declared_file_size: header.metadata().disk_index_file_size,
            legacy_reorder: None,
        });
    }
    if append_reorder_data != 1 {
        return Err(ANNError::log_invalid_file_format(format!(
            "disk index has invalid append_reorder_data value {append_reorder_data}"
        )));
    }

    let start_block = cursor.read_u64::<LittleEndian>()?;
    let reorder_dimension = cursor.read_u64::<LittleEndian>()?;
    let vectors_per_block = cursor.read_u64::<LittleEndian>()?;
    let declared_file_size = cursor.read_u64::<LittleEndian>()?;
    Ok(DiskBlockLayout {
        num_points,
        routing_dimension: Some(
            usize::try_from(reorder_dimension)
                .map_err(|_| ANNError::log_invalid_file_format("disk dimension exceeds usize"))?,
        ),
        node_len,
        nodes_per_block,
        block_size: DEFAULT_DISK_BLOCK_SIZE,
        declared_file_size,
        legacy_reorder: Some(LegacyReorderLayout {
            start_block,
            vectors_per_block,
        }),
    })
}

#[cfg(test)]
mod tests {
    use std::{io::Write, num::NonZeroUsize};

    use diskann_providers::storage::{StorageWriteProvider, VirtualStorageProvider};
    use diskann_vector::distance::Metric;

    use crate::data_model::{GraphLayoutVersion, GraphMetadata};

    use super::*;

    fn write_test_disk_index<P>(
        storage: &P,
        path: &str,
        num_points: u64,
        block_size: u64,
        node_len: u64,
        nodes_per_block: u64,
    ) where
        P: StorageWriteProvider,
    {
        write_test_disk_index_with_layout(
            storage,
            path,
            num_points,
            block_size,
            block_size,
            node_len,
            nodes_per_block,
            GraphLayoutVersion::new(1, 0),
            0,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn write_test_disk_index_with_layout<P>(
        storage: &P,
        path: &str,
        num_points: u64,
        effective_block_size: u64,
        stored_block_size: u64,
        node_len: u64,
        nodes_per_block: u64,
        layout_version: GraphLayoutVersion,
        trailing_bytes: usize,
    ) where
        P: StorageWriteProvider,
    {
        let data_blocks = if nodes_per_block > 0 {
            num_points.div_ceil(nodes_per_block)
        } else {
            num_points * node_len.div_ceil(effective_block_size)
        };
        let disk_size = (data_blocks + 1) * effective_block_size;
        let metadata = GraphMetadata::new(
            num_points,
            8,
            0,
            node_len,
            nodes_per_block,
            0,
            0,
            disk_size,
            0,
        );
        let header = GraphHeader::new(metadata, stored_block_size, layout_version)
            .to_bytes()
            .unwrap();
        let mut bytes = vec![0u8; disk_size as usize + trailing_bytes];
        bytes[DISK_INDEX_BINARY_PREFIX_SIZE..DISK_INDEX_BINARY_PREFIX_SIZE + header.len()]
            .copy_from_slice(&header);
        let mut writer = storage.create_for_write(path).unwrap();
        writer.write_all(&bytes).unwrap();
        writer.flush().unwrap();
    }

    fn write_legacy_reordered_test_disk_index<P>(
        storage: &P,
        path: &str,
        num_points: u64,
        dimension: u64,
        node_len: u64,
        nodes_per_block: u64,
    ) where
        P: StorageWriteProvider,
    {
        let data_blocks = num_points.div_ceil(nodes_per_block);
        let reorder_start_block = data_blocks + 1;
        let vectors_per_block =
            DEFAULT_DISK_BLOCK_SIZE / (dimension * std::mem::size_of::<f32>() as u64);
        let reorder_blocks = num_points.div_ceil(vectors_per_block);
        let disk_size = (reorder_start_block + reorder_blocks) * DEFAULT_DISK_BLOCK_SIZE;
        let fields = [
            num_points,
            16,
            0,
            node_len,
            nodes_per_block,
            0,
            0,
            1,
            reorder_start_block,
            dimension,
            vectors_per_block,
            disk_size,
        ];
        let mut bytes = vec![0u8; disk_size as usize];
        let mut header = vec![];
        for field in fields {
            header.write_u64::<LittleEndian>(field).unwrap();
        }
        bytes[DISK_INDEX_BINARY_PREFIX_SIZE..DISK_INDEX_BINARY_PREFIX_SIZE + header.len()]
            .copy_from_slice(&header);
        let mut writer = storage.create_for_write(path).unwrap();
        writer.write_all(&bytes).unwrap();
        writer.flush().unwrap();
    }

    fn write_legacy_nonreordered_test_disk_index<P>(
        storage: &P,
        path: &str,
        num_points: u64,
        graph_dimension: u64,
        node_len: u64,
        nodes_per_block: u64,
    ) where
        P: StorageWriteProvider,
    {
        let data_blocks = num_points.div_ceil(nodes_per_block);
        let disk_size = (data_blocks + 1) * DEFAULT_DISK_BLOCK_SIZE;
        let fields = [
            num_points,
            graph_dimension,
            0,
            node_len,
            nodes_per_block,
            0,
            0,
            0,
            disk_size,
            0,
            0,
            0,
        ];
        let mut bytes = vec![0u8; disk_size as usize];
        let mut header = vec![];
        for field in fields {
            header.write_u64::<LittleEndian>(field).unwrap();
        }
        bytes[DISK_INDEX_BINARY_PREFIX_SIZE..DISK_INDEX_BINARY_PREFIX_SIZE + header.len()]
            .copy_from_slice(&header);
        let mut writer = storage.create_for_write(path).unwrap();
        writer.write_all(&bytes).unwrap();
        writer.flush().unwrap();
    }

    #[test]
    fn routing_table_round_trip_and_selection() {
        let storage = VirtualStorageProvider::new_memory();
        let table =
            RoutingTable::new(vec![10, 20, 30], vec![0.0f32, 0.0, 5.0, 5.0, 9.0, 9.0], 2).unwrap();
        table.save("/routing.bin", &storage).unwrap();

        let loaded = RoutingTable::load("/routing.bin", &storage).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.dimension(), 2);
        assert_eq!(
            loaded
                .select(&[4.0, 4.0], Metric::L2, NonZeroUsize::new(2).unwrap())
                .unwrap(),
            vec![20, 10]
        );
        assert!(storage.exists("/routing.bin"));
    }

    #[test]
    fn selection_deduplicates_node_ids() {
        let table = RoutingTable::new(vec![10, 10, 20], vec![0.0f32, 0.1, 1.0], 1).unwrap();
        assert_eq!(
            table
                .select(&[0.0], Metric::L2, NonZeroUsize::new(2).unwrap())
                .unwrap(),
            vec![10, 20]
        );
    }

    #[test]
    fn selection_prefers_largest_inner_product() {
        let table =
            RoutingTable::new(vec![10, 20, 30], vec![1.0f32, 0.0, 0.0, 1.0, 2.0, 2.0], 2).unwrap();
        assert_eq!(
            table
                .select(&[1.0, 1.0], Metric::InnerProduct, NonZeroUsize::MIN)
                .unwrap(),
            vec![30]
        );
    }

    #[test]
    fn block_first_candidates_use_physical_block_stride() {
        let storage = VirtualStorageProvider::new_memory();
        write_test_disk_index(&storage, "/disk.index", 21, 4096, 1000, 4);

        assert_eq!(
            block_first_candidate_ids("/disk.index", 21, 8, &storage)
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![0, 4, 8, 12, 16, 20]
        );
    }

    #[test]
    fn block_first_stride_one_matches_all_nodes() {
        let storage = VirtualStorageProvider::new_memory();
        write_test_disk_index(&storage, "/one-per-block.index", 5, 4096, 3000, 1);
        write_test_disk_index(&storage, "/multi-block.index", 5, 4096, 5000, 0);
        let expected = vec![0, 1, 2, 3, 4];

        assert_eq!(
            block_first_candidate_ids("/one-per-block.index", 5, 8, &storage)
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            block_first_candidate_ids("/multi-block.index", 5, 8, &storage)
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn block_first_candidates_reject_point_count_mismatch() {
        let storage = VirtualStorageProvider::new_memory();
        write_test_disk_index(&storage, "/disk.index", 5, 4096, 1000, 4);

        let error = block_first_candidate_ids("/disk.index", 6, 8, &storage).unwrap_err();
        assert!(error
            .to_string()
            .contains("disk index contains 5 points but PQ codes contain 6"));
    }

    #[test]
    fn block_first_candidates_reject_inconsistent_layout() {
        let storage = VirtualStorageProvider::new_memory();
        write_test_disk_index(&storage, "/disk.index", 10, 4096, 1000, 3);

        let error = block_first_candidate_ids("/disk.index", 10, 8, &storage).unwrap_err();
        assert!(error
            .to_string()
            .contains("reports 3 nodes per block, expected 4"));
    }

    #[test]
    fn block_first_candidates_use_legacy_block_size_fallback() {
        let storage = VirtualStorageProvider::new_memory();
        write_test_disk_index_with_layout(
            &storage,
            "/legacy.index",
            10,
            DEFAULT_DISK_BLOCK_SIZE,
            0,
            1000,
            4,
            GraphLayoutVersion::new(0, 0),
            0,
        );

        assert_eq!(
            block_first_candidate_ids("/legacy.index", 10, 8, &storage)
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![0, 4, 8]
        );
    }

    #[test]
    fn block_first_candidates_allow_trailing_payload() {
        let storage = VirtualStorageProvider::new_memory();
        write_test_disk_index_with_layout(
            &storage,
            "/trailing-payload.index",
            10,
            4096,
            4096,
            1000,
            4,
            GraphLayoutVersion::new(1, 0),
            128,
        );

        assert_eq!(
            block_first_candidate_ids("/trailing-payload.index", 10, 8, &storage)
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![0, 4, 8]
        );
    }

    #[test]
    fn block_first_candidates_parse_legacy_reorder_header() {
        let storage = VirtualStorageProvider::new_memory();
        write_legacy_reordered_test_disk_index(&storage, "/legacy-reordered.index", 10, 8, 1000, 4);

        assert_eq!(
            block_first_candidate_ids("/legacy-reordered.index", 10, 8, &storage)
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![0, 4, 8]
        );
    }

    #[test]
    fn block_first_candidates_allow_legacy_disk_pq_dimension() {
        let storage = VirtualStorageProvider::new_memory();
        write_legacy_nonreordered_test_disk_index(
            &storage,
            "/legacy-disk-pq.index",
            10,
            4,
            1000,
            4,
        );

        assert_eq!(
            block_first_candidate_ids("/legacy-disk-pq.index", 10, 8, &storage)
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![0, 4, 8]
        );
    }

    #[test]
    fn block_first_candidates_reject_unknown_layout_version() {
        let storage = VirtualStorageProvider::new_memory();
        write_test_disk_index_with_layout(
            &storage,
            "/future.index",
            10,
            4096,
            4096,
            1000,
            4,
            GraphLayoutVersion::new(2, 0),
            0,
        );

        let error = block_first_candidate_ids("/future.index", 10, 8, &storage).unwrap_err();
        assert!(error
            .to_string()
            .contains("unsupported graph layout version 2.0"));
    }

    #[test]
    fn block_first_candidates_reject_header_overlapping_graph_data() {
        let storage = VirtualStorageProvider::new_memory();
        write_test_disk_index(&storage, "/tiny-block.index", 2, 96, 80, 1);

        let error = block_first_candidate_ids("/tiny-block.index", 2, 8, &storage).unwrap_err();
        assert!(error
            .to_string()
            .contains("block size 96 is too small for the 104-byte graph header"));
    }

    #[test]
    fn block_first_candidates_reject_zero_current_block_size() {
        let storage = VirtualStorageProvider::new_memory();
        write_test_disk_index_with_layout(
            &storage,
            "/zero-block.index",
            10,
            4096,
            0,
            1000,
            4,
            GraphLayoutVersion::new(1, 0),
            0,
        );

        let error = block_first_candidate_ids("/zero-block.index", 10, 8, &storage).unwrap_err();
        assert!(error
            .to_string()
            .contains("block size 0 is too small for the 104-byte graph header"));
    }

    #[test]
    fn block_first_candidates_reject_dimension_mismatch() {
        let storage = VirtualStorageProvider::new_memory();
        write_test_disk_index(&storage, "/disk.index", 10, 4096, 1000, 4);

        let error = block_first_candidate_ids("/disk.index", 10, 7, &storage).unwrap_err();
        assert!(error
            .to_string()
            .contains("disk index dimension is 8 but PQ pivots use dimension 7"));
    }
}
