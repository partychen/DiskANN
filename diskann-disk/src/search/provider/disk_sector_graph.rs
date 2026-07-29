/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![warn(missing_docs)]

//! Sector graph
use std::{ops::Deref, sync::Arc};

use diskann::{ANNError, ANNResult};
use diskann_quantization::alloc::{AlignedAllocator, Poly};

use crate::{
    data_model::GraphHeader,
    layout::PhysicalLayout,
    search::provider::aligned_file_reader::{traits::AlignedFileReader, AlignedRead, Alignment},
};

const DEFAULT_DISK_SECTOR_LEN: usize = 4096;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PhysicalReadMetrics {
    pub read_requests: u32,
    pub blocks_read: u64,
    pub bytes_read: u64,
}

#[derive(Debug, Default, Clone, Copy)]
struct LoadedNodeLocation {
    buffer_block_index: usize,
    node_offset: usize,
}

/// Sector graph read from disk index
pub struct DiskSectorGraph<AlignedReaderType: AlignedFileReader> {
    /// Ensure `sector_reader` is dropped before `sectors_data` by placing it before `sectors_data`.
    /// Graph storage to read sectors
    sector_reader: AlignedReaderType,
    /// Sector bytes from disk
    /// One sector has num_nodes_per_sector nodes
    /// Each node's layout: {full precision vector:[T; DIM]}{num_nbrs: u32}{neighbors: [u32; num_nbrs]}
    /// The fp vector is not aligned
    ///
    /// index info for multi-node sectors
    /// node `i` is in sector: [i / num_nodes_per_sector]
    /// offset in sector: [(i % num_nodes_per_sector) * node_len]
    ///
    /// index info for multi-sector nodes
    /// node `i` is in sector: [i * max_node_len.div_ceil(block_size)]
    /// offset in sector: [0]
    sectors_data: Poly<[u8], AlignedAllocator>,
    /// Current sector index into which the next read reads data
    cur_sector_idx: u64,

    /// 0 for multi-sector nodes, >0 for multi-node sectors
    num_nodes_per_sector: u64,

    node_len: u64,

    max_n_batch_sector_read: usize,

    num_sectors_per_node: usize,

    block_size: usize,

    physical_layout: Option<Arc<PhysicalLayout>>,

    loaded_node_locations: Vec<LoadedNodeLocation>,
}

impl<AlignedReaderType: AlignedFileReader> DiskSectorGraph<AlignedReaderType> {
    /// Create SectorGraph instance
    pub fn new(
        sector_reader: AlignedReaderType,
        header: &GraphHeader,
        max_n_batch_sector_read: usize,
    ) -> ANNResult<Self> {
        Self::new_with_layout(sector_reader, header, max_n_batch_sector_read, None)
    }

    pub(crate) fn new_with_layout(
        sector_reader: AlignedReaderType,
        header: &GraphHeader,
        max_n_batch_sector_read: usize,
        physical_layout: Option<Arc<PhysicalLayout>>,
    ) -> ANNResult<Self> {
        let mut block_size = header.block_size() as usize;
        let version = header.layout_version();
        if (version.major_version() == 0 && version.minor_version() == 0) || block_size == 0 {
            block_size = DEFAULT_DISK_SECTOR_LEN;
        }

        let num_nodes_per_sector = header.metadata().num_nodes_per_block;
        let node_len = header.metadata().node_len;
        let num_sectors_per_node = if num_nodes_per_sector > 0 {
            1
        } else {
            (node_len as usize).div_ceil(block_size)
        };

        Ok(Self {
            sector_reader,
            sectors_data: Poly::broadcast(
                0u8,
                max_n_batch_sector_read * num_sectors_per_node * block_size,
                AlignedAllocator::new(AlignedReaderType::Alignment::VALUE),
            )
            .map_err(ANNError::log_index_error)?,
            cur_sector_idx: 0,
            num_nodes_per_sector,
            node_len,
            max_n_batch_sector_read,
            num_sectors_per_node,
            block_size,
            physical_layout,
            loaded_node_locations: Vec::with_capacity(max_n_batch_sector_read),
        })
    }

    /// Reconfigure SectorGraph if the max number of sectors to read is larger than the current one
    pub fn reconfigure(&mut self, max_n_batch_sector_read: usize) -> ANNResult<()> {
        if max_n_batch_sector_read > self.max_n_batch_sector_read {
            self.max_n_batch_sector_read = max_n_batch_sector_read;
            self.sectors_data = Poly::broadcast(
                0u8,
                max_n_batch_sector_read * self.num_sectors_per_node * self.block_size,
                AlignedAllocator::new(AlignedReaderType::Alignment::VALUE),
            )
            .map_err(ANNError::log_index_error)?;
        }
        Ok(())
    }

    /// Reset SectorGraph
    pub fn reset(&mut self) {
        self.cur_sector_idx = 0;
        self.loaded_node_locations.clear();
    }

    /// Read sectors into sectors_data
    /// They are in the same order as sectors_to_fetch
    pub fn read_graph(&mut self, sectors_to_fetch: &[u64]) -> ANNResult<()> {
        let cur_sector_idx_usize: usize = self.cur_sector_idx.try_into()?;
        if sectors_to_fetch.len() > self.max_n_batch_sector_read - cur_sector_idx_usize {
            return Err(ANNError::log_index_error(format_args!(
                "Trying to read too many sectors. number of sectors to read: {}, max number of sectors can read: {}",
                sectors_to_fetch.len(),
                self.max_n_batch_sector_read - cur_sector_idx_usize,
            )));
        }

        let len_per_node = self.num_sectors_per_node * self.block_size;
        if len_per_node == 0 {
            return Err(ANNError::log_index_error(format_args!(
                "len_per_node is 0 (num_sectors_per_node={}, block_size={})",
                self.num_sectors_per_node, self.block_size,
            )));
        }
        let range = cur_sector_idx_usize * len_per_node
            ..(cur_sector_idx_usize + sectors_to_fetch.len()) * len_per_node;
        debug_assert!(
            range.len() % len_per_node == 0,
            "range length {} is not divisible by {}",
            range.len(),
            len_per_node
        );
        let mut sector_slices: Vec<&mut [u8]> =
            self.sectors_data[range].chunks_mut(len_per_node).collect();
        let mut read_requests: Vec<AlignedRead<'_, u8, AlignedReaderType::Alignment>> =
            Vec::with_capacity(sector_slices.len());
        for (local_sector_idx, slice) in sector_slices.iter_mut().enumerate() {
            let sector_id = sectors_to_fetch[local_sector_idx];
            read_requests.push(AlignedRead::new(sector_id * self.block_size as u64, slice)?);
        }

        self.sector_reader.read(&mut read_requests)?;
        self.cur_sector_idx += sectors_to_fetch.len() as u64;

        Ok(())
    }

    pub(crate) fn read_vertices(&mut self, vertex_ids: &[u32]) -> ANNResult<PhysicalReadMetrics> {
        if self.physical_layout.is_none() {
            let sectors_to_fetch = vertex_ids
                .iter()
                .map(|&vertex_id| self.node_sector_index(vertex_id))
                .collect::<ANNResult<Vec<_>>>()?;
            self.read_graph(&sectors_to_fetch)?;
            let blocks_read = (vertex_ids.len() * self.num_sectors_per_node) as u64;
            return Ok(PhysicalReadMetrics {
                read_requests: vertex_ids.len().try_into()?,
                blocks_read,
                bytes_read: blocks_read * self.block_size as u64,
            });
        }

        if vertex_ids.len() > self.max_n_batch_sector_read {
            return Err(ANNError::log_index_error(format_args!(
                "Trying to read too many vertices. number of vertices to read: {}, max: {}",
                vertex_ids.len(),
                self.max_n_batch_sector_read,
            )));
        }

        let mut requested_nodes = Vec::with_capacity(vertex_ids.len());
        let mut unique_sectors =
            Vec::with_capacity(vertex_ids.len().saturating_mul(self.num_sectors_per_node));
        for (request_index, &vertex_id) in vertex_ids.iter().enumerate() {
            let start_sector = self.node_sector_index(vertex_id)?;
            requested_nodes.push((
                request_index,
                start_sector,
                self.get_node_offset(vertex_id)?,
            ));
            unique_sectors.extend(
                start_sector..start_sector.saturating_add(self.num_sectors_per_node as u64),
            );
        }
        unique_sectors.sort_unstable();
        unique_sectors.dedup();

        self.loaded_node_locations
            .resize(vertex_ids.len(), LoadedNodeLocation::default());
        for (request_index, start_sector, node_offset) in requested_nodes {
            let buffer_block_index = unique_sectors.binary_search(&start_sector).map_err(|_| {
                ANNError::log_index_error("mapped vertex sector missing from read plan")
            })?;
            self.loaded_node_locations[request_index] = LoadedNodeLocation {
                buffer_block_index,
                node_offset,
            };
        }

        let blocks_read = unique_sectors.len() as u64;
        let bytes_read = blocks_read
            .checked_mul(self.block_size as u64)
            .ok_or_else(|| ANNError::log_index_error("physical read byte count overflow"))?;
        if unique_sectors.is_empty() {
            return Ok(PhysicalReadMetrics::default());
        }

        let required_bytes = unique_sectors
            .len()
            .checked_mul(self.block_size)
            .ok_or_else(|| ANNError::log_index_error("physical read buffer size overflow"))?;
        if required_bytes > self.sectors_data.len() {
            return Err(ANNError::log_index_error(format_args!(
                "Mapped read plan requires {} bytes, buffer has {}",
                required_bytes,
                self.sectors_data.len()
            )));
        }

        let mut ranges = Vec::new();
        let mut range_start = 0usize;
        for index in 1..=unique_sectors.len() {
            if index == unique_sectors.len()
                || unique_sectors[index] != unique_sectors[index - 1] + 1
            {
                ranges.push((range_start, index));
                range_start = index;
            }
        }

        let mut read_requests = Vec::with_capacity(ranges.len());
        let mut remaining = &mut self.sectors_data[..required_bytes];
        let mut consumed_bytes = 0usize;
        for (start, end) in ranges {
            let start_byte = start * self.block_size;
            let gap = start_byte - consumed_bytes;
            let (_, after_gap) = remaining.split_at_mut(gap);
            let len = (end - start) * self.block_size;
            let (buffer, after_range) = after_gap.split_at_mut(len);
            read_requests.push(AlignedRead::new(
                unique_sectors[start] * self.block_size as u64,
                buffer,
            )?);
            remaining = after_range;
            consumed_bytes = start_byte + len;
        }
        self.sector_reader.read(&mut read_requests)?;

        Ok(PhysicalReadMetrics {
            read_requests: read_requests.len().try_into()?,
            blocks_read,
            bytes_read,
        })
    }

    #[inline]
    /// Get node data by local index.
    pub fn node_disk_buf(&self, node_index_local: usize, vertex_id: u32) -> ANNResult<&[u8]> {
        if self.physical_layout.is_some() {
            let location = self
                .loaded_node_locations
                .get(node_index_local)
                .ok_or_else(|| {
                    ANNError::log_index_error(format_args!(
                        "No mapped read location for local node index {node_index_local}"
                    ))
                })?;
            let start = location
                .buffer_block_index
                .checked_mul(self.block_size)
                .and_then(|value| value.checked_add(location.node_offset))
                .ok_or_else(|| ANNError::log_index_error("mapped node offset overflow"))?;
            let end = start
                .checked_add(self.node_len as usize)
                .ok_or_else(|| ANNError::log_index_error("mapped node end overflow"))?;
            return self
                .sectors_data
                .get(start..end)
                .ok_or_else(|| ANNError::log_index_error("mapped node is outside read buffer"));
        }
        // get sector_buf where this node is located
        let sector_buf = self.get_sector_buf(node_index_local);
        let node_offset = self.get_node_offset(vertex_id)?;
        Ok(&sector_buf[node_offset..node_offset + self.node_len as usize])
    }

    /// Get sector data by local index
    #[inline]
    fn get_sector_buf(&self, local_sector_idx: usize) -> &[u8] {
        let len_per_node = self.num_sectors_per_node * self.block_size;
        &self.sectors_data[local_sector_idx * len_per_node..(local_sector_idx + 1) * len_per_node]
    }

    /// Get offset of node in sectors_data
    #[inline]
    fn get_node_offset(&self, vertex_id: u32) -> ANNResult<usize> {
        let physical_slot = self.physical_slot(vertex_id)?;
        if self.num_nodes_per_sector == 0 {
            // multi-sector node
            Ok(0)
        } else {
            // multi node in a sector
            Ok((physical_slot % self.num_nodes_per_sector * self.node_len) as usize)
        }
    }

    #[inline]
    /// Gets the index for the sector that contains the node with the given vertex_id
    pub fn node_sector_index(&self, vertex_id: u32) -> ANNResult<u64> {
        let physical_slot = self.physical_slot(vertex_id)?;
        Ok(1 + if self.num_nodes_per_sector > 0 {
            physical_slot / self.num_nodes_per_sector
        } else {
            physical_slot * self.num_sectors_per_node as u64
        })
    }

    fn physical_slot(&self, vertex_id: u32) -> ANNResult<u64> {
        match &self.physical_layout {
            Some(layout) => layout
                .physical_slot(vertex_id)
                .map(u64::from)
                .ok_or_else(|| {
                    ANNError::log_index_error(format_args!(
                        "Logical vertex ID {vertex_id} is outside physical layout"
                    ))
                }),
            None => Ok(u64::from(vertex_id)),
        }
    }
}

impl<AlignedReaderType: AlignedFileReader> Deref for DiskSectorGraph<AlignedReaderType> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.sectors_data
    }
}

#[cfg(test)]
mod disk_sector_graph_test {
    use diskann_utils::test_data_root;

    use super::*;
    use crate::{
        data_model::{GraphLayoutVersion, GraphMetadata},
        layout::PhysicalLayout,
        search::provider::aligned_file_reader::{
            traits::{AlignedFileReader, AlignedReaderFactory},
            AlignedFileReaderFactory, AlignedRead, A1,
        },
    };

    struct TestReader {
        data: Vec<u8>,
    }

    impl AlignedFileReader for TestReader {
        type Alignment = A1;

        fn read(&mut self, reads: &mut [AlignedRead<u8, A1>]) -> ANNResult<()> {
            for read in reads {
                let start = read.offset() as usize;
                let end = start + read.aligned_buf().len();
                read.aligned_buf_mut()
                    .copy_from_slice(&self.data[start..end]);
            }
            Ok(())
        }
    }

    fn test_index_path() -> String {
        test_data_root()
            .join("disk_index_misc/disk_index_siftsmall_learn_256pts_R4_L50_A1.2_aligned_reader_test.index")
            .to_string_lossy()
            .to_string()
    }

    fn test_initialize_disk_sector_graph(
        num_nodes_per_sector: u64,
        num_sectors_per_node: usize,
        sector_reader: <AlignedFileReaderFactory as AlignedReaderFactory>::AlignedReaderType,
    ) -> DiskSectorGraph<<AlignedFileReaderFactory as AlignedReaderFactory>::AlignedReaderType>
    {
        DiskSectorGraph {
            sectors_data: Poly::broadcast(0u8, 512, AlignedAllocator::A512).unwrap(),
            sector_reader,
            cur_sector_idx: 0,
            num_nodes_per_sector,
            node_len: 32,
            max_n_batch_sector_read: 4,
            num_sectors_per_node,
            block_size: 64,
            physical_layout: None,
            loaded_node_locations: Vec::new(),
        }
    }

    #[test]
    fn test_new_disk_sector_graph_multi_node_per_sector() {
        let metadata = GraphMetadata::new(1000, 32, 500, 32, 2, 20, 50, 1024, 256);
        let header = GraphHeader::new(metadata, 64, GraphLayoutVersion::new(1, 0));
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = DiskSectorGraph::new(reader, &header, 2).unwrap();
        assert_eq!(graph.sectors_data.len(), 128);
        assert_eq!(graph.num_sectors_per_node, 1);
        assert_eq!(graph.num_nodes_per_sector, 2);
    }

    #[test]
    fn test_new_disk_sector_graph_multi_sector_per_node() {
        let metadata = GraphMetadata::new(1000, 32, 500, 128, 0, 20, 50, 1024, 256);
        let header = GraphHeader::new(metadata, 64, GraphLayoutVersion::new(1, 0));
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = DiskSectorGraph::new(reader, &header, 2).unwrap();
        assert_eq!(graph.sectors_data.len(), 256);
        assert_eq!(graph.num_sectors_per_node, 2);
        assert_eq!(graph.num_nodes_per_sector, 0);
    }

    #[test]
    fn test_new_disk_sector_graph_old_version_data() {
        let metadata = GraphMetadata::new(1000, 32, 500, 128, 0, 20, 50, 1024, 256);
        let header = GraphHeader::new(metadata, 9999, GraphLayoutVersion::new(0, 0));
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = DiskSectorGraph::new(reader, &header, 2).unwrap();
        assert_eq!(graph.block_size, DEFAULT_DISK_SECTOR_LEN);
    }

    #[test]
    fn get_sector_buf_test() {
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = test_initialize_disk_sector_graph(2, 1, reader);
        let sector_buf = graph.get_sector_buf(0);
        assert_eq!(sector_buf.len(), 64);
    }

    #[test]
    fn get_node_offset_test_multi_node_per_sector() {
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = test_initialize_disk_sector_graph(4, 1, reader);

        assert_eq!(graph.get_node_offset(0).unwrap(), 0);
        assert_eq!(graph.get_node_offset(1).unwrap(), 32);
        assert_eq!(graph.get_node_offset(2).unwrap(), 64);
        assert_eq!(graph.get_node_offset(3).unwrap(), 96);
        assert_eq!(graph.get_node_offset(4).unwrap(), 0);
        assert_eq!(graph.get_node_offset(5).unwrap(), 32);
        assert_eq!(graph.get_node_offset(6).unwrap(), 64);
        assert_eq!(graph.get_node_offset(7).unwrap(), 96);
    }

    #[test]
    fn get_node_offset_test_multi_sector_per_node() {
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = test_initialize_disk_sector_graph(0, 2, reader);

        assert_eq!(graph.get_node_offset(0).unwrap(), 0);
        assert_eq!(graph.get_node_offset(1).unwrap(), 0);
        assert_eq!(graph.get_node_offset(2).unwrap(), 0);
        assert_eq!(graph.get_node_offset(3).unwrap(), 0);
        assert_eq!(graph.get_node_offset(4).unwrap(), 0);
        assert_eq!(graph.get_node_offset(5).unwrap(), 0);
    }

    #[test]
    fn node_sector_index_test_multi_node_per_sector() {
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = test_initialize_disk_sector_graph(4, 1, reader);

        assert_eq!(graph.node_sector_index(0).unwrap(), 1);
        assert_eq!(graph.node_sector_index(3).unwrap(), 1);
        assert_eq!(graph.node_sector_index(4).unwrap(), 2);
        assert_eq!(graph.node_sector_index(5).unwrap(), 2);
        assert_eq!(graph.node_sector_index(7).unwrap(), 2);
        assert_eq!(graph.node_sector_index(8).unwrap(), 3);
        assert_eq!(graph.node_sector_index(1023).unwrap(), 256);
        assert_eq!(graph.node_sector_index(1024).unwrap(), 257);
        assert_eq!(graph.node_sector_index(2047).unwrap(), 512);
        assert_eq!(graph.node_sector_index(2048).unwrap(), 513);
    }

    #[test]
    fn node_sector_index_test_multi_sector_per_node() {
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = test_initialize_disk_sector_graph(0, 2, reader);

        assert_eq!(graph.node_sector_index(0).unwrap(), 1);
        assert_eq!(graph.node_sector_index(3).unwrap(), 7);
        assert_eq!(graph.node_sector_index(4).unwrap(), 9);
        assert_eq!(graph.node_sector_index(5).unwrap(), 11);
        assert_eq!(graph.node_sector_index(7).unwrap(), 15);
        assert_eq!(graph.node_sector_index(8).unwrap(), 17);
        assert_eq!(graph.node_sector_index(1023).unwrap(), 2047);
        assert_eq!(graph.node_sector_index(1024).unwrap(), 2049);
        assert_eq!(graph.node_sector_index(2047).unwrap(), 4095);
        assert_eq!(graph.node_sector_index(2048).unwrap(), 4097);
    }

    #[test]
    fn test_read_graph_max_sectors() {
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let mut disk_sector_graph = test_initialize_disk_sector_graph(0, 2, reader);

        // Try to read more sectors than the maximum allowed
        let sectors_to_fetch = vec![1, 2, 3, 4, 5, 6];
        let result = disk_sector_graph.read_graph(&sectors_to_fetch);

        // Check that an error is returned
        // Trying to read too many sectors. number of sectors to read: {}, max number of sectors can read: {}",
        assert!(result.is_err());
    }

    #[test]
    fn mapped_reads_deduplicate_blocks_and_coalesce_adjacent_ranges() {
        let block_size = 64usize;
        let node_len = 16usize;
        let mapping = vec![0, 1, 4, 5, 2, 3, 6, 7];
        let layout = PhysicalLayout::from_mapping_for_test(mapping.clone());
        let metadata = GraphMetadata::new(8, 1, 0, node_len as u64, 4, 0, 0, 192, 0);
        let header = GraphHeader::new(
            metadata,
            block_size as u64,
            GraphHeader::GRAPH_AWARE_LAYOUT_VERSION,
        );
        let mut data = vec![0u8; block_size * 3];
        for (logical, &physical) in mapping.iter().enumerate() {
            let offset = block_size
                + (physical as usize / 4) * block_size
                + (physical as usize % 4) * node_len;
            data[offset] = logical as u8;
        }
        let reader = TestReader { data };
        let mut graph = DiskSectorGraph::new_with_layout(reader, &header, 4, Some(layout)).unwrap();

        let requested = [0, 1, 4, 2];
        let metrics = graph.read_vertices(&requested).unwrap();
        assert_eq!(
            metrics,
            PhysicalReadMetrics {
                read_requests: 1,
                blocks_read: 2,
                bytes_read: 128,
            }
        );
        for (index, logical) in requested.into_iter().enumerate() {
            assert_eq!(
                graph.node_disk_buf(index, logical).unwrap()[0],
                logical as u8
            );
        }

        graph.reset();
        let metrics = graph.read_vertices(&[0, 1, 4]).unwrap();
        assert_eq!(metrics.read_requests, 1);
        assert_eq!(metrics.blocks_read, 1);
        assert_eq!(metrics.bytes_read, 64);
    }

    #[test]
    fn test_disk_sector_graph_deref() {
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = test_initialize_disk_sector_graph(1, 1, reader);
        let data = &graph;
        assert_eq!(data.len(), 512);
    }

    #[test]
    fn test_reconfigure_grows_buffer() {
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let mut graph = test_initialize_disk_sector_graph(2, 1, reader);
        assert_eq!(graph.max_n_batch_sector_read, 4);

        // Reconfigure to larger batch — buffer must grow beyond initial 512 bytes
        graph.reconfigure(16).unwrap();
        assert_eq!(graph.max_n_batch_sector_read, 16);
        assert_eq!(graph.sectors_data.len(), 16 * 64);
    }

    #[test]
    fn test_reconfigure_noop_for_smaller_size() {
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let mut graph = test_initialize_disk_sector_graph(2, 1, reader);
        let original_len = graph.sectors_data.len();

        // Reconfigure with same or smaller size should be a no-op
        graph.reconfigure(4).unwrap();
        assert_eq!(graph.max_n_batch_sector_read, 4);
        assert_eq!(graph.sectors_data.len(), original_len);

        graph.reconfigure(2).unwrap();
        assert_eq!(graph.max_n_batch_sector_read, 4);
        assert_eq!(graph.sectors_data.len(), original_len);
    }

    #[test]
    fn test_new_disk_sector_graph_zero_block_size_defaults() {
        let metadata = GraphMetadata::new(1000, 32, 500, 32, 2, 20, 50, 1024, 256);
        // block_size = 0 should fall back to DEFAULT_DISK_SECTOR_LEN regardless of version
        let header = GraphHeader::new(metadata, 0, GraphLayoutVersion::new(1, 0));
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = DiskSectorGraph::new(reader, &header, 2).unwrap();
        assert_eq!(graph.block_size, DEFAULT_DISK_SECTOR_LEN);
    }
}
