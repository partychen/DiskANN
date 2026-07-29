/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{cmp::min, collections::VecDeque, sync::Arc, time::Instant};

use crate::data_model::GraphDataType;
use diskann::{graph::AdjacencyList, ANNError, ANNResult};
use diskann_quantization::{
    alloc::{AlignedAllocator, Poly},
    num::PowerOfTwo,
};
use hashbrown::HashSet;
use tracing::info;

use crate::{
    data_model::{Cache, CachingStrategy, GraphHeader},
    layout::{load_physical_layout, PhysicalLayout},
    search::provider::aligned_file_reader::{
        traits::{AlignedFileReader, AlignedReaderFactory},
        AlignedFileReaderFactory, AlignedRead,
    },
    search::{
        provider::{
            cached_disk_vertex_provider::CachedDiskVertexProvider,
            disk_vertex_provider::DiskVertexProvider,
        },
        traits::{VertexProvider, VertexProviderFactory},
    },
};

const DEFAULT_DISK_SECTOR_LEN: usize = 4096;
const BEAM_WIDTH_FOR_BFS: usize = 32;

/// DiskVertexProviderFactory. This is one of the implementations for the `VertexProviderFactory` trait.
pub struct DiskVertexProviderFactory<
    Data: GraphDataType<VectorIdType = u32>,
    ReaderFactory: AlignedReaderFactory,
> {
    pub aligned_reader_factory: ReaderFactory,
    pub caching_strategy: CachingStrategy,
    pub cache: Option<Arc<Cache<Data>>>,
    physical_layout: Option<Arc<PhysicalLayout>>,
    physical_layout_load_time_us: u128,
}

/// DiskVertexProviderFactory. This is one of the implementations for the `VertexProviderFactory` trait, for which the associated graph data is read from disk.
impl<Data, ReaderFactory> VertexProviderFactory<Data>
    for DiskVertexProviderFactory<Data, ReaderFactory>
where
    ReaderFactory: AlignedReaderFactory,
    Data: GraphDataType<VectorIdType = u32>,
{
    type VertexProviderType = CachedDiskVertexProvider<Data, ReaderFactory::AlignedReaderType>;

    fn get_header(&self) -> ANNResult<GraphHeader> {
        // Here we still need the hardcoded len, because the length of the read_buf needs to be the multiple times of DEFAULT_DISK_SECTOR_LEN.
        // since this is the implementation for the disk vertex provider, there're only two kinds of sector lengths: 4096 and 512.
        // it's okay to hardcoded at this place.
        let buffer_len = GraphHeader::get_size().next_multiple_of(DEFAULT_DISK_SECTOR_LEN);
        let mut read_buf = Poly::broadcast(
            0u8,
            buffer_len,
            AlignedAllocator::new(PowerOfTwo::new(buffer_len).map_err(ANNError::log_index_error)?),
        )
        .map_err(ANNError::log_index_error)?;
        let aligned_read = AlignedRead::new(0_u64, &mut read_buf)?;
        self.aligned_reader_factory
            .build()?
            .read(&mut [aligned_read])?;

        // Create a GraphHeader from the buffer.
        GraphHeader::try_from(&read_buf[8..])
    }

    fn create_vertex_provider(
        &self,
        max_batch_size: usize,
        header: &GraphHeader,
    ) -> ANNResult<Self::VertexProviderType> {
        let sector_reader = self.aligned_reader_factory.build()?;
        match self.caching_strategy {
            CachingStrategy::StaticCacheWithBfsNodes(_) => match self.cache {
                Some(ref cache) => CachedDiskVertexProvider::new_with_layout(
                    header,
                    max_batch_size,
                    sector_reader,
                    cache.clone(),
                    self.physical_layout.clone(),
                ),
                None => Err(ANNError::log_index_error(
                    "Cache must be initialised for StaticCacheWithBfsNodes caching strategy",
                )),
            },
            CachingStrategy::None => CachedDiskVertexProvider::new_with_layout(
                header,
                max_batch_size,
                sector_reader,
                Arc::new(Cache::new(0, 0)?),
                self.physical_layout.clone(),
            ),
        }
    }
}

impl<Data: GraphDataType<VectorIdType = u32>>
    DiskVertexProviderFactory<Data, AlignedFileReaderFactory>
{
    /// Creates a production `DiskVertexProviderFactory` that reads the on-disk index at
    /// `disk_index_path` using the platform's native aligned file reader.
    pub fn from_disk_index_path(
        disk_index_path: String,
        caching_strategy: CachingStrategy,
    ) -> ANNResult<Self> {
        let timer = Instant::now();
        let physical_layout =
            load_physical_layout(&disk_index_path, None).map_err(ANNError::log_index_error)?;
        let physical_layout_load_time_us = physical_layout
            .as_ref()
            .map_or(0, |_| timer.elapsed().as_micros());
        Self::new_with_layout(
            AlignedFileReaderFactory::new(disk_index_path),
            caching_strategy,
            physical_layout,
            physical_layout_load_time_us,
        )
    }
}

impl<Data: GraphDataType<VectorIdType = u32>, ReaderFactory: AlignedReaderFactory>
    DiskVertexProviderFactory<Data, ReaderFactory>
{
    /// Creates a DiskVertexProviderFactory instance.
    pub fn new(
        aligned_reader_factory: ReaderFactory,
        caching_strategy: CachingStrategy,
    ) -> ANNResult<Self> {
        Self::new_with_layout(aligned_reader_factory, caching_strategy, None, 0)
    }

    fn new_with_layout(
        aligned_reader_factory: ReaderFactory,
        caching_strategy: CachingStrategy,
        physical_layout: Option<Arc<PhysicalLayout>>,
        physical_layout_load_time_us: u128,
    ) -> ANNResult<Self> {
        let mut disk_vertex_provider_factory = DiskVertexProviderFactory {
            aligned_reader_factory,
            caching_strategy,
            cache: None,
            physical_layout,
            physical_layout_load_time_us,
        };

        if disk_vertex_provider_factory.caching_strategy != CachingStrategy::None {
            disk_vertex_provider_factory.setup_cache()?;
        }

        Ok(disk_vertex_provider_factory)
    }

    pub fn physical_layout_memory_bytes(&self) -> usize {
        self.physical_layout
            .as_ref()
            .map_or(0, |layout| layout.memory_bytes())
    }

    pub fn physical_layout_load_time_us(&self) -> u128 {
        self.physical_layout_load_time_us
    }

    fn create_disk_vertex_provider(
        &self,
        max_batch_size: usize,
        header: &GraphHeader,
    ) -> ANNResult<DiskVertexProvider<Data, ReaderFactory::AlignedReaderType>> {
        DiskVertexProvider::new_with_layout(
            header,
            max_batch_size,
            self.aligned_reader_factory.build()?,
            self.physical_layout.clone(),
        )
    }

    fn setup_cache(&mut self) -> ANNResult<()> {
        let timer = Instant::now();

        match self.caching_strategy {
            CachingStrategy::StaticCacheWithBfsNodes(mut num_nodes_to_cache) => {
                if num_nodes_to_cache == 0 {
                    ANNError::log_index_error(
                        "num_nodes_to_cache should be greater than 0 for StaticCacheWithBfsNodes caching strategy",
                    );
                }

                let graph_metadata = self.get_header()?;
                let graph_metadata = graph_metadata.metadata();

                if num_nodes_to_cache > graph_metadata.num_pts as usize {
                    info!(
                        "Reducing nodes to cache from: {} to: {} (total no. of nodes)",
                        num_nodes_to_cache, graph_metadata.num_pts
                    );
                    num_nodes_to_cache = graph_metadata.num_pts as usize;
                }

                let start_node = graph_metadata.medoid as u32;
                self.cache = Some(Arc::new(self.build_cache_via_bfs(
                    &[start_node],
                    num_nodes_to_cache,
                    graph_metadata.dims,
                )?));
            }
            CachingStrategy::None => {}
        }

        info!("Cache setup took: {} ms", timer.elapsed().as_millis());
        Ok(())
    }

    /// (Re)build the static node cache seeded from an explicit set of graph nodes
    /// (e.g. query-aware routing entry points) instead of the graph medoid.
    ///
    /// The BFS starts from every node in `seeds`, so the seeds and their surrounding
    /// neighborhood are made resident in memory, making the first search hops from any
    /// chosen entry point IO-free. Passing an empty `seeds` or a zero budget is a no-op.
    pub fn seed_cache_from_nodes(
        &mut self,
        seeds: &[u32],
        num_nodes_to_cache: usize,
    ) -> ANNResult<()> {
        if seeds.is_empty() || num_nodes_to_cache == 0 {
            return Ok(());
        }

        let graph_metadata = self.get_header()?;
        let graph_metadata = graph_metadata.metadata();
        let num_nodes_to_cache = min(num_nodes_to_cache, graph_metadata.num_pts as usize);

        let cache = self.build_cache_via_bfs(seeds, num_nodes_to_cache, graph_metadata.dims)?;
        self.cache = Some(Arc::new(cache));
        self.caching_strategy = CachingStrategy::StaticCacheWithBfsNodes(num_nodes_to_cache);
        Ok(())
    }

    /// Build a static cache containing the supplied graph nodes in the given order.
    ///
    /// Unlike [`Self::seed_cache_from_nodes`], this does not expand graph neighbors.
    /// Duplicate IDs are ignored and the first `num_nodes_to_cache` unique IDs are used.
    pub fn seed_cache_from_exact_nodes(
        &mut self,
        nodes: &[u32],
        num_nodes_to_cache: usize,
    ) -> ANNResult<usize> {
        if nodes.is_empty() || num_nodes_to_cache == 0 {
            return Ok(0);
        }

        let graph_metadata = self.get_header()?;
        let graph_metadata = graph_metadata.metadata();
        let num_nodes_to_cache = min(num_nodes_to_cache, graph_metadata.num_pts as usize);
        let mut seen = HashSet::with_capacity(num_nodes_to_cache);
        let mut selected = Vec::with_capacity(num_nodes_to_cache);

        for &node in nodes {
            if node as usize >= graph_metadata.num_pts as usize {
                return Err(ANNError::log_index_config_error(
                    "cache_node_id".into(),
                    format!(
                        "cache node ID {} is outside graph range 0..{}",
                        node, graph_metadata.num_pts
                    ),
                ));
            }
            if seen.insert(node) {
                selected.push(node);
                if selected.len() == num_nodes_to_cache {
                    break;
                }
            }
        }

        let cache = self.build_cache_from_exact_nodes(&selected, graph_metadata.dims)?;
        let cache_len = cache.len();
        self.cache = Some(Arc::new(cache));
        self.caching_strategy = CachingStrategy::StaticCacheWithBfsNodes(cache_len);
        Ok(cache_len)
    }

    fn build_cache_from_exact_nodes(
        &self,
        nodes: &[u32],
        dimension: usize,
    ) -> ANNResult<Cache<Data>> {
        info!("Building cache with {} exact nodes.", nodes.len());
        let mut cache = Cache::new(dimension, nodes.len())?;
        let mut vertex_provider =
            self.create_disk_vertex_provider(BEAM_WIDTH_FOR_BFS, &self.get_header()?)?;

        for batch in nodes.chunks(BEAM_WIDTH_FOR_BFS) {
            vertex_provider.load_vertices(batch)?;
            for (idx, node) in batch.iter().enumerate() {
                Self::insert_in_cache(node, idx, &mut vertex_provider, &mut cache)?;
            }
        }

        Ok(cache)
    }

    fn build_cache_via_bfs(
        &self,
        start_nodes: &[u32],
        num_nodes_to_cache: usize,
        dimension: usize,
    ) -> ANNResult<Cache<Data>> {
        info!("Building cache with {} nodes via BFS.", num_nodes_to_cache);
        let mut cache = Cache::new(dimension, num_nodes_to_cache)?;
        let mut vertex_provider =
            self.create_disk_vertex_provider(BEAM_WIDTH_FOR_BFS, &self.get_header()?)?;

        let mut visited = HashSet::with_capacity(num_nodes_to_cache);
        let mut queue = VecDeque::with_capacity(num_nodes_to_cache);
        let mut nodes_in_a_batch = Vec::with_capacity(BEAM_WIDTH_FOR_BFS);

        for &start_node in start_nodes {
            if visited.insert(start_node) {
                queue.push_back(start_node);
            }
        }

        while (!queue.is_empty()) && cache.len() < num_nodes_to_cache {
            nodes_in_a_batch.clear();
            let batch_size = min(queue.len(), BEAM_WIDTH_FOR_BFS);
            for _ in 0..batch_size {
                let node = queue.pop_front().ok_or_else(|| {
                    ANNError::log_index_error("Error while caching Nodes via BFS: Queue is empty")
                })?;
                nodes_in_a_batch.push(node);
            }

            vertex_provider.load_vertices(&nodes_in_a_batch)?;

            for (idx, node) in nodes_in_a_batch.iter().enumerate() {
                Self::insert_in_cache(node, idx, &mut vertex_provider, &mut cache)?;
                let adjacency_list = cache.get_adjacency_list(node).ok_or_else(|| {
                    ANNError::log_index_error(format!("Error while caching Nodes via BFS: Adjacency List not found for inserted node {} in cache.", node))
                })?;
                for neighbor_id in adjacency_list.iter() {
                    if !visited.contains(neighbor_id) {
                        queue.push_back(*neighbor_id);
                        visited.insert(*neighbor_id);
                    }
                }
                if cache.len() >= num_nodes_to_cache {
                    break;
                }
            }
        }

        ANNResult::Ok(cache)
    }

    fn insert_in_cache<AlignedReaderType>(
        node: &Data::VectorIdType,
        idx: usize,
        vertex_provider: &mut DiskVertexProvider<Data, AlignedReaderType>,
        cache: &mut Cache<Data>,
    ) -> ANNResult<()>
    where
        AlignedReaderType: AlignedFileReader,
    {
        vertex_provider.process_loaded_node(node, idx)?;
        let vector = vertex_provider.get_vector(node)?;
        let adjacency_list = vertex_provider.get_adjacency_list(node)?;
        let associated_data = vertex_provider.get_associated_data(node)?;

        cache.insert(
            node,
            vector,
            AdjacencyList::from_iter_untrusted(adjacency_list.iter().copied()),
            *associated_data,
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        search::provider::aligned_file_reader::VirtualAlignedReaderFactory,
        test_utils::GraphDataF32VectorUnitData,
    };
    use diskann_providers::storage::VirtualStorageProvider;
    use diskann_utils::test_data_root;
    use vfs::OverlayFS;

    // Use existing test data instead of generating new indices
    const TEST_INDEX_PATH: &str =
        "/disk_index_search/disk_index_sift_learn_R4_L50_A1.2_truth_search_disk.index";

    #[test]
    fn test_disk_vertex_provider_factory_new_with_no_cache() {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));

        let factory = DiskVertexProviderFactory::<
            GraphDataF32VectorUnitData,
            VirtualAlignedReaderFactory<OverlayFS>,
        >::new(
            VirtualAlignedReaderFactory::new(TEST_INDEX_PATH.to_string(), storage_provider.clone()),
            CachingStrategy::None,
        )
        .unwrap();

        assert!(factory.cache.is_none());
    }

    #[test]
    fn test_disk_vertex_provider_factory_with_static_cache() {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));

        let num_nodes_to_cache = 10;
        let factory = DiskVertexProviderFactory::<
            GraphDataF32VectorUnitData,
            VirtualAlignedReaderFactory<OverlayFS>,
        >::new(
            VirtualAlignedReaderFactory::new(TEST_INDEX_PATH.to_string(), storage_provider.clone()),
            CachingStrategy::StaticCacheWithBfsNodes(num_nodes_to_cache),
        )
        .unwrap();

        // Verify cache was created
        assert!(factory.cache.is_some());
        let cache = factory.cache.as_ref().unwrap();
        assert!(!cache.is_empty());
        assert!(cache.len() <= num_nodes_to_cache);
    }

    #[test]
    fn test_seed_cache_from_exact_nodes_preserves_order_and_deduplicates() {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));
        let mut factory = DiskVertexProviderFactory::<
            GraphDataF32VectorUnitData,
            VirtualAlignedReaderFactory<OverlayFS>,
        >::new(
            VirtualAlignedReaderFactory::new(TEST_INDEX_PATH.to_string(), storage_provider),
            CachingStrategy::None,
        )
        .unwrap();

        let cached = factory
            .seed_cache_from_exact_nodes(&[7, 3, 7, 20, 9], 3)
            .unwrap();

        assert_eq!(cached, 3);
        let cache = factory.cache.as_ref().unwrap();
        assert!(cache.contains(&7));
        assert!(cache.contains(&3));
        assert!(cache.contains(&20));
        assert!(!cache.contains(&9));
    }

    #[test]
    fn test_disk_vertex_provider_factory_cache_limit_exceeds_total_nodes() {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));

        // Request to cache more nodes than exist in the index
        let num_nodes_to_cache = 100000;
        let factory = DiskVertexProviderFactory::<
            GraphDataF32VectorUnitData,
            VirtualAlignedReaderFactory<OverlayFS>,
        >::new(
            VirtualAlignedReaderFactory::new(TEST_INDEX_PATH.to_string(), storage_provider.clone()),
            CachingStrategy::StaticCacheWithBfsNodes(num_nodes_to_cache),
        )
        .unwrap();

        // Verify cache was created but limited to actual number of nodes
        assert!(factory.cache.is_some());
        let cache = factory.cache.as_ref().unwrap();
        // The test index has 256 nodes
        assert!(cache.len() <= 256);
    }

    #[test]
    fn test_create_vertex_provider_with_no_cache() {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));

        let factory = DiskVertexProviderFactory::<
            GraphDataF32VectorUnitData,
            VirtualAlignedReaderFactory<OverlayFS>,
        >::new(
            VirtualAlignedReaderFactory::new(TEST_INDEX_PATH.to_string(), storage_provider.clone()),
            CachingStrategy::None,
        )
        .unwrap();

        let header = factory.get_header().unwrap();
        let vertex_provider = factory.create_vertex_provider(32, &header).unwrap();

        // Verify the provider was created successfully
        assert_eq!(vertex_provider.io_operations(), 0);
    }

    #[test]
    fn test_create_vertex_provider_with_cache() {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));

        let factory = DiskVertexProviderFactory::<
            GraphDataF32VectorUnitData,
            VirtualAlignedReaderFactory<OverlayFS>,
        >::new(
            VirtualAlignedReaderFactory::new(TEST_INDEX_PATH.to_string(), storage_provider.clone()),
            CachingStrategy::StaticCacheWithBfsNodes(10),
        )
        .unwrap();

        let header = factory.get_header().unwrap();
        let vertex_provider = factory.create_vertex_provider(32, &header).unwrap();

        // Verify the provider was created successfully with a cache
        assert_eq!(vertex_provider.io_operations(), 0);
    }

    #[test]
    fn test_create_vertex_provider_with_cache_but_none_initialized_should_error() {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));

        // Create a factory with a caching strategy but manually set cache to None
        let factory = DiskVertexProviderFactory::<
            GraphDataF32VectorUnitData,
            VirtualAlignedReaderFactory<OverlayFS>,
        > {
            aligned_reader_factory: VirtualAlignedReaderFactory::new(
                TEST_INDEX_PATH.to_string(),
                storage_provider.clone(),
            ),
            caching_strategy: CachingStrategy::StaticCacheWithBfsNodes(10),
            cache: None, // Intentionally None despite caching strategy requiring it
            physical_layout: None,
            physical_layout_load_time_us: 0,
        };

        let header = factory.get_header().unwrap();
        let result = factory.create_vertex_provider(32, &header);

        // Should error because cache is required but not initialized
        assert!(result.is_err());
    }

    #[test]
    fn test_get_header() {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));

        let factory = DiskVertexProviderFactory::<
            GraphDataF32VectorUnitData,
            VirtualAlignedReaderFactory<OverlayFS>,
        >::new(
            VirtualAlignedReaderFactory::new(TEST_INDEX_PATH.to_string(), storage_provider.clone()),
            CachingStrategy::None,
        )
        .unwrap();

        let header = factory.get_header().unwrap();
        assert_eq!(header.metadata().num_pts, 256);
    }
}
