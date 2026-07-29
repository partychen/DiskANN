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

/// Effective composition of a static hybrid node cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridCacheComposition {
    /// BFS node target requested by the configured fraction, before preserving roots.
    pub requested_bfs_nodes: usize,
    /// Number of unique routing roots that the cache must preserve.
    pub routing_root_nodes: usize,
    /// Unique nodes selected from the initial routing-root BFS prefix.
    pub bfs_nodes: usize,
    /// Unique visit-frequency nodes added after the BFS prefix.
    pub frequency_nodes: usize,
    /// Unique nodes added from the remaining BFS order when frequency ranks underfill.
    pub fallback_bfs_nodes: usize,
    /// Total unique nodes loaded into the cache.
    pub total_nodes: usize,
    /// Estimated cached graph payload in bytes, using the on-disk node length.
    pub estimated_payload_bytes: u64,
}

/// DiskVertexProviderFactory. This is one of the implementations for the `VertexProviderFactory` trait.
pub struct DiskVertexProviderFactory<
    Data: GraphDataType<VectorIdType = u32>,
    ReaderFactory: AlignedReaderFactory,
> {
    pub aligned_reader_factory: ReaderFactory,
    pub caching_strategy: CachingStrategy,
    pub cache: Option<Arc<Cache<Data>>>,
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
                Some(ref cache) => CachedDiskVertexProvider::new(
                    header,
                    max_batch_size,
                    sector_reader,
                    cache.clone(),
                ),
                None => Err(ANNError::log_index_error(
                    "Cache must be initialised for StaticCacheWithBfsNodes caching strategy",
                )),
            },
            CachingStrategy::None => CachedDiskVertexProvider::new(
                header,
                max_batch_size,
                sector_reader,
                Arc::new(Cache::new(0, 0)?),
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
        Self::new(
            AlignedFileReaderFactory::new(disk_index_path),
            caching_strategy,
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
        let mut disk_vertex_provider_factory = DiskVertexProviderFactory {
            aligned_reader_factory,
            caching_strategy,
            cache: None,
        };

        if disk_vertex_provider_factory.caching_strategy != CachingStrategy::None {
            disk_vertex_provider_factory.setup_cache()?;
        }

        Ok(disk_vertex_provider_factory)
    }

    fn create_disk_vertex_provider(
        &self,
        max_batch_size: usize,
        header: &GraphHeader,
    ) -> ANNResult<DiskVertexProvider<Data, ReaderFactory::AlignedReaderType>> {
        DiskVertexProvider::new(header, max_batch_size, self.aligned_reader_factory.build()?)
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

    /// Build one static cache from a routing-root BFS prefix and frequency-ranked nodes.
    ///
    /// Every unique routing root is retained even when the requested BFS target is smaller.
    /// If frequency-ranked nodes cannot fill the budget after deduplication, the remaining
    /// deterministic BFS order fills the cache.
    pub fn seed_cache_from_hybrid_nodes(
        &mut self,
        seeds: &[u32],
        frequency_nodes: &[u32],
        num_nodes_to_cache: usize,
        requested_bfs_nodes: usize,
    ) -> ANNResult<HybridCacheComposition> {
        if seeds.is_empty() {
            return Err(ANNError::log_index_config_error(
                "hybrid_cache_seeds".into(),
                "hybrid cache requires at least one routing root".into(),
            ));
        }
        if num_nodes_to_cache == 0 {
            return Err(ANNError::log_index_config_error(
                "hybrid_cache_budget".into(),
                "hybrid cache budget must be positive".into(),
            ));
        }

        let graph_header = self.get_header()?;
        let graph_metadata = graph_header.metadata();
        if num_nodes_to_cache > graph_metadata.num_pts as usize {
            return Err(ANNError::log_index_config_error(
                "hybrid_cache_budget".into(),
                format!(
                    "cache budget {num_nodes_to_cache} exceeds graph point count {}",
                    graph_metadata.num_pts
                ),
            ));
        }
        let mut unique_roots = HashSet::with_capacity(seeds.len());
        for &seed in seeds {
            if seed as u64 >= graph_metadata.num_pts {
                return Err(ANNError::log_index_config_error(
                    "hybrid_cache_seed".into(),
                    format!(
                        "routing root {seed} is outside graph range 0..{}",
                        graph_metadata.num_pts
                    ),
                ));
            }
            unique_roots.insert(seed);
        }
        if unique_roots.len() > num_nodes_to_cache {
            return Err(ANNError::log_index_config_error(
                "hybrid_cache_budget".into(),
                format!(
                    "cache budget {num_nodes_to_cache} cannot preserve {} unique routing roots",
                    unique_roots.len()
                ),
            ));
        }
        if let Some(&node) = frequency_nodes
            .iter()
            .find(|&&node| node as u64 >= graph_metadata.num_pts)
        {
            return Err(ANNError::log_index_config_error(
                "hybrid_cache_frequency_node".into(),
                format!(
                    "frequency-ranked node {node} is outside graph range 0..{}",
                    graph_metadata.num_pts
                ),
            ));
        }

        let bfs_nodes = self.collect_nodes_via_bfs(seeds, num_nodes_to_cache)?;
        let (selected, composition) = compose_hybrid_cache_nodes(
            &bfs_nodes,
            frequency_nodes,
            num_nodes_to_cache,
            requested_bfs_nodes,
            unique_roots.len(),
            graph_metadata.node_len,
        )?;
        let cache = self.build_cache_from_exact_nodes(&selected, graph_metadata.dims)?;
        if cache.len() != num_nodes_to_cache {
            return Err(ANNError::log_index_error(format!(
                "hybrid cache loaded {} nodes, expected {num_nodes_to_cache}",
                cache.len()
            )));
        }
        self.cache = Some(Arc::new(cache));
        self.caching_strategy = CachingStrategy::StaticCacheWithBfsNodes(num_nodes_to_cache);
        Ok(composition)
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
        self.walk_bfs(start_nodes, num_nodes_to_cache, Some(&mut cache))?;
        ANNResult::Ok(cache)
    }

    fn collect_nodes_via_bfs(
        &self,
        start_nodes: &[u32],
        num_nodes_to_collect: usize,
    ) -> ANNResult<Vec<u32>> {
        self.walk_bfs(start_nodes, num_nodes_to_collect, None)
    }

    fn walk_bfs(
        &self,
        start_nodes: &[u32],
        num_nodes: usize,
        mut cache: Option<&mut Cache<Data>>,
    ) -> ANNResult<Vec<u32>> {
        let mut vertex_provider =
            self.create_disk_vertex_provider(BEAM_WIDTH_FOR_BFS, &self.get_header()?)?;

        let mut visited = HashSet::with_capacity(num_nodes);
        let mut queue = VecDeque::with_capacity(num_nodes);
        let mut nodes_in_a_batch = Vec::with_capacity(BEAM_WIDTH_FOR_BFS);
        let mut bfs_nodes = Vec::with_capacity(num_nodes);

        for &start_node in start_nodes {
            if visited.insert(start_node) {
                queue.push_back(start_node);
            }
        }

        while !queue.is_empty() && bfs_nodes.len() < num_nodes {
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
                vertex_provider.process_loaded_node(node, idx)?;
                let adjacency_list = AdjacencyList::from_iter_untrusted(
                    vertex_provider.get_adjacency_list(node)?.iter().copied(),
                );
                if let Some(cache) = cache.as_deref_mut() {
                    let vector = vertex_provider.get_vector(node)?;
                    let associated_data = vertex_provider.get_associated_data(node)?;
                    cache.insert(node, vector, adjacency_list.clone(), *associated_data)?;
                }
                bfs_nodes.push(*node);
                for neighbor_id in adjacency_list.iter() {
                    if !visited.contains(neighbor_id) {
                        queue.push_back(*neighbor_id);
                        visited.insert(*neighbor_id);
                    }
                }
                if bfs_nodes.len() >= num_nodes {
                    break;
                }
            }
        }

        ANNResult::Ok(bfs_nodes)
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
        Self::insert_processed_node_in_cache(node, vertex_provider, cache)
    }

    fn insert_processed_node_in_cache<AlignedReaderType>(
        node: &Data::VectorIdType,
        vertex_provider: &DiskVertexProvider<Data, AlignedReaderType>,
        cache: &mut Cache<Data>,
    ) -> ANNResult<()>
    where
        AlignedReaderType: AlignedFileReader,
    {
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

fn compose_hybrid_cache_nodes(
    bfs_nodes: &[u32],
    frequency_nodes: &[u32],
    cache_budget: usize,
    requested_bfs_nodes: usize,
    routing_root_nodes: usize,
    node_len: u64,
) -> ANNResult<(Vec<u32>, HybridCacheComposition)> {
    if routing_root_nodes > cache_budget {
        return Err(ANNError::log_index_config_error(
            "hybrid_cache_budget".into(),
            format!(
                "cache budget {cache_budget} cannot preserve {routing_root_nodes} routing roots"
            ),
        ));
    }
    if bfs_nodes.len() < routing_root_nodes {
        return Err(ANNError::log_index_error(format!(
            "BFS produced {} nodes, fewer than {routing_root_nodes} routing roots",
            bfs_nodes.len()
        )));
    }

    let bfs_target = requested_bfs_nodes
        .max(routing_root_nodes)
        .min(cache_budget);
    let mut selected = Vec::with_capacity(cache_budget);
    let mut seen = HashSet::with_capacity(cache_budget);

    for &node in bfs_nodes.iter().take(bfs_target) {
        if seen.insert(node) {
            selected.push(node);
        }
    }
    let actual_bfs_nodes = selected.len();

    let mut frequency_count = 0usize;
    for &node in frequency_nodes {
        if selected.len() == cache_budget {
            break;
        }
        if seen.insert(node) {
            selected.push(node);
            frequency_count += 1;
        }
    }

    let mut fallback_bfs_nodes = 0usize;
    for &node in bfs_nodes {
        if selected.len() == cache_budget {
            break;
        }
        if seen.insert(node) {
            selected.push(node);
            fallback_bfs_nodes += 1;
        }
    }

    if selected.len() != cache_budget {
        return Err(ANNError::log_index_error(format!(
            "hybrid cache composition produced {} unique nodes, expected {cache_budget}",
            selected.len()
        )));
    }
    let estimated_payload_bytes = node_len
        .checked_mul(selected.len() as u64)
        .ok_or_else(|| ANNError::log_index_error("hybrid cache payload size overflow"))?;

    Ok((
        selected,
        HybridCacheComposition {
            requested_bfs_nodes,
            routing_root_nodes,
            bfs_nodes: actual_bfs_nodes,
            frequency_nodes: frequency_count,
            fallback_bfs_nodes,
            total_nodes: cache_budget,
            estimated_payload_bytes,
        },
    ))
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
    fn test_hybrid_composition_deduplicates_and_uses_fallback() {
        let (nodes, composition) =
            compose_hybrid_cache_nodes(&[10, 20, 30, 40, 50], &[20, 60], 5, 2, 1, 100).unwrap();

        assert_eq!(nodes, [10, 20, 60, 30, 40]);
        assert_eq!(
            composition,
            HybridCacheComposition {
                requested_bfs_nodes: 2,
                routing_root_nodes: 1,
                bfs_nodes: 2,
                frequency_nodes: 1,
                fallback_bfs_nodes: 2,
                total_nodes: 5,
                estimated_payload_bytes: 500,
            }
        );
    }

    #[test]
    fn test_hybrid_composition_preserves_roots_at_zero_bfs_fraction() {
        let (nodes, composition) =
            compose_hybrid_cache_nodes(&[10, 20, 30, 40], &[30, 50, 60], 4, 0, 2, 1).unwrap();

        assert_eq!(nodes, [10, 20, 30, 50]);
        assert_eq!(composition.bfs_nodes, 2);
        assert_eq!(composition.frequency_nodes, 2);
        assert_eq!(composition.fallback_bfs_nodes, 0);
    }

    #[test]
    fn test_hybrid_composition_honors_full_bfs_endpoint() {
        let (nodes, composition) =
            compose_hybrid_cache_nodes(&[10, 20, 30], &[40, 50], 3, 3, 1, 1).unwrap();

        assert_eq!(nodes, [10, 20, 30]);
        assert_eq!(composition.bfs_nodes, 3);
        assert_eq!(composition.frequency_nodes, 0);
        assert_eq!(composition.fallback_bfs_nodes, 0);
    }

    #[test]
    fn test_hybrid_composition_rejects_budget_smaller_than_roots() {
        let error = compose_hybrid_cache_nodes(&[10, 20], &[30], 1, 0, 2, 1).unwrap_err();

        assert!(error
            .to_string()
            .contains("cannot preserve 2 routing roots"));
    }

    #[test]
    fn test_hybrid_cache_rejects_budget_larger_than_graph() {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));
        let mut factory = DiskVertexProviderFactory::<
            GraphDataF32VectorUnitData,
            VirtualAlignedReaderFactory<OverlayFS>,
        >::new(
            VirtualAlignedReaderFactory::new(TEST_INDEX_PATH.to_string(), storage_provider),
            CachingStrategy::None,
        )
        .unwrap();

        let error = factory
            .seed_cache_from_hybrid_nodes(&[0], &[], 257, 128)
            .unwrap_err();

        assert!(error
            .to_string()
            .contains("cache budget 257 exceeds graph point count 256"));
    }

    #[test]
    fn test_collect_nodes_via_bfs_matches_cached_adjacency_order() {
        let storage_provider = Arc::new(VirtualStorageProvider::new_overlay(test_data_root()));
        let factory = DiskVertexProviderFactory::<
            GraphDataF32VectorUnitData,
            VirtualAlignedReaderFactory<OverlayFS>,
        >::new(
            VirtualAlignedReaderFactory::new(TEST_INDEX_PATH.to_string(), storage_provider.clone()),
            CachingStrategy::None,
        )
        .unwrap();
        let start_node = factory.get_header().unwrap().metadata().medoid as u32;
        let actual = factory.collect_nodes_via_bfs(&[start_node], 10).unwrap();

        let fully_cached = DiskVertexProviderFactory::<
            GraphDataF32VectorUnitData,
            VirtualAlignedReaderFactory<OverlayFS>,
        >::new(
            VirtualAlignedReaderFactory::new(TEST_INDEX_PATH.to_string(), storage_provider),
            CachingStrategy::StaticCacheWithBfsNodes(256),
        )
        .unwrap();
        let cache = fully_cached.cache.as_ref().unwrap();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::from([start_node]);
        let mut expected = Vec::new();
        visited.insert(start_node);
        while let Some(node) = queue.pop_front() {
            expected.push(node);
            if expected.len() == 10 {
                break;
            }
            for &neighbor in cache.get_adjacency_list(&node).unwrap().iter() {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        assert_eq!(actual, expected);
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
