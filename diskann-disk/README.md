# DiskANN Disk Index Crate

This crate provides disk-based indexing capabilities for DiskANN.

## Overview

The `disk-index` crate contains all the components specifically needed for building and searching disk-based indices:

## Structure

```text
src/
├── build/             # Disk index building pipeline
│   ├── builder/       # Core disk index builder and logic
│   ├── chunking/      # Checkpointing and continuation handling
│   └── configuration/ # Build parameters and quantization configuration
├── search/            # Disk index search infrastructure
│   ├── provider/      # Disk vertex providers and caching implementations
│   └── traits/        # Core traits for vertex providers and factories
├── data_model/        # Core data structures for disk indices
├── storage/           # Disk I/O operations and quantization
└── utils/             # Disk-specific utilities
```

## Implementation Status

This crate has been populated with the core disk index functionality from the main `diskann` crate. The refactor is complete with the following modules implemented:

### Build Module

- **Builder**: Core disk index builder, quantizer, and build operations
- **Chunking**: Checkpoint and continuation handling for large builds
- **Configuration**: Disk build parameters, filter parameters, and quantization types

### Search Module

- **Provider**: Disk vertex providers, caching implementations, and factory patterns
- **Traits**: Core traits for vertex providers and provider factories

#### Query-aware graph entry routing

An optional sidecar can select graph entry points from an in-memory set of real vectors. The disk
graph is unchanged, so the same index can be benchmarked with either its medoid or routed entries.
Routing vectors are stored as `f32` independently of the index's native vector type, preserving
Search-PQ reconstruction precision for `uint8`, `int8`, and `fp16` indexes.

Generate a table with k-means++ initialization and Lloyd iterations:

```bash
cargo run --release -p diskann-tools --bin generate_routing_table -- \
  --data-type float \
  --data-file /path/to/base.bin \
  --output-file /path/to/index.routing \
  --num-centers 256 \
  --sampling-rate 0.1 \
  --random-seed 42
```

The base file must be the dataset used to build the graph, in the same row order. Enable routing
after constructing a `DiskIndexSearcher`:

```rust
searcher.load_routing_table(
    "/path/to/index.routing",
    &storage_provider,
    NonZeroUsize::new(2).unwrap(),
)?;
```

`clear_routing_table()` restores medoid-based search. Compare `routing_time_us`,
`total_io_operations`, `search_hops`, recall, and end-to-end latency while sweeping the number of
centers and selected entries.

For representative-query cache training, the disk benchmark accepts `cache_sample_queries` together
with `num_nodes_to_cache`. It runs an untimed, uncached graph-search pass, ranks expanded frontier
nodes by visit frequency, and loads the exact Top-K IDs into the static cache before measurement.
Use a query sample that is separate from the measured query set. The output reports
`frontier_cache_hits`, `traversal_uncached_reads`, and `rerank_uncached_reads` separately.

To combine structural routing coverage with measured hot nodes, also set
`cache_bfs_fraction` in `[0, 1]`. The benchmark preserves every routing-table root, takes
the requested prefix of the deterministic multi-source BFS order, then appends unique
visit-frequency-ranked nodes. If those ranks do not fill `num_nodes_to_cache`, the
remaining BFS order fills the budget. The result records the requested fraction, effective
BFS/frequency/fallback composition, node coverage, and an estimated cached payload based on
the disk graph's node length. Cache setup remains outside measured query time.

To cluster Search-PQ reconstructed vectors instead of the original data, pass
`--pq-pivots-file`, `--pq-compressed-file`, and `--distance l2` or
`--distance innerproduct`.

### Data Model Module

- Graph headers, metadata, layout versioning, and caching structures

### Storage Module

- Disk I/O operations with reader and writer APIs
- Quantization compression and generation utilities

### Utils Module

- Disk-specific partitioning utilities

## Dependencies

This crate depends on:

- `diskann`: Core types and utilities
- `diskann-providers`: Main DiskANN library (including storage abstractions)
- `diskann-utils`: Utility functions
- `diskann-vector`: Vector operations
- `diskann-linalg`: Linear algebra operations
- `diskann-quantization`: Vector quantization
