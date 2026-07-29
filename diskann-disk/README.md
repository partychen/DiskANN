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

To cluster Search-PQ reconstructed vectors instead of the original data, pass
`--pq-pivots-file`, `--pq-compressed-file`, and `--distance l2` or
`--distance innerproduct`.

#### Graph-aware physical layout

The `graph_aware_layout` tool rewrites an existing disk graph without rebuilding Vamana or
Search-PQ. Graph adjacency lists and every externally visible node ID remain logical IDs. Layout
1.1 stores only node records in a different physical order and uses a checksummed
`<output-index>.layout` sidecar for logical-to-physical translation.

Estimate space and memory without creating index outputs:

```bash
cargo run --release -p diskann-tools --bin graph_aware_layout -- \
  --source-index /path/to/source_disk.index \
  --output-index /path/to/graph_aware_disk.index \
  --report /path/to/layout_estimate.json \
  --estimate-only
```

Remove `--estimate-only` and choose a new report path to perform the rewrite. The tool refuses to
overwrite the output index, sidecar, or report. The output has the same graph geometry and file size
as the source, so existing Search-PQ and routing sidecars remain valid under their original logical
IDs.

Packing is deterministic and uses no random seed. Each block starts with the lowest unassigned
logical ID. Remaining slots greedily select the unassigned node with the most outgoing edges from
nodes already in that block, breaking ties by lowest logical ID. Packed blocks are ordered from the
medoid block by greatest outgoing edge weight to an unplaced adjacent block, again breaking ties by
lowest original block ID; disconnected restarts use the lowest unplaced block.

Search loads the mapping only for layout 1.1. It deduplicates requested physical blocks and
coalesces adjacent blocks into range reads. `total_io_operations` counts read requests, while
`physical_blocks_read` and `physical_bytes_read` account for the full ranges. Legacy layouts retain
their direct logical-ID offset calculation and per-vertex read behavior.

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
