/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use anyhow::Result;
use clap::Parser;
use diskann_disk::routing::{generate_routing_table, generate_routing_table_pq};
use diskann_providers::{
    storage::FileStorageProvider,
    utils::{create_rnd_from_optional_seed, create_thread_pool},
};
use diskann_tools::utils::{get_num_threads, DataType};
use diskann_vector::distance::Metric;
use diskann_vector::Half;

/// Generate query-routing entry points for a disk index.
#[derive(Debug, Parser)]
#[command(name = "generate_routing_table")]
struct Args {
    /// Raw dataset vector type. PQ-space routing sidecars always store float32 vectors.
    #[arg(long = "data-type", default_value = "float")]
    data_type: DataType,

    /// DiskANN binary dataset whose row IDs match the disk graph IDs.
    #[arg(long = "data-file", required_unless_present = "pq_pivots_file")]
    data_file: Option<String>,

    /// Output routing sidecar.
    #[arg(long = "output-file", required = true)]
    output_file: String,

    /// Number of k-means routing regions.
    #[arg(long = "num-centers", default_value = "256")]
    num_centers: NonZeroUsize,

    /// Fraction of the dataset used to train k-means.
    #[arg(long = "sampling-rate", default_value = "0.1")]
    sampling_rate: f64,

    /// Maximum Lloyd iterations.
    #[arg(long = "max-kmeans-reps", default_value = "10")]
    max_kmeans_reps: NonZeroUsize,

    /// Number of worker threads.
    #[arg(long = "num-threads", short = 'T')]
    num_threads: Option<usize>,

    /// Random seed for reproducible training.
    #[arg(long = "random-seed")]
    random_seed: Option<u64>,

    /// Search-PQ pivots file of an existing disk index. When set together with
    /// `--pq-compressed-file`, clustering runs in the index's quantized space.
    #[arg(long = "pq-pivots-file", requires = "pq_compressed_file")]
    pq_pivots_file: Option<String>,

    /// Search-PQ compressed-codes file of an existing disk index.
    #[arg(long = "pq-compressed-file", requires = "pq_pivots_file")]
    pq_compressed_file: Option<String>,

    /// Distance metric for routing construction. PQ-space routing defaults to innerproduct;
    /// raw-data routing supports only l2.
    #[arg(long = "distance")]
    distance: Option<Metric>,
}

fn generate<T>(args: &Args) -> Result<()>
where
    T: diskann::utils::VectorRepr,
{
    let storage = FileStorageProvider;
    let pool = create_thread_pool(get_num_threads(args.num_threads))?;
    let mut rng = create_rnd_from_optional_seed(args.random_seed);
    let table = match (&args.pq_pivots_file, &args.pq_compressed_file) {
        (Some(pivots), Some(compressed)) => generate_routing_table_pq(
            pivots,
            compressed,
            &args.output_file,
            args.distance.unwrap_or(Metric::InnerProduct),
            args.num_centers,
            args.sampling_rate,
            args.max_kmeans_reps,
            &storage,
            &mut rng,
            pool.as_ref(),
        )?,
        _ => {
            if args.distance.is_some_and(|metric| metric != Metric::L2) {
                anyhow::bail!("raw-data routing supports only --distance l2");
            }
            let data_file = args
                .data_file
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("--data-file is required without PQ inputs"))?;
            generate_routing_table::<T, _>(
                data_file,
                &args.output_file,
                args.num_centers,
                args.sampling_rate,
                args.max_kmeans_reps,
                &storage,
                &mut rng,
                pool.as_ref(),
            )?
        }
    };
    println!(
        "Wrote {} routing entries of dimension {} to {}",
        table.len(),
        table.dimension(),
        args.output_file
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.data_type {
        DataType::Float => generate::<f32>(&args),
        DataType::Int8 => generate::<i8>(&args),
        DataType::Uint8 => generate::<u8>(&args),
        DataType::Fp16 => generate::<Half>(&args),
    }
}
