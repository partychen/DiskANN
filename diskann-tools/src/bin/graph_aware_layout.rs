/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fs::File,
    io::{self, Write},
    path::PathBuf,
};

use anyhow::{bail, Context, Result};
use clap::Parser;
use diskann_disk::layout::{
    estimate_graph_aware_layout, rewrite_graph_aware_layout, LayoutEstimate,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Rewrite a disk graph into deterministic graph-aware physical layout 1.1")]
struct Args {
    #[arg(long)]
    source_index: PathBuf,
    #[arg(long)]
    output_index: PathBuf,
    #[arg(long)]
    report: PathBuf,
    #[arg(long)]
    estimate_only: bool,
}

#[derive(Serialize)]
struct EstimateReport<'a> {
    source_index: String,
    output_index: String,
    sidecar: String,
    estimate: &'a LayoutEstimate,
}

fn write_new_json(path: &PathBuf, value: &impl Serialize) -> Result<()> {
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("refusing to overwrite report {}", path.display()))?;
    serde_json::to_writer_pretty(&mut file, value)
        .with_context(|| format!("writing report {}", path.display()))?;
    writeln!(file).with_context(|| format!("finishing report {}", path.display()))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.report.exists() {
        bail!("refusing to overwrite report {}", args.report.display());
    }

    if args.estimate_only {
        let estimate = estimate_graph_aware_layout(&args.source_index)?;
        let sidecar = diskann_disk::layout::default_sidecar_path(&args.output_index);
        let report = EstimateReport {
            source_index: args.source_index.display().to_string(),
            output_index: args.output_index.display().to_string(),
            sidecar: sidecar.display().to_string(),
            estimate: &estimate,
        };
        serde_json::to_writer_pretty(io::stdout().lock(), &report)?;
        println!();
        write_new_json(&args.report, &report)?;
    } else {
        let report = rewrite_graph_aware_layout(&args.source_index, &args.output_index)?;
        serde_json::to_writer_pretty(io::stdout().lock(), &report)?;
        println!();
        write_new_json(&args.report, &report)?;
    }
    Ok(())
}
