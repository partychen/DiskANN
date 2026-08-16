/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{utils::VectorRepr, ANNResult};
use diskann_utils::views::{MatrixView, MutMatrixView};

/// [`QuantCompressor`] defines the interface for quantizer with [`QuantDataGenerator`]
///
/// This trait serves as a general wrapper for different quantizers, allowing them to be
/// used interchangeably with QuantDataGenerator. Any type implementing this trait
/// can be used to compress vector data during the data generation process.
///
/// Preparation and compression are split across two types so that a compressor cannot be
/// observed in a half-initialized state: [`QuantCompressor`] only describes *how* to build a
/// quantizer, while [`QuantCompressor::prepare`] performs that work and hands back a
/// [`Compress`] implementation that is ready to use.
///
/// # Type Parameters
/// - `T`: The data type of the input vectors. Must impl Copy + Into<f32> + Pod + Sync
///   so that the [`QuantDataGenerator`] can parallelize computation, call compress_into and read from data file.
///
/// # Associated Types
/// - [`CompressorContext`]: An overloadable type that provides initialization parameters for the compressor
/// - [`Prepared`]: The ready-to-use compressor produced by [`prepare`](QuantCompressor::prepare)
///
/// [`QuantDataGenerator`]: crate::storage::quant::QuantDataGenerator
pub trait QuantCompressor<'a, T>: Sized
where
    T: VectorRepr,
{
    type CompressorContext: 'a;

    type Prepared: Compress + Sync;

    /// Record the parameters describing the quantizer.
    ///
    /// Construction is pure bookkeeping: it performs no I/O, runs no training, and cannot fail.
    fn new(context: &'a Self::CompressorContext) -> Self;

    /// Perform the work needed to obtain a usable quantizer, such as training and persisting a
    /// codebook or loading an existing one.
    ///
    /// # Errors
    ///
    /// Returns an error if the context is invalid or if the underlying preparation work
    /// (I/O, training, validation) fails.
    fn prepare(&self) -> ANNResult<Self::Prepared>;
}

/// A quantizer that is ready to compress vectors.
///
/// Implementations perform no I/O; obtaining one is the job of
/// [`QuantCompressor::prepare`].
pub trait Compress {
    /// Compress a batch of vectors into `output`.
    ///
    /// # Errors
    ///
    /// Returns an error if `vector` and `output` have incompatible shapes.
    fn compress(&self, vector: MatrixView<f32>, output: MutMatrixView<u8>) -> ANNResult<()>;

    /// The size in bytes of each compressed vector.
    fn compressed_bytes(&self) -> usize;
}
