/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Spherical tile MaxSim using the existing MinMax integer SIMD operations.

use std::{
    num::NonZeroUsize,
    ops::{Add, Mul, Sub},
};

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_wide::{
    SIMDCast, SIMDMinMax, SIMDVector,
    arch::{Architecture, Target},
};

use crate::{
    bits::{PermutationStrategy, Representation, Unsigned},
    matrix_kernels::{
        Cache, Drive,
        blocks::{packed, unpacked},
        bounds, driver,
        num::{Bytes, DimK, Elements},
        ptr::Slice,
        util::LoadStore,
    },
    multi_vector::{BlockTransposed, BuildMaxSimError, MaxSimError, Overflow},
    spherical::{
        CompensatedIP, DataMeta, DataMetaF32, DataRef, QueryMeta, QueryRef,
        compensation::{self, DataTerms, QueryTerms},
    },
};

use super::{
    packed_f32_x_unpacked_f32::Params,
    packed_minmax_x_unpacked_minmax::{
        UnpackedCode, UnsignedDot, dot_tile, load_native, max_k, reduce_native_max,
    },
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum PrepareError {
    #[error(transparent)]
    Storage(#[from] BuildMaxSimError),
    #[error("Spherical MaxSim requires a positive dimension")]
    ZeroDimension,
    #[error("compensation context dim {context} does not equal query dim {query}")]
    ContextDimension { context: f32, query: usize },
    #[error("query row {row} has dimension {found}, expected {expected}")]
    RowDimension {
        row: usize,
        found: usize,
        expected: usize,
    },
}

impl From<Overflow> for PrepareError {
    fn from(error: Overflow) -> Self {
        BuildMaxSimError::Overflow(error).into()
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ComputeError {
    #[error(transparent)]
    Scores(#[from] MaxSimError),
    #[error(transparent)]
    Overflow(#[from] Overflow),
    #[error("document bit width {found} does not equal prepared width {expected}")]
    BitWidth { found: usize, expected: usize },
    #[error("document bit width {bits} exceeds backend maximum {maximum}")]
    BackendWidth { bits: usize, maximum: usize },
    #[error("canonical document row has {found} bytes, expected {expected}")]
    CanonicalRow { found: usize, expected: usize },
}

/// Internal query preparation, without a public matrix representation or ISA factory.
pub(crate) struct PreparedQuery<T: Copy, const MR: usize, const PACK: usize> {
    codes: BlockTransposed<T, MR, PACK>,
    meta: Vec<QueryPanel<MR>>,
    a_bits: usize,
    document_bits: usize,
}

fn validate_query<const Q: usize, const D: usize>(
    dim: usize,
    context: CompensatedIP,
    rows: impl Iterator<Item = usize>,
) -> Result<(), PrepareError> {
    if dim == 0 {
        return Err(PrepareError::ZeroDimension);
    }
    let limit = max_k(Q, D);
    if dim > limit {
        return Err(BuildMaxSimError::DimTooLarge(dim, limit).into());
    }
    if context.dim != dim as f32 {
        return Err(PrepareError::ContextDimension {
            context: context.dim,
            query: dim,
        });
    }
    for (row, found) in rows.enumerate() {
        if found != dim {
            return Err(PrepareError::RowDimension {
                row,
                found,
                expected: dim,
            });
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Compensation {
    Symmetric,
    Mixed,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct QueryPanel<const MR: usize> {
    correction: [f32; MR],
    bit_sum: [f32; MR],
    offset: [f32; MR],
    metric_specific: [f32; MR],
    document_offset: f32,
    squared_shift_norm: f32,
    mode: Compensation,
}

impl<const MR: usize> QueryPanel<MR> {
    fn new<const B_BITS: usize>(context: CompensatedIP, mode: Compensation) -> Self {
        Self {
            correction: [0.0; MR],
            bit_sum: [0.0; MR],
            offset: [0.0; MR],
            metric_specific: [0.0; MR],
            document_offset: DataMeta::offset_term::<B_BITS>(),
            squared_shift_norm: context.squared_shift_norm,
            mode,
        }
    }

    fn set(&mut self, row: usize, meta: QueryMeta) {
        self.correction[row] = meta.inner_product_correction;
        self.bit_sum[row] = meta.bit_sum;
        self.offset[row] = meta.offset;
        self.metric_specific[row] = meta.metric_specific;
    }
}

#[derive(Default, Clone, Copy)]
pub(crate) struct Document {
    raw: DataMeta,
    full: DataMetaF32,
}

#[inline(always)]
fn prepare_documents<W: Architecture>(arch: W, docs: &mut [Document]) {
    #[cfg(test)]
    tests::PREPARATIONS.with(|count| {
        if let Some(value) = count.get() {
            count.set(Some(value + docs.len()));
        }
    });
    for doc in docs {
        doc.full = doc.raw.to_full(arch);
    }
}

#[inline(always)]
fn reduce_native<I, F, const MR: usize, const NR: usize>(
    arch: F::Arch,
    acc: impl IntoIterator<Item = I>,
    query: &QueryPanel<MR>,
    docs: &[Document; NR],
    k: usize,
    start: usize,
    scores: &mut [f32],
) where
    I: SIMDCast<f32, Cast = F>,
    F: SIMDVector<Scalar = f32> + SIMDMinMax + Add<Output = F> + Sub<Output = F> + Mul<Output = F>,
{
    let q = QueryTerms {
        correction: load_native::<F>(arch, &query.correction[start..start + F::LANES]),
        bit_sum: load_native::<F>(arch, &query.bit_sum[start..start + F::LANES]),
        offset: load_native::<F>(arch, &query.offset[start..start + F::LANES]),
        metric_specific: load_native::<F>(arch, &query.metric_specific[start..start + F::LANES]),
    };
    let offset = F::splat(arch, query.document_offset);
    let dim = F::splat(arch, k as f32);
    let shift = F::splat(arch, query.squared_shift_norm);
    let document = |column: usize| {
        let d = docs[column].full;
        DataTerms {
            correction: F::splat(arch, d.inner_product_correction),
            bit_sum: F::splat(arch, d.bit_sum),
            metric_specific: F::splat(arch, d.metric_specific),
        }
    };
    // The closed compensation choice is outside the integer K loop and column loop.
    match query.mode {
        Compensation::Symmetric => {
            let q = DataTerms {
                correction: q.correction,
                bit_sum: q.bit_sum,
                metric_specific: q.metric_specific,
            };
            let offset_squared = F::splat(arch, query.document_offset * query.document_offset);
            reduce_native_max(
                arch,
                acc,
                scores,
                #[inline(always)]
                |dot, column| {
                    compensation::symmetric(
                        dot,
                        q,
                        document(column),
                        offset,
                        offset_squared,
                        dim,
                        shift,
                    )
                },
            );
        }
        Compensation::Mixed => {
            reduce_native_max(
                arch,
                acc,
                scores,
                #[inline(always)]
                |dot, column| compensation::mixed(dot, q, document(column), offset, dim, shift),
            );
        }
    }
}

impl<T: Copy + Default + From<u8>, const MR: usize, const PACK: usize> PreparedQuery<T, MR, PACK> {
    pub(crate) fn symmetric<W: Architecture, const BITS: usize>(
        arch: W,
        query: &[DataRef<'_, BITS>],
        k: usize,
        context: CompensatedIP,
    ) -> Result<Self, PrepareError>
    where
        Unsigned: Representation<BITS>,
    {
        validate_query::<BITS, BITS>(k, context, query.iter().map(|row| row.len()))?;
        let mut packed = Self::try_new(
            query.len(),
            k,
            BITS,
            QueryPanel::new::<BITS>(context, Compensation::Symmetric),
            BITS,
        )?;
        for (i, row) in query.iter().enumerate() {
            let m = row.meta().to_full(arch);
            packed.meta[i / MR].set(
                i % MR,
                QueryMeta {
                    inner_product_correction: m.inner_product_correction,
                    bit_sum: m.bit_sum,
                    offset: 0.0,
                    metric_specific: m.metric_specific,
                },
            );
            let codes = row.vector();
            packed.pack_row(
                i,
                (0..k).map(|j| {
                    // SAFETY: Canonical rows expose exactly the logical K codes.
                    (unsafe { codes.get_unchecked(j) }) as u8
                }),
            );
        }
        Ok(packed)
    }

    pub(crate) fn mixed<const Q: usize, const D: usize, P>(
        query: &[QueryRef<'_, Q, P>],
        k: usize,
        context: CompensatedIP,
    ) -> Result<Self, PrepareError>
    where
        Unsigned: Representation<Q> + Representation<D>,
        P: PermutationStrategy<Q>,
    {
        validate_query::<Q, D>(k, context, query.iter().map(|row| row.len()))?;
        let mut packed = Self::try_new(
            query.len(),
            k,
            Q,
            QueryPanel::new::<D>(context, Compensation::Mixed),
            D,
        )?;
        for (i, row) in query.iter().enumerate() {
            packed.meta[i / MR].set(i % MR, row.meta());
            let codes = row.vector();
            packed.pack_row(
                i,
                (0..k).map(|j| {
                    // SAFETY: The logical accessor applies P before entering shared packed layout.
                    (unsafe { codes.get_unchecked(j) }) as u8
                }),
            );
        }
        Ok(packed)
    }

    /// Overwrite per-query scores with `min(-maximum_inner_product, f32::MAX)`.
    ///
    /// Query rows, canonical documents and context must originate from the same
    /// quantizer; canonical bytes do not carry provenance or a metric tag.
    /// NaNs are ignored, and empty/all-NaN documents produce `f32::MAX`.
    ///
    /// # Errors
    ///
    /// Bit width, logical dimension, canonical stride and score length are checked
    /// before any score write. Packed bytes alone cannot identify logical dimensions.
    pub(in crate::matrix_kernels) fn compute<W, const BITS: usize, const NR: usize>(
        &self,
        arch: W,
        documents: rowmajor::Ref<'_, u8>,
        dim: usize,
        scores: &mut [f32],
        cache: Cache,
    ) -> Result<(), ComputeError>
    where
        W: UnsignedDot<MR, PACK, AElem = T> + LoadStore<f32, MR>,
        Unsigned: Representation<BITS>,
        for<'a> PanelKernel<'a, W, MR, NR, PACK>: driver::PanelKernel,
    {
        if BITS != self.document_bits {
            return Err(ComputeError::BitWidth {
                found: BITS,
                expected: self.document_bits,
            });
        }
        if BITS > W::MAX_B_BITS {
            return Err(ComputeError::BackendWidth {
                bits: BITS,
                maximum: W::MAX_B_BITS,
            });
        }
        if scores.len() != self.codes.nrows() {
            return Err(MaxSimError::InvalidBufferLength(scores.len(), self.codes.nrows()).into());
        }
        if dim != self.codes.ncols() {
            return Err(MaxSimError::UnequalDim(dim, self.codes.ncols()).into());
        }
        let stride = DataRef::<BITS>::canonical_bytes(self.codes.ncols());
        if documents.ncols() != stride {
            return Err(ComputeError::CanonicalRow {
                found: documents.ncols(),
                expected: stride,
            });
        }
        if documents.nrows() == 0 {
            scores.fill(f32::MAX);
            return Ok(());
        }
        if scores.is_empty() {
            return Ok(());
        }
        let mut driver = Driver::<W, MR, NR, PACK>::from_canonical(
            arch,
            self,
            documents,
            scores,
            cache,
            BITS,
            decode_tile::<BITS, _>,
        );
        BScratch::<W::BElem>::check_layout::<PACK>(
            documents.nrows().min(driver.b_rows_per_tile.get()),
            driver.k,
        )?;
        driver.drive();
        for score in scores {
            *score = (-*score).min(f32::MAX);
        }
        Ok(())
    }
}

#[inline(never)]
#[expect(
    clippy::expect_used,
    reason = "driver supplies checked canonical rows and logical K"
)]
pub(crate) fn decode_tile<const BITS: usize, T: UnpackedCode>(
    source: rowmajor::Ref<'_, u8>,
    k: DimK,
    mut codes: rowmajor::Mut<'_, T>,
    metadata: &mut [Document],
) where
    Unsigned: Representation<BITS>,
{
    let dim = k.value().get();
    assert_eq!(source.nrows(), codes.nrows());
    assert_eq!(source.nrows(), metadata.len());
    assert!(dim <= codes.ncols());
    for ((bytes, dst), meta) in source.rows().zip(codes.rows_mut()).zip(metadata) {
        let row =
            DataRef::<BITS>::from_canonical_front(bytes, dim).expect("canonical document row");
        meta.raw = row.meta();
        row.vector()
            .decode_into(&mut dst[..dim], T::from_code)
            .expect("logical code count");
    }
}

/// Code-generation entry only; enabled by the existing assembly-inspection feature.
///
/// # Errors
///
/// Propagates query/context, document shape and score-buffer validation failures.
#[cfg(all(feature = "codegen", target_arch = "x86_64"))]
#[inline(never)]
pub fn codegen_v4(
    arch: diskann_wide::arch::x86_64::V4,
    query: &[QueryRef<'_, 4, crate::bits::Dense>],
    dim: usize,
    context: CompensatedIP,
    documents: rowmajor::Ref<'_, u8>,
    scores: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let kernel =
        PreparedQuery::<u8, 16, 4>::mixed::<4, 4, crate::bits::Dense>(query, dim, context)?;
    kernel.compute::<_, 4, 8>(arch, documents, dim, scores, Cache::detect())?;
    Ok(())
}

impl<T: Copy + Default + From<u8>, const MR: usize, const PACK: usize> PreparedQuery<T, MR, PACK> {
    pub(crate) fn try_new(
        rows: usize,
        k: usize,
        bits: usize,
        initial: QueryPanel<MR>,
        document_bits: usize,
    ) -> Result<Self, Overflow> {
        let codes = BlockTransposed::try_new(rows, k)?;
        let panels = codes.num_blocks();
        Overflow::check::<QueryPanel<MR>>(panels, 1)?;
        Ok(Self {
            codes,
            meta: vec![initial; panels],
            a_bits: bits,
            document_bits,
        })
    }

    #[expect(
        clippy::expect_used,
        reason = "adapters enumerate exactly the input rows"
    )]
    pub(crate) fn pack_row(&mut self, i: usize, codes: impl IntoIterator<Item = u8>) {
        let mut row = self.codes.get_row_mut(i).expect("query row in bounds");
        for (j, code) in codes.into_iter().enumerate() {
            row.set(j, T::from(code));
        }
    }

    fn panel_bytes(&self) -> Bytes {
        let codes = Elements::<T>::new(MR * self.codes.padded_ncols()).bytes();
        Bytes::new(codes.value() + size_of::<QueryPanel<MR>>())
    }
}

//--------//
// Driver //
//--------//

/// Each drive overwrites C; results do not accumulate across calls.
pub(crate) struct Driver<
    'a,
    W: UnsignedDot<MR, PACK>,
    const MR: usize,
    const NR: usize,
    const PACK: usize,
> {
    arch: W,
    a: packed::View<'a, W::AElem, MR, PACK>,
    meta: &'a [QueryPanel<MR>],
    k: DimK,
    b: rowmajor::Ref<'a, u8>,
    decode_tile: DecodeTileFn<W::BElem>,
    c: &'a mut [f32],
    pub(crate) b_rows_per_tile: NonZeroUsize,
}

impl<'a, W: UnsignedDot<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize>
    Driver<'a, W, MR, NR, PACK>
{
    // Keep initialization out of the bit-width-specific constructors.
    #[inline(never)]
    #[expect(clippy::expect_used)]
    pub(crate) fn from_canonical(
        arch: W,
        a: &'a PreparedQuery<W::AElem, MR, PACK>,
        b: rowmajor::Ref<'a, u8>,
        c: &'a mut [f32],
        cache: Cache,
        b_bits: usize,
        decode_tile: DecodeTileFn<W::BElem>,
    ) -> Self {
        assert!(
            b_bits <= W::MAX_B_BITS,
            "document bit width exceeds the backend's supported range"
        );
        let rows = a.codes.nrows();
        let a_bytes = a.panel_bytes();
        let a_bits = a.a_bits;
        let meta = &a.meta;
        let dim = a.codes.ncols();
        let k = DimK::new(NonZeroUsize::new(dim).expect("validated positive K"));
        let a =
            packed::View::from_block_transposed(a.codes.as_view()).expect("validated nonempty A");
        bounds::check_eq!(bounds::Bound::new(c.len()), rows);
        let max = max_k(a_bits, b_bits);
        assert!(
            dim <= max,
            "contraction dimension {dim} exceeds the unsigned {a_bits}-bit x {b_bits}-bit limit of {max}",
        );
        let b_rows_per_tile = Params::new(
            cache,
            a_bytes,
            BScratch::<W::BElem>::row_bytes::<PACK>(k),
            NR,
        )
        .b_cols_in_l1;
        Self {
            arch,
            a,
            meta,
            k,
            b,
            decode_tile,
            c,
            b_rows_per_tile,
        }
    }
}

impl<W, const MR: usize, const NR: usize, const PACK: usize> driver::Drive
    for Driver<'_, W, MR, NR, PACK>
where
    W: UnsignedDot<MR, PACK> + LoadStore<f32, MR>,
    for<'a> PanelKernel<'a, W, MR, NR, PACK>: driver::PanelKernel,
{
    fn drive(&mut self) {
        self.arch.run(self);
    }
}

impl<W, const MR: usize, const NR: usize, const PACK: usize> Target<W, ()>
    for &mut Driver<'_, W, MR, NR, PACK>
where
    W: UnsignedDot<MR, PACK> + LoadStore<f32, MR>,
    for<'a> PanelKernel<'a, W, MR, NR, PACK>: driver::PanelKernel,
{
    #[inline(always)]
    fn run(self, _: W) {
        self.c.fill(f32::NEG_INFINITY);
        let mut scratch = BScratch::<W::BElem>::default();
        let mut start = 0;
        while let Some(remaining) = NonZeroUsize::new(self.b.nrows() - start) {
            let count = remaining.min(self.b_rows_per_tile);
            #[expect(clippy::expect_used, reason = "count is bounded by the remaining rows")]
            let source = self
                .b
                .subview(start..start + count.get())
                .expect("B tile in bounds");
            scratch.prepare::<PACK>(source, self.k, self.decode_tile);
            prepare_documents(self.arch, &mut scratch.meta);
            let b = scratch.as_tile::<PACK>(self.k);
            // SAFETY: Preparation binds the same K and panel count to codes and metadata.
            unsafe {
                self.a.visit_panels(
                    self.k,
                    #[inline(always)]
                    |a, panel| {
                        let row = panel * MR;
                        let end = (row + MR).min(self.c.len());
                        let c = self.arch.load(&self.c[row..end]);
                        let kernel = PanelKernel::<W, MR, NR, PACK>::new(
                            self.arch,
                            a,
                            &self.meta[panel],
                            b,
                            c,
                        );
                        // Already inside the driver's target-feature entry.
                        let c = kernel.run(self.arch);
                        self.arch.store(c, &mut self.c[row..end]);
                    },
                );
            }
            start += count.get();
        }
    }
}

#[derive(Default)]
pub(crate) struct BScratch<T> {
    pub(crate) codes: Vec<T>,
    pub(crate) meta: Vec<Document>,
}

// Only decoding is specialized by bit width.
pub(crate) type DecodeTileFn<T> = fn(
    source: rowmajor::Ref<'_, u8>,
    k: DimK,
    codes: rowmajor::Mut<'_, T>,
    metadata: &mut [Document],
);

/// Codes and document metadata share row order; code rows are zero-padded to PACK.
#[derive(Clone, Copy)]
pub(crate) struct BTile<'a, T, const PACK: usize> {
    pub(crate) codes: unpacked::View<'a, T>,
    pub(crate) meta: &'a [Document],
    k: DimK,
    physical_k: DimK,
}

impl<T: Copy, const PACK: usize> BTile<'_, T, PACK> {
    /// # Safety
    ///
    /// `codes` must be the NR-row panel of this tile starting at `start`.
    #[inline(always)]
    unsafe fn bind_panel<'a, const NR: usize>(
        &'a self,
        codes: unpacked::Panel<'a, T, NR>,
        start: usize,
    ) -> BPanel<'a, T, NR, PACK> {
        // SAFETY: Preparation gives each code row one metadata record in the same order.
        let meta = unsafe { &*self.meta.as_ptr().add(start).cast::<[Document; NR]>() };
        BPanel {
            codes,
            meta,
            k: self.k,
            physical_k: self.physical_k,
        }
    }
}

struct BPanel<'a, T, const NR: usize, const PACK: usize> {
    codes: unpacked::Panel<'a, T, NR>,
    meta: &'a [Document; NR],
    k: DimK,
    physical_k: DimK,
}

#[inline]
#[expect(clippy::expect_used)]
fn padded_k<const PACK: usize>(k: DimK) -> DimK {
    DimK::new(
        NonZeroUsize::new(k.value().get().next_multiple_of(PACK))
            .expect("padding a positive dimension remains positive"),
    )
}

impl<T: Copy + Default> BScratch<T> {
    fn check_layout<const PACK: usize>(rows: usize, k: DimK) -> Result<(), Overflow> {
        Overflow::check::<T>(rows, padded_k::<PACK>(k).value().get())?;
        Overflow::check::<Document>(rows, 1)
    }

    /// Prepare a nonempty tile. The decoder writes logical columns; padding bypasses it.
    ///
    /// # Panics
    ///
    /// Panics if the source tile is empty or its output storage size overflows.
    #[expect(clippy::expect_used)]
    pub(crate) fn prepare<const PACK: usize>(
        &mut self,
        source: rowmajor::Ref<'_, u8>,
        k: DimK,
        decode: impl FnOnce(rowmajor::Ref<'_, u8>, DimK, rowmajor::Mut<'_, T>, &mut [Document]),
    ) -> BTile<'_, T, PACK> {
        let count = NonZeroUsize::new(source.nrows()).expect("B tile must be nonempty");
        let dim = k.value().get();
        let physical_k = padded_k::<PACK>(k);
        let stride = physical_k.value().get();
        self.codes.resize(count.get() * stride, T::default());
        self.meta.resize(count.get(), Document::default());
        let mut codes = rowmajor::Mut::try_from_data(&mut self.codes, count.get(), stride)
            .expect("resized tile storage matches its shape");
        // Empty per-row fills can still lower to memset calls.
        if dim < stride {
            for dst in codes.rows_mut() {
                dst[dim..].fill(T::default());
            }
        }
        decode(source, k, codes, &mut self.meta);
        self.as_tile::<PACK>(k)
    }

    #[expect(clippy::expect_used, reason = "prepare constructs only nonempty tiles")]
    fn as_tile<const PACK: usize>(&self, k: DimK) -> BTile<'_, T, PACK> {
        let count = NonZeroUsize::new(self.meta.len()).expect("prepared nonempty tile");
        let physical_k = padded_k::<PACK>(k);
        BTile {
            // SAFETY: Preparation initialized every row and its PACK padding.
            codes: unsafe { unpacked::View::new(Slice::new(&self.codes), count, physical_k) },
            meta: &self.meta,
            k,
            physical_k,
        }
    }

    fn row_bytes<const PACK: usize>(k: DimK) -> Bytes {
        let codes = Elements::<T>::new(padded_k::<PACK>(k).value().get()).bytes();
        Bytes::new(codes.value() + size_of::<Document>())
    }
}

//-------------//
// PanelKernel //
//-------------//

pub(crate) struct PanelKernel<
    'a,
    W: UnsignedDot<MR, PACK>,
    const MR: usize,
    const NR: usize,
    const PACK: usize,
> {
    arch: W,
    a: packed::Panel<'a, W::AElem, MR, PACK>,
    query: &'a QueryPanel<MR>,
    b: BTile<'a, W::BElem, PACK>,
    c: [f32; MR],
}

impl<'a, W: UnsignedDot<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize>
    PanelKernel<'a, W, MR, NR, PACK>
{
    /// # Safety
    ///
    /// A's logical dimension must equal B's.
    unsafe fn new(
        arch: W,
        a: packed::Panel<'a, W::AElem, MR, PACK>,
        query: &'a QueryPanel<MR>,
        b: BTile<'a, W::BElem, PACK>,
        c: [f32; MR],
    ) -> Self {
        bounds::check_eq!(a.k(), b.k);
        Self {
            arch,
            a,
            query,
            b,
            c,
        }
    }
}

impl<W: UnsignedDot<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize, const N: usize>
    unpacked::PanelVisitor<W::BElem, N> for &mut PanelKernel<'_, W, MR, NR, PACK>
{
    #[inline(always)]
    fn visit(&mut self, codes: unpacked::Panel<'_, W::BElem, N>, start: usize) {
        let this = &mut **self;
        // SAFETY: Both full and trailing panels come from this kernel's B tile.
        let b = unsafe { this.b.bind_panel(codes, start) };
        let mut kernel = MicroKernel {
            arch: this.arch,
            a: this.a,
            query: this.query,
            b,
            c: &mut this.c,
        };
        driver::MicroKernel::micro_kernel(&mut kernel);
    }
}

impl<W: UnsignedDot<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize>
    Target<W, [f32; MR]> for PanelKernel<'_, W, MR, NR, PACK>
where
    Self: driver::PanelKernel,
{
    #[inline(always)]
    fn run(mut self, _: W) -> [f32; MR] {
        driver::PanelKernel::panel_kernel(&mut self);
        self.c
    }
}

macro_rules! panel_kernel {
    ($nr:literal, [$($tail:literal),*]) => {
        impl<W: UnsignedDot<MR, PACK>, const MR: usize, const PACK: usize>
            driver::PanelKernel for PanelKernel<'_, W, MR, $nr, PACK>
        {
            #[inline(always)]
            fn panel_kernel(&mut self) {
                let b = self.b;
                let mut visitor = self;
                // SAFETY: Preparation constructs B with physical_k elements per row.
                let tail = unsafe {
                    b.codes.visit_panels::<$nr>(b.physical_k, &mut *visitor)
                };
                if let Some(tail) = tail {
                    $(
                        if let Some(panel) = tail.try_as_panel::<$tail>() {
                            unpacked::PanelVisitor::visit(&mut visitor, panel, tail.start());
                            return;
                        }
                    )*
                    unreachable!("all remainders below NR are handled");
                }
            }
        }
    };
}

panel_kernel!(2, [1]);
panel_kernel!(4, [1, 2, 3]);
panel_kernel!(8, [1, 2, 3, 4, 5, 6, 7]);

//-------------//
// MicroKernel //
//-------------//

struct MicroKernel<
    'a,
    W: UnsignedDot<MR, PACK>,
    const MR: usize,
    const NR: usize,
    const PACK: usize,
> {
    arch: W,
    a: packed::Panel<'a, W::AElem, MR, PACK>,
    query: &'a QueryPanel<MR>,
    b: BPanel<'a, W::BElem, NR, PACK>,
    c: &'a mut [f32; MR],
}

impl<W: UnsignedDot<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize>
    driver::MicroKernel for MicroKernel<'_, W, MR, NR, PACK>
{
    #[inline(always)]
    fn micro_kernel(&mut self) {
        let acc = dot_tile::<W, MR, NR, PACK>(self.arch, self.a, self.b.codes, self.b.physical_k);
        self.arch.visit_native(
            acc,
            #[inline(always)]
            |start, native| {
                let end = start + W::NativeFloat::LANES;
                reduce_native::<W::NativeAcc, W::NativeFloat, MR, NR>(
                    self.arch,
                    native,
                    self.query,
                    self.b.meta,
                    self.b.k.value().get(),
                    start,
                    &mut self.c[start..end],
                );
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bits::{BitTranspose, Dense},
        distances::MathematicalResult,
        meta::{VectorMut, VectorRef},
    };
    use diskann_wide::arch::Scalar;
    use std::num::NonZeroUsize;

    std::thread_local! {
        pub(super) static PREPARATIONS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    }

    fn matrix<const BITS: usize, M: bytemuck::Pod, P: PermutationStrategy<BITS>>(
        rows: usize,
        dim: usize,
        metadata: impl Fn(usize, u32) -> M,
    ) -> rowmajor::Owned<u8>
    where
        Unsigned: Representation<BITS>,
    {
        let mut docs = rowmajor::Owned::from_fn(
            rows,
            VectorRef::<BITS, Unsigned, M, P>::canonical_bytes(dim),
            |_| 0,
        );
        let max = (1u32 << BITS) - 1;
        for (i, bytes) in docs.rows_mut().enumerate() {
            let mut row =
                VectorMut::<BITS, Unsigned, M, P>::from_canonical_front_mut(bytes, dim).unwrap();
            let mut sum = 0;
            for j in 0..dim {
                let value = match (i + j) % 7 {
                    0 => 0,
                    1 => max,
                    2 => max / 2,
                    3 => max / 2 + 1,
                    _ => ((i * 17 + j * 29 + 11) as u32) & max,
                };
                row.vector_mut().set(j, value.into()).unwrap();
                sum += value;
            }
            row.set_meta(metadata(i, sum));
        }
        docs
    }

    fn data_meta(i: usize, sum: u32) -> DataMeta {
        DataMeta::new((i % 5 + 1) as f32 * 0.125, i as f32 * 0.25 - 2.0, sum).unwrap()
    }

    fn query_meta(i: usize, sum: u32) -> QueryMeta {
        QueryMeta {
            inner_product_correction: (i % 7 + 1) as f32 * 0.125,
            bit_sum: sum as f32,
            offset: i as f32 * 0.125 - 1.375,
            metric_specific: 1.0 - i as f32 * 0.375,
        }
    }

    fn documents(rows: usize, dim: usize) -> rowmajor::Owned<u8> {
        matrix::<4, DataMeta, Dense>(rows, dim, data_meta)
    }

    fn refs<const BITS: usize, M: bytemuck::Pod, P: PermutationStrategy<BITS>>(
        bytes: &rowmajor::Owned<u8>,
        dim: usize,
    ) -> Vec<VectorRef<'_, BITS, Unsigned, M, P>>
    where
        Unsigned: Representation<BITS>,
    {
        bytes
            .rows()
            .map(|row| VectorRef::from_canonical_front(row, dim).unwrap())
            .collect()
    }

    #[test]
    fn scratch_allocation_boundaries_include_padding_and_metadata() {
        let k = DimK::new(NonZeroUsize::new(65).unwrap());
        let rows = isize::MAX as usize / (68 * size_of::<i16>());
        assert!(BScratch::<i16>::check_layout::<4>(rows, k).is_ok());
        assert!(BScratch::<i16>::check_layout::<4>(rows + 1, k).is_err());

        let k = DimK::new(NonZeroUsize::new(1).unwrap());
        let rows = isize::MAX as usize / size_of::<Document>();
        assert!(BScratch::<u8>::check_layout::<4>(rows, k).is_ok());
        assert!(BScratch::<u8>::check_layout::<4>(rows + 1, k).is_err());
    }

    #[test]
    fn decoder_scratch_reuse_clears_padding_and_refreshes_metadata() {
        let scalar = Scalar::new();
        let mut scratch = BScratch::<i8>::default();
        for (rows, dim) in [(17, 65), (2, 3), (31, 128), (1, 1), (9, 65)] {
            let docs = documents(rows, dim);
            let k = DimK::new(NonZeroUsize::new(dim).unwrap());
            scratch.prepare::<4>(docs.as_view(), k, decode_tile::<4, _>);
            prepare_documents(scalar, &mut scratch.meta);
            let stride = dim.next_multiple_of(4);
            for (i, row) in refs::<4, DataMeta, Dense>(&docs, dim).iter().enumerate() {
                let expected = row.meta().to_full(scalar);
                assert_eq!(scratch.meta[i].raw, row.meta());
                assert_eq!(scratch.meta[i].full, expected);
                assert!(
                    scratch.codes[i * stride + dim..(i + 1) * stride]
                        .iter()
                        .all(|v| *v == 0)
                );
                for j in 0..dim {
                    assert_eq!(
                        scratch.codes[i * stride + j] as i64,
                        row.vector().get(j).unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn metadata_preparation_occurs_once_per_tile_not_per_query_panel() {
        for rows in [1, 17, 65] {
            let query = documents(rows, 65);
            let context = CompensatedIP::new(&[0.25; 65], 65);
            let packed = PreparedQuery::<u8, 16, 4>::symmetric(
                Scalar::new(),
                &refs::<4, DataMeta, Dense>(&query, 65),
                65,
                context,
            )
            .unwrap();
            for count in [1, 3, 4, 5, 17] {
                let docs = documents(count, 65);
                let mut scores = vec![0.0; rows];
                let mut driver = Driver::<Scalar, 16, 2, 4>::from_canonical(
                    Scalar::new(),
                    &packed,
                    docs.as_view(),
                    &mut scores,
                    Cache::detect(),
                    4,
                    decode_tile::<4, _>,
                );
                driver.b_rows_per_tile = NonZeroUsize::new(4).unwrap();
                PREPARATIONS.with(|value| value.set(Some(0)));
                driver.drive();
                PREPARATIONS.with(|value| {
                    assert_eq!(value.get(), Some(count));
                    value.set(None);
                });
            }
        }
    }

    const SHAPES: &[(usize, usize, usize)] = &[
        (0, 0, 65),
        (0, 9, 128),
        (1, 0, 1),
        (1, 1, 1),
        (7, 3, 2),
        (8, 4, 3),
        (15, 5, 4),
        (16, 7, 5),
        (17, 8, 63),
        (31, 9, 64),
        (33, 17, 65),
        (17, 513, 128),
    ];

    fn compare<W, const BITS: usize, const MR: usize, const NR: usize, const PACK: usize>(
        arch: W,
        kernel: &PreparedQuery<W::AElem, MR, PACK>,
        documents: &rowmajor::Owned<u8>,
        expected: &[f32],
    ) where
        W: UnsignedDot<MR, PACK> + LoadStore<f32, MR>,
        Unsigned: Representation<BITS>,
        for<'a> PanelKernel<'a, W, MR, NR, PACK>: driver::PanelKernel,
    {
        for initial in [f32::NAN, -1234.0] {
            let mut scores = vec![initial; expected.len()];
            kernel
                .compute::<W, BITS, NR>(
                    arch,
                    documents.as_view(),
                    kernel.codes.ncols(),
                    &mut scores,
                    Cache::detect(),
                )
                .unwrap();
            assert_eq!(
                scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "ISA={}",
                std::any::type_name::<W>(),
            );
        }
    }

    fn symmetric<W, const BITS: usize, const MR: usize, const NR: usize, const PACK: usize>(arch: W)
    where
        W: UnsignedDot<MR, PACK> + LoadStore<f32, MR>,
        Unsigned: Representation<BITS>,
        for<'a> PanelKernel<'a, W, MR, NR, PACK>: driver::PanelKernel,
        for<'a> CompensatedIP: diskann_wide::arch::Target2<
                Scalar,
                MathematicalResult<f32>,
                DataRef<'a, BITS>,
                DataRef<'a, BITS>,
            >,
    {
        for nonfinite in [false, true] {
            for &(rows, documents, dim) in SHAPES {
                let query = matrix::<BITS, DataMeta, Dense>(rows, dim, |i, sum| {
                    let mut meta = data_meta(i, sum);
                    if nonfinite {
                        meta.metric_specific = [
                            half::f16::ZERO,
                            half::f16::NAN,
                            half::f16::INFINITY,
                            half::f16::NEG_INFINITY,
                        ][i % 4];
                    }
                    meta
                });
                let docs = matrix::<BITS, DataMeta, Dense>(documents, dim, |i, sum| {
                    let mut meta = data_meta(i, sum);
                    if nonfinite && i % 3 == 0 {
                        meta.inner_product_correction = [
                            half::f16::ZERO,
                            half::f16::NAN,
                            half::f16::INFINITY,
                            half::f16::NEG_INFINITY,
                        ][i % 4];
                    }
                    meta
                });
                let query = refs::<BITS, DataMeta, Dense>(&query, dim);
                let document_refs = refs::<BITS, DataMeta, Dense>(&docs, dim);
                let context = CompensatedIP::new(&vec![0.25; dim], dim);
                let expected: Vec<_> = query
                    .iter()
                    .map(|q| {
                        let best = document_refs.iter().fold(f32::NEG_INFINITY, |best, d| {
                            let ip: MathematicalResult<f32> = Scalar::new().run2(context, *q, *d);
                            best.max(ip.unwrap().into_inner())
                        });
                        (-best).min(f32::MAX)
                    })
                    .collect();
                let kernel =
                    PreparedQuery::<W::AElem, MR, PACK>::symmetric(arch, &query, dim, context)
                        .unwrap();
                compare::<W, BITS, MR, NR, PACK>(arch, &kernel, &docs, &expected);
            }
        }
    }

    fn mixed<
        W,
        const Q: usize,
        const D: usize,
        P,
        const MR: usize,
        const NR: usize,
        const PACK: usize,
    >(
        arch: W,
    ) where
        W: UnsignedDot<MR, PACK> + LoadStore<f32, MR>,
        Unsigned: Representation<Q> + Representation<D>,
        P: PermutationStrategy<Q> + Copy,
        for<'a> PanelKernel<'a, W, MR, NR, PACK>: driver::PanelKernel,
        for<'a> CompensatedIP: diskann_wide::arch::Target2<
                Scalar,
                MathematicalResult<f32>,
                QueryRef<'a, Q, P>,
                DataRef<'a, D>,
            >,
    {
        for nonfinite in [false, true] {
            for &(rows, documents, dim) in SHAPES {
                let query = matrix::<Q, QueryMeta, P>(rows, dim, |i, sum| {
                    let mut meta = query_meta(i, sum);
                    if nonfinite {
                        meta.metric_specific =
                            [0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY][i % 4];
                    }
                    meta
                });
                let docs = matrix::<D, DataMeta, Dense>(documents, dim, |i, sum| {
                    let mut meta = data_meta(i, sum);
                    if nonfinite && i % 3 == 0 {
                        meta.metric_specific = half::f16::NAN;
                    }
                    meta
                });
                let query = refs::<Q, QueryMeta, P>(&query, dim);
                let document_refs = refs::<D, DataMeta, Dense>(&docs, dim);
                let context = CompensatedIP::new(&vec![0.25; dim], dim);
                let expected: Vec<_> = query
                    .iter()
                    .map(|q| {
                        let best = document_refs.iter().fold(f32::NEG_INFINITY, |best, d| {
                            let ip: MathematicalResult<f32> = Scalar::new().run2(context, *q, *d);
                            best.max(ip.unwrap().into_inner())
                        });
                        (-best).min(f32::MAX)
                    })
                    .collect();
                let kernel =
                    PreparedQuery::<W::AElem, MR, PACK>::mixed::<Q, D, P>(&query, dim, context)
                        .unwrap();
                compare::<W, D, MR, NR, PACK>(arch, &kernel, &docs, &expected);
            }
        }
    }

    macro_rules! backends {
        ($bits:literal, $run:ident, [$($arg:tt),*]) => {
            if $bits <= 7 {
                $run::<Scalar, $($arg,)* 16, 2, 4>(Scalar::new());
            } else {
                $run::<Scalar, $($arg,)* 8, 2, 2>(Scalar::new());
            }
            #[cfg(target_arch = "x86_64")]
            {
                use diskann_wide::arch::x86_64::{V3, V4};
                if let Some(arch) = V3::new_checked() {
                    if $bits <= 6 {
                        $run::<V3, $($arg,)* 16, 4, 4>(arch);
                    } else {
                        $run::<V3, $($arg,)* 16, 4, 2>(arch);
                    }
                }
                if let Some(arch) = V4::new_checked() {
                    if $bits <= 7 {
                        $run::<V4, $($arg,)* 16, 8, 4>(arch);
                    } else {
                        $run::<V4, $($arg,)* 16, 8, 2>(arch);
                    }
                }
            }
            #[cfg(target_arch = "aarch64")]
            if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
                $run::<_, $($arg,)* 8, 4, 4>(arch);
            }
        };
    }

    #[test]
    fn symmetric_1() {
        backends!(1, symmetric, [1]);
    }
    #[test]
    fn symmetric_2() {
        backends!(2, symmetric, [2]);
    }
    #[test]
    fn symmetric_3() {
        backends!(3, symmetric, [3]);
    }
    #[test]
    fn symmetric_4() {
        backends!(4, symmetric, [4]);
    }
    #[test]
    fn symmetric_8() {
        backends!(8, symmetric, [8]);
    }
    #[test]
    fn mixed_2() {
        backends!(2, mixed, [2, 2, Dense]);
    }
    #[test]
    fn mixed_3() {
        backends!(3, mixed, [3, 3, Dense]);
    }
    #[test]
    fn mixed_4() {
        backends!(4, mixed, [4, 4, Dense]);
    }
    #[test]
    fn mixed_8() {
        backends!(8, mixed, [8, 8, Dense]);
    }
    #[test]
    fn transposed_4_x_1() {
        backends!(1, mixed, [4, 1, BitTranspose]);
    }

    #[test]
    fn independent_formula_and_noninteger_winner() {
        let mut query = matrix::<4, QueryMeta, Dense>(1, 1, |_, _| QueryMeta {
            inner_product_correction: 2.0,
            bit_sum: 15.0,
            offset: -2.0,
            metric_specific: 3.0,
        });
        VectorMut::<4, Unsigned, QueryMeta>::from_canonical_front_mut(query.row_mut(0), 1)
            .unwrap()
            .vector_mut()
            .set(0, 15)
            .unwrap();
        let mut docs = matrix::<4, DataMeta, Dense>(2, 1, |i, _| {
            DataMeta::new(
                if i == 0 { 0.125 } else { 4.0 },
                1.0,
                if i == 0 { 15 } else { 14 },
            )
            .unwrap()
        });
        for (i, bytes) in docs.rows_mut().enumerate() {
            VectorMut::<4, Unsigned, DataMeta>::from_canonical_front_mut(bytes, 1)
                .unwrap()
                .vector_mut()
                .set(0, if i == 0 { 15 } else { 14 })
                .unwrap();
        }
        let expected: Vec<f32> = [225.0, 210.0]
            .into_iter()
            .enumerate()
            .map(|(i, dot)| {
                let correction = if i == 0 { 0.125 } else { 4.0 };
                let sum = if i == 0 { 15.0 } else { 14.0 };
                2.0 * correction * (dot - 7.5 * 15.0 + -2.0 * sum - 7.5 * -2.0) + 1.0 + 3.0 + 4.0
            })
            .collect();
        assert!(expected[1] > expected[0]);
        let kernel = PreparedQuery::<u8, 16, 4>::mixed::<4, 4, Dense>(
            &refs::<4, QueryMeta, Dense>(&query, 1),
            1,
            CompensatedIP::new(&[2.0], 1),
        )
        .unwrap();
        compare::<Scalar, 4, 16, 2, 4>(Scalar::new(), &kernel, &docs, &[-expected[1]]);
    }

    #[test]
    fn validation_is_atomic_and_accumulator_boundaries_are_checked() {
        let query = matrix::<8, QueryMeta, Dense>(2, 65, query_meta);
        let query = refs::<8, QueryMeta, Dense>(&query, 65);
        let context = CompensatedIP::new(&[0.25; 65], 65);
        let kernel = PreparedQuery::<i16, 8, 2>::mixed::<8, 8, Dense>(&query, 65, context).unwrap();
        let docs = matrix::<8, DataMeta, Dense>(0, 64, data_meta);
        let mut scores = [123.0; 2];
        assert!(
            kernel
                .compute::<Scalar, 8, 2>(
                    Scalar::new(),
                    docs.as_view(),
                    65,
                    &mut scores,
                    Cache::detect()
                )
                .is_err()
        );
        assert_eq!(scores, [123.0; 2]);
        let docs = matrix::<8, DataMeta, Dense>(0, 65, data_meta);
        let mut scores = [123.0; 3];
        assert!(
            kernel
                .compute::<Scalar, 8, 2>(
                    Scalar::new(),
                    docs.as_view(),
                    65,
                    &mut scores,
                    Cache::detect()
                )
                .is_err()
        );
        assert_eq!(scores, [123.0; 3]);
        let mut scores = [123.0; 2];
        assert!(
            kernel
                .compute::<Scalar, 4, 2>(
                    Scalar::new(),
                    docs.as_view(),
                    65,
                    &mut scores,
                    Cache::detect()
                )
                .is_err()
        );
        assert_eq!(scores, [123.0; 2]);
        let byte_kernel =
            PreparedQuery::<u8, 16, 4>::mixed::<8, 8, Dense>(&query, 65, context).unwrap();
        assert!(matches!(
            byte_kernel.compute::<Scalar, 8, 2>(
                Scalar::new(),
                docs.as_view(),
                65,
                &mut scores,
                Cache::detect()
            ),
            Err(ComputeError::BackendWidth { .. }),
        ));
        assert_eq!(scores, [123.0; 2]);
        let narrow_query = matrix::<4, QueryMeta, Dense>(2, 65, query_meta);
        let narrow = PreparedQuery::<u8, 16, 4>::mixed::<4, 4, Dense>(
            &refs::<4, QueryMeta, Dense>(&narrow_query, 65),
            65,
            context,
        )
        .unwrap();
        let ambiguous = documents(0, 66);
        assert_eq!(ambiguous.ncols(), documents(0, 65).ncols());
        assert!(
            narrow
                .compute::<Scalar, 4, 2>(
                    Scalar::new(),
                    ambiguous.as_view(),
                    66,
                    &mut scores,
                    Cache::detect(),
                )
                .is_err()
        );
        assert_eq!(scores, [123.0; 2]);
        assert!(matches!(
            PreparedQuery::<i16, 8, 2>::mixed::<8, 8, Dense>(
                &query,
                65,
                CompensatedIP::new(&[], 64)
            ),
            Err(PrepareError::ContextDimension { .. }),
        ));
        assert!(matches!(
            PreparedQuery::<i16, 8, 2>::mixed::<8, 8, Dense>(
                &query,
                64,
                CompensatedIP::new(&[], 64)
            ),
            Err(PrepareError::RowDimension { .. }),
        ));
        assert!(matches!(
            PreparedQuery::<i16, 8, 2>::mixed::<8, 8, Dense>(&[], 0, CompensatedIP::new(&[], 0)),
            Err(PrepareError::ZeroDimension),
        ));
        for dim in [max_k(8, 8), max_k(8, 8) + 1, usize::MAX] {
            let result = PreparedQuery::<i16, 8, 2>::mixed::<8, 8, Dense>(
                &[],
                dim,
                CompensatedIP::new(&[], dim),
            );
            assert_eq!(result.is_ok(), dim == max_k(8, 8));
        }
        assert!(rowmajor::Ref::try_from_data(&[] as &[u8], usize::MAX, 65).is_err());
    }

    #[test]
    fn query_is_owned_and_reusable_for_different_documents() {
        let query = matrix::<4, QueryMeta, Dense>(17, 65, query_meta);
        let context = CompensatedIP::new(&[0.25; 65], 65);
        let kernel = PreparedQuery::<u8, 16, 4>::mixed::<4, 4, Dense>(
            &refs::<4, QueryMeta, Dense>(&query, 65),
            65,
            context,
        )
        .unwrap();
        drop(query);
        let mut expected = [0.0; 17];
        let docs = documents(17, 65);
        kernel
            .compute::<Scalar, 4, 2>(
                Scalar::new(),
                docs.as_view(),
                65,
                &mut expected,
                Cache::detect(),
            )
            .unwrap();
        for rows in [513, 1, 17, 0, 9, 17] {
            let docs = documents(rows, 65);
            let mut scores = [0.0; 17];
            kernel
                .compute::<Scalar, 4, 2>(
                    Scalar::new(),
                    docs.as_view(),
                    65,
                    &mut scores,
                    Cache::detect(),
                )
                .unwrap();
            if rows == 17 {
                assert_eq!(scores, expected);
            }
            if rows == 0 {
                assert_eq!(scores, [f32::MAX; 17]);
            }
        }
    }

    #[test]
    fn actual_quantizer_rows_match_pairwise_inner_product_and_cosine() {
        use crate::{
            AsFunctor, CompressIntoWith,
            algorithms::TransformKind,
            alloc::{GlobalAllocator, ScopedAllocator},
            spherical::{PreScale, SphericalQuantizer, SupportedMetric},
        };
        use rand::{SeedableRng, rngs::StdRng};
        let values: Vec<_> = (0..8 * 65)
            .map(|i| ((i * 19 % 113) as f32 - 25.0) / 17.0)
            .collect();
        let source = rowmajor::Ref::try_from_data(&values, 8, 65).unwrap();
        let mut rng = StdRng::seed_from_u64(0x1673ac82);
        for metric in [SupportedMetric::InnerProduct, SupportedMetric::Cosine] {
            let quantizer = SphericalQuantizer::train(
                source,
                TransformKind::Null,
                metric,
                PreScale::None,
                &mut rng,
                GlobalAllocator,
            )
            .unwrap();
            let dim = quantizer.output_dim();
            let mut query = matrix::<4, QueryMeta, Dense>(8, dim, query_meta);
            let mut docs = documents(8, dim);
            for ((from, q), d) in source.rows().zip(query.rows_mut()).zip(docs.rows_mut()) {
                quantizer
                    .compress_into_with(
                        from,
                        VectorMut::<4, Unsigned, QueryMeta>::from_canonical_front_mut(q, dim)
                            .unwrap(),
                        ScopedAllocator::new(&GlobalAllocator),
                    )
                    .unwrap();
                quantizer
                    .compress_into_with(
                        from,
                        VectorMut::<4, Unsigned, DataMeta>::from_canonical_front_mut(d, dim)
                            .unwrap(),
                        ScopedAllocator::new(&GlobalAllocator),
                    )
                    .unwrap();
            }
            let context: CompensatedIP = quantizer.as_functor();
            let query = refs::<4, QueryMeta, Dense>(&query, dim);
            let document_refs = refs::<4, DataMeta, Dense>(&docs, dim);
            let expected: Vec<_> = query
                .iter()
                .map(|q| {
                    let best = document_refs.iter().fold(f32::NEG_INFINITY, |best, d| {
                        let ip: MathematicalResult<f32> = Scalar::new().run2(context, *q, *d);
                        best.max(ip.unwrap().into_inner())
                    });
                    (-best).min(f32::MAX)
                })
                .collect();
            let kernel =
                PreparedQuery::<u8, 16, 4>::mixed::<4, 4, Dense>(&query, dim, context).unwrap();
            compare::<Scalar, 4, 16, 2, 4>(Scalar::new(), &kernel, &docs, &expected);
            #[cfg(target_arch = "x86_64")]
            if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                compare::<_, 4, 16, 4, 4>(arch, &kernel, &docs, &expected);
            }
        }
    }

    #[test]
    #[ignore = "manual paired tile/pairwise timings, not a performance assertion"]
    fn manual_stage_timings() {
        use diskann_wide::arch::{Target, Target2};
        use std::{hint::black_box, time::Instant};

        fn measure(mut f: impl FnMut()) -> u128 {
            for _ in 0..10 {
                f();
            }
            let mut rounds = [0u128; 5];
            for round in &mut rounds {
                let start = Instant::now();
                for _ in 0..100 {
                    f();
                }
                *round = start.elapsed().as_nanos() / 100;
            }
            rounds.sort_unstable();
            rounds[2]
        }

        struct Pairwise<'a> {
            query: &'a [QueryRef<'a, 4, Dense>],
            documents: &'a [DataRef<'a, 4>],
            context: CompensatedIP,
            scores: &'a mut [f32],
        }

        impl<W: Architecture> Target<W, ()> for Pairwise<'_>
        where
            for<'a> CompensatedIP:
                Target2<W, MathematicalResult<f32>, QueryRef<'a, 4, Dense>, DataRef<'a, 4>>,
        {
            #[inline(always)]
            fn run(self, arch: W) {
                for (query, score) in self.query.iter().zip(self.scores) {
                    let best = self
                        .documents
                        .iter()
                        .fold(f32::NEG_INFINITY, |best, document| {
                            let ip: MathematicalResult<f32> =
                                arch.run2_inline(self.context, *query, *document);
                            best.max(ip.unwrap().into_inner())
                        });
                    *score = (-best).min(f32::MAX);
                }
            }
        }

        fn paired(
            mut tile: impl FnMut(),
            mut pairwise: impl FnMut(),
            iterations: usize,
        ) -> [(u128, u128); 9] {
            for _ in 0..10 {
                tile();
                pairwise();
            }
            let time = |f: &mut dyn FnMut()| {
                let start = Instant::now();
                for _ in 0..iterations {
                    f();
                }
                start.elapsed().as_nanos() / iterations as u128
            };
            std::array::from_fn(|round| {
                if round % 2 == 0 {
                    let tile_ns = time(&mut tile);
                    (tile_ns, time(&mut pairwise))
                } else {
                    let pairwise_ns = time(&mut pairwise);
                    (time(&mut tile), pairwise_ns)
                }
            })
        }

        fn stages<W: UnsignedDot<16, 4> + LoadStore<f32, 16>, const NR: usize>(
            arch: W,
            query_rows: usize,
            document_rows: usize,
            dim: usize,
        ) where
            for<'a> PanelKernel<'a, W, 16, NR, 4>: driver::PanelKernel,
            for<'a> CompensatedIP:
                Target2<W, MathematicalResult<f32>, QueryRef<'a, 4, Dense>, DataRef<'a, 4>>,
        {
            let query = matrix::<4, QueryMeta, Dense>(query_rows, dim, query_meta);
            let query = refs::<4, QueryMeta, Dense>(&query, dim);
            let docs = documents(document_rows, dim);
            let document_refs = refs::<4, DataMeta, Dense>(&docs, dim);
            let context = CompensatedIP::new(&vec![0.25; dim], dim);
            let q_time = measure(|| {
                black_box(
                    PreparedQuery::<W::AElem, 16, 4>::mixed::<4, 4, Dense>(
                        black_box(&query),
                        dim,
                        black_box(context),
                    )
                    .unwrap(),
                );
            });
            let kernel =
                PreparedQuery::<W::AElem, 16, 4>::mixed::<4, 4, Dense>(&query, dim, context)
                    .unwrap();
            let mut scores = vec![0.0; query_rows];
            let mut pairwise_scores = vec![0.0; query_rows];
            kernel
                .compute::<W, 4, NR>(arch, docs.as_view(), dim, &mut scores, Cache::detect())
                .unwrap();
            arch.run(Pairwise {
                query: &query,
                documents: &document_refs,
                context,
                scores: &mut pairwise_scores,
            });
            assert_eq!(
                scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                pairwise_scores
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
            );
            let iterations = (100_000_000 / (query_rows * document_rows * dim)).clamp(20, 2000);
            let rounds = paired(
                || {
                    black_box(&kernel)
                        .compute::<W, 4, NR>(
                            arch,
                            black_box(docs.as_view()),
                            dim,
                            black_box(&mut scores),
                            Cache::detect(),
                        )
                        .unwrap();
                    black_box(&scores);
                },
                || {
                    arch.run(Pairwise {
                        query: black_box(&query),
                        documents: black_box(&document_refs),
                        context: black_box(context),
                        scores: black_box(&mut pairwise_scores),
                    });
                    black_box(&pairwise_scores);
                },
                iterations,
            );
            let mut tile_times = rounds.map(|(tile, _)| tile);
            let mut pairwise_times = rounds.map(|(_, pairwise)| pairwise);
            tile_times.sort_unstable();
            pairwise_times.sort_unstable();
            let tile_time = tile_times[4];
            let pairwise_time = pairwise_times[4];
            let mut driver = Driver::<W, 16, NR, 4>::from_canonical(
                arch,
                &kernel,
                docs.as_view(),
                &mut scores,
                Cache::detect(),
                4,
                decode_tile::<4, _>,
            );
            let rows_per_tile = driver.b_rows_per_tile.get();
            driver.drive();
            let mut scratch = BScratch::<W::BElem>::default();
            let k = DimK::new(NonZeroUsize::new(dim).unwrap());
            let bytes = docs.as_view();
            let b_time = measure(|| {
                arch.run(
                    #[inline(always)]
                    || {
                        for start in (0..document_rows).step_by(rows_per_tile) {
                            let source = bytes
                                .subview(start..(start + rows_per_tile).min(document_rows))
                                .unwrap();
                            scratch.prepare::<4>(source, k, decode_tile::<4, _>);
                            prepare_documents(arch, &mut scratch.meta);
                            black_box(&scratch);
                        }
                    },
                );
            });
            println!(
                "ISA={}; Q={query_rows} D={document_rows} K={dim}; tile_rows={rows_per_tile}; iterations={iterations}; median ns: query_prepare={q_time}, B_decode_and_metadata={b_time}, tile={tile_time}, pairwise={pairwise_time}; speedup={:.3}; paired_ns={rounds:?}",
                std::any::type_name::<W>(),
                pairwise_time as f64 / tile_time as f64,
            );
        }

        for (queries, documents, dim) in [
            (1, 1, 128),
            (1, 513, 128),
            (8, 17, 65),
            (33, 513, 128),
            (128, 513, 128),
        ] {
            stages::<_, 2>(Scalar::new(), queries, documents, dim);
            #[cfg(target_arch = "x86_64")]
            if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                stages::<_, 4>(arch, queries, documents, dim);
            }
        }
    }
}
