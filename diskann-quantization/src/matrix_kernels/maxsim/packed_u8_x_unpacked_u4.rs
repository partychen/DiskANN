/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Raw dot products for a block-transposed `u8` query matrix and dense row-major `u4`
//! document matrix.
//!
//! This follows the `packed_f32_x_unpacked_f32` driver, panel-kernel, and micro-kernel
//! structure. Quantization metadata and correction are intentionally outside this kernel.

use std::num::NonZeroUsize;

use diskann_wide::arch::{Architecture, Scalar};

use crate::{
    matrix_kernels::{
        Cache,
        blocks::{packed, unpacked},
        bounds, driver,
        num::DimK,
        ptr::Slice,
    },
    multi_vector::BlockTransposedRef,
};

use super::packed_f32_x_unpacked_f32::Params;

pub(crate) const QUERY_K_PACK: usize = 2;
pub(crate) const SCALAR_MR: usize = 8;
pub(crate) const SCALAR_NR: usize = 2;
#[cfg(target_arch = "x86_64")]
pub(crate) const V3_MR: usize = 16;
#[cfg(target_arch = "x86_64")]
pub(crate) const V3_NR: usize = 4;

const MAX_PRODUCT: usize = u8::MAX as usize * 0x0f;
const MAX_SCALAR_DIM: usize = u32::MAX as usize / MAX_PRODUCT;
#[cfg(target_arch = "x86_64")]
const MAX_V3_DIM: usize = i32::MAX as usize / MAX_PRODUCT;

/// Errors returned when logical matrix dimensions do not match their backing storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShapeError {
    EmptyDimension,
    DimensionMismatch { query: usize, document: usize },
    DocumentStorage { expected: usize, actual: usize },
    OutputStorage { expected: usize, actual: usize },
    DimensionTooLarge { dimension: usize, maximum: usize },
    SizeOverflow,
}

/// A row-major matrix whose logical `u4` values are densely packed, low nibble first.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UnpackedU4Rows<'a> {
    data: &'a [u8],
    rows: usize,
    dim: usize,
    row_bytes: usize,
}

impl<'a> UnpackedU4Rows<'a> {
    pub(crate) fn new(data: &'a [u8], rows: usize, dim: usize) -> Result<Self, ShapeError> {
        if dim == 0 {
            return Err(ShapeError::EmptyDimension);
        }

        let row_bytes = dim.div_ceil(QUERY_K_PACK);
        let expected = rows
            .checked_mul(row_bytes)
            .ok_or(ShapeError::SizeOverflow)?;
        if data.len() != expected {
            return Err(ShapeError::DocumentStorage {
                expected,
                actual: data.len(),
            });
        }

        Ok(Self {
            data,
            rows,
            dim,
            row_bytes,
        })
    }
}

fn validate<const MR: usize>(
    query: BlockTransposedRef<'_, u8, MR, QUERY_K_PACK>,
    docs: UnpackedU4Rows<'_>,
    output: &[u32],
    maximum_dimension: usize,
) -> Result<(), ShapeError> {
    if query.ncols() != docs.dim {
        return Err(ShapeError::DimensionMismatch {
            query: query.ncols(),
            document: docs.dim,
        });
    }
    if query.ncols() > maximum_dimension {
        return Err(ShapeError::DimensionTooLarge {
            dimension: query.ncols(),
            maximum: maximum_dimension,
        });
    }

    let expected = query
        .nrows()
        .checked_mul(docs.rows)
        .ok_or(ShapeError::SizeOverflow)?;
    if output.len() != expected {
        return Err(ShapeError::OutputStorage {
            expected,
            actual: output.len(),
        });
    }

    Ok(())
}

/// Compute raw unsigned dot products with the portable scalar kernel.
///
/// Output is row-major with shape `query.nrows() x docs.rows`.
pub(crate) fn compute_scalar(
    query: BlockTransposedRef<'_, u8, SCALAR_MR, QUERY_K_PACK>,
    docs: UnpackedU4Rows<'_>,
    output: &mut [u32],
) -> Result<(), ShapeError> {
    compute::<_, SCALAR_MR, SCALAR_NR>(
        Scalar::new(),
        query,
        docs,
        output,
        MAX_SCALAR_DIM,
        Cache::detect(),
    )
}

/// Compute raw unsigned dot products with the x86-64 V3 kernel.
#[cfg(target_arch = "x86_64")]
pub(crate) fn compute_v3(
    arch: diskann_wide::arch::x86_64::V3,
    query: BlockTransposedRef<'_, u8, V3_MR, QUERY_K_PACK>,
    docs: UnpackedU4Rows<'_>,
    output: &mut [u32],
) -> Result<(), ShapeError> {
    compute::<_, V3_MR, V3_NR>(arch, query, docs, output, MAX_V3_DIM, Cache::detect())
}

fn compute<A, const MR: usize, const NR: usize>(
    arch: A,
    query: BlockTransposedRef<'_, u8, MR, QUERY_K_PACK>,
    docs: UnpackedU4Rows<'_>,
    output: &mut [u32],
    maximum_dimension: usize,
    cache: Cache,
) -> Result<(), ShapeError>
where
    A: Architecture,
    for<'a> Driver<'a, A, MR, NR>: driver::Drive,
{
    validate(query, docs, output, maximum_dimension)?;
    if query.nrows() == 0 || docs.rows == 0 {
        return Ok(());
    }

    let a_k = DimK::new(NonZeroUsize::new(query.padded_ncols()).ok_or(ShapeError::EmptyDimension)?);
    let b_k = DimK::new(NonZeroUsize::new(docs.row_bytes).ok_or(ShapeError::EmptyDimension)?);

    // SAFETY: `BlockTransposedRef` validates its storage. Its physical block stride is
    // `MR * padded_ncols`, exactly the packed view layout used here.
    let a = unsafe {
        packed::View::new(
            Slice::new(query.as_slice()),
            NonZeroUsize::new(query.num_blocks()).ok_or(ShapeError::EmptyDimension)?,
            a_k,
        )
    };
    // SAFETY: `UnpackedU4Rows::new` validates `data.len() == rows * row_bytes`.
    let b = unsafe {
        unpacked::View::new(
            Slice::new(docs.data),
            NonZeroUsize::new(docs.rows).ok_or(ShapeError::EmptyDimension)?,
            b_k,
        )
    };

    // SAFETY: Validation and the view constructors establish all driver invariants.
    let mut driver = unsafe {
        Driver::new(
            arch,
            a,
            b,
            output,
            query.nrows(),
            query.ncols(),
            a_k,
            b_k,
            cache,
        )
    };
    driver::Drive::drive(&mut driver);
    Ok(())
}

//--------//
// Driver //
//--------//

/// Driver for raw packed-`u8` by dense-`u4` matrix multiplication.
///
/// # Class Invariants
///
/// 1. `a.k() == a_k` and `b.k() == b_k`.
/// 2. `a_k == QUERY_K_PACK * b_k`.
/// 3. `logical_k` is either `a_k` or `a_k - 1`.
/// 4. `query_rows.div_ceil(MR) == a.blocks()`.
/// 5. `c.len() == query_rows * b.extent()`.
pub(crate) struct Driver<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::View<'a, u8, MR>,
    b: unpacked::View<'a, u8>,
    c: &'a mut [u32],
    query_rows: usize,
    logical_k: usize,
    a_k: DimK,
    b_k: DimK,
    params: Params,
}

impl<'a, A, const MR: usize, const NR: usize> Driver<'a, A, MR, NR> {
    /// # Safety
    ///
    /// The caller must uphold all class invariants.
    #[expect(
        clippy::too_many_arguments,
        reason = "dimensions carry independent safety invariants"
    )]
    unsafe fn new(
        arch: A,
        a: packed::View<'a, u8, MR>,
        b: unpacked::View<'a, u8>,
        c: &'a mut [u32],
        query_rows: usize,
        logical_k: usize,
        a_k: DimK,
        b_k: DimK,
        cache: Cache,
    ) -> Self {
        bounds::check_eq!(a.k(), a_k);
        bounds::check_eq!(b.k(), b_k);
        bounds::check_eq!(
            bounds::Bound::new(a_k.value().get()),
            QUERY_K_PACK * b_k.value().get()
        );
        bounds::check_eq!(
            bounds::Bound::new(a.blocks().get()),
            query_rows.div_ceil(MR)
        );
        bounds::check_eq!(bounds::Bound::new(c.len()), query_rows * b.extent().get());

        debug_assert!(logical_k == a_k.value().get() || logical_k + 1 == a_k.value().get());

        let params = Params::new(
            cache,
            a.block_stride(a_k).bytes(),
            b.stride(b_k).bytes(),
            NR,
        );
        Self {
            arch,
            a,
            b,
            c,
            query_rows,
            logical_k,
            a_k,
            b_k,
            params,
        }
    }
}

impl<A, const MR: usize, const NR: usize> driver::Drive for Driver<'_, A, MR, NR>
where
    A: Architecture,
    for<'a> PanelKernel<'a, A, MR, NR>: driver::PanelKernel,
{
    fn drive(&mut self) {
        self.arch.run(
            #[inline]
            || {
                let c_ncols = self.b.extent().get();
                let c = &mut *self.c;

                let on_a_panels = |a_panels: packed::View<'_, u8, MR>, a_block_base: usize| {
                    let on_b_panels = |b_panels: unpacked::View<'_, u8>, b_base: usize| {
                        let panel_kernel =
                            |a_panel: packed::Panel<'_, u8, MR>, a_block_offset: usize| {
                                let query_base = (a_block_base + a_block_offset) * MR;
                                let valid_queries = (self.query_rows - query_base).min(MR);

                                // SAFETY: Driver invariants are inherited by every sub-view and
                                // panel. The visitor offsets determine a valid output tile.
                                let mut kernel = unsafe {
                                    PanelKernel::new(
                                        self.arch,
                                        a_panel,
                                        b_panels,
                                        c,
                                        c_ncols,
                                        query_base,
                                        valid_queries,
                                        b_base,
                                        self.logical_k,
                                        self.a_k,
                                        self.b_k,
                                    )
                                };
                                driver::PanelKernel::panel_kernel(&mut kernel);
                            };

                        // SAFETY: The driver invariant establishes
                        // `a_panels.k() == a_k`.
                        unsafe { a_panels.visit_panels(self.a_k, panel_kernel) };
                    };

                    // SAFETY: The driver invariant establishes `self.b.k() == b_k`.
                    unsafe {
                        self.b
                            .visit_sub_views(self.params.b_cols_in_l1, self.b_k, on_b_panels);
                    }
                };

                // SAFETY: The driver invariant establishes `self.a.k() == a_k`.
                unsafe {
                    self.a
                        .visit_sub_views(self.params.a_panels_in_l2, self.a_k, on_a_panels);
                }
            },
        );
    }
}

//-------------//
// PanelKernel //
//-------------//

struct PanelKernel<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, u8, MR>,
    b: unpacked::View<'a, u8>,
    c: &'a mut [u32],
    c_ncols: usize,
    query_base: usize,
    valid_queries: usize,
    doc_base: usize,
    logical_k: usize,
    a_k: DimK,
    b_k: DimK,
}

impl<'a, A, const MR: usize, const NR: usize> PanelKernel<'a, A, MR, NR> {
    /// # Safety
    ///
    /// The panel and view dimensions must match `a_k` and `b_k`; output tile indices must
    /// be in bounds; and the packed dimensions must describe `logical_k`.
    #[expect(
        clippy::too_many_arguments,
        reason = "tile coordinates and dimensions are distinct"
    )]
    unsafe fn new(
        arch: A,
        a: packed::Panel<'a, u8, MR>,
        b: unpacked::View<'a, u8>,
        c: &'a mut [u32],
        c_ncols: usize,
        query_base: usize,
        valid_queries: usize,
        doc_base: usize,
        logical_k: usize,
        a_k: DimK,
        b_k: DimK,
    ) -> Self {
        bounds::check_eq!(a.k(), a_k);
        bounds::check_eq!(b.k(), b_k);
        debug_assert!(valid_queries <= MR);
        debug_assert!(logical_k == a_k.value().get() || logical_k + 1 == a_k.value().get());
        debug_assert!((query_base + valid_queries) * c_ncols <= c.len());
        debug_assert!(doc_base + b.extent().get() <= c_ncols);

        Self {
            arch,
            a,
            b,
            c,
            c_ncols,
            query_base,
            valid_queries,
            doc_base,
            logical_k,
            a_k,
            b_k,
        }
    }
}

struct Visitor<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, u8, MR>,
    c: &'a mut [u32],
    c_ncols: usize,
    query_base: usize,
    valid_queries: usize,
    doc_base: usize,
    logical_k: usize,
    a_k: DimK,
    b_k: DimK,
}

impl<A, const MR: usize, const NR: usize> unpacked::PanelVisitor<u8, NR> for Visitor<'_, A, MR, NR>
where
    A: Copy,
    for<'a> MicroKernel<'a, A, MR, NR>: driver::MicroKernel,
{
    #[inline(always)]
    fn visit(&mut self, b: unpacked::Panel<'_, u8, NR>, start: usize) {
        // SAFETY: Panel visitation preserves the dimensions established by the driver.
        let mut micro =
            unsafe { MicroKernel::new(self.arch, self.a, b, self.logical_k, self.a_k, self.b_k) };
        driver::MicroKernel::micro_kernel(&mut micro);

        let tile = micro.take();
        let doc_base = self.doc_base + start;
        for (doc_offset, column) in tile.iter().enumerate() {
            for (query_offset, &value) in column.iter().take(self.valid_queries).enumerate() {
                self.c[(self.query_base + query_offset) * self.c_ncols + doc_base + doc_offset] =
                    value;
            }
        }
    }
}

macro_rules! panel_kernel {
    ($arch:ty, $mr:literal, $nr:literal, [ $($ns:literal),+ $(,)? ]) => {
        impl driver::PanelKernel for PanelKernel<'_, $arch, $mr, $nr> {
            #[inline(always)]
            fn panel_kernel(&mut self) {
                let visitor = Visitor {
                    arch: self.arch,
                    a: self.a,
                    c: &mut *self.c,
                    c_ncols: self.c_ncols,
                    query_base: self.query_base,
                    valid_queries: self.valid_queries,
                    doc_base: self.doc_base,
                    logical_k: self.logical_k,
                    a_k: self.a_k,
                    b_k: self.b_k,
                };

                // SAFETY: The panel-kernel invariant establishes `self.b.k() == b_k`.
                let tail = unsafe { self.b.visit_panels::<$nr>(self.b_k, visitor) };
                if let Some(tail) = tail {
                    $(
                        const { assert!($ns < $nr) };
                        if let Some(b) = tail.try_as_panel::<$ns>() {
                            // SAFETY: The tail panel retains the parent view's dimensions.
                            let mut micro = unsafe {
                                MicroKernel::new(
                                    self.arch,
                                    self.a,
                                    b,
                                    self.logical_k,
                                    self.a_k,
                                    self.b_k,
                                )
                            };
                            driver::MicroKernel::micro_kernel(&mut micro);

                            let tile = micro.take();
                            let doc_base = self.doc_base + tail.start();
                            for (doc_offset, column) in tile.iter().enumerate() {
                                for (query_offset, &value) in
                                    column.iter().take(self.valid_queries).enumerate()
                                {
                                    self.c[(self.query_base + query_offset) * self.c_ncols
                                        + doc_base
                                        + doc_offset] = value;
                                }
                            }
                        }
                    )+
                }
            }
        }
    };
}

panel_kernel!(Scalar, 8, 2, [1]);

//--------------//
// MicroKernel  //
//--------------//

struct MicroKernel<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, u8, MR>,
    b: unpacked::Panel<'a, u8, NR>,
    c: [[u32; MR]; NR],
    logical_k: usize,
    a_k: DimK,
    b_k: DimK,
}

impl<'a, A, const MR: usize, const NR: usize> MicroKernel<'a, A, MR, NR> {
    /// # Safety
    ///
    /// `a.k() == a_k`, `b.k() == b_k`, `a_k == 2 * b_k`, and `logical_k` must be
    /// either `a_k` or `a_k - 1`.
    unsafe fn new(
        arch: A,
        a: packed::Panel<'a, u8, MR>,
        b: unpacked::Panel<'a, u8, NR>,
        logical_k: usize,
        a_k: DimK,
        b_k: DimK,
    ) -> Self {
        bounds::check_eq!(a.k(), a_k);
        bounds::check_eq!(b.k(), b_k);
        bounds::check_eq!(
            bounds::Bound::new(a_k.value().get()),
            QUERY_K_PACK * b_k.value().get()
        );
        debug_assert!(logical_k == a_k.value().get() || logical_k + 1 == a_k.value().get());

        Self {
            arch,
            a,
            b,
            c: [[0; MR]; NR],
            logical_k,
            a_k,
            b_k,
        }
    }

    fn take(self) -> [[u32; MR]; NR] {
        self.c
    }
}

#[inline(always)]
unsafe fn micro_kernel_scalar<const MR: usize, const NR: usize>(
    _arch: Scalar,
    a: packed::Panel<'_, u8, MR>,
    b: unpacked::Panel<'_, u8, NR>,
    c: &mut [[u32; MR]; NR],
    logical_k: usize,
    a_k: DimK,
    b_k: DimK,
) {
    bounds::check_eq!(a.k(), a_k);
    bounds::check_eq!(b.k(), b_k);

    let a = a.as_ptr().as_ptr();
    let b = b.as_ptr().as_ptr();
    let pairs = b_k.value().get();
    let odd_k = !logical_k.is_multiple_of(QUERY_K_PACK);

    // SAFETY: Panel invariants guarantee `MR * 2 * pairs` query bytes and `NR * pairs`
    // document bytes. Scalar validation bounds every accumulation to `u32`.
    unsafe {
        for pair in 0..pairs {
            for (doc, column) in c.iter_mut().enumerate() {
                let packed_doc = *b.add(doc * pairs + pair);
                let lo = u32::from(packed_doc & 0x0f);
                let hi = if odd_k && pair + 1 == pairs {
                    0
                } else {
                    u32::from(packed_doc >> 4)
                };

                for (query, value) in column.iter_mut().enumerate() {
                    let query_pair = a.add(pair * MR * QUERY_K_PACK + query * QUERY_K_PACK);
                    *value += u32::from(*query_pair) * lo + u32::from(*query_pair.add(1)) * hi;
                }
            }
        }
    }
}

macro_rules! scalar_micro_kernel {
    ($($nr:literal),+ $(,)?) => {
        $(
            impl driver::MicroKernel for MicroKernel<'_, Scalar, SCALAR_MR, $nr> {
                #[inline(always)]
                fn micro_kernel(&mut self) {
                    // SAFETY: The micro-kernel class invariants establish all dimensions.
                    unsafe {
                        micro_kernel_scalar(
                            self.arch,
                            self.a,
                            self.b,
                            &mut self.c,
                            self.logical_k,
                            self.a_k,
                            self.b_k,
                        );
                    }
                }
            }
        )+
    };
}

scalar_micro_kernel!(1, 2);

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;

    use diskann_wide::arch::x86_64::V3;

    panel_kernel!(V3, 16, 4, [1, 2, 3]);

    macro_rules! v3_micro_kernel {
        ($($nr:literal),+ $(,)?) => {
            $(
                impl driver::MicroKernel for MicroKernel<'_, V3, V3_MR, $nr> {
                    #[inline(always)]
                    fn micro_kernel(&mut self) {
                        // SAFETY: The micro-kernel class invariants establish all dimensions.
                        unsafe {
                            micro_kernel_v3(
                                self.arch,
                                self.a,
                                self.b,
                                &mut self.c,
                                self.logical_k,
                                self.a_k,
                                self.b_k,
                            );
                        }
                    }
                }
            )+
        };
    }

    v3_micro_kernel!(1, 2, 3, 4);

    #[inline(always)]
    unsafe fn micro_kernel_v3<const NR: usize>(
        _arch: V3,
        a: packed::Panel<'_, u8, V3_MR>,
        b: unpacked::Panel<'_, u8, NR>,
        c: &mut [[u32; V3_MR]; NR],
        logical_k: usize,
        a_k: DimK,
        b_k: DimK,
    ) {
        use std::arch::x86_64::{
            __m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi16_epi32,
            _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_maddubs_epi16, _mm256_set1_epi16,
            _mm256_setzero_si256, _mm256_storeu_si256,
        };

        bounds::check_eq!(a.k(), a_k);
        bounds::check_eq!(b.k(), b_k);

        let a = a.as_ptr().as_ptr();
        let b = b.as_ptr().as_ptr();
        let pairs = b_k.value().get();
        let odd_k = !logical_k.is_multiple_of(QUERY_K_PACK);

        // SAFETY: V3 supplies AVX2. Panel invariants guarantee complete query and document
        // panels. Each `vpmaddubsw` lane is at most `2 * 255 * 15 = 7,650`, so it cannot
        // saturate. V3 validation bounds completed dots to `i32::MAX`.
        unsafe {
            let zero = _mm256_setzero_si256();
            let mut low: [__m256i; NR] = [zero; NR];
            let mut high: [__m256i; NR] = [zero; NR];

            for pair in 0..pairs {
                let query =
                    _mm256_loadu_si256(a.add(pair * V3_MR * QUERY_K_PACK).cast::<__m256i>());

                for doc in 0..NR {
                    let packed_doc = *b.add(doc * pairs + pair);
                    let lo = packed_doc & 0x0f;
                    let hi = if odd_k && pair + 1 == pairs {
                        0
                    } else {
                        packed_doc >> 4
                    };
                    let document = _mm256_set1_epi16(i16::from(lo) | (i16::from(hi) << 8));
                    let partial = _mm256_maddubs_epi16(query, document);
                    let partial_low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(partial));
                    let partial_high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(partial, 1));
                    low[doc] = _mm256_add_epi32(low[doc], partial_low);
                    high[doc] = _mm256_add_epi32(high[doc], partial_high);
                }
            }

            for doc in 0..NR {
                let mut signed = [0i32; V3_MR];
                _mm256_storeu_si256(signed.as_mut_ptr().cast(), low[doc]);
                _mm256_storeu_si256(signed.as_mut_ptr().add(8).cast(), high[doc]);
                for (dst, value) in c[doc].iter_mut().zip(signed) {
                    debug_assert!(value >= 0);
                    *dst = value as u32;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        bits::{BoxedBitSlice, Unsigned},
        multi_vector::{BlockTransposed, MatRef, Standard},
    };

    fn query_values(rows: usize, dim: usize) -> Vec<u8> {
        (0..rows * dim)
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .collect()
    }

    fn doc_values(rows: usize, dim: usize) -> Vec<u8> {
        (0..rows * dim).map(|i| ((i * 7 + 3) % 16) as u8).collect()
    }

    fn pack_dense_u4_rows(values: &[u8], rows: usize, dim: usize) -> Vec<u8> {
        let row_bytes = dim.div_ceil(QUERY_K_PACK);
        let mut packed = Vec::with_capacity(rows * row_bytes);
        for values in values.chunks_exact(dim) {
            let mut row = BoxedBitSlice::<4, Unsigned>::new_boxed(dim);
            for (col, &value) in values.iter().enumerate() {
                row.set(col, i64::from(value)).unwrap();
            }
            packed.extend_from_slice(row.as_slice());
        }
        assert_eq!(packed.len(), rows * row_bytes);
        packed
    }

    fn reference(
        query: &[u8],
        query_rows: usize,
        docs: &[u8],
        doc_rows: usize,
        dim: usize,
    ) -> Vec<u32> {
        let mut output = vec![0u32; query_rows * doc_rows];
        for q in 0..query_rows {
            for d in 0..doc_rows {
                output[q * doc_rows + d] = (0..dim)
                    .map(|k| u32::from(query[q * dim + k]) * u32::from(docs[d * dim + k]))
                    .sum();
            }
        }
        output
    }

    fn run_scalar(query: &[u8], query_rows: usize, docs: UnpackedU4Rows<'_>) -> Vec<u32> {
        let query_mat =
            MatRef::new(Standard::<u8>::new(query_rows, docs.dim).unwrap(), query).unwrap();
        let packed = BlockTransposed::<u8, SCALAR_MR, QUERY_K_PACK>::from_matrix_view(
            query_mat.as_matrix_view(),
        );
        let mut output = vec![0; query_rows * docs.rows];
        compute_scalar(packed.as_view(), docs, &mut output).unwrap();
        output
    }

    #[cfg(target_arch = "x86_64")]
    fn run_v3(query: &[u8], query_rows: usize, docs: UnpackedU4Rows<'_>) -> Option<Vec<u32>> {
        let arch = diskann_wide::arch::x86_64::V3::new_checked()?;
        let query_mat =
            MatRef::new(Standard::<u8>::new(query_rows, docs.dim).unwrap(), query).unwrap();
        let packed = BlockTransposed::<u8, V3_MR, QUERY_K_PACK>::from_matrix_view(
            query_mat.as_matrix_view(),
        );
        let mut output = vec![0; query_rows * docs.rows];
        compute_v3(arch, packed.as_view(), docs, &mut output).unwrap();
        Some(output)
    }

    fn run_case(query_rows: usize, doc_rows: usize, dim: usize) {
        let query = query_values(query_rows, dim);
        let docs = doc_values(doc_rows, dim);
        let packed_docs = pack_dense_u4_rows(&docs, doc_rows, dim);
        let docs_view = UnpackedU4Rows::new(&packed_docs, doc_rows, dim).unwrap();
        let expected = reference(&query, query_rows, &docs, doc_rows, dim);

        assert_eq!(run_scalar(&query, query_rows, docs_view), expected);
        #[cfg(target_arch = "x86_64")]
        if let Some(actual) = run_v3(&query, query_rows, docs_view) {
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn scalar_and_v3_match_reference_at_panel_boundaries() {
        for query_rows in [1, 7, 8, 9, 15, 16, 17, 33] {
            for doc_rows in [1, 2, 3, 4, 5, 9] {
                for dim in [1, 2, 3, 31, 32, 33] {
                    run_case(query_rows, doc_rows, dim);
                }
            }
        }
    }

    #[test]
    fn accepts_repository_dense_bitslice_layout() {
        let docs = [1, 2, 3, 4, 5, 6];
        let packed = pack_dense_u4_rows(&docs, 2, 3);
        assert_eq!(packed, [0x21, 0x03, 0x54, 0x06]);
        run_case(3, 2, 3);
    }

    #[test]
    fn maximum_values_do_not_overflow_intermediate_lanes() {
        const QUERY_ROWS: usize = 17;
        const DOC_ROWS: usize = 5;
        const DIM: usize = 257;

        let query = vec![u8::MAX; QUERY_ROWS * DIM];
        let docs = vec![0x0f; DOC_ROWS * DIM];
        let packed_docs = pack_dense_u4_rows(&docs, DOC_ROWS, DIM);
        let docs_view = UnpackedU4Rows::new(&packed_docs, DOC_ROWS, DIM).unwrap();
        let expected = u32::from(u8::MAX) * 0x0f * DIM as u32;

        let scalar = run_scalar(&query, QUERY_ROWS, docs_view);
        assert!(scalar.iter().all(|&value| value == expected));
        #[cfg(target_arch = "x86_64")]
        if let Some(actual) = run_v3(&query, QUERY_ROWS, docs_view) {
            assert_eq!(actual, scalar);
        }
    }

    #[test]
    fn ignores_unused_high_nibble_for_odd_dimensions() {
        const QUERY_ROWS: usize = 9;
        const DOC_ROWS: usize = 5;
        const DIM: usize = 3;

        let query = query_values(QUERY_ROWS, DIM);
        let docs = doc_values(DOC_ROWS, DIM);
        let mut packed_docs = pack_dense_u4_rows(&docs, DOC_ROWS, DIM);
        for row in 0..DOC_ROWS {
            packed_docs[row * DIM.div_ceil(QUERY_K_PACK) + 1] |= 0xf0;
        }
        let docs_view = UnpackedU4Rows::new(&packed_docs, DOC_ROWS, DIM).unwrap();
        let expected = reference(&query, QUERY_ROWS, &docs, DOC_ROWS, DIM);

        assert_eq!(run_scalar(&query, QUERY_ROWS, docs_view), expected);
        #[cfg(target_arch = "x86_64")]
        if let Some(actual) = run_v3(&query, QUERY_ROWS, docs_view) {
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn rejects_invalid_shapes() {
        assert_eq!(
            UnpackedU4Rows::new(&[], 1, 0).unwrap_err(),
            ShapeError::EmptyDimension
        );
        assert_eq!(
            UnpackedU4Rows::new(&[0; 3], 2, 4).unwrap_err(),
            ShapeError::DocumentStorage {
                expected: 4,
                actual: 3,
            }
        );
        assert_eq!(
            UnpackedU4Rows::new(&[], usize::MAX, 3).unwrap_err(),
            ShapeError::SizeOverflow
        );

        let query = query_values(2, 4);
        let query_mat = MatRef::new(Standard::<u8>::new(2, 4).unwrap(), &query).unwrap();
        let packed = BlockTransposed::<u8, SCALAR_MR, QUERY_K_PACK>::from_matrix_view(
            query_mat.as_matrix_view(),
        );
        let docs = UnpackedU4Rows::new(&[0; 2], 1, 4).unwrap();
        assert_eq!(
            compute_scalar(packed.as_view(), docs, &mut [0; 1]),
            Err(ShapeError::OutputStorage {
                expected: 2,
                actual: 1,
            })
        );

        let mismatched = UnpackedU4Rows::new(&[0; 2], 1, 3).unwrap();
        assert_eq!(
            compute_scalar(packed.as_view(), mismatched, &mut [0; 2]),
            Err(ShapeError::DimensionMismatch {
                query: 4,
                document: 3,
            })
        );
    }

    #[test]
    fn accumulator_dimension_limits_are_exact() {
        assert!(MAX_SCALAR_DIM * MAX_PRODUCT <= u32::MAX as usize);
        assert!((MAX_SCALAR_DIM + 1) * MAX_PRODUCT > u32::MAX as usize);

        #[cfg(target_arch = "x86_64")]
        {
            assert!(MAX_V3_DIM * MAX_PRODUCT <= i32::MAX as usize);
            assert!((MAX_V3_DIM + 1) * MAX_PRODUCT > i32::MAX as usize);
        }
    }
}
