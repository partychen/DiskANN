/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Raw dot products for packed `u8` queries and dense row-major `u4` documents.
//!
//! Quantization correction and MaxSim reduction are intentionally left to the caller.

use diskann_wide::arch::{Architecture, Scalar};

use crate::matrix_kernels::{
    Cache,
    blocks::{packed, unpacked},
    bounds, driver,
    num::{DimK, value_or_one},
};

use super::packed_f32_x_unpacked_f32::Params;

const PACK: usize = 2;
const MAX_PRODUCT: usize = u8::MAX as usize * 0x0f;

pub(crate) trait KernelArch {
    const MAX_DIM: usize;
}

impl KernelArch for Scalar {
    const MAX_DIM: usize = u32::MAX as usize / MAX_PRODUCT;
}

#[cfg(target_arch = "x86_64")]
impl KernelArch for diskann_wide::arch::x86_64::V3 {
    const MAX_DIM: usize = i32::MAX as usize / MAX_PRODUCT;
}

fn query_storage_k(k: DimK) -> DimK {
    DimK::new(value_or_one(k.value().get().div_ceil(PACK) * PACK))
}

fn document_storage_k(k: DimK) -> DimK {
    DimK::new(value_or_one(k.value().get().div_ceil(PACK)))
}

//--------//
// Driver //
//--------//

/// Driver for packed-`u8` by dense-`u4` raw dot products.
///
/// Results are written row-major into `c`.
///
/// # Class Invariants
///
/// 1. `a.k()` equals `k` rounded up to a multiple of two.
/// 2. `b.k()` equals `k.div_ceil(2)`.
/// 3. `c.len()` is a multiple of `b.extent()`.
/// 4. `c.len() / b.extent()` occupies exactly the packed blocks in `a`.
/// 5. The maximum dot product for `k` fits the accumulator used by `A`.
pub(crate) struct Driver<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::View<'a, u8, MR>,
    b: unpacked::View<'a, u8>,
    c: &'a mut [u32],
    k: DimK,
    params: Params,
}

impl<'a, A, const MR: usize, const NR: usize> Driver<'a, A, MR, NR>
where
    A: KernelArch,
{
    /// Prepare a raw matrix multiplication with results stored directly into `c`.
    ///
    /// `c` does not need any specific initial value.
    ///
    /// # Safety
    ///
    /// The caller must uphold all class invariants.
    pub(crate) unsafe fn new(
        arch: A,
        a: packed::View<'a, u8, MR>,
        b: unpacked::View<'a, u8>,
        c: &'a mut [u32],
        k: DimK,
        cache: Cache,
    ) -> Self {
        assert!(
            k.value().get() <= A::MAX_DIM,
            "dimension exceeds the kernel accumulator bound"
        );

        let query_k = query_storage_k(k);
        let document_k = document_storage_k(k);
        let n = b.extent().get();

        bounds::check_eq!(a.k(), query_k, "invalid packed query storage");
        bounds::check_eq!(b.k(), document_k, "invalid packed document storage");
        bounds::check_eq!(
            bounds::Bound::new(c.len() % n),
            0,
            "output must contain complete rows"
        );
        bounds::check_eq!(
            bounds::Bound::new((c.len() / n).div_ceil(MR)),
            a.blocks().get(),
            "output rows must occupy exactly the packed query blocks"
        );

        let params = Params::new(
            cache,
            a.block_stride(query_k).bytes(),
            b.stride(document_k).bytes(),
            NR,
        );

        Self {
            arch,
            a,
            b,
            c,
            k,
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
                let query_k = query_storage_k(self.k);
                let document_k = document_storage_k(self.k);
                let n = self.b.extent().get();
                let m = self.c.len() / n;
                let c = &mut *self.c;

                let on_a_panels = |a_panels: packed::View<'_, u8, MR>, a_block_base: usize| {
                    let on_b_panels = |b_panels: unpacked::View<'_, u8>, b_base: usize| {
                        let on_a_panel = |a: packed::Panel<'_, u8, MR>, a_block_offset: usize| {
                            let a_base = (a_block_base + a_block_offset) * MR;
                            let a_extent = (m - a_base).min(MR);

                            // SAFETY: The driver invariants are retained by all visited
                            // sub-views and panels. The visitor offsets identify a valid
                            // output tile.
                            let mut kernel = unsafe {
                                PanelKernel::new(
                                    self.arch, a, b_panels, c, n, a_base, a_extent, b_base, self.k,
                                )
                            };
                            driver::PanelKernel::panel_kernel(&mut kernel);
                        };

                        // SAFETY: `a_panels.k()` equals `query_k` by class invariant.
                        unsafe { a_panels.visit_panels(query_k, on_a_panel) };
                    };

                    // SAFETY: `self.b.k()` equals `document_k` by class invariant.
                    unsafe {
                        self.b
                            .visit_sub_views(self.params.b_cols_in_l1, document_k, on_b_panels);
                    }
                };

                // SAFETY: `self.a.k()` equals `query_k` by class invariant.
                unsafe {
                    self.a
                        .visit_sub_views(self.params.a_panels_in_l2, query_k, on_a_panels);
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
    c_stride: usize,
    a_base: usize,
    a_extent: usize,
    b_base: usize,
    k: DimK,
}

impl<'a, A, const MR: usize, const NR: usize> PanelKernel<'a, A, MR, NR> {
    /// # Safety
    ///
    /// The input dimensions and output tile must satisfy the [`Driver`] class invariants.
    #[expect(
        clippy::too_many_arguments,
        reason = "matrix tile coordinates are independent"
    )]
    unsafe fn new(
        arch: A,
        a: packed::Panel<'a, u8, MR>,
        b: unpacked::View<'a, u8>,
        c: &'a mut [u32],
        c_stride: usize,
        a_base: usize,
        a_extent: usize,
        b_base: usize,
        k: DimK,
    ) -> Self {
        bounds::check_eq!(a.k(), query_storage_k(k));
        bounds::check_eq!(b.k(), document_storage_k(k));
        bounds::check_ge!(bounds::Bound::new(MR), a_extent);
        bounds::check_ge!(bounds::Bound::new(c.len()), (a_base + a_extent) * c_stride);
        bounds::check_ge!(bounds::Bound::new(c_stride), b_base + b.extent().get());

        Self {
            arch,
            a,
            b,
            c,
            c_stride,
            a_base,
            a_extent,
            b_base,
            k,
        }
    }
}

struct Visitor<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, u8, MR>,
    c: &'a mut [u32],
    c_stride: usize,
    a_base: usize,
    a_extent: usize,
    b_base: usize,
    k: DimK,
}

fn write_tile<const MR: usize, const NR: usize>(
    c: &mut [u32],
    c_stride: usize,
    a_base: usize,
    a_extent: usize,
    b_base: usize,
    tile: [[u32; MR]; NR],
) {
    for (j, column) in tile.iter().enumerate() {
        for (i, &value) in column.iter().take(a_extent).enumerate() {
            c[(a_base + i) * c_stride + b_base + j] = value;
        }
    }
}

impl<A, const MR: usize, const NR: usize> unpacked::PanelVisitor<u8, NR> for Visitor<'_, A, MR, NR>
where
    A: Copy,
    for<'a> MicroKernel<'a, A, MR, NR>: driver::MicroKernel,
{
    #[inline(always)]
    fn visit(&mut self, b: unpacked::Panel<'_, u8, NR>, offset: usize) {
        // SAFETY: The panel visitor retains the dimensions established by `PanelKernel`.
        let mut micro = unsafe { MicroKernel::new(self.arch, self.a, b, self.k) };
        driver::MicroKernel::micro_kernel(&mut micro);
        write_tile(
            self.c,
            self.c_stride,
            self.a_base,
            self.a_extent,
            self.b_base + offset,
            micro.take(),
        );
    }
}

macro_rules! panel_kernel {
    ($arch:ty, $mr:literal, $nr:literal, [ $($tail:literal),+ $(,)? ]) => {
        impl driver::PanelKernel for PanelKernel<'_, $arch, $mr, $nr> {
            #[inline(always)]
            fn panel_kernel(&mut self) {
                let visitor = Visitor {
                    arch: self.arch,
                    a: self.a,
                    c: &mut *self.c,
                    c_stride: self.c_stride,
                    a_base: self.a_base,
                    a_extent: self.a_extent,
                    b_base: self.b_base,
                    k: self.k,
                };

                // SAFETY: `self.b.k()` equals `document_storage_k(self.k)`.
                let tail = unsafe {
                    self.b
                        .visit_panels::<$nr>(document_storage_k(self.k), visitor)
                };

                if let Some(tail) = tail {
                    $(
                        const { assert!($tail < $nr) };
                        if let Some(b) = tail.try_as_panel::<$tail>() {
                            // SAFETY: The tail retains the parent view's dimensions.
                            let mut micro = unsafe {
                                MicroKernel::new(self.arch, self.a, b, self.k)
                            };
                            driver::MicroKernel::micro_kernel(&mut micro);
                            write_tile(
                                self.c,
                                self.c_stride,
                                self.a_base,
                                self.a_extent,
                                self.b_base + tail.start(),
                                micro.take(),
                            );
                        }
                    )+
                }
            }
        }
    };
}

panel_kernel!(Scalar, 8, 2, [1]);

//-------------//
// MicroKernel //
//-------------//

struct MicroKernel<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, u8, MR>,
    b: unpacked::Panel<'a, u8, NR>,
    c: [[u32; MR]; NR],
    k: DimK,
}

impl<'a, A, const MR: usize, const NR: usize> MicroKernel<'a, A, MR, NR> {
    /// # Safety
    ///
    /// `a.k()` and `b.k()` must match the packed storage dimensions derived from `k`.
    unsafe fn new(
        arch: A,
        a: packed::Panel<'a, u8, MR>,
        b: unpacked::Panel<'a, u8, NR>,
        k: DimK,
    ) -> Self {
        bounds::check_eq!(a.k(), query_storage_k(k));
        bounds::check_eq!(b.k(), document_storage_k(k));
        Self {
            arch,
            a,
            b,
            c: [[0; MR]; NR],
            k,
        }
    }

    fn take(self) -> [[u32; MR]; NR] {
        self.c
    }
}

#[inline(always)]
unsafe fn scalar_micro_kernel<const MR: usize, const NR: usize>(
    _arch: Scalar,
    a: packed::Panel<'_, u8, MR>,
    b: unpacked::Panel<'_, u8, NR>,
    c: &mut [[u32; MR]; NR],
    k: DimK,
) {
    bounds::check_eq!(a.k(), query_storage_k(k));
    bounds::check_eq!(b.k(), document_storage_k(k));

    let ap = a.as_ptr().as_ptr();
    let bp = b.as_ptr().as_ptr();
    let pairs = document_storage_k(k).value().get();
    let odd_k = !k.value().get().is_multiple_of(PACK);

    // SAFETY: The panel bounds guarantee `MR * 2 * pairs` query bytes and `NR * pairs`
    // document bytes. `Driver` bounds completed products to `u32::MAX`.
    unsafe {
        for pair in 0..pairs {
            for (j, column) in c.iter_mut().enumerate() {
                let packed = *bp.add(j * pairs + pair);
                let lo = u32::from(packed & 0x0f);
                let hi = if odd_k && pair + 1 == pairs {
                    0
                } else {
                    u32::from(packed >> 4)
                };

                for (i, value) in column.iter_mut().enumerate() {
                    let query = ap.add(pair * MR * PACK + i * PACK);
                    *value += u32::from(*query) * lo + u32::from(*query.add(1)) * hi;
                }
            }
        }
    }
}

macro_rules! scalar_micro_kernels {
    ($($nr:literal),+ $(,)?) => {
        $(
            impl driver::MicroKernel for MicroKernel<'_, Scalar, 8, $nr> {
                #[inline(always)]
                fn micro_kernel(&mut self) {
                    // SAFETY: The micro-kernel class invariant establishes the dimensions.
                    unsafe {
                        scalar_micro_kernel(self.arch, self.a, self.b, &mut self.c, self.k);
                    }
                }
            }
        )+
    };
}

scalar_micro_kernels!(1, 2);

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;

    use diskann_wide::arch::x86_64::V3;

    panel_kernel!(V3, 16, 4, [1, 2, 3]);

    macro_rules! v3_micro_kernels {
        ($($nr:literal),+ $(,)?) => {
            $(
                impl driver::MicroKernel for MicroKernel<'_, V3, 16, $nr> {
                    #[inline(always)]
                    fn micro_kernel(&mut self) {
                        // SAFETY: The micro-kernel class invariant establishes the dimensions.
                        unsafe {
                            v3_micro_kernel(self.arch, self.a, self.b, &mut self.c, self.k);
                        }
                    }
                }
            )+
        };
    }

    v3_micro_kernels!(1, 2, 3, 4);

    #[inline(always)]
    unsafe fn v3_micro_kernel<const NR: usize>(
        _arch: V3,
        a: packed::Panel<'_, u8, 16>,
        b: unpacked::Panel<'_, u8, NR>,
        c: &mut [[u32; 16]; NR],
        k: DimK,
    ) {
        use std::arch::x86_64::{
            __m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi16_epi32,
            _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_maddubs_epi16, _mm256_set1_epi16,
            _mm256_setzero_si256, _mm256_storeu_si256,
        };

        bounds::check_eq!(a.k(), query_storage_k(k));
        bounds::check_eq!(b.k(), document_storage_k(k));

        let ap = a.as_ptr().as_ptr();
        let bp = b.as_ptr().as_ptr();
        let pairs = document_storage_k(k).value().get();
        let odd_k = !k.value().get().is_multiple_of(PACK);

        // SAFETY: V3 supplies AVX2. Each `vpmaddubsw` lane is at most
        // `2 * 255 * 15 = 7,650`, so it cannot saturate. `Driver` bounds completed dots
        // to `i32::MAX`, making conversion of the nonnegative lanes to `u32` exact.
        unsafe {
            let zero = _mm256_setzero_si256();
            let mut low: [__m256i; NR] = [zero; NR];
            let mut high: [__m256i; NR] = [zero; NR];

            for pair in 0..pairs {
                let query = _mm256_loadu_si256(ap.add(pair * 32).cast::<__m256i>());

                for j in 0..NR {
                    let packed = *bp.add(j * pairs + pair);
                    let lo = packed & 0x0f;
                    let hi = if odd_k && pair + 1 == pairs {
                        0
                    } else {
                        packed >> 4
                    };
                    let document = _mm256_set1_epi16(i16::from(lo) | (i16::from(hi) << 8));
                    let partial = _mm256_maddubs_epi16(query, document);
                    let partial_low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(partial));
                    let partial_high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(partial, 1));
                    low[j] = _mm256_add_epi32(low[j], partial_low);
                    high[j] = _mm256_add_epi32(high[j], partial_high);
                }
            }

            for j in 0..NR {
                let mut signed = [0i32; 16];
                _mm256_storeu_si256(signed.as_mut_ptr().cast(), low[j]);
                _mm256_storeu_si256(signed.as_mut_ptr().add(8).cast(), high[j]);
                for (dst, value) in c[j].iter_mut().zip(signed) {
                    debug_assert!(value >= 0);
                    *dst = value as u32;
                }
            }
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::num::NonZeroUsize;

    use diskann_utils::views::MatrixView;
    use diskann_wide::arch::Scalar;

    use crate::{
        bits::{BoxedBitSlice, Unsigned},
        matrix_kernels::{Drive, ptr::Slice, test_util::panic_message_for},
        multi_vector::{BlockTransposed, MatRef, Standard},
    };

    fn run<A, const MR: usize, const NR: usize>(
        arch: A,
        query: MatrixView<'_, u8>,
        docs: MatrixView<'_, u8>,
        k: DimK,
    ) -> Vec<u32>
    where
        A: KernelArch,
        for<'a> Driver<'a, A, MR, NR>: Drive,
    {
        let query = BlockTransposed::<u8, MR, PACK>::from_matrix_view(query);
        // SAFETY: `query` owns complete, padded `MR` by `PACK` blocks.
        let a = unsafe {
            packed::View::new(
                Slice::new(query.as_slice()),
                NonZeroUsize::new(query.num_blocks()).unwrap(),
                query_storage_k(k),
            )
        };
        let b = unpacked::View::from_matrix_view(docs).unwrap();
        let mut c = vec![0; query.nrows() * docs.nrows()];

        // SAFETY: The test constructors establish all driver invariants.
        let mut driver = unsafe { Driver::new(arch, a, b, &mut c, k, Cache::detect()) };
        driver.drive();
        c
    }

    fn query_values(rows: usize, dim: usize) -> Vec<u8> {
        (0..rows * dim)
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .collect()
    }

    fn document_values(rows: usize, dim: usize) -> Vec<u8> {
        (0..rows * dim).map(|i| ((i * 7 + 3) % 16) as u8).collect()
    }

    fn pack_documents(values: &[u8], rows: usize, dim: usize) -> Vec<u8> {
        let mut packed = Vec::with_capacity(rows * dim.div_ceil(PACK));
        for row in values.chunks_exact(dim) {
            let mut bits = BoxedBitSlice::<4, Unsigned>::new_boxed(dim);
            for (i, &value) in row.iter().enumerate() {
                bits.set(i, i64::from(value)).unwrap();
            }
            packed.extend_from_slice(bits.as_slice());
        }
        packed
    }

    fn reference(query: MatrixView<'_, u8>, docs: MatrixView<'_, u8>) -> Vec<u32> {
        query
            .row_iter()
            .flat_map(|query| {
                docs.row_iter().map(move |doc| {
                    query
                        .iter()
                        .zip(doc)
                        .map(|(&a, &b)| u32::from(a) * u32::from(b))
                        .sum()
                })
            })
            .collect()
    }

    fn check_case(query_rows: usize, document_rows: usize, dim: usize) {
        let query = query_values(query_rows, dim);
        let documents = document_values(document_rows, dim);
        let packed_documents = pack_documents(&documents, document_rows, dim);
        let query = MatRef::new(Standard::<u8>::new(query_rows, dim).unwrap(), &query).unwrap();
        let documents =
            MatRef::new(Standard::<u8>::new(document_rows, dim).unwrap(), &documents).unwrap();
        let packed_documents = MatRef::new(
            Standard::<u8>::new(document_rows, dim.div_ceil(PACK)).unwrap(),
            &packed_documents,
        )
        .unwrap();
        let k = DimK::new(NonZeroUsize::new(dim).unwrap());
        let expected = reference(query.as_matrix_view(), documents.as_matrix_view());

        let scalar = run::<_, 8, 2>(
            Scalar::new(),
            query.as_matrix_view(),
            packed_documents.as_matrix_view(),
            k,
        );
        assert_eq!(scalar, expected);

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            let actual = run::<_, 16, 4>(
                arch,
                query.as_matrix_view(),
                packed_documents.as_matrix_view(),
                k,
            );
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn matches_reference_at_panel_boundaries() {
        for m in [1, 7, 8, 9, 15, 16, 17, 33] {
            for n in [1, 2, 3, 4, 5, 9] {
                for k in [1, 2, 3, 31, 32, 33] {
                    check_case(m, n, k);
                }
            }
        }
    }

    #[test]
    fn handles_maximum_values_and_dirty_odd_nibble() {
        const M: usize = 17;
        const N: usize = 5;
        const K: usize = 257;

        let query = vec![u8::MAX; M * K];
        let documents = vec![0x0f; N * K];
        let mut packed_documents = pack_documents(&documents, N, K);
        for row in 0..N {
            packed_documents[row * K.div_ceil(PACK) + K / PACK] |= 0xf0;
        }

        let query = MatRef::new(Standard::<u8>::new(M, K).unwrap(), &query).unwrap();
        let packed_documents = MatRef::new(
            Standard::<u8>::new(N, K.div_ceil(PACK)).unwrap(),
            &packed_documents,
        )
        .unwrap();
        let k = DimK::new(NonZeroUsize::new(K).unwrap());
        let expected = u32::from(u8::MAX) * 0x0f * K as u32;

        let scalar = run::<_, 8, 2>(
            Scalar::new(),
            query.as_matrix_view(),
            packed_documents.as_matrix_view(),
            k,
        );
        assert!(scalar.iter().all(|&value| value == expected));

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            let actual = run::<_, 16, 4>(
                arch,
                query.as_matrix_view(),
                packed_documents.as_matrix_view(),
                k,
            );
            assert_eq!(actual, scalar);
        }
    }

    #[test]
    fn rejects_inconsistent_output_and_storage_dimensions() {
        let query = vec![0; 2 * 4];
        let query = MatRef::new(Standard::<u8>::new(2, 4).unwrap(), &query).unwrap();
        let query = BlockTransposed::<u8, 8, PACK>::from_matrix_view(query.as_matrix_view());
        // SAFETY: `query` owns one complete, padded `8` by `PACK` block.
        let a = unsafe {
            packed::View::new(
                Slice::new(query.as_slice()),
                NonZeroUsize::new(query.num_blocks()).unwrap(),
                DimK::new(NonZeroUsize::new(4).unwrap()),
            )
        };
        let documents = [0; 4];
        let documents = MatRef::new(Standard::<u8>::new(2, 2).unwrap(), &documents).unwrap();
        let b = unpacked::View::from_matrix_view(documents.as_matrix_view()).unwrap();
        let k = DimK::new(NonZeroUsize::new(4).unwrap());

        let message = panic_message_for(|| {
            let mut c = [0; 3];
            // SAFETY: The deliberate output mismatch is caught by checked test bounds.
            let _ =
                unsafe { Driver::<_, 8, 2>::new(Scalar::new(), a, b, &mut c, k, Cache::detect()) };
        });
        assert!(message.contains("complete rows"));

        let wrong_k = DimK::new(NonZeroUsize::new(2).unwrap());
        let message = panic_message_for(|| {
            let mut c = [0; 4];
            // SAFETY: The deliberate storage mismatch is caught by checked test bounds.
            let _ = unsafe {
                Driver::<_, 8, 2>::new(Scalar::new(), a, b, &mut c, wrong_k, Cache::detect())
            };
        });
        assert!(message.contains("invalid packed query storage"));
    }

    #[test]
    fn accumulator_bounds_are_exact() {
        assert!(Scalar::MAX_DIM * MAX_PRODUCT <= u32::MAX as usize);
        assert!((Scalar::MAX_DIM + 1) * MAX_PRODUCT > u32::MAX as usize);

        #[cfg(target_arch = "x86_64")]
        {
            use diskann_wide::arch::x86_64::V3;
            assert!(V3::MAX_DIM * MAX_PRODUCT <= i32::MAX as usize);
            assert!((V3::MAX_DIM + 1) * MAX_PRODUCT > i32::MAX as usize);
        }
    }
}
