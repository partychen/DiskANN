/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! This follows [`super::packed_f32_x_unpacked_f32`], with two adjacent `u8` values
//! represented as `[u8; 2]` and the matching pair of `u4` values represented as one byte.

use diskann_wide::arch::{Architecture, Scalar};

use crate::matrix_kernels::{
    Cache,
    blocks::{packed, unpacked},
    bounds, driver,
    num::{DimK, Elements},
    ptr::{MutSlice, Slice},
    util::{Fold, Folder},
};

use super::packed_f32_x_unpacked_f32::Params;

type U8Pair = [u8; 2];
const MAX_K_PAIRS: usize = u32::MAX as usize / (2 * u8::MAX as usize * 0x0f);

//--------//
// Driver //
//--------//

/// A driver for prepacked `u8` by unpacked `u4` maxsim computations.
///
/// Each element of `b` stores two `u4` values, low nibble first. Consequently, `k`
/// is measured in pairs and odd logical dimensions rely on the zero-padding in `a`.
pub(crate) struct Driver<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::View<'a, U8Pair, MR>,
    b: unpacked::View<'a, u8>,
    c: &'a mut [u32],
    k: DimK,
    params: Params,
}

impl<'a, A, const MR: usize, const NR: usize> Driver<'a, A, MR, NR> {
    /// # Safety
    ///
    /// 1. `a.k()` and `b.k()` must be equal to `k`.
    /// 2. `c.len().div_ceil(MR)` must be equal to `a.blocks()`.
    /// 3. `k` must not exceed [`MAX_K_PAIRS`].
    #[allow(dead_code)]
    pub(crate) unsafe fn new(
        arch: A,
        a: packed::View<'a, U8Pair, MR>,
        b: unpacked::View<'a, u8>,
        c: &'a mut [u32],
        k: DimK,
        cache: Cache,
    ) -> Self {
        bounds::check_eq!(a.k(), k, "contraction dimensions do not agree");
        bounds::check_eq!(b.k(), k, "contraction dimensions do not agree");
        bounds::check_eq!(
            bounds::Bound::new(a.blocks().get()),
            c.len().div_ceil(MR),
            "output length must occupy exactly the packed A blocks",
        );
        assert!(
            k.value().get() <= MAX_K_PAIRS,
            "contraction dimension exceeds u32 accumulator capacity",
        );

        // SAFETY: Inherited from caller.
        unsafe {
            Self::new_inner(
                arch,
                a,
                b,
                c,
                k,
                Params::new(cache, a.block_stride(k).bytes(), b.stride(k).bytes(), NR),
            )
        }
    }

    /// # Safety
    ///
    /// 1. `a.k()` and `b.k()` must be equal to `k`.
    /// 2. `c.len().div_ceil(MR)` must be equal to `a.blocks()`.
    /// 3. `k` must not exceed [`MAX_K_PAIRS`].
    unsafe fn new_inner(
        arch: A,
        a: packed::View<'a, U8Pair, MR>,
        b: unpacked::View<'a, u8>,
        c: &'a mut [u32],
        k: DimK,
        params: Params,
    ) -> Self {
        bounds::check_eq!(a.k(), k, "contraction dimensions do not agree");
        bounds::check_eq!(b.k(), k, "contraction dimensions do not agree");
        bounds::check_eq!(
            bounds::Bound::new(a.blocks().get()),
            c.len().div_ceil(MR),
            "output length must occupy exactly the packed A blocks",
        );
        assert!(
            k.value().get() <= MAX_K_PAIRS,
            "contraction dimension exceeds u32 accumulator capacity",
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
                self.c.fill(0);

                let remainder = self.c.len() % MR;
                let last_a_block = self.a.blocks().get() - 1;
                let mut c = MutSlice::new(self.c);

                let on_a_panels = |a_panels: packed::View<'_, U8Pair, MR>, a_block_base| {
                    let on_b_panels = |b_panels: unpacked::View<'_, u8>, _| {
                        let panel_kernel =
                            |a_panel: packed::Panel<'_, U8Pair, MR>, a_block_offset| {
                                let a_block = a_block_base + a_block_offset;
                                let handling_tail = a_block == last_a_block && remainder != 0;
                                let bound = bounds::Bound::from_fn(|| {
                                    if handling_tail { remainder } else { MR }
                                });

                                // SAFETY: By class invariant, this region is contained in `c`.
                                let mut region = unsafe { c.subslice(MR * a_block, bound) };
                                let mut values = [0; MR];
                                if handling_tail {
                                    // SAFETY: `region` has length exactly `remainder`.
                                    values[..remainder]
                                        .copy_from_slice(unsafe { region.as_std_slice(remainder) });
                                } else {
                                    // SAFETY: `region` has length exactly `MR`.
                                    values = unsafe { *region.as_array::<MR>() };
                                }

                                // SAFETY: By class invariant, both panels have contraction
                                // dimension `self.k`.
                                let mut kernel = unsafe {
                                    PanelKernel::new(self.arch, a_panel, b_panels, values, self.k)
                                };
                                driver::PanelKernel::panel_kernel(&mut kernel);

                                if handling_tail {
                                    // SAFETY: `region` has length exactly `remainder`.
                                    unsafe { region.as_std_mut_slice(remainder) }
                                        .copy_from_slice(&kernel.take()[..remainder]);
                                } else {
                                    // SAFETY: `region` has length exactly `MR`.
                                    unsafe { *region.as_array::<MR>() = kernel.take() };
                                }
                            };

                        // SAFETY: By class invariant, `a_panels.k() == self.k`.
                        unsafe { a_panels.visit_panels(self.k, panel_kernel) };
                    };

                    // SAFETY: By class invariant, `self.b.k() == self.k`.
                    unsafe {
                        self.b
                            .visit_sub_views(self.params.b_cols_in_l1, self.k, on_b_panels);
                    }
                };

                // SAFETY: By class invariant, `self.a.k() == self.k`.
                unsafe {
                    self.a
                        .visit_sub_views(self.params.a_panels_in_l2, self.k, on_a_panels);
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
    a: packed::Panel<'a, U8Pair, MR>,
    b: unpacked::View<'a, u8>,
    c: [u32; MR],
    k: DimK,
}

impl<'a, A, const MR: usize, const NR: usize> PanelKernel<'a, A, MR, NR> {
    /// # Safety
    ///
    /// `a.k()` and `b.k()` must both equal `k`.
    unsafe fn new(
        arch: A,
        a: packed::Panel<'a, U8Pair, MR>,
        b: unpacked::View<'a, u8>,
        c: [u32; MR],
        k: DimK,
    ) -> Self {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.k(), k);
        Self { arch, a, b, c, k }
    }

    fn take(self) -> [u32; MR] {
        self.c
    }
}

struct Visitor<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, U8Pair, MR>,
    c: &'a mut [u32; MR],
    k: DimK,
}

impl<A, const MR: usize, const NR: usize> unpacked::PanelVisitor<u8, NR> for Visitor<'_, A, MR, NR>
where
    A: Copy,
    for<'a> MicroKernel<'a, A, MR, NR>: driver::MicroKernel,
{
    #[inline(always)]
    fn visit(&mut self, b: unpacked::Panel<'_, u8, NR>, _: usize) {
        // SAFETY: The panel visitor preserves the shared contraction dimension.
        let mut micro = unsafe { MicroKernel::new(self.arch, self.a, b, self.c, self.k) };
        driver::MicroKernel::micro_kernel(&mut micro);
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
                    c: &mut self.c,
                    k: self.k,
                };

                // SAFETY: By class invariant, `self.b.k() == self.k`.
                let b_tail = unsafe { self.b.visit_panels::<$nr>(self.k, visitor) };

                if let Some(b_tail) = b_tail {
                    $(
                        const { assert!($ns < $nr) };
                        if let Some(b_panel) = b_tail.try_as_panel::<$ns>() {
                            // SAFETY: Both panels have contraction dimension `self.k`.
                            let mut micro = unsafe {
                                MicroKernel::new(
                                    self.arch,
                                    self.a,
                                    b_panel,
                                    &mut self.c,
                                    self.k,
                                )
                            };
                            driver::MicroKernel::micro_kernel(&mut micro);
                        }
                    )+
                }
            }
        }
    };
}

panel_kernel!(Scalar, 8, 2, [1]);

//--------------//
// Micro Kernel //
//--------------//

struct MicroKernel<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, U8Pair, MR>,
    b: unpacked::Panel<'a, u8, NR>,
    c: &'a mut [u32; MR],
    k: DimK,
}

impl<'a, A, const MR: usize, const NR: usize> MicroKernel<'a, A, MR, NR> {
    /// # Safety
    ///
    /// `a.k()` and `b.k()` must both equal `k`.
    unsafe fn new(
        arch: A,
        a: packed::Panel<'a, U8Pair, MR>,
        b: unpacked::Panel<'a, u8, NR>,
        c: &'a mut [u32; MR],
        k: DimK,
    ) -> Self {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.k(), k);
        Self { arch, a, b, c, k }
    }
}

#[inline(always)]
unsafe fn micro_kernel<W, const MR: usize, const NR: usize>(
    wide: W,
    a: packed::Panel<'_, U8Pair, MR>,
    b: unpacked::Panel<'_, u8, NR>,
    c: &mut [u32; MR],
    k: DimK,
) where
    W: ExtraWide<MR>,
    Folder: Fold<NR>,
{
    bounds::check_eq!(a.k(), k);
    bounds::check_eq!(b.k(), k);

    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let astride = Elements::<U8Pair>::new(MR);
    let bstride = b.stride(k);
    let mut acc = [wide.default(); NR];

    for i in 0..k.value().get() {
        // SAFETY: `a` contains `k` complete groups of `MR` pairs.
        let ai = unsafe { wide.load(ap.add(astride * i).truncate(astride)) };
        for (j, acc) in acc.iter_mut().enumerate() {
            // SAFETY: `b` contains `NR` bands of `k` packed bytes.
            let bj = *unsafe { bp.add(bstride * j + Elements::new(i)).as_unit().as_ref() };
            *acc = W::mul_add_pair(ai, bj, *acc);
        }
    }

    wide.max_into(Folder::fold(acc, W::max), c);
}

macro_rules! micro_kernel {
    ($arch:ty, $mr:literal, $nr:literal) => {
        impl driver::MicroKernel for MicroKernel<'_, $arch, $mr, $nr> {
            #[inline(always)]
            fn micro_kernel(&mut self) {
                // SAFETY: By class invariant, both panels have contraction dimension `self.k`.
                unsafe { micro_kernel(self.arch, self.a, self.b, self.c, self.k) }
            }
        }
    };
    ($arch:ty, $mr:literal, { $($nr:literal),+ $(,)? }) => {
        $(micro_kernel!($arch, $mr, $nr);)+
    };
}

micro_kernel!(Scalar, 8, { 2, 1 });

trait ExtraWide<const ELEMENTS: usize>: Copy {
    type PackedA: Copy;
    type Wide: Copy;

    /// # Safety
    ///
    /// `slice.len()` must be exactly `ELEMENTS`.
    unsafe fn load(self, slice: Slice<'_, U8Pair>) -> Self::PackedA;

    fn default(self) -> Self::Wide;
    fn mul_add_pair(a: Self::PackedA, b: u8, acc: Self::Wide) -> Self::Wide;
    fn max(lhs: Self::Wide, rhs: Self::Wide) -> Self::Wide;
    fn max_into(self, max: Self::Wide, into: &mut [u32; ELEMENTS]);
}

impl ExtraWide<8> for Scalar {
    type PackedA = [U8Pair; 8];
    type Wide = [u32; 8];

    #[inline(always)]
    unsafe fn load(self, slice: Slice<'_, U8Pair>) -> Self::PackedA {
        bounds::check_eq!(slice.len(), 8);
        // SAFETY: The slice contains exactly eight pairs.
        unsafe { *slice.as_ptr().cast::<[U8Pair; 8]>() }
    }

    #[inline(always)]
    fn default(self) -> Self::Wide {
        [0; 8]
    }

    #[inline(always)]
    fn mul_add_pair(a: Self::PackedA, b: u8, acc: Self::Wide) -> Self::Wide {
        let lo = u32::from(b & 0x0f);
        let hi = u32::from(b >> 4);
        core::array::from_fn(|i| {
            acc[i]
                .wrapping_add(u32::from(a[i][0]) * lo)
                .wrapping_add(u32::from(a[i][1]) * hi)
        })
    }

    #[inline(always)]
    fn max(lhs: Self::Wide, rhs: Self::Wide) -> Self::Wide {
        core::array::from_fn(|i| lhs[i].max(rhs[i]))
    }

    #[inline(always)]
    fn max_into(self, max: Self::Wide, into: &mut [u32; 8]) {
        *into = Self::max(max, *into);
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepu16_epi32,
        _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_maddubs_epi16, _mm256_max_epu32,
        _mm256_set1_epi16, _mm256_setzero_si256, _mm256_storeu_si256,
    };

    use diskann_wide::arch::x86_64::V3;

    use super::*;

    panel_kernel!(V3, 16, 6, [1, 2, 3, 4, 5]);
    micro_kernel!(V3, 16, { 6, 5, 4, 3, 2, 1 });

    impl ExtraWide<16> for V3 {
        type PackedA = __m256i;
        type Wide = [__m256i; 2];

        #[inline(always)]
        unsafe fn load(self, slice: Slice<'_, U8Pair>) -> Self::PackedA {
            bounds::check_eq!(slice.len(), 16);
            // SAFETY: Sixteen pairs occupy exactly one 256-bit vector.
            unsafe { _mm256_loadu_si256(slice.as_ptr().cast()) }
        }

        #[inline(always)]
        fn default(self) -> Self::Wide {
            // SAFETY: V3 guarantees AVX2.
            unsafe { [_mm256_setzero_si256(); 2] }
        }

        #[inline(always)]
        fn mul_add_pair(a: Self::PackedA, b: u8, acc: Self::Wide) -> Self::Wide {
            // SAFETY: V3 guarantees AVX2. Pair products are at most
            // `2 * 255 * 15 = 7,650`, so `maddubs` cannot saturate.
            unsafe {
                let pair = i16::from(b & 0x0f) | (i16::from(b >> 4) << 8);
                let packed_b = _mm256_set1_epi16(pair);
                let partial = _mm256_maddubs_epi16(a, packed_b);
                [
                    _mm256_add_epi32(
                        acc[0],
                        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(partial)),
                    ),
                    _mm256_add_epi32(
                        acc[1],
                        _mm256_cvtepu16_epi32(_mm256_extracti128_si256(partial, 1)),
                    ),
                ]
            }
        }

        #[inline(always)]
        fn max(lhs: Self::Wide, rhs: Self::Wide) -> Self::Wide {
            // SAFETY: V3 guarantees AVX2.
            unsafe {
                [
                    _mm256_max_epu32(lhs[0], rhs[0]),
                    _mm256_max_epu32(lhs[1], rhs[1]),
                ]
            }
        }

        #[inline(always)]
        fn max_into(self, max: Self::Wide, into: &mut [u32; 16]) {
            // SAFETY: V3 guarantees AVX2, and both loads and stores cover exactly 16 values.
            unsafe {
                let previous = [
                    _mm256_loadu_si256(into.as_ptr().cast()),
                    _mm256_loadu_si256(into.as_ptr().add(8).cast()),
                ];
                let max = Self::max(max, previous);
                _mm256_storeu_si256(into.as_mut_ptr().cast(), max[0]);
                _mm256_storeu_si256(into.as_mut_ptr().add(8).cast(), max[1]);
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

    use diskann_utils::views::{Init, Matrix};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[cfg(target_arch = "x86_64")]
    use diskann_wide::arch::x86_64::V3;

    use crate::{matrix_kernels::maxsim, multi_vector::BlockTransposed};

    fn generate(
        m: usize,
        logical_k: usize,
        n: usize,
        rng: &mut impl Rng,
    ) -> (Matrix<u8>, Matrix<u8>, Vec<u32>) {
        let a = Matrix::new(Init(|| rng.random::<u8>()), m, logical_k);
        let docs = Matrix::new(Init(|| rng.random_range(0u8..16)), n, logical_k);
        let pair_k = logical_k.div_ceil(2);
        let mut b = Matrix::new(Init(|| 0u8), n, pair_k);

        for row in 0..n {
            for col in 0..logical_k {
                b[(row, col / 2)] |= docs[(row, col)] << (4 * (col % 2));
            }
            if !logical_k.is_multiple_of(2) {
                b[(row, pair_k - 1)] |= 0xf0;
            }
        }

        let c = a
            .row_iter()
            .map(|a_row| {
                docs.row_iter()
                    .map(|doc| {
                        a_row
                            .iter()
                            .zip(doc)
                            .map(|(&x, &y)| u32::from(x) * u32::from(y))
                            .sum()
                    })
                    .max()
                    .unwrap()
            })
            .collect();

        (a, b, c)
    }

    fn test_micro_kernel<A, const MR: usize, const NR: usize>(
        arch: A,
        logical_k: usize,
        rng: &mut impl Rng,
    ) where
        A: Copy,
        for<'a> MicroKernel<'a, A, MR, NR>: driver::MicroKernel,
    {
        let (a, b, expected) = generate(MR, logical_k, NR, rng);
        let a = BlockTransposed::<u8, MR, 2>::from_matrix_view(a.as_view());
        let a = packed::View::from_u8_pairs(a.as_view()).unwrap();
        let k = DimK::new(NonZeroUsize::new(logical_k.div_ceil(2)).unwrap());
        let mut actual = [0; MR];

        // SAFETY: The test views share `k` and contain exactly one A panel and one B panel.
        unsafe {
            a.visit_panels(k, |a, _| {
                let mut micro = MicroKernel::new(
                    arch,
                    a,
                    unpacked::Panel::new(Slice::new(b.as_slice()), k),
                    &mut actual,
                    k,
                );
                driver::MicroKernel::micro_kernel(&mut micro);
            });
        }

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_micro_kernel_scalar() {
        let mut rng = StdRng::seed_from_u64(0x759722e1a83e4566);
        for logical_k in [1, 2, 5, 8, 257] {
            test_micro_kernel::<_, 8, 2>(Scalar::new(), logical_k, &mut rng);
            test_micro_kernel::<_, 8, 1>(Scalar::new(), logical_k, &mut rng);
        }
    }

    #[test]
    fn rejects_nonzero_odd_column_padding() {
        let a = Matrix::new(Init(|| 1u8), 8, 3);
        let mut a = BlockTransposed::<u8, 8, 2>::from_matrix_view(a.as_view());
        a.as_mut_slice()[2 * 8 + 1] = 1;
        assert!(packed::View::from_u8_pairs(a.as_view()).is_none());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_micro_kernel_v3() {
        if let Some(arch) = V3::new_checked() {
            let mut rng = StdRng::seed_from_u64(0x157f59d99f648437);
            for logical_k in [1, 2, 5, 8, 257] {
                test_micro_kernel::<_, 16, 6>(arch, logical_k, &mut rng);
                test_micro_kernel::<_, 16, 5>(arch, logical_k, &mut rng);
                test_micro_kernel::<_, 16, 1>(arch, logical_k, &mut rng);
            }
        }
    }

    fn test_driver<A, const MR: usize, const NR: usize>(arch: A, rng: &mut impl Rng)
    where
        A: Copy,
        for<'a> Driver<'a, A, MR, NR>: driver::Drive,
    {
        for case in maxsim::test::packed_x_unpacked_test_dims(MR, NR) {
            let (a, b, expected) = generate(case.total_a_rows, case.k, case.total_b_cols, rng);
            let a = BlockTransposed::<u8, MR, 2>::from_matrix_view(a.as_view());
            let a = packed::View::from_u8_pairs(a.as_view()).unwrap();
            let b = unpacked::View::from_matrix_view(b.as_view()).unwrap();
            let k = DimK::new(NonZeroUsize::new(case.k.div_ceil(2)).unwrap());
            let mut actual = vec![u32::MAX; case.total_a_rows];

            // SAFETY: Test builds verify the matching view and output bounds.
            let mut driver = unsafe {
                Driver::new_inner(
                    arch,
                    a,
                    b,
                    &mut actual,
                    k,
                    Params {
                        a_panels_in_l2: NonZeroUsize::new(case.a_panels_per_tile).unwrap(),
                        b_cols_in_l1: NonZeroUsize::new(case.b_cols_per_tile).unwrap(),
                    },
                )
            };
            driver::Drive::drive(&mut driver);

            assert_eq!(expected, actual, "setup: {case:?}");
        }
    }

    #[test]
    fn test_driver_scalar() {
        test_driver::<_, 8, 2>(
            Scalar::new(),
            &mut StdRng::seed_from_u64(0x2c03eb9ee51d30c3),
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_driver_v3() {
        if let Some(arch) = V3::new_checked() {
            test_driver::<_, 16, 6>(arch, &mut StdRng::seed_from_u64(0x2c03eb9ee51d30c3));
        }
    }
}
