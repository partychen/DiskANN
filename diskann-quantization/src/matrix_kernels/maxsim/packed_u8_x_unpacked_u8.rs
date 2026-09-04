/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The lowering of operations mimics a GEMM style operation with inplace application of the
//! max-sim reduction operation. Currently, blocking across the contraction dimension "k"
//! is not implemented. As such, expect a performance penalty for large-dimensional vectors.
//!
//! The kernel is implemented as follows:
//!
//! * Partition `a` into sub-views `suba` that roughly occupy the L2 cache.
//! * Partition `b` into sub-views `subb` that roughly occupy a portion of the L1 cache.
//! * Partition `suba` into panels `pa`. We want `pa + subb` to fit in L1.
//! * Perform micro-kernel operations on `pa + subb`. This computes the max-sim in-place.
//!
//! There is plenty of room for improvement. This is just a starting point.

use diskann_wide::arch::{Architecture, Scalar};

use crate::matrix_kernels::{
    Cache,
    blocks::{packed, unpacked},
    bounds, driver,
    num::{DimK, Elements},
    ptr::{MutSlice, Slice},
    util::{self, Fold, Folder},
};

use super::Params;

const MAX_PRODUCT: usize = u8::MAX as usize * u8::MAX as usize;
const MAX_K: usize = (u32::MAX as usize / MAX_PRODUCT) / 2 * 2;

//--------//
// Driver //
//--------//

/// A driver for prepacked by unpacked `u8` MaxSim computations.
///
/// Results are returned directly in `c`.
///
/// # Class Invariants
///
/// 1. `a.k()` and `b.k()` must be equal to `k`.
/// 2. `c.len().div_ceil(MR)` must be equal to `a.blocks()`.
/// 3. The maximum dot product for `k` elements fits in `u32`.
/// 4. `k` is even and `a` uses PACK=2 block-transposed ordering.
pub(crate) struct Driver<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::View<'a, u8, MR>,
    b: unpacked::View<'a, u8>,
    c: &'a mut [u32],
    k: DimK,
    params: Params,
}

impl<'a, A, const MR: usize, const NR: usize> Driver<'a, A, MR, NR> {
    /// Prepare for a MaxSim on `a` and `b` with the results stored directly into `c`.
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
        bounds::check_eq!(a.k(), k, "contraction dimensions do not agree");
        bounds::check_eq!(b.k(), k, "contraction dimensions do not agree");
        bounds::check_eq!(
            bounds::Bound::new(a.blocks().get()),
            c.len().div_ceil(MR),
            "output length must occupy exactly the packed A blocks"
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
    /// The caller must uphold all class invariants.
    unsafe fn new_inner(
        arch: A,
        a: packed::View<'a, u8, MR>,
        b: unpacked::View<'a, u8>,
        c: &'a mut [u32],
        k: DimK,
        params: Params,
    ) -> Self {
        assert!(
            k.value().get() <= MAX_K,
            "dimension exceeds accumulator bound"
        );
        bounds::check_eq!(
            bounds::Bound::new(k.value().get() % 2),
            0,
            "PACK=2 requires an even storage dimension"
        );
        bounds::check_eq!(a.k(), k, "contraction dimensions do not agree");
        bounds::check_eq!(b.k(), k, "contraction dimensions do not agree");
        bounds::check_eq!(
            bounds::Bound::new(a.blocks().get()),
            c.len().div_ceil(MR),
            "output length must occupy exactly the packed A blocks"
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
    A: util::LoadStore<u32, MR> + Architecture,
    for<'a> PanelKernel<'a, A, MR, NR>: driver::PanelKernel,
{
    #[inline(never)]
    fn drive(&mut self) {
        self.arch.run(
            #[inline]
            || {
                // Pre-fill `c`.
                self.c.fill(0);

                // We allow the final physical A block to be only logically filled.
                let remainder = self.c.len() % MR;
                let last_a_block = self.a.blocks().get() - 1;
                let mut c = MutSlice::new(self.c);

                let on_a_panels = |a_panels: packed::View<'_, u8, MR>, a_block_base| {
                    let on_b_panels = |b_panels: unpacked::View<'_, u8>, _| {
                        let panel_kernel =
                            |a_panel: packed::Panel<'_, u8, MR>, a_block_offset| {
                                let a_block = a_block_base + a_block_offset;
                                let handling_tail = a_block == last_a_block && remainder != 0;
                                let bound = bounds::Bound::from_fn(|| {
                                    if handling_tail { remainder } else { MR }
                                });

                                // SAFETY: The output occupies exactly the packed A blocks.
                                let mut region = unsafe { c.subslice(MR * a_block, bound) };
                                let c = if handling_tail {
                                    util::LoadStore::<u32, MR>::load(
                                        self.arch,
                                        // SAFETY: `region` has length exactly `remainder`.
                                        unsafe { region.as_std_slice(remainder) },
                                    )
                                } else {
                                    // SAFETY: `region` has length exactly `MR`.
                                    unsafe { *region.as_array::<MR>() }
                                };

                                // Run the kernel.
                                //
                                // SAFETY: By class invariant, `a_panel.k()` and `b_panels.k()`
                                // are both equal to `self.k`.
                                let mut kernel = unsafe {
                                    PanelKernel::new(self.arch, a_panel, b_panels, c, self.k)
                                };
                                driver::PanelKernel::panel_kernel(&mut kernel);

                                let c_final = kernel.take();

                                // Put back `C`.
                                if handling_tail {
                                    util::LoadStore::<u32, MR>::store(
                                        self.arch,
                                        c_final,
                                        // SAFETY: `region` has length exactly `remainder`.
                                        unsafe { region.as_std_mut_slice(remainder) },
                                    );
                                } else {
                                    // SAFETY: `region` has length exactly `MR`.
                                    unsafe { *region.as_array::<MR>() = c_final };
                                }
                            };

                        // SAFETY: By class invariant, `a_panels.k() == self.k`.
                        unsafe { a_panels.visit_panels(self.k, panel_kernel) };
                    };

                    // SAFETY: `self.b.k()` equals `self.k` by class invariant.
                    unsafe {
                        self.b
                            .visit_sub_views(self.params.b_cols_in_l1, self.k, on_b_panels);
                    }
                };

                // SAFETY: `self.a.k()` equals `self.k` by class invariant.
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

#[derive(Debug)]
pub(super) struct PanelKernel<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, u8, MR>,
    b: unpacked::View<'a, u8>,
    c: [u32; MR],
    k: DimK,
}

impl<'a, A, const MR: usize, const NR: usize> PanelKernel<'a, A, MR, NR> {
    /// Construct a new kernel.
    ///
    /// # Safety
    ///
    /// Bounds `a.k()` and `b.k()` must both be equal to `k`.
    pub(super) unsafe fn new(
        arch: A,
        a: packed::Panel<'a, u8, MR>,
        b: unpacked::View<'a, u8>,
        c: [u32; MR],
        k: DimK,
    ) -> Self {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.k(), k);
        bounds::check_eq!(bounds::Bound::new(k.value().get() % 2), 0);

        Self { arch, a, b, c, k }
    }

    pub(super) fn take(self) -> [u32; MR] {
        self.c
    }
}

/// A custom visitor for the [`MicroKernel`].
///
/// This is needed to ensure the visitor body is inlined to inherit target features.
#[derive(Debug)]
struct Visitor<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, u8, MR>,
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
        // SAFETY: This is only used in contexts where `self.a.k()`, `b.k()`, and `self.k`
        // are all equal.
        let mut micro = unsafe { MicroKernel::new(self.arch, self.a, b, self.c, self.k) };
        driver::MicroKernel::micro_kernel(&mut micro);
    }
}

macro_rules! panel_kernel {
    ($arch:ty, $mr:literal, $nr:literal, [ $($ns:literal),+ $(,)? ]) => {
        impl driver::PanelKernel for PanelKernel<'_, $arch, $mr, $nr> {
            #[inline(always)]
            fn panel_kernel(&mut self) {
                // NOTE: A `Visitor` is used here instead of a closure because a `Visitor`
                // is more reliably inlined, which means target features are inherited
                // more reliably.
                let on_b_panels = Visitor {
                    arch: self.arch,
                    a: self.a,
                    c: &mut self.c,
                    k: self.k,
                };

                // SAFETY: By class invariant, `self.k` is equal to `self.b.k()`.
                let b_tail = unsafe { self.b.visit_panels::<$nr>(self.k, on_b_panels) };

                if let Some(b_tail) = b_tail {
                    $(
                        const { assert!($ns < $nr) };
                        if let Some(b_panel) = b_tail.try_as_panel::<$ns>() {
                            // SAFETY: By class invariant, `self.a.k()` and `self.b.k()`
                            // are equal to `self.k`.
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

/// # Class Invariants
///
/// `a.k()` and `b.k()` are equal to `k`.
struct MicroKernel<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, u8, MR>,
    b: unpacked::Panel<'a, u8, NR>,
    c: &'a mut [u32; MR],
    k: DimK,
}

impl<'a, A, const MR: usize, const NR: usize> MicroKernel<'a, A, MR, NR> {
    /// # Safety
    ///
    /// Bounds `a.k()` and `b.k()` must be equal to `k`.
    unsafe fn new(
        arch: A,
        a: packed::Panel<'a, u8, MR>,
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
    a: packed::Panel<'_, u8, MR>,
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
    let mut acc = [wide.default(); NR];
    let astride = Elements::<u8>::new(MR * 2);
    let bstride = b.stride(k);

    for pair in 0..k.value().get() / 2 {
        // SAFETY: `ap` contains `k / 2` groups of `MR * 2` query values.
        let ai = unsafe { wide.load(ap.add(astride * pair).truncate(astride)) };

        for (j, acc) in acc.iter_mut().enumerate() {
            let offset = bstride * j + Elements::new(pair * 2);
            // SAFETY: `bp` contains `NR` rows of exactly `k` elements.
            let lo = *unsafe { bp.add(offset).as_unit().as_ref() };
            // SAFETY: `k` is even, so every pair has a second element.
            let hi = *unsafe { bp.add(offset + Elements::new(1)).as_unit().as_ref() };
            let bj = wide.splat(lo, hi);
            *acc = W::mul_add_pair(ai, bj, *acc);
        }
    }

    let acc = acc.map(|value| wide.to_u32_array(value));
    let max = Folder::fold(acc, |lhs, rhs| core::array::from_fn(|i| lhs[i].max(rhs[i])));
    for (value, maximum) in c.iter_mut().zip(max) {
        *value = (*value).max(maximum);
    }
}

macro_rules! micro_kernel {
    ($arch:ty, $mr:literal, $nr:literal) => {
        impl driver::MicroKernel for MicroKernel<'_, $arch, $mr, $nr> {
            #[inline(always)]
            fn micro_kernel(&mut self) {
                // SAFETY: By class invariant, `self.a.k()` and `self.b.k()` equal `self.k`.
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
    type Wide: Copy;
    type Splat: Copy;
    type Accumulator: Copy;

    /// # Safety
    ///
    /// `slice.len()` must be exactly `ELEMENTS * 2`.
    unsafe fn load(self, slice: Slice<'_, u8>) -> Self::Wide;

    fn default(self) -> Self::Accumulator;
    fn splat(self, lo: u8, hi: u8) -> Self::Splat;
    fn mul_add_pair(a: Self::Wide, b: Self::Splat, acc: Self::Accumulator) -> Self::Accumulator;
    fn to_u32_array(self, value: Self::Accumulator) -> [u32; ELEMENTS];
}

impl ExtraWide<8> for Scalar {
    type Wide = [u8; 16];
    type Splat = [u8; 2];
    type Accumulator = [u32; 8];

    #[inline(always)]
    unsafe fn load(self, slice: Slice<'_, u8>) -> Self::Wide {
        bounds::check_eq!(slice.len(), 16);
        let mut result = [0; 16];
        // SAFETY: `slice` contains exactly 16 elements.
        result.copy_from_slice(unsafe { slice.as_std_slice(16) });
        result
    }

    #[inline(always)]
    fn default(self) -> Self::Accumulator {
        [0; 8]
    }

    #[inline(always)]
    fn splat(self, lo: u8, hi: u8) -> Self::Splat {
        [lo, hi]
    }

    #[inline(always)]
    fn mul_add_pair(
        a: Self::Wide,
        [lo, hi]: Self::Splat,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        core::array::from_fn(|i| {
            acc[i] + u32::from(a[i * 2]) * u32::from(lo) + u32::from(a[i * 2 + 1]) * u32::from(hi)
        })
    }

    #[inline(always)]
    fn to_u32_array(self, value: Self::Accumulator) -> [u32; 8] {
        value
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;

    use diskann_wide::{
        SIMDDotProduct, SIMDReinterpret, SIMDVector,
        arch::x86_64::{V3, V4},
    };

    panel_kernel!(V3, 16, 4, [1, 2, 3]);
    panel_kernel!(V4, 16, 6, [1, 2, 3, 4, 5]);
    panel_kernel!(V4, 32, 6, [1, 2, 3, 4, 5]);

    micro_kernel!(V3, 16, { 4, 3, 2, 1 });
    micro_kernel!(V4, 16, { 6, 5, 4, 3, 2, 1 });
    micro_kernel!(V4, 32, { 6, 5, 4, 3, 2, 1 });

    diskann_wide::alias!(v3_u8x16 = <V3>::u8x16);
    diskann_wide::alias!(v3_u32x8 = <V3>::u32x8);
    diskann_wide::alias!(v3_i16x16 = <V3>::i16x16);
    diskann_wide::alias!(v3_i32x8 = <V3>::i32x8);

    impl ExtraWide<16> for V3 {
        type Wide = [v3_i16x16; 2];
        type Splat = v3_i16x16;
        type Accumulator = [v3_i32x8; 2];

        #[inline(always)]
        unsafe fn load(self, slice: Slice<'_, u8>) -> Self::Wide {
            bounds::check_eq!(slice.len(), 32);
            // SAFETY: `slice` contains exactly 32 elements.
            unsafe {
                [
                    v3_u8x16::load_simd(self, slice.as_ptr()).into(),
                    v3_u8x16::load_simd(self, slice.as_ptr().add(16)).into(),
                ]
            }
        }

        #[inline(always)]
        fn default(self) -> Self::Accumulator {
            [v3_i32x8::default(self); 2]
        }

        #[inline(always)]
        fn splat(self, lo: u8, hi: u8) -> Self::Splat {
            let pair = u32::from(lo) | (u32::from(hi) << 16);
            v3_u32x8::splat(self, pair).reinterpret_simd()
        }

        #[inline(always)]
        fn mul_add_pair(
            a: Self::Wide,
            b: Self::Splat,
            acc: Self::Accumulator,
        ) -> Self::Accumulator {
            core::array::from_fn(|i| acc[i].dot_simd(a[i], b))
        }

        #[inline(always)]
        fn to_u32_array(self, value: Self::Accumulator) -> [u32; 16] {
            let lo = value[0].to_array();
            let hi = value[1].to_array();
            core::array::from_fn(|i| {
                if i < 8 {
                    lo[i] as u32
                } else {
                    hi[i - 8] as u32
                }
            })
        }
    }

    diskann_wide::alias!(v4_u8x32 = <V4>::u8x32);
    diskann_wide::alias!(v4_u32x16 = <V4>::u32x16);
    diskann_wide::alias!(v4_i16x32 = <V4>::i16x32);
    diskann_wide::alias!(v4_i32x16 = <V4>::i32x16);

    impl ExtraWide<16> for V4 {
        type Wide = v4_i16x32;
        type Splat = v4_i16x32;
        type Accumulator = v4_i32x16;

        #[inline(always)]
        unsafe fn load(self, slice: Slice<'_, u8>) -> Self::Wide {
            bounds::check_eq!(slice.len(), 32);
            // SAFETY: `slice` contains exactly 32 elements.
            unsafe { v4_u8x32::load_simd(self, slice.as_ptr()).into() }
        }

        #[inline(always)]
        fn default(self) -> Self::Accumulator {
            v4_i32x16::default(self)
        }

        #[inline(always)]
        fn splat(self, lo: u8, hi: u8) -> Self::Splat {
            let pair = u32::from(lo) | (u32::from(hi) << 16);
            v4_u32x16::splat(self, pair).reinterpret_simd()
        }

        #[inline(always)]
        fn mul_add_pair(
            a: Self::Wide,
            b: Self::Splat,
            acc: Self::Accumulator,
        ) -> Self::Accumulator {
            acc.dot_simd(a, b)
        }

        #[inline(always)]
        fn to_u32_array(self, value: Self::Accumulator) -> [u32; 16] {
            value.to_array().map(|value| value as u32)
        }
    }

    impl ExtraWide<32> for V4 {
        type Wide = [v4_i16x32; 2];
        type Splat = v4_i16x32;
        type Accumulator = [v4_i32x16; 2];

        #[inline(always)]
        unsafe fn load(self, slice: Slice<'_, u8>) -> Self::Wide {
            bounds::check_eq!(slice.len(), 64);
            // SAFETY: `slice` contains exactly 64 elements.
            unsafe {
                [
                    v4_u8x32::load_simd(self, slice.as_ptr()).into(),
                    v4_u8x32::load_simd(self, slice.as_ptr().add(32)).into(),
                ]
            }
        }

        #[inline(always)]
        fn default(self) -> Self::Accumulator {
            [v4_i32x16::default(self); 2]
        }

        #[inline(always)]
        fn splat(self, lo: u8, hi: u8) -> Self::Splat {
            let pair = u32::from(lo) | (u32::from(hi) << 16);
            v4_u32x16::splat(self, pair).reinterpret_simd()
        }

        #[inline(always)]
        fn mul_add_pair(
            a: Self::Wide,
            b: Self::Splat,
            acc: Self::Accumulator,
        ) -> Self::Accumulator {
            core::array::from_fn(|i| acc[i].dot_simd(a[i], b))
        }

        #[inline(always)]
        fn to_u32_array(self, value: Self::Accumulator) -> [u32; 32] {
            let lo = value[0].to_array();
            let hi = value[1].to_array();
            core::array::from_fn(|i| {
                if i < 16 {
                    lo[i] as u32
                } else {
                    hi[i - 16] as u32
                }
            })
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::*;

    use diskann_wide::{SIMDDotProduct, SIMDVector, SplitJoin, arch::aarch64::Neon};

    panel_kernel!(Neon, 8, 6, [1, 2, 3, 4, 5]);
    micro_kernel!(Neon, 8, { 6, 5, 4, 3, 2, 1 });

    diskann_wide::alias!(neon_u8x16 = <Neon>::u8x16);
    diskann_wide::alias!(neon_i16x8 = <Neon>::i16x8);
    diskann_wide::alias!(neon_i16x16 = <Neon>::i16x16);
    diskann_wide::alias!(neon_i32x4 = <Neon>::i32x4);

    impl ExtraWide<8> for Neon {
        type Wide = [neon_i16x8; 2];
        type Splat = neon_i16x8;
        type Accumulator = [neon_i32x4; 2];

        #[inline(always)]
        unsafe fn load(self, slice: Slice<'_, u8>) -> Self::Wide {
            bounds::check_eq!(slice.len(), 16);
            // SAFETY: `slice` contains exactly 16 elements.
            let wide: neon_i16x16 = unsafe { neon_u8x16::load_simd(self, slice.as_ptr()) }.into();
            let halves = wide.split();
            [halves.lo, halves.hi]
        }

        #[inline(always)]
        fn default(self) -> Self::Accumulator {
            [neon_i32x4::default(self); 2]
        }

        #[inline(always)]
        fn splat(self, lo: u8, hi: u8) -> Self::Splat {
            neon_i16x8::from_array(
                self,
                core::array::from_fn(|i| i16::from(if i % 2 == 0 { lo } else { hi })),
            )
        }

        #[inline(always)]
        fn mul_add_pair(
            a: Self::Wide,
            b: Self::Splat,
            acc: Self::Accumulator,
        ) -> Self::Accumulator {
            core::array::from_fn(|i| acc[i].dot_simd(a[i], b))
        }

        #[inline(always)]
        fn to_u32_array(self, value: Self::Accumulator) -> [u32; 8] {
            let lo = value[0].to_array();
            let hi = value[1].to_array();
            core::array::from_fn(|i| {
                if i < 4 {
                    lo[i] as u32
                } else {
                    hi[i - 4] as u32
                }
            })
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
        matrix_kernels::{Drive, test_util::panic_message_for},
        multi_vector::{BlockTransposed, MatRef, Standard},
    };

    fn run<A, const MR: usize, const NR: usize>(
        arch: A,
        query: MatrixView<'_, u8>,
        docs: MatrixView<'_, u8>,
    ) -> Vec<u32>
    where
        for<'a> Driver<'a, A, MR, NR>: Drive,
    {
        let query = BlockTransposed::<u8, MR, 2>::from_matrix_view(query);
        let a = packed::View::from_block_transposed(query.as_view()).unwrap();
        let k = DimK::from_bound(a.k());
        let b = unpacked::View::from_matrix_view(docs).unwrap();
        let mut c = vec![0; query.nrows()];

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

    fn pad_documents(values: &[u8], rows: usize, dim: usize) -> Vec<u8> {
        let padded_dim = dim.next_multiple_of(2);
        let mut padded = Vec::with_capacity(rows * padded_dim);
        for row in values.chunks_exact(dim) {
            padded.extend_from_slice(row);
            padded.resize(padded.len() + padded_dim - dim, 0);
        }
        padded
    }

    fn reference(query: MatrixView<'_, u8>, docs: MatrixView<'_, u8>) -> Vec<u32> {
        query
            .row_iter()
            .map(|query| {
                docs.row_iter()
                    .map(|doc| {
                        query
                            .iter()
                            .zip(doc)
                            .map(|(&a, &b)| u32::from(a) * u32::from(b))
                            .sum()
                    })
                    .max()
                    .unwrap()
            })
            .collect()
    }

    fn check_case(query_rows: usize, document_rows: usize, dim: usize) {
        let query = query_values(query_rows, dim);
        let documents = document_values(document_rows, dim);
        let padded_documents = pad_documents(&documents, document_rows, dim);
        let query = MatRef::new(Standard::<u8>::new(query_rows, dim).unwrap(), &query).unwrap();
        let documents =
            MatRef::new(Standard::<u8>::new(document_rows, dim).unwrap(), &documents).unwrap();
        let padded_documents = MatRef::new(
            Standard::<u8>::new(document_rows, dim.next_multiple_of(2)).unwrap(),
            &padded_documents,
        )
        .unwrap();
        let expected = reference(query.as_matrix_view(), documents.as_matrix_view());

        let scalar = run::<_, 8, 2>(
            Scalar::new(),
            query.as_matrix_view(),
            padded_documents.as_matrix_view(),
        );
        assert_eq!(scalar, expected);

        #[cfg(target_arch = "x86_64")]
        {
            if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                let actual = run::<_, 16, 4>(
                    arch,
                    query.as_matrix_view(),
                    padded_documents.as_matrix_view(),
                );
                assert_eq!(actual, expected);
            }
            if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
                let actual = run::<_, 16, 6>(
                    arch,
                    query.as_matrix_view(),
                    padded_documents.as_matrix_view(),
                );
                assert_eq!(actual, expected);
                let actual = run::<_, 32, 6>(
                    arch,
                    query.as_matrix_view(),
                    padded_documents.as_matrix_view(),
                );
                assert_eq!(actual, expected);
            }
        }

        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            let actual = run::<_, 8, 6>(
                arch,
                query.as_matrix_view(),
                padded_documents.as_matrix_view(),
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
    fn handles_maximum_values() {
        const M: usize = 17;
        const N: usize = 5;
        const K: usize = 257;

        let query = vec![u8::MAX; M * K];
        let documents = vec![u8::MAX; N * K];
        let padded_documents = pad_documents(&documents, N, K);

        let query = MatRef::new(Standard::<u8>::new(M, K).unwrap(), &query).unwrap();
        let padded_documents = MatRef::new(
            Standard::<u8>::new(N, K.next_multiple_of(2)).unwrap(),
            &padded_documents,
        )
        .unwrap();
        let expected = u32::from(u8::MAX).pow(2) * K as u32;

        let scalar = run::<_, 8, 2>(
            Scalar::new(),
            query.as_matrix_view(),
            padded_documents.as_matrix_view(),
        );
        assert!(scalar.iter().all(|&value| value == expected));

        #[cfg(target_arch = "x86_64")]
        {
            if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                let actual = run::<_, 16, 4>(
                    arch,
                    query.as_matrix_view(),
                    padded_documents.as_matrix_view(),
                );
                assert_eq!(actual, scalar);
            }
            if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
                let actual = run::<_, 32, 6>(
                    arch,
                    query.as_matrix_view(),
                    padded_documents.as_matrix_view(),
                );
                assert_eq!(actual, scalar);
            }
        }
    }

    #[test]
    fn accumulates_past_i32_max() {
        const K: usize = 40_000;

        let query = vec![u8::MAX; K];
        let documents = vec![u8::MAX; K];
        let query = MatRef::new(Standard::<u8>::new(1, K).unwrap(), &query).unwrap();
        let documents = MatRef::new(Standard::<u8>::new(1, K).unwrap(), &documents).unwrap();
        let expected = u32::from(u8::MAX).pow(2) * K as u32;
        assert!(expected > i32::MAX as u32);

        let scalar = run::<_, 8, 2>(
            Scalar::new(),
            query.as_matrix_view(),
            documents.as_matrix_view(),
        );
        assert_eq!(scalar, [expected]);

        #[cfg(target_arch = "x86_64")]
        {
            if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                let actual =
                    run::<_, 16, 4>(arch, query.as_matrix_view(), documents.as_matrix_view());
                assert_eq!(actual, scalar);
            }
            if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
                let actual =
                    run::<_, 32, 6>(arch, query.as_matrix_view(), documents.as_matrix_view());
                assert_eq!(actual, scalar);
            }
        }
    }

    #[test]
    fn rejects_inconsistent_output_and_storage_dimensions() {
        let query = vec![0; 2 * 4];
        let query = MatRef::new(Standard::<u8>::new(2, 4).unwrap(), &query).unwrap();
        let query = BlockTransposed::<u8, 8, 2>::from_matrix_view(query.as_matrix_view());
        let k = DimK::new(NonZeroUsize::new(4).unwrap());
        let a = packed::View::from_block_transposed(query.as_view()).unwrap();
        let documents = [0; 8];
        let documents = MatRef::new(Standard::<u8>::new(2, 4).unwrap(), &documents).unwrap();
        let b = unpacked::View::from_matrix_view(documents.as_matrix_view()).unwrap();

        let message = panic_message_for(|| {
            let mut c = [0; 9];
            // SAFETY: The deliberate output mismatch is caught by checked test bounds.
            let _ =
                unsafe { Driver::<_, 8, 2>::new(Scalar::new(), a, b, &mut c, k, Cache::detect()) };
        });
        assert!(message.contains("packed A blocks"));

        let wrong_k = DimK::new(NonZeroUsize::new(2).unwrap());
        let message = panic_message_for(|| {
            let mut c = [0; 2];
            // SAFETY: The deliberate storage mismatch is caught by checked test bounds.
            let _ = unsafe {
                Driver::<_, 8, 2>::new(Scalar::new(), a, b, &mut c, wrong_k, Cache::detect())
            };
        });
        assert!(message.contains("contraction dimensions do not agree"));

        let short_documents = [0; 4];
        let short_documents =
            MatRef::new(Standard::<u8>::new(2, 2).unwrap(), &short_documents).unwrap();
        let short_b = unpacked::View::from_matrix_view(short_documents.as_matrix_view()).unwrap();
        let message = panic_message_for(|| {
            let mut c = [0; 2];
            // SAFETY: The deliberate document storage mismatch is checked by the driver.
            let _ = unsafe {
                Driver::<_, 8, 2>::new(Scalar::new(), a, short_b, &mut c, k, Cache::detect())
            };
        });
        assert!(message.contains("contraction dimensions do not agree"));
    }

    #[test]
    fn accumulator_bounds_are_exact() {
        assert!(MAX_K * MAX_PRODUCT <= u32::MAX as usize);
        assert!((MAX_K + 2) * MAX_PRODUCT > u32::MAX as usize);
    }
}
