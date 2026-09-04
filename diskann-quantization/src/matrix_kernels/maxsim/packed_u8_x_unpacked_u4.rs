/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! This closely follows the implementation in [`super::packed_f32_x_unpacked_f16`].
//!
//! Dense `u4` sub-views of `b` are expanded to `u8` before any panel-kernel operation.

use std::num::NonZeroUsize;

use diskann_wide::arch::Architecture;

use crate::matrix_kernels::{
    Cache,
    blocks::{packed, unpacked},
    bounds, driver,
    num::{DimK, Elements},
    ptr::{MutSlice, Slice},
    util,
};

use super::{Params, packed_u8_x_unpacked_u8::PanelKernel};

const MAX_PRODUCT: usize = u8::MAX as usize * 0x0f;
const MAX_K: usize = (u32::MAX as usize / MAX_PRODUCT) / 2 * 2;

/// A driver for prepacked `u8` by dense `u4` MaxSim computations.
///
/// Results are returned directly in `c`.
///
/// # Class Invariants
///
/// 1. `a.k()` must be equal to `k`, and `b.k() * 2` must be equal to `k`.
/// 2. `c.len().div_ceil(MR)` must be equal to `a.blocks()`.
/// 3. The maximum dot product for `k` elements fits in `u32`.
pub(crate) struct Driver<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::View<'a, u8, MR>,
    b: unpacked::View<'a, u8>,
    c: &'a mut [u32],
    k: DimK,
    b_converted: Vec<u8>,
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
        let params = Params::new(
            cache,
            a.block_stride(k).bytes(),
            Elements::<u8>::new(k.value().get()).bytes(),
            NR,
        );

        // SAFETY: Inherited from caller.
        unsafe { Self::new_inner(arch, a, b, c, k, params) }
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
        bounds::check_eq!(a.k(), k, "contraction dimensions do not agree");
        bounds::check_eq!(
            b.k() * bounds::Bound::new(2),
            k,
            "dense-u4 storage does not match the contraction dimension"
        );
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
            b_converted: vec![0; params.b_cols_in_l1.get() * k.value().get()],
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
                self.c.fill(0);

                let remainder = self.c.len() % MR;
                let last_a_block = self.a.blocks().get() - 1;
                let mut c = MutSlice::new(self.c);
                let b_k = DimK::new(
                    NonZeroUsize::new(self.k.value().get() / 2).unwrap_or(NonZeroUsize::MIN),
                );

                let on_a_panels = |a_panels: packed::View<'_, u8, MR>, a_block_base| {
                    let on_b_panels = |b_panels: unpacked::View<'_, u8>, _| {
                        // SAFETY: `b_panels.k()` equals `b_k` by class invariant.
                        let b_packed = unsafe { b_panels.as_std_slice(b_k) };
                        let b_converted =
                            &mut self.b_converted[..b_panels.extent().get() * self.k.value().get()];

                        for (source, target) in b_packed
                            .chunks_exact(b_k.value().get())
                            .zip(b_converted.chunks_exact_mut(self.k.value().get()))
                        {
                            for (&pair, values) in source.iter().zip(target.chunks_exact_mut(2)) {
                                values[0] = pair & 0x0f;
                                values[1] = pair >> 4;
                            }
                        }

                        // SAFETY: `b_converted` contains
                        // `b_panels.extent() * self.k` elements.
                        let b_panels_converted = unsafe {
                            unpacked::View::new(Slice::new(b_converted), b_panels.extent(), self.k)
                        };

                        let panel_kernel = |a_panel: packed::Panel<'_, u8, MR>, a_block_offset| {
                            let a_block = a_block_base + a_block_offset;
                            let handling_tail = a_block == last_a_block && remainder != 0;
                            let bound =
                                bounds::Bound::from_fn(
                                    || {
                                        if handling_tail { remainder } else { MR }
                                    },
                                );

                            // SAFETY: `c` occupies exactly the packed A blocks.
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

                            // SAFETY: `a_panel.k()` and `b_panels_converted.k()` both equal
                            // `self.k`.
                            let mut kernel = unsafe {
                                PanelKernel::new(self.arch, a_panel, b_panels_converted, c, self.k)
                            };
                            driver::PanelKernel::panel_kernel(&mut kernel);

                            let c_final = kernel.take();
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

                        // SAFETY: `a_panels.k()` equals `self.k`.
                        unsafe { a_panels.visit_panels(self.k, panel_kernel) };
                    };

                    // SAFETY: `self.b.k()` equals `b_k`.
                    unsafe {
                        self.b
                            .visit_sub_views(self.params.b_cols_in_l1, b_k, on_b_panels);
                    }
                };

                // SAFETY: `self.a.k()` equals `self.k`.
                unsafe {
                    self.a
                        .visit_sub_views(self.params.a_panels_in_l2, self.k, on_a_panels);
                }
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use diskann_utils::views::MatrixView;
    use diskann_wide::arch::Scalar;

    use crate::{
        bits::{BoxedBitSlice, Dense, Unsigned},
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

    fn pack_documents(values: &[u8], rows: usize, dim: usize) -> Vec<u8> {
        let mut packed = Vec::with_capacity(rows * dim.div_ceil(2));
        for row in values.chunks_exact(dim) {
            let mut bits = BoxedBitSlice::<4, Unsigned, Dense>::new_boxed(dim);
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
        let query_values = query_values(query_rows, dim);
        let document_values = document_values(document_rows, dim);
        let packed_documents = pack_documents(&document_values, document_rows, dim);
        let query =
            MatRef::new(Standard::<u8>::new(query_rows, dim).unwrap(), &query_values).unwrap();
        let documents = MatRef::new(
            Standard::<u8>::new(document_rows, dim).unwrap(),
            &document_values,
        )
        .unwrap();
        let packed_documents = MatRef::new(
            Standard::<u8>::new(document_rows, dim.div_ceil(2)).unwrap(),
            &packed_documents,
        )
        .unwrap();
        let expected = reference(query.as_matrix_view(), documents.as_matrix_view());

        assert_eq!(
            run::<_, 8, 2>(
                Scalar::new(),
                query.as_matrix_view(),
                packed_documents.as_matrix_view(),
            ),
            expected
        );

        #[cfg(target_arch = "x86_64")]
        {
            if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                assert_eq!(
                    run::<_, 16, 4>(
                        arch,
                        query.as_matrix_view(),
                        packed_documents.as_matrix_view(),
                    ),
                    expected
                );
            }
            if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
                assert_eq!(
                    run::<_, 16, 6>(
                        arch,
                        query.as_matrix_view(),
                        packed_documents.as_matrix_view(),
                    ),
                    expected
                );
                assert_eq!(
                    run::<_, 32, 6>(
                        arch,
                        query.as_matrix_view(),
                        packed_documents.as_matrix_view(),
                    ),
                    expected
                );
            }
        }

        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            assert_eq!(
                run::<_, 8, 6>(
                    arch,
                    query.as_matrix_view(),
                    packed_documents.as_matrix_view(),
                ),
                expected
            );
        }
    }

    #[test]
    fn matches_reference_at_panel_boundaries() {
        for m in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            for n in [1, 2, 3, 4, 5, 6, 7, 13] {
                for k in [1, 2, 3, 31, 32, 33] {
                    check_case(m, n, k);
                }
            }
        }
    }

    #[test]
    fn handles_maximum_values_and_dirty_odd_nibble() {
        const M: usize = 33;
        const N: usize = 7;
        const K: usize = 257;

        let query = vec![u8::MAX; M * K];
        let documents = vec![0x0f; N * K];
        let mut packed_documents = pack_documents(&documents, N, K);
        for row in 0..N {
            packed_documents[row * K.div_ceil(2) + K / 2] |= 0xf0;
        }

        let query = MatRef::new(Standard::<u8>::new(M, K).unwrap(), &query).unwrap();
        let packed_documents = MatRef::new(
            Standard::<u8>::new(N, K.div_ceil(2)).unwrap(),
            &packed_documents,
        )
        .unwrap();
        let expected = u32::from(u8::MAX) * 0x0f * K as u32;
        let actual = run::<_, 8, 2>(
            Scalar::new(),
            query.as_matrix_view(),
            packed_documents.as_matrix_view(),
        );
        assert!(actual.iter().all(|&value| value == expected));
    }

    #[test]
    fn reads_repository_dense_bitslice_layout() {
        const K: usize = 9;
        let query_values = [3u8, 5, 7, 11, 13, 17, 19, 23, 29];
        let document_values = [1u8, 2, 3, 4, 5, 6, 7, 8, 15];
        let query = MatRef::new(Standard::<u8>::new(1, K).unwrap(), &query_values).unwrap();

        let mut document = BoxedBitSlice::<4, Unsigned, Dense>::new_boxed(K);
        for (index, &value) in document_values.iter().enumerate() {
            document.set(index, i64::from(value)).unwrap();
        }
        let packed_document = MatRef::new(
            Standard::<u8>::new(1, K.div_ceil(2)).unwrap(),
            document.as_slice(),
        )
        .unwrap();

        let expected = query_values
            .iter()
            .zip(document_values)
            .map(|(&a, b)| u32::from(a) * u32::from(b))
            .sum::<u32>();
        assert_eq!(
            run::<_, 8, 2>(
                Scalar::new(),
                query.as_matrix_view(),
                packed_document.as_matrix_view(),
            ),
            [expected]
        );
    }

    #[test]
    fn accumulates_past_i32_max() {
        const K: usize = 600_000;
        let query = vec![u8::MAX; K];
        let documents = vec![0x0f; K];
        let packed_documents = pack_documents(&documents, 1, K);
        let query = MatRef::new(Standard::<u8>::new(1, K).unwrap(), &query).unwrap();
        let packed_documents =
            MatRef::new(Standard::<u8>::new(1, K / 2).unwrap(), &packed_documents).unwrap();
        let expected = u32::from(u8::MAX) * 0x0f * K as u32;
        assert!(expected > i32::MAX as u32);
        assert_eq!(
            run::<_, 8, 2>(
                Scalar::new(),
                query.as_matrix_view(),
                packed_documents.as_matrix_view(),
            ),
            [expected]
        );
    }

    #[test]
    fn rejects_inconsistent_output_and_storage_dimensions() {
        let query = [0; 8];
        let query = MatRef::new(Standard::<u8>::new(2, 4).unwrap(), &query).unwrap();
        let query = BlockTransposed::<u8, 8, 2>::from_matrix_view(query.as_matrix_view());
        let k = DimK::new(NonZeroUsize::new(4).unwrap());
        let a = packed::View::from_block_transposed(query.as_view()).unwrap();
        let documents = [0; 4];
        let documents = MatRef::new(Standard::<u8>::new(2, 2).unwrap(), &documents).unwrap();
        let b = unpacked::View::from_matrix_view(documents.as_matrix_view()).unwrap();

        let message = panic_message_for(|| {
            let mut c = [0; 9];
            // SAFETY: The deliberate output mismatch is checked by the driver.
            let _ =
                unsafe { Driver::<_, 8, 2>::new(Scalar::new(), a, b, &mut c, k, Cache::detect()) };
        });
        assert!(message.contains("packed A blocks"));

        let wrong_k = DimK::new(NonZeroUsize::new(2).unwrap());
        let message = panic_message_for(|| {
            let mut c = [0; 2];
            // SAFETY: The deliberate contraction mismatch is checked by the driver.
            let _ = unsafe {
                Driver::<_, 8, 2>::new(Scalar::new(), a, b, &mut c, wrong_k, Cache::detect())
            };
        });
        assert!(message.contains("expected 4 to be equal to 2"));

        let short_documents = [0; 2];
        let short_documents =
            MatRef::new(Standard::<u8>::new(2, 1).unwrap(), &short_documents).unwrap();
        let short_b = unpacked::View::from_matrix_view(short_documents.as_matrix_view()).unwrap();
        let message = panic_message_for(|| {
            let mut c = [0; 2];
            // SAFETY: The deliberate dense-u4 mismatch is checked by the driver.
            let _ = unsafe {
                Driver::<_, 8, 2>::new(Scalar::new(), a, short_b, &mut c, k, Cache::detect())
            };
        });
        assert!(message.contains("dense-u4 storage"));
    }

    #[test]
    fn accumulator_bounds_are_exact() {
        assert!(MAX_K * MAX_PRODUCT <= u32::MAX as usize);
        assert!((MAX_K + 2) * MAX_PRODUCT > u32::MAX as usize);
    }
}
