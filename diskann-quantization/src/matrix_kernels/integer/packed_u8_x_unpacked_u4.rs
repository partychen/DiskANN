/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Raw dot products for a packed `u8` query matrix and unpacked `u4` document matrix.
//!
//! This kernel intentionally stops at completed integer dot products. Quantization correction
//! and reduction belong in a later layer because both MinMax and Spherical need their metadata
//! after the integer contraction completes.
//!
//! "Packed" and "unpacked" describe the matrix layouts, matching the other matrix-kernel modules.
//! The unpacked document matrix is row-major; its `u4` elements are encoded two per byte.

use crate::multi_vector::BlockTransposedRef;

const QUERY_K_PACK: usize = 2;
pub(crate) const V3_MR: usize = 16;
pub(crate) const V3_NR: usize = 4;
const MAX_DIM: usize = i32::MAX as usize / (u8::MAX as usize * 0x0f);

/// Errors returned when the logical matrix dimensions do not match their backing storage.
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

        let row_bytes = dim.div_ceil(2);
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

    fn value(self, row: usize, col: usize) -> u8 {
        let byte = self.data[row * self.row_bytes + col / 2];
        if col.is_multiple_of(2) {
            byte & 0x0f
        } else {
            byte >> 4
        }
    }
}

fn validate<const MR: usize>(
    query: BlockTransposedRef<'_, u8, MR, QUERY_K_PACK>,
    docs: UnpackedU4Rows<'_>,
    output: &[i32],
) -> Result<(), ShapeError> {
    if query.ncols() != docs.dim {
        return Err(ShapeError::DimensionMismatch {
            query: query.ncols(),
            document: docs.dim,
        });
    }
    if query.ncols() > MAX_DIM {
        return Err(ShapeError::DimensionTooLarge {
            dimension: query.ncols(),
            maximum: MAX_DIM,
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

/// Compute raw integer dot products with a scalar implementation.
///
/// Output is row-major with shape `query.nrows() x docs.rows`.
pub(crate) fn compute_scalar<const MR: usize>(
    query: BlockTransposedRef<'_, u8, MR, QUERY_K_PACK>,
    docs: UnpackedU4Rows<'_>,
    output: &mut [i32],
) -> Result<(), ShapeError> {
    validate(query, docs, output)?;

    let packed = query.as_slice();
    let padded_k = query.padded_ncols();
    let block_stride = MR * padded_k;

    for q in 0..query.nrows() {
        let block = q / MR;
        let row = q % MR;
        let block_base = block * block_stride;

        for d in 0..docs.rows {
            let mut sum = 0i32;
            for k in 0..query.ncols() {
                let query_offset = block_base
                    + (k / QUERY_K_PACK) * MR * QUERY_K_PACK
                    + row * QUERY_K_PACK
                    + k % QUERY_K_PACK;
                sum += i32::from(packed[query_offset]) * i32::from(docs.value(d, k));
            }
            output[q * docs.rows + d] = sum;
        }
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn compute_v3(
    arch: diskann_wide::arch::x86_64::V3,
    query: BlockTransposedRef<'_, u8, V3_MR, QUERY_K_PACK>,
    docs: UnpackedU4Rows<'_>,
    output: &mut [i32],
) -> Result<(), ShapeError> {
    use diskann_wide::Architecture;

    validate(query, docs, output)?;

    arch.run(
        #[inline]
        || {
            let packed = query.as_slice();
            let block_stride = V3_MR * query.padded_ncols();
            let query_blocks = query.padded_nrows() / V3_MR;

            for query_block in 0..query_blocks {
                let query_base = query_block * V3_MR;
                let valid_queries = (query.nrows() - query_base).min(V3_MR);
                let query_panel = packed[query_block * block_stride..].as_ptr();

                for doc_base in (0..docs.rows).step_by(V3_NR) {
                    let remaining = docs.rows - doc_base;
                    match remaining.min(V3_NR) {
                        1 => compute_doc_block_v3::<1>(
                            query_panel,
                            docs,
                            doc_base,
                            query_base,
                            valid_queries,
                            output,
                        ),
                        2 => compute_doc_block_v3::<2>(
                            query_panel,
                            docs,
                            doc_base,
                            query_base,
                            valid_queries,
                            output,
                        ),
                        3 => compute_doc_block_v3::<3>(
                            query_panel,
                            docs,
                            doc_base,
                            query_base,
                            valid_queries,
                            output,
                        ),
                        4 => compute_doc_block_v3::<4>(
                            query_panel,
                            docs,
                            doc_base,
                            query_base,
                            valid_queries,
                            output,
                        ),
                        _ => unreachable!("step_by always leaves between one and V3_NR documents"),
                    }
                }
            }
        },
    );

    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn compute_doc_block_v3<const NR: usize>(
    query_panel: *const u8,
    docs: UnpackedU4Rows<'_>,
    doc_base: usize,
    query_base: usize,
    valid_queries: usize,
    output: &mut [i32],
) {
    // SAFETY: `compute_v3` validates both matrix shapes. `query_panel` addresses a complete
    // padded V3 query block, and the dispatch guarantees `doc_base + NR <= docs.rows`.
    let results = unsafe { microkernel_v3::<NR>(query_panel, docs, doc_base) };
    for (doc_offset, column) in results.iter().enumerate() {
        for query_offset in 0..valid_queries {
            output[(query_base + query_offset) * docs.rows + doc_base + doc_offset] =
                column[query_offset];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn microkernel_v3<const NR: usize>(
    query_panel: *const u8,
    docs: UnpackedU4Rows<'_>,
    doc_base: usize,
) -> [[i32; V3_MR]; NR] {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi16_epi32,
        _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_maddubs_epi16, _mm256_set1_epi16,
        _mm256_setzero_si256, _mm256_storeu_si256,
    };

    // SAFETY: The caller guarantees a complete padded query panel and `NR` valid document rows.
    // V3 guarantees all AVX2 instructions used below. Pair products are at most
    // `2 * 255 * 15 = 7,650`, so `_mm256_maddubs_epi16` cannot saturate. `validate` bounds the
    // contraction dimension so the i32 accumulators cannot overflow.
    unsafe {
        let zero = _mm256_setzero_si256();
        let mut low: [__m256i; NR] = [zero; NR];
        let mut high: [__m256i; NR] = [zero; NR];
        let pairs = docs.dim.div_ceil(2);

        for pair_index in 0..pairs {
            // `BlockTransposed<_, V3_MR, 2>` stores one complete query column-pair in
            // `V3_MR * 2` contiguous bytes.
            let a = _mm256_loadu_si256(query_panel.add(pair_index * V3_MR * QUERY_K_PACK).cast());

            for doc_offset in 0..NR {
                let packed_doc = docs.data[(doc_base + doc_offset) * docs.row_bytes + pair_index];
                let low_nibble = packed_doc & 0x0f;
                let high_nibble =
                    if pair_index + 1 == pairs && !docs.dim.is_multiple_of(QUERY_K_PACK) {
                        0
                    } else {
                        packed_doc >> 4
                    };
                let pair = i16::from(low_nibble) | (i16::from(high_nibble) << 8);
                let b = _mm256_set1_epi16(pair);

                // Each i16 lane is one query row's two-dimension partial dot product.
                let partial = _mm256_maddubs_epi16(a, b);
                let partial_low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(partial));
                let partial_high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(partial, 1));

                low[doc_offset] = _mm256_add_epi32(low[doc_offset], partial_low);
                high[doc_offset] = _mm256_add_epi32(high[doc_offset], partial_high);
            }
        }

        let mut output = [[0i32; V3_MR]; NR];
        for doc_offset in 0..NR {
            let mut signed = [0i32; V3_MR];
            _mm256_storeu_si256(signed.as_mut_ptr().cast(), low[doc_offset]);
            _mm256_storeu_si256(signed.as_mut_ptr().add(8).cast(), high[doc_offset]);
            output[doc_offset] = signed;
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::multi_vector::{BlockTransposed, MatRef, Standard};

    fn query_values(rows: usize, dim: usize) -> Vec<u8> {
        (0..rows * dim)
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .collect()
    }

    fn doc_values(rows: usize, dim: usize) -> Vec<u8> {
        (0..rows * dim).map(|i| ((i * 7 + 3) % 16) as u8).collect()
    }

    fn pack_u4(values: &[u8], rows: usize, dim: usize) -> Vec<u8> {
        let row_bytes = dim.div_ceil(2);
        let mut packed = vec![0u8; rows * row_bytes];
        for row in 0..rows {
            for col in 0..dim {
                let value = values[row * dim + col];
                assert!(value <= 0x0f);
                let dst = &mut packed[row * row_bytes + col / 2];
                if col.is_multiple_of(2) {
                    *dst |= value;
                } else {
                    *dst |= value << 4;
                }
            }
        }
        packed
    }

    fn reference(
        query: &[u8],
        query_rows: usize,
        docs: &[u8],
        doc_rows: usize,
        dim: usize,
    ) -> Vec<i32> {
        let mut output = vec![0i32; query_rows * doc_rows];
        for q in 0..query_rows {
            for d in 0..doc_rows {
                output[q * doc_rows + d] = (0..dim)
                    .map(|k| i32::from(query[q * dim + k]) * i32::from(docs[d * dim + k]))
                    .sum();
            }
        }
        output
    }

    fn run_case(query_rows: usize, doc_rows: usize, dim: usize) {
        let query = query_values(query_rows, dim);
        let docs = doc_values(doc_rows, dim);
        let packed_docs = pack_u4(&docs, doc_rows, dim);

        let query_mat = MatRef::new(Standard::<u8>::new(query_rows, dim).unwrap(), &query).unwrap();
        let packed_query = BlockTransposed::<u8, V3_MR, QUERY_K_PACK>::from_matrix_view(
            query_mat.as_matrix_view(),
        );
        let docs_view = UnpackedU4Rows::new(&packed_docs, doc_rows, dim).unwrap();
        let expected = reference(&query, query_rows, &docs, doc_rows, dim);

        let mut scalar = vec![0i32; expected.len()];
        compute_scalar(packed_query.as_view(), docs_view, &mut scalar).unwrap();
        assert_eq!(
            scalar, expected,
            "scalar shape ({query_rows}, {doc_rows}, {dim})"
        );

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            let mut actual = vec![0i32; expected.len()];
            compute_v3(arch, packed_query.as_view(), docs_view, &mut actual).unwrap();
            assert_eq!(
                actual, expected,
                "V3 shape ({query_rows}, {doc_rows}, {dim})"
            );
        }
    }

    #[test]
    fn scalar_and_v3_match_reference() {
        for (query_rows, doc_rows, dim) in [
            (1, 1, 1),
            (5, 3, 7),
            (16, 4, 64),
            (16, 16, 256),
            (17, 5, 65),
            (32, 9, 128),
        ] {
            run_case(query_rows, doc_rows, dim);
        }
    }

    #[test]
    fn maximum_values_do_not_overflow_intermediate_lanes() {
        const QUERY_ROWS: usize = 16;
        const DOC_ROWS: usize = 5;
        const DIM: usize = 257;

        let query = vec![u8::MAX; QUERY_ROWS * DIM];
        let docs = vec![0x0f; DOC_ROWS * DIM];
        let packed_docs = pack_u4(&docs, DOC_ROWS, DIM);
        let query_mat = MatRef::new(Standard::<u8>::new(QUERY_ROWS, DIM).unwrap(), &query).unwrap();
        let packed_query = BlockTransposed::<u8, V3_MR, QUERY_K_PACK>::from_matrix_view(
            query_mat.as_matrix_view(),
        );
        let docs_view = UnpackedU4Rows::new(&packed_docs, DOC_ROWS, DIM).unwrap();
        let expected = i32::from(u8::MAX) * 0x0f * DIM as i32;

        let mut scalar = vec![0i32; QUERY_ROWS * DOC_ROWS];
        compute_scalar(packed_query.as_view(), docs_view, &mut scalar).unwrap();
        assert!(scalar.iter().all(|&value| value == expected));

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            let mut actual = vec![0i32; QUERY_ROWS * DOC_ROWS];
            compute_v3(arch, packed_query.as_view(), docs_view, &mut actual).unwrap();
            assert_eq!(actual, scalar);
        }
    }

    #[test]
    fn ignores_unused_high_nibble_for_odd_dimensions() {
        const QUERY_ROWS: usize = 3;
        const DOC_ROWS: usize = 2;
        const DIM: usize = 3;

        let query = query_values(QUERY_ROWS, DIM);
        let docs = doc_values(DOC_ROWS, DIM);
        let mut packed_docs = pack_u4(&docs, DOC_ROWS, DIM);
        for row in 0..DOC_ROWS {
            packed_docs[row * DIM.div_ceil(2) + 1] |= 0xf0;
        }

        let query_mat = MatRef::new(Standard::<u8>::new(QUERY_ROWS, DIM).unwrap(), &query).unwrap();
        let packed_query = BlockTransposed::<u8, V3_MR, QUERY_K_PACK>::from_matrix_view(
            query_mat.as_matrix_view(),
        );
        let docs_view = UnpackedU4Rows::new(&packed_docs, DOC_ROWS, DIM).unwrap();
        let expected = reference(&query, QUERY_ROWS, &docs, DOC_ROWS, DIM);

        let mut scalar = vec![0i32; expected.len()];
        compute_scalar(packed_query.as_view(), docs_view, &mut scalar).unwrap();
        assert_eq!(scalar, expected);

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            let mut actual = vec![0i32; expected.len()];
            compute_v3(arch, packed_query.as_view(), docs_view, &mut actual).unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn rejects_invalid_storage_shapes() {
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

        let query = query_values(2, 4);
        let query_mat = MatRef::new(Standard::<u8>::new(2, 4).unwrap(), &query).unwrap();
        let packed_query = BlockTransposed::<u8, V3_MR, QUERY_K_PACK>::from_matrix_view(
            query_mat.as_matrix_view(),
        );
        let docs = UnpackedU4Rows::new(&[0; 2], 1, 4).unwrap();
        assert_eq!(
            compute_scalar(packed_query.as_view(), docs, &mut [0; 1]),
            Err(ShapeError::OutputStorage {
                expected: 2,
                actual: 1,
            })
        );
    }
}
