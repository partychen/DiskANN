/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

use crate::SIMDVector;

use super::{V3, V4, v3, v4};

/// Expand the four unsigned nibbles in `packed`, least significant first, and repeat
/// the resulting four bytes across the vector. Each output lane is in `0..=15`.
#[inline(always)]
pub fn splat_u4x4(arch: V3, packed: u16) -> v3::i8x32 {
    if cfg!(miri) {
        v3::i8x32::from_array(
            arch,
            core::array::from_fn(|i| ((packed >> (4 * (i % 4))) & 0x0f) as i8),
        )
    } else {
        let packed = v3::i16x16::splat(arch, packed as i16).to_underlying();
        let mask = v3::i8x32::splat(arch, 0x0f);
        let low = v3::i8x32::from_underlying(arch, packed) & mask;
        // SAFETY: V3 provides AVX2.
        unsafe {
            let high = v3::i8x32::from_underlying(arch, _mm256_srli_epi16::<4>(packed)) & mask;
            v3::i8x32::from_underlying(
                arch,
                _mm256_unpacklo_epi8(low.to_underlying(), high.to_underlying()),
            )
        }
    }
}

/// Multiply unsigned bytes by signed bytes, add adjacent products, and saturate
/// each sum to `i16`. Output lane `i` uses input lanes `2 * i` and `2 * i + 1`.
///
/// Unlike [`crate::SIMDDotProduct`], this operation saturates before any wider accumulation.
#[inline(always)]
pub fn multiply_sum_saturating_u8x32_i8x32(a: v3::u8x32, b: v3::i8x32) -> v3::i16x16 {
    let arch = a.arch();
    if cfg!(miri) {
        let a = a.to_array();
        let b = b.to_array();
        v3::i16x16::from_array(
            arch,
            core::array::from_fn(|i| {
                let x0 = i32::from(a[2 * i]) * i32::from(b[2 * i]);
                let x1 = i32::from(a[2 * i + 1]) * i32::from(b[2 * i + 1]);
                (x0 + x1).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
            }),
        )
    } else {
        // SAFETY: The input vectors imply V3, which provides AVX2.
        v3::i16x16::from_underlying(arch, unsafe {
            _mm256_maddubs_epi16(a.to_underlying(), b.to_underlying())
        })
    }
}

/// Expand the eight unsigned nibbles in `packed`, least significant first, and repeat
/// the resulting eight bytes across the vector. Each output lane is in `0..=15`.
///
/// Uses BMI2 bit deposit followed by a 512-bit broadcast.
#[inline(always)]
pub fn splat_u4x8(arch: V4, packed: u32) -> v4::i8x64 {
    #[cfg(miri)]
    {
        v4::i8x64::from_array(
            arch,
            core::array::from_fn(|i| ((packed >> (4 * (i % 8))) & 0x0f) as i8),
        )
    }
    #[cfg(not(miri))]
    {
        // SAFETY: V4 includes BMI2.
        let expanded = unsafe { _pdep_u64(u64::from(packed), 0x0f0f_0f0f_0f0f_0f0f) };
        v4::i8x64::from_underlying(arch, v4::u64x8::splat(arch, expanded).to_underlying())
    }
}

/// Sum adjacent `i32` lanes with wrapping arithmetic.
///
/// Output lane `i` is `values[2 * i].wrapping_add(values[2 * i + 1])`.
#[inline(always)]
pub fn sum_adjacent_i32x16(values: v4::i32x16) -> v4::i32x8 {
    let arch = values.arch();
    #[cfg(miri)]
    {
        let values = values.to_array();
        v4::i32x8::from_array(
            arch,
            core::array::from_fn(|i| values[2 * i].wrapping_add(values[2 * i + 1])),
        )
    }
    #[cfg(not(miri))]
    {
        let upper = v4::u64x8::from_underlying(arch, values.to_underlying()) >> 32;
        let pairs = values + v4::i32x16::from_underlying(arch, upper.to_underlying());
        // SAFETY: The input vector implies V4, which provides AVX-512F.
        v4::i32x8::from_underlying(arch, unsafe {
            _mm512_cvtepi64_epi32(pairs.to_underlying())
        })
    }
}

/// Efficiently load the first `8 < bytes < 16` bytes from `ptr` without accessing memory
/// outside of `[ptr, ptr + bytes)`.
///
/// # Safety
///
/// * `bytes` must be in the range `(8, 16)`.
/// * The memory in `[ptr, ptr + bytes)` must be readable and valid.
#[inline(always)]
unsafe fn __load_8_to_16_bytes(_: V3, ptr: *const u8, bytes: usize) -> __m128i {
    debug_assert!(bytes > 8 && bytes < 16);

    // The trick here is to use 2 8-byte loads. One (call it X) beginning at `ptr` loading
    // `[ptr, ptr + 8)` and the other (call it Y) loading `[ptr + bytes - 8, ptr + bytes)`.
    //
    // Then, we need a way to glue Y after the first `bytes - 8` bytes of X (formulating the
    // problem this way is done intentionally as we'll see below).
    //
    // We do this using the powerful `_mm_shuffle_epi8` instruction.
    //
    // This is set up by using an identity shuffle adjusted by subtracting the shift amount.
    // Lanes that underflow become negative (high bit set), which `_mm_shuffle_epi8` zeroes.
    // Lanes beyond the loaded 8 bytes read from the zero-extended upper half of
    // `_mm_loadl_epi64`, producing zeros that are harmless under OR.
    //
    // For example, if `bytes` is 13 (8 + 5), the adjusted shuffle mask is
    // ```
    // [-X, -X, -X, -X, -X, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    // |-----------------|
    //  output lanes here
    //    will be zeroed
    // ```
    // This will effectively move the 8 bytes of Y each over by 5 lanes. When OR'ed with X,
    // this becomes the 13 bytes we want.
    //
    // SAFETY: Both reads are within `[ptr, ptr + bytes)`. The intrinsics require SSSE3/SSE2,
    // available on V3.
    unsafe {
        let base = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let lo = _mm_loadl_epi64(ptr as *const __m128i);
        let hi = _mm_loadl_epi64(ptr.add(bytes - 8) as *const __m128i);
        let mask = _mm_sub_epi8(base, _mm_set1_epi8((bytes - 8) as i8));
        _mm_or_si128(lo, _mm_shuffle_epi8(hi, mask))
    }
}

/// Perform a load of the first `first` bytes beginning at `ptr` into
/// an unsigned 128-bit integer.
///
/// For clarity, the first byte beginning at `ptr` will occupy the lowest 8-bits of the
/// returned integer. The next byte will occupy bits 8 to 15 etc.
///
/// Memory addresses equal to and above `ptr + first` will not be accessed.
///
/// Note: This is actually faster than memcpy (since memcpy cannot be inlined).
///
/// # Safety
///
/// The memory addresses in  `[ptr, ptr + first)` must all be readable and valid.
///
/// Guarantee: Memory addresses in `[ptr + first, ptr + 16)` will not be accessed.
#[inline(always)]
pub(crate) unsafe fn __load_first_of_16_bytes(arch: V3, ptr: *const u8, first: usize) -> u128 {
    if first >= 16 {
        // SAFETY:
        // * Pointer Cast: The instruction `_mm_loadu_si128` does not have any alignment
        //     restrictions, so if `[ptr, ptr + first)` is valid, the cast will be valid.
        // * `_mm_loadu_si128`: The intrinsic requires SSE2, implied by V3.
        //     The load is valid since the caller passed a value greater than 16.
        // *`__m128i` and `u128` are both the same size, do not own any resources, and are
        //     valid for all bit patterns.
        return unsafe {
            std::mem::transmute::<__m128i, u128>(_mm_loadu_si128(ptr as *const __m128i))
        };
    }

    // For `first > 8`, use the optimized two-load method.
    if first > 8 {
        // SAFETY: `first` is in `(8, 16)` and `[ptr, ptr + first)` is valid.
        return unsafe {
            std::mem::transmute::<__m128i, u128>(__load_8_to_16_bytes(arch, ptr, first))
        };
    }

    // For `first <= 8`, everything fits in general purpose registers.
    //
    // Use two overlapping reads whose results are combined with a single shift + OR.
    //
    // SAFETY: All reads are within `[ptr, ptr + first)`, which the caller asserts is valid.
    unsafe {
        if first == 8 {
            std::ptr::read_unaligned(ptr as *const u64) as u128
        } else if first >= 4 {
            let lo = std::ptr::read_unaligned(ptr as *const u32) as u64;
            let hi = std::ptr::read_unaligned(ptr.add(first - 4) as *const u32) as u64;
            (lo | (hi << ((first - 4) * 8))) as u128
        } else if first >= 2 {
            let lo = std::ptr::read_unaligned(ptr as *const u16) as u64;
            let hi = std::ptr::read_unaligned(ptr.add(first - 2) as *const u16) as u64;
            (lo | (hi << ((first - 2) * 8))) as u128
        } else if first == 1 {
            std::ptr::read(ptr) as u128
        } else {
            0
        }
    }
}

/// Load the first `first` 16-bit words from `ptr` and return the result as a `__m128i`.
///
/// # Safety
///
/// The memory addresses in `[ptr, ptr + first)` must all be readable and valid.
///
/// This function guarantees that the memory addresses in `[ptr + first, ptr + 16)` will not
/// be accessed.
#[inline(always)]
pub(crate) unsafe fn __load_first_u16_of_16_bytes(
    arch: V3,
    ptr: *const u16,
    first: usize,
) -> __m128i {
    if first >= 8 {
        // SAFETY: All lanes are readable. The intrinsic can be used because `arch` is present.
        return unsafe { _mm_loadu_si128(ptr as *const __m128i) };
    }

    let byte_ptr = ptr as *const u8;
    let bytes = first * 2;

    // For `bytes > 8` (i.e., `first > 4`), use the optimized two-load method.
    if bytes > 8 {
        // SAFETY: `bytes` is in `(8, 16)` and `[byte_ptr, byte_ptr + bytes)` is valid.
        return unsafe { __load_8_to_16_bytes(arch, byte_ptr, bytes) };
    }

    // For `bytes <= 8`, everything fits in general purpose registers.
    //
    // SAFETY: All reads are within `[ptr, ptr + first)`, which the caller
    // asserts is valid.
    unsafe {
        if bytes == 8 {
            let v = std::ptr::read_unaligned(byte_ptr as *const u64);
            _mm_cvtsi64_si128(v as i64)
        } else if bytes >= 4 {
            let lo = std::ptr::read_unaligned(byte_ptr as *const u32) as u64;
            let hi = std::ptr::read_unaligned(byte_ptr.add(bytes - 4) as *const u32) as u64;
            _mm_cvtsi64_si128((lo | (hi << ((bytes - 4) * 8))) as i64)
        } else if bytes >= 2 {
            _mm_cvtsi32_si128(std::ptr::read_unaligned(byte_ptr as *const u16) as i32)
        } else {
            _mm_setzero_si128()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::driver;

    fn test_v4() -> Option<V4> {
        if cfg!(miri) {
            V4::new_checked_miri()
        } else {
            V4::new_checked_uncached()
        }
    }

    #[test]
    fn test_splat_u4x4() {
        let Some(arch) = V3::new_checked_uncached() else {
            return;
        };
        let check = move |input: &[u16]| {
            let bytes = input[0].to_le_bytes();
            let expected = bytes.map(|b| [b & 0x0f, b >> 4]);
            let got = splat_u4x4(arch, input[0]).to_array();
            for (i, lane) in got.into_iter().enumerate() {
                assert_eq!(lane, expected[(i % 4) / 2][i % 2] as i8);
            }
        };
        driver::drive_unary(&check, 1, 0xe60aa814);
        for lane in 0..4 {
            for nibble in 0..=15 {
                check(&[nibble << (4 * lane)]);
            }
        }
    }

    #[test]
    fn test_multiply_sum_saturating_u8x32_i8x32() {
        let Some(arch) = V3::new_checked_uncached() else {
            return;
        };
        let check = move |a: &[u8], b: &[i8]| {
            let expected: [i16; 16] = core::array::from_fn(|i| {
                let sum = (0..2)
                    .map(|j| i32::from(a[2 * i + j]) * i32::from(b[2 * i + j]))
                    .sum::<i32>();
                sum.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
            });
            let a = v3::u8x32::from_array(arch, a.try_into().unwrap());
            let b = v3::i8x32::from_array(arch, b.try_into().unwrap());
            assert_eq!(
                multiply_sum_saturating_u8x32_i8x32(a, b).to_array(),
                expected
            );
        };
        driver::drive_binary(&check, (32, 32), 0x67b693ff);
        check(&[255; 32], &[127; 32]);
        check(&[255; 32], &[-128; 32]);
        check(&[255; 32], &[15; 32]);
    }

    #[test]
    fn test_splat_u4x8() {
        let Some(arch) = test_v4() else {
            return;
        };
        let check = move |input: &[u32]| {
            let bytes = input[0].to_le_bytes();
            let expected = bytes.map(|b| [b & 0x0f, b >> 4]);
            let got = splat_u4x8(arch, input[0]).to_array();
            for (i, lane) in got.into_iter().enumerate() {
                assert_eq!(lane, expected[(i % 8) / 2][i % 2] as i8);
            }
        };
        driver::drive_unary(&check, 1, 0x9578d17b);
        for lane in 0..8 {
            for nibble in 0..=15 {
                check(&[nibble << (4 * lane)]);
            }
        }
    }

    #[test]
    fn test_sum_adjacent_i32x16() {
        let Some(arch) = test_v4() else {
            return;
        };
        let check = move |input: &[i32]| {
            let expected: [i32; 8] =
                core::array::from_fn(|i| input[2 * i].wrapping_add(input[2 * i + 1]));
            let values = v4::i32x16::from_array(arch, input.try_into().unwrap());
            assert_eq!(sum_adjacent_i32x16(values).to_array(), expected);
        };
        driver::drive_unary(&check, 16, 0xf7d2c381);
        check(&[i32::MAX; 16]);
        check(&[i32::MIN; 16]);
        check(&core::array::from_fn::<_, 16, _>(|i| {
            if i % 2 == 0 { i32::MAX } else { 1 }
        }));
    }
}
