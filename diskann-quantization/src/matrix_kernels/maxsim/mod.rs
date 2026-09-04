/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Implementation of "maxsim" kernels.
//!
//! Let:
//!
//! * `A` be a `M x K` matrix.
//! * `B` be a `K x N` matrix.
//!
//! The "maxsim" result `C` is a `M`-dimensional vector where
//!
//! ```text
//! C[i] = max(j in 0..N, dot(A[i, :], B[:, j]))
//! ```

use std::num::NonZeroUsize;

use super::{
    Cache,
    num::{Bytes, value_or_one},
};

/// Blocking parameters for packed by unpacked MaxSim kernels.
#[derive(Debug, Clone, Copy)]
pub(super) struct Params {
    /// The approximate number of A panels that fit in the L2 cache.
    pub(super) a_panels_in_l2: NonZeroUsize,
    /// The approximate number of B columns that fit in the L1 cache.
    pub(super) b_cols_in_l1: NonZeroUsize,
}

impl Params {
    pub(super) fn new(cache: Cache, a_panel: Bytes, b_col: Bytes, nr: usize) -> Self {
        let a_panels_in_l2 = value_or_one(cache.l2().get() / a_panel.value());
        let b_budget = cache.l1().get().saturating_sub(a_panel.value()).max(1);
        let b_cols_in_l1 = value_or_one(nr * b_budget.div_ceil(nr * b_col.value()));

        Self {
            a_panels_in_l2,
            b_cols_in_l1,
        }
    }
}

pub(crate) mod packed_f32_x_unpacked_f16;
pub(crate) mod packed_f32_x_unpacked_f32;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod packed_u8_x_unpacked_u4;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod packed_u8_x_unpacked_u8;

#[cfg(test)]
mod test;
