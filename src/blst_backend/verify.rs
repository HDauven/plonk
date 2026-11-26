// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! GLV endomorphism decomposition, static Comb table, and pipelined single
//! and batched PLONK verification backed by `blst`.

use alloc::vec::Vec;
use core::ptr;

use blst::{
    blst_fp, blst_fp_cneg, blst_fp_eucl_inverse, blst_fp_mul, blst_fp6,
    blst_fp12, blst_fp12_finalverify, blst_miller_loop_lines, blst_p1,
    blst_p1_add_or_double, blst_p1_add_or_double_affine, blst_p1_affine,
    blst_p1_affine_generator, blst_p1_affine_is_inf, blst_p1_double,
    blst_p1_from_affine, blst_p1_mult, blst_p1_to_affine,
    blst_p1s_mult_pippenger_scratch_sizeof, blst_p1s_to_affine,
};
use dusk_bls12_381::{BlsScalar, G1Affine};
use rayon::prelude::*;

use super::msm::{
    blst_msm_serial_raw_nbits, blst_tile_pippenger_raw, pippenger_window,
    reduce_window_tiles,
};
use super::{
    SCALAR_BYTES, blst_p1s_to_g1_affines, fp_eq, g1_affine_to_blst,
    scalar_to_blst_le_bytes,
};
use crate::proof_system::verifier_msm::{
    DYNAMIC_POINT_COUNT, STATIC_POINT_COUNT,
};

/// BLS12-381 `x^2 = (-0xd201000000010000)^2`, the exact 128-bit GLV eigenvalue
/// magnitude (`r = x^4 - x^2 + 1`).
const GLV_X_SQ: u128 = 0xac45_a401_0001_a402_0000_0001_0000_0000;
pub(super) const GLV_X_SQ_LO: u64 = GLV_X_SQ as u64;
pub(super) const GLV_X_SQ_HI: u64 = (GLV_X_SQ >> 64) as u64;

const COMB_GROUPS: usize = 2;
const COMB_GROUP_SIZE: usize = 8;
const COMB_TEETH: usize = 4;
const COMB_SUBSETS: usize = 256;

/// Divide 3-limb `(u2, u1, u0)` by 2-limb `(d1, d0)` assuming `d1 >= 2^63` and
/// `(u2, u1) < (d1, d0)`. Returns `(q, r1, r0)`.
#[inline(always)]
fn divmod_3_by_2(
    u2: u64,
    u1: u64,
    u0: u64,
    d1: u64,
    d0: u64,
) -> (u64, u64, u64) {
    let mut q = if u2 >= d1 {
        u64::MAX
    } else {
        ((((u2 as u128) << 64) | (u1 as u128)) / (d1 as u128)) as u64
    };
    // Compute 192-bit product p = q * (d1*2^64 + d0)
    let lo = (q as u128) * (d0 as u128);
    let hi = (q as u128) * (d1 as u128) + (lo >> 64);
    let mut p0 = lo as u64;
    let mut p1 = hi as u64;
    let mut p2 = (hi >> 64) as u64;

    while (p2, p1, p0) > (u2, u1, u0) {
        q -= 1;
        let (np0, b0) = p0.overflowing_sub(d0);
        let (np1a, b1a) = p1.overflowing_sub(d1);
        let (np1, b1b) = np1a.overflowing_sub(u64::from(b0));
        let np2 = p2.wrapping_sub(u64::from(b1a || b1b));
        p0 = np0;
        p1 = np1;
        p2 = np2;
    }

    let (r0, b0) = u0.overflowing_sub(p0);
    let r1 = u1.wrapping_sub(p1).wrapping_sub(u64::from(b0));
    (q, r1, r0)
}

/// Decompose canonical 255-bit scalar `s < r = x^4 - x^2 + 1` into
/// `(q, rem)` such that `s = q * x^2 + rem` with `0 <= q < x^2 < 2^128` and
/// `0 <= rem < x^2 < 2^128`.
///
/// Since `phi(P) = (beta * x, y) = [-x^2]P`, `[s]P = [rem]P + [q](-phi(P))`
/// with purely unsigned 128-bit half-scalars!
#[inline(always)]
fn glv_decompose_bytes(bytes: &[u8; SCALAR_BYTES]) -> (u128, u128) {
    let s0 = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let s1 = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let s2 = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let s3 = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
    let (q1, r1, r0) = divmod_3_by_2(s3, s2, s1, GLV_X_SQ_HI, GLV_X_SQ_LO);
    let (q0, rem1, rem0) = divmod_3_by_2(r1, r0, s0, GLV_X_SQ_HI, GLV_X_SQ_LO);
    let q = ((q1 as u128) << 64) | (q0 as u128);
    let rem = ((rem1 as u128) << 64) | (rem0 as u128);
    (q, rem)
}

/// Return the BLS12-381 `Fp` cube root of unity `beta` (in `blst` Montgomery
/// form) such that `phi(x, y) = (beta * x, y) = [-x^2]P` on G1. Derived and
/// self-verified once against `G1::generator()` and `2 * G1::generator()` at
/// initialization time.
fn glv_beta() -> &'static blst_fp {
    static BETA: std::sync::OnceLock<blst_fp> = std::sync::OnceLock::new();
    BETA.get_or_init(|| {
        let x_sq_le = GLV_X_SQ.to_le_bytes();
        let mut g_proj = blst_p1::default();
        let mut g2_proj = blst_p1::default();
        let mut g2_aff = blst_p1_affine::default();
        let mut neg_xsq_g = blst_p1::default();
        let mut neg_xsq_g_aff = blst_p1_affine::default();
        let mut neg_xsq_g2 = blst_p1::default();
        let mut neg_xsq_g2_aff = blst_p1_affine::default();
        let g_aff: blst_p1_affine;
        let mut inv_gx = blst_fp::default();
        let mut beta = blst_fp::default();
        let mut test_x = blst_fp::default();
        // SAFETY: valid `blst` generator, point, and `Fp` pointers.
        unsafe {
            g_aff = *blst_p1_affine_generator();
            blst_p1_from_affine(&mut g_proj, &g_aff);
            blst_p1_double(&mut g2_proj, &g_proj);
            blst_p1_to_affine(&mut g2_aff, &g2_proj);

            blst_p1_mult(&mut neg_xsq_g, &g_proj, x_sq_le.as_ptr(), 128);
            blst_fp_cneg(&mut neg_xsq_g.y, &neg_xsq_g.y, true);
            blst_p1_to_affine(&mut neg_xsq_g_aff, &neg_xsq_g);

            blst_p1_mult(&mut neg_xsq_g2, &g2_proj, x_sq_le.as_ptr(), 128);
            blst_fp_cneg(&mut neg_xsq_g2.y, &neg_xsq_g2.y, true);
            blst_p1_to_affine(&mut neg_xsq_g2_aff, &neg_xsq_g2);

            blst_fp_eucl_inverse(&mut inv_gx, &g_aff.x);
            blst_fp_mul(&mut beta, &neg_xsq_g_aff.x, &inv_gx);
            blst_fp_mul(&mut test_x, &g2_aff.x, &beta);
        }
        assert!(
            fp_eq(&g_aff.y, &neg_xsq_g_aff.y)
                && fp_eq(&test_x, &neg_xsq_g2_aff.x)
                && fp_eq(&g2_aff.y, &neg_xsq_g2_aff.y),
            "BLS12-381 GLV endomorphism self-check failed"
        );
        beta
    })
}

/// Apply `-phi(x, y) = (beta * x, -y)` so `[-phi(P)] = [x^2]P`.
#[inline(always)]
fn glv_neg_phi(p: &blst_p1_affine, beta: &blst_fp) -> blst_p1_affine {
    let mut out = *p;
    // SAFETY: valid `blst_fp` pointers.
    unsafe {
        blst_fp_mul(&mut out.x, &p.x, beta);
        blst_fp_cneg(&mut out.y, &p.y, true);
    }
    out
}

/// Encode a 128-bit unsigned scalar into width-5 signed wNAF digits (`<= 129`
/// entries, odd magnitudes in `[-15, 15]`).
fn encode_wnaf5_u128(mut s: u128, digits: &mut [i8; 129]) -> usize {
    digits.fill(0);
    let mut i = 0usize;
    let mut len = 0usize;
    while s != 0 {
        if (s & 1) == 1 {
            let mut d = (s & 31) as i32;
            if d >= 16 {
                d -= 32;
                s = s.wrapping_add((-d) as u128);
            } else {
                s -= d as u128;
            }
            digits[i] = d as i8;
            len = i + 1;
        }
        s >>= 1;
        i += 1;
    }
    len
}

/// Variable-time 128-bit GLV Straus wNAF (`w = 5`) MSM over `blst_p1_affine`
/// points and 32-byte LE scalars (optimal for small verifier MSMs `1..=16`).
pub(super) fn glv_straus_blst(
    points: &[blst_p1_affine],
    scalar_bytes: &[u8],
) -> blst_p1 {
    let n = points.len();
    debug_assert_eq!(scalar_bytes.len(), n * SCALAR_BYTES);
    if n == 0 {
        return blst_p1::default();
    }

    let beta = glv_beta();
    // Build `[1P, 3P, 5P, 7P, 9P, 11P, 13P, 15P]` for each point with a single
    // batched affine normalization.
    let mut projs = Vec::with_capacity(n * 7);
    for pt in points {
        let mut p1 = blst_p1::default();
        let mut p2 = blst_p1::default();
        // SAFETY: valid point pointers.
        unsafe {
            blst_p1_from_affine(&mut p1, pt);
            blst_p1_double(&mut p2, &p1);
        }
        let mut cur = p1;
        for _ in 0..7 {
            let prev = cur;
            // SAFETY: valid point pointers.
            unsafe { blst_p1_add_or_double(&mut cur, &prev, &p2) };
            projs.push(cur);
        }
    }
    let mut norm = vec![blst_p1_affine::default(); projs.len()];
    if !projs.is_empty() {
        let ptrs: [*const blst_p1; 2] = [projs.as_ptr(), ptr::null()];
        // SAFETY: contiguous slice of `projs.len()` points.
        unsafe {
            blst_p1s_to_affine(norm.as_mut_ptr(), ptrs.as_ptr(), projs.len());
        }
    }

    let mut tables: Vec<[blst_p1_affine; 8]> = Vec::with_capacity(n);
    for (i, &pt) in points.iter().enumerate() {
        let mut row = [blst_p1_affine::default(); 8];
        row[0] = pt;
        row[1..].copy_from_slice(&norm[i * 7..(i + 1) * 7]);
        tables.push(row);
    }

    let mut rem_digits = vec![[0i8; 129]; n];
    let mut q_digits = vec![[0i8; 129]; n];
    let mut rem_lens = vec![0usize; n];
    let mut q_lens = vec![0usize; n];
    let mut max_len = 0usize;

    for i in 0..n {
        let sc: &[u8; SCALAR_BYTES] = scalar_bytes
            [i * SCALAR_BYTES..(i + 1) * SCALAR_BYTES]
            .try_into()
            .unwrap();
        let (q, rem) = glv_decompose_bytes(sc);
        let lr = encode_wnaf5_u128(rem, &mut rem_digits[i]);
        let lq = encode_wnaf5_u128(q, &mut q_digits[i]);
        rem_lens[i] = lr;
        q_lens[i] = lq;
        max_len = max_len.max(lr).max(lq);
    }

    let mut acc = blst_p1::default();
    let mut started = false;
    for bit in (0..max_len).rev() {
        if started {
            let cur = acc;
            // SAFETY: valid Jacobian point.
            unsafe { blst_p1_double(&mut acc, &cur) };
        }
        for i in 0..n {
            if bit < rem_lens[i] {
                let d = rem_digits[i][bit];
                if d != 0 {
                    let mut pt = tables[i][(d.unsigned_abs() as usize) >> 1];
                    if d < 0 {
                        // SAFETY: negates y in place.
                        unsafe { blst_fp_cneg(&mut pt.y, &pt.y, true) };
                    }
                    let cur = acc;
                    // SAFETY: valid Jacobian + affine pointers.
                    unsafe {
                        blst_p1_add_or_double_affine(&mut acc, &cur, &pt)
                    };
                    started = true;
                }
            }
            if bit < q_lens[i] {
                let d = q_digits[i][bit];
                if d != 0 {
                    let base = &tables[i][(d.unsigned_abs() as usize) >> 1];
                    let mut pt = *base;
                    // `d > 0` adds `[|d|](-phi(P)) = (beta*x, -y)`;
                    // `d < 0` adds `[|d|](+phi(P)) = (beta*x, +y)`.
                    // SAFETY: valid `blst_fp` pointers.
                    unsafe {
                        blst_fp_mul(&mut pt.x, &base.x, beta);
                        if d > 0 {
                            blst_fp_cneg(&mut pt.y, &pt.y, true);
                        }
                    }
                    let cur = acc;
                    // SAFETY: valid Jacobian + affine pointers.
                    unsafe {
                        blst_p1_add_or_double_affine(&mut acc, &cur, &pt)
                    };
                    started = true;
                }
            }
        }
    }
    acc
}

/// Precomputed 4-tooth × 2-group × 8-point variable-time Comb table for the
/// 16 static commitments on a `Verifier`.
///
/// Each group of 8 points stores `4` teeth (`[2^(64*t)] P_i` for `t in 0..4`)
/// with all `256` affine subset sums (`2 * 4 * 256 * 96 B = 192 KiB`). Because
/// PLONK verifier scalars are public, direct O(1) indexing replaces
/// constant-time linear table scans, evaluating each 8-point group in `63`
/// doublings and at most `255` mixed affine additions (~70 µs).
#[derive(Clone, Debug)]
pub(crate) struct BlstStaticVerifierTable {
    active_mask: [u8; 2],
    /// Flat `[group][tooth][mask]` lookup table of length `2 * 4 * 256` (empty
    /// when all static points are identity).
    tables: Vec<blst_p1_affine>,
}

impl BlstStaticVerifierTable {
    pub(crate) fn new(points: &[G1Affine; STATIC_POINT_COUNT]) -> Self {
        const {
            assert!(COMB_GROUPS * COMB_GROUP_SIZE == STATIC_POINT_COUNT);
        }
        let mut active_mask = [0u8; COMB_GROUPS];
        let mut blst_pts = [blst_p1_affine::default(); STATIC_POINT_COUNT];

        for (slot_idx, pt) in points.iter().enumerate() {
            if !bool::from(pt.is_identity()) {
                active_mask[slot_idx / COMB_GROUP_SIZE] |=
                    1u8 << (slot_idx % COMB_GROUP_SIZE);
                blst_pts[slot_idx] = g1_affine_to_blst(pt);
            }
        }

        if (active_mask[0] | active_mask[1]) == 0 {
            return Self {
                active_mask,
                tables: Vec::new(),
            };
        }

        let mut tables = vec![
            blst_p1_affine::default();
            COMB_GROUPS * COMB_TEETH * COMB_SUBSETS
        ];
        let mut proj_entries =
            Vec::with_capacity(COMB_GROUPS * COMB_TEETH * (COMB_SUBSETS - 1));
        let mut table_indices =
            Vec::with_capacity(COMB_GROUPS * COMB_TEETH * (COMB_SUBSETS - 1));

        for g in 0..COMB_GROUPS {
            let amask = active_mask[g];
            if amask == 0 {
                continue;
            }
            // Start tooth 0 from the base points `[P_{8g + i}]`.
            let mut tooth_bases = [blst_p1::default(); COMB_GROUP_SIZE];
            for i in 0..COMB_GROUP_SIZE {
                if (amask & (1u8 << i)) != 0 {
                    // SAFETY: valid affine/Jacobian pointers.
                    unsafe {
                        blst_p1_from_affine(
                            &mut tooth_bases[i],
                            &blst_pts[g * COMB_GROUP_SIZE + i],
                        );
                    }
                }
            }

            let mut sums = [blst_p1::default(); COMB_SUBSETS];
            for t in 0..COMB_TEETH {
                if t > 0 {
                    // Advance each active base by `2^64` via 64 doublings.
                    for (i, base) in tooth_bases.iter_mut().enumerate() {
                        if (amask & (1u8 << i)) != 0 {
                            for _ in 0..64 {
                                let cur = *base;
                                // SAFETY: valid Jacobian point.
                                unsafe {
                                    blst_p1_double(base, &cur);
                                }
                            }
                        }
                    }
                }

                sums[0] = blst_p1::default();
                let base_offset = (g * COMB_TEETH + t) * COMB_SUBSETS;
                for mask in 1usize..COMB_SUBSETS {
                    if (mask as u8 & !amask) != 0 {
                        continue;
                    }
                    let lsb = mask & mask.wrapping_neg();
                    let prev = mask ^ lsb;
                    let bit = lsb.trailing_zeros() as usize;
                    if prev == 0 {
                        sums[mask] = tooth_bases[bit];
                    } else {
                        let prev_pt = sums[prev];
                        // SAFETY: valid Jacobian points.
                        unsafe {
                            blst_p1_add_or_double(
                                &mut sums[mask],
                                &prev_pt,
                                &tooth_bases[bit],
                            );
                        }
                    }
                    proj_entries.push(sums[mask]);
                    table_indices.push(base_offset + mask);
                }
            }
        }

        // Batch-normalize all active subset sums in one `Fp` inversion.
        if !proj_entries.is_empty() {
            let mut norm = vec![blst_p1_affine::default(); proj_entries.len()];
            let ptrs: [*const blst_p1; 2] =
                [proj_entries.as_ptr(), ptr::null()];
            // SAFETY: contiguous slice of `proj_entries.len()` points.
            unsafe {
                blst_p1s_to_affine(
                    norm.as_mut_ptr(),
                    ptrs.as_ptr(),
                    proj_entries.len(),
                );
            }
            for (dst_idx, pt) in table_indices.into_iter().zip(norm) {
                tables[dst_idx] = pt;
            }
        }

        Self {
            active_mask,
            tables,
        }
    }

    /// Evaluate all 16 static points serially in a single shared 64-bit
    /// doubling loop (63 doublings total instead of 126).
    pub(super) fn eval_serial(
        &self,
        scalars: &[BlsScalar; STATIC_POINT_COUNT],
    ) -> blst_p1 {
        if self.tables.is_empty() {
            return blst_p1::default();
        }
        let mut limbs = [[[0u64; COMB_GROUP_SIZE]; COMB_TEETH]; COMB_GROUPS];
        let mut active_g = [false; COMB_GROUPS];
        for g in 0..COMB_GROUPS {
            let amask = self.active_mask[g];
            if amask == 0 {
                continue;
            }
            let base_slot = g * COMB_GROUP_SIZE;
            for i in 0..COMB_GROUP_SIZE {
                if (amask & (1u8 << i)) == 0 {
                    continue;
                }
                let sc = &scalars[base_slot + i];
                if *sc == BlsScalar::zero() {
                    continue;
                }
                active_g[g] = true;
                let b = scalar_to_blst_le_bytes(sc);
                limbs[g][0][i] =
                    u64::from_le_bytes(b[0..8].try_into().unwrap());
                limbs[g][1][i] =
                    u64::from_le_bytes(b[8..16].try_into().unwrap());
                limbs[g][2][i] =
                    u64::from_le_bytes(b[16..24].try_into().unwrap());
                limbs[g][3][i] =
                    u64::from_le_bytes(b[24..32].try_into().unwrap());
            }
        }
        if !(active_g[0] || active_g[1]) {
            return blst_p1::default();
        }

        let mut acc = blst_p1::default();
        let mut started = false;
        for shift in (0..64).rev() {
            if started {
                let cur = acc;
                // SAFETY: valid Jacobian point.
                unsafe { blst_p1_double(&mut acc, &cur) };
            }
            for (g, group_limbs) in limbs.iter().enumerate() {
                if !active_g[g] {
                    continue;
                }
                let base_g = g * COMB_TEETH * COMB_SUBSETS;
                for (t, tooth) in group_limbs.iter().enumerate() {
                    let mask = (((tooth[0] >> shift) & 1)
                        | (((tooth[1] >> shift) & 1) << 1)
                        | (((tooth[2] >> shift) & 1) << 2)
                        | (((tooth[3] >> shift) & 1) << 3)
                        | (((tooth[4] >> shift) & 1) << 4)
                        | (((tooth[5] >> shift) & 1) << 5)
                        | (((tooth[6] >> shift) & 1) << 6)
                        | (((tooth[7] >> shift) & 1) << 7))
                        as usize;
                    if mask != 0 {
                        let pt = &self.tables[base_g + t * COMB_SUBSETS + mask];
                        let cur = acc;
                        // SAFETY: valid Jacobian + affine pointers.
                        unsafe {
                            blst_p1_add_or_double_affine(&mut acc, &cur, pt);
                        }
                        started = true;
                    }
                }
            }
        }
        acc
    }
}

/// Verify the PLONK 2-pair BLS12-381 pairing equation
/// `e(left, x_h) * e(right, h) == 1` using precomputed G2 lines and `blst`.
///
/// Both `h` and `x_h` are guaranteed non-identity prime-order G2 points by
/// `OpeningKey::try_new`.
#[cfg(test)]
pub(crate) fn verify_pairing_lines(
    left: &G1Affine,
    right: &G1Affine,
    lines_x_h: &[blst_fp6; 68],
    lines_h: &[blst_fp6; 68],
) -> bool {
    let left_inf = bool::from(left.is_identity());
    let right_inf = bool::from(right.is_identity());
    if left_inf || right_inf {
        return left_inf && right_inf;
    }

    let mut neg_left = g1_affine_to_blst(left);
    // e(left, x_h) * e(right, h) == 1  <=>  e(-left, x_h) == e(right, h)
    // SAFETY: negates `y` coordinate in place.
    unsafe { blst_fp_cneg(&mut neg_left.y, &neg_left.y, true) };
    let right_blst = g1_affine_to_blst(right);

    let mut ml1 = blst_fp12::default();
    let mut ml2 = blst_fp12::default();
    // SAFETY: valid precomputed 68-line tables and non-identity G1 points.
    // Running both precomputed Miller loops serially avoids sub-millisecond
    // Rayon wakeup overhead (~40-100 µs on hybrid CPUs) and scales cleanly when
    // callers verify multiple proofs concurrently.
    unsafe {
        blst_miller_loop_lines(&mut ml1, lines_x_h.as_ptr(), &neg_left);
        blst_miller_loop_lines(&mut ml2, lines_h.as_ptr(), &right_blst);
        blst_fp12_finalverify(&ml1, &ml2)
    }
}

/// Compute the two affine pairing inputs `(left, right)` for single-proof PLONK
/// verification using the precomputed 4-tooth Comb table + 128-bit GLV Straus
/// wNAF (`w = 5`).
pub(crate) fn verifier_msm_single_blst(
    static_table: &BlstStaticVerifierTable,
    static_scalars: &[BlsScalar; STATIC_POINT_COUNT],
    dynamic_points: &[G1Affine; DYNAMIC_POINT_COUNT],
    dynamic_scalars: &[BlsScalar; DYNAMIC_POINT_COUNT],
    w_z: &G1Affine,
    w_zw: &G1Affine,
    u_challenge: &BlsScalar,
) -> (G1Affine, G1Affine) {
    let compute_left = || -> blst_p1 {
        let wz_aff = g1_affine_to_blst(w_z);
        let mut acc = blst_p1::default();
        // SAFETY: valid point pointers.
        unsafe { blst_p1_from_affine(&mut acc, &wz_aff) };

        if !bool::from(w_zw.is_identity()) && *u_challenge != BlsScalar::zero()
        {
            let wzw_aff = g1_affine_to_blst(w_zw);
            let u_bytes = scalar_to_blst_le_bytes(u_challenge);
            let u_wzw = glv_straus_blst(&[wzw_aff], &u_bytes);
            let cur = acc;
            // SAFETY: valid Jacobian points.
            unsafe { blst_p1_add_or_double(&mut acc, &cur, &u_wzw) };
        }
        // Negate to produce `-(w_z + u * w_zw)`.
        // SAFETY: negates Jacobian y coordinate in place.
        unsafe { blst_fp_cneg(&mut acc.y, &acc.y, true) };
        acc
    };

    let mut pts = [blst_p1_affine::default(); DYNAMIC_POINT_COUNT];
    let mut sbytes = [0u8; DYNAMIC_POINT_COUNT * SCALAR_BYTES];
    let mut count = 0usize;
    for (pt, sc) in dynamic_points.iter().zip(dynamic_scalars.iter()) {
        if bool::from(pt.is_identity()) || *sc == BlsScalar::zero() {
            continue;
        }
        pts[count] = g1_affine_to_blst(pt);
        sbytes[count * SCALAR_BYTES..(count + 1) * SCALAR_BYTES]
            .copy_from_slice(&scalar_to_blst_le_bytes(sc));
        count += 1;
    }

    let (left_proj, right_proj) = if rayon::current_num_threads() > 1
        && count >= 4
    {
        // One clean 2-way split (~230 µs each): (left + static Comb) vs
        // dynamic GLV Straus. Avoids waking 5+ sleeping threads on hybrid CPUs.
        let ((left, static_sum), dyn_sum) = rayon::join(
            || {
                let l = compute_left();
                let s = static_table.eval_serial(static_scalars);
                (l, s)
            },
            || glv_straus_blst(&pts[..count], &sbytes[..count * SCALAR_BYTES]),
        );
        let mut right = static_sum;
        // SAFETY: valid Jacobian points.
        unsafe { blst_p1_add_or_double(&mut right, &right, &dyn_sum) };
        (left, right)
    } else {
        let left = compute_left();
        let mut right = static_table.eval_serial(static_scalars);
        let dyn_sum =
            glv_straus_blst(&pts[..count], &sbytes[..count * SCALAR_BYTES]);
        // SAFETY: valid Jacobian points.
        unsafe { blst_p1_add_or_double(&mut right, &right, &dyn_sum) };
        (left, right)
    };

    let [left_aff, right_aff] =
        blst_p1s_to_g1_affines(&[left_proj, right_proj]);
    (left_aff, right_aff)
}

/// Evaluate a variable-base `blst` MSM returning Jacobian `blst_p1`, using
/// 128-bit GLV Straus for small sizes and 129-bit GLV-decomposed tiled
/// Pippenger for larger batch-verification MSMs.
pub(super) fn msm_variable_base_proj(
    points: &[G1Affine],
    scalars: &[BlsScalar],
) -> blst_p1 {
    assert_eq!(points.len(), scalars.len());
    if points.is_empty() {
        return blst_p1::default();
    }

    let mut blst_pts = Vec::with_capacity(points.len());
    let mut sbytes = Vec::with_capacity(scalars.len() * SCALAR_BYTES);
    for (pt, sc) in points.iter().zip(scalars.iter()) {
        if bool::from(pt.is_identity()) || *sc == BlsScalar::zero() {
            continue;
        }
        blst_pts.push(g1_affine_to_blst(pt));
        sbytes.extend_from_slice(&scalar_to_blst_le_bytes(sc));
    }

    let n = blst_pts.len();
    if n == 0 {
        return blst_p1::default();
    }
    if n < 24 {
        return glv_straus_blst(&blst_pts, &sbytes);
    }

    // GLV-decompose each `(P, s)` into `(P, rem)` and `(-phi(P), q)` with
    // `0 <= rem, q < x^2 < 2^128`. Using `GLV_NBITS = 129` (`17` bytes, top
    // byte = 0) guarantees zero terminal Booth carry for every Pippenger window
    // width while halving the number of windows!
    const GLV_NBITS: usize = 129;
    const GLV_NBYTES: usize = 17;
    let beta = glv_beta();
    let mut glv_pts = Vec::with_capacity(n * 2);
    let mut glv_scs = Vec::with_capacity(n * 2 * GLV_NBYTES);
    for i in 0..n {
        let sc: &[u8; SCALAR_BYTES] = sbytes
            [i * SCALAR_BYTES..(i + 1) * SCALAR_BYTES]
            .try_into()
            .unwrap();
        let (q, rem) = glv_decompose_bytes(sc);
        if rem != 0 {
            glv_pts.push(blst_pts[i]);
            glv_scs.extend_from_slice(&rem.to_le_bytes());
            glv_scs.push(0u8);
        }
        if q != 0 {
            glv_pts.push(glv_neg_phi(&blst_pts[i], beta));
            glv_scs.extend_from_slice(&q.to_le_bytes());
            glv_scs.push(0u8);
        }
    }

    let m = glv_pts.len();
    if m == 0 {
        return blst_p1::default();
    }
    if rayon::current_num_threads() <= 1 {
        return blst_msm_serial_raw_nbits(&glv_pts, &glv_scs, GLV_NBITS);
    }

    let window = pippenger_window(m);
    let ny = GLV_NBITS.div_ceil(window);
    // SAFETY: pure size query.
    let tile_unit_u64 =
        unsafe { blst_p1s_mult_pippenger_scratch_sizeof(0) }.div_ceil(8);

    let tile_results: Vec<blst_p1> = (0..ny)
        .into_par_iter()
        .map(|w_idx| {
            blst_tile_pippenger_raw(
                &glv_pts,
                &glv_scs,
                GLV_NBITS,
                w_idx * window,
                window,
                tile_unit_u64,
            )
        })
        .collect();

    reduce_window_tiles(&tile_results, window, ny, 1)
}

/// Compute the single-proof verifier MSM AND pairing check in one pipelined
/// pass: `neg_left = w_z + u * w_zw` and its Miller loop `ml1(neg_left, x_h)`
/// run concurrently with the `right` MSM (static Comb + dynamic GLV Straus),
/// hiding the entire first Miller loop behind the MSM!
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_single_pipelined_blst(
    static_table: &BlstStaticVerifierTable,
    static_scalars: &[BlsScalar; STATIC_POINT_COUNT],
    dynamic_points: &[G1Affine; DYNAMIC_POINT_COUNT],
    dynamic_scalars: &[BlsScalar; DYNAMIC_POINT_COUNT],
    w_z: &G1Affine,
    w_zw: &G1Affine,
    u_challenge: &BlsScalar,
    lines_x_h: &[blst_fp6; 68],
    lines_h: &[blst_fp6; 68],
) -> bool {
    let compute_neg_left_and_ml1 = || -> (bool, blst_fp12) {
        let wz_aff = g1_affine_to_blst(w_z);
        let mut neg_left_proj = blst_p1::default();
        // SAFETY: valid point pointers.
        unsafe { blst_p1_from_affine(&mut neg_left_proj, &wz_aff) };
        if !bool::from(w_zw.is_identity()) && *u_challenge != BlsScalar::zero()
        {
            let wzw_aff = g1_affine_to_blst(w_zw);
            let u_bytes = scalar_to_blst_le_bytes(u_challenge);
            let u_wzw = glv_straus_blst(&[wzw_aff], &u_bytes);
            let cur = neg_left_proj;
            // SAFETY: valid Jacobian points.
            unsafe {
                blst_p1_add_or_double(&mut neg_left_proj, &cur, &u_wzw);
            }
        }
        let mut neg_left_aff = blst_p1_affine::default();
        let mut ml1 = blst_fp12::default();
        // SAFETY: valid Jacobian/affine and precomputed 68-line pointers.
        let left_inf = unsafe {
            blst_p1_to_affine(&mut neg_left_aff, &neg_left_proj);
            let is_inf = blst_p1_affine_is_inf(&neg_left_aff);
            if !is_inf {
                blst_miller_loop_lines(
                    &mut ml1,
                    lines_x_h.as_ptr(),
                    &neg_left_aff,
                );
            }
            is_inf
        };
        (left_inf, ml1)
    };

    let mut pts = [blst_p1_affine::default(); DYNAMIC_POINT_COUNT];
    let mut sbytes = [0u8; DYNAMIC_POINT_COUNT * SCALAR_BYTES];
    let mut count = 0usize;
    for (pt, sc) in dynamic_points.iter().zip(dynamic_scalars.iter()) {
        if bool::from(pt.is_identity()) || *sc == BlsScalar::zero() {
            continue;
        }
        pts[count] = g1_affine_to_blst(pt);
        sbytes[count * SCALAR_BYTES..(count + 1) * SCALAR_BYTES]
            .copy_from_slice(&scalar_to_blst_le_bytes(sc));
        count += 1;
    }

    let ((left_inf, ml1), right_proj) = if rayon::current_num_threads() > 1 {
        // Balanced 2-way split (~330 µs each):
        //   Branch A: neg_left + ml1 (~165 µs) + first ~60% of dyn (~165 µs)
        //   Branch B: static Comb (~218 µs) + remaining ~40% of dyn (~120 µs)
        let split = (count * 3) / 5;
        let (((l_inf, ml1), dyn_a), (static_sum, dyn_b)) = rayon::join(
            || {
                let l = compute_neg_left_and_ml1();
                let da = glv_straus_blst(
                    &pts[..split],
                    &sbytes[..split * SCALAR_BYTES],
                );
                (l, da)
            },
            || {
                let s = static_table.eval_serial(static_scalars);
                let db = glv_straus_blst(
                    &pts[split..count],
                    &sbytes[split * SCALAR_BYTES..count * SCALAR_BYTES],
                );
                (s, db)
            },
        );
        let mut right = static_sum;
        // SAFETY: valid Jacobian points.
        unsafe {
            blst_p1_add_or_double(&mut right, &right, &dyn_a);
            blst_p1_add_or_double(&mut right, &right, &dyn_b);
        }
        ((l_inf, ml1), right)
    } else {
        let l = compute_neg_left_and_ml1();
        let mut right = static_table.eval_serial(static_scalars);
        let dyn_sum =
            glv_straus_blst(&pts[..count], &sbytes[..count * SCALAR_BYTES]);
        // SAFETY: valid Jacobian points.
        unsafe { blst_p1_add_or_double(&mut right, &right, &dyn_sum) };
        (l, right)
    };

    let mut right_aff = blst_p1_affine::default();
    // SAFETY: valid Jacobian/affine pointers.
    let right_inf = unsafe {
        blst_p1_to_affine(&mut right_aff, &right_proj);
        blst_p1_affine_is_inf(&right_aff)
    };
    if left_inf || right_inf {
        return left_inf && right_inf;
    }

    let mut ml2 = blst_fp12::default();
    // SAFETY: valid precomputed 68-line table, non-identity `right_aff`, and
    // valid `ml1`/`ml2` Miller loop outputs.
    unsafe {
        blst_miller_loop_lines(&mut ml2, lines_h.as_ptr(), &right_aff);
        blst_fp12_finalverify(&ml1, &ml2)
    }
}

/// Compute the batched verifier pairing check in one pipelined pass: the
/// smaller `left` MSM (`2*B` points) immediately computes `ml1(neg_left, x_h)`
/// concurrently while the larger `right` MSM (`11*B` points) finishes!
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_batch_pipelined_blst(
    left_pts: &[G1Affine],
    left_scs: &[BlsScalar],
    right_pts: &[G1Affine],
    right_scs: &[BlsScalar],
    static_part: Option<(
        &BlstStaticVerifierTable,
        &[BlsScalar; STATIC_POINT_COUNT],
    )>,
    lines_x_h: &[blst_fp6; 68],
    lines_h: &[blst_fp6; 68],
) -> bool {
    let compute_neg_left_and_ml1 = || -> (bool, blst_fp12) {
        let neg_left_proj = msm_variable_base_proj(left_pts, left_scs);
        let mut neg_left_aff = blst_p1_affine::default();
        let mut ml1 = blst_fp12::default();
        // SAFETY: valid Jacobian/affine and precomputed 68-line pointers.
        let left_inf = unsafe {
            blst_p1_to_affine(&mut neg_left_aff, &neg_left_proj);
            let is_inf = blst_p1_affine_is_inf(&neg_left_aff);
            if !is_inf {
                blst_miller_loop_lines(
                    &mut ml1,
                    lines_x_h.as_ptr(),
                    &neg_left_aff,
                );
            }
            is_inf
        };
        (left_inf, ml1)
    };

    let ((left_inf, ml1), right_proj) = if rayon::current_num_threads() > 1 {
        let ((l, s_opt), mut r) = rayon::join(
            || {
                rayon::join(compute_neg_left_and_ml1, || {
                    static_part.map(|(tbl, scs)| tbl.eval_serial(scs))
                })
            },
            || msm_variable_base_proj(right_pts, right_scs),
        );
        if let Some(s) = s_opt {
            // SAFETY: valid Jacobian points.
            unsafe { blst_p1_add_or_double(&mut r, &r, &s) };
        }
        (l, r)
    } else {
        let l = compute_neg_left_and_ml1();
        let mut r = msm_variable_base_proj(right_pts, right_scs);
        if let Some((tbl, scs)) = static_part {
            let s = tbl.eval_serial(scs);
            // SAFETY: valid Jacobian points.
            unsafe { blst_p1_add_or_double(&mut r, &r, &s) };
        }
        (l, r)
    };

    let mut right_aff = blst_p1_affine::default();
    // SAFETY: valid Jacobian/affine pointers.
    let right_inf = unsafe {
        blst_p1_to_affine(&mut right_aff, &right_proj);
        blst_p1_affine_is_inf(&right_aff)
    };
    if left_inf || right_inf {
        return left_inf && right_inf;
    }

    let mut ml2 = blst_fp12::default();
    // SAFETY: valid precomputed 68-line table, non-identity `right_aff`, and
    // valid `ml1`/`ml2` Miller loop outputs.
    unsafe {
        blst_miller_loop_lines(&mut ml2, lines_h.as_ptr(), &right_aff);
        blst_fp12_finalverify(&ml1, &ml2)
    }
}
