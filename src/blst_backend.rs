// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! High-performance `blst` (x86_64 ADX/BMI2 & aarch64 assembly) backend for
//! BLS12-381 multi-scalar multiplication, pairings, and point decoding.
//!
//! All `unsafe` FFI calls to `blst` are isolated inside this module. Every
//! conversion preserves bit-exact compatibility with `dusk-bls12_381` and
//! Dusk PLONK V3 proofs.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::ptr;

use blst::{
    BLST_ERROR, blst_fp, blst_fp_cneg, blst_fp_eucl_inverse, blst_fp_mul,
    blst_fp_sqr, blst_fp_sub, blst_fp6, blst_fp12, blst_fp12_finalverify,
    blst_fr, blst_fr_eucl_inverse, blst_miller_loop_lines, blst_p1,
    blst_p1_add_or_double, blst_p1_add_or_double_affine, blst_p1_affine,
    blst_p1_affine_generator, blst_p1_affine_in_g1, blst_p1_affine_is_inf,
    blst_p1_double, blst_p1_from_affine, blst_p1_mult, blst_p1_to_affine,
    blst_p1_uncompress, blst_p1s_mult_pippenger,
    blst_p1s_mult_pippenger_scratch_sizeof, blst_p1s_tile_pippenger,
    blst_p1s_to_affine, blst_p2_affine, blst_p2_uncompress,
    blst_precompute_lines, blst_scalar, blst_scalar_from_fr,
};
use dusk_bls12_381::{BlsScalar, G1Affine, G2Affine};
use rayon::prelude::*;

use crate::fft::EvaluationDomain;
use crate::proof_system::verifier_msm::{
    DYNAMIC_POINT_COUNT, STATIC_POINT_COUNT,
};

// BLS12-381 scalars satisfy `s < r < 2^255`, so bit 255 is always 0.
// Passing 256 bits guarantees zero terminal Booth carry for every Pippenger
// window width (including divisors of 255 and 256).
const SCALAR_NBITS: usize = 256;
const SCALAR_BYTES: usize = 32;

std::thread_local! {
    static PIPPENGER_SCRATCH: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
    static AFFINE_SCRATCH: RefCell<AffinePippengerScratch> =
        RefCell::new(AffinePippengerScratch::default());
}

/// Fast binary-Euclidean `Fr` inversion via `blst` ADX/BMI2 assembly.
///
/// Caller guarantees `scalar != 0`.
#[inline]
pub(crate) fn fr_invert_nonzero(scalar: &BlsScalar) -> BlsScalar {
    let input = blst_fr {
        l: *scalar.internal_repr(),
    };
    let mut inv_fr = blst_fr::default();
    let mut out_sc = blst_scalar::default();
    // SAFETY: valid `blst_fr` and `blst_scalar` pointers.
    unsafe {
        blst_fr_eucl_inverse(&mut inv_fr, &input);
        blst_scalar_from_fr(&mut out_sc, &inv_fr);
    }
    let l0 = u64::from_le_bytes(out_sc.b[0..8].try_into().unwrap());
    let l1 = u64::from_le_bytes(out_sc.b[8..16].try_into().unwrap());
    let l2 = u64::from_le_bytes(out_sc.b[16..24].try_into().unwrap());
    let l3 = u64::from_le_bytes(out_sc.b[24..32].try_into().unwrap());
    BlsScalar::from_raw([l0, l1, l2, l3])
}

/// BLS12-381 `x^2 = (-0xd201000000010000)^2`, the exact 128-bit GLV eigenvalue
/// magnitude (`r = x^4 - x^2 + 1`).
const GLV_X_SQ: u128 = 0xac45_a401_0001_a402_0000_0001_0000_0000;
const GLV_X_SQ_LO: u64 = GLV_X_SQ as u64;
const GLV_X_SQ_HI: u64 = (GLV_X_SQ >> 64) as u64;

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
fn glv_straus_blst(points: &[blst_p1_affine], scalar_bytes: &[u8]) -> blst_p1 {
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

/// Reusable per-thread workspace for Batched-Affine Pippenger.
struct AffinePippengerScratch {
    buckets: Vec<blst_p1_affine>,
    occupied: Vec<u64>,
    in_batch: Vec<u64>,
    deferred: Vec<(u16, u32)>,
    next_deferred: Vec<(u16, u32)>,
    dxs: Vec<blst_fp>,
    prods: Vec<blst_fp>,
    special: Vec<bool>,
}

impl Default for AffinePippengerScratch {
    fn default() -> Self {
        Self {
            buckets: Vec::new(),
            occupied: Vec::new(),
            in_batch: Vec::new(),
            deferred: Vec::new(),
            next_deferred: Vec::new(),
            dxs: vec![blst_fp::default(); AFFINE_BATCH_SIZE],
            prods: vec![blst_fp::default(); AFFINE_BATCH_SIZE],
            special: vec![false; AFFINE_BATCH_SIZE],
        }
    }
}

const AFFINE_BATCH_SIZE: usize = 256;

/// `1` in BLS12-381 `Fp` Montgomery form (`R = 2^384 mod p`), used as a safe
/// non-zero placeholder during Montgomery batch inversion when `x1 == x2`.
const FP_ONE_MONT: blst_fp = blst_fp {
    l: [
        0x7609_0000_0002_fffd,
        0xebf4_000b_c40c_0002,
        0x5f48_9857_53c7_58ba,
        0x77ce_5853_7052_5745,
        0x5c07_1a97_a256_ec6d,
        0x15f6_5ec3_fa80_e493,
    ],
};

#[inline(always)]
fn fp_is_zero(x: &blst_fp) -> bool {
    (x.l[0] | x.l[1] | x.l[2] | x.l[3] | x.l[4] | x.l[5]) == 0
}

#[inline(always)]
fn fp_eq(a: &blst_fp, b: &blst_fp) -> bool {
    a.l == b.l
}

/// Extract signed Booth digit `(mag, neg)` for window `[bit0 .. bit0 + w)`
/// from a 32-byte LE scalar (`scalar < 2^255`, `1 <= w <= 15`).
///
/// Returns `mag in 0..=2^(w-1)` and `neg: bool`. `mag == 0` means skip.
#[inline(always)]
fn extract_booth_digit(scalar: &[u8], bit0: usize, w: usize) -> (u32, bool) {
    let raw = if bit0 == 0 {
        let word = u32::from_le_bytes([scalar[0], scalar[1], scalar[2], 0]);
        (word << 1) & ((1u32 << (w + 1)) - 1)
    } else {
        let start = bit0 - 1;
        let byte_idx = start >> 3;
        let bit_off = start & 7;
        let b0 = *scalar.get(byte_idx).unwrap_or(&0) as u32;
        let b1 = *scalar.get(byte_idx + 1).unwrap_or(&0) as u32;
        let b2 = *scalar.get(byte_idx + 2).unwrap_or(&0) as u32;
        let b3 = *scalar.get(byte_idx + 3).unwrap_or(&0) as u32;
        let word = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        (word >> bit_off) & ((1u32 << (w + 1)) - 1)
    };
    let neg = ((raw >> w) & 1) != 0;
    let mag = if neg {
        ((1u32 << (w + 1)) - raw) >> 1
    } else {
        (raw + 1) >> 1
    };
    (mag, neg)
}

/// Flush `count` collision-free `(bucket_idx, point_idx_and_sign)` affine
/// additions using ONE `blst_fp_eucl_inverse` via Montgomery's trick.
#[inline]
#[allow(clippy::too_many_arguments)]
fn flush_affine_batch(
    buckets: &mut [blst_p1_affine],
    occupied: &mut [u64],
    in_batch: &mut [u64],
    dxs: &mut [blst_fp],
    prods: &mut [blst_fp],
    special: &mut [bool],
    points: &[blst_p1_affine],
    batch_bidx: &[u16; AFFINE_BATCH_SIZE],
    batch_psign: &[u32; AFFINE_BATCH_SIZE],
    count: usize,
) {
    if count == 0 {
        return;
    }

    // Forward pass: compute `dx[k] = x2 - x1` and running prefix products.
    for k in 0..count {
        let bidx = batch_bidx[k] as usize;
        let pidx = (batch_psign[k] & 0x7FFF_FFFF) as usize;
        let mut dx = blst_fp::default();
        // SAFETY: valid `blst_fp` pointers.
        unsafe { blst_fp_sub(&mut dx, &points[pidx].x, &buckets[bidx].x) };
        if fp_is_zero(&dx) {
            special[k] = true;
            dx = FP_ONE_MONT;
        } else {
            special[k] = false;
        }
        dxs[k] = dx;
        if k == 0 {
            prods[0] = dx;
        } else {
            // SAFETY: valid `blst_fp` pointers.
            unsafe {
                let prev = prods[k - 1];
                blst_fp_mul(&mut prods[k], &prev, &dx);
            }
        }
    }

    // One single binary-Euclidean inversion for the entire batch!
    let mut acc_inv = blst_fp::default();
    // SAFETY: `prods[count - 1]` is guaranteed nonzero (zeros replaced by 1).
    unsafe { blst_fp_eucl_inverse(&mut acc_inv, &prods[count - 1]) };

    // Backward pass: recover each `dx[k]^{-1}` and complete affine addition.
    for k in (0..count).rev() {
        let bidx = batch_bidx[k] as usize;
        in_batch[bidx >> 6] = 0;

        let mut inv_dx = blst_fp::default();
        if k > 0 {
            // SAFETY: valid `blst_fp` pointers.
            unsafe {
                blst_fp_mul(&mut inv_dx, &acc_inv, &prods[k - 1]);
                let next_inv = acc_inv;
                blst_fp_mul(&mut acc_inv, &next_inv, &dxs[k]);
            }
        } else {
            inv_dx = acc_inv;
        }

        let psign = batch_psign[k];
        let pidx = (psign & 0x7FFF_FFFF) as usize;
        let neg = (psign >> 31) != 0;
        let mut y2 = points[pidx].y;
        if neg {
            // SAFETY: negates y in place.
            unsafe { blst_fp_cneg(&mut y2, &y2, true) };
        }

        if special[k] {
            // x1 == x2: either P2 == -P1 (cancel to empty bucket) or P2 == P1
            // (point doubling).
            if !fp_eq(&buckets[bidx].y, &y2) || fp_is_zero(&y2) {
                occupied[bidx >> 6] &= !(1u64 << (bidx & 63));
            } else {
                let mut proj = blst_p1::default();
                let mut dbl = blst_p1::default();
                // SAFETY: valid point pointers.
                unsafe {
                    blst_p1_from_affine(&mut proj, &buckets[bidx]);
                    blst_p1_double(&mut dbl, &proj);
                    blst_p1_to_affine(&mut buckets[bidx], &dbl);
                }
            }
            continue;
        }

        let x1 = buckets[bidx].x;
        let y1 = buckets[bidx].y;
        let x2 = points[pidx].x;

        let mut dy = blst_fp::default();
        let mut lambda = blst_fp::default();
        let mut x3 = blst_fp::default();
        let mut y3 = blst_fp::default();
        let mut tmp = blst_fp::default();
        // SAFETY: valid `blst_fp` pointers.
        unsafe {
            blst_fp_sub(&mut dy, &y2, &y1);
            blst_fp_mul(&mut lambda, &dy, &inv_dx);
            blst_fp_sqr(&mut x3, &lambda);
            blst_fp_sub(&mut tmp, &x3, &x1);
            blst_fp_sub(&mut x3, &tmp, &x2);
            blst_fp_sub(&mut tmp, &x1, &x3);
            blst_fp_mul(&mut y3, &lambda, &tmp);
            blst_fp_sub(&mut tmp, &y3, &y1);
        }
        buckets[bidx] = blst_p1_affine { x: x3, y: tmp };
    }
}

/// Evaluate one Pippenger window tile using Batched-Affine buckets over
/// `blst_fp` ADX/BMI2 assembly (`~5.4M + 1S` per addition, 96 B/bucket).
fn batched_affine_tile_pippenger(
    points: &[blst_p1_affine],
    scalar_bytes: &[u8],
    bit0: usize,
    window: usize,
) -> blst_p1 {
    let n = points.len();
    debug_assert_eq!(scalar_bytes.len(), n * SCALAR_BYTES);
    debug_assert!((2..=15).contains(&window));
    if n == 0 {
        return blst_p1::default();
    }

    let num_buckets = 1usize << (window - 1);
    let words = num_buckets.div_ceil(64);

    AFFINE_SCRATCH.with(|cell| {
        let mut ws = cell.borrow_mut();
        if ws.buckets.len() < num_buckets {
            ws.buckets.resize(num_buckets, blst_p1_affine::default());
        }
        if ws.occupied.len() < words {
            ws.occupied.resize(words, 0);
        }
        if ws.in_batch.len() < words {
            ws.in_batch.resize(words, 0);
        }
        ws.occupied[..words].fill(0);
        ws.in_batch[..words].fill(0);
        ws.deferred.clear();
        ws.next_deferred.clear();

        let AffinePippengerScratch {
            buckets,
            occupied,
            in_batch,
            deferred,
            next_deferred,
            dxs,
            prods,
            special,
        } = &mut *ws;

        let mut batch_bidx = [0u16; AFFINE_BATCH_SIZE];
        let mut batch_psign = [0u32; AFFINE_BATCH_SIZE];
        let mut batch_len = 0usize;
        let batch_cap = (num_buckets / 4).clamp(64, AFFINE_BATCH_SIZE);

        for i in 0..n {
            if fp_is_zero(&points[i].x) && fp_is_zero(&points[i].y) {
                continue;
            }
            let sc = &scalar_bytes[i * SCALAR_BYTES..(i + 1) * SCALAR_BYTES];
            let (mag, neg) = extract_booth_digit(sc, bit0, window);
            if mag == 0 {
                continue;
            }
            let bidx = (mag - 1) as usize;
            let word = bidx >> 6;
            let mask = 1u64 << (bidx & 63);

            if (occupied[word] & mask) == 0 {
                let mut pt = points[i];
                if neg {
                    // SAFETY: negates y in place.
                    unsafe { blst_fp_cneg(&mut pt.y, &pt.y, true) };
                }
                buckets[bidx] = pt;
                occupied[word] |= mask;
            } else if (in_batch[word] & mask) == 0 {
                in_batch[word] |= mask;
                batch_bidx[batch_len] = bidx as u16;
                batch_psign[batch_len] = (i as u32) | ((neg as u32) << 31);
                batch_len += 1;
                if batch_len == batch_cap {
                    flush_affine_batch(
                        buckets,
                        occupied,
                        in_batch,
                        dxs,
                        prods,
                        special,
                        points,
                        &batch_bidx,
                        &batch_psign,
                        batch_len,
                    );
                    batch_len = 0;
                }
            } else {
                deferred.push((bidx as u16, (i as u32) | ((neg as u32) << 31)));
            }
        }

        if batch_len > 0 {
            flush_affine_batch(
                buckets,
                occupied,
                in_batch,
                dxs,
                prods,
                special,
                points,
                &batch_bidx,
                &batch_psign,
                batch_len,
            );
            batch_len = 0;
        }

        // Drain deferred collisions in conflict-free mini-batches. If the
        // remaining deferred queue is skewed onto fewer than 32 distinct
        // buckets (e.g., repeated witness scalars), spill the remaining items
        // directly into Jacobian overflow accumulators (zero inversions!).
        let mut overflow: Vec<(u16, blst_p1)> = Vec::new();
        while !deferred.is_empty() {
            next_deferred.clear();
            let mut pass_flushed = 0usize;
            for &(bidx_u16, psign) in deferred.iter() {
                let bidx = bidx_u16 as usize;
                let word = bidx >> 6;
                let mask = 1u64 << (bidx & 63);
                if (occupied[word] & mask) == 0 {
                    let pidx = (psign & 0x7FFF_FFFF) as usize;
                    let neg = (psign >> 31) != 0;
                    let mut pt = points[pidx];
                    if neg {
                        // SAFETY: negates y in place.
                        unsafe { blst_fp_cneg(&mut pt.y, &pt.y, true) };
                    }
                    buckets[bidx] = pt;
                    occupied[word] |= mask;
                } else if (in_batch[word] & mask) == 0 {
                    in_batch[word] |= mask;
                    batch_bidx[batch_len] = bidx_u16;
                    batch_psign[batch_len] = psign;
                    batch_len += 1;
                    if batch_len == batch_cap {
                        pass_flushed += batch_len;
                        flush_affine_batch(
                            buckets,
                            occupied,
                            in_batch,
                            dxs,
                            prods,
                            special,
                            points,
                            &batch_bidx,
                            &batch_psign,
                            batch_len,
                        );
                        batch_len = 0;
                    }
                } else {
                    next_deferred.push((bidx_u16, psign));
                }
            }
            if batch_len > 0 {
                if pass_flushed == 0 && batch_len < 32 {
                    // Clear `in_batch` bits and accumulate the skewed tail
                    // (`batch[0..batch_len]` union `next_deferred`) via
                    // Jacobian mixed additions without paying serial Fp
                    // inversions.
                    let mut add_overflow = |bidx_u16: u16, psign: u32| {
                        let pidx = (psign & 0x7FFF_FFFF) as usize;
                        let neg = (psign >> 31) != 0;
                        let mut pt = points[pidx];
                        if neg {
                            // SAFETY: negates y in place.
                            unsafe { blst_fp_cneg(&mut pt.y, &pt.y, true) };
                        }
                        if let Some((_, acc)) =
                            overflow.iter_mut().find(|(b, _)| *b == bidx_u16)
                        {
                            let cur = *acc;
                            // SAFETY: valid Jacobian + affine pointers.
                            unsafe {
                                blst_p1_add_or_double_affine(acc, &cur, &pt);
                            }
                        } else {
                            let mut acc = blst_p1::default();
                            // SAFETY: valid affine/Jacobian pointers.
                            unsafe { blst_p1_from_affine(&mut acc, &pt) };
                            overflow.push((bidx_u16, acc));
                        }
                    };
                    for k in 0..batch_len {
                        let b = batch_bidx[k];
                        in_batch[(b as usize) >> 6] = 0;
                        add_overflow(b, batch_psign[k]);
                    }
                    for &(b, psign) in next_deferred.iter() {
                        add_overflow(b, psign);
                    }
                    deferred.clear();
                    break;
                }
                flush_affine_batch(
                    buckets,
                    occupied,
                    in_batch,
                    dxs,
                    prods,
                    special,
                    points,
                    &batch_bidx,
                    &batch_psign,
                    batch_len,
                );
                batch_len = 0;
            }
            core::mem::swap(deferred, next_deferred);
        }

        // Integrate affine buckets (plus any skewed-tail Jacobian overflows)
        // into the Jacobian window sum.
        overflow.sort_unstable_by_key(|a| core::cmp::Reverse(a.0));
        let mut ov_idx = 0usize;
        let mut running = blst_p1::default();
        let mut sum = blst_p1::default();
        let mut started = false;
        for b in (0..num_buckets).rev() {
            if ((occupied[b >> 6] >> (b & 63)) & 1) != 0 {
                // SAFETY: valid Jacobian + affine pointers.
                unsafe {
                    blst_p1_add_or_double_affine(
                        &mut running,
                        &running,
                        &buckets[b],
                    );
                }
                started = true;
            }
            if ov_idx < overflow.len() && overflow[ov_idx].0 as usize == b {
                let cur = running;
                // SAFETY: valid Jacobian pointers.
                unsafe {
                    blst_p1_add_or_double(
                        &mut running,
                        &cur,
                        &overflow[ov_idx].1,
                    );
                }
                ov_idx += 1;
                started = true;
            }
            if started {
                // SAFETY: valid Jacobian pointers.
                unsafe { blst_p1_add_or_double(&mut sum, &sum, &running) };
            }
        }
        sum
    })
}

/// Empirically optimal L2-resident Batched-Affine Pippenger window width on
/// BLS12-381 (`96 B/bucket`): `2^12 -> 10, 2^13 -> 11, 2^14..2^15 -> 12,
/// >=2^16 -> 13`.
#[inline]
fn choose_affine_window(n: usize) -> usize {
    let lg = (usize::BITS - 1 - n.max(1).leading_zeros()) as usize;
    match lg {
        0..=5 => 5,
        6..=10 => lg - 1,
        11 => 9,
        12 => 10,
        13 => 11,
        14..=15 => 12,
        _ => 13,
    }
}

/// Evaluate a batch of `K` MSMs using Batched-Affine Pippenger tiled across
/// Rayon with configurable window override (`None` selects the optimal affine
/// window for the tile size).
pub(crate) fn msm_g1_batch_affine<const K: usize>(
    bases: &[blst_p1_affine],
    polys: [&[BlsScalar]; K],
    window_override: Option<usize>,
) -> [G1Affine; K] {
    let mut lengths = [0usize; K];
    for (i, poly) in polys.iter().enumerate() {
        let mut len = poly.len();
        while len > 0 && poly[len - 1] == BlsScalar::zero() {
            len -= 1;
        }
        assert!(
            len <= bases.len(),
            "MSM polynomial length exceeds CommitKey bases"
        );
        lengths[i] = len;
    }

    let max_len = lengths.iter().copied().max().unwrap_or(0);
    if max_len == 0 {
        return [G1Affine::identity(); K];
    }

    let num_threads = rayon::current_num_threads();
    if num_threads <= 1 || max_len < 64 {
        let mut projs = [blst_p1::default(); K];
        for i in 0..K {
            let len = lengths[i];
            if len > 0 {
                let sbytes = scalars_to_le_bytes_vec(&polys[i][..len]);
                projs[i] = blst_msm_serial_raw(&bases[..len], &sbytes);
            }
        }
        return blst_p1s_to_g1_affines(&projs);
    }

    let mut scalar_bufs: Vec<Vec<u8>> = lengths
        .iter()
        .map(|&len| vec![0u8; len * SCALAR_BYTES])
        .collect();
    {
        let total_scalars: usize = lengths.iter().sum();
        if total_scalars >= 1024 {
            scalar_bufs
                .par_iter_mut()
                .zip(polys.par_iter())
                .zip(lengths.par_iter())
                .for_each(|((buf, poly), &len)| {
                    if len == 0 {
                        return;
                    }
                    buf.par_chunks_exact_mut(SCALAR_BYTES)
                        .zip(poly[..len].par_iter())
                        .for_each(|(dst, sc)| {
                            dst.copy_from_slice(&scalar_to_blst_le_bytes(sc));
                        });
                });
        } else {
            for i in 0..K {
                let len = lengths[i];
                for (dst, sc) in scalar_bufs[i]
                    .as_chunks_mut::<SCALAR_BYTES>()
                    .0
                    .iter_mut()
                    .zip(polys[i][..len].iter())
                {
                    *dst = scalar_to_blst_le_bytes(sc);
                }
            }
        }
    }

    struct AffineTile {
        poly_idx: usize,
        x: usize,
        dx: usize,
        w_idx: usize,
        window: usize,
    }

    let mut poly_windows = [0usize; K];
    let mut poly_ny = [0usize; K];
    let mut poly_nx = [0usize; K];
    let mut tiles = Vec::new();
    let active_polys = lengths.iter().filter(|&&l| l >= 64).count().max(1);

    for (poly_idx, &len) in lengths.iter().enumerate() {
        if len < 64 {
            continue;
        }
        let choose_w = |n: usize| -> usize {
            window_override
                .map(|w| w.clamp(2, 15))
                .unwrap_or_else(|| choose_affine_window(n))
        };
        let initial_w = choose_w(len);
        let initial_ny = SCALAR_NBITS.div_ceil(initial_w);
        let desired_total_tiles = if len >= 4096 {
            num_threads.saturating_mul(2)
        } else {
            num_threads
        };
        let target_tiles_per_poly =
            desired_total_tiles.div_ceil(active_polys).max(initial_ny);
        let nx = if len >= 2048 {
            target_tiles_per_poly
                .div_ceil(initial_ny)
                .min((len / 2048).max(1))
                .max(1)
        } else {
            1
        };
        let window = choose_w(len.div_ceil(nx));
        let ny = SCALAR_NBITS.div_ceil(window);
        poly_windows[poly_idx] = window;
        poly_ny[poly_idx] = ny;
        poly_nx[poly_idx] = nx;

        let base_dx = len / nx;
        let rem = len % nx;
        for w_idx in 0..ny {
            let mut x = 0usize;
            for xi in 0..nx {
                let dx = base_dx + usize::from(xi < rem);
                tiles.push(AffineTile {
                    poly_idx,
                    x,
                    dx,
                    w_idx,
                    window,
                });
                x += dx;
            }
        }
    }

    // SAFETY: pure size query for one XYZZ tile bucket unit (used for narrow
    // top partial windows where `eff_bits < 10` would cause heavy affine
    // bucket collisions).
    let tile_unit_u64 =
        unsafe { blst_p1s_mult_pippenger_scratch_sizeof(0) }.div_ceil(8);

    let tile_results: Vec<blst_p1> = tiles
        .par_iter()
        .map(|tile| {
            let bit0 = tile.w_idx * tile.window;
            let eff_bits = 255usize.saturating_sub(bit0).min(tile.window);
            if eff_bits < 10 {
                return PIPPENGER_SCRATCH.with(|cell| {
                    let mut scratch = cell.borrow_mut();
                    let need_u64 = tile_unit_u64 << (tile.window - 1);
                    if scratch.len() < need_u64 {
                        scratch.resize(need_u64, 0);
                    }
                    let p_ptrs: [*const blst_p1_affine; 2] =
                        [bases[tile.x..].as_ptr(), ptr::null()];
                    let s_ptrs: [*const u8; 2] = [
                        scalar_bufs[tile.poly_idx][tile.x * SCALAR_BYTES..]
                            .as_ptr(),
                        ptr::null(),
                    ];
                    let mut out = blst_p1::default();
                    // SAFETY: valid pointers and sized scratch.
                    unsafe {
                        blst_p1s_tile_pippenger(
                            &mut out,
                            p_ptrs.as_ptr(),
                            tile.dx,
                            s_ptrs.as_ptr(),
                            SCALAR_NBITS,
                            scratch.as_mut_ptr(),
                            bit0,
                            tile.window,
                        );
                    }
                    out
                });
            }
            let pts = &bases[tile.x..tile.x + tile.dx];
            let scs = &scalar_bufs[tile.poly_idx]
                [tile.x * SCALAR_BYTES..(tile.x + tile.dx) * SCALAR_BYTES];
            batched_affine_tile_pippenger(pts, scs, bit0, tile.window)
        })
        .collect();

    let mut projs = [blst_p1::default(); K];
    let mut cursor = 0usize;
    for poly_idx in 0..K {
        let len = lengths[poly_idx];
        if len == 0 {
            continue;
        }
        if len < 64 {
            projs[poly_idx] =
                blst_msm_serial_raw(&bases[..len], &scalar_bufs[poly_idx]);
            continue;
        }
        let window = poly_windows[poly_idx];
        let ny = poly_ny[poly_idx];
        let nx = poly_nx[poly_idx];
        let slice = &tile_results[cursor..cursor + ny * nx];
        cursor += ny * nx;

        let mut acc = blst_p1::default();
        for w_idx in (0..ny).rev() {
            let row = &slice[w_idx * nx..(w_idx + 1) * nx];
            for tile_pt in row {
                // SAFETY: valid Jacobian points.
                unsafe { blst_p1_add_or_double(&mut acc, &acc, tile_pt) };
            }
            if w_idx > 0 {
                for _ in 0..window {
                    // SAFETY: valid Jacobian point.
                    unsafe { blst_p1_double(&mut acc, &acc) };
                }
            }
        }
        projs[poly_idx] = acc;
    }

    blst_p1s_to_g1_affines(&projs)
}

/// Convert a `dusk_bls12_381::G1Affine` to `blst_p1_affine` via raw Montgomery
/// limb copy (zero field inversions or Montgomery conversions).
#[inline]
pub(crate) fn g1_affine_to_blst(point: &G1Affine) -> blst_p1_affine {
    if bool::from(point.is_identity()) {
        return blst_p1_affine::default();
    }
    let raw = point.to_raw_bytes();
    let mut x = [0u64; 6];
    let mut y = [0u64; 6];
    for i in 0..6 {
        x[i] = u64::from_le_bytes(
            raw[i * 8..(i + 1) * 8].try_into().expect("8-byte limb"),
        );
        y[i] = u64::from_le_bytes(
            raw[48 + i * 8..48 + (i + 1) * 8]
                .try_into()
                .expect("8-byte limb"),
        );
    }
    blst_p1_affine {
        x: blst_fp { l: x },
        y: blst_fp { l: y },
    }
}

/// Convert a slice of `G1Affine` points into `blst_p1_affine` in parallel for
/// large slices (such as `CommitKey::powers_of_g`).
pub(crate) fn g1_affines_to_blst(points: &[G1Affine]) -> Vec<blst_p1_affine> {
    if points.len() >= 1024 && rayon::current_num_threads() > 1 {
        points.par_iter().map(g1_affine_to_blst).collect()
    } else {
        points.iter().map(g1_affine_to_blst).collect()
    }
}

/// Convert a valid prime-order `blst_p1_affine` back into `G1Affine`.
///
/// # Safety contract (internal)
/// `point` must be a valid affine point in the BLS12-381 prime-order G1
/// subgroup (or infinity represented as `x = 0, y = 0`), with canonical
/// Montgomery coordinates (guaranteed for all `blst` outputs).
#[inline]
pub(crate) fn blst_p1_affine_to_g1(point: &blst_p1_affine) -> G1Affine {
    // SAFETY: `blst_p1_affine_is_inf` only reads `point`.
    if unsafe { blst_p1_affine_is_inf(point) } {
        return G1Affine::identity();
    }
    let mut raw = [0u8; G1Affine::RAW_SIZE];
    for i in 0..6 {
        raw[i * 8..(i + 1) * 8].copy_from_slice(&point.x.l[i].to_le_bytes());
        raw[48 + i * 8..48 + (i + 1) * 8]
            .copy_from_slice(&point.y.l[i].to_le_bytes());
    }
    raw[G1Affine::RAW_SIZE - 1] = 0;
    // SAFETY: `raw` is 97 bytes containing canonical Montgomery `Fp` limbs of a
    // validated prime-order G1 point produced by `blst`.
    unsafe { G1Affine::from_slice_unchecked(&raw) }
}

/// Normalize a single `blst_p1` Jacobian point into `G1Affine`.
#[inline]
pub(crate) fn blst_p1_to_g1_affine(point: &blst_p1) -> G1Affine {
    let mut affine = blst_p1_affine::default();
    // SAFETY: valid input and output pointers.
    unsafe { blst_p1_to_affine(&mut affine, point) };
    blst_p1_affine_to_g1(&affine)
}

/// Batch-normalize `K` Jacobian `blst_p1` points into `G1Affine` using a single
/// `Fp` inversion.
pub(crate) fn blst_p1s_to_g1_affines<const K: usize>(
    points: &[blst_p1; K],
) -> [G1Affine; K] {
    if K == 0 {
        return [G1Affine::identity(); K];
    }
    if K == 1 {
        let mut out = [G1Affine::identity(); K];
        out[0] = blst_p1_to_g1_affine(&points[0]);
        return out;
    }
    let mut affines = [blst_p1_affine::default(); K];
    let ptrs: [*const blst_p1; 2] = [points.as_ptr(), ptr::null()];
    // SAFETY: `ptrs` is a contiguous-array sentinel (`[base, null]`) for `K`
    // elements, and `affines` has capacity `K`.
    unsafe { blst_p1s_to_affine(affines.as_mut_ptr(), ptrs.as_ptr(), K) };
    core::array::from_fn(|i| blst_p1_affine_to_g1(&affines[i]))
}

/// Convert a `BlsScalar` (Montgomery `[u64; 4]`) into canonical 32-byte LE
/// scalar bytes using `blst`'s assembly Montgomery reduction.
#[inline]
fn scalar_to_blst_le_bytes(scalar: &BlsScalar) -> [u8; SCALAR_BYTES] {
    let fr = blst_fr {
        l: *scalar.internal_repr(),
    };
    let mut out = blst_scalar::default();
    // SAFETY: valid input/output pointers.
    unsafe { blst_scalar_from_fr(&mut out, &fr) };
    out.b
}

/// Convert a slice of `BlsScalar`s into tightly packed 32-byte LE scalars.
fn scalars_to_le_bytes_vec(scalars: &[BlsScalar]) -> Vec<u8> {
    let n = scalars.len();
    let mut out = vec![0u8; n * SCALAR_BYTES];
    if n >= 1024 && rayon::current_num_threads() > 1 {
        out.par_chunks_exact_mut(SCALAR_BYTES)
            .zip(scalars.par_iter())
            .for_each(|(dst, sc)| {
                dst.copy_from_slice(&scalar_to_blst_le_bytes(sc));
            });
    } else {
        for (dst, sc) in out
            .as_chunks_mut::<SCALAR_BYTES>()
            .0
            .iter_mut()
            .zip(scalars.iter())
        {
            *dst = scalar_to_blst_le_bytes(sc);
        }
    }
    out
}

/// Decode a canonical 48-byte compressed BLS12-381 G1 point using `blst`
/// (checking flags, canonical field coordinate, curve equation, and prime-order
/// subgroup membership).
#[inline]
pub(crate) fn decode_g1_compressed(
    bytes: &[u8; 48],
) -> Result<G1Affine, dusk_bytes::Error> {
    let mut affine = blst_p1_affine::default();
    // SAFETY: `bytes` is 48 bytes; `blst_p1_uncompress` enforces the Zcash/IETF
    // compressed G1 encoding (compression bit, infinity encoding, canonical x,
    // curve membership, and lexicographic y sign).
    let err = unsafe { blst_p1_uncompress(&mut affine, bytes.as_ptr()) };
    if err != BLST_ERROR::BLST_SUCCESS {
        return Err(dusk_bytes::Error::InvalidData);
    }
    // Check prime-order subgroup (for infinity, `blst_p1_affine_in_g1` returns
    // true, matching `G1Affine::from_compressed`).
    // SAFETY: `affine` is a valid on-curve point or infinity.
    if !unsafe { blst_p1_affine_in_g1(&affine) } {
        return Err(dusk_bytes::Error::InvalidData);
    }
    Ok(blst_p1_affine_to_g1(&affine))
}

/// Precompute Miller-loop line coefficients for a validated `G2Affine` point.
pub(crate) fn precompute_g2_lines(point: &G2Affine) -> Box<[blst_fp6; 68]> {
    let compressed = point.to_compressed();
    let mut g2_aff = blst_p2_affine::default();
    // SAFETY: `compressed` is 96 bytes from a validated `G2Affine`.
    let err = unsafe { blst_p2_uncompress(&mut g2_aff, compressed.as_ptr()) };
    assert_eq!(
        err,
        BLST_ERROR::BLST_SUCCESS,
        "OpeningKey G2Affine must decompress cleanly in blst"
    );
    let mut lines = Box::new([blst_fp6::default(); 68]);
    // SAFETY: `lines` has 68 `blst_fp6` slots as required by `blst`.
    unsafe { blst_precompute_lines(lines.as_mut_ptr(), &g2_aff) };
    lines
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

/// Choose the optimal Pippenger window width for `n` points per tile,
/// matching `blst`'s `pippenger_window_size` heuristic so bucket integration
/// (`2^(w-1)` buckets) stays balanced against `n` and L2-cache-friendly on
/// hybrid P+E cores.
#[inline]
fn pippenger_window(n: usize) -> usize {
    let lg = (usize::BITS - 1 - n.max(1).leading_zeros()) as usize;
    if lg > 12 {
        (lg - 3).min(13)
    } else if lg > 8 {
        lg - 2
    } else if lg > 4 {
        lg - 1
    } else if lg > 0 {
        2
    } else {
        1
    }
}

/// Run a single-threaded `blst` MSM on `points` and packed `nbytes`-per-scalar
/// LE `scalar_bytes` with bit-width `nbits`.
fn blst_msm_serial_raw_nbits(
    points: &[blst_p1_affine],
    scalar_bytes: &[u8],
    nbits: usize,
) -> blst_p1 {
    let n = points.len();
    let nbytes = nbits.div_ceil(8);
    debug_assert_eq!(scalar_bytes.len(), n * nbytes);
    if n == 0 {
        return blst_p1::default();
    }
    if n == 1 {
        let mut p = blst_p1::default();
        let mut out = blst_p1::default();
        // SAFETY: valid point and `nbytes`-byte scalar pointer.
        unsafe {
            blst_p1_from_affine(&mut p, &points[0]);
            blst_p1_mult(&mut out, &p, scalar_bytes.as_ptr(), nbits);
        }
        return out;
    }

    PIPPENGER_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        // SAFETY: pure size query.
        let need_u64 =
            unsafe { blst_p1s_mult_pippenger_scratch_sizeof(n) }.div_ceil(8);
        if scratch.len() < need_u64 {
            scratch.resize(need_u64, 0);
        }
        let p_ptrs: [*const blst_p1_affine; 2] = [points.as_ptr(), ptr::null()];
        let s_ptrs: [*const u8; 2] = [scalar_bytes.as_ptr(), ptr::null()];
        let mut out = blst_p1::default();
        // SAFETY: `p_ptrs` and `s_ptrs` point to contiguous slices of length
        // `n`; `scratch` is sized via `blst_p1s_mult_pippenger_scratch_sizeof`.
        unsafe {
            blst_p1s_mult_pippenger(
                &mut out,
                p_ptrs.as_ptr(),
                n,
                s_ptrs.as_ptr(),
                nbits,
                scratch.as_mut_ptr(),
            );
        }
        out
    })
}

/// Run a single-threaded `blst` MSM on `points` and packed 32-byte LE `scalars`.
fn blst_msm_serial_raw(
    points: &[blst_p1_affine],
    scalar_bytes: &[u8],
) -> blst_p1 {
    blst_msm_serial_raw_nbits(points, scalar_bytes, SCALAR_NBITS)
}

/// Evaluate a batch of `K` MSMs sharing the same `bases` slice using
/// `blst_p1s_tile_pippenger` scheduled cleanly across Rayon with thread-local
/// scratch buffers and a single batched affine normalization.
pub(crate) fn msm_g1_batch<const K: usize>(
    bases: &[blst_p1_affine],
    polys: [&[BlsScalar]; K],
) -> [G1Affine; K] {
    let mut lengths = [0usize; K];
    for (i, poly) in polys.iter().enumerate() {
        let mut len = poly.len();
        while len > 0 && poly[len - 1] == BlsScalar::zero() {
            len -= 1;
        }
        assert!(
            len <= bases.len(),
            "MSM polynomial length exceeds CommitKey bases"
        );
        lengths[i] = len;
    }

    let max_len = lengths.iter().copied().max().unwrap_or(0);
    if max_len == 0 {
        return [G1Affine::identity(); K];
    }

    let num_threads = rayon::current_num_threads();
    if num_threads > 1 && max_len >= 1024 {
        return msm_g1_batch_affine(bases, polys, None);
    }
    if num_threads <= 1 || max_len < 64 {
        let mut projs = [blst_p1::default(); K];
        for i in 0..K {
            let len = lengths[i];
            if len > 0 {
                let sbytes = scalars_to_le_bytes_vec(&polys[i][..len]);
                projs[i] = blst_msm_serial_raw(&bases[..len], &sbytes);
            }
        }
        return blst_p1s_to_g1_affines(&projs);
    }

    // Convert all active polynomials' scalars into packed 32-byte LE buffers in
    // one flat parallel pass.
    let mut scalar_bufs: Vec<Vec<u8>> = lengths
        .iter()
        .map(|&len| vec![0u8; len * SCALAR_BYTES])
        .collect();
    {
        // Flatten chunk conversion across all polynomials.
        let total_scalars: usize = lengths.iter().sum();
        if total_scalars >= 1024 {
            scalar_bufs
                .par_iter_mut()
                .zip(polys.par_iter())
                .zip(lengths.par_iter())
                .for_each(|((buf, poly), &len)| {
                    if len == 0 {
                        return;
                    }
                    buf.par_chunks_exact_mut(SCALAR_BYTES)
                        .zip(poly[..len].par_iter())
                        .for_each(|(dst, sc)| {
                            dst.copy_from_slice(&scalar_to_blst_le_bytes(sc));
                        });
                });
        } else {
            for i in 0..K {
                let len = lengths[i];
                for (dst, sc) in scalar_bufs[i]
                    .as_chunks_mut::<SCALAR_BYTES>()
                    .0
                    .iter_mut()
                    .zip(polys[i][..len].iter())
                {
                    *dst = scalar_to_blst_le_bytes(sc);
                }
            }
        }
    }

    // Build a flat list of `(poly_idx, x_start, x_len, window_idx, window_bits)`
    // tiles across all K polynomials so every core stays saturated without
    // nested Rayon barriers.
    struct TileSpec {
        poly_idx: usize,
        x: usize,
        dx: usize,
        w_idx: usize,
        window: usize,
    }

    let mut poly_windows = [0usize; K];
    let mut poly_ny = [0usize; K];
    let mut poly_nx = [0usize; K];
    let mut tiles = Vec::new();

    // Count how many polynomials are actually non-trivial.
    let active_polys = lengths.iter().filter(|&&l| l >= 64).count().max(1);

    for (poly_idx, &len) in lengths.iter().enumerate() {
        if len < 64 {
            continue;
        }
        // Choose `nx` and `window` jointly so `window` is optimal for the
        // actual per-tile point count `dx = len / nx`, and emit enough tiles
        // (~2x thread count when `len >= 4096`) to absorb P/E-core speed
        // asymmetry via Rayon work-stealing without straggler stalls.
        let initial_w = pippenger_window(len);
        let initial_ny = SCALAR_NBITS.div_ceil(initial_w);
        let desired_total_tiles = if len >= 4096 {
            num_threads.saturating_mul(2)
        } else {
            num_threads
        };
        let target_tiles_per_poly =
            desired_total_tiles.div_ceil(active_polys).max(initial_ny);
        let nx = if len >= 2048 {
            target_tiles_per_poly
                .div_ceil(initial_ny)
                .min((len / 2048).max(1))
                .max(1)
        } else {
            1
        };
        let dx_approx = len.div_ceil(nx);
        let window = pippenger_window(dx_approx);
        let ny = SCALAR_NBITS.div_ceil(window);
        poly_windows[poly_idx] = window;
        poly_ny[poly_idx] = ny;
        poly_nx[poly_idx] = nx;

        let base_dx = len / nx;
        let rem = len % nx;
        for w_idx in 0..ny {
            let mut x = 0usize;
            for xi in 0..nx {
                let dx = base_dx + usize::from(xi < rem);
                tiles.push(TileSpec {
                    poly_idx,
                    x,
                    dx,
                    w_idx,
                    window,
                });
                x += dx;
            }
        }
    }

    // SAFETY: pure size query for one tile bucket unit.
    let tile_unit_u64 =
        unsafe { blst_p1s_mult_pippenger_scratch_sizeof(0) }.div_ceil(8);

    let tile_results: Vec<blst_p1> = tiles
        .par_iter()
        .map(|tile| {
            PIPPENGER_SCRATCH.with(|cell| {
                let mut scratch = cell.borrow_mut();
                let need_u64 = tile_unit_u64 << (tile.window - 1);
                if scratch.len() < need_u64 {
                    scratch.resize(need_u64, 0);
                }
                let p_ptrs: [*const blst_p1_affine; 2] =
                    [bases[tile.x..].as_ptr(), ptr::null()];
                let s_ptrs: [*const u8; 2] = [
                    scalar_bufs[tile.poly_idx][tile.x * SCALAR_BYTES..]
                        .as_ptr(),
                    ptr::null(),
                ];
                let mut out = blst_p1::default();
                // SAFETY: `tile.x + tile.dx <= lengths[tile.poly_idx] <= bases.len()`,
                // `scratch` has `tile_unit_u64 << (window - 1)` limbs as required
                // by `blst_p1s_tile_pippenger`.
                unsafe {
                    blst_p1s_tile_pippenger(
                        &mut out,
                        p_ptrs.as_ptr(),
                        tile.dx,
                        s_ptrs.as_ptr(),
                        SCALAR_NBITS,
                        scratch.as_mut_ptr(),
                        tile.w_idx * tile.window,
                        tile.window,
                    );
                }
                out
            })
        })
        .collect();

    // Reduce tile results per polynomial from highest window down to 0.
    let mut projs = [blst_p1::default(); K];
    let mut cursor = 0usize;
    for poly_idx in 0..K {
        let len = lengths[poly_idx];
        if len == 0 {
            continue;
        }
        if len < 64 {
            projs[poly_idx] =
                blst_msm_serial_raw(&bases[..len], &scalar_bufs[poly_idx]);
            continue;
        }
        let window = poly_windows[poly_idx];
        let ny = poly_ny[poly_idx];
        let nx = poly_nx[poly_idx];
        let slice = &tile_results[cursor..cursor + ny * nx];
        cursor += ny * nx;

        let mut acc = blst_p1::default();
        for w_idx in (0..ny).rev() {
            let row = &slice[w_idx * nx..(w_idx + 1) * nx];
            for tile_pt in row {
                // SAFETY: valid Jacobian points.
                unsafe { blst_p1_add_or_double(&mut acc, &acc, tile_pt) };
            }
            if w_idx > 0 {
                for _ in 0..window {
                    // SAFETY: valid Jacobian point.
                    unsafe { blst_p1_double(&mut acc, &acc) };
                }
            }
        }
        projs[poly_idx] = acc;
    }

    blst_p1s_to_g1_affines(&projs)
}

/// Evaluate a single MSM sharing `bases` (e.g., `z_poly` commitment).
pub(crate) fn msm_g1(
    bases: &[blst_p1_affine],
    scalars: &[BlsScalar],
) -> G1Affine {
    msm_g1_batch(bases, [scalars])[0]
}

/// Evaluate a variable-base MSM over `G1Affine` points and `BlsScalar`s
/// (used for batch verification).
pub(crate) fn msm_g1_variable_base(
    points: &[G1Affine],
    scalars: &[BlsScalar],
) -> G1Affine {
    assert_eq!(points.len(), scalars.len());
    if points.is_empty() {
        return G1Affine::identity();
    }

    // Filter identity points and zero scalars so `blst` only processes active
    // terms.
    let mut blst_pts = Vec::with_capacity(points.len());
    let mut active_scs = Vec::with_capacity(scalars.len());
    for (pt, sc) in points.iter().zip(scalars.iter()) {
        if bool::from(pt.is_identity()) || *sc == BlsScalar::zero() {
            continue;
        }
        blst_pts.push(g1_affine_to_blst(pt));
        active_scs.push(*sc);
    }

    if blst_pts.is_empty() {
        return G1Affine::identity();
    }

    msm_g1(&blst_pts, &active_scs)
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

const COMB_GROUPS: usize = 2;
const COMB_GROUP_SIZE: usize = 8;
const COMB_TEETH: usize = 4;
const COMB_SUBSETS: usize = 256;

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
    fn eval_serial(
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
fn msm_variable_base_proj(
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
            PIPPENGER_SCRATCH.with(|cell| {
                let mut scratch = cell.borrow_mut();
                let need_u64 = tile_unit_u64 << (window - 1);
                if scratch.len() < need_u64 {
                    scratch.resize(need_u64, 0);
                }
                let p_ptrs: [*const blst_p1_affine; 2] =
                    [glv_pts.as_ptr(), ptr::null()];
                let s_ptrs: [*const u8; 2] = [glv_scs.as_ptr(), ptr::null()];
                let mut out = blst_p1::default();
                // SAFETY: valid pointers and sized scratch.
                unsafe {
                    blst_p1s_tile_pippenger(
                        &mut out,
                        p_ptrs.as_ptr(),
                        m,
                        s_ptrs.as_ptr(),
                        GLV_NBITS,
                        scratch.as_mut_ptr(),
                        w_idx * window,
                        window,
                    );
                }
                out
            })
        })
        .collect();

    let mut acc = blst_p1::default();
    for w_idx in (0..ny).rev() {
        // SAFETY: valid Jacobian points.
        unsafe { blst_p1_add_or_double(&mut acc, &acc, &tile_results[w_idx]) };
        if w_idx > 0 {
            for _ in 0..window {
                // SAFETY: valid Jacobian point.
                unsafe { blst_p1_double(&mut acc, &acc) };
            }
        }
    }
    acc
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

/// Compute the Lagrange-basis SRS `[L_0(tau)]_1, ..., [L_{n-1}(tau)]_1` from
/// the monomial SRS `[tau^0]_1, ..., [tau^{n-1}]_1` via parallel radix-2 G1
/// inverse FFT in `blst` Jacobian coordinates followed by batched affine
/// normalization.
pub(crate) fn g1_ifft_blst(
    monomial_powers: &[blst_p1_affine],
    domain: &EvaluationDomain,
) -> Vec<blst_p1_affine> {
    let n = domain.size();
    assert!(monomial_powers.len() >= n);
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![monomial_powers[0]];
    }
    let log_n = domain.log_size_of_group;

    // Bit-reverse monomial bases into Jacobian buffer.
    let mut buf: Vec<blst_p1> = (0..n)
        .into_par_iter()
        .map(|i| {
            let src = &monomial_powers[crate::fast_fr::bitrev(i as u32, log_n)];
            let mut p = blst_p1::default();
            // SAFETY: valid affine/Jacobian pointers.
            unsafe { blst_p1_from_affine(&mut p, src) };
            p
        })
        .collect();

    // Precompute `n/2` inverse roots of unity in 32-byte LE scalar form.
    let inv_roots_bls = domain.precompute_ifft_roots();
    let inv_roots_le: Vec<[u8; SCALAR_BYTES]> = inv_roots_bls
        .par_iter()
        .map(scalar_to_blst_le_bytes)
        .collect();

    let mut m = 1usize;
    while m < n {
        let step = m << 1;
        let stride = n / step;
        if m == 1 {
            buf.par_chunks_exact_mut(2).for_each(|pair| {
                let u = pair[0];
                let mut neg_v = pair[1];
                // SAFETY: valid Jacobian points.
                unsafe {
                    blst_p1_add_or_double(&mut pair[0], &u, &pair[1]);
                    blst_fp_cneg(&mut neg_v.y, &neg_v.y, true);
                    blst_p1_add_or_double(&mut pair[1], &u, &neg_v);
                }
            });
        } else if n / step >= 16 {
            buf.par_chunks_exact_mut(step).for_each(|block| {
                let (lo, hi) = block.split_at_mut(m);
                // j = 0: twiddle = 1
                {
                    let u = lo[0];
                    let mut neg_v = hi[0];
                    // SAFETY: valid Jacobian points.
                    unsafe {
                        blst_p1_add_or_double(&mut lo[0], &u, &hi[0]);
                        blst_fp_cneg(&mut neg_v.y, &neg_v.y, true);
                        blst_p1_add_or_double(&mut hi[0], &u, &neg_v);
                    }
                }
                for j in 1..m {
                    let u = lo[j];
                    let mut v = blst_p1::default();
                    let w = &inv_roots_le[j * stride];
                    // SAFETY: valid Jacobian points and 32-byte scalar.
                    unsafe {
                        blst_p1_mult(&mut v, &hi[j], w.as_ptr(), SCALAR_NBITS);
                        blst_p1_add_or_double(&mut lo[j], &u, &v);
                        blst_fp_cneg(&mut v.y, &v.y, true);
                        blst_p1_add_or_double(&mut hi[j], &u, &v);
                    }
                }
            });
        } else {
            for block in buf.chunks_exact_mut(step) {
                let (lo, hi) = block.split_at_mut(m);
                lo.par_iter_mut()
                    .zip(hi.par_iter_mut())
                    .enumerate()
                    .for_each(|(j, (u_slot, v_slot))| {
                        let u = *u_slot;
                        let mut v = *v_slot;
                        if j > 0 {
                            let w = &inv_roots_le[j * stride];
                            let mut prod = blst_p1::default();
                            // SAFETY: valid Jacobian point and 32-byte scalar.
                            unsafe {
                                blst_p1_mult(
                                    &mut prod,
                                    &v,
                                    w.as_ptr(),
                                    SCALAR_NBITS,
                                );
                            }
                            v = prod;
                        }
                        // SAFETY: valid Jacobian points.
                        unsafe {
                            blst_p1_add_or_double(u_slot, &u, &v);
                            blst_fp_cneg(&mut v.y, &v.y, true);
                            blst_p1_add_or_double(v_slot, &u, &v);
                        }
                    });
            }
        }
        m = step;
    }

    // Scale by `1/n` and batch-normalize to affine in parallel chunks.
    let size_inv_le = scalar_to_blst_le_bytes(&domain.size_inv);
    buf.par_iter_mut().for_each(|p| {
        let cur = *p;
        // SAFETY: valid Jacobian point and 32-byte scalar.
        unsafe { blst_p1_mult(p, &cur, size_inv_le.as_ptr(), SCALAR_NBITS) };
    });

    let mut affines = vec![blst_p1_affine::default(); n];
    const NORM_CHUNK: usize = 512;
    affines
        .par_chunks_mut(NORM_CHUNK)
        .zip(buf.par_chunks(NORM_CHUNK))
        .for_each(|(dst, src)| {
            let ptrs: [*const blst_p1; 2] = [src.as_ptr(), ptr::null()];
            // SAFETY: contiguous slice of `src.len()` points.
            unsafe {
                blst_p1s_to_affine(dst.as_mut_ptr(), ptrs.as_ptr(), src.len());
            }
        });
    affines
}

/// Subtract `b` from `MODULUS` assuming `b < MODULUS`.
#[inline(always)]
fn sub_from_modulus(b: [u64; 4]) -> [u64; 4] {
    let m = crate::fast_fr::MODULUS;
    let (d0, borrow) = m[0].overflowing_sub(b[0]);
    let (d1, b1a) = m[1].overflowing_sub(b[1]);
    let (d1, b1b) = d1.overflowing_sub(u64::from(borrow));
    let borrow = b1a || b1b;
    let (d2, b2a) = m[2].overflowing_sub(b[2]);
    let (d2, b2b) = d2.overflowing_sub(u64::from(borrow));
    let borrow = b2a || b2b;
    let d3 = m[3].wrapping_sub(b[3]).wrapping_sub(u64::from(borrow));
    [d0, d1, d2, d3]
}

const NEG_ONE_MONT: [u64; 4] = {
    let m = crate::fast_fr::MODULUS;
    let o = crate::fast_fr::ONE_MONT;
    let (d0, b0) = m[0].overflowing_sub(o[0]);
    let (d1a, b1a) = m[1].overflowing_sub(o[1]);
    let (d1, b1b) = d1a.overflowing_sub(b0 as u64);
    let b1 = b1a || b1b;
    let (d2a, b2a) = m[2].overflowing_sub(o[2]);
    let (d2, b2b) = d2a.overflowing_sub(b1 as u64);
    let b2 = b2a || b2b;
    let d3 = m[3] - o[3] - (b2 as u64);
    [d0, d1, d2, d3]
};

#[inline(always)]
fn neg_affine_blst(mut p: blst_p1_affine) -> blst_p1_affine {
    // SAFETY: negates y coordinate in place (infinity has y=0 -> unchanged).
    unsafe { blst_fp_cneg(&mut p.y, &p.y, true) };
    p
}

/// Commit to `K` blinded wire polynomials directly from their Lagrange
/// evaluations `evals` (`len == n`) and blinding scalars `blinders`
/// (`(b_0 + b_1 X + ...)(X^n - 1)`), using sparse + bit-width bucketed MSMs.
///
/// Produces bit-for-bit identical commitments to monomial `msm_g1_batch` on
/// the blinded coefficient polynomials while skipping zeros, folding `±1..=±15`
/// in a single mixed add per element, and running narrow `nbits = 16 / 64`
/// Pippenger passes for small-integer witnesses.
pub(crate) fn msm_g1_lagrange_batch<const K: usize>(
    lagrange_bases: &[blst_p1_affine],
    monomial_bases: &[blst_p1_affine],
    evals: [&[BlsScalar]; K],
    blinders: [&[BlsScalar]; K],
) -> [G1Affine; K] {
    let n = lagrange_bases.len();
    for i in 0..K {
        debug_assert_eq!(evals[i].len(), n);
        assert!(n + blinders[i].len() <= monomial_bases.len());
    }
    if n == 0 {
        return [G1Affine::identity(); K];
    }

    struct ChunkAcc {
        wire_idx: usize,
        tiny_buckets: [blst_p1; 15],
        pts16: Vec<blst_p1_affine>,
        scs16: Vec<u8>,
        pts64: Vec<blst_p1_affine>,
        scs64: Vec<u8>,
        pts256: Vec<blst_p1_affine>,
        scs256: Vec<u8>,
    }

    const CLASSIFY_CHUNK: usize = 2048;
    let chunks_per_wire = n.div_ceil(CLASSIFY_CHUNK);
    let tasks: Vec<(usize, usize, usize)> = (0..K)
        .flat_map(|wire_idx| {
            (0..chunks_per_wire).map(move |c| {
                let start = c * CLASSIFY_CHUNK;
                let end = (start + CLASSIFY_CHUNK).min(n);
                (wire_idx, start, end)
            })
        })
        .collect();

    let chunk_accs: Vec<ChunkAcc> = tasks
        .into_par_iter()
        .map(|(wire_idx, start, end)| {
            let mut tiny = [blst_p1::default(); 15];
            let mut pts16 = Vec::new();
            let mut scs16 = Vec::new();
            let mut pts64 = Vec::new();
            let mut scs64 = Vec::new();
            let mut pts256 = Vec::new();
            let mut scs256 = Vec::new();

            let ev = &evals[wire_idx][start..end];
            let bases = &lagrange_bases[start..end];
            let m3 = crate::fast_fr::MODULUS[3];

            for (sc, &base) in ev.iter().zip(bases.iter()) {
                let raw = *sc.internal_repr();
                if (raw[0] | raw[1] | raw[2] | raw[3]) == 0 {
                    continue;
                }
                if raw == crate::fast_fr::ONE_MONT {
                    // SAFETY: valid Jacobian + affine pointers.
                    unsafe {
                        blst_p1_add_or_double_affine(
                            &mut tiny[0],
                            &tiny[0],
                            &base,
                        );
                    }
                    continue;
                }
                if raw == NEG_ONE_MONT {
                    let nb = neg_affine_blst(base);
                    // SAFETY: valid Jacobian + affine pointers.
                    unsafe {
                        blst_p1_add_or_double_affine(
                            &mut tiny[0],
                            &tiny[0],
                            &nb,
                        );
                    }
                    continue;
                }

                let bytes = scalar_to_blst_le_bytes(sc);
                let l0 = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
                let l1 = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
                let l2 = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
                let l3 = u64::from_le_bytes(bytes[24..32].try_into().unwrap());

                if (l1 | l2 | l3) == 0 {
                    if l0 <= 15 {
                        let slot = (l0 as usize) - 1;
                        // SAFETY: valid Jacobian + affine pointers.
                        unsafe {
                            blst_p1_add_or_double_affine(
                                &mut tiny[slot],
                                &tiny[slot],
                                &base,
                            );
                        }
                    } else if l0 < 32768 {
                        pts16.push(base);
                        scs16.extend_from_slice(&(l0 as u16).to_le_bytes());
                    } else if l0 < (1u64 << 63) {
                        pts64.push(base);
                        scs64.extend_from_slice(&l0.to_le_bytes());
                    } else {
                        pts256.push(base);
                        scs256.extend_from_slice(&bytes);
                    }
                    continue;
                }

                if l3 == m3 {
                    let neg = sub_from_modulus([l0, l1, l2, l3]);
                    if (neg[1] | neg[2] | neg[3]) == 0 {
                        let v = neg[0];
                        let nb = neg_affine_blst(base);
                        if v <= 15 {
                            let slot = (v as usize) - 1;
                            // SAFETY: valid Jacobian + affine pointers.
                            unsafe {
                                blst_p1_add_or_double_affine(
                                    &mut tiny[slot],
                                    &tiny[slot],
                                    &nb,
                                );
                            }
                        } else if v < 32768 {
                            pts16.push(nb);
                            scs16.extend_from_slice(&(v as u16).to_le_bytes());
                        } else if v < (1u64 << 63) {
                            pts64.push(nb);
                            scs64.extend_from_slice(&v.to_le_bytes());
                        } else {
                            pts256.push(base);
                            scs256.extend_from_slice(&bytes);
                        }
                        continue;
                    }
                }

                pts256.push(base);
                scs256.extend_from_slice(&bytes);
            }

            ChunkAcc {
                wire_idx,
                tiny_buckets: tiny,
                pts16,
                scs16,
                pts64,
                scs64,
                pts256,
                scs256,
            }
        })
        .collect();

    // Merge per-wire tiny buckets and concatenate bucket point/scalar arrays.
    let mut wire_tiny = [[blst_p1::default(); 15]; K];
    let mut w_pts16: [Vec<blst_p1_affine>; K] =
        core::array::from_fn(|_| Vec::new());
    let mut w_scs16: [Vec<u8>; K] = core::array::from_fn(|_| Vec::new());
    let mut w_pts64: [Vec<blst_p1_affine>; K] =
        core::array::from_fn(|_| Vec::new());
    let mut w_scs64: [Vec<u8>; K] = core::array::from_fn(|_| Vec::new());
    let mut w_pts256: [Vec<blst_p1_affine>; K] =
        core::array::from_fn(|_| Vec::new());
    let mut w_scs256: [Vec<u8>; K] = core::array::from_fn(|_| Vec::new());

    for acc in chunk_accs {
        let w = acc.wire_idx;
        for (dst, src) in wire_tiny[w].iter_mut().zip(acc.tiny_buckets.iter()) {
            // SAFETY: valid Jacobian points.
            unsafe {
                blst_p1_add_or_double(dst, dst, src);
            }
        }
        w_pts16[w].extend_from_slice(&acc.pts16);
        w_scs16[w].extend_from_slice(&acc.scs16);
        w_pts64[w].extend_from_slice(&acc.pts64);
        w_scs64[w].extend_from_slice(&acc.scs64);
        w_pts256[w].extend_from_slice(&acc.pts256);
        w_scs256[w].extend_from_slice(&acc.scs256);
    }

    // Fold wire blinders `sum_j b_j * ([tau^{n+j}]_1 - [tau^j]_1)` into the
    // 256-bit bucket for each wire.
    for w in 0..K {
        for (j, b) in blinders[w].iter().enumerate() {
            if *b == BlsScalar::zero() {
                continue;
            }
            let b_bytes = scalar_to_blst_le_bytes(b);
            w_pts256[w].push(monomial_bases[n + j]);
            w_scs256[w].extend_from_slice(&b_bytes);
            w_pts256[w].push(neg_affine_blst(monomial_bases[j]));
            w_scs256[w].extend_from_slice(&b_bytes);
        }
    }

    // Build a flat list of Pippenger jobs (across all K wires and bit-widths
    // 16 / 64 / 256) and tile them concurrently across all cores!
    struct BucketJob<'a> {
        wire_idx: usize,
        nbits: usize,
        nbytes: usize,
        pts: &'a [blst_p1_affine],
        scs: &'a [u8],
    }
    let mut jobs: Vec<BucketJob<'_>> = Vec::with_capacity(K * 3);
    for w in 0..K {
        if !w_pts16[w].is_empty() {
            jobs.push(BucketJob {
                wire_idx: w,
                nbits: 16,
                nbytes: 2,
                pts: &w_pts16[w],
                scs: &w_scs16[w],
            });
        }
        if !w_pts64[w].is_empty() {
            jobs.push(BucketJob {
                wire_idx: w,
                nbits: 64,
                nbytes: 8,
                pts: &w_pts64[w],
                scs: &w_scs64[w],
            });
        }
        if !w_pts256[w].is_empty() {
            jobs.push(BucketJob {
                wire_idx: w,
                nbits: 256,
                nbytes: 32,
                pts: &w_pts256[w],
                scs: &w_scs256[w],
            });
        }
    }

    struct LTile {
        job_idx: usize,
        x: usize,
        dx: usize,
        w_idx: usize,
        window: usize,
    }

    let num_threads = rayon::current_num_threads();
    let mut tiles = Vec::new();
    let mut job_meta: Vec<(usize, usize, usize)> =
        Vec::with_capacity(jobs.len());
    let mut small_jobs: Vec<usize> = Vec::new();

    for (job_idx, job) in jobs.iter().enumerate() {
        let len = job.pts.len();
        if len < 64 || num_threads <= 1 {
            job_meta.push((0, 0, 0));
            small_jobs.push(job_idx);
            continue;
        }
        let use_affine = job.nbits == SCALAR_NBITS && len >= 1024;
        let choose_w = |n: usize| -> usize {
            if use_affine {
                choose_affine_window(n).min(job.nbits)
            } else {
                pippenger_window(n).min(job.nbits)
            }
        };
        let window = choose_w(len);
        let ny = job.nbits.div_ceil(window);
        let nx = if len >= 4096 && ny < num_threads {
            (num_threads.div_ceil(ny)).min(len / 2048).max(1)
        } else {
            1
        };
        let window = choose_w(len.div_ceil(nx));
        let ny = job.nbits.div_ceil(window);
        job_meta.push((window, ny, nx));

        let base_dx = len / nx;
        let rem = len % nx;
        for w_idx in 0..ny {
            let mut x = 0usize;
            for xi in 0..nx {
                let dx = base_dx + usize::from(xi < rem);
                tiles.push(LTile {
                    job_idx,
                    x,
                    dx,
                    w_idx,
                    window,
                });
                x += dx;
            }
        }
    }

    // SAFETY: pure size query for one tile bucket unit.
    let tile_unit_u64 =
        unsafe { blst_p1s_mult_pippenger_scratch_sizeof(0) }.div_ceil(8);

    let (tile_results, small_results): (Vec<blst_p1>, Vec<(usize, blst_p1)>) =
        rayon::join(
            || {
                tiles
                    .par_iter()
                    .map(|tile| {
                        let job = &jobs[tile.job_idx];
                        let bit0 = tile.w_idx * tile.window;
                        if job.nbits == SCALAR_NBITS && tile.dx >= 1024 {
                            let eff_bits =
                                255usize.saturating_sub(bit0).min(tile.window);
                            if eff_bits >= 10 {
                                let pts = &job.pts[tile.x..tile.x + tile.dx];
                                let scs = &job.scs[tile.x * job.nbytes
                                    ..(tile.x + tile.dx) * job.nbytes];
                                return batched_affine_tile_pippenger(
                                    pts,
                                    scs,
                                    bit0,
                                    tile.window,
                                );
                            }
                        }
                        PIPPENGER_SCRATCH.with(|cell| {
                            let mut scratch = cell.borrow_mut();
                            let need_u64 = tile_unit_u64 << (tile.window - 1);
                            if scratch.len() < need_u64 {
                                scratch.resize(need_u64, 0);
                            }
                            let p_ptrs: [*const blst_p1_affine; 2] =
                                [job.pts[tile.x..].as_ptr(), ptr::null()];
                            let s_ptrs: [*const u8; 2] = [
                                job.scs[tile.x * job.nbytes..].as_ptr(),
                                ptr::null(),
                            ];
                            let mut out = blst_p1::default();
                            // SAFETY: bounds checked by tile partitioning.
                            unsafe {
                                blst_p1s_tile_pippenger(
                                    &mut out,
                                    p_ptrs.as_ptr(),
                                    tile.dx,
                                    s_ptrs.as_ptr(),
                                    job.nbits,
                                    scratch.as_mut_ptr(),
                                    bit0,
                                    tile.window,
                                );
                            }
                            out
                        })
                    })
                    .collect()
            },
            || {
                small_jobs
                    .par_iter()
                    .map(|&job_idx| {
                        let job = &jobs[job_idx];
                        (
                            job_idx,
                            blst_msm_serial_raw_nbits(
                                job.pts, job.scs, job.nbits,
                            ),
                        )
                    })
                    .collect()
            },
        );

    // Initialize wire totals from the 15-bucket triangle sums.
    let mut projs = [blst_p1::default(); K];
    for w in 0..K {
        let mut running = blst_p1::default();
        let mut sum = blst_p1::default();
        for b in (0..15).rev() {
            // SAFETY: valid Jacobian points.
            unsafe {
                blst_p1_add_or_double(&mut running, &running, &wire_tiny[w][b]);
                blst_p1_add_or_double(&mut sum, &sum, &running);
            }
        }
        projs[w] = sum;
    }

    // Add small job results.
    for (job_idx, pt) in small_results {
        let w = jobs[job_idx].wire_idx;
        // SAFETY: valid Jacobian points.
        unsafe { blst_p1_add_or_double(&mut projs[w], &projs[w], &pt) };
    }

    // Reduce tiled job results into each wire's accumulator.
    let mut cursor = 0usize;
    for (job_idx, &(window, ny, nx)) in job_meta.iter().enumerate() {
        if ny == 0 {
            continue;
        }
        let slice = &tile_results[cursor..cursor + ny * nx];
        cursor += ny * nx;
        let mut acc = blst_p1::default();
        for w_idx in (0..ny).rev() {
            let row = &slice[w_idx * nx..(w_idx + 1) * nx];
            for tile_pt in row {
                // SAFETY: valid Jacobian points.
                unsafe { blst_p1_add_or_double(&mut acc, &acc, tile_pt) };
            }
            if w_idx > 0 {
                for _ in 0..window {
                    // SAFETY: valid Jacobian point.
                    unsafe { blst_p1_double(&mut acc, &acc) };
                }
            }
        }
        let w = jobs[job_idx].wire_idx;
        // SAFETY: valid Jacobian points.
        unsafe { blst_p1_add_or_double(&mut projs[w], &projs[w], &acc) };
    }

    blst_p1s_to_g1_affines(&projs)
}

#[cfg(test)]
mod tests {
    use dusk_bls12_381::multiscalar_mul::msm_variable_base;
    use dusk_bls12_381::{G1Projective, G2Projective};
    use dusk_bytes::Serializable;
    use ff::Field;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;

    #[test]
    fn blst_g1_and_scalar_conversions_roundtrip() {
        let mut rng = StdRng::seed_from_u64(0x1234_abcd);
        assert_eq!(
            blst_p1_affine_to_g1(&g1_affine_to_blst(&G1Affine::identity())),
            G1Affine::identity()
        );
        assert_eq!(
            blst_p1_affine_to_g1(&g1_affine_to_blst(&G1Affine::generator())),
            G1Affine::generator()
        );

        for _ in 0..32 {
            let s = BlsScalar::random(&mut rng);
            assert_eq!(scalar_to_blst_le_bytes(&s), s.to_bytes());
            let p = G1Affine::from(G1Projective::generator() * s);
            assert_eq!(blst_p1_affine_to_g1(&g1_affine_to_blst(&p)), p);
            assert_eq!(decode_g1_compressed(&p.to_bytes()), Ok(p));
        }
        assert_eq!(
            decode_g1_compressed(&G1Affine::identity().to_bytes()),
            Ok(G1Affine::identity())
        );

        // Malformed compressed points must be rejected identically.
        let mut no_comp = G1Affine::generator().to_bytes();
        no_comp[0] &= !0x80;
        let mut nonzero_inf = G1Affine::identity().to_bytes();
        nonzero_inf[47] = 1;
        assert!(decode_g1_compressed(&no_comp).is_err());
        assert!(decode_g1_compressed(&nonzero_inf).is_err());
        assert!(decode_g1_compressed(&[0u8; 48]).is_err());
    }

    #[test]
    fn blst_msm_matches_reference_across_sizes() {
        let mut rng = StdRng::seed_from_u64(0xbeef_5678);
        for n in [0usize, 1, 2, 17, 64, 129, 513] {
            let pts: Vec<G1Affine> = (0..n)
                .map(|_| {
                    G1Affine::from(
                        G1Projective::generator() * BlsScalar::random(&mut rng),
                    )
                })
                .collect();
            let scs: Vec<BlsScalar> =
                (0..n).map(|_| BlsScalar::random(&mut rng)).collect();
            let blst_pts = g1_affines_to_blst(&pts);

            let expected = G1Affine::from(msm_variable_base(&pts, &scs));
            assert_eq!(msm_g1(&blst_pts, &scs), expected, "n={n}");
            assert_eq!(msm_g1_variable_base(&pts, &scs), expected, "var n={n}");
        }

        // Batch of 4 polynomials with different lengths and trailing zeros.
        let n = 300;
        let pts: Vec<G1Affine> = (0..n)
            .map(|_| {
                G1Affine::from(
                    G1Projective::generator() * BlsScalar::random(&mut rng),
                )
            })
            .collect();
        let blst_pts = g1_affines_to_blst(&pts);
        let p0: Vec<BlsScalar> =
            (0..n).map(|_| BlsScalar::random(&mut rng)).collect();
        let mut p1: Vec<BlsScalar> =
            (0..150).map(|_| BlsScalar::random(&mut rng)).collect();
        p1.extend_from_slice(&[BlsScalar::zero(); 50]);
        let p2: Vec<BlsScalar> = vec![BlsScalar::zero(); 80];
        let p3: Vec<BlsScalar> =
            (0..11).map(|_| BlsScalar::random(&mut rng)).collect();

        let got = msm_g1_batch(&blst_pts, [&p0, &p1, &p2, &p3]);
        assert_eq!(got[0], G1Affine::from(msm_variable_base(&pts, &p0)));
        assert_eq!(got[1], G1Affine::from(msm_variable_base(&pts, &p1)));
        assert_eq!(got[2], G1Affine::identity());
        assert_eq!(got[3], G1Affine::from(msm_variable_base(&pts, &p3)));
    }

    #[test]
    fn blst_pairing_matches_reference_including_identities() {
        let mut rng = StdRng::seed_from_u64(0xcafe_9999);
        let x = BlsScalar::random(&mut rng);
        let h = G2Affine::generator();
        let x_h = G2Affine::from(G2Projective::generator() * x);
        let lines_h = precompute_g2_lines(&h);
        let lines_x_h = precompute_g2_lines(&x_h);

        // Valid pair: e(-s*G, x*H) * e(s*x*G, H) == 1
        let s = BlsScalar::random(&mut rng);
        let left = G1Affine::from(-(G1Projective::generator() * s));
        let right = G1Affine::from(G1Projective::generator() * (s * x));
        assert!(verify_pairing_lines(&left, &right, &lines_x_h, &lines_h));

        // Invalid pair
        let bad_right = G1Affine::from(G1Projective::generator() * s);
        assert!(!verify_pairing_lines(
            &left, &bad_right, &lines_x_h, &lines_h
        ));

        // Identity edge cases
        assert!(verify_pairing_lines(
            &G1Affine::identity(),
            &G1Affine::identity(),
            &lines_x_h,
            &lines_h
        ));
        assert!(!verify_pairing_lines(
            &G1Affine::identity(),
            &right,
            &lines_x_h,
            &lines_h
        ));
        assert!(!verify_pairing_lines(
            &left,
            &G1Affine::identity(),
            &lines_x_h,
            &lines_h
        ));
    }

    #[test]
    fn blst_lagrange_msm_matches_monomial() {
        let mut rng = StdRng::seed_from_u64(0x1a6e_0001);
        for log_n in [2u32, 6, 8, 11] {
            let n = 1usize << log_n;
            let domain = EvaluationDomain::new(n).unwrap();
            let tau = BlsScalar::random(&mut rng);
            let mut pow = BlsScalar::one();
            let mut powers_g1 = Vec::with_capacity(n + 4);
            for _ in 0..n + 4 {
                powers_g1.push(G1Affine::from(G1Projective::generator() * pow));
                pow *= tau;
            }
            let mono = g1_affines_to_blst(&powers_g1);
            let lagr = g1_ifft_blst(&mono[..n], &domain);

            // Mix every scalar bucket class: 0, ±1, ±2..±15, ±16..±32767,
            // ±(2^63-1), 2^63, and full random 255-bit field elements.
            let edge_vals = [
                BlsScalar::zero(),
                BlsScalar::one(),
                -BlsScalar::one(),
                BlsScalar::from(2u64),
                -BlsScalar::from(2u64),
                BlsScalar::from(15u64),
                -BlsScalar::from(15u64),
                BlsScalar::from(16u64),
                -BlsScalar::from(16u64),
                BlsScalar::from(32767u64),
                -BlsScalar::from(32767u64),
                BlsScalar::from(32768u64),
                -BlsScalar::from(32768u64),
                BlsScalar::from((1u64 << 63) - 1),
                -BlsScalar::from((1u64 << 63) - 1),
                BlsScalar::from(1u64 << 63),
                -BlsScalar::from(1u64 << 63),
            ];

            let mut w0 = vec![BlsScalar::zero(); n];
            let mut w1 = vec![BlsScalar::zero(); n];
            let mut w2 = vec![BlsScalar::zero(); n];
            let mut w3 = vec![BlsScalar::zero(); n];
            for i in 0..n {
                w0[i] = edge_vals[i % edge_vals.len()];
                w1[i] = if i % 3 == 0 {
                    BlsScalar::random(&mut rng)
                } else {
                    BlsScalar::from((i % 7) as u64)
                };
                w2[i] = BlsScalar::zero();
                w3[i] = BlsScalar::random(&mut rng);
            }

            let b0: [BlsScalar; 3] =
                core::array::from_fn(|_| BlsScalar::random(&mut rng));
            let b1: [BlsScalar; 3] =
                core::array::from_fn(|_| BlsScalar::random(&mut rng));
            let b2: [BlsScalar; 2] =
                core::array::from_fn(|_| BlsScalar::random(&mut rng));
            let b3: [BlsScalar; 3] =
                core::array::from_fn(|_| BlsScalar::random(&mut rng));

            let blind = |evals: &[BlsScalar], blinders: &[BlsScalar]| {
                let mut coeffs = domain.ifft(evals);
                for (j, &b) in blinders.iter().enumerate() {
                    coeffs[j] -= b;
                    coeffs.push(b);
                }
                coeffs
            };
            let p0 = blind(&w0, &b0);
            let p1 = blind(&w1, &b1);
            let p2 = blind(&w2, &b2);
            let p3 = blind(&w3, &b3);

            let expected = msm_g1_batch(&mono, [&p0, &p1, &p2, &p3]);
            let got = msm_g1_lagrange_batch(
                &lagr,
                &mono,
                [&w0, &w1, &w2, &w3],
                [&b0[..], &b1[..], &b2[..], &b3[..]],
            );
            assert_eq!(got, expected, "lagrange mismatch log_n={log_n}");
        }
    }

    #[test]
    fn blst_batched_affine_msm_matches_pippenger_including_collisions() {
        let mut rng = StdRng::seed_from_u64(0xaff1_0001);
        for n in [64usize, 257, 1024, 4096] {
            let mut pts: Vec<G1Affine> = (0..n)
                .map(|_| {
                    G1Affine::from(
                        G1Projective::generator() * BlsScalar::random(&mut rng),
                    )
                })
                .collect();
            // Inject repeated bases and identity points to exercise doubling
            // (`P + P`) and cancellation (`P + (-P)`) edge cases in affine
            // batching!
            pts[1] = pts[0];
            pts[2] = pts[0];
            pts[3] = G1Affine::identity();

            let mut s0: Vec<BlsScalar> =
                (0..n).map(|_| BlsScalar::random(&mut rng)).collect();
            let mut s1: Vec<BlsScalar> =
                (0..n).map(|_| BlsScalar::random(&mut rng)).collect();
            // Force exact doubling (`s[0] == s[1]`) and exact cancellation
            // (`s[2] == -s[0]`) on identical base `pts[0]`.
            s0[1] = s0[0];
            s0[2] = -s0[0];
            s1[0] = BlsScalar::one();
            s1[1] = BlsScalar::one();
            s1[2] = -BlsScalar::one();

            let blst_pts = g1_affines_to_blst(&pts);
            let expected = msm_g1_batch(&blst_pts, [&s0, &s1]);
            for w in [None, Some(8), Some(11), Some(13)] {
                let got = msm_g1_batch_affine(&blst_pts, [&s0, &s1], w);
                assert_eq!(got, expected, "n={n}, w={w:?}");
            }
        }
    }

    #[test]
    fn blst_fr_invert_and_glv_match_reference() {
        let mut rng = StdRng::seed_from_u64(0x6176_0001);
        let x_sq_scalar = BlsScalar::from_raw([GLV_X_SQ_LO, GLV_X_SQ_HI, 0, 0]);
        let edge_scalars = [
            BlsScalar::one(),
            -BlsScalar::one(),
            x_sq_scalar,
            x_sq_scalar - BlsScalar::one(),
            x_sq_scalar + BlsScalar::one(),
            -x_sq_scalar,
        ];
        for &s in &edge_scalars {
            let inv = fr_invert_nonzero(&s);
            assert_eq!(inv * s, BlsScalar::one());
            assert_eq!(inv, s.invert().unwrap());
        }
        for _ in 0..64 {
            let s = BlsScalar::random(&mut rng);
            if s != BlsScalar::zero() {
                assert_eq!(fr_invert_nonzero(&s), s.invert().unwrap());
            }
        }

        // Test GLV Straus and GLV 129-bit Pippenger across small and large sizes.
        for n in [1usize, 2, 11, 16, 24, 64, 176] {
            let pts: Vec<G1Affine> = (0..n)
                .map(|_| {
                    G1Affine::from(
                        G1Projective::generator() * BlsScalar::random(&mut rng),
                    )
                })
                .collect();
            let mut scs: Vec<BlsScalar> =
                (0..n).map(|_| BlsScalar::random(&mut rng)).collect();
            for (i, &es) in edge_scalars.iter().enumerate() {
                if i < n {
                    scs[i] = es;
                }
            }
            let blst_pts = g1_affines_to_blst(&pts);
            let sbytes = scalars_to_le_bytes_vec(&scs);
            let expected = G1Affine::from(msm_variable_base(&pts, &scs));
            if n <= 24 {
                assert_eq!(
                    blst_p1_to_g1_affine(&glv_straus_blst(&blst_pts, &sbytes)),
                    expected,
                    "glv_straus mismatch n={n}"
                );
            }
            assert_eq!(
                blst_p1_to_g1_affine(&msm_variable_base_proj(&pts, &scs)),
                expected,
                "msm_variable_base_proj mismatch n={n}"
            );
        }
    }

    #[test]
    fn blst_static_comb_table_matches_reference() {
        let mut rng = StdRng::seed_from_u64(0xc0ab_0001);
        let mut pts: [G1Affine; STATIC_POINT_COUNT] =
            core::array::from_fn(|_| {
                G1Affine::from(
                    G1Projective::generator() * BlsScalar::random(&mut rng),
                )
            });
        // Include identity slots to exercise `active_mask` skipping.
        pts[2] = G1Affine::identity();
        pts[9] = G1Affine::identity();
        pts[15] = G1Affine::identity();

        let tbl = BlstStaticVerifierTable::new(&pts);
        for trial in 0..8 {
            let mut scs: [BlsScalar; STATIC_POINT_COUNT] =
                core::array::from_fn(|_| BlsScalar::random(&mut rng));
            if trial == 0 {
                scs.fill(BlsScalar::zero());
            } else if trial == 1 {
                scs[0] = BlsScalar::zero();
                scs[7] = -BlsScalar::one();
                scs[8] = BlsScalar::one();
            }
            let expected = G1Affine::from(msm_variable_base(&pts, &scs));
            assert_eq!(
                blst_p1_to_g1_affine(&tbl.eval_serial(&scs)),
                expected,
                "comb table mismatch trial={trial}"
            );
        }

        // All-identity static table edge case.
        let empty_tbl = BlstStaticVerifierTable::new(
            &[G1Affine::identity(); STATIC_POINT_COUNT],
        );
        let scs: [BlsScalar; STATIC_POINT_COUNT] =
            core::array::from_fn(|_| BlsScalar::random(&mut rng));
        assert_eq!(
            blst_p1_to_g1_affine(&empty_tbl.eval_serial(&scs)),
            G1Affine::identity()
        );
    }

    #[test]
    fn blst_pipelined_verify_matches_split() {
        let mut rng = StdRng::seed_from_u64(0x919e_0001);
        let x = BlsScalar::random(&mut rng);
        let h = G2Affine::generator();
        let x_h = G2Affine::from(G2Projective::generator() * x);
        let lines_h = precompute_g2_lines(&h);
        let lines_x_h = precompute_g2_lines(&x_h);

        let static_pts: [G1Affine; STATIC_POINT_COUNT] =
            core::array::from_fn(|i| {
                if i == 3 {
                    G1Affine::identity()
                } else {
                    G1Affine::from(
                        G1Projective::generator() * BlsScalar::random(&mut rng),
                    )
                }
            });
        let static_tbl = BlstStaticVerifierTable::new(&static_pts);
        let static_scs: [BlsScalar; STATIC_POINT_COUNT] =
            core::array::from_fn(|_| BlsScalar::random(&mut rng));
        let mut dyn_pts: [G1Affine; DYNAMIC_POINT_COUNT] =
            core::array::from_fn(|_| {
                G1Affine::from(
                    G1Projective::generator() * BlsScalar::random(&mut rng),
                )
            });
        let mut dyn_scs: [BlsScalar; DYNAMIC_POINT_COUNT] =
            core::array::from_fn(|_| BlsScalar::random(&mut rng));
        let w_z = G1Affine::from(
            G1Projective::generator() * BlsScalar::random(&mut rng),
        );
        let w_zw = G1Affine::from(
            G1Projective::generator() * BlsScalar::random(&mut rng),
        );
        let u = BlsScalar::random(&mut rng);

        // Adjust the last dynamic slot so `right == x * (w_z + u * w_zw)`,
        // satisfying `e(-(w_z + u * w_zw), x * H) * e(right, H) == 1`.
        let neg_left_proj =
            G1Projective::from(w_z) + G1Projective::from(w_zw) * u;
        let target_right = neg_left_proj * x;
        let partial_right = msm_variable_base(&static_pts, &static_scs)
            + msm_variable_base(
                &dyn_pts[..DYNAMIC_POINT_COUNT - 1],
                &dyn_scs[..DYNAMIC_POINT_COUNT - 1],
            );
        dyn_pts[DYNAMIC_POINT_COUNT - 1] =
            G1Affine::from(target_right - partial_right);
        dyn_scs[DYNAMIC_POINT_COUNT - 1] = BlsScalar::one();

        assert!(verify_single_pipelined_blst(
            &static_tbl,
            &static_scs,
            &dyn_pts,
            &dyn_scs,
            &w_z,
            &w_zw,
            &u,
            &lines_x_h,
            &lines_h,
        ));
        assert!(verify_batch_pipelined_blst(
            &[w_z, w_zw],
            &[BlsScalar::one(), u],
            &dyn_pts,
            &dyn_scs,
            Some((&static_tbl, &static_scs)),
            &lines_x_h,
            &lines_h,
        ));

        // Perturb scalar -> both must reject.
        let mut bad_dyn_scs = dyn_scs;
        bad_dyn_scs[0] += BlsScalar::one();
        assert!(!verify_single_pipelined_blst(
            &static_tbl,
            &static_scs,
            &dyn_pts,
            &bad_dyn_scs,
            &w_z,
            &w_zw,
            &u,
            &lines_x_h,
            &lines_h,
        ));
        assert!(!verify_batch_pipelined_blst(
            &[w_z, w_zw],
            &[BlsScalar::one(), u],
            &dyn_pts,
            &bad_dyn_scs,
            Some((&static_tbl, &static_scs)),
            &lines_x_h,
            &lines_h,
        ));

        // Identity edge cases: both identity -> accept; only one -> reject.
        let zero_static = [BlsScalar::zero(); STATIC_POINT_COUNT];
        let zero_dyn = [BlsScalar::zero(); DYNAMIC_POINT_COUNT];
        assert!(verify_single_pipelined_blst(
            &static_tbl,
            &zero_static,
            &dyn_pts,
            &zero_dyn,
            &G1Affine::identity(),
            &G1Affine::identity(),
            &u,
            &lines_x_h,
            &lines_h,
        ));
        assert!(!verify_single_pipelined_blst(
            &static_tbl,
            &static_scs,
            &dyn_pts,
            &dyn_scs,
            &G1Affine::identity(),
            &G1Affine::identity(),
            &u,
            &lines_x_h,
            &lines_h,
        ));
        assert!(verify_batch_pipelined_blst(
            &[],
            &[],
            &[],
            &[],
            None,
            &lines_x_h,
            &lines_h,
        ));
        assert!(!verify_batch_pipelined_blst(
            &[w_z],
            &[BlsScalar::one()],
            &[],
            &[],
            None,
            &lines_x_h,
            &lines_h,
        ));
    }
}
