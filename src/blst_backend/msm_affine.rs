// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! Batched-Affine Pippenger MSM over `blst_fp` ADX/BMI2 assembly.

use alloc::vec::Vec;

use blst::{
    blst_fp, blst_fp_cneg, blst_fp_eucl_inverse, blst_fp_mul, blst_fp_sqr,
    blst_fp_sub, blst_p1, blst_p1_add_or_double, blst_p1_add_or_double_affine,
    blst_p1_affine, blst_p1_double, blst_p1_from_affine, blst_p1_to_affine,
    blst_p1s_mult_pippenger_scratch_sizeof,
};
use dusk_bls12_381::{BlsScalar, G1Affine};
use rayon::prelude::*;

use super::msm::{
    blst_tile_pippenger_raw, msm_g1_batch_serial, pack_batch_scalars,
    plan_batch_tiles, reduce_batch_tiles, trimmed_batch_lengths,
};
use super::{
    AFFINE_SCRATCH, FP_ONE_MONT, SCALAR_BYTES, SCALAR_NBITS,
    blst_p1s_to_g1_affines, fp_eq, fp_is_zero,
};

const AFFINE_BATCH_SIZE: usize = 256;

/// Reusable per-thread workspace for Batched-Affine Pippenger.
pub(super) struct AffinePippengerScratch {
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
pub(super) fn batched_affine_tile_pippenger(
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
pub(super) fn choose_affine_window(n: usize) -> usize {
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
    let lengths = trimmed_batch_lengths(bases.len(), &polys);

    let max_len = lengths.iter().copied().max().unwrap_or(0);
    if max_len == 0 {
        return [G1Affine::identity(); K];
    }

    let num_threads = rayon::current_num_threads();
    if num_threads <= 1 || max_len < 64 {
        return msm_g1_batch_serial(bases, &polys, &lengths);
    }

    let scalar_bufs = pack_batch_scalars(&polys, &lengths);
    let choose_w = |n: usize| -> usize {
        window_override
            .map(|w| w.clamp(2, 15))
            .unwrap_or_else(|| choose_affine_window(n))
    };
    let plan = plan_batch_tiles(&lengths, num_threads, choose_w);

    // SAFETY: pure size query for one XYZZ tile bucket unit (used for narrow
    // top partial windows where `eff_bits < 10` would cause heavy affine
    // bucket collisions).
    let tile_unit_u64 =
        unsafe { blst_p1s_mult_pippenger_scratch_sizeof(0) }.div_ceil(8);

    let tile_results: Vec<blst_p1> = plan
        .tiles
        .par_iter()
        .map(|tile| {
            let bit0 = tile.w_idx * tile.window;
            let eff_bits = 255usize.saturating_sub(bit0).min(tile.window);
            let pts = &bases[tile.x..tile.x + tile.dx];
            let scs = &scalar_bufs[tile.poly_idx]
                [tile.x * SCALAR_BYTES..(tile.x + tile.dx) * SCALAR_BYTES];
            if eff_bits < 10 {
                return blst_tile_pippenger_raw(
                    pts,
                    scs,
                    SCALAR_NBITS,
                    bit0,
                    tile.window,
                    tile_unit_u64,
                );
            }
            batched_affine_tile_pippenger(pts, scs, bit0, tile.window)
        })
        .collect();

    let projs =
        reduce_batch_tiles(bases, &scalar_bufs, &lengths, &plan, &tile_results);
    blst_p1s_to_g1_affines(&projs)
}
