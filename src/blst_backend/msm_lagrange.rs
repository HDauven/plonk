// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! Lagrange-basis SRS conversion (`g1_ifft_blst`) and sparse/bit-width
//! bucketed Lagrange MSM commitment (`msm_g1_lagrange_batch`).

use alloc::vec::Vec;
use core::ptr;

use blst::{
    blst_fp_cneg, blst_p1, blst_p1_add_or_double, blst_p1_add_or_double_affine,
    blst_p1_affine, blst_p1_from_affine, blst_p1_mult,
    blst_p1s_mult_pippenger_scratch_sizeof, blst_p1s_to_affine,
};
use dusk_bls12_381::{BlsScalar, G1Affine};
use rayon::prelude::*;

use super::msm::{
    blst_msm_serial_raw_nbits, blst_tile_pippenger_raw, pippenger_window,
    reduce_window_tiles,
};
use super::msm_affine::{batched_affine_tile_pippenger, choose_affine_window};
use super::{
    SCALAR_BYTES, SCALAR_NBITS, blst_p1s_to_g1_affines, scalar_to_blst_le_bytes,
};
use crate::fft::EvaluationDomain;

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
                        let pts = &job.pts[tile.x..tile.x + tile.dx];
                        let scs = &job.scs[tile.x * job.nbytes
                            ..(tile.x + tile.dx) * job.nbytes];
                        if job.nbits == SCALAR_NBITS && tile.dx >= 1024 {
                            let eff_bits =
                                255usize.saturating_sub(bit0).min(tile.window);
                            if eff_bits >= 10 {
                                return batched_affine_tile_pippenger(
                                    pts,
                                    scs,
                                    bit0,
                                    tile.window,
                                );
                            }
                        }
                        blst_tile_pippenger_raw(
                            pts,
                            scs,
                            job.nbits,
                            bit0,
                            tile.window,
                            tile_unit_u64,
                        )
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
        let acc = reduce_window_tiles(slice, window, ny, nx);
        let w = jobs[job_idx].wire_idx;
        // SAFETY: valid Jacobian points.
        unsafe { blst_p1_add_or_double(&mut projs[w], &projs[w], &acc) };
    }

    blst_p1s_to_g1_affines(&projs)
}
