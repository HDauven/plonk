// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! Multi-scalar multiplication (MSM) primitives, tiled Pippenger scheduling,
//! and shared batch helpers backed by `blst`.

use alloc::vec::Vec;
use core::ptr;

use blst::{
    blst_p1, blst_p1_add_or_double, blst_p1_affine, blst_p1_double,
    blst_p1_from_affine, blst_p1_mult, blst_p1s_mult_pippenger,
    blst_p1s_mult_pippenger_scratch_sizeof, blst_p1s_tile_pippenger,
};
use dusk_bls12_381::{BlsScalar, G1Affine};
use rayon::prelude::*;

use super::{
    PIPPENGER_SCRATCH, SCALAR_BYTES, SCALAR_NBITS, blst_p1s_to_g1_affines,
    g1_affine_to_blst, msm_g1_batch_affine, scalar_to_blst_le_bytes,
    scalars_to_le_bytes_vec,
};

/// Descriptor for one Pippenger window tile in a batched MSM.
pub(super) struct TileSpec {
    pub(super) poly_idx: usize,
    pub(super) x: usize,
    pub(super) dx: usize,
    pub(super) w_idx: usize,
    pub(super) window: usize,
}

/// Partitioning metadata and flat tile list for a batch of `K` MSMs.
pub(super) struct BatchTilePlan<const K: usize> {
    pub(super) poly_windows: [usize; K],
    pub(super) poly_ny: [usize; K],
    pub(super) poly_nx: [usize; K],
    pub(super) tiles: Vec<TileSpec>,
}

/// Choose the optimal Pippenger window width for `n` points per tile,
/// matching `blst`'s `pippenger_window_size` heuristic so bucket integration
/// (`2^(w-1)` buckets) stays balanced against `n` and L2-cache-friendly on
/// hybrid P+E cores.
#[inline]
pub(super) fn pippenger_window(n: usize) -> usize {
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

/// Trim trailing zero scalars from each polynomial in the batch and verify
/// every effective length fits within `bases_len`.
pub(super) fn trimmed_batch_lengths<const K: usize>(
    bases_len: usize,
    polys: &[&[BlsScalar]; K],
) -> [usize; K] {
    let mut lengths = [0usize; K];
    for (i, poly) in polys.iter().enumerate() {
        let mut len = poly.len();
        while len > 0 && poly[len - 1] == BlsScalar::zero() {
            len -= 1;
        }
        assert!(
            len <= bases_len,
            "MSM polynomial length exceeds CommitKey bases"
        );
        lengths[i] = len;
    }
    lengths
}

/// Convert all active polynomials' scalars into packed 32-byte LE buffers in
/// one flat parallel pass (or serially for small total counts).
pub(super) fn pack_batch_scalars<const K: usize>(
    polys: &[&[BlsScalar]; K],
    lengths: &[usize; K],
) -> Vec<Vec<u8>> {
    let mut scalar_bufs: Vec<Vec<u8>> = lengths
        .iter()
        .map(|&len| vec![0u8; len * SCALAR_BYTES])
        .collect();
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
    scalar_bufs
}

/// Build a flat list of `(poly_idx, x_start, x_len, window_idx, window_bits)`
/// tiles across all `K` polynomials so every core stays saturated without
/// nested Rayon barriers.
pub(super) fn plan_batch_tiles<const K: usize>(
    lengths: &[usize; K],
    num_threads: usize,
    choose_w: impl Fn(usize) -> usize,
) -> BatchTilePlan<K> {
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

    BatchTilePlan {
        poly_windows,
        poly_ny,
        poly_nx,
        tiles,
    }
}

/// Evaluate a single Pippenger window tile using `blst_p1s_tile_pippenger` and
/// the thread-local `PIPPENGER_SCRATCH` workspace.
#[inline]
pub(super) fn blst_tile_pippenger_raw(
    points: &[blst_p1_affine],
    scalar_bytes: &[u8],
    nbits: usize,
    bit0: usize,
    window: usize,
    tile_unit_u64: usize,
) -> blst_p1 {
    let dx = points.len();
    debug_assert!(
        dx >= 2,
        "blst_p1s_tile_pippenger requires at least 2 points"
    );
    debug_assert_eq!(scalar_bytes.len(), dx * nbits.div_ceil(8));
    PIPPENGER_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let need_u64 = tile_unit_u64 << (window - 1);
        if scratch.len() < need_u64 {
            scratch.resize(need_u64, 0);
        }
        let p_ptrs: [*const blst_p1_affine; 2] = [points.as_ptr(), ptr::null()];
        let s_ptrs: [*const u8; 2] = [scalar_bytes.as_ptr(), ptr::null()];
        let mut out = blst_p1::default();
        // SAFETY: `points` and `scalar_bytes` cover `dx` elements, and
        // `scratch` has `tile_unit_u64 << (window - 1)` limbs as required by
        // `blst_p1s_tile_pippenger`.
        unsafe {
            blst_p1s_tile_pippenger(
                &mut out,
                p_ptrs.as_ptr(),
                dx,
                s_ptrs.as_ptr(),
                nbits,
                scratch.as_mut_ptr(),
                bit0,
                window,
            );
        }
        out
    })
}

/// Reduce a row-major `ny x nx` grid of Pippenger window-tile results from the
/// highest window down to 0.
#[inline]
pub(super) fn reduce_window_tiles(
    slice: &[blst_p1],
    window: usize,
    ny: usize,
    nx: usize,
) -> blst_p1 {
    debug_assert_eq!(slice.len(), ny * nx);
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
    acc
}

/// Reduce per-polynomial Pippenger tile results (plus any short `< 64`
/// polynomials evaluated serially) into `K` Jacobian accumulators.
pub(super) fn reduce_batch_tiles<const K: usize>(
    bases: &[blst_p1_affine],
    scalar_bufs: &[Vec<u8>],
    lengths: &[usize; K],
    plan: &BatchTilePlan<K>,
    tile_results: &[blst_p1],
) -> [blst_p1; K] {
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
        let window = plan.poly_windows[poly_idx];
        let ny = plan.poly_ny[poly_idx];
        let nx = plan.poly_nx[poly_idx];
        let slice = &tile_results[cursor..cursor + ny * nx];
        cursor += ny * nx;
        projs[poly_idx] = reduce_window_tiles(slice, window, ny, nx);
    }
    projs
}

/// Evaluate a batch of `K` MSMs serially and batch-normalize the results.
pub(super) fn msm_g1_batch_serial<const K: usize>(
    bases: &[blst_p1_affine],
    polys: &[&[BlsScalar]; K],
    lengths: &[usize; K],
) -> [G1Affine; K] {
    let mut projs = [blst_p1::default(); K];
    for i in 0..K {
        let len = lengths[i];
        if len > 0 {
            let sbytes = scalars_to_le_bytes_vec(&polys[i][..len]);
            projs[i] = blst_msm_serial_raw(&bases[..len], &sbytes);
        }
    }
    blst_p1s_to_g1_affines(&projs)
}

/// Run a single-threaded `blst` MSM on `points` and packed `nbytes`-per-scalar
/// LE `scalar_bytes` with bit-width `nbits`.
pub(super) fn blst_msm_serial_raw_nbits(
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

/// Run a single-threaded `blst` MSM on `points` and packed 32-byte LE
/// `scalars`.
pub(super) fn blst_msm_serial_raw(
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
    let lengths = trimmed_batch_lengths(bases.len(), &polys);

    let max_len = lengths.iter().copied().max().unwrap_or(0);
    if max_len == 0 {
        return [G1Affine::identity(); K];
    }

    let num_threads = rayon::current_num_threads();
    if num_threads > 1 && max_len >= 1024 {
        return msm_g1_batch_affine(bases, polys, None);
    }
    if num_threads <= 1 || max_len < 64 {
        return msm_g1_batch_serial(bases, &polys, &lengths);
    }

    let scalar_bufs = pack_batch_scalars(&polys, &lengths);
    let plan = plan_batch_tiles(&lengths, num_threads, pippenger_window);

    // SAFETY: pure size query for one tile bucket unit.
    let tile_unit_u64 =
        unsafe { blst_p1s_mult_pippenger_scratch_sizeof(0) }.div_ceil(8);

    let tile_results: Vec<blst_p1> = plan
        .tiles
        .par_iter()
        .map(|tile| {
            let pts = &bases[tile.x..tile.x + tile.dx];
            let scs = &scalar_bufs[tile.poly_idx]
                [tile.x * SCALAR_BYTES..(tile.x + tile.dx) * SCALAR_BYTES];
            blst_tile_pippenger_raw(
                pts,
                scs,
                SCALAR_NBITS,
                tile.w_idx * tile.window,
                tile.window,
                tile_unit_u64,
            )
        })
        .collect();

    let projs =
        reduce_batch_tiles(bases, &scalar_bufs, &lengths, &plan, &tile_results);
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
