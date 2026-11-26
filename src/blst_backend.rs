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
    BLST_ERROR, blst_fp, blst_fp_cneg, blst_fp6, blst_fp12,
    blst_fp12_finalverify, blst_fr, blst_miller_loop_lines, blst_p1,
    blst_p1_add_or_double, blst_p1_add_or_double_affine, blst_p1_affine,
    blst_p1_affine_in_g1, blst_p1_affine_is_inf, blst_p1_double,
    blst_p1_from_affine, blst_p1_mult, blst_p1_to_affine, blst_p1_uncompress,
    blst_p1s_mult_pippenger, blst_p1s_mult_pippenger_scratch_sizeof,
    blst_p1s_mult_wbits, blst_p1s_mult_wbits_precompute,
    blst_p1s_mult_wbits_precompute_sizeof, blst_p1s_mult_wbits_scratch_sizeof,
    blst_p1s_tile_pippenger, blst_p1s_to_affine, blst_p2_affine,
    blst_p2_uncompress, blst_precompute_lines, blst_scalar,
    blst_scalar_from_fr,
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
    static WBITS_SCRATCH: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
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

    let (ml1, ml2) = if rayon::current_num_threads() > 1 {
        rayon::join(
            || {
                let mut ml = blst_fp12::default();
                // SAFETY: `lines_x_h` has 68 entries and `neg_left` is
                // non-identity.
                unsafe {
                    blst_miller_loop_lines(
                        &mut ml,
                        lines_x_h.as_ptr(),
                        &neg_left,
                    )
                };
                ml
            },
            || {
                let mut ml = blst_fp12::default();
                // SAFETY: `lines_h` has 68 entries and `right_blst` is
                // non-identity.
                unsafe {
                    blst_miller_loop_lines(
                        &mut ml,
                        lines_h.as_ptr(),
                        &right_blst,
                    )
                };
                ml
            },
        )
    } else {
        let mut ml1 = blst_fp12::default();
        let mut ml2 = blst_fp12::default();
        // SAFETY: valid precomputed 68-line tables and non-identity G1 points.
        unsafe {
            blst_miller_loop_lines(&mut ml1, lines_x_h.as_ptr(), &neg_left);
            blst_miller_loop_lines(&mut ml2, lines_h.as_ptr(), &right_blst);
        }
        (ml1, ml2)
    };

    // SAFETY: checks `final_exp(conjugate(ml1) * ml2) == 1`.
    unsafe { blst_fp12_finalverify(&ml1, &ml2) }
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

/// Fixed-base `blst` `wbits` precomputed table for the 16 static commitments on
/// a `Verifier`.
///
/// With `wbits = 6`, each active static point uses 32 affine points (~3 KiB),
/// totaling at most ~48 KiB per `Verifier` (L1/L2 resident) and reducing
/// static MSM time to ~40 µs in ADX/BMI2 assembly.
#[derive(Clone, Debug)]
pub(crate) struct BlstStaticVerifierTable {
    slots: [Option<u8>; STATIC_POINT_COUNT],
    active_count: usize,
    table: Vec<blst_p1_affine>,
    scratch_u64_len: usize,
}

const VERIFIER_WBITS: usize = 8;

impl BlstStaticVerifierTable {
    pub(crate) fn new(points: &[G1Affine; STATIC_POINT_COUNT]) -> Self {
        let mut slots = [None; STATIC_POINT_COUNT];
        let mut active_pts = Vec::with_capacity(STATIC_POINT_COUNT);

        for (slot_idx, pt) in points.iter().enumerate() {
            if !bool::from(pt.is_identity()) {
                slots[slot_idx] = Some(active_pts.len() as u8);
                active_pts.push(g1_affine_to_blst(pt));
            }
        }

        let active_count = active_pts.len();
        if active_count == 0 {
            return Self {
                slots,
                active_count: 0,
                table: Vec::new(),
                scratch_u64_len: 0,
            };
        }

        // SAFETY: pure size queries.
        let table_bytes = unsafe {
            blst_p1s_mult_wbits_precompute_sizeof(VERIFIER_WBITS, active_count)
        };
        let table_len =
            table_bytes.div_ceil(core::mem::size_of::<blst_p1_affine>());
        let mut table = vec![blst_p1_affine::default(); table_len];

        let p_ptrs: [*const blst_p1_affine; 2] =
            [active_pts.as_ptr(), ptr::null()];
        // SAFETY: `table` is sized by `blst_p1s_mult_wbits_precompute_sizeof`,
        // `p_ptrs` is a contiguous-array sentinel for `active_count` points.
        unsafe {
            blst_p1s_mult_wbits_precompute(
                table.as_mut_ptr(),
                VERIFIER_WBITS,
                p_ptrs.as_ptr(),
                active_count,
            );
        }

        // SAFETY: pure size query.
        let scratch_u64_len =
            unsafe { blst_p1s_mult_wbits_scratch_sizeof(active_count) }
                .div_ceil(8);

        Self {
            slots,
            active_count,
            table,
            scratch_u64_len,
        }
    }

    /// Evaluate the static linear combination using the precomputed `wbits`
    /// table.
    fn eval(&self, scalars: &[BlsScalar; STATIC_POINT_COUNT]) -> blst_p1 {
        if self.active_count == 0 {
            return blst_p1::default();
        }
        let mut sbytes = [0u8; STATIC_POINT_COUNT * SCALAR_BYTES];
        let mut any_nonzero = false;
        for (slot_idx, sc) in scalars.iter().enumerate() {
            if let Some(active_idx) = self.slots[slot_idx]
                && *sc != BlsScalar::zero()
            {
                any_nonzero = true;
                let start = (active_idx as usize) * SCALAR_BYTES;
                sbytes[start..start + SCALAR_BYTES]
                    .copy_from_slice(&scalar_to_blst_le_bytes(sc));
            }
        }
        if !any_nonzero {
            return blst_p1::default();
        }

        WBITS_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            if scratch.len() < self.scratch_u64_len {
                scratch.resize(self.scratch_u64_len, 0);
            }
            let s_ptrs: [*const u8; 2] = [sbytes.as_ptr(), ptr::null()];
            let mut out = blst_p1::default();
            // SAFETY: `self.table` and `scratch` were sized by the matching
            // `blst_p1s_mult_wbits_*_sizeof` functions for `self.active_count`.
            unsafe {
                blst_p1s_mult_wbits(
                    &mut out,
                    self.table.as_ptr(),
                    VERIFIER_WBITS,
                    self.active_count,
                    s_ptrs.as_ptr(),
                    SCALAR_NBITS,
                    scratch.as_mut_ptr(),
                );
            }
            out
        })
    }
}

/// Compute the two affine pairing inputs `(left, right)` for single-proof PLONK
/// verification using `blst`'s precomputed `wbits` table + Pippenger assembly.
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
        let mut wz_proj = blst_p1::default();
        // SAFETY: valid point pointers.
        unsafe { blst_p1_from_affine(&mut wz_proj, &wz_aff) };

        let mut acc = wz_proj;
        if !bool::from(w_zw.is_identity()) && *u_challenge != BlsScalar::zero()
        {
            let wzw_aff = g1_affine_to_blst(w_zw);
            let mut wzw_proj = blst_p1::default();
            let u_bytes = scalar_to_blst_le_bytes(u_challenge);
            // SAFETY: valid point and 32-byte scalar pointers.
            unsafe {
                blst_p1_from_affine(&mut wzw_proj, &wzw_aff);
                blst_p1_mult(
                    &mut wzw_proj,
                    &wzw_proj,
                    u_bytes.as_ptr(),
                    SCALAR_NBITS,
                );
                blst_p1_add_or_double(&mut acc, &acc, &wzw_proj);
            }
        }
        // Negate to produce `-(w_z + u * w_zw)`.
        // SAFETY: negates Jacobian y coordinate in place.
        unsafe { blst_fp_cneg(&mut acc.y, &acc.y, true) };
        acc
    };

    let compute_dynamic = || -> blst_p1 {
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
        blst_msm_serial_raw(&pts[..count], &sbytes[..count * SCALAR_BYTES])
    };

    let (left_proj, right_proj) = if rayon::current_num_threads() > 1 {
        // Balance 3 concurrent tasks: (left + static wbits) vs dynamic half A
        // vs dynamic half B.
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
        let mid = count / 2;
        let ((left, static_sum), (dyn_a, dyn_b)) = rayon::join(
            || {
                let l = compute_left();
                let s = static_table.eval(static_scalars);
                (l, s)
            },
            || {
                if count >= 6 {
                    rayon::join(
                        || {
                            blst_msm_serial_raw(
                                &pts[..mid],
                                &sbytes[..mid * SCALAR_BYTES],
                            )
                        },
                        || {
                            blst_msm_serial_raw(
                                &pts[mid..count],
                                &sbytes
                                    [mid * SCALAR_BYTES..count * SCALAR_BYTES],
                            )
                        },
                    )
                } else {
                    (
                        blst_msm_serial_raw(
                            &pts[..count],
                            &sbytes[..count * SCALAR_BYTES],
                        ),
                        blst_p1::default(),
                    )
                }
            },
        );
        let mut right = static_sum;
        // SAFETY: valid Jacobian points.
        unsafe {
            blst_p1_add_or_double(&mut right, &right, &dyn_a);
            blst_p1_add_or_double(&mut right, &right, &dyn_b);
        }
        (left, right)
    } else {
        let left = compute_left();
        let mut right = static_table.eval(static_scalars);
        let dyn_sum = compute_dynamic();
        // SAFETY: valid Jacobian points.
        unsafe { blst_p1_add_or_double(&mut right, &right, &dyn_sum) };
        (left, right)
    };

    let [left_aff, right_aff] =
        blst_p1s_to_g1_affines(&[left_proj, right_proj]);
    (left_aff, right_aff)
}

/// Evaluate a variable-base `blst` MSM returning Jacobian `blst_p1`, tiled
/// across Rayon when large enough.
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
    if n < 64 || rayon::current_num_threads() <= 1 {
        return blst_msm_serial_raw(&blst_pts, &sbytes);
    }

    let window = pippenger_window(n);
    let ny = SCALAR_NBITS.div_ceil(window);
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
                    [blst_pts.as_ptr(), ptr::null()];
                let s_ptrs: [*const u8; 2] = [sbytes.as_ptr(), ptr::null()];
                let mut out = blst_p1::default();
                // SAFETY: valid pointers and sized scratch.
                unsafe {
                    blst_p1s_tile_pippenger(
                        &mut out,
                        p_ptrs.as_ptr(),
                        n,
                        s_ptrs.as_ptr(),
                        SCALAR_NBITS,
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

/// Compute the batched verifier pairing inputs `(left = -MSM(left_pts), right =
/// MSM(right_pts) [+ static_table])` concurrently in one parallel pass with a
/// single batched affine normalization.
pub(crate) fn verify_batch_msm_blst(
    left_pts: &[G1Affine],
    left_scs: &[BlsScalar],
    right_pts: &[G1Affine],
    right_scs: &[BlsScalar],
    static_part: Option<(
        &BlstStaticVerifierTable,
        &[BlsScalar; STATIC_POINT_COUNT],
    )>,
) -> (G1Affine, G1Affine) {
    let (mut left_proj, right_proj) = if rayon::current_num_threads() > 1 {
        let ((l, s_opt), r) = rayon::join(
            || {
                let l = msm_variable_base_proj(left_pts, left_scs);
                let s = static_part.map(|(tbl, scs)| tbl.eval(scs));
                (l, s)
            },
            || msm_variable_base_proj(right_pts, right_scs),
        );
        let mut r = r;
        if let Some(s) = s_opt {
            // SAFETY: valid Jacobian points.
            unsafe { blst_p1_add_or_double(&mut r, &r, &s) };
        }
        (l, r)
    } else {
        let l = msm_variable_base_proj(left_pts, left_scs);
        let mut r = msm_variable_base_proj(right_pts, right_scs);
        if let Some((tbl, scs)) = static_part {
            let s = tbl.eval(scs);
            // SAFETY: valid Jacobian points.
            unsafe { blst_p1_add_or_double(&mut r, &r, &s) };
        }
        (l, r)
    };

    // Negate `left` in Jacobian form before normalizing both together.
    // SAFETY: negates Jacobian y coordinate in place.
    unsafe { blst_fp_cneg(&mut left_proj.y, &left_proj.y, true) };
    let [left_aff, right_aff] =
        blst_p1s_to_g1_affines(&[left_proj, right_proj]);
    (left_aff, right_aff)
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
        let window = pippenger_window(len).min(job.nbits);
        let ny = job.nbits.div_ceil(window);
        let nx = if len >= 4096 && ny < num_threads {
            (num_threads.div_ceil(ny)).min(len / 2048).max(1)
        } else {
            1
        };
        let window = pippenger_window(len.div_ceil(nx)).min(job.nbits);
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
                                    tile.w_idx * tile.window,
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
        for log_n in [2u32, 6, 8] {
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
}
