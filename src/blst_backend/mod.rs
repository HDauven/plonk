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

mod msm;
mod msm_affine;
mod msm_lagrange;
mod verify;

#[cfg(test)]
mod tests;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::ptr;

use blst::{
    BLST_ERROR, blst_fp, blst_fp6, blst_fr, blst_fr_eucl_inverse, blst_p1,
    blst_p1_affine, blst_p1_affine_in_g1, blst_p1_affine_is_inf,
    blst_p1_to_affine, blst_p1_uncompress, blst_p1s_to_affine, blst_p2_affine,
    blst_p2_uncompress, blst_precompute_lines, blst_scalar,
    blst_scalar_from_fr,
};
use dusk_bls12_381::{BlsScalar, G1Affine, G2Affine};
pub(crate) use msm::{msm_g1, msm_g1_batch, msm_g1_variable_base};
pub(crate) use msm_affine::msm_g1_batch_affine;
pub(crate) use msm_lagrange::{g1_ifft_blst, msm_g1_lagrange_batch};
use rayon::prelude::*;
#[cfg(test)]
pub(crate) use verify::verify_pairing_lines;
pub(crate) use verify::{
    BlstStaticVerifierTable, verifier_msm_single_blst,
    verify_batch_pipelined_blst, verify_single_pipelined_blst,
};

// BLS12-381 scalars satisfy `s < r < 2^255`, so bit 255 is always 0.
// Passing 256 bits guarantees zero terminal Booth carry for every Pippenger
// window width (including divisors of 255 and 256).
const SCALAR_NBITS: usize = 256;
const SCALAR_BYTES: usize = 32;

std::thread_local! {
    static PIPPENGER_SCRATCH: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
    static AFFINE_SCRATCH: RefCell<msm_affine::AffinePippengerScratch> =
        RefCell::new(msm_affine::AffinePippengerScratch::default());
}

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
