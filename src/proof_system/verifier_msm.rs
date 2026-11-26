// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

#![forbid(unsafe_code)]

//! Variable-time G1 multi-scalar multiplication optimized for PLONK proof
//! verification.
//!
//! # Timing
//!
//! Every routine in this module operates exclusively on **public** verifier
//! challenges and claimed proof evaluations. They are intentionally
//! variable-time with respect to scalars and points and must never be called
//! with secret witnesses or blinding scalars.

use alloc::vec::Vec;

use dusk_bls12_381::{BlsScalar, G1Affine, G1Projective};

/// Number of static commitments per [`crate::compiler::Verifier`]:
/// 15 verifier-key commitments plus `opening_key.g`.
pub(crate) const STATIC_POINT_COUNT: usize = 16;

/// Number of dynamic proof commitments contributing to the right pairing input.
pub(crate) const DYNAMIC_POINT_COUNT: usize = 11;

/// Odd-multiple table width for dynamic verifier points.
///
/// Under `std`, `w = 5` (8 affine odd multiples per base) minimizes mixed
/// additions in Straus. Under `no_std`, `w = 4` (4 affine odd multiples)
/// halves heap and stack footprint for constrained targets.
#[cfg(feature = "std")]
const DYNAMIC_WNAF_WIDTH: usize = 5;
#[cfg(feature = "std")]
const DYNAMIC_ODD_COUNT: usize = 1 << (DYNAMIC_WNAF_WIDTH - 2);

#[cfg(not(feature = "std"))]
const DYNAMIC_WNAF_WIDTH: usize = 4;
#[cfg(not(feature = "std"))]
const DYNAMIC_ODD_COUNT: usize = 1 << (DYNAMIC_WNAF_WIDTH - 2);

/// Odd-multiple table width for static `Verifier` commitments under `std`.
///
/// `w = 6` stores 16 affine odd multiples (`[1P, 3P, ..., 31P]`) per
/// non-identity static commitment (~1.6 KiB per active point, <= 26.6 KiB total
/// per `Verifier`).
#[cfg(feature = "std")]
const STATIC_WNAF_WIDTH: usize = 6;
#[cfg(feature = "std")]
const STATIC_ODD_COUNT: usize = 1 << (STATIC_WNAF_WIDTH - 2);

/// Precomputed odd-multiple tables for the 16 static commitments owned by a
/// [`crate::compiler::Verifier`].
///
/// Under `no_std`, this is zero-sized so embedded verifiers incur zero extra
/// memory overhead.
#[cfg(feature = "std")]
#[derive(Clone, Debug)]
pub(crate) struct StaticVerifierTables {
    /// Maps each static slot `0..16` to its index in `tables`, or `None` if
    /// the static commitment is the identity point.
    slots: [Option<u8>; STATIC_POINT_COUNT],
    /// Compact storage of `[1P, 3P, ..., 31P]` for each non-identity static
    /// point.
    tables: Vec<[G1Affine; STATIC_ODD_COUNT]>,
}

#[cfg(not(feature = "std"))]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StaticVerifierTables;

impl StaticVerifierTables {
    /// Precompute static odd-multiple tables for a `Verifier`'s 16 fixed
    /// commitments.
    #[cfg(feature = "std")]
    pub(crate) fn new(points: &[G1Affine; STATIC_POINT_COUNT]) -> Self {
        let mut slots = [None; STATIC_POINT_COUNT];
        let mut active_points = Vec::with_capacity(STATIC_POINT_COUNT);

        for (slot_idx, point) in points.iter().enumerate() {
            if !bool::from(point.is_identity()) {
                slots[slot_idx] = Some(active_points.len() as u8);
                active_points.push(*point);
            }
        }

        let tables =
            build_odd_tables_batched::<STATIC_ODD_COUNT>(&active_points);
        Self { slots, tables }
    }

    #[cfg(not(feature = "std"))]
    pub(crate) fn new(_points: &[G1Affine; STATIC_POINT_COUNT]) -> Self {
        Self
    }
}

/// Convert a `BlsScalar` to canonical little-endian 64-bit limbs.
#[inline]
fn scalar_to_le_limbs(scalar: &BlsScalar) -> [u64; 4] {
    let bytes = scalar.to_bytes();
    [
        u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5],
            bytes[6], bytes[7],
        ]),
        u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13],
            bytes[14], bytes[15],
        ]),
        u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21],
            bytes[22], bytes[23],
        ]),
        u64::from_le_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29],
            bytes[30], bytes[31],
        ]),
    ]
}

/// Compute signed `W`-bit wNAF representation of `scalar` into `digits`.
///
/// Returns the number of significant digits (`<= 256`). Nonzero entries are
/// odd integers in `[-(2^(W-1) - 1), 2^(W-1) - 1]`.
fn encode_wnaf<const W: usize>(
    scalar: &BlsScalar,
    digits: &mut [i8; 256],
) -> usize {
    debug_assert!((2..=7).contains(&W));
    digits.fill(0);

    let mut limbs = scalar_to_le_limbs(scalar);
    let width = 1i64 << W;
    let half = width >> 1;
    let mask = (width - 1) as u64;

    let mut i = 0usize;
    let mut len = 0usize;

    while (limbs[0] | limbs[1] | limbs[2] | limbs[3]) != 0 {
        if (limbs[0] & 1) == 1 {
            let mut d = (limbs[0] & mask) as i64;
            if d >= half {
                d -= width;
                let add = (-d) as u64;
                let (v0, c0) = limbs[0].overflowing_add(add);
                limbs[0] = v0;
                if c0 {
                    let (v1, c1) = limbs[1].overflowing_add(1);
                    limbs[1] = v1;
                    if c1 {
                        let (v2, c2) = limbs[2].overflowing_add(1);
                        limbs[2] = v2;
                        if c2 {
                            limbs[3] = limbs[3].wrapping_add(1);
                        }
                    }
                }
            } else {
                limbs[0] -= d as u64;
            }
            digits[i] = d as i8;
            len = i + 1;
        }

        limbs[0] = (limbs[0] >> 1) | (limbs[1] << 63);
        limbs[1] = (limbs[1] >> 1) | (limbs[2] << 63);
        limbs[2] = (limbs[2] >> 1) | (limbs[3] << 63);
        limbs[3] >>= 1;
        i += 1;
    }

    len
}

/// Build affine odd-multiple tables `[1P, 3P, ..., (2K-1)P]` for a slice of
/// affine points using a single batched inversion.
fn build_odd_tables_batched<const K: usize>(
    points: &[G1Affine],
) -> Vec<[G1Affine; K]> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }

    let extra = K - 1;
    let mut projs = Vec::with_capacity(n * extra);
    for point in points {
        let p_proj = G1Projective::from(point);
        let two_p = p_proj.double();
        let mut cur = p_proj;
        for _ in 0..extra {
            cur += two_p;
            projs.push(cur);
        }
    }

    let mut normalized = alloc::vec![G1Affine::identity(); projs.len()];
    if !projs.is_empty() {
        G1Projective::batch_normalize(&projs, &mut normalized);
    }

    let mut tables = Vec::with_capacity(n);
    for (idx, point) in points.iter().enumerate() {
        let mut row = [G1Affine::identity(); K];
        row[0] = *point;
        if extra > 0 {
            let start = idx * extra;
            row[1..].copy_from_slice(&normalized[start..start + extra]);
        }
        tables.push(row);
    }
    tables
}

/// Variable-time scalar multiplication `scalar * point` for a public scalar.
pub(crate) fn scalar_mul_public(
    point: &G1Affine,
    scalar: &BlsScalar,
) -> G1Projective {
    if bool::from(point.is_identity()) || *scalar == BlsScalar::zero() {
        return G1Projective::identity();
    }
    if *scalar == BlsScalar::one() {
        return G1Projective::from(point);
    }
    if *scalar == -BlsScalar::one() {
        return -G1Projective::from(point);
    }

    let p_proj = G1Projective::from(point);
    let two_p = p_proj.double();
    let mut projs = [G1Projective::identity(); DYNAMIC_ODD_COUNT - 1];
    let mut cur = p_proj;
    for slot in &mut projs {
        cur += two_p;
        *slot = cur;
    }
    let mut table = [G1Affine::identity(); DYNAMIC_ODD_COUNT];
    table[0] = *point;
    G1Projective::batch_normalize(&projs, &mut table[1..]);

    let mut digits = [0i8; 256];
    let len = encode_wnaf::<DYNAMIC_WNAF_WIDTH>(scalar, &mut digits);

    let mut acc = G1Projective::identity();
    for &d in digits[..len].iter().rev() {
        acc = acc.double();
        if d > 0 {
            acc = acc.add_mixed(&table[(d as usize) >> 1]);
        } else if d < 0 {
            acc = acc.add_mixed(&(-table[((-d) as usize) >> 1]));
        }
    }
    acc
}

/// Evaluate a slice of pre-built odd-multiple tables with signed wNAF digits
/// in a single shared doubling chain (Straus simultaneous
/// multi-exponentiation).
fn straus_eval_slice(
    tables: &[&[G1Affine]],
    digits: &[i8],
    lens: &[usize],
) -> G1Projective {
    let max_len = lens.iter().copied().max().unwrap_or(0);
    if max_len == 0 {
        return G1Projective::identity();
    }

    let mut acc = G1Projective::identity();
    for bit in (0..max_len).rev() {
        acc = acc.double();
        for (i, table) in tables.iter().enumerate() {
            if bit < lens[i] {
                let d = digits[i * 256 + bit];
                if d > 0 {
                    acc = acc.add_mixed(&table[(d as usize) >> 1]);
                } else if d < 0 {
                    acc = acc.add_mixed(&(-table[((-d) as usize) >> 1]));
                }
            }
        }
    }
    acc
}

/// Compute the two pairing inputs `(left, right)` for a single PLONK proof
/// verification:
/// - `left = -(w_z + u_challenge * w_zw)`
/// - `right = sum(static_points[i] * static_scalars[i])` plus
///   `sum(dynamic_points[j] * dynamic_scalars[j])`
pub(crate) fn verifier_msm_single(
    static_points: &[G1Affine; STATIC_POINT_COUNT],
    static_tables: &StaticVerifierTables,
    static_scalars: &[BlsScalar; STATIC_POINT_COUNT],
    dynamic_points: &[G1Affine; DYNAMIC_POINT_COUNT],
    dynamic_scalars: &[BlsScalar; DYNAMIC_POINT_COUNT],
    w_z: &G1Affine,
    w_zw: &G1Affine,
    u_challenge: &BlsScalar,
) -> (G1Projective, G1Projective) {
    let compute_left =
        || -(G1Projective::from(w_z) + scalar_mul_public(w_zw, u_challenge));

    #[cfg(feature = "std")]
    {
        // Gather active dynamic points (non-identity, nonzero scalar),
        // folding ±1 directly to avoid unnecessary table construction.
        let mut dyn_pts = Vec::with_capacity(DYNAMIC_POINT_COUNT);
        let mut dyn_scs = Vec::with_capacity(DYNAMIC_POINT_COUNT);
        let mut unit_sum = G1Projective::identity();
        for (pt, sc) in dynamic_points.iter().zip(dynamic_scalars) {
            if bool::from(pt.is_identity()) || *sc == BlsScalar::zero() {
                continue;
            }
            if *sc == BlsScalar::one() {
                unit_sum = unit_sum.add_mixed(pt);
            } else if *sc == -BlsScalar::one() {
                unit_sum = unit_sum.add_mixed(&(-*pt));
            } else {
                dyn_pts.push(*pt);
                dyn_scs.push(*sc);
            }
        }

        // Gather active static table indices & scalars.
        let mut active_static: Vec<(usize, BlsScalar)> =
            Vec::with_capacity(STATIC_POINT_COUNT);
        for (slot_idx, sc) in static_scalars.iter().enumerate() {
            if *sc == BlsScalar::zero() {
                continue;
            }
            if let Some(tbl_idx) = static_tables.slots[slot_idx] {
                if *sc == BlsScalar::one() {
                    unit_sum = unit_sum.add_mixed(&static_points[slot_idx]);
                } else if *sc == -BlsScalar::one() {
                    unit_sum = unit_sum.add_mixed(&(-static_points[slot_idx]));
                } else {
                    active_static.push((tbl_idx as usize, *sc));
                }
            }
        }

        let total_active = active_static.len() + dyn_pts.len();
        if total_active == 0 {
            return (compute_left(), unit_sum);
        }

        let num_threads = rayon::current_num_threads();
        if num_threads <= 1 {
            let dyn_tables =
                build_odd_tables_batched::<DYNAMIC_ODD_COUNT>(&dyn_pts);

            let mut tables: Vec<&[G1Affine]> = Vec::with_capacity(total_active);
            let mut digits = alloc::vec![0i8; total_active * 256];
            let mut lens = Vec::with_capacity(total_active);

            let mut idx = 0usize;
            for &(tbl_idx, ref sc) in &active_static {
                tables.push(&static_tables.tables[tbl_idx][..]);
                let slice: &mut [i8; 256] = (&mut digits
                    [idx * 256..(idx + 1) * 256])
                    .try_into()
                    .expect("chunk is 256 bytes");
                lens.push(encode_wnaf::<STATIC_WNAF_WIDTH>(sc, slice));
                idx += 1;
            }
            for (i, sc) in dyn_scs.iter().enumerate() {
                tables.push(&dyn_tables[i][..]);
                let slice: &mut [i8; 256] = (&mut digits
                    [idx * 256..(idx + 1) * 256])
                    .try_into()
                    .expect("chunk is 256 bytes");
                lens.push(encode_wnaf::<DYNAMIC_WNAF_WIDTH>(sc, slice));
                idx += 1;
            }

            let right = unit_sum + straus_eval_slice(&tables, &digits, &lens);
            return (compute_left(), right);
        }

        // Single flat 2-way split with no nested parallelism:
        // Branch A computes `left` + static-table Straus (zero table builds).
        // Branch B builds dynamic odd tables + dynamic Straus.
        let eval_static_and_left = || {
            let left = compute_left();
            let static_sum = if active_static.is_empty() {
                G1Projective::identity()
            } else {
                let n = active_static.len();
                let mut tables: Vec<&[G1Affine]> = Vec::with_capacity(n);
                let mut digits = alloc::vec![0i8; n * 256];
                let mut lens = Vec::with_capacity(n);
                for (i, &(tbl_idx, ref sc)) in active_static.iter().enumerate()
                {
                    tables.push(&static_tables.tables[tbl_idx][..]);
                    let slice: &mut [i8; 256] = (&mut digits
                        [i * 256..(i + 1) * 256])
                        .try_into()
                        .expect("chunk is 256 bytes");
                    lens.push(encode_wnaf::<STATIC_WNAF_WIDTH>(sc, slice));
                }
                straus_eval_slice(&tables, &digits, &lens)
            };
            (left, static_sum)
        };

        let eval_dynamic = || {
            if dyn_pts.is_empty() {
                return G1Projective::identity();
            }
            let dyn_tables =
                build_odd_tables_batched::<DYNAMIC_ODD_COUNT>(&dyn_pts);
            let n = dyn_pts.len();
            let table_refs: Vec<&[G1Affine]> =
                dyn_tables.iter().map(|t| &t[..]).collect();
            let mut digits = alloc::vec![0i8; n * 256];
            let mut lens = Vec::with_capacity(n);
            for (i, sc) in dyn_scs.iter().enumerate() {
                let slice: &mut [i8; 256] = (&mut digits
                    [i * 256..(i + 1) * 256])
                    .try_into()
                    .expect("chunk is 256 bytes");
                lens.push(encode_wnaf::<DYNAMIC_WNAF_WIDTH>(sc, slice));
            }
            straus_eval_slice(&table_refs, &digits, &lens)
        };

        let ((left, static_sum), dyn_sum) =
            rayon::join(eval_static_and_left, eval_dynamic);
        (left, unit_sum + static_sum + dyn_sum)
    }

    #[cfg(not(feature = "std"))]
    {
        let _ = static_tables;
        let mut pts =
            Vec::with_capacity(STATIC_POINT_COUNT + DYNAMIC_POINT_COUNT);
        let mut scs =
            Vec::with_capacity(STATIC_POINT_COUNT + DYNAMIC_POINT_COUNT);
        let mut unit_sum = G1Projective::identity();
        for (pt, sc) in static_points
            .iter()
            .zip(static_scalars)
            .chain(dynamic_points.iter().zip(dynamic_scalars))
        {
            if bool::from(pt.is_identity()) || *sc == BlsScalar::zero() {
                continue;
            }
            if *sc == BlsScalar::one() {
                unit_sum = unit_sum.add_mixed(pt);
            } else if *sc == -BlsScalar::one() {
                unit_sum = unit_sum.add_mixed(&(-*pt));
            } else {
                pts.push(*pt);
                scs.push(*sc);
            }
        }

        let left = compute_left();
        let right = if pts.is_empty() {
            unit_sum
        } else if pts.len() == 1 {
            unit_sum + scalar_mul_public(&pts[0], &scs[0])
        } else {
            let owned_tables =
                build_odd_tables_batched::<DYNAMIC_ODD_COUNT>(&pts);
            let table_refs: Vec<&[G1Affine]> =
                owned_tables.iter().map(|t| &t[..]).collect();
            let mut digits = alloc::vec![0i8; pts.len() * 256];
            let mut lens = Vec::with_capacity(pts.len());
            for (i, sc) in scs.iter().enumerate() {
                let slice: &mut [i8; 256] = (&mut digits
                    [i * 256..(i + 1) * 256])
                    .try_into()
                    .expect("chunk is 256 bytes");
                lens.push(encode_wnaf::<DYNAMIC_WNAF_WIDTH>(sc, slice));
            }
            unit_sum + straus_eval_slice(&table_refs, &digits, &lens)
        };
        (left, right)
    }
}

#[cfg(test)]
mod tests {
    use dusk_bls12_381::multiscalar_mul::msm_variable_base;
    use ff::Field;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;

    fn edge_scalars(rng: &mut StdRng) -> Vec<BlsScalar> {
        let neg_one = -BlsScalar::one();
        let neg_two = -BlsScalar::from(2u64);
        let pow_254 = BlsScalar::from(2u64).pow(&[254, 0, 0, 0]);
        let pow_252_minus_1 =
            BlsScalar::from(2u64).pow(&[252, 0, 0, 0]) - BlsScalar::one();
        let pow_250_minus_1 =
            BlsScalar::from(2u64).pow(&[250, 0, 0, 0]) - BlsScalar::one();
        let mut out = alloc::vec![
            BlsScalar::zero(),
            BlsScalar::one(),
            BlsScalar::from(2u64),
            BlsScalar::from(3u64),
            BlsScalar::from(15u64),
            BlsScalar::from(16u64),
            BlsScalar::from(31u64),
            BlsScalar::from(32u64),
            neg_one,
            neg_two,
            pow_254,
            pow_254 - BlsScalar::one(),
            pow_252_minus_1,
            pow_250_minus_1,
            neg_one - BlsScalar::from(127u64),
        ];
        for _ in 0..17 {
            out.push(BlsScalar::random(&mut *rng));
        }
        out
    }

    #[test]
    fn scalar_mul_public_matches_reference_on_edge_scalars() {
        let mut rng = StdRng::seed_from_u64(0x51a1_cafe);
        let p = G1Affine::from(
            G1Projective::generator() * BlsScalar::random(&mut rng),
        );
        for s in edge_scalars(&mut rng) {
            assert_eq!(scalar_mul_public(&p, &s), p * s);
            assert_eq!(
                scalar_mul_public(&G1Affine::identity(), &s),
                G1Projective::identity()
            );
        }
    }

    #[test]
    fn verifier_msm_single_matches_reference_including_identities() {
        let mut rng = StdRng::seed_from_u64(0xbeef_1001);
        let mut static_points = [G1Affine::identity(); STATIC_POINT_COUNT];
        let mut static_scalars = [BlsScalar::zero(); STATIC_POINT_COUNT];
        let edges = edge_scalars(&mut rng);

        for i in 0..STATIC_POINT_COUNT {
            // Mix identity and non-identity static points.
            if i % 3 != 0 {
                static_points[i] = G1Affine::from(
                    G1Projective::generator() * BlsScalar::random(&mut rng),
                );
            }
            static_scalars[i] = edges[i % edges.len()];
        }

        let mut dynamic_points = [G1Affine::identity(); DYNAMIC_POINT_COUNT];
        let mut dynamic_scalars = [BlsScalar::zero(); DYNAMIC_POINT_COUNT];
        for i in 0..DYNAMIC_POINT_COUNT {
            if i != 5 {
                dynamic_points[i] = G1Affine::from(
                    G1Projective::generator() * BlsScalar::random(&mut rng),
                );
            }
            dynamic_scalars[i] = edges[(i + 7) % edges.len()];
        }

        let w_z = G1Affine::from(
            G1Projective::generator() * BlsScalar::random(&mut rng),
        );
        let w_zw = G1Affine::from(
            G1Projective::generator() * BlsScalar::random(&mut rng),
        );
        let u = BlsScalar::random(&mut rng);

        let tables = StaticVerifierTables::new(&static_points);
        let (left, right) = verifier_msm_single(
            &static_points,
            &tables,
            &static_scalars,
            &dynamic_points,
            &dynamic_scalars,
            &w_z,
            &w_zw,
            &u,
        );

        let expected_left = -(G1Projective::from(w_z) + (w_zw * u));
        let mut all_pts = Vec::new();
        all_pts.extend_from_slice(&static_points);
        all_pts.extend_from_slice(&dynamic_points);
        let mut all_scs = Vec::new();
        all_scs.extend_from_slice(&static_scalars);
        all_scs.extend_from_slice(&dynamic_scalars);
        let expected_right = msm_variable_base(&all_pts, &all_scs);

        assert_eq!(left, expected_left);
        assert_eq!(right, expected_right);
    }
}
