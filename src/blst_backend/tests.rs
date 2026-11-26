// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use dusk_bls12_381::multiscalar_mul::msm_variable_base;
use dusk_bls12_381::{G1Projective, G2Projective};
use dusk_bytes::Serializable;
use ff::Field;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::verify::{
    GLV_X_SQ_HI, GLV_X_SQ_LO, glv_straus_blst, msm_variable_base_proj,
};
use super::*;
use crate::fft::EvaluationDomain;
use crate::proof_system::verifier_msm::{
    DYNAMIC_POINT_COUNT, STATIC_POINT_COUNT,
};

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
    let mut pts: [G1Affine; STATIC_POINT_COUNT] = core::array::from_fn(|_| {
        G1Affine::from(G1Projective::generator() * BlsScalar::random(&mut rng))
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
    let w_z =
        G1Affine::from(G1Projective::generator() * BlsScalar::random(&mut rng));
    let w_zw =
        G1Affine::from(G1Projective::generator() * BlsScalar::random(&mut rng));
    let u = BlsScalar::random(&mut rng);

    // Adjust the last dynamic slot so `right == x * (w_z + u * w_zw)`,
    // satisfying `e(-(w_z + u * w_zw), x * H) * e(right, H) == 1`.
    let neg_left_proj = G1Projective::from(w_z) + G1Projective::from(w_zw) * u;
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

#[test]
fn blst_batched_affine_explicit_cancellation_and_doubling() {
    let mut rng = StdRng::seed_from_u64(0xca0c_e101);
    for n in [1024usize, 4096] {
        let mut pts: Vec<G1Affine> = (0..n)
            .map(|_| {
                G1Affine::from(
                    G1Projective::generator() * BlsScalar::random(&mut rng),
                )
            })
            .collect();
        let mut scs: Vec<BlsScalar> =
            (0..n).map(|_| BlsScalar::random(&mut rng)).collect();
        // Pair `pts[2*k]` and `-pts[2*k]` with identical scalars so they land
        // in the exact same bucket and exercise true `P + (-P)` cancellation,
        // and pair identical bases with identical scalars for true `P + P`.
        for k in 0..32 {
            pts[2 * k + 1] = -pts[2 * k];
            scs[2 * k + 1] = scs[2 * k];
        }
        for k in 32..64 {
            pts[2 * k + 1] = pts[2 * k];
            scs[2 * k + 1] = scs[2 * k];
        }
        let blst_pts = g1_affines_to_blst(&pts);
        let expected = G1Affine::from(msm_variable_base(&pts, &scs));
        for w in [None, Some(10), Some(12), Some(13)] {
            let [got] = msm_g1_batch_affine(&blst_pts, [&scs[..]], w);
            assert_eq!(got, expected, "cancellation mismatch n={n} w={w:?}");
        }
    }
}
