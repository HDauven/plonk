// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use dusk_plonk::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Default)]
struct MulPublicCircuit {
    a: BlsScalar,
    b: BlsScalar,
}

impl Circuit for MulPublicCircuit {
    fn circuit(&self, composer: &mut Composer) -> Result<(), Error> {
        let w_a = composer.append_witness(self.a);
        let w_b = composer.append_witness(self.b);
        let prod = composer.gate_mul(Constraint::new().mult(1).a(w_a).b(w_b));
        composer.assert_equal_constant(prod, 0u64, Some(self.a * self.b));
        Ok(())
    }
}

#[derive(Default)]
struct MixedGatesCircuit {
    a: BlsScalar,
    b: BlsScalar,
    k: JubJubScalar,
}

impl Circuit for MixedGatesCircuit {
    fn circuit(&self, composer: &mut Composer) -> Result<(), Error> {
        let w_a = composer.append_witness(self.a);
        let w_b = composer.append_witness(self.b);
        let sum =
            composer.gate_add(Constraint::new().left(1).right(1).a(w_a).b(w_b));
        composer.assert_equal_constant(sum, 0u64, Some(self.a + self.b));
        composer.component_range_bits::<8>(w_a);
        let _xor = composer.append_logic_xor::<4>(w_a, w_b);
        let w_k = composer.append_witness(self.k);
        let p = composer
            .component_mul_generator(w_k, dusk_jubjub::GENERATOR_EXTENDED)?;
        let expected = dusk_jubjub::GENERATOR_EXTENDED * self.k;
        composer.assert_equal_public_point(p.into(), expected)?;
        Ok(())
    }
}

#[test]
fn same_circuit_verify_batch_sizes_and_corruptions() {
    let mut rng = StdRng::seed_from_u64(0xba7c_0001);
    let pp = PublicParameters::setup(1 << 6, &mut rng).expect("setup");
    let (prover, verifier) =
        Compiler::compile::<MulPublicCircuit>(&pp, b"batch-mul")
            .expect("compile");

    // Empty batch succeeds.
    let empty: [(&Proof, &[BlsScalar]); 0] = [];
    verifier.verify_batch(&empty).expect("empty batch");

    // Generate 16 distinct valid proofs (covers single, Straus, and Pippenger
    // regimes: K = 1, 2, 4, 8, 16).
    let proofs: Vec<(Proof, Vec<BlsScalar>)> = (1..=16u64)
        .map(|i| {
            let circuit = MulPublicCircuit {
                a: BlsScalar::from(i + 2),
                b: BlsScalar::from(3 * i + 5),
            };
            prover.prove(&mut rng, &circuit).expect("prove")
        })
        .collect();

    for k in [1usize, 2, 4, 8, 16] {
        let items: Vec<(&Proof, &[BlsScalar])> = proofs[..k]
            .iter()
            .map(|(p, pi)| (p, pi.as_slice()))
            .collect();
        verifier
            .verify_batch(&items)
            .unwrap_or_else(|e| panic!("valid batch k={k} failed: {e}"));
    }

    // Wrong public input length in the middle of the batch returns
    // InconsistentPublicInputsLen.
    let bad_len: Vec<(&Proof, &[BlsScalar])> = vec![
        (&proofs[0].0, proofs[0].1.as_slice()),
        (&proofs[1].0, &[]),
        (&proofs[2].0, proofs[2].1.as_slice()),
    ];
    assert_eq!(
        verifier.verify_batch(&bad_len),
        Err(Error::InconsistentPublicInputsLen {
            expected: 1,
            provided: 0,
        })
    );

    // Corrupting any single proof's public input (first, middle, or last) must
    // reject the entire batch.
    for target in [0usize, 5, 15] {
        let bad_pi = vec![proofs[target].1[0] + BlsScalar::one()];
        let items: Vec<(&Proof, &[BlsScalar])> = proofs
            .iter()
            .enumerate()
            .map(|(idx, (p, pi))| {
                if idx == target {
                    (p, bad_pi.as_slice())
                } else {
                    (p, pi.as_slice())
                }
            })
            .collect();
        assert_eq!(
            verifier.verify_batch(&items),
            Err(Error::ProofVerificationError),
            "corrupted public input at index {target} must fail batch"
        );
    }

    // Swapping two proofs' public inputs must fail.
    let mut swapped: Vec<(&Proof, &[BlsScalar])> =
        proofs.iter().map(|(p, pi)| (p, pi.as_slice())).collect();
    swapped[2].1 = proofs[3].1.as_slice();
    swapped[3].1 = proofs[2].1.as_slice();
    assert_eq!(
        verifier.verify_batch(&swapped),
        Err(Error::ProofVerificationError)
    );
}

#[test]
fn heterogeneous_batch_verifier_across_circuits_and_srs_checks() {
    let mut rng = StdRng::seed_from_u64(0xba7c_0002);
    let pp = PublicParameters::setup(1 << 11, &mut rng).expect("setup");

    let (prover_a, verifier_a) =
        Compiler::compile::<MulPublicCircuit>(&pp, b"circuit-a")
            .expect("compile A");
    let (prover_b, verifier_b) =
        Compiler::compile::<MixedGatesCircuit>(&pp, b"circuit-b")
            .expect("compile B");
    // Same circuit compiled under a different label (identical VK, different
    // transcript label).
    let (prover_a2, verifier_a2) =
        Compiler::compile::<MulPublicCircuit>(&pp, b"circuit-a-alt")
            .expect("compile A2");

    // Different SRS -> incompatible opening key.
    let pp_other = PublicParameters::setup(1 << 6, &mut rng).expect("setup 2");
    let (prover_other, verifier_other) =
        Compiler::compile::<MulPublicCircuit>(&pp_other, b"circuit-other")
            .expect("compile other");

    let proofs_a: Vec<_> = (0..5u64)
        .map(|i| {
            let c = MulPublicCircuit {
                a: BlsScalar::from(i + 10),
                b: BlsScalar::from(i + 20),
            };
            prover_a.prove(&mut rng, &c).expect("prove A")
        })
        .collect();
    let proofs_b: Vec<_> = (0..4u64)
        .map(|i| {
            let c = MixedGatesCircuit {
                a: BlsScalar::from(i + 1),
                b: BlsScalar::from(i + 2),
                k: JubJubScalar::from(i + 5),
            };
            prover_b.prove(&mut rng, &c).expect("prove B")
        })
        .collect();
    let (proof_a2, pi_a2) = prover_a2
        .prove(
            &mut rng,
            &MulPublicCircuit {
                a: BlsScalar::from(9),
                b: BlsScalar::from(11),
            },
        )
        .expect("prove A2");
    let (proof_other, pi_other) = prover_other
        .prove(
            &mut rng,
            &MulPublicCircuit {
                a: BlsScalar::from(3),
                b: BlsScalar::from(4),
            },
        )
        .expect("prove other");

    // Empty BatchVerifier succeeds.
    let empty = BatchVerifier::new();
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    empty.verify().expect("empty batch verifier");

    // Single-entry BatchVerifier delegates cleanly.
    let mut single = BatchVerifier::with_capacity(1);
    single
        .push(&verifier_a, &proofs_a[0].0, &proofs_a[0].1)
        .expect("push single");
    assert_eq!(single.len(), 1);
    assert!(!single.is_empty());
    single.verify().expect("single batch verifier");

    // Valid mixed-circuit batch (same VK + different labels + different
    // circuit).
    let mut batch = BatchVerifier::with_capacity(10);
    for (p, pi) in &proofs_a {
        batch.push(&verifier_a, p, pi).expect("push A");
    }
    for (p, pi) in &proofs_b {
        batch.push(&verifier_b, p, pi).expect("push B");
    }
    batch
        .push(&verifier_a2, &proof_a2, &pi_a2)
        .expect("push A2");

    // Pushing a verifier from a different SRS is rejected immediately.
    assert_eq!(
        batch.push(&verifier_other, &proof_other, &pi_other),
        Err(Error::IncompatibleOpeningKey)
    );
    // Failed push does not corrupt already queued valid entries.
    assert_eq!(batch.len(), 10);
    batch.verify().expect("mixed-circuit batch should verify");

    // Cross-label replay (valid proof_a2 verified under verifier_a) must fail.
    let mut bad_label_batch = BatchVerifier::new();
    bad_label_batch
        .push(&verifier_a, &proofs_a[0].0, &proofs_a[0].1)
        .unwrap();
    bad_label_batch
        .push(&verifier_b, &proofs_b[0].0, &proofs_b[0].1)
        .unwrap();
    bad_label_batch
        .push(&verifier_a, &proof_a2, &pi_a2)
        .unwrap();
    assert_eq!(bad_label_batch.verify(), Err(Error::ProofVerificationError));
}

#[test]
fn batch_verifier_supports_upstream_mixed_v2_and_v3_fixtures() {
    use dusk_bytes::DeserializableSlice;

    let verifier = Verifier::try_from_bytes(include_bytes!(
        "fixtures/merlin-3/verifier.bin"
    ))
    .expect("upstream verifier");
    let v2_proof =
        Proof::from_slice(include_bytes!("fixtures/merlin-3/v2.proof"))
            .expect("v2 proof");
    let v3_proof =
        Proof::from_slice(include_bytes!("fixtures/merlin-3/v3.proof"))
            .expect("v3 proof");
    let pi = [BlsScalar::from(12u64)];

    let mut batch = BatchVerifier::new();
    batch
        .push_with_version(&verifier, &v2_proof, &pi, PlonkVersion::V2)
        .unwrap();
    batch
        .push_with_version(&verifier, &v3_proof, &pi, PlonkVersion::V3)
        .unwrap();
    batch
        .verify()
        .expect("mixed V2+V3 fixture batch must verify");

    // Verifying a V3 proof under V1 inside a batch must fail.
    let mut bad_v1 = BatchVerifier::new();
    bad_v1
        .push_with_version(&verifier, &v2_proof, &pi, PlonkVersion::V2)
        .unwrap();
    bad_v1
        .push_with_version(&verifier, &v3_proof, &pi, PlonkVersion::V1)
        .unwrap();
    assert_eq!(bad_v1.verify(), Err(Error::ProofVerificationError));
}

#[cfg(feature = "legacy-proving")]
#[test]
fn batch_verifier_supports_mixed_plonk_versions() {
    let mut rng = StdRng::seed_from_u64(0xba7c_0003);
    let pp = PublicParameters::setup(1 << 6, &mut rng).expect("setup");
    let (prover, verifier) =
        Compiler::compile::<MulPublicCircuit>(&pp, b"batch-versions")
            .expect("compile");

    let c2 = MulPublicCircuit {
        a: BlsScalar::from(8),
        b: BlsScalar::from(9),
    };
    let c3 = MulPublicCircuit {
        a: BlsScalar::from(11),
        b: BlsScalar::from(13),
    };

    let (p_v2, pi_v2) = prover
        .prove_with_version(&mut rng, &c2, PlonkVersion::V2)
        .expect("prove v2");
    let (p_v3, pi_v3) = prover
        .prove_with_version(&mut rng, &c3, PlonkVersion::V3)
        .expect("prove v3");

    // Mixed V2 + V3 in a single BatchVerifier.
    let mut batch = BatchVerifier::new();
    batch
        .push_with_version(&verifier, &p_v2, &pi_v2, PlonkVersion::V2)
        .unwrap();
    batch
        .push_with_version(&verifier, &p_v3, &pi_v3, PlonkVersion::V3)
        .unwrap();
    batch.verify().expect("mixed-version batch should verify");

    // Verifying a V2 proof under V3 inside the batch must fail.
    let mut bad = BatchVerifier::new();
    bad.push_with_version(&verifier, &p_v3, &pi_v3, PlonkVersion::V3)
        .unwrap();
    bad.push_with_version(&verifier, &p_v2, &pi_v2, PlonkVersion::V3)
        .unwrap();
    assert_eq!(bad.verify(), Err(Error::ProofVerificationError));

    // Same-circuit verify_batch_with_version for V2.
    let (p_v2b, pi_v2b) = prover
        .prove_with_version(&mut rng, &c3, PlonkVersion::V2)
        .expect("prove v2b");
    verifier
        .verify_batch_with_version(
            &[(&p_v2, pi_v2.as_slice()), (&p_v2b, pi_v2b.as_slice())],
            PlonkVersion::V2,
        )
        .expect("v2 batch should verify");
    assert_eq!(
        verifier.verify_batch_with_version(
            &[(&p_v2, pi_v2.as_slice()), (&p_v2b, pi_v2b.as_slice())],
            PlonkVersion::V3,
        ),
        Err(Error::ProofVerificationError)
    );
}
