// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

#![forbid(unsafe_code)]

//! A Proof stores the commitments to all of the elements that
//! are needed to univocally identify a prove of some statement.

use dusk_bytes::{DeserializableSlice, Serializable};
#[cfg(feature = "std")]
use rayon::prelude::*;

use super::linearization_poly::ProofEvaluations;
use crate::BufferWriter;
use crate::commitment_scheme::Commitment;

// Number of (unshifted) polynomials opened at `z`, excluding the linearization
// polynomial `r(X)`.
//
// We open at `z`:
//   a, b, c, d, s_sigma_1, s_sigma_2, s_sigma_3,
//   q_arith, q_c, q_l, q_r
#[cfg(feature = "alloc")]
const V_MAX_DEGREE: usize = 11;
// Legacy number of (unshifted) polynomials opened at `z`, excluding the
// linearization polynomial `r(X)`.
//
// This matches the pre-soundness-fix batching that does NOT bind selector /
// constant evaluations in the batched opening at `z`.
#[cfg(feature = "alloc")]
const V_MAX_DEGREE_LEGACY: usize = 7;

#[cfg(feature = "rkyv-impl")]
use bytecheck::CheckBytes;
#[cfg(feature = "rkyv-impl")]
use rkyv::{
    Archive, Deserialize, Serialize,
    ser::{ScratchSpace, Serializer},
};

/// A Proof is a composition of `Commitment`s to the Witness, Permutation,
/// Quotient, Shifted and Opening polynomials as well as the
/// `ProofEvaluations`.
///
/// It's main goal is to allow the `Verifier` to formally verify that the secret
/// witnesses used to generate the [`Proof`] satisfy a circuit that both
/// [`Composer`] and [`Verifier`] have in common succintly and without any
/// capabilities of adquiring any kind of knowledge about the witness used to
/// construct the Proof.
///
/// [`Composer`]: [`crate::prelude::Composer`]
/// [`Verifier`]: [`crate::prelude::Verifier`]
#[derive(Debug, Eq, PartialEq, Clone, Default)]
#[cfg_attr(
    feature = "rkyv-impl",
    derive(Archive, Deserialize, Serialize),
    archive(bound(serialize = "__S: Serializer + ScratchSpace")),
    archive_attr(derive(CheckBytes))
)]
pub struct Proof {
    /// Commitment to the witness polynomial for the left wires.
    #[cfg_attr(feature = "rkyv-impl", omit_bounds)]
    pub(crate) a_comm: Commitment,
    /// Commitment to the witness polynomial for the right wires.
    #[cfg_attr(feature = "rkyv-impl", omit_bounds)]
    pub(crate) b_comm: Commitment,
    /// Commitment to the witness polynomial for the output wires.
    #[cfg_attr(feature = "rkyv-impl", omit_bounds)]
    pub(crate) c_comm: Commitment,
    /// Commitment to the witness polynomial for the fourth wires.
    #[cfg_attr(feature = "rkyv-impl", omit_bounds)]
    pub(crate) d_comm: Commitment,

    /// Commitment to the permutation polynomial.
    #[cfg_attr(feature = "rkyv-impl", omit_bounds)]
    pub(crate) z_comm: Commitment,

    /// Commitment to the quotient polynomial.
    #[cfg_attr(feature = "rkyv-impl", omit_bounds)]
    pub(crate) t_low_comm: Commitment,
    /// Commitment to the quotient polynomial.
    #[cfg_attr(feature = "rkyv-impl", omit_bounds)]
    pub(crate) t_mid_comm: Commitment,
    /// Commitment to the quotient polynomial.
    #[cfg_attr(feature = "rkyv-impl", omit_bounds)]
    pub(crate) t_high_comm: Commitment,
    /// Commitment to the quotient polynomial.
    #[cfg_attr(feature = "rkyv-impl", omit_bounds)]
    pub(crate) t_fourth_comm: Commitment,

    /// Commitment to the opening polynomial.
    #[cfg_attr(feature = "rkyv-impl", omit_bounds)]
    pub(crate) w_z_chall_comm: Commitment,
    /// Commitment to the shifted opening polynomial.
    #[cfg_attr(feature = "rkyv-impl", omit_bounds)]
    pub(crate) w_z_chall_w_comm: Commitment,
    /// Subset of all of the evaluations added to the proof.
    #[cfg_attr(feature = "rkyv-impl", omit_bounds)]
    pub(crate) evaluations: ProofEvaluations,
}

// The struct Proof has 11 commitments + 1 ProofEvaluations
impl Serializable<{ 11 * Commitment::SIZE + ProofEvaluations::SIZE }>
    for Proof
{
    type Error = dusk_bytes::Error;

    fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        let mut writer = BufferWriter::new(&mut buf);
        writer.write(&self.a_comm.to_bytes());
        writer.write(&self.b_comm.to_bytes());
        writer.write(&self.c_comm.to_bytes());
        writer.write(&self.d_comm.to_bytes());
        writer.write(&self.z_comm.to_bytes());
        writer.write(&self.t_low_comm.to_bytes());
        writer.write(&self.t_mid_comm.to_bytes());
        writer.write(&self.t_high_comm.to_bytes());
        writer.write(&self.t_fourth_comm.to_bytes());
        writer.write(&self.w_z_chall_comm.to_bytes());
        writer.write(&self.w_z_chall_w_comm.to_bytes());
        writer.write(&self.evaluations.to_bytes());
        writer.finish();

        buf
    }

    fn from_bytes(buf: &[u8; Self::SIZE]) -> Result<Self, Self::Error> {
        #[cfg(feature = "std")]
        {
            let (comm_bytes, mut eval_bytes) =
                buf.split_at(11 * Commitment::SIZE);
            let (comm_chunks, _) =
                comm_bytes.as_chunks::<{ Commitment::SIZE }>();
            let mut comms = [Ok(Commitment::default()); 11];
            if rayon::current_num_threads() > 1 {
                // 4-way split (3+3+3+2 points) stays well inside P-cores
                // without excessive thread-wakeup tail latency.
                comms
                    .par_chunks_mut(3)
                    .zip(comm_chunks.par_chunks(3))
                    .for_each(|(slots, chunks)| {
                        for (slot, chunk) in slots.iter_mut().zip(chunks.iter())
                        {
                            *slot = Commitment::from_bytes(chunk);
                        }
                    });
            } else {
                for (slot, chunk) in comms.iter_mut().zip(comm_chunks.iter()) {
                    *slot = Commitment::from_bytes(chunk);
                }
            }

            let a_comm = comms[0]?;
            let b_comm = comms[1]?;
            let c_comm = comms[2]?;
            let d_comm = comms[3]?;
            let z_comm = comms[4]?;
            let t_low_comm = comms[5]?;
            let t_mid_comm = comms[6]?;
            let t_high_comm = comms[7]?;
            let t_fourth_comm = comms[8]?;
            let w_z_chall_comm = comms[9]?;
            let w_z_chall_w_comm = comms[10]?;
            let evaluations = ProofEvaluations::from_reader(&mut eval_bytes)?;

            Ok(Proof {
                a_comm,
                b_comm,
                c_comm,
                d_comm,
                z_comm,
                t_low_comm,
                t_mid_comm,
                t_high_comm,
                t_fourth_comm,
                w_z_chall_comm,
                w_z_chall_w_comm,
                evaluations,
            })
        }

        #[cfg(not(feature = "std"))]
        {
            let mut buffer = &buf[..];

            let a_comm = Commitment::from_reader(&mut buffer)?;
            let b_comm = Commitment::from_reader(&mut buffer)?;
            let c_comm = Commitment::from_reader(&mut buffer)?;
            let d_comm = Commitment::from_reader(&mut buffer)?;
            let z_comm = Commitment::from_reader(&mut buffer)?;
            let t_low_comm = Commitment::from_reader(&mut buffer)?;
            let t_mid_comm = Commitment::from_reader(&mut buffer)?;
            let t_high_comm = Commitment::from_reader(&mut buffer)?;
            let t_fourth_comm = Commitment::from_reader(&mut buffer)?;
            let w_z_chall_comm = Commitment::from_reader(&mut buffer)?;
            let w_z_chall_w_comm = Commitment::from_reader(&mut buffer)?;
            let evaluations = ProofEvaluations::from_reader(&mut buffer)?;

            Ok(Proof {
                a_comm,
                b_comm,
                c_comm,
                d_comm,
                z_comm,
                t_low_comm,
                t_mid_comm,
                t_high_comm,
                t_fourth_comm,
                w_z_chall_comm,
                w_z_chall_w_comm,
                evaluations,
            })
        }
    }
}

impl Proof {
    /// Decodes a [`Proof`] from exactly [`Proof::SIZE`] bytes.
    ///
    /// Unlike [`DeserializableSlice::from_slice`], which ignores trailing data,
    /// this rejects any other length with [`dusk_bytes::Error::BadLength`].
    pub fn from_slice_exact(bytes: &[u8]) -> Result<Self, dusk_bytes::Error> {
        let exact =
            bytes.try_into().map_err(|_| dusk_bytes::Error::BadLength {
                found: bytes.len(),
                expected: Self::SIZE,
            })?;
        Self::from_bytes(exact)
    }
}

#[cfg(feature = "alloc")]
#[allow(unused_imports)]
pub(crate) mod alloc {
    use super::*;
    use crate::commitment_scheme::{AggregateProof, OpeningKey};
    use crate::error::Error;
    use crate::fft::EvaluationDomain;
    use crate::proof_system::verifier_msm::{
        DYNAMIC_POINT_COUNT, STATIC_POINT_COUNT, StaticVerifierTables,
        verifier_msm_single,
    };
    use crate::proof_system::widget::{
        VerifierKey, arithmetic, ecc, logic, permutation, range,
    };
    use crate::transcript::TranscriptProtocol;
    use crate::util::batch_inversion;
    #[rustfmt::skip]
    use ::alloc::vec::Vec;
    #[cfg(test)]
    use dusk_bls12_381::multiscalar_mul::msm_variable_base;
    use dusk_bls12_381::{BlsScalar, G1Affine, G1Projective};
    use merlin::Transcript;
    #[cfg(feature = "std")]
    use rayon::prelude::*;

    /// Scalar weights and dynamic proof commitments produced by replaying a
    /// proof's Fiat-Shamir transcript. Shared by single-proof and batched
    /// verification.
    pub(crate) struct PreparedVerification {
        pub(crate) static_scalars: [BlsScalar; STATIC_POINT_COUNT],
        pub(crate) dynamic_points: [G1Affine; DYNAMIC_POINT_COUNT],
        pub(crate) dynamic_scalars: [BlsScalar; DYNAMIC_POINT_COUNT],
        pub(crate) w_z: G1Affine,
        pub(crate) w_zw: G1Affine,
        pub(crate) u_challenge: BlsScalar,
    }

    #[cfg(test)]
    struct ReferenceCheckData {
        alpha: BlsScalar,
        beta: BlsScalar,
        gamma: BlsScalar,
        range_sep_challenge: BlsScalar,
        logic_sep_challenge: BlsScalar,
        fixed_base_sep_challenge: BlsScalar,
        var_base_sep_challenge: BlsScalar,
        z_challenge: BlsScalar,
        u_challenge: BlsScalar,
        l1_eval: BlsScalar,
        e_scalar: BlsScalar,
        shifted_witness_scalar: BlsScalar,
        f_scalars: Vec<BlsScalar>,
    }

    #[cfg(test)]
    type RefCheck = ReferenceCheckData;
    #[cfg(not(test))]
    type RefCheck = ();

    impl Proof {
        /// Absorbs every proof commitment and evaluation into a batch
        /// verification transcript.
        pub(crate) fn append_to_batch_transcript(
            &self,
            transcript: &mut Transcript,
        ) {
            let Self {
                a_comm,
                b_comm,
                c_comm,
                d_comm,
                z_comm,
                t_low_comm,
                t_mid_comm,
                t_high_comm,
                t_fourth_comm,
                w_z_chall_comm,
                w_z_chall_w_comm,
                evaluations,
            } = self;

            transcript.append_commitment(b"a_comm", a_comm);
            transcript.append_commitment(b"b_comm", b_comm);
            transcript.append_commitment(b"c_comm", c_comm);
            transcript.append_commitment(b"d_comm", d_comm);
            transcript.append_commitment(b"z_comm", z_comm);
            transcript.append_commitment(b"t_low_comm", t_low_comm);
            transcript.append_commitment(b"t_mid_comm", t_mid_comm);
            transcript.append_commitment(b"t_high_comm", t_high_comm);
            transcript.append_commitment(b"t_fourth_comm", t_fourth_comm);
            transcript.append_commitment(b"w_z_chall_comm", w_z_chall_comm);
            transcript.append_commitment(b"w_z_chall_w_comm", w_z_chall_w_comm);

            let ProofEvaluations {
                a_eval,
                b_eval,
                c_eval,
                d_eval,
                a_w_eval,
                b_w_eval,
                d_w_eval,
                q_arith_eval,
                q_c_eval,
                q_l_eval,
                q_r_eval,
                s_sigma_1_eval,
                s_sigma_2_eval,
                s_sigma_3_eval,
                z_eval,
            } = evaluations;

            transcript.append_scalar(b"a_eval", a_eval);
            transcript.append_scalar(b"b_eval", b_eval);
            transcript.append_scalar(b"c_eval", c_eval);
            transcript.append_scalar(b"d_eval", d_eval);
            transcript.append_scalar(b"a_w_eval", a_w_eval);
            transcript.append_scalar(b"b_w_eval", b_w_eval);
            transcript.append_scalar(b"d_w_eval", d_w_eval);
            transcript.append_scalar(b"q_arith_eval", q_arith_eval);
            transcript.append_scalar(b"q_c_eval", q_c_eval);
            transcript.append_scalar(b"q_l_eval", q_l_eval);
            transcript.append_scalar(b"q_r_eval", q_r_eval);
            transcript.append_scalar(b"s_sigma_1_eval", s_sigma_1_eval);
            transcript.append_scalar(b"s_sigma_2_eval", s_sigma_2_eval);
            transcript.append_scalar(b"s_sigma_3_eval", s_sigma_3_eval);
            transcript.append_scalar(b"z_eval", z_eval);
        }

        /// Replay the proof's Fiat-Shamir transcript and return its scalar
        /// weights and dynamic commitments for batched verification.
        pub(crate) fn prepare_verification(
            &self,
            transcript: &mut Transcript,
            domain: &EvaluationDomain,
            public_input_roots: &[BlsScalar],
            pub_inputs: &[BlsScalar],
            bind_selectors: bool,
        ) -> Result<PreparedVerification, Error> {
            let (prepared, _) = self.prepare_with_ref(
                transcript,
                domain,
                public_input_roots,
                pub_inputs,
                bind_selectors,
            )?;
            Ok(prepared)
        }

        /// Execute the final 2-pair BLS12-381 pairing check on affine inputs.
        #[cfg(all(feature = "blst-backend", test))]
        pub(crate) fn verify_pairing_affine(
            opening_key: &OpeningKey,
            left_affine: G1Affine,
            right_affine: G1Affine,
        ) -> Result<(), Error> {
            if !crate::blst_backend::verify_pairing_lines(
                &left_affine,
                &right_affine,
                &opening_key.blst_lines_x_h,
                &opening_key.blst_lines_h,
            ) {
                return Err(Error::ProofVerificationError);
            }
            Ok(())
        }

        /// Execute the final 2-pair BLS12-381 pairing check shared by single
        /// and batched verification.
        #[cfg(not(all(feature = "blst-backend", not(test))))]
        pub(crate) fn verify_pairing(
            opening_key: &OpeningKey,
            left_projective: G1Projective,
            right_projective: G1Projective,
        ) -> Result<(), Error> {
            let projectives = [left_projective, right_projective];
            let mut affines = [G1Affine::identity(); 2];
            G1Projective::batch_normalize(&projectives, &mut affines);

            #[cfg(feature = "blst-backend")]
            {
                Self::verify_pairing_affine(opening_key, affines[0], affines[1])
            }
            #[cfg(not(feature = "blst-backend"))]
            {
                let pairing = dusk_bls12_381::multi_miller_loop(&[
                    (&affines[0], &opening_key.prepared_x_h),
                    (&affines[1], &opening_key.prepared_h),
                ])
                .final_exponentiation();

                if pairing != dusk_bls12_381::Gt::identity() {
                    return Err(Error::ProofVerificationError);
                }

                Ok(())
            }
        }

        #[allow(non_snake_case)]
        fn prepare_with_ref(
            &self,
            transcript: &mut Transcript,
            domain: &EvaluationDomain,
            public_input_roots: &[BlsScalar],
            pub_inputs: &[BlsScalar],
            bind_selectors: bool,
        ) -> Result<(PreparedVerification, RefCheck), Error> {
            // Subgroup checks are done when the proof is deserialized.
            transcript.append_commitment(b"a_comm", &self.a_comm);
            transcript.append_commitment(b"b_comm", &self.b_comm);
            transcript.append_commitment(b"c_comm", &self.c_comm);
            transcript.append_commitment(b"d_comm", &self.d_comm);

            let beta = transcript.challenge_scalar(b"beta");
            transcript.append_scalar(b"beta", &beta);
            let gamma = transcript.challenge_scalar(b"gamma");

            transcript.append_commitment(b"z_comm", &self.z_comm);

            let alpha = transcript.challenge_scalar(b"alpha");
            let range_sep_challenge =
                transcript.challenge_scalar(b"range separation challenge");
            let logic_sep_challenge =
                transcript.challenge_scalar(b"logic separation challenge");
            let fixed_base_sep_challenge =
                transcript.challenge_scalar(b"fixed base separation challenge");
            let var_base_sep_challenge = transcript
                .challenge_scalar(b"variable base separation challenge");

            transcript.append_commitment(b"t_low_comm", &self.t_low_comm);
            transcript.append_commitment(b"t_mid_comm", &self.t_mid_comm);
            transcript.append_commitment(b"t_high_comm", &self.t_high_comm);
            transcript.append_commitment(b"t_fourth_comm", &self.t_fourth_comm);

            let z_challenge = transcript.challenge_scalar(b"z_challenge");

            transcript.append_scalar(b"a_eval", &self.evaluations.a_eval);
            transcript.append_scalar(b"b_eval", &self.evaluations.b_eval);
            transcript.append_scalar(b"c_eval", &self.evaluations.c_eval);
            transcript.append_scalar(b"d_eval", &self.evaluations.d_eval);

            transcript.append_scalar(
                b"s_sigma_1_eval",
                &self.evaluations.s_sigma_1_eval,
            );
            transcript.append_scalar(
                b"s_sigma_2_eval",
                &self.evaluations.s_sigma_2_eval,
            );
            transcript.append_scalar(
                b"s_sigma_3_eval",
                &self.evaluations.s_sigma_3_eval,
            );

            transcript.append_scalar(b"z_eval", &self.evaluations.z_eval);

            transcript.append_scalar(b"a_w_eval", &self.evaluations.a_w_eval);
            transcript.append_scalar(b"b_w_eval", &self.evaluations.b_w_eval);
            transcript.append_scalar(b"d_w_eval", &self.evaluations.d_w_eval);
            transcript
                .append_scalar(b"q_arith_eval", &self.evaluations.q_arith_eval);
            transcript.append_scalar(b"q_c_eval", &self.evaluations.q_c_eval);
            transcript.append_scalar(b"q_l_eval", &self.evaluations.q_l_eval);
            transcript.append_scalar(b"q_r_eval", &self.evaluations.q_r_eval);

            let v_challenge = transcript.challenge_scalar(b"v_challenge");
            let v_w_challenge = transcript.challenge_scalar(b"v_w_challenge");

            transcript
                .append_commitment(b"w_z_chall_comm", &self.w_z_chall_comm);
            transcript
                .append_commitment(b"w_z_chall_w_comm", &self.w_z_chall_w_comm);

            let u_challenge = transcript.challenge_scalar(b"u_challenge");

            let z_h_eval = domain.evaluate_vanishing_polynomial(&z_challenge);

            let (l1_eval, pi_eval) =
                compute_lagrange_and_barycentric_evaluations(
                    public_input_roots,
                    pub_inputs,
                    &z_challenge,
                    &z_h_eval,
                    domain,
                )?;

            let r_0_eval = pi_eval
                - l1_eval * alpha.square()
                - alpha
                    * (self.evaluations.a_eval
                        + beta * self.evaluations.s_sigma_1_eval
                        + gamma)
                    * (self.evaluations.b_eval
                        + beta * self.evaluations.s_sigma_2_eval
                        + gamma)
                    * (self.evaluations.c_eval
                        + beta * self.evaluations.s_sigma_3_eval
                        + gamma)
                    * (self.evaluations.d_eval + gamma)
                    * self.evaluations.z_eval;

            let v_deg = if bind_selectors {
                V_MAX_DEGREE
            } else {
                V_MAX_DEGREE_LEGACY
            };

            let mut f_scalars = [BlsScalar::zero(); V_MAX_DEGREE];
            f_scalars[0] = v_challenge;
            for i in 1..v_deg {
                f_scalars[i] = f_scalars[i - 1] * v_challenge;
            }

            let u_vw = v_w_challenge * u_challenge;
            let u_vw2 = u_vw * v_w_challenge;
            let u_vw3 = u_vw2 * v_w_challenge;

            let mut E_scalar = self.evaluations.a_eval * f_scalars[0]
                + self.evaluations.b_eval * f_scalars[1]
                + self.evaluations.c_eval * f_scalars[2]
                + self.evaluations.d_eval * f_scalars[3]
                + self.evaluations.s_sigma_1_eval * f_scalars[4]
                + self.evaluations.s_sigma_2_eval * f_scalars[5]
                + self.evaluations.s_sigma_3_eval * f_scalars[6]
                + self.evaluations.a_w_eval * u_vw
                + self.evaluations.b_w_eval * u_vw2
                + self.evaluations.d_w_eval * u_vw3
                - r_0_eval
                + (u_challenge * self.evaluations.z_eval);

            if bind_selectors {
                E_scalar += self.evaluations.q_arith_eval * f_scalars[7]
                    + self.evaluations.q_c_eval * f_scalars[8]
                    + self.evaluations.q_l_eval * f_scalars[9]
                    + self.evaluations.q_r_eval * f_scalars[10];
            }

            // Group shifted wire coefficients onto [a]_1, [b]_1, [d]_1.
            f_scalars[0] += u_vw;
            f_scalars[1] += u_vw2;
            f_scalars[3] += u_vw3;

            let arith = arithmetic::VerifierKey::linearization_scalars(
                &self.evaluations,
            );
            let range_scalar = range::VerifierKey::linearization_scalar(
                &range_sep_challenge,
                &self.evaluations,
            );
            let logic_scalar = logic::VerifierKey::linearization_scalar(
                &logic_sep_challenge,
                &self.evaluations,
            );
            let fixed_base_scalar =
                ecc::scalar_mul::fixed_base::VerifierKey::linearization_scalar(
                    &fixed_base_sep_challenge,
                    &self.evaluations,
                );
            let var_base_scalar =
                ecc::curve_addition::VerifierKey::linearization_scalar(
                    &var_base_sep_challenge,
                    &self.evaluations,
                );
            let (perm_z_scalar, perm_s_sigma_4_scalar) =
                permutation::VerifierKey::linearization_scalars(
                    &self.evaluations,
                    &z_challenge,
                    &u_challenge,
                    (&alpha, &beta, &gamma),
                    &l1_eval,
                );

            let z_h_eval_neg = -z_h_eval;
            let z_pow_n = z_h_eval + BlsScalar::one();
            let z_n = z_pow_n * z_h_eval_neg;
            let z_two_n = z_pow_n.square() * z_h_eval_neg;
            let z_three_n = z_two_n * z_pow_n;
            let shifted_witness_scalar =
                u_challenge * z_challenge * domain.group_gen;

            // Static slot order matches `VerifierKey::static_points`:
            // [q_m, q_l, q_r, q_o, q_f, q_c, q_arith, q_range, q_logic,
            //  q_fixed_group_add, q_variable_group_add,
            //  s_sigma_1, s_sigma_2, s_sigma_3, s_sigma_4, opening_key.g]
            // Deduplicating q_l, q_r, q_c merges the arithmetic linearization
            // terms with the selector-binding opening terms.
            let mut static_scalars = [
                arith[0],
                arith[1],
                arith[2],
                arith[3],
                arith[4],
                arith[5],
                BlsScalar::zero(),
                range_scalar,
                logic_scalar,
                fixed_base_scalar,
                var_base_scalar,
                f_scalars[4],
                f_scalars[5],
                f_scalars[6],
                perm_s_sigma_4_scalar,
                -E_scalar,
            ];
            if bind_selectors {
                static_scalars[6] = f_scalars[7]; // q_arith
                static_scalars[5] += f_scalars[8]; // q_c
                static_scalars[1] += f_scalars[9]; // q_l
                static_scalars[2] += f_scalars[10]; // q_r
            }

            let dynamic_points = [
                self.a_comm.0,
                self.b_comm.0,
                self.c_comm.0,
                self.d_comm.0,
                self.z_comm.0,
                self.t_low_comm.0,
                self.t_mid_comm.0,
                self.t_high_comm.0,
                self.t_fourth_comm.0,
                self.w_z_chall_comm.0,
                self.w_z_chall_w_comm.0,
            ];
            let dynamic_scalars = [
                f_scalars[0],
                f_scalars[1],
                f_scalars[2],
                f_scalars[3],
                perm_z_scalar,
                z_h_eval_neg,
                z_n,
                z_two_n,
                z_three_n,
                z_challenge,
                shifted_witness_scalar,
            ];

            let prepared = PreparedVerification {
                static_scalars,
                dynamic_points,
                dynamic_scalars,
                w_z: self.w_z_chall_comm.0,
                w_zw: self.w_z_chall_w_comm.0,
                u_challenge,
            };

            #[cfg(test)]
            let ref_data = ReferenceCheckData {
                alpha,
                beta,
                gamma,
                range_sep_challenge,
                logic_sep_challenge,
                fixed_base_sep_challenge,
                var_base_sep_challenge,
                z_challenge,
                u_challenge,
                l1_eval,
                e_scalar: E_scalar,
                shifted_witness_scalar,
                f_scalars: f_scalars[..v_deg].to_vec(),
            };
            #[cfg(not(test))]
            let ref_data = ();

            Ok((prepared, ref_data))
        }

        /// Performs the verification of a [`Proof`] returning `Ok(())` if
        /// valid.
        #[allow(non_snake_case)]
        pub(crate) fn verify(
            &self,
            verifier_key: &VerifierKey,
            transcript: &mut Transcript,
            opening_key: &OpeningKey,
            domain: &EvaluationDomain,
            public_input_roots: &[BlsScalar],
            pub_inputs: &[BlsScalar],
            static_points: &[G1Affine; STATIC_POINT_COUNT],
            static_tables: &StaticVerifierTables,
        ) -> Result<(), Error> {
            self.verify_inner(
                verifier_key,
                transcript,
                opening_key,
                domain,
                public_input_roots,
                pub_inputs,
                static_points,
                static_tables,
                true,
            )
        }

        /// Performs proof verification using the legacy batching behavior
        /// (PLONK v1).
        #[allow(non_snake_case)]
        pub(crate) fn verify_legacy(
            &self,
            verifier_key: &VerifierKey,
            transcript: &mut Transcript,
            opening_key: &OpeningKey,
            domain: &EvaluationDomain,
            public_input_roots: &[BlsScalar],
            pub_inputs: &[BlsScalar],
            static_points: &[G1Affine; STATIC_POINT_COUNT],
            static_tables: &StaticVerifierTables,
        ) -> Result<(), Error> {
            self.verify_inner(
                verifier_key,
                transcript,
                opening_key,
                domain,
                public_input_roots,
                pub_inputs,
                static_points,
                static_tables,
                false,
            )
        }

        fn verify_inner(
            &self,
            #[cfg_attr(not(test), allow(unused_variables))]
            verifier_key: &VerifierKey,
            transcript: &mut Transcript,
            opening_key: &OpeningKey,
            domain: &EvaluationDomain,
            public_input_roots: &[BlsScalar],
            pub_inputs: &[BlsScalar],
            static_points: &[G1Affine; STATIC_POINT_COUNT],
            static_tables: &StaticVerifierTables,
            bind_selectors: bool,
        ) -> Result<(), Error> {
            let (prepared, _ref_data) = self.prepare_with_ref(
                transcript,
                domain,
                public_input_roots,
                pub_inputs,
                bind_selectors,
            )?;

            #[cfg(all(feature = "blst-backend", not(test)))]
            {
                let _ = (verifier_key, static_points);
                if !crate::blst_backend::verify_single_pipelined_blst(
                    &static_tables.blst,
                    &prepared.static_scalars,
                    &prepared.dynamic_points,
                    &prepared.dynamic_scalars,
                    &prepared.w_z,
                    &prepared.w_zw,
                    &prepared.u_challenge,
                    &opening_key.blst_lines_x_h,
                    &opening_key.blst_lines_h,
                ) {
                    return Err(Error::ProofVerificationError);
                }
                Ok(())
            }

            #[cfg(not(all(feature = "blst-backend", not(test))))]
            {
                let (left_projective, right_projective) = verifier_msm_single(
                    static_points,
                    static_tables,
                    &prepared.static_scalars,
                    &prepared.dynamic_points,
                    &prepared.dynamic_scalars,
                    &prepared.w_z,
                    &prepared.w_zw,
                    &prepared.u_challenge,
                );

                #[cfg(test)]
                {
                    let ref_data = _ref_data;
                    let f_points_full = [
                        self.a_comm.0,
                        self.b_comm.0,
                        self.c_comm.0,
                        self.d_comm.0,
                        verifier_key.permutation.s_sigma_1.0,
                        verifier_key.permutation.s_sigma_2.0,
                        verifier_key.permutation.s_sigma_3.0,
                        verifier_key.arithmetic.q_arith.0,
                        verifier_key.arithmetic.q_c.0,
                        verifier_key.arithmetic.q_l.0,
                        verifier_key.arithmetic.q_r.0,
                    ];
                    let f_points = if bind_selectors {
                        &f_points_full[..]
                    } else {
                        &f_points_full[..V_MAX_DEGREE_LEGACY]
                    };
                    assert_grouped_msm_equivalence(
                        &right_projective,
                        self.compute_reference_linearization_commitment(
                            &ref_data.alpha,
                            &ref_data.beta,
                            &ref_data.gamma,
                            (
                                &ref_data.range_sep_challenge,
                                &ref_data.logic_sep_challenge,
                                &ref_data.fixed_base_sep_challenge,
                                &ref_data.var_base_sep_challenge,
                            ),
                            &ref_data.z_challenge,
                            &ref_data.u_challenge,
                            ref_data.l1_eval,
                            verifier_key,
                            domain,
                        ),
                        f_points,
                        &ref_data.f_scalars,
                        opening_key.g,
                        ref_data.e_scalar,
                        self.w_z_chall_comm.0,
                        ref_data.z_challenge,
                        self.w_z_chall_w_comm.0,
                        ref_data.shifted_witness_scalar,
                    );
                }

                Self::verify_pairing(
                    opening_key,
                    left_projective,
                    right_projective,
                )
            }
        }

        /// Reconstruct the pre-regrouping linearization commitment directly
        /// from the widget reference helpers.
        ///
        /// Keeping the direct powers and the separate MSM here makes the
        /// equivalence assertion independent of the optimized verifier path.
        #[cfg(test)]
        fn compute_reference_linearization_commitment(
            &self,
            alpha: &BlsScalar,
            beta: &BlsScalar,
            gamma: &BlsScalar,
            (
                range_sep_challenge,
                logic_sep_challenge,
                fixed_base_sep_challenge,
                var_base_sep_challenge,
            ): (&BlsScalar, &BlsScalar, &BlsScalar, &BlsScalar),
            z_challenge: &BlsScalar,
            u_challenge: &BlsScalar,
            l1_eval: BlsScalar,
            verifier_key: &VerifierKey,
            domain: &EvaluationDomain,
        ) -> G1Projective {
            let mut scalars = Vec::with_capacity(10);
            let mut points = Vec::with_capacity(10);

            verifier_key.arithmetic.compute_linearization_commitment(
                &mut scalars,
                &mut points,
                &self.evaluations,
            );
            verifier_key.range.compute_linearization_commitment(
                range_sep_challenge,
                &mut scalars,
                &mut points,
                &self.evaluations,
            );
            verifier_key.logic.compute_linearization_commitment(
                logic_sep_challenge,
                &mut scalars,
                &mut points,
                &self.evaluations,
            );
            verifier_key.fixed_base.compute_linearization_commitment(
                fixed_base_sep_challenge,
                &mut scalars,
                &mut points,
                &self.evaluations,
            );
            verifier_key.variable_base.compute_linearization_commitment(
                var_base_sep_challenge,
                &mut scalars,
                &mut points,
                &self.evaluations,
            );
            verifier_key.permutation.compute_linearization_commitment(
                &mut scalars,
                &mut points,
                &self.evaluations,
                z_challenge,
                u_challenge,
                (alpha, beta, gamma),
                &l1_eval,
                self.z_comm.0,
            );

            let domain_size = domain.size() as u64;
            let z_h_eval = -domain.evaluate_vanishing_polynomial(z_challenge);

            scalars.push(z_h_eval);
            points.push(self.t_low_comm.0);
            scalars.push(z_challenge.pow(&[domain_size, 0, 0, 0]) * z_h_eval);
            points.push(self.t_mid_comm.0);
            scalars
                .push(z_challenge.pow(&[2 * domain_size, 0, 0, 0]) * z_h_eval);
            points.push(self.t_high_comm.0);
            scalars
                .push(z_challenge.pow(&[3 * domain_size, 0, 0, 0]) * z_h_eval);
            points.push(self.t_fourth_comm.0);

            msm_variable_base(&points, &scalars)
        }
    }

    #[cfg(test)]
    fn assert_grouped_msm_equivalence(
        grouped: &G1Projective,
        reference_linearization: G1Projective,
        f_points: &[G1Affine],
        f_scalars: &[BlsScalar],
        generator: G1Affine,
        e_scalar: BlsScalar,
        witness: G1Affine,
        z_challenge: BlsScalar,
        shifted_witness: G1Affine,
        shifted_witness_scalar: BlsScalar,
    ) {
        let expected = reference_linearization
            + msm_variable_base(f_points, f_scalars)
            - generator * e_scalar
            + witness * z_challenge
            + shifted_witness * shifted_witness_scalar;

        assert_eq!(*grouped, expected);
    }

    fn compute_lagrange_and_barycentric_evaluations(
        public_input_roots: &[BlsScalar],
        evaluations: &[BlsScalar],
        point: &BlsScalar,
        z_h_eval: &BlsScalar,
        domain: &EvaluationDomain,
    ) -> Result<(BlsScalar, BlsScalar), Error> {
        let non_zero_inputs: Vec<_> = public_input_roots
            .iter()
            .copied()
            .zip(evaluations.iter().copied())
            .filter(|(_, evaluation)| *evaluation != BlsScalar::zero())
            .collect();

        let mut denominators = Vec::with_capacity(non_zero_inputs.len() + 1);
        denominators
            .push(domain.size_as_field_element * (point - BlsScalar::one()));

        denominators.extend(
            non_zero_inputs
                .iter()
                .map(|(root, _)| (*root * point) - BlsScalar::one()),
        );

        if denominators
            .iter()
            .any(|denominator| denominator == &BlsScalar::zero())
        {
            return Err(Error::ProofVerificationError);
        }

        batch_inversion(&mut denominators);

        let l1_eval = z_h_eval * denominators[0];
        let pi_eval: BlsScalar = non_zero_inputs
            .iter()
            .zip(&denominators[1..])
            .map(|((_, evaluation), inverse)| *inverse * *evaluation)
            .sum::<BlsScalar>()
            * z_h_eval
            * domain.size_inv;

        Ok((l1_eval, pi_eval))
    }

    pub(crate) fn compute_barycentric_eval(
        evaluations: &[BlsScalar],
        point: &BlsScalar,
        domain: &EvaluationDomain,
    ) -> BlsScalar {
        if evaluations.is_empty() {
            return BlsScalar::zero();
        }

        let non_zero_evaluations: Vec<usize> = evaluations
            .iter()
            .enumerate()
            .filter_map(|(i, evaluation)| {
                (*evaluation != BlsScalar::zero()).then_some(i)
            })
            .collect();

        if non_zero_evaluations.is_empty() {
            return BlsScalar::zero();
        }

        let numerator = (point.pow(&[domain.size() as u64, 0, 0, 0])
            - BlsScalar::one())
            * domain.size_inv;

        // Only compute the denominators with non-zero evaluations
        #[cfg(not(feature = "std"))]
        let range = 0..non_zero_evaluations.len();

        #[cfg(feature = "std")]
        let range = (0..non_zero_evaluations.len()).into_par_iter();

        let mut denominators: Vec<BlsScalar> = range
            .clone()
            .map(|i| {
                // index of non-zero evaluation
                let index = non_zero_evaluations[i];

                (domain.group_gen_inv.pow(&[index as u64, 0, 0, 0]) * point)
                    - BlsScalar::one()
            })
            .collect();
        batch_inversion(&mut denominators);

        let result: BlsScalar = range
            .map(|i| {
                let eval_index = non_zero_evaluations[i];
                let eval = evaluations[eval_index];

                denominators[i] * eval
            })
            .sum();

        result * numerator
    }

    #[cfg(test)]
    fn compute_barycentric_eval_sparse(
        indexes: &[usize],
        evaluations: &[BlsScalar],
        point: &BlsScalar,
        domain: &EvaluationDomain,
    ) -> BlsScalar {
        if indexes.is_empty() {
            return BlsScalar::zero();
        }

        let numerator = (point.pow(&[domain.size() as u64, 0, 0, 0])
            - BlsScalar::one())
            * domain.size_inv;

        let non_zero_evaluations: Vec<_> = indexes
            .iter()
            .copied()
            .zip(evaluations.iter().copied())
            .filter(|(_, evaluation)| evaluation != &BlsScalar::zero())
            .collect();

        if non_zero_evaluations.is_empty() {
            return BlsScalar::zero();
        }

        let mut denominators: Vec<BlsScalar> = non_zero_evaluations
            .iter()
            .map(|(index, _)| {
                (domain.group_gen_inv.pow(&[*index as u64, 0, 0, 0]) * point)
                    - BlsScalar::one()
            })
            .collect();
        batch_inversion(&mut denominators);

        let result: BlsScalar = non_zero_evaluations
            .iter()
            .zip(denominators.iter())
            .map(|((_, evaluation), denominator)| *denominator * *evaluation)
            .sum();

        result * numerator
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fused_lagrange_and_sparse_evaluation_matches_separate_formulas() {
            let domain = EvaluationDomain::new(1 << 5).unwrap();
            let indexes = [0usize, 3, 17, 31];
            let evaluations = [
                BlsScalar::from(2u64),
                BlsScalar::zero(),
                BlsScalar::from(5u64),
                BlsScalar::from(9u64),
            ];
            let roots: Vec<_> = indexes
                .iter()
                .map(|index| {
                    domain.group_gen_inv.pow(&[*index as u64, 0, 0, 0])
                })
                .collect();
            let point = BlsScalar::from(7u64);
            let z_h_eval = domain.evaluate_vanishing_polynomial(&point);

            let (l1_eval, pi_eval) =
                compute_lagrange_and_barycentric_evaluations(
                    &roots,
                    &evaluations,
                    &point,
                    &z_h_eval,
                    &domain,
                )
                .unwrap();

            let l1_denominator =
                domain.size_as_field_element * (point - BlsScalar::one());
            let expected_l1 = z_h_eval * l1_denominator.invert().unwrap();
            let expected_pi = compute_barycentric_eval_sparse(
                &indexes,
                &evaluations,
                &point,
                &domain,
            );

            assert_eq!(l1_eval, expected_l1);
            assert_eq!(pi_eval, expected_pi);
        }

        #[test]
        fn fused_evaluation_rejects_a_domain_point() {
            let domain = EvaluationDomain::new(1 << 5).unwrap();
            let result = compute_lagrange_and_barycentric_evaluations(
                &[],
                &[],
                &BlsScalar::one(),
                &BlsScalar::zero(),
                &domain,
            );

            assert!(matches!(result, Err(Error::ProofVerificationError)));
        }

        #[test]
        fn fused_evaluation_rejects_a_public_input_domain_point() {
            let domain = EvaluationDomain::new(1 << 5).unwrap();
            let index = 3u64;
            let point = domain.group_gen.pow(&[index, 0, 0, 0]);
            let root = domain.group_gen_inv.pow(&[index, 0, 0, 0]);
            let result = compute_lagrange_and_barycentric_evaluations(
                &[root],
                &[BlsScalar::one()],
                &point,
                &BlsScalar::zero(),
                &domain,
            );

            assert!(matches!(result, Err(Error::ProofVerificationError)));
        }
    }
}

#[cfg(test)]
mod proof_tests {
    use dusk_bls12_381::BlsScalar;
    use ff::Field;
    use rand_core::OsRng;

    use super::*;

    #[test]
    fn test_dusk_bytes_serde_proof() {
        let proof = Proof {
            a_comm: Commitment::default(),
            b_comm: Commitment::default(),
            c_comm: Commitment::default(),
            d_comm: Commitment::default(),
            z_comm: Commitment::default(),
            t_low_comm: Commitment::default(),
            t_mid_comm: Commitment::default(),
            t_high_comm: Commitment::default(),
            t_fourth_comm: Commitment::default(),
            w_z_chall_comm: Commitment::default(),
            w_z_chall_w_comm: Commitment::default(),
            evaluations: ProofEvaluations {
                a_eval: BlsScalar::random(&mut OsRng),
                b_eval: BlsScalar::random(&mut OsRng),
                c_eval: BlsScalar::random(&mut OsRng),
                d_eval: BlsScalar::random(&mut OsRng),
                a_w_eval: BlsScalar::random(&mut OsRng),
                b_w_eval: BlsScalar::random(&mut OsRng),
                d_w_eval: BlsScalar::random(&mut OsRng),
                q_arith_eval: BlsScalar::random(&mut OsRng),
                q_c_eval: BlsScalar::random(&mut OsRng),
                q_l_eval: BlsScalar::random(&mut OsRng),
                q_r_eval: BlsScalar::random(&mut OsRng),
                s_sigma_1_eval: BlsScalar::random(&mut OsRng),
                s_sigma_2_eval: BlsScalar::random(&mut OsRng),
                s_sigma_3_eval: BlsScalar::random(&mut OsRng),
                z_eval: BlsScalar::random(&mut OsRng),
            },
        };

        let proof_bytes = proof.to_bytes();
        let got_proof = Proof::from_bytes(&proof_bytes).unwrap();
        assert_eq!(got_proof, proof);

        #[cfg(feature = "rkyv-impl")]
        {
            let bytes = rkyv::to_bytes::<_, 256>(&proof).unwrap();
            let decoded = rkyv::from_bytes::<Proof>(&bytes).unwrap();
            assert_eq!(decoded, proof);
            assert_eq!(
                rkyv::to_bytes::<_, 256>(&decoded).unwrap().as_slice(),
                bytes.as_slice()
            );
            assert!(
                rkyv::from_bytes::<Proof>(&bytes[..bytes.len() - 1]).is_err()
            );
        }
    }

    #[test]
    fn from_slice_exact_rejects_other_lengths() {
        let bytes = Proof::default().to_bytes();
        assert_eq!(Proof::from_slice_exact(&bytes), Ok(Proof::default()));

        let padded = [&bytes[..], &[0]].concat();
        for bad in [&padded[..], &bytes[..Proof::SIZE - 1]] {
            let found = bad.len();
            let expected = Proof::SIZE;
            let err = dusk_bytes::Error::BadLength { found, expected };
            assert_eq!(Proof::from_slice_exact(bad), Err(err));
        }
    }

    #[test]
    fn from_bytes_rejects_malformed_commitment_in_every_slot() {
        let valid = Proof::default().to_bytes();
        for slot in 0..11 {
            let mut corrupted = valid;
            // Clear compression bit / set invalid compressed encoding.
            corrupted[slot * Commitment::SIZE] = 0xff;
            assert_eq!(
                Proof::from_bytes(&corrupted),
                Err(dusk_bytes::Error::InvalidData),
                "commitment slot {slot} should fail deserialization"
            );
        }

        // Non-canonical scalar in evaluations tail (all 0xff > r).
        let mut bad_eval = valid;
        let eval_start = 11 * Commitment::SIZE;
        bad_eval[eval_start..eval_start + 32].fill(0xff);
        assert_eq!(
            Proof::from_bytes(&bad_eval),
            Err(dusk_bytes::Error::InvalidData)
        );
    }
}

/// Regression test for CVE: unbound selector evaluations in batch opening.
///
/// The selector evaluations `q_arith_eval`, `q_c_eval`, `q_l_eval`,
/// `q_r_eval` are included in `ProofEvaluations` but are NOT verified
/// against their polynomial commitments via the batch opening proof.
/// A malicious prover can forge these values after seeing `z_challenge`
/// to pass verification for a false statement.
///
/// This test constructs such a forged proof. If the vulnerability is
/// present, the forged proof passes verification and the test fails.
/// Once the fix is applied, verification rejects the proof and the test
/// passes.
#[cfg(test)]
#[cfg(feature = "std")]
mod soundness_tests {
    use dusk_bls12_381::BlsScalar;
    use ff::Field;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_core::{CryptoRng, RngCore};

    use crate::commitment_scheme::{CommitKey, PublicParameters};
    use crate::compiler::{Compiler, PlonkVersion, Prover, Verifier};
    use crate::composer::{Circuit, Composer, Constraint};
    use crate::error::Error;
    use crate::fft::{EvaluationDomain, Polynomial};
    use crate::proof_system::linearization_poly::{self, ProofEvaluations};
    use crate::proof_system::proof::{self, Proof};
    use crate::transcript::TranscriptProtocol;

    // Simple arithmetic circuit: a + b + a*b + d + public + 1 = result
    #[derive(Default)]
    struct ArithCircuit {
        a: BlsScalar,
        b: BlsScalar,
        d: BlsScalar,
        public: BlsScalar,
        result: BlsScalar,
    }

    impl Circuit for ArithCircuit {
        fn circuit(&self, composer: &mut Composer) -> Result<(), Error> {
            let w_a = composer.append_witness(self.a);
            let w_b = composer.append_witness(self.b);
            let w_d = composer.append_witness(self.d);
            let w_result = composer.append_witness(self.result);

            let constraint = Constraint::new()
                .left(1)
                .right(1)
                .mult(1)
                .fourth(1)
                .a(w_a)
                .b(w_b)
                .d(w_d)
                .public(self.public)
                .constant(BlsScalar::one());

            let result = composer.gate_add(constraint);
            composer.assert_equal(w_result, result);

            Ok(())
        }
    }

    /// Construct a forged proof exploiting the unbound selector evaluations.
    ///
    /// The proof uses:
    /// - Honest wire polynomials (circuit is satisfied at gate level)
    /// - A RANDOM permutation polynomial z (breaking the permutation argument —
    ///   gates are not properly wired together)
    /// - RANDOM quotient polynomials (not the actual quotient)
    /// - A FORGED `q_arith_eval` computed to balance the verification equation
    ///   after observing `z_challenge`
    ///
    /// This proof is invalid (the permutation argument does not hold)
    /// but exploits the fact that selector evaluations are not bound
    /// by the opening proof to make the pairing check pass.
    fn forge_proof(
        prover: &Prover,
        _verifier: &Verifier,
        circuit: &ArithCircuit,
        rng: &mut (impl RngCore + CryptoRng),
    ) -> (Proof, Vec<BlsScalar>) {
        let composed = Composer::prove(prover.constraints, circuit).unwrap();
        let size = prover.size;
        let domain = EvaluationDomain::new(prover.constraints).unwrap();

        let mut transcript = prover.transcript.clone();

        let public_inputs = composed.public_inputs();
        let public_input_indexes = composed.public_input_indexes();
        let dense_public_inputs = Composer::dense_public_inputs(
            &public_input_indexes,
            &public_inputs,
            prover.size,
        );

        public_inputs
            .iter()
            .for_each(|pi| transcript.append_scalar(b"pi", pi));

        // Round 1: honest wire polynomials
        let mut a_scalars = vec![BlsScalar::zero(); size];
        let mut b_scalars = vec![BlsScalar::zero(); size];
        let mut c_scalars = vec![BlsScalar::zero(); size];
        let mut d_scalars = vec![BlsScalar::zero(); size];

        composed
            .constraints
            .iter()
            .enumerate()
            .for_each(|(i, constraint)| {
                a_scalars[i] = composed[constraint.a];
                b_scalars[i] = composed[constraint.b];
                c_scalars[i] = composed[constraint.c];
                d_scalars[i] = composed[constraint.d];
            });

        let a_poly = Prover::blind_poly(rng, &a_scalars, 1, &domain);
        let b_poly = Prover::blind_poly(rng, &b_scalars, 1, &domain);
        let c_poly = Prover::blind_poly(rng, &c_scalars, 1, &domain);
        let d_poly = Prover::blind_poly(rng, &d_scalars, 1, &domain);

        let a_comm = prover.commit_key.commit(&a_poly).unwrap();
        let b_comm = prover.commit_key.commit(&b_poly).unwrap();
        let c_comm = prover.commit_key.commit(&c_poly).unwrap();
        let d_comm = prover.commit_key.commit(&d_poly).unwrap();

        transcript.append_commitment(b"a_comm", &a_comm);
        transcript.append_commitment(b"b_comm", &b_comm);
        transcript.append_commitment(b"c_comm", &c_comm);
        transcript.append_commitment(b"d_comm", &d_comm);

        // Round 2: RANDOM permutation polynomial (not honestly
        // computed). This breaks the permutation argument.
        let beta = transcript.challenge_scalar(b"beta");
        transcript.append_scalar(b"beta", &beta);
        let gamma = transcript.challenge_scalar(b"gamma");

        let mut z_scalars = vec![BlsScalar::zero(); size];
        for z in z_scalars.iter_mut() {
            *z = BlsScalar::random(&mut *rng);
        }
        let z_poly = Prover::blind_poly(rng, &z_scalars, 2, &domain);
        let z_comm = prover.commit_key.commit(&z_poly).unwrap();
        transcript.append_commitment(b"z_comm", &z_comm);

        // Round 3: RANDOM quotient polynomials
        let alpha = transcript.challenge_scalar(b"alpha");
        let range_sep_challenge =
            transcript.challenge_scalar(b"range separation challenge");
        let logic_sep_challenge =
            transcript.challenge_scalar(b"logic separation challenge");
        let fixed_base_sep_challenge =
            transcript.challenge_scalar(b"fixed base separation challenge");
        let var_base_sep_challenge =
            transcript.challenge_scalar(b"variable base separation challenge");

        let rand_linear_poly = |rng: &mut dyn RngCore| -> Polynomial {
            let c0 = BlsScalar::random(&mut *rng);
            let mut c1 = BlsScalar::random(&mut *rng);
            while c1 == BlsScalar::zero() {
                c1 = BlsScalar::random(&mut *rng);
            }
            Polynomial::from_coefficients_vec(vec![c0, c1])
        };

        let t_low_poly = rand_linear_poly(rng);
        let t_mid_poly = rand_linear_poly(rng);
        let t_high_poly = rand_linear_poly(rng);
        let t_fourth_poly = rand_linear_poly(rng);

        let t_low_comm = prover.commit_key.commit(&t_low_poly).unwrap();
        let t_mid_comm = prover.commit_key.commit(&t_mid_poly).unwrap();
        let t_high_comm = prover.commit_key.commit(&t_high_poly).unwrap();
        let t_fourth_comm = prover.commit_key.commit(&t_fourth_poly).unwrap();

        transcript.append_commitment(b"t_low_comm", &t_low_comm);
        transcript.append_commitment(b"t_mid_comm", &t_mid_comm);
        transcript.append_commitment(b"t_high_comm", &t_high_comm);
        transcript.append_commitment(b"t_fourth_comm", &t_fourth_comm);

        // Round 4: honest evaluations of wire and sigma
        // polynomials at z_challenge
        let z_challenge = transcript.challenge_scalar(b"z_challenge");

        let a_eval = a_poly.evaluate(&z_challenge);
        let b_eval = b_poly.evaluate(&z_challenge);
        let c_eval = c_poly.evaluate(&z_challenge);
        let d_eval = d_poly.evaluate(&z_challenge);

        let s_sigma_1_eval = prover
            .prover_key
            .permutation
            .s_sigma_1
            .0
            .evaluate(&z_challenge);
        let s_sigma_2_eval = prover
            .prover_key
            .permutation
            .s_sigma_2
            .0
            .evaluate(&z_challenge);
        let s_sigma_3_eval = prover
            .prover_key
            .permutation
            .s_sigma_3
            .0
            .evaluate(&z_challenge);

        let z_eval = z_poly.evaluate(&(z_challenge * domain.group_gen));

        transcript.append_scalar(b"a_eval", &a_eval);
        transcript.append_scalar(b"b_eval", &b_eval);
        transcript.append_scalar(b"c_eval", &c_eval);
        transcript.append_scalar(b"d_eval", &d_eval);
        transcript.append_scalar(b"s_sigma_1_eval", &s_sigma_1_eval);
        transcript.append_scalar(b"s_sigma_2_eval", &s_sigma_2_eval);
        transcript.append_scalar(b"s_sigma_3_eval", &s_sigma_3_eval);
        transcript.append_scalar(b"z_eval", &z_eval);

        let a_w_eval = a_poly.evaluate(&(z_challenge * domain.group_gen));
        let b_w_eval = b_poly.evaluate(&(z_challenge * domain.group_gen));
        let d_w_eval = d_poly.evaluate(&(z_challenge * domain.group_gen));

        // compute honest selector evaluations (except q_arith)
        let q_c_eval = prover.prover_key.logic.q_c.0.evaluate(&z_challenge);
        let q_l_eval =
            prover.prover_key.fixed_base.q_l.0.evaluate(&z_challenge);
        let q_r_eval =
            prover.prover_key.fixed_base.q_r.0.evaluate(&z_challenge);

        transcript.append_scalar(b"a_w_eval", &a_w_eval);
        transcript.append_scalar(b"b_w_eval", &b_w_eval);
        transcript.append_scalar(b"d_w_eval", &d_w_eval);

        // ---- FORGE q_arith_eval ----
        //
        // Compute the linearization poly value at z with
        // q_arith_eval = 0, then with q_arith_eval = 1, and solve
        // for the value that makes the verification equation hold.
        let q_arith_eval = {
            let z_h_eval = domain.evaluate_vanishing_polynomial(&z_challenge);
            let n_fr = BlsScalar::from(domain.size() as u64);
            let denom = n_fr * (z_challenge - BlsScalar::one());
            let _l1_eval = z_h_eval * denom.invert().unwrap();

            let pi_eval = proof::alloc::compute_barycentric_eval(
                &dense_public_inputs,
                &z_challenge,
                &domain,
            );

            let r_0_eval = pi_eval
                - _l1_eval * alpha.square()
                - alpha
                    * (a_eval + beta * s_sigma_1_eval + gamma)
                    * (b_eval + beta * s_sigma_2_eval + gamma)
                    * (c_eval + beta * s_sigma_3_eval + gamma)
                    * (d_eval + gamma)
                    * z_eval;

            // Evaluate linearization poly with q_arith_eval = 0
            let evals_q0 = ProofEvaluations {
                a_eval,
                b_eval,
                c_eval,
                d_eval,
                a_w_eval,
                b_w_eval,
                d_w_eval,
                q_arith_eval: BlsScalar::zero(),
                q_c_eval,
                q_l_eval,
                q_r_eval,
                s_sigma_1_eval,
                s_sigma_2_eval,
                s_sigma_3_eval,
                z_eval,
            };

            let r_poly_q0 = linearization_poly::compute(
                &prover.prover_key,
                &linearization_poly::LinearizationChallenges {
                    alpha,
                    beta,
                    gamma,
                    range_separation: range_sep_challenge,
                    logic_separation: logic_sep_challenge,
                    fixed_base_separation: fixed_base_sep_challenge,
                    variable_base_separation: var_base_sep_challenge,
                    z: z_challenge,
                },
                &z_poly,
                &evals_q0,
                &domain,
                &t_low_poly,
                &t_mid_poly,
                &t_high_poly,
                &t_fourth_poly,
                &dense_public_inputs,
            );
            let r_q0_at_z = r_poly_q0.evaluate(&z_challenge);

            // Evaluate the per-unit contribution of q_arith_eval
            let evals_q1 = ProofEvaluations {
                q_arith_eval: BlsScalar::one(),
                ..evals_q0
            };
            let arith_base_poly = prover
                .prover_key
                .arithmetic
                .compute_linearization(&evals_q1);
            let arith_base_at_z = arith_base_poly.evaluate(&z_challenge);

            // Solve for the q_arith_eval that balances r(z) = target
            let target_r_at_z = -r_0_eval + pi_eval;
            (target_r_at_z - r_q0_at_z) * arith_base_at_z.invert().unwrap()
        };

        // Append forged selector evaluations to transcript
        transcript.append_scalar(b"q_arith_eval", &q_arith_eval);
        transcript.append_scalar(b"q_c_eval", &q_c_eval);
        transcript.append_scalar(b"q_l_eval", &q_l_eval);
        transcript.append_scalar(b"q_r_eval", &q_r_eval);

        let evaluations = ProofEvaluations {
            a_eval,
            b_eval,
            c_eval,
            d_eval,
            a_w_eval,
            b_w_eval,
            d_w_eval,
            q_arith_eval,
            q_c_eval,
            q_l_eval,
            q_r_eval,
            s_sigma_1_eval,
            s_sigma_2_eval,
            s_sigma_3_eval,
            z_eval,
        };

        // Round 5: compute opening proofs using forged evaluations
        let v_challenge = transcript.challenge_scalar(b"v_challenge");

        let r_poly = linearization_poly::compute(
            &prover.prover_key,
            &linearization_poly::LinearizationChallenges {
                alpha,
                beta,
                gamma,
                range_separation: range_sep_challenge,
                logic_separation: logic_sep_challenge,
                fixed_base_separation: fixed_base_sep_challenge,
                variable_base_separation: var_base_sep_challenge,
                z: z_challenge,
            },
            &z_poly,
            &evaluations,
            &domain,
            &t_low_poly,
            &t_mid_poly,
            &t_high_poly,
            &t_fourth_poly,
            &dense_public_inputs,
        );

        let aggregate_witness = CommitKey::compute_aggregate_witness(
            &[
                &r_poly,
                &a_poly,
                &b_poly,
                &c_poly,
                &d_poly,
                &prover.prover_key.permutation.s_sigma_1.0,
                &prover.prover_key.permutation.s_sigma_2.0,
                &prover.prover_key.permutation.s_sigma_3.0,
            ],
            &z_challenge,
            &v_challenge,
        );
        let w_z_chall_comm =
            prover.commit_key.commit(&aggregate_witness).unwrap();

        let v_w_challenge = transcript.challenge_scalar(b"v_w_challenge");

        let shifted_aggregate_witness = CommitKey::compute_aggregate_witness(
            &[&z_poly, &a_poly, &b_poly, &d_poly],
            &(z_challenge * domain.group_gen),
            &v_w_challenge,
        );
        let w_z_chall_w_comm = prover
            .commit_key
            .commit(&shifted_aggregate_witness)
            .unwrap();

        let proof = Proof {
            a_comm,
            b_comm,
            c_comm,
            d_comm,
            z_comm,
            t_low_comm,
            t_mid_comm,
            t_high_comm,
            t_fourth_comm,
            w_z_chall_comm,
            w_z_chall_w_comm,
            evaluations,
        };

        (proof, public_inputs)
    }

    /// Verify that a forged proof exploiting unbound selector evaluations
    /// is rejected by the verifier.
    ///
    /// This test FAILS when the vulnerability is present (the forged proof
    /// passes verification). After applying the fix (binding selector
    /// evaluations in the batch opening proof), this test PASSES.
    #[test]
    fn forged_selector_eval_proof_must_be_rejected() {
        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let capacity = 1 << 4;
        let pp =
            PublicParameters::setup(capacity, &mut rng).expect("setup failed");

        let a = BlsScalar::from(3);
        let b = BlsScalar::from(5);
        let d = BlsScalar::from(7);
        let public = BlsScalar::from(11);
        let result = a + b + a * b + d + public + BlsScalar::one();

        let circuit = ArithCircuit {
            a,
            b,
            d,
            public,
            result,
        };

        let (prover, verifier) =
            Compiler::compile::<ArithCircuit>(&pp, b"soundness_test")
                .expect("compile failed");

        // Sanity: honest proof passes
        let (honest_proof, honest_pi) = prover
            .prove(&mut rng, &circuit)
            .expect("honest proof failed");
        verifier
            .verify(&honest_proof, &honest_pi)
            .expect("honest proof should verify");
        assert!(matches!(
            verifier.verify_with_version(
                &honest_proof,
                &honest_pi,
                PlonkVersion::V1,
            ),
            Err(Error::ProofVerificationError)
        ));

        // Forged proof must NOT pass
        let (forged_proof, forged_pi) =
            forge_proof(&prover, &verifier, &circuit, &mut rng);

        assert!(
            verifier.verify(&forged_proof, &forged_pi).is_err(),
            "VULNERABILITY PRESENT: forged proof with fabricated \
             selector evaluations was accepted by the verifier. \
             q_arith_eval, q_c_eval, q_l_eval, and q_r_eval are \
             not bound by the batch opening proof, allowing a \
             malicious prover to forge them after seeing z_challenge."
        );
        assert!(
            verifier
                .verify_batch(&[
                    (&honest_proof, &honest_pi),
                    (&forged_proof, &forged_pi),
                ])
                .is_err(),
            "forged selector-eval proof must also be rejected in verify_batch"
        );
    }
}
