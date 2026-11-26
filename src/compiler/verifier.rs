// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

use alloc::vec::Vec;

use dusk_bls12_381::{BlsScalar, G1Affine};
use dusk_bytes::{DeserializableSlice, Serializable};
use merlin::Transcript;

use super::PlonkVersion;
use crate::commitment_scheme::OpeningKey;
use crate::error::Error;
use crate::fft::EvaluationDomain;
use crate::proof_system::proof::alloc::PreparedVerification;
#[cfg(not(feature = "blst-backend"))]
use crate::proof_system::verifier_msm::msm_public_variable_base;
use crate::proof_system::verifier_msm::{
    DYNAMIC_POINT_COUNT, STATIC_POINT_COUNT, StaticVerifierTables,
};
use crate::proof_system::{Proof, VerifierKey};
use crate::transcript::TranscriptProtocol;
/// Verify proofs of a given circuit
pub struct Verifier {
    label: Vec<u8>,
    verifier_key: VerifierKey,
    opening_key: OpeningKey,
    public_input_indexes: Vec<usize>,
    public_input_roots: Vec<BlsScalar>,
    domain: EvaluationDomain,
    transcript: Transcript,
    transcript_v3: Transcript,
    static_points: [G1Affine; STATIC_POINT_COUNT],
    static_tables: StaticVerifierTables,
    size: usize,
    constraints: usize,
}

impl Verifier {
    pub(crate) fn new(
        label: Vec<u8>,
        verifier_key: VerifierKey,
        opening_key: OpeningKey,
        public_input_indexes: Vec<usize>,
        size: usize,
        constraints: usize,
    ) -> Result<Self, Error> {
        let domain = EvaluationDomain::new(verifier_key.n)?;

        if constraints.checked_next_power_of_two() != Some(size)
            || verifier_key.n != constraints
            || public_input_indexes
                .iter()
                .any(|index| *index >= constraints)
            || public_input_indexes
                .windows(2)
                .any(|indexes| indexes[0] >= indexes[1])
        {
            return Err(dusk_bytes::Error::InvalidData.into());
        }
        let public_input_roots = public_input_indexes
            .iter()
            .map(|index| domain.group_gen_inv.pow(&[*index as u64, 0, 0, 0]))
            .collect();
        let transcript =
            Transcript::base(label.as_slice(), &verifier_key, constraints);
        let transcript_v3 =
            Transcript::base_v3(label.as_slice(), &verifier_key, constraints);
        let static_points = verifier_key.static_points(opening_key.g);
        let static_tables = StaticVerifierTables::new(&static_points);

        Ok(Self {
            label,
            verifier_key,
            opening_key,
            public_input_indexes,
            public_input_roots,
            domain,
            transcript,
            transcript_v3,
            static_points,
            static_tables,
            size,
            constraints,
        })
    }

    fn prepare_serialize(
        &self,
    ) -> (usize, [u8; VerifierKey::SIZE], [u8; OpeningKey::SIZE]) {
        let verifier_key = self.verifier_key.to_bytes();
        let opening_key = self.opening_key.to_bytes();

        let label_len = self.label.len();
        let verifier_key_len = verifier_key.len();
        let opening_key_len = opening_key.len();
        let public_input_indexes_len = self.public_input_indexes.len() * 8;

        let size = 48
            + label_len
            + verifier_key_len
            + opening_key_len
            + public_input_indexes_len;

        (size, verifier_key, opening_key)
    }

    /// Serialized size in bytes
    pub fn serialized_size(&self) -> usize {
        self.prepare_serialize().0
    }

    /// Serialize the verifier into bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let (size, verifier_key, opening_key) = self.prepare_serialize();
        let mut bytes = Vec::with_capacity(size);

        let label_len = self.label.len() as u64;
        let verifier_key_len = verifier_key.len() as u64;
        let opening_key_len = opening_key.len() as u64;
        let public_input_indexes_len = self.public_input_indexes.len() as u64;
        let size = self.size as u64;
        let constraints = self.constraints as u64;

        bytes.extend(label_len.to_be_bytes());
        bytes.extend(verifier_key_len.to_be_bytes());
        bytes.extend(opening_key_len.to_be_bytes());
        bytes.extend(public_input_indexes_len.to_be_bytes());
        bytes.extend(size.to_be_bytes());
        bytes.extend(constraints.to_be_bytes());

        bytes.extend(self.label.as_slice());
        bytes.extend(verifier_key);
        bytes.extend(opening_key);

        self.public_input_indexes
            .iter()
            .map(|i| *i as u64)
            .map(u64::to_be_bytes)
            .for_each(|i| bytes.extend(i));

        bytes
    }

    /// Attempt to deserialize the verifier from bytes generated via
    /// [`Self::to_bytes`].
    ///
    /// Serialized values and byte-length arithmetic that overflow `usize`
    /// return [`Error::BytesError`] containing
    /// [`dusk_bytes::Error::InvalidData`]. Short reads retain
    /// [`Error::NotEnoughBytes`] or nested [`dusk_bytes::Error::BadLength`].
    pub fn try_from_bytes<B>(bytes: B) -> Result<Self, Error>
    where
        B: AsRef<[u8]>,
    {
        let mut bytes = bytes.as_ref();

        if bytes.len() < 48 {
            return Err(Error::NotEnoughBytes);
        }

        let label_len = <[u8; 8]>::try_from(&bytes[..8]).expect("checked len");
        let label_len = usize::try_from(u64::from_be_bytes(label_len))
            .map_err(|_| dusk_bytes::Error::InvalidData)?;
        bytes = &bytes[8..];

        let verifier_key_len =
            <[u8; 8]>::try_from(&bytes[..8]).expect("checked len");
        let verifier_key_len =
            usize::try_from(u64::from_be_bytes(verifier_key_len))
                .map_err(|_| dusk_bytes::Error::InvalidData)?;
        bytes = &bytes[8..];

        let opening_key_len =
            <[u8; 8]>::try_from(&bytes[..8]).expect("checked len");
        let opening_key_len =
            usize::try_from(u64::from_be_bytes(opening_key_len))
                .map_err(|_| dusk_bytes::Error::InvalidData)?;
        bytes = &bytes[8..];

        let public_input_indexes_len =
            <[u8; 8]>::try_from(&bytes[..8]).expect("checked len");
        let public_input_indexes_len =
            usize::try_from(u64::from_be_bytes(public_input_indexes_len))
                .map_err(|_| dusk_bytes::Error::InvalidData)?;
        bytes = &bytes[8..];
        let public_input_indexes_bytes_len = public_input_indexes_len
            .checked_mul(8)
            .ok_or(dusk_bytes::Error::InvalidData)?;

        let size = <[u8; 8]>::try_from(&bytes[..8]).expect("checked len");
        let size = usize::try_from(u64::from_be_bytes(size))
            .map_err(|_| dusk_bytes::Error::InvalidData)?;
        bytes = &bytes[8..];

        let constraints =
            <[u8; 8]>::try_from(&bytes[..8]).expect("checked len");
        let constraints = usize::try_from(u64::from_be_bytes(constraints))
            .map_err(|_| dusk_bytes::Error::InvalidData)?;
        bytes = &bytes[8..];

        if public_input_indexes_len > constraints {
            return Err(dusk_bytes::Error::InvalidData.into());
        }

        let required_len = label_len
            .checked_add(verifier_key_len)
            .and_then(|len| len.checked_add(opening_key_len))
            .and_then(|len| len.checked_add(public_input_indexes_bytes_len))
            .ok_or(dusk_bytes::Error::InvalidData)?;

        if bytes.len() < required_len {
            return Err(Error::NotEnoughBytes);
        }
        if bytes.len() > required_len {
            return Err(dusk_bytes::Error::InvalidData.into());
        }

        // Both keys have fixed-size encodings: a longer segment would decode
        // from its prefix.
        if verifier_key_len > VerifierKey::SIZE
            || opening_key_len > OpeningKey::SIZE
        {
            return Err(dusk_bytes::Error::InvalidData.into());
        }

        let label = &bytes[..label_len];
        bytes = &bytes[label_len..];

        let verifier_key = &bytes[..verifier_key_len];
        bytes = &bytes[verifier_key_len..];

        let opening_key = &bytes[..opening_key_len];
        bytes = &bytes[opening_key_len..];

        let public_input_indexes = &bytes[..public_input_indexes_bytes_len];

        let label = label.to_vec();
        let verifier_key = VerifierKey::from_slice(verifier_key)?;
        let opening_key = OpeningKey::from_slice(opening_key)?;
        let public_input_indexes = public_input_indexes
            .as_chunks::<{ u64::SIZE }>()
            .0
            .iter()
            .map(|bytes| {
                let index = u64::from_be_bytes(*bytes);
                usize::try_from(index)
                    .map_err(|_| dusk_bytes::Error::InvalidData)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::new(
            label,
            verifier_key,
            opening_key,
            public_input_indexes,
            size,
            constraints,
        )
    }

    /// Verify a generated proof using the current (latest) verification
    /// behavior.
    pub fn verify(
        &self,
        proof: &Proof,
        public_inputs: &[BlsScalar],
    ) -> Result<(), Error> {
        self.verify_with_version(proof, public_inputs, PlonkVersion::current())
    }

    /// Verify a generated proof using an explicitly selected version.
    pub fn verify_with_version(
        &self,
        proof: &Proof,
        public_inputs: &[BlsScalar],
        version: PlonkVersion,
    ) -> Result<(), Error> {
        if public_inputs.len() != self.public_input_indexes.len() {
            return Err(Error::InconsistentPublicInputsLen {
                expected: self.public_input_indexes.len(),
                provided: public_inputs.len(),
            });
        }

        let mut transcript = self.transcript_for_version(version);

        public_inputs
            .iter()
            .for_each(|pi| transcript.append_scalar(b"pi", pi));

        match version {
            PlonkVersion::V1 => proof.verify_legacy(
                &self.verifier_key,
                &mut transcript,
                &self.opening_key,
                &self.domain,
                &self.public_input_roots,
                public_inputs,
                &self.static_points,
                &self.static_tables,
            ),
            PlonkVersion::V2 | PlonkVersion::V3 => proof.verify(
                &self.verifier_key,
                &mut transcript,
                &self.opening_key,
                &self.domain,
                &self.public_input_roots,
                public_inputs,
                &self.static_points,
                &self.static_tables,
            ),
        }
    }

    /// Verify a batch of proofs for this circuit using the current (latest)
    /// verification behavior.
    ///
    /// Shares a single BLS12-381 pairing check and merges the 16 static
    /// verifier commitments across the batch. Returns `Ok(())` when `items` is
    /// empty or every proof is valid.
    pub fn verify_batch<P>(&self, items: &[(&Proof, P)]) -> Result<(), Error>
    where
        P: AsRef<[BlsScalar]> + Sync,
    {
        self.verify_batch_with_version(items, PlonkVersion::current())
    }

    /// Verify a batch of proofs for this circuit using an explicitly selected
    /// version.
    ///
    /// Shares a single BLS12-381 pairing check and merges the 16 static
    /// verifier commitments across the batch. Returns `Ok(())` when `items` is
    /// empty or every proof is valid.
    pub fn verify_batch_with_version<P>(
        &self,
        items: &[(&Proof, P)],
        version: PlonkVersion,
    ) -> Result<(), Error>
    where
        P: AsRef<[BlsScalar]> + Sync,
    {
        if items.is_empty() {
            return Ok(());
        }
        if items.len() == 1 {
            return self.verify_with_version(
                items[0].0,
                items[0].1.as_ref(),
                version,
            );
        }

        for (_, pi) in items {
            let pi = pi.as_ref();
            if pi.len() != self.public_input_indexes.len() {
                return Err(Error::InconsistentPublicInputsLen {
                    expected: self.public_input_indexes.len(),
                    provided: pi.len(),
                });
            }
        }

        let mut batch_transcript =
            Transcript::new(b"dusk-plonk-verify-batch-v1");
        self.opening_key.append_to_transcript(&mut batch_transcript);
        batch_transcript.append_u64(b"version", version_tag(version));
        self.append_context_to_batch_transcript(&mut batch_transcript);
        batch_transcript.append_u64(b"batch-len", items.len() as u64);
        for (proof, pi) in items {
            proof.append_to_batch_transcript(&mut batch_transcript);
            let pi = pi.as_ref();
            batch_transcript.append_u64(b"pi-len", pi.len() as u64);
            for scalar in pi {
                batch_transcript.append_scalar(b"pi", scalar);
            }
        }
        let rhos = sample_batch_randomizers(&mut batch_transcript, items.len());

        let prepared = self.prepare_batch_items(items, version)?;

        let mut merged_static = [BlsScalar::zero(); STATIC_POINT_COUNT];
        #[cfg(feature = "blst-backend")]
        {
            let mut left_pts = Vec::with_capacity(2 * items.len());
            let mut left_scs = Vec::with_capacity(2 * items.len());
            let mut right_dyn_pts =
                Vec::with_capacity(DYNAMIC_POINT_COUNT * items.len());
            let mut right_dyn_scs =
                Vec::with_capacity(DYNAMIC_POINT_COUNT * items.len());

            for (prep, &rho) in prepared.iter().zip(rhos.iter()) {
                if rho == BlsScalar::one() {
                    for (dst, src) in
                        merged_static.iter_mut().zip(prep.static_scalars.iter())
                    {
                        *dst += *src;
                    }
                    left_pts.push(prep.w_z);
                    left_scs.push(BlsScalar::one());
                    left_pts.push(prep.w_zw);
                    left_scs.push(prep.u_challenge);
                    right_dyn_pts.extend_from_slice(&prep.dynamic_points);
                    right_dyn_scs.extend_from_slice(&prep.dynamic_scalars);
                } else {
                    for (dst, src) in
                        merged_static.iter_mut().zip(prep.static_scalars.iter())
                    {
                        *dst += rho * *src;
                    }
                    left_pts.push(prep.w_z);
                    left_scs.push(rho);
                    left_pts.push(prep.w_zw);
                    left_scs.push(rho * prep.u_challenge);
                    right_dyn_pts.extend_from_slice(&prep.dynamic_points);
                    for sc in &prep.dynamic_scalars {
                        right_dyn_scs.push(rho * *sc);
                    }
                }
            }

            let (left, right) = crate::blst_backend::verify_batch_msm_blst(
                &left_pts,
                &left_scs,
                &right_dyn_pts,
                &right_dyn_scs,
                Some((&self.static_tables.blst, &merged_static)),
            );
            Proof::verify_pairing_affine(&self.opening_key, left, right)
        }
        #[cfg(not(feature = "blst-backend"))]
        {
            let mut left_pts = Vec::with_capacity(2 * items.len());
            let mut left_scs = Vec::with_capacity(2 * items.len());
            let mut right_pts = Vec::with_capacity(
                STATIC_POINT_COUNT + DYNAMIC_POINT_COUNT * items.len(),
            );
            let mut right_scs = Vec::with_capacity(
                STATIC_POINT_COUNT + DYNAMIC_POINT_COUNT * items.len(),
            );

            right_pts.extend_from_slice(&self.static_points);
            right_scs.resize(STATIC_POINT_COUNT, BlsScalar::zero());

            for (prep, &rho) in prepared.iter().zip(rhos.iter()) {
                if rho == BlsScalar::one() {
                    for (dst, src) in
                        merged_static.iter_mut().zip(prep.static_scalars.iter())
                    {
                        *dst += *src;
                    }
                    left_pts.push(prep.w_z);
                    left_scs.push(BlsScalar::one());
                    left_pts.push(prep.w_zw);
                    left_scs.push(prep.u_challenge);
                    right_pts.extend_from_slice(&prep.dynamic_points);
                    right_scs.extend_from_slice(&prep.dynamic_scalars);
                } else {
                    for (dst, src) in
                        merged_static.iter_mut().zip(prep.static_scalars.iter())
                    {
                        *dst += rho * *src;
                    }
                    left_pts.push(prep.w_z);
                    left_scs.push(rho);
                    left_pts.push(prep.w_zw);
                    left_scs.push(rho * prep.u_challenge);
                    right_pts.extend_from_slice(&prep.dynamic_points);
                    for sc in &prep.dynamic_scalars {
                        right_scs.push(rho * *sc);
                    }
                }
            }
            right_scs[..STATIC_POINT_COUNT].copy_from_slice(&merged_static);

            let left = -msm_public_variable_base(&left_pts, &left_scs);
            let right = msm_public_variable_base(&right_pts, &right_scs);
            Proof::verify_pairing(&self.opening_key, left, right)
        }
    }

    fn prepare_one(
        &self,
        proof: &Proof,
        public_inputs: &[BlsScalar],
        version: PlonkVersion,
    ) -> Result<PreparedVerification, Error> {
        let mut transcript = self.transcript_for_version(version);
        for pi in public_inputs {
            transcript.append_scalar(b"pi", pi);
        }
        let bind_selectors = !matches!(version, PlonkVersion::V1);
        proof.prepare_verification(
            &mut transcript,
            &self.domain,
            &self.public_input_roots,
            public_inputs,
            bind_selectors,
        )
    }

    fn prepare_batch_items<P>(
        &self,
        items: &[(&Proof, P)],
        version: PlonkVersion,
    ) -> Result<Vec<PreparedVerification>, Error>
    where
        P: AsRef<[BlsScalar]> + Sync,
    {
        #[cfg(feature = "std")]
        if items.len() >= 4 && rayon::current_num_threads() > 1 {
            use rayon::prelude::*;

            return items
                .par_iter()
                .with_min_len(2)
                .map(|(proof, pi)| {
                    self.prepare_one(proof, pi.as_ref(), version)
                })
                .collect();
        }

        items
            .iter()
            .map(|(proof, pi)| self.prepare_one(proof, pi.as_ref(), version))
            .collect()
    }

    fn append_context_to_batch_transcript(&self, transcript: &mut Transcript) {
        transcript.append_message(b"label", &self.label);
        transcript.append_u64(b"constraints", self.constraints as u64);
        transcript
            .append_u64(b"pi-idx-len", self.public_input_indexes.len() as u64);
        for &idx in &self.public_input_indexes {
            transcript.append_u64(b"pi-idx", idx as u64);
        }
        self.verifier_key.seed_transcript(transcript);
    }

    fn transcript_for_version(&self, version: PlonkVersion) -> Transcript {
        match version {
            PlonkVersion::V1 | PlonkVersion::V2 => self.transcript.clone(),
            PlonkVersion::V3 => self.transcript_v3.clone(),
        }
    }
}

const fn version_tag(version: PlonkVersion) -> u64 {
    match version {
        PlonkVersion::V1 => 1,
        PlonkVersion::V2 => 2,
        PlonkVersion::V3 => 3,
    }
}

fn sample_batch_randomizers(
    transcript: &mut Transcript,
    count: usize,
) -> Vec<BlsScalar> {
    let mut rhos = Vec::with_capacity(count);
    if count == 0 {
        return rhos;
    }
    rhos.push(BlsScalar::one());
    for _ in 1..count {
        let mut rho = transcript.challenge_scalar(b"batch-rho");
        while rho == BlsScalar::zero() {
            rho = transcript.challenge_scalar(b"batch-rho");
        }
        rhos.push(rho);
    }
    rhos
}

struct BatchEntry<'a> {
    verifier: &'a Verifier,
    proof: &'a Proof,
    public_inputs: &'a [BlsScalar],
    version: PlonkVersion,
}

/// Accumulates proofs across one or more [`Verifier`] instances that share the
/// same SRS opening key and verifies them together in a single batched
/// multi-scalar multiplication and BLS12-381 pairing check.
///
/// Proofs sharing the same circuit verification key automatically fold their
/// static verifier-key commitments together, while all proofs across all
/// circuits share the opening-key generator and a single 2-pair pairing.
#[derive(Default)]
pub struct BatchVerifier<'a> {
    opening_key: Option<&'a OpeningKey>,
    entries: Vec<BatchEntry<'a>>,
}

impl<'a> BatchVerifier<'a> {
    /// Creates an empty batch verifier.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty batch verifier with preallocated entry capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            opening_key: None,
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of proofs currently queued in the batch.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no proofs have been pushed to the batch.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Adds a proof and its public inputs for `verifier` using
    /// [`PlonkVersion::current`].
    ///
    /// Returns [`Error::InconsistentPublicInputsLen`] if `public_inputs` does
    /// not match `verifier`, or [`Error::IncompatibleOpeningKey`] if
    /// `verifier` uses a different SRS opening key than previously pushed
    /// verifiers.
    pub fn push(
        &mut self,
        verifier: &'a Verifier,
        proof: &'a Proof,
        public_inputs: &'a [BlsScalar],
    ) -> Result<(), Error> {
        self.push_with_version(
            verifier,
            proof,
            public_inputs,
            PlonkVersion::current(),
        )
    }

    /// Adds a proof and its public inputs for `verifier` under an explicitly
    /// selected [`PlonkVersion`].
    ///
    /// Returns [`Error::InconsistentPublicInputsLen`] if `public_inputs` does
    /// not match `verifier`, or [`Error::IncompatibleOpeningKey`] if
    /// `verifier` uses a different SRS opening key than previously pushed
    /// verifiers.
    pub fn push_with_version(
        &mut self,
        verifier: &'a Verifier,
        proof: &'a Proof,
        public_inputs: &'a [BlsScalar],
        version: PlonkVersion,
    ) -> Result<(), Error> {
        if public_inputs.len() != verifier.public_input_indexes.len() {
            return Err(Error::InconsistentPublicInputsLen {
                expected: verifier.public_input_indexes.len(),
                provided: public_inputs.len(),
            });
        }

        match self.opening_key {
            Some(existing) if !existing.matches(&verifier.opening_key) => {
                return Err(Error::IncompatibleOpeningKey);
            }
            None => self.opening_key = Some(&verifier.opening_key),
            Some(_) => {}
        }

        self.entries.push(BatchEntry {
            verifier,
            proof,
            public_inputs,
            version,
        });
        Ok(())
    }

    /// Verifies all queued proofs in a single batched check.
    ///
    /// Returns `Ok(())` when the batch is empty or every queued proof is
    /// valid, or [`Error::ProofVerificationError`] if any proof fails.
    pub fn verify(self) -> Result<(), Error> {
        const VK_STATIC_COUNT: usize = STATIC_POINT_COUNT - 1;

        if self.entries.is_empty() {
            return Ok(());
        }
        if self.entries.len() == 1 {
            let entry = &self.entries[0];
            return entry.verifier.verify_with_version(
                entry.proof,
                entry.public_inputs,
                entry.version,
            );
        }

        let opening_key = self
            .opening_key
            .expect("non-empty batch has a validated opening key");

        let mut batch_transcript =
            Transcript::new(b"dusk-plonk-batch-verifier-v1");
        opening_key.append_to_transcript(&mut batch_transcript);
        batch_transcript.append_u64(b"batch-len", self.entries.len() as u64);
        for entry in &self.entries {
            batch_transcript.append_u64(b"version", version_tag(entry.version));
            entry
                .verifier
                .append_context_to_batch_transcript(&mut batch_transcript);
            entry
                .proof
                .append_to_batch_transcript(&mut batch_transcript);
            batch_transcript
                .append_u64(b"pi-len", entry.public_inputs.len() as u64);
            for scalar in entry.public_inputs {
                batch_transcript.append_scalar(b"pi", scalar);
            }
        }
        let rhos =
            sample_batch_randomizers(&mut batch_transcript, self.entries.len());

        let prepare_entry = |entry: &BatchEntry<'a>| {
            entry.verifier.prepare_one(
                entry.proof,
                entry.public_inputs,
                entry.version,
            )
        };

        #[cfg(feature = "std")]
        let prepared: Vec<PreparedVerification> =
            if self.entries.len() >= 4 && rayon::current_num_threads() > 1 {
                use rayon::prelude::*;

                self.entries
                    .par_iter()
                    .with_min_len(2)
                    .map(prepare_entry)
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                self.entries
                    .iter()
                    .map(prepare_entry)
                    .collect::<Result<Vec<_>, _>>()?
            };

        #[cfg(not(feature = "std"))]
        let prepared: Vec<PreparedVerification> = self
            .entries
            .iter()
            .map(prepare_entry)
            .collect::<Result<Vec<_>, _>>()?;

        let mut vk_groups: Vec<(
            &'a VerifierKey,
            &'a [G1Affine; STATIC_POINT_COUNT],
            [BlsScalar; VK_STATIC_COUNT],
        )> = Vec::new();
        let mut g_scalar = BlsScalar::zero();

        let mut left_pts = Vec::with_capacity(2 * self.entries.len());
        let mut left_scs = Vec::with_capacity(2 * self.entries.len());
        let mut right_pts = Vec::with_capacity(
            STATIC_POINT_COUNT + DYNAMIC_POINT_COUNT * self.entries.len(),
        );
        let mut right_scs = Vec::with_capacity(
            STATIC_POINT_COUNT + DYNAMIC_POINT_COUNT * self.entries.len(),
        );

        for ((entry, prep), &rho) in
            self.entries.iter().zip(prepared.iter()).zip(rhos.iter())
        {
            let group_idx = if let Some(idx) = vk_groups
                .iter()
                .position(|(vk, _, _)| **vk == entry.verifier.verifier_key)
            {
                idx
            } else {
                vk_groups.push((
                    &entry.verifier.verifier_key,
                    &entry.verifier.static_points,
                    [BlsScalar::zero(); VK_STATIC_COUNT],
                ));
                vk_groups.len() - 1
            };

            let group_scalars = &mut vk_groups[group_idx].2;
            if rho == BlsScalar::one() {
                for (dst, src) in group_scalars
                    .iter_mut()
                    .zip(prep.static_scalars[..VK_STATIC_COUNT].iter())
                {
                    *dst += *src;
                }
                g_scalar += prep.static_scalars[VK_STATIC_COUNT];
                left_pts.push(prep.w_z);
                left_scs.push(BlsScalar::one());
                left_pts.push(prep.w_zw);
                left_scs.push(prep.u_challenge);
                right_pts.extend_from_slice(&prep.dynamic_points);
                right_scs.extend_from_slice(&prep.dynamic_scalars);
            } else {
                for (dst, src) in group_scalars
                    .iter_mut()
                    .zip(prep.static_scalars[..VK_STATIC_COUNT].iter())
                {
                    *dst += rho * *src;
                }
                g_scalar += rho * prep.static_scalars[VK_STATIC_COUNT];
                left_pts.push(prep.w_z);
                left_scs.push(rho);
                left_pts.push(prep.w_zw);
                left_scs.push(rho * prep.u_challenge);
                right_pts.extend_from_slice(&prep.dynamic_points);
                for sc in &prep.dynamic_scalars {
                    right_scs.push(rho * *sc);
                }
            }
        }

        for (_, static_pts, scalars) in &vk_groups {
            right_pts.extend_from_slice(&static_pts[..VK_STATIC_COUNT]);
            right_scs.extend_from_slice(scalars);
        }
        right_pts.push(opening_key.g);
        right_scs.push(g_scalar);

        #[cfg(feature = "blst-backend")]
        {
            let (left, right) = crate::blst_backend::verify_batch_msm_blst(
                &left_pts, &left_scs, &right_pts, &right_scs, None,
            );
            Proof::verify_pairing_affine(opening_key, left, right)
        }
        #[cfg(not(feature = "blst-backend"))]
        {
            let left = -msm_public_variable_base(&left_pts, &left_scs);
            let right = msm_public_variable_base(&right_pts, &right_scs);
            Proof::verify_pairing(opening_key, left, right)
        }
    }
}

#[cfg(test)]
mod tests {
    use dusk_bls12_381::BlsScalar;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::commitment_scheme::Commitment;
    use crate::prelude::{
        Circuit, Compiler, Composer, Prover, PublicParameters,
    };

    #[derive(Default)]
    struct MinimalCircuit;

    impl Circuit for MinimalCircuit {
        fn circuit(&self, composer: &mut Composer) -> Result<(), Error> {
            let witness = composer.append_witness(BlsScalar::from(7u64));
            composer.assert_equal_constant(
                witness,
                BlsScalar::from(7u64),
                None,
            );
            Ok(())
        }
    }

    #[derive(Default)]
    struct PublicInputCircuit;

    impl Circuit for PublicInputCircuit {
        fn circuit(&self, composer: &mut Composer) -> Result<(), Error> {
            for value in [7u64, 11] {
                let value = BlsScalar::from(value);
                let witness = composer.append_witness(value);
                composer.assert_equal_constant(witness, 0u64, Some(value));
            }
            Ok(())
        }
    }

    fn header_word(bytes: &[u8], index: usize) -> usize {
        let offset = index * u64::SIZE;
        u64::from_be_bytes(
            bytes[offset..offset + u64::SIZE]
                .try_into()
                .expect("header is complete"),
        ) as usize
    }

    fn assert_invalid_verifier(bytes: &[u8]) {
        assert!(matches!(
            Verifier::try_from_bytes(bytes),
            Err(Error::BytesError(dusk_bytes::Error::InvalidData))
        ));
    }

    #[test]
    fn verifier_try_from_bytes_rejects_overflow_lengths_without_panicking() {
        let mut bytes = Vec::with_capacity(48);
        bytes.extend_from_slice(&0u64.to_be_bytes()); // label_len
        bytes.extend_from_slice(&0u64.to_be_bytes()); // verifier_key_len
        bytes.extend_from_slice(&0u64.to_be_bytes()); // opening_key_len
        bytes.extend_from_slice(&u64::MAX.to_be_bytes()); // public_input_indexes_len
        bytes.extend_from_slice(&0u64.to_be_bytes()); // size
        bytes.extend_from_slice(&0u64.to_be_bytes()); // constraints

        let result =
            std::panic::catch_unwind(|| Verifier::try_from_bytes(&bytes));
        assert!(
            result.is_ok(),
            "try_from_bytes panicked on overflow lengths"
        );
        assert!(matches!(
            result.unwrap(),
            Err(Error::BytesError(dusk_bytes::Error::InvalidData))
        ));
    }

    #[test]
    fn verifier_try_from_bytes_rejects_oversized_domain_without_panicking() {
        let mut rng = StdRng::seed_from_u64(42);
        let pp = PublicParameters::setup(1 << 5, &mut rng)
            .expect("public parameters should build");
        let (_, verifier) =
            Compiler::compile::<MinimalCircuit>(&pp, b"oversized-domain")
                .expect("circuit should compile");

        let mut bytes = verifier.to_bytes();
        let label_len = u64::from_be_bytes(
            bytes[..u64::SIZE].try_into().expect("header is complete"),
        ) as usize;
        let verifier_key_offset = 48 + label_len;
        bytes[verifier_key_offset..verifier_key_offset + u64::SIZE].fill(0xff);

        let result =
            std::panic::catch_unwind(|| Verifier::try_from_bytes(&bytes));
        assert!(result.is_ok(), "deserializer should never panic");
        let result = result.expect("checked above");
        if usize::BITS == 32 {
            assert!(matches!(
                result,
                Err(Error::BytesError(dusk_bytes::Error::InvalidData))
            ));
        } else {
            assert!(matches!(result, Err(Error::InvalidEvalDomainSize { .. })));
        }
    }

    #[test]
    fn verifier_rejects_public_input_count_mismatch() {
        let mut rng = StdRng::seed_from_u64(43);
        let pp = PublicParameters::setup(1 << 5, &mut rng)
            .expect("public parameters should build");
        let (prover, verifier) =
            Compiler::compile::<MinimalCircuit>(&pp, b"pi-count")
                .expect("circuit should compile");
        let (proof, public_inputs) = prover
            .prove(&mut rng, &MinimalCircuit)
            .expect("proof should build");

        assert!(public_inputs.is_empty());
        assert_eq!(
            verifier.verify(&proof, &[BlsScalar::from(1u64)]),
            Err(Error::InconsistentPublicInputsLen {
                expected: 0,
                provided: 1,
            })
        );
    }

    #[test]
    fn verifier_serialization_requires_valid_public_input_layout() {
        let mut rng = StdRng::seed_from_u64(44);
        let pp = PublicParameters::setup(1 << 5, &mut rng)
            .expect("public parameters should build");
        let (_, verifier) =
            Compiler::compile::<PublicInputCircuit>(&pp, b"verifier-layout")
                .expect("circuit should compile");
        let bytes = verifier.to_bytes();
        let index_offset = 6 * u64::SIZE
            + header_word(&bytes, 0)
            + header_word(&bytes, 1)
            + header_word(&bytes, 2);
        let first: [u8; u64::SIZE] = bytes
            [index_offset..index_offset + u64::SIZE]
            .try_into()
            .expect("first index is complete");
        let second: [u8; u64::SIZE] = bytes
            [index_offset + u64::SIZE..index_offset + 2 * u64::SIZE]
            .try_into()
            .expect("second index is complete");

        let mut duplicate = bytes.clone();
        duplicate[index_offset + u64::SIZE..index_offset + 2 * u64::SIZE]
            .copy_from_slice(&first);

        let mut descending = bytes.clone();
        descending[index_offset..index_offset + u64::SIZE]
            .copy_from_slice(&second);
        descending[index_offset + u64::SIZE..index_offset + 2 * u64::SIZE]
            .copy_from_slice(&first);

        let mut out_of_range = bytes.clone();
        out_of_range[index_offset..index_offset + u64::SIZE]
            .copy_from_slice(&(verifier.constraints as u64).to_be_bytes());

        let mut excessive_count = bytes.clone();
        excessive_count[3 * u64::SIZE..4 * u64::SIZE].copy_from_slice(
            &((verifier.constraints + 1) as u64).to_be_bytes(),
        );

        let mut incoherent_size = bytes.clone();
        incoherent_size[4 * u64::SIZE..5 * u64::SIZE]
            .copy_from_slice(&((verifier.size + 1) as u64).to_be_bytes());

        let mut incoherent_key = bytes.clone();
        let verifier_key_offset = 6 * u64::SIZE + header_word(&bytes, 0);
        incoherent_key[verifier_key_offset..verifier_key_offset + u64::SIZE]
            .copy_from_slice(&((verifier.constraints + 1) as u64).to_bytes());

        let mut trailing = bytes;
        trailing.push(0);

        for malformed in [
            duplicate,
            descending,
            out_of_range,
            excessive_count,
            incoherent_size,
            incoherent_key,
            trailing,
        ] {
            assert_invalid_verifier(&malformed);
        }
    }

    #[test]
    fn try_from_bytes_requires_canonical_key_encodings() {
        let mut rng = StdRng::seed_from_u64(46);
        let pp = PublicParameters::setup(1 << 5, &mut rng).expect("setup");
        let (prover, verifier) =
            Compiler::compile::<MinimalCircuit>(&pp, b"keys").expect("compile");

        // Declare the key segment ending at `end` one byte longer, zero-padded.
        let pad = |bytes: &[u8], header: usize, end: usize| {
            let mut padded = bytes.to_vec();
            let len = header_word(bytes, header) as u64 + 1;
            padded[header * 8..][..8].copy_from_slice(&len.to_be_bytes());
            padded.insert(end, 0);
            padded
        };

        let bytes = verifier.to_bytes();
        let key_end = 48 + header_word(&bytes, 0) + VerifierKey::SIZE;
        assert_invalid_verifier(&pad(&bytes, 1, key_end));
        assert_invalid_verifier(&pad(&bytes, 2, key_end + OpeningKey::SIZE));
        // A nonzero byte at either end of the five retired slots.
        for byte in [key_end - 5 * Commitment::SIZE, key_end - 1] {
            let mut retired_slot = bytes.clone();
            retired_slot[byte] = 1;
            assert_invalid_verifier(&retired_slot);
        }

        // Padded prover-key and verifier-key segments of a prover.
        let bytes = prover.to_bytes();
        let prover_key_end =
            56 + (1..3).map(|i| header_word(&bytes, i)).sum::<usize>();
        let key_end =
            56 + (1..5).map(|i| header_word(&bytes, i)).sum::<usize>();
        for padded in [pad(&bytes, 2, prover_key_end), pad(&bytes, 4, key_end)]
        {
            assert!(matches!(
                Prover::try_from_bytes(padded),
                Err(Error::BytesError(dusk_bytes::Error::InvalidData))
            ));
        }
    }

    #[test]
    fn decoded_verifier_preserves_public_input_layout() {
        let mut rng = StdRng::seed_from_u64(45);
        let pp = PublicParameters::setup(1 << 5, &mut rng)
            .expect("public parameters should build");
        let (prover, verifier) =
            Compiler::compile::<PublicInputCircuit>(&pp, b"verifier-roundtrip")
                .expect("circuit should compile");
        let bytes = verifier.to_bytes();
        let decoded = Verifier::try_from_bytes(&bytes)
            .expect("serialized verifier should decode");
        let (proof, public_inputs) = prover
            .prove(&mut rng, &PublicInputCircuit)
            .expect("proof should build");

        assert_eq!(decoded.to_bytes(), bytes);
        decoded
            .verify(&proof, &public_inputs)
            .expect("decoded verifier should verify the proof");
    }

    #[test]
    fn batch_verification_rejects_cancelling_invalid_proofs() {
        // Construct two invalid proofs whose individual pairing defects cancel
        // if summed with equal weights (P1 = valid + delta*G, P2 = valid -
        // delta*G). Independent Fiat-Shamir randomizers must still
        // reject the pair.
        use dusk_bls12_381::G1Projective;

        let mut rng = StdRng::seed_from_u64(0xca0c_e101);
        let pp = PublicParameters::setup(1 << 5, &mut rng).expect("setup");
        let (prover, verifier) =
            Compiler::compile::<PublicInputCircuit>(&pp, b"cancel-pair")
                .expect("compile");
        let (proof, pi) =
            prover.prove(&mut rng, &PublicInputCircuit).expect("prove");

        let delta = G1Projective::generator() * BlsScalar::from(12345u64);
        let mut p_plus = proof.clone();
        p_plus.a_comm = Commitment(G1Affine::from(
            G1Projective::from(proof.a_comm.0) + delta,
        ));
        let mut p_minus = proof;
        p_minus.a_comm = Commitment(G1Affine::from(
            G1Projective::from(p_minus.a_comm.0) - delta,
        ));

        assert_eq!(
            verifier.verify(&p_plus, &pi),
            Err(Error::ProofVerificationError)
        );
        assert_eq!(
            verifier.verify(&p_minus, &pi),
            Err(Error::ProofVerificationError)
        );
        assert_eq!(
            verifier.verify_batch(&[(&p_plus, &pi), (&p_minus, &pi)]),
            Err(Error::ProofVerificationError)
        );

        let mut batch = BatchVerifier::new();
        batch.push(&verifier, &p_plus, &pi).unwrap();
        batch.push(&verifier, &p_minus, &pi).unwrap();
        assert_eq!(batch.verify(), Err(Error::ProofVerificationError));
    }

    #[test]
    fn batch_transcript_binds_every_proof_field() {
        use dusk_bls12_381::G1Projective;

        let base = Proof::default();
        let challenge_for = |proof: &Proof| {
            let mut t = Transcript::new(b"field-binding-test");
            proof.append_to_batch_transcript(&mut t);
            t.challenge_scalar(b"out")
        };
        let baseline = challenge_for(&base);
        let g = Commitment(G1Affine::from(G1Projective::generator()));
        let one = BlsScalar::one();

        let mut mutated_proofs = Vec::new();
        macro_rules! push_comm_mutation {
            ($field:ident) => {{
                let mut p = base.clone();
                p.$field = g;
                mutated_proofs.push((stringify!($field), p));
            }};
        }
        macro_rules! push_eval_mutation {
            ($field:ident) => {{
                let mut p = base.clone();
                p.evaluations.$field = one;
                mutated_proofs.push((stringify!($field), p));
            }};
        }

        push_comm_mutation!(a_comm);
        push_comm_mutation!(b_comm);
        push_comm_mutation!(c_comm);
        push_comm_mutation!(d_comm);
        push_comm_mutation!(z_comm);
        push_comm_mutation!(t_low_comm);
        push_comm_mutation!(t_mid_comm);
        push_comm_mutation!(t_high_comm);
        push_comm_mutation!(t_fourth_comm);
        push_comm_mutation!(w_z_chall_comm);
        push_comm_mutation!(w_z_chall_w_comm);
        push_eval_mutation!(a_eval);
        push_eval_mutation!(b_eval);
        push_eval_mutation!(c_eval);
        push_eval_mutation!(d_eval);
        push_eval_mutation!(a_w_eval);
        push_eval_mutation!(b_w_eval);
        push_eval_mutation!(d_w_eval);
        push_eval_mutation!(q_arith_eval);
        push_eval_mutation!(q_c_eval);
        push_eval_mutation!(q_l_eval);
        push_eval_mutation!(q_r_eval);
        push_eval_mutation!(s_sigma_1_eval);
        push_eval_mutation!(s_sigma_2_eval);
        push_eval_mutation!(s_sigma_3_eval);
        push_eval_mutation!(z_eval);

        assert_eq!(mutated_proofs.len(), 26);
        for (name, mutated) in mutated_proofs {
            assert_ne!(
                challenge_for(&mutated),
                baseline,
                "batch transcript did not bind proof field {name}"
            );
        }
    }
}
