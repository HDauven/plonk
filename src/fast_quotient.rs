// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! L2-cache-resident 8-coset quotient polynomial engine backed by inline
//! `x86_64` ADX/BMI2 `FastFr` arithmetic.
//!
//! Decomposes the `8n`-point quotient coset evaluation and inverse coset FFT
//! into 8 independent `n`-point cosets (`x = g * omega_{8n}^k * omega_n^j`),
//! precomputes static `L_1(x)` evaluations and de-interleaved stride-1
//! selectors once in `Prover::new`, runs all `n`-point wire/quotient FFTs in
//! L2 cache, and recombines the 8 coset IFFTs with an in-register 8-point
//! radix-8 inverse butterfly (producing `t_low`, `t_mid`, `t_high`, `t_fourth`
//! directly without allocating an intermediate `8n` coefficient vector).

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

use dusk_bls12_381::{BlsScalar, GENERATOR};
use dusk_jubjub::EDWARDS_D;
use rayon::prelude::*;

use crate::composer::permutation::constants::{K1, K2, K3};
use crate::error::Error;
use crate::fast_fr::{
    FastFr, bls_slice_to_fast, coset_n_fft_fast, has_adx_bmi2, mul_portable,
    serial_ifft_fast,
};
use crate::fft::{EvaluationDomain, Polynomial};
use crate::proof_system::ProverKey;
use crate::util::batch_inversion;

/// Precomputed de-interleaved `FastFr` selector and domain data for one of the
/// 8 quotient cosets `k in 0..8`.
#[derive(Clone)]
struct CosetData {
    c_k_pow_n: FastFr,
    coset_powers: Vec<FastFr>,
    inv_twiddle: Vec<FastFr>,
    vanishing_inv: FastFr,
    q_m: Vec<FastFr>,
    q_l: Vec<FastFr>,
    q_r: Vec<FastFr>,
    q_o: Vec<FastFr>,
    q_f: Vec<FastFr>,
    q_c: Vec<FastFr>,
    q_arith: Vec<FastFr>,
    q_range: Option<Vec<FastFr>>,
    q_logic: Option<Vec<FastFr>>,
    q_fixed: Option<Vec<FastFr>>,
    q_var: Option<Vec<FastFr>>,
    s_sigma_1: Vec<FastFr>,
    s_sigma_2: Vec<FastFr>,
    s_sigma_3: Vec<FastFr>,
    s_sigma_4: Vec<FastFr>,
    linear_evals: Vec<FastFr>,
    l1_evals: Vec<FastFr>,
}

/// Precomputed `Prover` acceleration tables for wire IFFTs and 8-coset
/// quotient computation.
#[derive(Clone)]
pub(crate) struct FastProverCache {
    pub(crate) domain_fft_roots: Vec<FastFr>,
    pub(crate) domain_ifft_roots: Vec<FastFr>,
    pub(crate) domain_size_inv: FastFr,
    cosets: [CosetData; 8],
    omega_8_inv_1: FastFr,
    omega_8_inv_2: FastFr,
    omega_8_inv_3: FastFr,
    gen_inv_t_scaled: Vec<FastFr>,
    gen_inv_sn: [FastFr; 8],
    #[cfg(feature = "gpu-cuda")]
    gpu_ntt: std::sync::Arc<
        std::sync::OnceLock<Option<crate::gpu_cuda::GpuNttTables>>,
    >,
}

#[inline]
fn deinterleave_fast(evals: &[BlsScalar], k: usize, n: usize) -> Vec<FastFr> {
    let mut out = Vec::with_capacity(n);
    for j in 0..n {
        out.push(FastFr::from_bls(&evals[8 * j + k]));
    }
    out
}

#[inline]
fn deinterleave_opt_fast(
    evals: &[BlsScalar],
    k: usize,
    n: usize,
) -> Option<Vec<FastFr>> {
    let mut any_nonzero = false;
    let mut out = Vec::with_capacity(n);
    for j in 0..n {
        let v = FastFr::from_bls(&evals[8 * j + k]);
        if !v.is_zero() {
            any_nonzero = true;
        }
        out.push(v);
    }
    any_nonzero.then_some(out)
}

impl FastProverCache {
    pub(crate) fn new(
        domain: &EvaluationDomain,
        quotient_domain: &EvaluationDomain,
        prover_key: &ProverKey,
        vanishing_coset_inverses: &[BlsScalar; 8],
    ) -> Self {
        let n = domain.size();
        let domain_fft_roots =
            bls_slice_to_fast(&domain.precompute_fft_roots());
        let domain_ifft_roots =
            bls_slice_to_fast(&domain.precompute_ifft_roots());
        let domain_size_inv = FastFr::from_bls(&domain.size_inv);

        let omega_8n = quotient_domain.group_gen;
        let omega_8n_inv = quotient_domain.group_gen_inv;
        let gen_inv = quotient_domain.generator_inv;
        let eight_inv = BlsScalar::from(8u64).invert().unwrap();

        let omega_8_inv = omega_8n_inv.pow_vartime(&[n as u64, 0, 0, 0]);
        let omega_8_inv_1 = FastFr::from_bls(&omega_8_inv);
        let omega_8_inv_2 = FastFr::from_bls(&omega_8_inv.square());
        let omega_8_inv_3 =
            FastFr::from_bls(&(omega_8_inv.square() * omega_8_inv));

        let mut gen_inv_t_scaled = Vec::with_capacity(n);
        let mut acc = eight_inv;
        for _ in 0..n {
            gen_inv_t_scaled.push(FastFr::from_bls(&acc));
            acc *= gen_inv;
        }

        let gen_inv_n = gen_inv.pow_vartime(&[n as u64, 0, 0, 0]);
        let mut gen_inv_sn = [FastFr::ONE; 8];
        let mut sn_acc = BlsScalar::one();
        for slot in &mut gen_inv_sn {
            *slot = FastFr::from_bls(&sn_acc);
            sn_acc *= gen_inv_n;
        }

        let linear_8n = &prover_key.permutation.linear_evaluations.evals;
        let v_h_8n = &prover_key.v_h_coset_8n().evals;
        let n_inv = domain.size_inv;

        let coset_vec: Vec<CosetData> = (0..8usize)
            .into_par_iter()
            .map(|k| {
                let omega_k = omega_8n.pow_vartime(&[k as u64, 0, 0, 0]);
                let c_k = GENERATOR * omega_k;
                let c_k_pow_n =
                    FastFr::from_bls(&c_k.pow_vartime(&[n as u64, 0, 0, 0]));

                let mut coset_powers = Vec::with_capacity(n);
                let mut cp = BlsScalar::one();
                for _ in 0..n {
                    coset_powers.push(FastFr::from_bls(&cp));
                    cp *= c_k;
                }

                let omega_inv_k =
                    omega_8n_inv.pow_vartime(&[k as u64, 0, 0, 0]);
                let mut inv_twiddle = Vec::with_capacity(n);
                let mut tw = BlsScalar::one();
                for _ in 0..n {
                    inv_twiddle.push(FastFr::from_bls(&tw));
                    tw *= omega_inv_k;
                }

                // Precompute static `L_1(x) = v_h(x) / (n * (x - 1))` for this
                // coset using one batch inversion.
                let mut denoms = Vec::with_capacity(n);
                for j in 0..n {
                    denoms.push(linear_8n[8 * j + k] - BlsScalar::one());
                }
                batch_inversion(&mut denoms);
                let mut l1_evals = Vec::with_capacity(n);
                for j in 0..n {
                    let l1 = denoms[j] * v_h_8n[8 * j + k] * n_inv;
                    l1_evals.push(FastFr::from_bls(&l1));
                }

                CosetData {
                    c_k_pow_n,
                    coset_powers,
                    inv_twiddle,
                    vanishing_inv: FastFr::from_bls(
                        &vanishing_coset_inverses[k],
                    ),
                    q_m: deinterleave_fast(
                        &prover_key.arithmetic.q_m.1.evals,
                        k,
                        n,
                    ),
                    q_l: deinterleave_fast(
                        &prover_key.arithmetic.q_l.1.evals,
                        k,
                        n,
                    ),
                    q_r: deinterleave_fast(
                        &prover_key.arithmetic.q_r.1.evals,
                        k,
                        n,
                    ),
                    q_o: deinterleave_fast(
                        &prover_key.arithmetic.q_o.1.evals,
                        k,
                        n,
                    ),
                    q_f: deinterleave_fast(
                        &prover_key.arithmetic.q_f.1.evals,
                        k,
                        n,
                    ),
                    q_c: deinterleave_fast(
                        &prover_key.arithmetic.q_c.1.evals,
                        k,
                        n,
                    ),
                    q_arith: deinterleave_fast(
                        &prover_key.arithmetic.q_arith.1.evals,
                        k,
                        n,
                    ),
                    q_range: deinterleave_opt_fast(
                        &prover_key.range.q_range.1.evals,
                        k,
                        n,
                    ),
                    q_logic: deinterleave_opt_fast(
                        &prover_key.logic.q_logic.1.evals,
                        k,
                        n,
                    ),
                    q_fixed: deinterleave_opt_fast(
                        &prover_key.fixed_base.q_fixed_group_add.1.evals,
                        k,
                        n,
                    ),
                    q_var: deinterleave_opt_fast(
                        &prover_key.variable_base.q_variable_group_add.1.evals,
                        k,
                        n,
                    ),
                    s_sigma_1: deinterleave_fast(
                        &prover_key.permutation.s_sigma_1.1.evals,
                        k,
                        n,
                    ),
                    s_sigma_2: deinterleave_fast(
                        &prover_key.permutation.s_sigma_2.1.evals,
                        k,
                        n,
                    ),
                    s_sigma_3: deinterleave_fast(
                        &prover_key.permutation.s_sigma_3.1.evals,
                        k,
                        n,
                    ),
                    s_sigma_4: deinterleave_fast(
                        &prover_key.permutation.s_sigma_4.1.evals,
                        k,
                        n,
                    ),
                    linear_evals: deinterleave_fast(linear_8n, k, n),
                    l1_evals,
                }
            })
            .collect();

        let cosets: [CosetData; 8] = coset_vec
            .try_into()
            .unwrap_or_else(|_| unreachable!("8 cosets"));

        Self {
            domain_fft_roots,
            domain_ifft_roots,
            domain_size_inv,
            cosets,
            omega_8_inv_1,
            omega_8_inv_2,
            omega_8_inv_3,
            gen_inv_t_scaled,
            gen_inv_sn,
            #[cfg(feature = "gpu-cuda")]
            gpu_ntt: std::sync::Arc::new(std::sync::OnceLock::new()),
        }
    }

    #[cfg(feature = "gpu-cuda")]
    fn gpu_ntt_tables(
        &self,
        n: usize,
        log_n: u32,
    ) -> Option<&crate::gpu_cuda::GpuNttTables> {
        self.gpu_ntt
            .get_or_init(|| {
                let coset_powers: [&[FastFr]; 8] = core::array::from_fn(|k| {
                    self.cosets[k].coset_powers.as_slice()
                });
                let ck_pown: [FastFr; 8] =
                    core::array::from_fn(|k| self.cosets[k].c_k_pow_n);
                let vanishing_inv: [FastFr; 8] =
                    core::array::from_fn(|k| self.cosets[k].vanishing_inv);
                let opt_sel =
                    |f: fn(&CosetData) -> Option<&[FastFr]>| -> Option<[Option<&[FastFr]>; 8]> {
                        let arr: [Option<&[FastFr]>; 8] =
                            core::array::from_fn(|k| f(&self.cosets[k]));
                        arr.iter().any(Option::is_some).then_some(arr)
                    };
                let req_sel =
                    |f: fn(&CosetData) -> &[FastFr]| -> Option<[Option<&[FastFr]>; 8]> {
                        Some(core::array::from_fn(|k| Some(f(&self.cosets[k]))))
                    };
                let selectors: [Option<[Option<&[FastFr]>; 8]>; 17] = [
                    req_sel(|c| &c.q_m),
                    req_sel(|c| &c.q_l),
                    req_sel(|c| &c.q_r),
                    req_sel(|c| &c.q_o),
                    req_sel(|c| &c.q_f),
                    req_sel(|c| &c.q_c),
                    req_sel(|c| &c.q_arith),
                    opt_sel(|c| c.q_range.as_deref()),
                    opt_sel(|c| c.q_logic.as_deref()),
                    opt_sel(|c| c.q_fixed.as_deref()),
                    opt_sel(|c| c.q_var.as_deref()),
                    req_sel(|c| &c.s_sigma_1),
                    req_sel(|c| &c.s_sigma_2),
                    req_sel(|c| &c.s_sigma_3),
                    req_sel(|c| &c.s_sigma_4),
                    req_sel(|c| &c.linear_evals),
                    req_sel(|c| &c.l1_evals),
                ];
                crate::gpu_cuda::GpuNttTables::upload(
                    n,
                    log_n,
                    &self.domain_fft_roots,
                    &self.domain_ifft_roots,
                    self.domain_size_inv,
                    coset_powers,
                    ck_pown,
                    vanishing_inv,
                    selectors,
                )
            })
            .as_ref()
    }

    fn finalize_radix8_combine(
        &self,
        coset_iffts: [&[FastFr]; 8],
        n: usize,
    ) -> Result<[Vec<BlsScalar>; 4], Error> {
        let mut t_low = vec![BlsScalar::zero(); n];
        let mut t_mid = vec![BlsScalar::zero(); n];
        let mut t_high = vec![BlsScalar::zero(); n];
        let mut t_fourth = vec![BlsScalar::zero(); n];
        let unsatisfied = AtomicBool::new(false);

        const COMBINE_CHUNK: usize = 512;
        let slices: Vec<_> = t_low
            .chunks_mut(COMBINE_CHUNK)
            .zip(t_mid.chunks_mut(COMBINE_CHUNK))
            .zip(t_high.chunks_mut(COMBINE_CHUNK))
            .zip(t_fourth.chunks_mut(COMBINE_CHUNK))
            .enumerate()
            .collect();

        let extra_tails: Vec<Vec<(usize, BlsScalar)>> = slices
            .into_par_iter()
            .map(|(chunk_idx, (((c0, c1), c2), c3))| {
                let start = chunk_idx * COMBINE_CHUNK;
                combine_radix8_chunk(
                    self,
                    coset_iffts,
                    n,
                    start,
                    [c0, c1, c2, c3],
                    &unsatisfied,
                )
            })
            .collect();

        if unsatisfied.load(Ordering::Relaxed) {
            return Err(Error::CircuitUnsatisfied);
        }

        let max_tail_idx = extra_tails
            .iter()
            .flat_map(|v| v.iter().map(|&(idx, _)| idx))
            .max();
        if let Some(max_idx) = max_tail_idx {
            t_fourth.resize(n + max_idx + 1, BlsScalar::zero());
            for tail in extra_tails {
                for (idx, val) in tail {
                    t_fourth[n + idx] = val;
                }
            }
        } else {
            let mut l = t_fourth.len();
            while l > 0 && t_fourth[l - 1] == BlsScalar::zero() {
                l -= 1;
            }
            t_fourth.truncate(l);
        }
        if t_fourth.is_empty() {
            t_fourth.push(BlsScalar::zero());
        }

        Ok([t_low, t_mid, t_high, t_fourth])
    }

    /// Compute the split quotient polynomial chunks `(t_low, t_mid, t_high,
    /// t_fourth)` directly (before blinding) using the 8-coset `FastFr`
    /// engine, enforcing the exact `> 7n` unsatisfied-circuit check.
    pub(crate) fn compute_split_quotient(
        &self,
        domain: &EvaluationDomain,
        z_poly: &Polynomial,
        (a_poly, b_poly, c_poly, d_poly): (
            &Polynomial,
            &Polynomial,
            &Polynomial,
            &Polynomial,
        ),
        pi_poly: &Polynomial,
        challenges: &(
            BlsScalar,
            BlsScalar,
            BlsScalar,
            BlsScalar,
            BlsScalar,
            BlsScalar,
            BlsScalar,
        ),
    ) -> Result<[Vec<BlsScalar>; 4], Error> {
        let n = domain.size();
        let log_n = domain.log_size_of_group;
        let has_pi = !pi_poly.is_zero();
        let polys_per_coset = if has_pi { 6usize } else { 5usize };
        let src_polys: [&[BlsScalar]; 6] = [
            &z_poly[..],
            &a_poly[..],
            &b_poly[..],
            &c_poly[..],
            &d_poly[..],
            &pi_poly[..],
        ];

        let ch = QuotientFastChallenges::new(challenges);

        // Full GPU VRAM path: upload 5 (or 6) source polynomials once, run all
        // 40 forward coset NTTs + 524K pointwise widget evaluations + 8 inverse
        // coset IFFTs in VRAM, and download only the 8 coset IFFTs once.
        #[cfg(feature = "gpu-cuda")]
        if let Some(gpu) = self.gpu_ntt_tables(n, log_n) {
            let gpu_ch = ch.to_gpu_array();
            if let Some(res) = gpu.with_full_quotient_iffts(
                &src_polys[..polys_per_coset],
                &gpu_ch,
                |flat| {
                    let coset_iffts: [&[FastFr]; 8] =
                        core::array::from_fn(|k| &flat[k * n..(k + 1) * n]);
                    self.finalize_radix8_combine(coset_iffts, n)
                },
            ) {
                return res;
            }
        }

        // CPU fallback: process the 8 cosets in 2 waves of 4 cosets each to
        // halve peak memory while keeping 20-24 independent n-point FFTs
        // saturating all cores per wave, then run all 8 n-point IFFTs in one
        // parallel pass.
        let mut q_evals_all: [Vec<FastFr>; 8] = Default::default();

        for wave in 0..2usize {
            let k_base = wave * 4;
            let total_ffts = 4 * polys_per_coset;

            let wave_evals: Vec<Vec<FastFr>> = (0..total_ffts)
                .into_par_iter()
                .map(|idx| {
                    let local_k = idx / polys_per_coset;
                    let p_idx = idx % polys_per_coset;
                    let coset = &self.cosets[k_base + local_k];
                    coset_n_fft_fast(
                        src_polys[p_idx],
                        n,
                        log_n,
                        coset.c_k_pow_n,
                        &coset.coset_powers,
                        &self.domain_fft_roots,
                    )
                })
                .collect();

            let mut q_evals: [Vec<FastFr>; 4] =
                core::array::from_fn(|_| vec![FastFr::ZERO; n]);

            const CHUNK: usize = 512;
            let chunks_per_coset = n.div_ceil(CHUNK);
            let total_chunks = 4 * chunks_per_coset;

            let mut out_chunks: Vec<(usize, usize, &mut [FastFr])> =
                Vec::with_capacity(total_chunks);
            for (local_k, q_buf) in q_evals.iter_mut().enumerate() {
                for (c_idx, slice) in q_buf.chunks_mut(CHUNK).enumerate() {
                    out_chunks.push((local_k, c_idx * CHUNK, slice));
                }
            }

            out_chunks
                .into_par_iter()
                .for_each(|(local_k, start, out)| {
                    let coset = &self.cosets[k_base + local_k];
                    let base = local_k * polys_per_coset;
                    let z_ev = &wave_evals[base];
                    let a_ev = &wave_evals[base + 1];
                    let b_ev = &wave_evals[base + 2];
                    let c_ev = &wave_evals[base + 3];
                    let d_ev = &wave_evals[base + 4];
                    let pi_ev = has_pi.then(|| wave_evals[base + 5].as_slice());
                    eval_quotient_chunk(
                        coset, &ch, z_ev, a_ev, b_ev, c_ev, d_ev, pi_ev, start,
                        out,
                    );
                });

            for (offset, q_ev) in q_evals.into_iter().enumerate() {
                q_evals_all[k_base + offset] = q_ev;
            }
        }

        let coset_iffts_vec: Vec<Vec<FastFr>> = q_evals_all
            .into_par_iter()
            .map(|q_ev| {
                serial_ifft_fast(
                    &q_ev,
                    &self.domain_ifft_roots,
                    self.domain_size_inv,
                    log_n,
                )
            })
            .collect();
        let coset_iffts: [&[FastFr]; 8] =
            core::array::from_fn(|k| coset_iffts_vec[k].as_slice());

        self.finalize_radix8_combine(coset_iffts, n)
    }

    #[cfg(all(test, feature = "gpu-cuda"))]
    pub(crate) fn compute_split_quotient_raw_iffts_gpu(
        &self,
        domain: &EvaluationDomain,
        z_poly: &Polynomial,
        (a_poly, b_poly, c_poly, d_poly): (
            &Polynomial,
            &Polynomial,
            &Polynomial,
            &Polynomial,
        ),
        pi_poly: &Polynomial,
        challenges: &(
            BlsScalar,
            BlsScalar,
            BlsScalar,
            BlsScalar,
            BlsScalar,
            BlsScalar,
            BlsScalar,
        ),
    ) -> [Vec<FastFr>; 8] {
        let n = domain.size();
        let log_n = domain.log_size_of_group;
        let has_pi = !pi_poly.is_zero();
        let polys_per_coset = if has_pi { 6usize } else { 5usize };
        let src_polys: [&[BlsScalar]; 6] = [
            &z_poly[..],
            &a_poly[..],
            &b_poly[..],
            &c_poly[..],
            &d_poly[..],
            &pi_poly[..],
        ];
        let ch = QuotientFastChallenges::new(challenges);
        let gpu_ch = ch.to_gpu_array();
        let gpu = self
            .gpu_ntt_tables(n, log_n)
            .expect("GPU NTT tables should initialize");
        gpu.with_full_quotient_iffts(
            &src_polys[..polys_per_coset],
            &gpu_ch,
            |flat| core::array::from_fn(|k| flat[k * n..(k + 1) * n].to_vec()),
        )
        .expect("GPU full quotient should succeed")
    }

    #[cfg(all(test, feature = "gpu-cuda"))]
    pub(crate) fn compute_split_quotient_raw_iffts_cpu(
        &self,
        domain: &EvaluationDomain,
        z_poly: &Polynomial,
        (a_poly, b_poly, c_poly, d_poly): (
            &Polynomial,
            &Polynomial,
            &Polynomial,
            &Polynomial,
        ),
        pi_poly: &Polynomial,
        challenges: &(
            BlsScalar,
            BlsScalar,
            BlsScalar,
            BlsScalar,
            BlsScalar,
            BlsScalar,
            BlsScalar,
        ),
    ) -> [Vec<FastFr>; 8] {
        let n = domain.size();
        let log_n = domain.log_size_of_group;
        let has_pi = !pi_poly.is_zero();
        let polys_per_coset = if has_pi { 6usize } else { 5usize };
        let src_polys: [&[BlsScalar]; 6] = [
            &z_poly[..],
            &a_poly[..],
            &b_poly[..],
            &c_poly[..],
            &d_poly[..],
            &pi_poly[..],
        ];
        let ch = QuotientFastChallenges::new(challenges);
        let mut q_evals_all: [Vec<FastFr>; 8] = Default::default();
        for wave in 0..2usize {
            let k_base = wave * 4;
            let total_ffts = 4 * polys_per_coset;
            let wave_evals: Vec<Vec<FastFr>> = (0..total_ffts)
                .into_par_iter()
                .map(|idx| {
                    let local_k = idx / polys_per_coset;
                    let p_idx = idx % polys_per_coset;
                    let coset = &self.cosets[k_base + local_k];
                    coset_n_fft_fast(
                        src_polys[p_idx],
                        n,
                        log_n,
                        coset.c_k_pow_n,
                        &coset.coset_powers,
                        &self.domain_fft_roots,
                    )
                })
                .collect();
            for local_k in 0..4 {
                let coset = &self.cosets[k_base + local_k];
                let base = local_k * polys_per_coset;
                let mut out = vec![FastFr::ZERO; n];
                let pi_ev = has_pi.then(|| wave_evals[base + 5].as_slice());
                eval_quotient_chunk(
                    coset,
                    &ch,
                    &wave_evals[base],
                    &wave_evals[base + 1],
                    &wave_evals[base + 2],
                    &wave_evals[base + 3],
                    &wave_evals[base + 4],
                    pi_ev,
                    0,
                    &mut out,
                );
                q_evals_all[k_base + local_k] = out;
            }
        }
        core::array::from_fn(|k| {
            serial_ifft_fast(
                &q_evals_all[k],
                &self.domain_ifft_roots,
                self.domain_size_inv,
                log_n,
            )
        })
    }
}

#[derive(Clone, Copy)]
struct QuotientFastChallenges {
    // Permutation
    alpha: FastFr,
    alpha_sq: FastFr,
    beta: FastFr,
    beta_k1: FastFr,
    beta_k2: FastFr,
    beta_k3: FastFr,
    gamma: FastFr,
    // Factored permutation constants (when `beta != 0`)
    has_inv_beta: bool,
    inv_beta: FastFr,
    gamma_over_beta: FastFr,
    beta4_alpha: FastFr,
    k1: FastFr,
    k2: FastFr,
    k3: FastFr,
    // Constants for delta(f) = (f^2 - 3f)(f^2 - 3f + 2)
    two: FastFr,
    three: FastFr,
    // Range
    range_ch: FastFr,
    range_k1: FastFr,
    range_k2: FastFr,
    range_k3: FastFr,
    // Logic
    logic_ch: FastFr,
    logic_k1: FastFr,
    logic_k2: FastFr,
    logic_k3: FastFr,
    logic_k4: FastFr,
    eighty_one: FastFr,
    eighty_three: FastFr,
    // Fixed-base ECC
    fixed_ch: FastFr,
    fixed_k1: FastFr,
    fixed_k2: FastFr,
    fixed_k3: FastFr,
    edwards_d: FastFr,
    // Variable-base ECC
    var_ch: FastFr,
    var_k1: FastFr,
    var_k2: FastFr,
}

impl QuotientFastChallenges {
    fn new(
        (alpha, beta, gamma, range_c, logic_c, fixed_c, var_c): &(
            BlsScalar,
            BlsScalar,
            BlsScalar,
            BlsScalar,
            BlsScalar,
            BlsScalar,
            BlsScalar,
        ),
    ) -> Self {
        let rk1 = range_c.square();
        let rk2 = rk1.square();
        let rk3 = rk2 * rk1;

        let lk1 = logic_c.square();
        let lk2 = lk1.square();
        let lk3 = lk2 * lk1;
        let lk4 = lk3 * lk1;

        let fk1 = fixed_c.square();
        let fk2 = fk1.square();
        let fk3 = fk2 * fk1;

        let vk1 = var_c.square();
        let vk2 = vk1.square();

        let (has_inv_beta, inv_beta, gamma_over_beta, beta4_alpha) =
            match beta.invert() {
                Some(ib) => {
                    let b2 = beta.square();
                    let b4 = b2.square();
                    (
                        true,
                        FastFr::from_bls(&ib),
                        FastFr::from_bls(&(gamma * ib)),
                        FastFr::from_bls(&(b4 * alpha)),
                    )
                }
                None => (false, FastFr::ZERO, FastFr::ZERO, FastFr::ZERO),
            };

        Self {
            alpha: FastFr::from_bls(alpha),
            alpha_sq: FastFr::from_bls(&alpha.square()),
            beta: FastFr::from_bls(beta),
            beta_k1: FastFr::from_bls(&(beta * K1)),
            beta_k2: FastFr::from_bls(&(beta * K2)),
            beta_k3: FastFr::from_bls(&(beta * K3)),
            gamma: FastFr::from_bls(gamma),
            has_inv_beta,
            inv_beta,
            gamma_over_beta,
            beta4_alpha,
            k1: FastFr::from_bls(&K1),
            k2: FastFr::from_bls(&K2),
            k3: FastFr::from_bls(&K3),
            two: FastFr::from_bls(&BlsScalar::from(2u64)),
            three: FastFr::from_bls(&BlsScalar::from(3u64)),
            range_ch: FastFr::from_bls(range_c),
            range_k1: FastFr::from_bls(&rk1),
            range_k2: FastFr::from_bls(&rk2),
            range_k3: FastFr::from_bls(&rk3),
            logic_ch: FastFr::from_bls(logic_c),
            logic_k1: FastFr::from_bls(&lk1),
            logic_k2: FastFr::from_bls(&lk2),
            logic_k3: FastFr::from_bls(&lk3),
            logic_k4: FastFr::from_bls(&lk4),
            eighty_one: FastFr::from_bls(&BlsScalar::from(81u64)),
            eighty_three: FastFr::from_bls(&BlsScalar::from(83u64)),
            fixed_ch: FastFr::from_bls(fixed_c),
            fixed_k1: FastFr::from_bls(&fk1),
            fixed_k2: FastFr::from_bls(&fk2),
            fixed_k3: FastFr::from_bls(&fk3),
            edwards_d: FastFr::from_bls(&EDWARDS_D),
            var_ch: FastFr::from_bls(var_c),
            var_k1: FastFr::from_bls(&vk1),
            var_k2: FastFr::from_bls(&vk2),
        }
    }

    #[cfg(feature = "gpu-cuda")]
    fn to_gpu_array(self) -> [FastFr; 36] {
        [
            self.alpha,
            self.alpha_sq,
            self.beta,
            self.beta_k1,
            self.beta_k2,
            self.beta_k3,
            self.gamma,
            self.inv_beta,
            self.gamma_over_beta,
            self.beta4_alpha,
            self.k1,
            self.k2,
            self.k3,
            FastFr::ONE,
            self.two,
            self.three,
            self.range_ch,
            self.range_k1,
            self.range_k2,
            self.range_k3,
            self.logic_ch,
            self.logic_k1,
            self.logic_k2,
            self.logic_k3,
            self.logic_k4,
            self.eighty_one,
            self.eighty_three,
            self.fixed_ch,
            self.fixed_k1,
            self.fixed_k2,
            self.fixed_k3,
            self.edwards_d,
            self.var_ch,
            self.var_k1,
            self.var_k2,
            FastFr([u64::from(self.has_inv_beta), 0, 0, 0]),
        ]
    }
}

macro_rules! define_quotient_chunk_kernel {
    ($fn_name:ident, $add:expr, $sub:expr, $mul:expr, $dbl:expr, $($attrs:meta)*) => {
        $(#[$attrs])*
        fn $fn_name(
            coset: &CosetData,
            ch: &QuotientFastChallenges,
            z_ev: &[FastFr],
            a_ev: &[FastFr],
            b_ev: &[FastFr],
            c_ev: &[FastFr],
            d_ev: &[FastFr],
            pi_ev: Option<&[FastFr]>,
            start: usize,
            out: &mut [FastFr],
        ) {
            let n = z_ev.len();
            let mask = n - 1;
            let one = FastFr::ONE;
            let two = ch.two;
            let three = ch.three;

            let add = |x: FastFr, y: &FastFr| -> FastFr { $add(x, y) };
            let sub = |x: FastFr, y: &FastFr| -> FastFr { $sub(x, y) };
            let mul = |x: &FastFr, y: &FastFr| -> FastFr { $mul(x, y) };
            let dbl = |x: FastFr| -> FastFr { $dbl(x) };
            // f*(f-1)*(f-2)*(f-3) = (f*(f-3)) * (f*(f-3) + 2): 2 muls instead of 3!
            let delta = |f: FastFr| -> FastFr {
                let f_m3 = sub(f, &three);
                let g = mul(&f, &f_m3);
                let g_p2 = add(g, &two);
                mul(&g, &g_p2)
            };

            let q_range_opt = coset.q_range.as_deref();
            let q_logic_opt = coset.q_logic.as_deref();
            let q_fixed_opt = coset.q_fixed.as_deref();
            let q_var_opt = coset.q_var.as_deref();
            let v_inv = coset.vanishing_inv;

            for (offset, slot) in out.iter_mut().enumerate() {
                let j = start + offset;
                let j_w = (j + 1) & mask;

                let a = a_ev[j];
                let b = b_ev[j];
                let c = c_ev[j];
                let d = d_ev[j];
                let a_w = a_ev[j_w];
                let b_w = b_ev[j_w];
                let d_w = d_ev[j_w];

                let mut t_1 = FastFr::ZERO;

                let ab = mul(&a, &b);

                // Arithmetic widget
                let q_arith = coset.q_arith[j];
                if !q_arith.is_zero() {
                    let s0 = add(mul(&ab, &coset.q_m[j]), &mul(&a, &coset.q_l[j]));
                    let s1 = add(mul(&b, &coset.q_r[j]), &mul(&c, &coset.q_o[j]));
                    let s2 = add(mul(&d, &coset.q_f[j]), &coset.q_c[j]);
                    let s = add(add(s0, &s1), &s2);
                    t_1 = add(t_1, &mul(&s, &q_arith));
                }

                // Shared 4*a, 4*b, 4*d for range & logic widgets
                let need_quads = q_range_opt.is_some() || q_logic_opt.is_some();
                let four_a = if need_quads { dbl(dbl(a)) } else { FastFr::ZERO };
                let four_b = if need_quads { dbl(dbl(b)) } else { FastFr::ZERO };
                let four_d = if need_quads { dbl(dbl(d)) } else { FastFr::ZERO };

                // Range widget
                if let Some(q_range_s) = q_range_opt {
                    let q_range = q_range_s[j];
                    if !q_range.is_zero() {
                        let b1 = delta(sub(c, &four_d));
                        let d2 = delta(sub(b, &dbl(dbl(c))));
                        let b2 = mul(&d2, &ch.range_k1);
                        let d3 = delta(sub(a, &four_b));
                        let b3 = mul(&d3, &ch.range_k2);
                        let d4 = delta(sub(d_w, &four_a));
                        let b4 = mul(&d4, &ch.range_k3);
                        let rsum = add(add(b1, &b2), &add(b3, &b4));
                        let rq = mul(&rsum, &q_range);
                        t_1 = add(t_1, &mul(&rq, &ch.range_ch));
                    }
                }

                // Logic widget
                if let Some(q_logic_s) = q_logic_opt {
                    let q_logic = q_logic_s[j];
                    if !q_logic.is_zero() {
                        let la = sub(a_w, &four_a);
                        let lb = sub(b_w, &four_b);
                        let ld = sub(d_w, &four_d);
                        let c0 = delta(la);
                        let dl1 = delta(lb);
                        let c1 = mul(&dl1, &ch.logic_k1);
                        let dl2 = delta(ld);
                        let c2 = mul(&dl2, &ch.logic_k2);
                        let la_lb = mul(&la, &lb);
                        let c_sub = sub(c, &la_lb);
                        let c3 = mul(&c_sub, &ch.logic_k3);

                        // delta_xor_and_with_constants(la, lb, c, ld, q_c)
                        // la^2 + lb^2 = (la + lb)^2 - 2*(la*lb) saves 1 mul!
                        let apb = add(la, &lb);
                        let three_apb = add(dbl(apb), &apb);
                        let nine_apb = add(dbl(three_apb), &three_apb);
                        let eighteen_apb = dbl(nine_apb);
                        let twenty_seven_apb = add(eighteen_apb, &nine_apb);
                        let eighty_one_apb =
                            add(dbl(twenty_seven_apb), &twenty_seven_apb);

                        let apb_sq = mul(&apb, &apb);
                        let sq_sum = sub(apb_sq, &dbl(la_lb));
                        let nine_sq_sum = add(dbl(dbl(dbl(sq_sum))), &sq_sum);
                        let eighteen_sq_sum = dbl(nine_sq_sum);

                        let four_w = dbl(dbl(c));
                        let inner0 = add(sub(four_w, &eighteen_apb), &ch.eighty_one);
                        let inner1 = add(mul(&c, &inner0), &eighteen_sq_sum);
                        let f_inner = add(sub(inner1, &eighty_one_apb), &ch.eighty_three);
                        let f_val = mul(&c, &f_inner);
                        let three_c = add(dbl(ld), &ld);
                        let nine_c = add(dbl(three_c), &three_c);
                        let e_val = sub(add(three_apb, &three_c), &dbl(f_val));
                        let diff_c = sub(nine_c, &three_apb);
                        let b_val = mul(&coset.q_c[j], &diff_c);
                        let be = add(b_val, &e_val);
                        let c4 = mul(&be, &ch.logic_k4);

                        let l_sum = add(add(add(c3, &c0), &add(c1, &c2)), &c4);
                        let lq = mul(&q_logic, &l_sum);
                        t_1 = add(t_1, &mul(&lq, &ch.logic_ch));
                    }
                }

                // Fixed-base ECC widget
                if let Some(q_fixed_s) = q_fixed_opt {
                    let q_fixed = q_fixed_s[j];
                    if !q_fixed.is_zero() {
                        let bit = sub(d_w, &dbl(d));
                        let bit_sq = mul(&bit, &bit);
                        let bit_sq_m1 = sub(bit_sq, &one);
                        let bit_cons = mul(&bit, &bit_sq_m1);
                        let qr_m1 = sub(coset.q_r[j], &one);
                        let y_alpha = add(mul(&bit_sq, &qr_m1), &one);
                        let x_alpha = mul(&bit, &coset.q_l[j]);
                        let bqc_mc = sub(mul(&bit, &coset.q_c[j]), &c);
                        let xy_cons = mul(&bqc_mc, &ch.fixed_k1);
                        let cab = mul(&c, &ab);
                        let d_term = mul(&cab, &ch.edwards_d);

                        let x_lhs = add(a_w, &mul(&a_w, &d_term));
                        let x_rhs = add(mul(&a, &y_alpha), &mul(&b, &x_alpha));
                        let x_diff = sub(x_lhs, &x_rhs);
                        let x_cons = mul(&x_diff, &ch.fixed_k2);

                        let y_lhs = sub(b_w, &mul(&b_w, &d_term));
                        let y_rhs = add(mul(&b, &y_alpha), &mul(&a, &x_alpha));
                        let y_diff = sub(y_lhs, &y_rhs);
                        let y_cons = mul(&y_diff, &ch.fixed_k3);

                        let id = add(add(bit_cons, &x_cons), &add(y_cons, &xy_cons));
                        let idq = mul(&id, &q_fixed);
                        t_1 = add(t_1, &mul(&idq, &ch.fixed_ch));
                    }
                }

                // Variable-base ECC widget
                if let Some(q_var_s) = q_var_opt {
                    let q_var = q_var_s[j];
                    if !q_var.is_zero() {
                        let xy_cons = sub(mul(&a, &d), &d_w);
                        let y1_x2 = mul(&b, &c);
                        let y1_y2 = mul(&b, &d);
                        let x1_x2 = mul(&a, &c);
                        let ed_dw = mul(&ch.edwards_d, &d_w);
                        let d_prod = mul(&ed_dw, &y1_x2);

                        let x3_lhs = add(d_w, &y1_x2);
                        let x3_rhs = add(a_w, &mul(&a_w, &d_prod));
                        let x3_diff = sub(x3_lhs, &x3_rhs);
                        let x3_cons = mul(&x3_diff, &ch.var_k1);

                        let y3_lhs = add(y1_y2, &x1_x2);
                        let y3_rhs = sub(b_w, &mul(&b_w, &d_prod));
                        let y3_diff = sub(y3_lhs, &y3_rhs);
                        let y3_cons = mul(&y3_diff, &ch.var_k2);

                        let id = add(add(xy_cons, &x3_cons), &y3_cons);
                        let idq = mul(&id, &q_var);
                        t_1 = add(t_1, &mul(&idq, &ch.var_ch));
                    }
                }

                if let Some(pi) = pi_ev {
                    t_1 = add(t_1, &pi[j]);
                }

                // Permutation widget (with precomputed static L_1 and factored beta!)
                let x = coset.linear_evals[j];
                let z_j = z_ev[j];
                let z_jw = z_ev[j_w];
                let perm_diff_scaled = if ch.has_inv_beta {
                    let a_gb = add(mul(&a, &ch.inv_beta), &ch.gamma_over_beta);
                    let b_gb = add(mul(&b, &ch.inv_beta), &ch.gamma_over_beta);
                    let c_gb = add(mul(&c, &ch.inv_beta), &ch.gamma_over_beta);
                    let d_gb = add(mul(&d, &ch.inv_beta), &ch.gamma_over_beta);

                    let id0 = add(a_gb, &x);
                    let id1 = add(b_gb, &mul(&ch.k1, &x));
                    let id2 = add(c_gb, &mul(&ch.k2, &x));
                    let id3 = add(d_gb, &mul(&ch.k3, &x));
                    let id01 = mul(&id0, &id1);
                    let id23 = mul(&id2, &id3);
                    let id_prod = mul(&id01, &id23);
                    let id_term = mul(&id_prod, &z_j);

                    let cp0 = add(a_gb, &coset.s_sigma_1[j]);
                    let cp1 = add(b_gb, &coset.s_sigma_2[j]);
                    let cp2 = add(c_gb, &coset.s_sigma_3[j]);
                    let cp3 = add(d_gb, &coset.s_sigma_4[j]);
                    let cp01 = mul(&cp0, &cp1);
                    let cp23 = mul(&cp2, &cp3);
                    let cp_prod = mul(&cp01, &cp23);
                    let copy_term = mul(&cp_prod, &z_jw);

                    let diff = sub(id_term, &copy_term);
                    mul(&diff, &ch.beta4_alpha)
                } else {
                    let a_g = add(a, &ch.gamma);
                    let b_g = add(b, &ch.gamma);
                    let c_g = add(c, &ch.gamma);
                    let d_g = add(d, &ch.gamma);

                    let id0 = add(a_g, &mul(&ch.beta, &x));
                    let id1 = add(b_g, &mul(&ch.beta_k1, &x));
                    let id2 = add(c_g, &mul(&ch.beta_k2, &x));
                    let id3 = add(d_g, &mul(&ch.beta_k3, &x));
                    let id01 = mul(&id0, &id1);
                    let id23 = mul(&id2, &id3);
                    let id_prod = mul(&id01, &id23);
                    let id_term = mul(&id_prod, &z_j);

                    let cp0 = add(a_g, &mul(&ch.beta, &coset.s_sigma_1[j]));
                    let cp1 = add(b_g, &mul(&ch.beta, &coset.s_sigma_2[j]));
                    let cp2 = add(c_g, &mul(&ch.beta, &coset.s_sigma_3[j]));
                    let cp3 = add(d_g, &mul(&ch.beta, &coset.s_sigma_4[j]));
                    let cp01 = mul(&cp0, &cp1);
                    let cp23 = mul(&cp2, &cp3);
                    let cp_prod = mul(&cp01, &cp23);
                    let copy_term = mul(&cp_prod, &z_jw);

                    let diff = sub(id_term, &copy_term);
                    mul(&diff, &ch.alpha)
                };

                let l1_alpha_sq = mul(&coset.l1_evals[j], &ch.alpha_sq);
                let zm1 = sub(z_j, &one);
                let check_one = mul(&zm1, &l1_alpha_sq);
                let t_2 = add(perm_diff_scaled, &check_one);
                let total = add(t_1, &t_2);

                *slot = mul(&total, &v_inv);
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
define_quotient_chunk_kernel!(
    eval_quotient_chunk_adx,
    |x, y| unsafe { crate::fast_fr::add_adx_ref(x, y) },
    |x, y| unsafe { crate::fast_fr::sub_adx_ref(x, y) },
    |x, y| unsafe { crate::fast_fr::mul_adx_ref(x, y) },
    |x| unsafe { crate::fast_fr::dbl_adx_inner(x) },
    target_feature(enable = "adx,bmi2")
);

define_quotient_chunk_kernel!(
    eval_quotient_chunk_portable,
    |x: FastFr, y: &FastFr| x.add_portable(*y),
    |x: FastFr, y: &FastFr| x.sub_portable(*y),
    |x: &FastFr, y: &FastFr| mul_portable(*x, *y),
    |x: FastFr| x.add_portable(x),
);

fn eval_quotient_chunk(
    coset: &CosetData,
    ch: &QuotientFastChallenges,
    z_ev: &[FastFr],
    a_ev: &[FastFr],
    b_ev: &[FastFr],
    c_ev: &[FastFr],
    d_ev: &[FastFr],
    pi_ev: Option<&[FastFr]>,
    start: usize,
    out: &mut [FastFr],
) {
    #[cfg(target_arch = "x86_64")]
    if has_adx_bmi2() {
        // SAFETY: runtime checked ADX + BMI2.
        unsafe {
            eval_quotient_chunk_adx(
                coset, ch, z_ev, a_ev, b_ev, c_ev, d_ev, pi_ev, start, out,
            );
        }
        return;
    }
    eval_quotient_chunk_portable(
        coset, ch, z_ev, a_ev, b_ev, c_ev, d_ev, pi_ev, start, out,
    );
}

macro_rules! define_combine_radix8_kernel {
    ($fn_name:ident, $add:expr, $sub:expr, $mul:expr, $($attrs:meta)*) => {
        $(#[$attrs])*
        fn $fn_name(
            cache: &FastProverCache,
            coset_iffts: [&[FastFr]; 8],
            n: usize,
            start: usize,
            outs: [&mut [BlsScalar]; 4],
            unsatisfied: &AtomicBool,
        ) -> Vec<(usize, BlsScalar)> {
            let add = |x: FastFr, y: FastFr| -> FastFr { $add(x, y) };
            let sub = |x: FastFr, y: FastFr| -> FastFr { $sub(x, y) };
            let mul = |x: FastFr, y: FastFr| -> FastFr { $mul(x, y) };

            let w1 = cache.omega_8_inv_1;
            let w2 = cache.omega_8_inv_2;
            let w3 = cache.omega_8_inv_3;
            let sn = cache.gen_inv_sn;
            let len = outs[0].len();
            let mut extra_tail = Vec::new();

            for offset in 0..len {
                let t = start + offset;
                let u0 = coset_iffts[0][t];
                let u1 = mul(coset_iffts[1][t], cache.cosets[1].inv_twiddle[t]);
                let u2 = mul(coset_iffts[2][t], cache.cosets[2].inv_twiddle[t]);
                let u3 = mul(coset_iffts[3][t], cache.cosets[3].inv_twiddle[t]);
                let u4 = mul(coset_iffts[4][t], cache.cosets[4].inv_twiddle[t]);
                let u5 = mul(coset_iffts[5][t], cache.cosets[5].inv_twiddle[t]);
                let u6 = mul(coset_iffts[6][t], cache.cosets[6].inv_twiddle[t]);
                let u7 = mul(coset_iffts[7][t], cache.cosets[7].inv_twiddle[t]);

                // 4-point DFT on even indices (u0, u2, u4, u6) with root w2
                let e0 = add(u0, u4);
                let e1 = sub(u0, u4);
                let e2 = add(u2, u6);
                let e3 = mul(sub(u2, u6), w2);
                let ev0 = add(e0, e2);
                let ev2 = sub(e0, e2);
                let ev1 = add(e1, e3);
                let ev3 = sub(e1, e3);

                // 4-point DFT on odd indices (u1, u3, u5, u7) with root w2
                let o0 = add(u1, u5);
                let o1 = sub(u1, u5);
                let o2 = add(u3, u7);
                let o3 = mul(sub(u3, u7), w2);
                let od0 = add(o0, o2);
                let od2 = mul(sub(o0, o2), w2);
                let od1 = mul(add(o1, o3), w1);
                let od3 = mul(sub(o1, o3), w3);

                // Combine to 8-point inverse DFT outputs v[0..8]
                let v0 = add(ev0, od0);
                let v4 = sub(ev0, od0);
                let v1 = add(ev1, od1);
                let v5 = sub(ev1, od1);
                let v2 = add(ev2, od2);
                let v6 = sub(ev2, od2);
                let v3 = add(ev3, od3);
                let v7 = sub(ev3, od3);

                if !v7.is_zero() {
                    unsatisfied.store(true, Ordering::Relaxed);
                }

                let gt = cache.gen_inv_t_scaled[t];
                let vs_low = [v0, v1, v2, v3];
                for s in 0..4 {
                    let val = vs_low[s];
                    if !val.is_zero() {
                        outs[s][offset] = mul(mul(val, gt), sn[s]).to_bls();
                    }
                }
                let vs_tail = [v4, v5, v6];
                for (k, val) in vs_tail.into_iter().enumerate() {
                    if !val.is_zero() {
                        extra_tail.push((
                            k * n + t,
                            mul(mul(val, gt), sn[4 + k]).to_bls(),
                        ));
                    }
                }
            }
            extra_tail
        }
    };
}

#[cfg(target_arch = "x86_64")]
define_combine_radix8_kernel!(
    combine_radix8_chunk_adx,
    |x, y| unsafe { crate::fast_fr::add_adx_inner(x, y) },
    |x, y| unsafe { crate::fast_fr::sub_adx_inner(x, y) },
    |x, y| unsafe { crate::fast_fr::mul_adx_inner(x, y) },
    target_feature(enable = "adx,bmi2")
);

define_combine_radix8_kernel!(
    combine_radix8_chunk_portable,
    |x: FastFr, y: FastFr| x.add_portable(y),
    |x: FastFr, y: FastFr| x.sub_portable(y),
    |x: FastFr, y: FastFr| mul_portable(x, y),
);

fn combine_radix8_chunk(
    cache: &FastProverCache,
    coset_iffts: [&[FastFr]; 8],
    n: usize,
    start: usize,
    outs: [&mut [BlsScalar]; 4],
    unsatisfied: &AtomicBool,
) -> Vec<(usize, BlsScalar)> {
    #[cfg(target_arch = "x86_64")]
    if has_adx_bmi2() {
        // SAFETY: runtime checked ADX + BMI2.
        return unsafe {
            combine_radix8_chunk_adx(
                cache,
                coset_iffts,
                n,
                start,
                outs,
                unsatisfied,
            )
        };
    }
    combine_radix8_chunk_portable(
        cache,
        coset_iffts,
        n,
        start,
        outs,
        unsatisfied,
    )
}
