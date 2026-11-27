// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! Native open-source CUDA GPU acceleration backend (NVIDIA Driver API +
//! embedded PTX, zero `nvcc` / `cmake` / license-server dependencies).
//!
//! Dynamically loads `libcuda.so.1` at runtime, JIT-compiles BLS12-381 `Fp` /
//! `Fr` Montgomery + XYZZ Pippenger MSM kernels directly on the NVIDIA driver
//! (supporting Volta through Blackwell SM 12.0), keeps SRS bases resident in
//! VRAM, and falls back gracefully to the CPU `blst-backend` whenever no CUDA
//! GPU is present or for small circuits (`< GPU_MSM_MIN_LEN`).

mod driver;
mod ptx;

use alloc::vec::Vec;
use core::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use blst::{
    blst_fp, blst_fp_eucl_inverse, blst_fp_mul, blst_p1,
    blst_p1_add_or_double_affine, blst_p1_affine, blst_p1_double,
};
use dusk_bls12_381::{BlsScalar, G1Affine};
use rayon::prelude::*;

use crate::blst_backend::blst_p1s_to_g1_affines;
use crate::fast_fr::FastFr;
use driver::{CuFunction, CudaBuf, CudaDriver, CudaHostBuf};

/// Minimum polynomial length to offload an MSM to GPU.
pub(crate) const GPU_MSM_MIN_LEN: usize = 4096;

/// Minimum domain size `n` to offload the full 8-coset quotient engine
/// (NTT + pointwise widgets + IFFT) to GPU.
pub(crate) const GPU_NTT_MIN_LEN: usize = 4096;

/// `1` in BLS12-381 `Fp` Montgomery form.
const FP_ONE_BLST: blst_fp = blst_fp {
    l: ptx::FP_ONE_MONT,
};

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuXyzz {
    x: [u64; 6],
    y: [u64; 6],
    zz: [u64; 6],
    zzz: [u64; 6],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuAccTask {
    start: u32,
    end: u32,
    dst: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuSplitMerge {
    dst_bucket: u32,
    tmp_start: u32,
    tmp_count: u32,
}

struct GpuMsmScratch {
    scalars_mont: CudaBuf,
    scalars_std: CudaBuf,
    counts: CudaBuf,
    cursors: CudaBuf,
    sorted: CudaBuf,
    tasks: CudaBuf,
    merges: CudaBuf,
    buckets: CudaBuf,
    window_sums: CudaBuf,
    host_packed: CudaHostBuf<[u64; 4]>,
    host_counts: CudaHostBuf<u32>,
    host_cursors: CudaHostBuf<u32>,
}

impl GpuMsmScratch {
    fn new(drv: &'static CudaDriver) -> Result<Self, i32> {
        Ok(Self {
            scalars_mont: CudaBuf::alloc(drv, 16)?,
            scalars_std: CudaBuf::alloc(drv, 16)?,
            counts: CudaBuf::alloc(drv, 16)?,
            cursors: CudaBuf::alloc(drv, 16)?,
            sorted: CudaBuf::alloc(drv, 16)?,
            tasks: CudaBuf::alloc(drv, 16)?,
            merges: CudaBuf::alloc(drv, 16)?,
            buckets: CudaBuf::alloc(drv, 16)?,
            window_sums: CudaBuf::alloc(drv, 16)?,
            host_packed: CudaHostBuf::new(drv),
            host_counts: CudaHostBuf::new(drv),
            host_cursors: CudaHostBuf::new(drv),
        })
    }
}

struct GpuNttScratch {
    src: CudaBuf,
    out: CudaBuf,
    q_evals: CudaBuf,
    ch_buf: CudaBuf,
    host_src: CudaHostBuf<FastFr>,
    host_out: CudaHostBuf<FastFr>,
}

impl GpuNttScratch {
    fn new(drv: &'static CudaDriver) -> Result<Self, i32> {
        Ok(Self {
            src: CudaBuf::alloc(drv, 16)?,
            out: CudaBuf::alloc(drv, 16)?,
            q_evals: CudaBuf::alloc(drv, 16)?,
            ch_buf: CudaBuf::alloc(drv, 44 * core::mem::size_of::<FastFr>())?,
            host_src: CudaHostBuf::new(drv),
            host_out: CudaHostBuf::new(drv),
        })
    }
}

struct GpuRuntime {
    drv: &'static CudaDriver,
    fn_count: CuFunction,
    fn_scatter: CuFunction,
    fn_accumulate: CuFunction,
    fn_merge: CuFunction,
    fn_reduce: CuFunction,
    fn_ntt_coset_stage8: CuFunction,
    fn_ntt_bitrev_stage8: CuFunction,
    fn_ntt_radix2: CuFunction,
    fn_quot_eval: CuFunction,
    #[cfg(test)]
    fn_test_fp: CuFunction,
    scratch: Mutex<GpuMsmScratch>,
    ntt_scratch: Mutex<GpuNttScratch>,
}

// SAFETY: `CuFunction` handles are process-lifetime immutable handles owned by
// the retained primary CUDA context; scratch buffers are serialized by `Mutex`.
unsafe impl Send for GpuRuntime {}
unsafe impl Sync for GpuRuntime {}

static CUDA_DRIVER: OnceLock<Option<CudaDriver>> = OnceLock::new();
static GPU_RUNTIME: OnceLock<Option<GpuRuntime>> = OnceLock::new();

fn gpu_runtime() -> Option<&'static GpuRuntime> {
    GPU_RUNTIME
        .get_or_init(|| {
            let drv = CUDA_DRIVER.get_or_init(CudaDriver::load).as_ref()?;
            let src = ptx::generate_msm_ptx();
            let module = match drv.load_ptx(&src) {
                Ok(m) => m,
                Err(_err) => {
                    #[cfg(test)]
                    eprintln!("fast-plonk gpu-cuda PTX JIT failed: {_err}");
                    return None;
                }
            };
            let fn_count =
                drv.get_function(module, "msm_count_buckets").ok()?;
            let fn_scatter =
                drv.get_function(module, "msm_scatter_indices").ok()?;
            let fn_accumulate =
                drv.get_function(module, "msm_accumulate_tasks").ok()?;
            let fn_merge =
                drv.get_function(module, "msm_merge_split_buckets").ok()?;
            let fn_reduce =
                drv.get_function(module, "msm_reduce_windows").ok()?;
            let fn_ntt_coset_stage8 =
                drv.get_function(module, "ntt_coset_fold_stage8").ok()?;
            let fn_ntt_bitrev_stage8 =
                drv.get_function(module, "ntt_bitrev_stage8").ok()?;
            let fn_ntt_radix2 =
                drv.get_function(module, "ntt_radix2_stage").ok()?;
            let fn_quot_eval =
                drv.get_function(module, "quotient_eval_wave").ok()?;
            #[cfg(test)]
            let fn_test_fp = drv.get_function(module, "test_fp_ops").ok()?;
            let scratch = GpuMsmScratch::new(drv).ok()?;
            let ntt_scratch = GpuNttScratch::new(drv).ok()?;
            Some(GpuRuntime {
                drv,
                fn_count,
                fn_scatter,
                fn_accumulate,
                fn_merge,
                fn_reduce,
                fn_ntt_coset_stage8,
                fn_ntt_bitrev_stage8,
                fn_ntt_radix2,
                fn_quot_eval,
                #[cfg(test)]
                fn_test_fp,
                scratch: Mutex::new(scratch),
                ntt_scratch: Mutex::new(ntt_scratch),
            })
        })
        .as_ref()
}

/// Returns `true` if a CUDA GPU is present and the PTX module JIT-compiled.
pub(crate) fn is_gpu_available() -> bool {
    gpu_runtime().is_some()
}

/// Choose optimal GPU Pippenger window width `w` (`num_buckets = 2^(w-1)`
/// always divisible by 64).
#[inline]
fn choose_gpu_window(msm_len: usize) -> usize {
    if msm_len <= 4096 {
        10 // 512 buckets, 26 windows
    } else if msm_len <= 16384 {
        11 // 1024 buckets, 24 windows
    } else if msm_len <= 32768 {
        12 // 2048 buckets, 22 windows
    } else {
        13 // 4096 buckets, 20 windows
    }
}

/// Batch-convert GPU XYZZ window sums into `blst_p1_affine` using ONE
/// `blst_fp_eucl_inverse` via Montgomery's trick.
fn xyzz_batch_to_blst_affine(points: &[GpuXyzz]) -> Vec<blst_p1_affine> {
    let n = points.len();
    let mut out = vec![blst_p1_affine::default(); n];
    if n == 0 {
        return out;
    }

    // Collect non-identity indices; invert both ZZ and ZZZ in one batch!
    let mut active = Vec::with_capacity(n);
    let mut zs = Vec::with_capacity(n * 2);
    for (i, p) in points.iter().enumerate() {
        let zz_zero =
            (p.zz[0] | p.zz[1] | p.zz[2] | p.zz[3] | p.zz[4] | p.zz[5]) == 0;
        if zz_zero {
            continue;
        }
        active.push(i);
        zs.push(blst_fp { l: p.zz });
        zs.push(blst_fp { l: p.zzz });
    }
    if zs.is_empty() {
        return out;
    }

    let m = zs.len();
    let mut prods = vec![FP_ONE_BLST; m];
    prods[0] = zs[0];
    for k in 1..m {
        let prev = prods[k - 1];
        // SAFETY: valid `blst_fp` pointers.
        unsafe { blst_fp_mul(&mut prods[k], &prev, &zs[k]) };
    }
    let mut acc_inv = blst_fp::default();
    // SAFETY: every entry in `zs` is non-zero (`ZZ != 0 => ZZZ != 0`).
    unsafe { blst_fp_eucl_inverse(&mut acc_inv, &prods[m - 1]) };

    let mut invs = vec![blst_fp::default(); m];
    for k in (0..m).rev() {
        if k > 0 {
            // SAFETY: valid `blst_fp` pointers.
            unsafe {
                blst_fp_mul(&mut invs[k], &acc_inv, &prods[k - 1]);
                let cur = acc_inv;
                blst_fp_mul(&mut acc_inv, &cur, &zs[k]);
            }
        } else {
            invs[0] = acc_inv;
        }
    }

    for (slot, &orig_idx) in active.iter().enumerate() {
        let p = &points[orig_idx];
        let x_in = blst_fp { l: p.x };
        let y_in = blst_fp { l: p.y };
        let mut x_aff = blst_fp::default();
        let mut y_aff = blst_fp::default();
        // SAFETY: valid `blst_fp` pointers.
        unsafe {
            blst_fp_mul(&mut x_aff, &x_in, &invs[slot * 2]);
            blst_fp_mul(&mut y_aff, &y_in, &invs[slot * 2 + 1]);
        }
        out[orig_idx] = blst_p1_affine { x: x_aff, y: y_aff };
    }
    out
}

/// Device-resident SRS bases owned by a `Prover` (lazily uploaded on first GPU
/// MSM and automatically freed on drop).
pub(crate) struct GpuCommitBases {
    base_len: usize,
    bases_d: CudaBuf,
}

impl GpuCommitBases {
    /// Upload `bases` (`blst_p1_affine` = 96-byte Montgomery `x || y`) to GPU
    /// VRAM once.
    pub(crate) fn upload(bases: &[blst_p1_affine]) -> Option<Self> {
        let rt = gpu_runtime()?;
        let base_len = bases.len();
        if base_len < GPU_MSM_MIN_LEN {
            return None;
        }
        rt.drv.bind_ctx().ok()?;
        let bytes = core::mem::size_of_val(bases);
        let bases_d = CudaBuf::alloc(rt.drv, bytes).ok()?;
        rt.drv.copy_htod(bases_d.ptr, bases).ok()?;
        Some(Self { base_len, bases_d })
    }

    /// Evaluate a batch of `K` MSMs sharing the device-resident SRS bases on
    /// the CUDA GPU in one serialized scratch pass.
    pub(crate) fn msm_batch<const K: usize>(
        &self,
        polys: [&[BlsScalar]; K],
    ) -> Option<[G1Affine; K]> {
        #[cfg(test)]
        {
            self.msm_batch_inner(polys).unwrap_or_else(|(step, rc)| {
                eprintln!(
                    "fast-plonk gpu-cuda msm_batch failed at {step}: rc={rc}"
                );
                None
            })
        }
        #[cfg(not(test))]
        {
            self.msm_batch_inner(polys).unwrap_or_default()
        }
    }

    fn msm_batch_inner<const K: usize>(
        &self,
        polys: [&[BlsScalar]; K],
    ) -> Result<Option<[G1Affine; K]>, (&'static str, i32)> {
        let mut lengths = [0usize; K];
        for (i, poly) in polys.iter().enumerate() {
            let mut len = poly.len();
            while len > 0 && poly[len - 1] == BlsScalar::zero() {
                len -= 1;
            }
            if len > self.base_len {
                return Ok(None);
            }
            lengths[i] = len;
        }

        let max_len = lengths.iter().copied().max().unwrap_or(0);
        if max_len == 0 {
            return Ok(Some([G1Affine::identity(); K]));
        }
        if max_len < GPU_MSM_MIN_LEN {
            return Ok(None);
        }

        let Some(rt) = gpu_runtime() else {
            return Ok(None);
        };
        rt.drv.bind_ctx().map_err(|e| ("bind_ctx", e))?;

        let mut active_indices = [0usize; K];
        let mut active_count = 0usize;
        for (i, &len) in lengths.iter().enumerate() {
            if len > 0 {
                active_indices[active_count] = i;
                active_count += 1;
            }
        }

        let msm_len = max_len;
        let total_scalars = active_count * msm_len;
        let w = choose_gpu_window(msm_len);
        let ny = 256usize.div_ceil(w);
        let num_buckets = 1usize << (w - 1);
        let num_windows_total = active_count * ny;
        let total_buckets = num_windows_total * num_buckets;
        let log_chunk = (num_buckets / 64).trailing_zeros();

        let mut scratch_guard = rt.scratch.lock().map_err(|_| ("mutex", -1))?;
        let scratch = &mut *scratch_guard;
        let scalars_bytes = total_scalars * 32;
        scratch
            .scalars_mont
            .ensure_capacity(scalars_bytes)
            .map_err(|e| ("alloc_sm", e))?;
        scratch
            .scalars_std
            .ensure_capacity(scalars_bytes)
            .map_err(|e| ("alloc_ss", e))?;
        scratch
            .counts
            .ensure_capacity(total_buckets * 4)
            .map_err(|e| ("alloc_cnt", e))?;
        scratch
            .cursors
            .ensure_capacity(total_buckets * 4)
            .map_err(|e| ("alloc_cur", e))?;
        scratch
            .window_sums
            .ensure_capacity(
                num_windows_total * core::mem::size_of::<GpuXyzz>(),
            )
            .map_err(|e| ("alloc_wsum", e))?;

        // Pack active polynomials' raw Montgomery limbs `[u64; 4]` directly
        // into reusable pinned host memory (zero allocation / zero page faults).
        let packed = scratch.host_packed.ensure_len(total_scalars);
        packed.par_chunks_exact_mut(msm_len).enumerate().for_each(
            |(slot, dst)| {
                let poly_idx = active_indices[slot];
                let src = &polys[poly_idx][..lengths[poly_idx]];
                for (d, s) in dst[..src.len()].iter_mut().zip(src.iter()) {
                    *d = *s.internal_repr();
                }
                for d in &mut dst[src.len()..] {
                    *d = [0u64; 4];
                }
            },
        );

        // 1. Upload scalars via pinned DMA and zero counts.
        rt.drv
            .copy_htod(scratch.scalars_mont.ptr, packed)
            .map_err(|e| ("htod_sc", e))?;
        rt.drv
            .memset_zero(scratch.counts.ptr, total_buckets * 4)
            .map_err(|e| ("zero_cnt", e))?;

        // 2. Count bucket hits & convert scalars Montgomery -> standard on GPU.
        let mut p_sm = scratch.scalars_mont.ptr;
        let mut p_ss = scratch.scalars_std.ptr;
        let mut p_cnt = scratch.counts.ptr;
        let mut p_msm_len = msm_len as u32;
        let mut p_total_sc = total_scalars as u32;
        let mut p_w = w as u32;
        let mut p_ny = ny as u32;
        let mut p_nb = num_buckets as u32;
        let mut params_cnt: [*mut c_void; 8] = [
            (&mut p_sm as *mut u64).cast(),
            (&mut p_ss as *mut u64).cast(),
            (&mut p_cnt as *mut u64).cast(),
            (&mut p_msm_len as *mut u32).cast(),
            (&mut p_total_sc as *mut u32).cast(),
            (&mut p_w as *mut u32).cast(),
            (&mut p_ny as *mut u32).cast(),
            (&mut p_nb as *mut u32).cast(),
        ];
        let grid_sc = (total_scalars as u32).div_ceil(256);
        // SAFETY: parameter array matches `msm_count_buckets` PTX signature.
        unsafe {
            rt.drv
                .launch(
                    rt.fn_count,
                    (grid_sc, 1, 1),
                    (256, 1, 1),
                    0,
                    &mut params_cnt,
                )
                .map_err(|e| ("launch_count", e))?;
        }

        // 3. Exclusive prefix sum over `counts` -> `cursors` + load-balanced
        // sub-bucket tasks (`<= max_bucket_chunk` additions per thread).
        let counts_ptr = scratch.counts.ptr;
        let counts_h = scratch.host_counts.ensure_len(total_buckets);
        rt.drv
            .copy_dtoh(counts_h, counts_ptr)
            .map_err(|e| ("dtoh_cnt", e))?;
        // Adaptive split threshold: ~2x the average bucket occupancy (min 32)
        // so normal Poisson buckets stay single-task with zero merge overhead,
        // while narrow top-window or skewed buckets split across parallel threads.
        let max_bucket_chunk = ((msm_len / num_buckets) * 2).max(32) as u32;
        let cursors_h = scratch.host_cursors.ensure_len(total_buckets);
        let mut tasks_h = Vec::with_capacity(total_buckets);
        let mut merges_h = Vec::new();
        let mut running = 0u32;
        let mut next_tmp = total_buckets as u32;
        for (b, &c) in counts_h.iter().enumerate() {
            let start = running;
            cursors_h[b] = start;
            running += c;
            if c == 0 {
                continue;
            }
            if c <= max_bucket_chunk {
                tasks_h.push(GpuAccTask {
                    start,
                    end: running,
                    dst: b as u32,
                });
            } else {
                let tmp_start = next_tmp;
                let mut cur = start;
                while cur < running {
                    let end = (cur + max_bucket_chunk).min(running);
                    tasks_h.push(GpuAccTask {
                        start: cur,
                        end,
                        dst: next_tmp,
                    });
                    next_tmp += 1;
                    cur = end;
                }
                merges_h.push(GpuSplitMerge {
                    dst_bucket: b as u32,
                    tmp_start,
                    tmp_count: next_tmp - tmp_start,
                });
            }
        }
        let total_entries = running as usize;
        if total_entries == 0 {
            return Ok(Some([G1Affine::identity(); K]));
        }
        let total_bucket_slots = next_tmp as usize;

        rt.drv
            .copy_htod(scratch.cursors.ptr, cursors_h)
            .map_err(|e| ("htod_cur", e))?;
        scratch
            .sorted
            .ensure_capacity(total_entries * 4)
            .map_err(|e| ("alloc_srt", e))?;
        scratch
            .tasks
            .ensure_capacity(tasks_h.len() * core::mem::size_of::<GpuAccTask>())
            .map_err(|e| ("alloc_tsk", e))?;
        scratch
            .buckets
            .ensure_capacity(
                total_bucket_slots * core::mem::size_of::<GpuXyzz>(),
            )
            .map_err(|e| ("alloc_bkt", e))?;
        if !merges_h.is_empty() {
            scratch
                .merges
                .ensure_capacity(
                    merges_h.len() * core::mem::size_of::<GpuSplitMerge>(),
                )
                .map_err(|e| ("alloc_mrg", e))?;
            rt.drv
                .copy_htod(scratch.merges.ptr, &merges_h)
                .map_err(|e| ("htod_mrg", e))?;
        }
        rt.drv
            .copy_htod(scratch.tasks.ptr, &tasks_h)
            .map_err(|e| ("htod_tsk", e))?;
        rt.drv
            .memset_zero(
                scratch.buckets.ptr,
                total_buckets * core::mem::size_of::<GpuXyzz>(),
            )
            .map_err(|e| ("zero_bkt", e))?;

        // 4. Scatter `(point_idx | (neg << 31))` into bucket-sorted array.
        let mut p_cur = scratch.cursors.ptr;
        let mut p_srt = scratch.sorted.ptr;
        let mut params_sct: [*mut c_void; 8] = [
            (&mut p_ss as *mut u64).cast(),
            (&mut p_cur as *mut u64).cast(),
            (&mut p_srt as *mut u64).cast(),
            (&mut p_msm_len as *mut u32).cast(),
            (&mut p_total_sc as *mut u32).cast(),
            (&mut p_w as *mut u32).cast(),
            (&mut p_ny as *mut u32).cast(),
            (&mut p_nb as *mut u32).cast(),
        ];
        // SAFETY: parameter array matches `msm_scatter_indices` PTX signature.
        unsafe {
            rt.drv
                .launch(
                    rt.fn_scatter,
                    (grid_sc, 1, 1),
                    (256, 1, 1),
                    0,
                    &mut params_sct,
                )
                .map_err(|e| ("launch_scatter", e))?;
        }
        // 5. Accumulate all load-balanced bucket tasks in parallel.
        let mut p_bases = self.bases_d.ptr;
        let mut p_tsk = scratch.tasks.ptr;
        let mut p_bkt = scratch.buckets.ptr;
        let mut p_num_t = tasks_h.len() as u32;
        let mut params_acc: [*mut c_void; 5] = [
            (&mut p_bases as *mut u64).cast(),
            (&mut p_tsk as *mut u64).cast(),
            (&mut p_srt as *mut u64).cast(),
            (&mut p_bkt as *mut u64).cast(),
            (&mut p_num_t as *mut u32).cast(),
        ];
        let grid_tsk = p_num_t.div_ceil(128);
        // SAFETY: parameter array matches `msm_accumulate_tasks` signature.
        unsafe {
            rt.drv
                .launch(
                    rt.fn_accumulate,
                    (grid_tsk, 1, 1),
                    (128, 1, 1),
                    0,
                    &mut params_acc,
                )
                .map_err(|e| ("launch_accumulate", e))?;
        }
        if !merges_h.is_empty() {
            let mut p_mrg = scratch.merges.ptr;
            let mut params_mrg: [*mut c_void; 2] = [
                (&mut p_bkt as *mut u64).cast(),
                (&mut p_mrg as *mut u64).cast(),
            ];
            // SAFETY: parameter array matches `msm_merge_split_buckets` signature.
            unsafe {
                rt.drv
                    .launch(
                        rt.fn_merge,
                        (merges_h.len() as u32, 1, 1),
                        (32, 1, 1),
                        0,
                        &mut params_mrg,
                    )
                    .map_err(|e| ("launch_merge", e))?;
            }
        }
        // 6. Reduce each window (64 threads per window, parallel 6-level tree).
        let mut p_wsums = scratch.window_sums.ptr;
        let mut p_log_c = log_chunk;
        let mut params_red: [*mut c_void; 4] = [
            (&mut p_bkt as *mut u64).cast(),
            (&mut p_wsums as *mut u64).cast(),
            (&mut p_nb as *mut u32).cast(),
            (&mut p_log_c as *mut u32).cast(),
        ];
        // SAFETY: parameter array matches `msm_reduce_windows` signature.
        unsafe {
            rt.drv
                .launch(
                    rt.fn_reduce,
                    (num_windows_total as u32, 1, 1),
                    (64, 1, 1),
                    0,
                    &mut params_red,
                )
                .map_err(|e| ("launch_reduce", e))?;
        }
        // 7. Copy window sums (~17 KB) to host and combine into `K` G1Affines.
        let mut wsums_h = vec![GpuXyzz::default(); num_windows_total];
        rt.drv
            .copy_dtoh(&mut wsums_h, scratch.window_sums.ptr)
            .map_err(|e| ("dtoh_wsums", e))?;
        drop(scratch_guard);

        let wsums_aff = xyzz_batch_to_blst_affine(&wsums_h);
        let mut projs = [blst_p1::default(); K];
        for (slot, &poly_idx) in
            active_indices[..active_count].iter().enumerate()
        {
            let slice = &wsums_aff[slot * ny..(slot + 1) * ny];
            let mut acc = blst_p1::default();
            for wi in (0..ny).rev() {
                // SAFETY: valid Jacobian + affine pointers.
                unsafe {
                    blst_p1_add_or_double_affine(&mut acc, &acc, &slice[wi]);
                }
                if wi > 0 {
                    for _ in 0..w {
                        // SAFETY: valid Jacobian point.
                        unsafe { blst_p1_double(&mut acc, &acc) };
                    }
                }
            }
            projs[poly_idx] = acc;
        }

        Ok(Some(blst_p1s_to_g1_affines(&projs)))
    }
}

/// Device-resident `Fr` NTT twiddle, coset, and selector tables owned lazily by
/// `FastProverCache`. Keeps the entire 8-coset quotient pipeline (40 forward
/// coset NTTs + 524K pointwise PLONK widget evaluations + 8 inverse coset
/// IFFTs) resident in VRAM with zero intermediate PCIe transfers.
pub(crate) struct GpuNttTables {
    n: usize,
    log_n: u32,
    fft_roots_d: CudaBuf,
    ifft_roots_d: CudaBuf,
    size_inv_d: CudaBuf,
    /// Flat `[8 * n]` `FastFr` coset powers (`coset k` at offset `k * n`).
    coset_powers_d: CudaBuf,
    /// Flat `[8]` `FastFr` `c_k^n` values.
    ck_pown_d: CudaBuf,
    /// Owned VRAM buffers for the 13–17 flat `[8 * n]` selector/precomputed
    /// evaluation tables.
    _sel_bufs: Vec<CudaBuf>,
    /// Device `[u64; 17]` pointer table (`0` for absent optional selectors).
    sel_ptrs_d: CudaBuf,
    /// Precomputed `vanishing_inv[0..8]` for each coset.
    vanishing_inv: [FastFr; 8],
}

impl GpuNttTables {
    /// Upload `domain_fft_roots`, `domain_ifft_roots`, `domain_size_inv`,
    /// coset powers, and all 17 selector/precomputed evaluation tables once to
    /// GPU VRAM.
    ///
    /// `selectors[s]` (`0..17`) holds `Option<[&[FastFr]; 8]>` in the exact
    /// order expected by `quotient_eval_wave`:
    /// `q_m, q_l, q_r, q_o, q_f, q_c, q_arith, q_range, q_logic, q_fixed,
    ///  q_var, s_sigma_1..4, linear_evals, l1_evals`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn upload(
        n: usize,
        log_n: u32,
        fft_roots: &[FastFr],
        ifft_roots: &[FastFr],
        size_inv: FastFr,
        coset_powers: [&[FastFr]; 8],
        ck_pown: [FastFr; 8],
        vanishing_inv: [FastFr; 8],
        selectors: [Option<[Option<&[FastFr]>; 8]>; 17],
    ) -> Option<Self> {
        if n < GPU_NTT_MIN_LEN || !n.is_power_of_two() || log_n < 8 {
            return None;
        }
        let rt = gpu_runtime()?;
        rt.drv.bind_ctx().ok()?;

        let fft_roots_d =
            CudaBuf::alloc(rt.drv, core::mem::size_of_val(fft_roots)).ok()?;
        rt.drv.copy_htod(fft_roots_d.ptr, fft_roots).ok()?;

        let ifft_roots_d =
            CudaBuf::alloc(rt.drv, core::mem::size_of_val(ifft_roots)).ok()?;
        rt.drv.copy_htod(ifft_roots_d.ptr, ifft_roots).ok()?;

        let size_inv_d =
            CudaBuf::alloc(rt.drv, core::mem::size_of::<FastFr>()).ok()?;
        rt.drv.copy_htod(size_inv_d.ptr, &[size_inv]).ok()?;

        let mut flat = Vec::with_capacity(8 * n);
        for cp in coset_powers {
            flat.extend_from_slice(cp);
        }
        let coset_powers_d =
            CudaBuf::alloc(rt.drv, core::mem::size_of_val(flat.as_slice()))
                .ok()?;
        rt.drv.copy_htod(coset_powers_d.ptr, &flat).ok()?;

        let ck_pown_d =
            CudaBuf::alloc(rt.drv, core::mem::size_of_val(&ck_pown)).ok()?;
        rt.drv.copy_htod(ck_pown_d.ptr, &ck_pown).ok()?;

        let mut sel_bufs = Vec::with_capacity(17);
        let mut sel_ptrs = [0u64; 17];
        let sel_bytes = 8 * n * core::mem::size_of::<FastFr>();
        for (s_idx, opt_cosets) in selectors.into_iter().enumerate() {
            let Some(coset_slices) = opt_cosets else {
                continue;
            };
            flat.clear();
            for opt_slice in coset_slices {
                if let Some(sl) = opt_slice {
                    flat.extend_from_slice(sl);
                } else {
                    flat.resize(flat.len() + n, FastFr::ZERO);
                }
            }
            let buf = CudaBuf::alloc(rt.drv, sel_bytes).ok()?;
            rt.drv.copy_htod(buf.ptr, &flat).ok()?;
            sel_ptrs[s_idx] = buf.ptr;
            sel_bufs.push(buf);
        }

        let sel_ptrs_d =
            CudaBuf::alloc(rt.drv, core::mem::size_of_val(&sel_ptrs)).ok()?;
        rt.drv.copy_htod(sel_ptrs_d.ptr, &sel_ptrs).ok()?;

        Some(Self {
            n,
            log_n,
            fft_roots_d,
            ifft_roots_d,
            size_inv_d,
            coset_powers_d,
            ck_pown_d,
            _sel_bufs: sel_bufs,
            sel_ptrs_d,
            vanishing_inv,
        })
    }

    /// Run the **entire** 8-coset quotient pipeline in GPU VRAM:
    /// 1. Upload `5` (or `6`) source polynomials ONCE via pinned DMA.
    /// 2. For both waves (`0..4` and `4..8`): fused coset NTTs + pointwise
    ///    6-widget PLONK constraint evaluation + vanishing division in VRAM.
    /// 3. Run all 8 coset IFFTs in VRAM (with fused `size_inv` scaling).
    /// 4. Download ONLY the 8 `n`-point coset IFFTs ONCE via pinned DMA and
    ///    invoke `f` with the flat `&[FastFr]` slice (`[8 * n]`, zero copy).
    pub(crate) fn with_full_quotient_iffts<F, R>(
        &self,
        polys: &[&[BlsScalar]],
        challenges: &[FastFr; 36],
        f: F,
    ) -> Option<R>
    where
        F: FnOnce(&[FastFr]) -> R,
    {
        let num_polys = polys.len();
        if num_polys != 5 && num_polys != 6 {
            return None;
        }
        let max_len = polys.iter().map(|p| p.len()).max().unwrap_or(0);
        if max_len > 2 * self.n {
            return None;
        }
        let src_stride = max_len.max(self.n);
        let rt = gpu_runtime()?;
        rt.drv.bind_ctx().ok()?;

        let wave_batch = 4 * num_polys;
        let wave_elems = wave_batch * self.n;
        let q_elems = 8 * self.n;
        let out_elems = wave_elems.max(q_elems);
        let src_elems = num_polys * src_stride;

        let mut scratch_guard = rt.ntt_scratch.lock().ok()?;
        let scratch = &mut *scratch_guard;
        scratch
            .src
            .ensure_capacity(src_elems * core::mem::size_of::<FastFr>())
            .ok()?;
        scratch
            .out
            .ensure_capacity(out_elems * core::mem::size_of::<FastFr>())
            .ok()?;
        scratch
            .q_evals
            .ensure_capacity(q_elems * core::mem::size_of::<FastFr>())
            .ok()?;

        let src_ptr = scratch.src.ptr;
        let out_ptr = scratch.out.ptr;
        let q_ptr = scratch.q_evals.ptr;
        let ch_ptr = scratch.ch_buf.ptr;

        // 1. Pack & DMA-upload source polynomials ONCE + tiny 1.4 KB challenges.
        let packed = scratch.host_src.ensure_len(src_elems);
        packed
            .par_chunks_exact_mut(src_stride)
            .zip(polys.par_iter())
            .for_each(|(dst, src)| {
                for (d, s) in dst[..src.len()].iter_mut().zip(src.iter()) {
                    *d = FastFr::from_bls(s);
                }
                for d in &mut dst[src.len()..] {
                    *d = FastFr::ZERO;
                }
            });
        rt.drv.copy_htod(src_ptr, packed).ok()?;

        let mut ch_all = [FastFr::ZERO; 44];
        ch_all[..36].copy_from_slice(challenges);
        ch_all[36..44].copy_from_slice(&self.vanishing_inv);
        rt.drv.copy_htod(ch_ptr, &ch_all).ok()?;

        // 2. Wave 0 and Wave 1: forward coset NTTs + pointwise evaluation in VRAM.
        let mut p_src = src_ptr;
        let mut p_fft_roots = self.fft_roots_d.ptr;
        let mut p_out = out_ptr;
        let mut p_q_out = q_ptr;
        let mut p_sel_ptrs = self.sel_ptrs_d.ptr;
        let mut p_ch = ch_ptr;
        let mut p_n = self.n as u32;
        let mut p_log_n = self.log_n;
        let mut p_stride = src_stride as u32;
        let mut p_npolys = num_polys as u32;
        let mut p_zero_scale: u64 = 0;
        let mut p_wave_bf = (wave_batch * (self.n / 2)) as u32;
        let grid_cf = (wave_batch * (self.n / 256)) as u32;
        let grid_wave_bf = p_wave_bf.div_ceil(256);
        let grid_eval = ((4 * self.n) as u32).div_ceil(256);

        for wave in 0..2usize {
            let k_base = wave * 4;
            let mut p_cp = self.coset_powers_d.ptr
                + (k_base * self.n * core::mem::size_of::<FastFr>()) as u64;
            let mut p_ckp = self.ck_pown_d.ptr
                + (k_base * core::mem::size_of::<FastFr>()) as u64;
            let mut params_cf: [*mut c_void; 9] = [
                (&mut p_src as *mut u64).cast(),
                (&mut p_cp as *mut u64).cast(),
                (&mut p_ckp as *mut u64).cast(),
                (&mut p_fft_roots as *mut u64).cast(),
                (&mut p_out as *mut u64).cast(),
                (&mut p_n as *mut u32).cast(),
                (&mut p_log_n as *mut u32).cast(),
                (&mut p_stride as *mut u32).cast(),
                (&mut p_npolys as *mut u32).cast(),
            ];
            // SAFETY: matches `ntt_coset_fold_stage8` signature.
            unsafe {
                rt.drv
                    .launch(
                        rt.fn_ntt_coset_stage8,
                        (grid_cf, 1, 1),
                        (128, 1, 1),
                        0,
                        &mut params_cf,
                    )
                    .ok()?;
            }

            for stage in 8..self.log_n {
                let mut p_stage = stage;
                let mut params_r2: [*mut c_void; 7] = [
                    (&mut p_out as *mut u64).cast(),
                    (&mut p_fft_roots as *mut u64).cast(),
                    (&mut p_zero_scale as *mut u64).cast(),
                    (&mut p_n as *mut u32).cast(),
                    (&mut p_log_n as *mut u32).cast(),
                    (&mut p_stage as *mut u32).cast(),
                    (&mut p_wave_bf as *mut u32).cast(),
                ];
                // SAFETY: matches `ntt_radix2_stage` signature.
                unsafe {
                    rt.drv
                        .launch(
                            rt.fn_ntt_radix2,
                            (grid_wave_bf, 1, 1),
                            (256, 1, 1),
                            0,
                            &mut params_r2,
                        )
                        .ok()?;
                }
            }

            let mut p_k_base = k_base as u32;
            let mut params_ev: [*mut c_void; 8] = [
                (&mut p_out as *mut u64).cast(),
                (&mut p_sel_ptrs as *mut u64).cast(),
                (&mut p_ch as *mut u64).cast(),
                (&mut p_q_out as *mut u64).cast(),
                (&mut p_n as *mut u32).cast(),
                (&mut p_log_n as *mut u32).cast(),
                (&mut p_k_base as *mut u32).cast(),
                (&mut p_npolys as *mut u32).cast(),
            ];
            // SAFETY: matches `quotient_eval_wave` signature.
            unsafe {
                rt.drv
                    .launch(
                        rt.fn_quot_eval,
                        (grid_eval, 1, 1),
                        (256, 1, 1),
                        0,
                        &mut params_ev,
                    )
                    .ok()?;
            }
        }

        // 3. Run all 8 coset IFFTs directly from `q_evals` -> `out` in VRAM.
        let mut p_ifft_roots = self.ifft_roots_d.ptr;
        let mut params_br: [*mut c_void; 5] = [
            (&mut p_q_out as *mut u64).cast(),
            (&mut p_ifft_roots as *mut u64).cast(),
            (&mut p_out as *mut u64).cast(),
            (&mut p_n as *mut u32).cast(),
            (&mut p_log_n as *mut u32).cast(),
        ];
        let grid_br = (8 * (self.n / 256)) as u32;
        // SAFETY: matches `ntt_bitrev_stage8` signature.
        unsafe {
            rt.drv
                .launch(
                    rt.fn_ntt_bitrev_stage8,
                    (grid_br, 1, 1),
                    (128, 1, 1),
                    0,
                    &mut params_br,
                )
                .ok()?;
        }

        let mut p_ifft_bf = (8 * (self.n / 2)) as u32;
        let grid_ifft_bf = p_ifft_bf.div_ceil(256);
        for stage in 8..self.log_n {
            let mut p_stage = stage;
            let mut p_scale = if stage + 1 == self.log_n {
                self.size_inv_d.ptr
            } else {
                0
            };
            let mut params_r2: [*mut c_void; 7] = [
                (&mut p_out as *mut u64).cast(),
                (&mut p_ifft_roots as *mut u64).cast(),
                (&mut p_scale as *mut u64).cast(),
                (&mut p_n as *mut u32).cast(),
                (&mut p_log_n as *mut u32).cast(),
                (&mut p_stage as *mut u32).cast(),
                (&mut p_ifft_bf as *mut u32).cast(),
            ];
            // SAFETY: matches `ntt_radix2_stage` signature.
            unsafe {
                rt.drv
                    .launch(
                        rt.fn_ntt_radix2,
                        (grid_ifft_bf, 1, 1),
                        (256, 1, 1),
                        0,
                        &mut params_r2,
                    )
                    .ok()?;
            }
        }

        // 4. Single pinned DMA download of the 8 coset IFFTs (`8 * n` FastFrs).
        let flat_out = scratch.host_out.ensure_len(q_elems);
        rt.drv.copy_dtoh(flat_out, out_ptr).ok()?;
        Some(f(flat_out))
    }

    /// Evaluate 4 cosets (`k_base .. k_base + 4`) × `polys.len()` polynomials
    /// (`5` or `6`) in one batched GPU NTT pass and invoke `f` with the flat
    /// pinned host slice (`[4 * polys.len() * n]`, zero extra copies/allocs!).
    #[cfg(test)]
    pub(crate) fn with_coset_ffts_wave<F>(
        &self,
        k_base: usize,
        polys: &[&[BlsScalar]],
        f: F,
    ) -> Option<()>
    where
        F: FnOnce(&[FastFr]),
    {
        let num_polys = polys.len();
        if num_polys == 0 || k_base + 4 > 8 {
            return None;
        }
        let max_len = polys.iter().map(|p| p.len()).max().unwrap_or(0);
        if max_len > 2 * self.n {
            return None;
        }
        let src_stride = max_len.max(self.n);
        let rt = gpu_runtime()?;
        rt.drv.bind_ctx().ok()?;

        let batch = 4 * num_polys;
        let out_elems = batch * self.n;
        let src_elems = num_polys * src_stride;
        let mut scratch_guard = rt.ntt_scratch.lock().ok()?;
        let scratch = &mut *scratch_guard;
        scratch
            .src
            .ensure_capacity(src_elems * core::mem::size_of::<FastFr>())
            .ok()?;
        scratch
            .out
            .ensure_capacity(out_elems * core::mem::size_of::<FastFr>())
            .ok()?;

        let src_ptr = scratch.src.ptr;
        let out_ptr = scratch.out.ptr;
        let packed = scratch.host_src.ensure_len(src_elems);
        packed
            .par_chunks_exact_mut(src_stride)
            .zip(polys.par_iter())
            .for_each(|(dst, src)| {
                for (d, s) in dst[..src.len()].iter_mut().zip(src.iter()) {
                    *d = FastFr::from_bls(s);
                }
                for d in &mut dst[src.len()..] {
                    *d = FastFr::ZERO;
                }
            });
        rt.drv.copy_htod(src_ptr, packed).ok()?;

        let mut p_src = src_ptr;
        let mut p_cp = self.coset_powers_d.ptr
            + (k_base * self.n * core::mem::size_of::<FastFr>()) as u64;
        let mut p_ckp = self.ck_pown_d.ptr
            + (k_base * core::mem::size_of::<FastFr>()) as u64;
        let mut p_roots = self.fft_roots_d.ptr;
        let mut p_out = out_ptr;
        let mut p_n = self.n as u32;
        let mut p_log_n = self.log_n;
        let mut p_stride = src_stride as u32;
        let mut p_npolys = num_polys as u32;
        let mut params_cf: [*mut c_void; 9] = [
            (&mut p_src as *mut u64).cast(),
            (&mut p_cp as *mut u64).cast(),
            (&mut p_ckp as *mut u64).cast(),
            (&mut p_roots as *mut u64).cast(),
            (&mut p_out as *mut u64).cast(),
            (&mut p_n as *mut u32).cast(),
            (&mut p_log_n as *mut u32).cast(),
            (&mut p_stride as *mut u32).cast(),
            (&mut p_npolys as *mut u32).cast(),
        ];
        let grid_cf = (batch * (self.n / 256)) as u32;
        // SAFETY: parameter array matches `ntt_coset_fold_stage8` signature.
        unsafe {
            rt.drv
                .launch(
                    rt.fn_ntt_coset_stage8,
                    (grid_cf, 1, 1),
                    (128, 1, 1),
                    0,
                    &mut params_cf,
                )
                .ok()?;
        }

        let mut p_zero_scale: u64 = 0;
        let mut p_total_bf = (batch * (self.n / 2)) as u32;
        let grid_bf = p_total_bf.div_ceil(256);
        for stage in 8..self.log_n {
            let mut p_stage = stage;
            let mut params_r2: [*mut c_void; 7] = [
                (&mut p_out as *mut u64).cast(),
                (&mut p_roots as *mut u64).cast(),
                (&mut p_zero_scale as *mut u64).cast(),
                (&mut p_n as *mut u32).cast(),
                (&mut p_log_n as *mut u32).cast(),
                (&mut p_stage as *mut u32).cast(),
                (&mut p_total_bf as *mut u32).cast(),
            ];
            // SAFETY: parameter array matches `ntt_radix2_stage` signature.
            unsafe {
                rt.drv
                    .launch(
                        rt.fn_ntt_radix2,
                        (grid_bf, 1, 1),
                        (256, 1, 1),
                        0,
                        &mut params_r2,
                    )
                    .ok()?;
            }
        }

        let flat_out = scratch.host_out.ensure_len(out_elems);
        rt.drv.copy_dtoh(flat_out, out_ptr).ok()?;
        f(flat_out);
        Some(())
    }

    #[cfg(test)]
    pub(crate) fn coset_ffts_wave(
        &self,
        k_base: usize,
        polys: &[&[BlsScalar]],
    ) -> Option<Vec<Vec<FastFr>>> {
        let mut result = Vec::new();
        self.with_coset_ffts_wave(k_base, polys, |flat| {
            result = flat.chunks_exact(self.n).map(|c| c.to_vec()).collect();
        })?;
        Some(result)
    }

    /// Run a batch of `n`-point inverse NTTs (`evals -> coeffs`, scaled by
    /// `domain_size_inv`) on GPU using reusable pinned host DMA buffers.
    #[cfg(test)]
    pub(crate) fn batch_ifft(
        &self,
        evals: &[Vec<FastFr>],
    ) -> Option<Vec<Vec<FastFr>>> {
        let batch = evals.len();
        if batch == 0 || evals.iter().any(|v| v.len() != self.n) {
            return None;
        }
        let rt = gpu_runtime()?;
        rt.drv.bind_ctx().ok()?;

        let total_elems = batch * self.n;
        let bytes = total_elems * core::mem::size_of::<FastFr>();
        let mut scratch_guard = rt.ntt_scratch.lock().ok()?;
        let scratch = &mut *scratch_guard;
        scratch.src.ensure_capacity(bytes).ok()?;
        scratch.out.ensure_capacity(bytes).ok()?;

        let src_ptr = scratch.src.ptr;
        let out_ptr = scratch.out.ptr;
        let flat_in = scratch.host_src.ensure_len(total_elems);
        flat_in
            .par_chunks_exact_mut(self.n)
            .zip(evals.par_iter())
            .for_each(|(dst, src)| dst.copy_from_slice(src));
        rt.drv.copy_htod(src_ptr, flat_in).ok()?;

        let mut p_src = src_ptr;
        let mut p_roots = self.ifft_roots_d.ptr;
        let mut p_out = out_ptr;
        let mut p_n = self.n as u32;
        let mut p_log_n = self.log_n;
        let mut params_br: [*mut c_void; 5] = [
            (&mut p_src as *mut u64).cast(),
            (&mut p_roots as *mut u64).cast(),
            (&mut p_out as *mut u64).cast(),
            (&mut p_n as *mut u32).cast(),
            (&mut p_log_n as *mut u32).cast(),
        ];
        let grid_br = (batch * (self.n / 256)) as u32;
        // SAFETY: parameter array matches `ntt_bitrev_stage8` signature.
        unsafe {
            rt.drv
                .launch(
                    rt.fn_ntt_bitrev_stage8,
                    (grid_br, 1, 1),
                    (128, 1, 1),
                    0,
                    &mut params_br,
                )
                .ok()?;
        }

        let mut p_total_bf = (batch * (self.n / 2)) as u32;
        let grid_bf = p_total_bf.div_ceil(256);
        for stage in 8..self.log_n {
            let mut p_stage = stage;
            let mut p_scale = if stage + 1 == self.log_n {
                self.size_inv_d.ptr
            } else {
                0
            };
            let mut params_r2: [*mut c_void; 7] = [
                (&mut p_out as *mut u64).cast(),
                (&mut p_roots as *mut u64).cast(),
                (&mut p_scale as *mut u64).cast(),
                (&mut p_n as *mut u32).cast(),
                (&mut p_log_n as *mut u32).cast(),
                (&mut p_stage as *mut u32).cast(),
                (&mut p_total_bf as *mut u32).cast(),
            ];
            // SAFETY: parameter array matches `ntt_radix2_stage` signature.
            unsafe {
                rt.drv
                    .launch(
                        rt.fn_ntt_radix2,
                        (grid_bf, 1, 1),
                        (256, 1, 1),
                        0,
                        &mut params_r2,
                    )
                    .ok()?;
            }
        }

        let flat_out = scratch.host_out.ensure_len(total_elems);
        rt.drv.copy_dtoh(flat_out, out_ptr).ok()?;
        let result: Vec<Vec<FastFr>> = flat_out
            .par_chunks_exact(self.n)
            .map(|chunk| chunk.to_vec())
            .collect();
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use blst::{blst_fp_add, blst_fp_cneg, blst_fp_sub};
    use dusk_bls12_381::G1Projective;
    use ff::Field;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::blst_backend::{g1_affines_to_blst, msm_g1_batch};

    #[test]
    fn gpu_cuda_fp_ops_match_blst() {
        let Some(rt) = gpu_runtime() else {
            eprintln!("skipping gpu_cuda Fp test: no CUDA GPU available");
            return;
        };
        rt.drv.bind_ctx().unwrap();

        let mut rng = StdRng::seed_from_u64(0xf00d_1234);
        let n = 64usize;
        let pts: Vec<G1Affine> = (0..n)
            .map(|_| {
                G1Affine::from(
                    G1Projective::generator() * BlsScalar::random(&mut rng),
                )
            })
            .collect();
        let blst_pts = g1_affines_to_blst(&pts);
        let mut a: Vec<blst_fp> = blst_pts.iter().map(|p| p.x).collect();
        let mut b: Vec<blst_fp> = blst_pts.iter().map(|p| p.y).collect();
        // Inject edge cases: 0 and 1_mont.
        a[0] = blst_fp::default();
        b[0] = FP_ONE_BLST;
        a[1] = FP_ONE_BLST;
        b[1] = blst_fp::default();
        a[2] = b[2];

        let bytes = n * 48;
        let d_a = CudaBuf::alloc(rt.drv, bytes).unwrap();
        let d_b = CudaBuf::alloc(rt.drv, bytes).unwrap();
        let d_mul = CudaBuf::alloc(rt.drv, bytes).unwrap();
        let d_add = CudaBuf::alloc(rt.drv, bytes).unwrap();
        let d_sub = CudaBuf::alloc(rt.drv, bytes).unwrap();
        let d_neg = CudaBuf::alloc(rt.drv, bytes).unwrap();
        rt.drv.copy_htod(d_a.ptr, &a).unwrap();
        rt.drv.copy_htod(d_b.ptr, &b).unwrap();

        let mut pa = d_a.ptr;
        let mut pb = d_b.ptr;
        let mut pm = d_mul.ptr;
        let mut pad = d_add.ptr;
        let mut ps = d_sub.ptr;
        let mut pn = d_neg.ptr;
        let mut pn_u32 = n as u32;
        let mut params: [*mut c_void; 7] = [
            (&mut pa as *mut u64).cast(),
            (&mut pb as *mut u64).cast(),
            (&mut pm as *mut u64).cast(),
            (&mut pad as *mut u64).cast(),
            (&mut ps as *mut u64).cast(),
            (&mut pn as *mut u64).cast(),
            (&mut pn_u32 as *mut u32).cast(),
        ];
        unsafe {
            rt.drv
                .launch(
                    rt.fn_test_fp,
                    (1, 1, 1),
                    (n as u32, 1, 1),
                    0,
                    &mut params,
                )
                .unwrap();
        }
        rt.drv.synchronize().unwrap();

        let mut out_mul = vec![blst_fp::default(); n];
        let mut out_add = vec![blst_fp::default(); n];
        let mut out_sub = vec![blst_fp::default(); n];
        let mut out_neg = vec![blst_fp::default(); n];
        rt.drv.copy_dtoh(&mut out_mul, d_mul.ptr).unwrap();
        rt.drv.copy_dtoh(&mut out_add, d_add.ptr).unwrap();
        rt.drv.copy_dtoh(&mut out_sub, d_sub.ptr).unwrap();
        rt.drv.copy_dtoh(&mut out_neg, d_neg.ptr).unwrap();

        for i in 0..n {
            let mut exp_mul = blst_fp::default();
            let mut exp_add = blst_fp::default();
            let mut exp_sub = blst_fp::default();
            let mut exp_neg = blst_fp::default();
            unsafe {
                blst_fp_mul(&mut exp_mul, &a[i], &b[i]);
                blst_fp_add(&mut exp_add, &a[i], &b[i]);
                blst_fp_sub(&mut exp_sub, &a[i], &b[i]);
                blst_fp_cneg(&mut exp_neg, &a[i], true);
            }
            assert_eq!(out_mul[i].l, exp_mul.l, "fp_mul mismatch at i={i}");
            assert_eq!(out_add[i].l, exp_add.l, "fp_add mismatch at i={i}");
            assert_eq!(out_sub[i].l, exp_sub.l, "fp_sub mismatch at i={i}");
            assert_eq!(out_neg[i].l, exp_neg.l, "fp_neg mismatch at i={i}");
        }
    }

    #[test]
    fn gpu_cuda_msm_matches_blst_including_collisions() {
        let Some(rt) = gpu_runtime() else {
            eprintln!("skipping gpu_cuda MSM test: no CUDA GPU available");
            return;
        };
        eprintln!("Running CUDA MSM test on GPU: {}", rt.drv.device_name);

        let mut rng = StdRng::seed_from_u64(0xc0da_5070);
        for n in [4096usize, 16384] {
            let mut pts: Vec<G1Affine> = (0..n + 6)
                .map(|_| {
                    G1Affine::from(
                        G1Projective::generator() * BlsScalar::random(&mut rng),
                    )
                })
                .collect();
            // Inject repeated base & identity points to test doubling and
            // cancellation inside GPU XYZZ accumulation!
            pts[1] = pts[0];
            pts[2] = pts[0];
            pts[3] = G1Affine::identity();

            let blst_pts = g1_affines_to_blst(&pts);
            let gpu_bases = GpuCommitBases::upload(&blst_pts)
                .expect("GPU base upload should succeed");

            let mut p0: Vec<BlsScalar> =
                (0..n + 3).map(|_| BlsScalar::random(&mut rng)).collect();
            p0[1] = p0[0];
            p0[2] = -p0[0];
            let p1: Vec<BlsScalar> =
                (0..n).map(|_| BlsScalar::random(&mut rng)).collect();
            let p2: Vec<BlsScalar> = vec![BlsScalar::zero(); n];
            let mut p3: Vec<BlsScalar> =
                (0..n + 6).map(|_| BlsScalar::random(&mut rng)).collect();
            p3[0] = BlsScalar::one();
            p3[1] = BlsScalar::one();
            p3[2] = -BlsScalar::one();

            let expected = msm_g1_batch(&blst_pts, [&p0, &p1, &p2, &p3]);
            let got = gpu_bases
                .msm_batch([&p0[..], &p1[..], &p2[..], &p3[..]])
                .expect("GPU batched MSM should succeed");
            assert_eq!(got, expected, "GPU batched MSM mismatch at n={n}");
        }
    }

    #[test]
    fn gpu_cuda_ntt_matches_cpu_fast_fr() {
        if gpu_runtime().is_none() {
            eprintln!("skipping gpu_cuda NTT test: no CUDA GPU available");
            return;
        }
        use crate::fast_fr::{
            bls_slice_to_fast, coset_n_fft_fast, serial_ifft_fast,
        };
        use crate::fft::EvaluationDomain;

        let n = 8192usize;
        let domain = EvaluationDomain::new(n).unwrap();
        let qdomain = EvaluationDomain::new(8 * n).unwrap();
        let log_n = domain.log_size_of_group;
        let fft_roots = bls_slice_to_fast(&domain.precompute_fft_roots());
        let ifft_roots = bls_slice_to_fast(&domain.precompute_ifft_roots());
        let size_inv = FastFr::from_bls(&domain.size_inv);

        let mut cp_vecs: Vec<Vec<FastFr>> = Vec::with_capacity(8);
        let mut ck_pown = [FastFr::ONE; 8];
        for (k, ck_slot) in ck_pown.iter_mut().enumerate() {
            let omega_k = qdomain.group_gen.pow_vartime(&[k as u64, 0, 0, 0]);
            let c_k = dusk_bls12_381::GENERATOR * omega_k;
            *ck_slot = FastFr::from_bls(&c_k.pow_vartime(&[n as u64, 0, 0, 0]));
            let mut cp = Vec::with_capacity(n);
            let mut acc = BlsScalar::one();
            for _ in 0..n {
                cp.push(FastFr::from_bls(&acc));
                acc *= c_k;
            }
            cp_vecs.push(cp);
        }
        let cp_refs: [&[FastFr]; 8] =
            core::array::from_fn(|k| cp_vecs[k].as_slice());

        let tables = GpuNttTables::upload(
            n,
            log_n,
            &fft_roots,
            &ifft_roots,
            size_inv,
            cp_refs,
            ck_pown,
            [FastFr::ONE; 8],
            [None; 17],
        )
        .expect("GPU NTT table upload should succeed");

        let mut rng = StdRng::seed_from_u64(0x5070_a770);
        let p0: Vec<BlsScalar> =
            (0..n + 6).map(|_| BlsScalar::random(&mut rng)).collect();
        let p1: Vec<BlsScalar> =
            (0..n + 2).map(|_| BlsScalar::random(&mut rng)).collect();
        let p2: Vec<BlsScalar> =
            (0..n).map(|_| BlsScalar::random(&mut rng)).collect();
        let p3: Vec<BlsScalar> =
            (0..n / 2).map(|_| BlsScalar::random(&mut rng)).collect();
        let p4: Vec<BlsScalar> = vec![BlsScalar::zero(); n];
        let src: [&[BlsScalar]; 5] = [&p0, &p1, &p2, &p3, &p4];

        for wave in 0..2usize {
            let k_base = wave * 4;
            let got = tables
                .coset_ffts_wave(k_base, &src)
                .expect("GPU coset_ffts_wave should succeed");
            for local_k in 0..4 {
                for (p_idx, poly) in src.iter().enumerate() {
                    let expected = coset_n_fft_fast(
                        poly,
                        n,
                        log_n,
                        ck_pown[k_base + local_k],
                        &cp_vecs[k_base + local_k],
                        &fft_roots,
                    );
                    assert_eq!(
                        got[local_k * src.len() + p_idx],
                        expected,
                        "GPU coset FFT mismatch at wave={wave} k={local_k} p={p_idx}"
                    );
                }
            }
        }

        let evals: Vec<Vec<FastFr>> = (0..8)
            .map(|_| {
                (0..n)
                    .map(|_| FastFr::from_bls(&BlsScalar::random(&mut rng)))
                    .collect()
            })
            .collect();
        let got_ifft = tables
            .batch_ifft(&evals)
            .expect("GPU batch_ifft should succeed");
        for k in 0..8 {
            let expected =
                serial_ifft_fast(&evals[k], &ifft_roots, size_inv, log_n);
            assert_eq!(
                got_ifft[k], expected,
                "GPU batch IFFT mismatch at k={k}"
            );
        }
    }

    #[test]
    fn gpu_cuda_full_quotient_matches_cpu_all_widgets() {
        if gpu_runtime().is_none() {
            eprintln!("skipping gpu_cuda quotient test: no CUDA GPU available");
            return;
        }
        use crate::composer::{Composer, Constraint, TorsionFreeWitnessPoint};
        use crate::fast_quotient::FastProverCache;
        use crate::fft::{EvaluationDomain, Polynomial};
        use crate::prelude::{Circuit, Compiler, Error, PublicParameters};
        use dusk_jubjub::JubJubScalar;

        #[derive(Default)]
        struct AllWidgetsCircuit;
        impl Circuit for AllWidgetsCircuit {
            fn circuit(&self, composer: &mut Composer) -> Result<(), Error> {
                let a = composer.append_witness(BlsScalar::from(2u64));
                let b = composer.append_witness(BlsScalar::from(3u64));
                let x = composer.append_witness(BlsScalar::from(6u64));
                let y = composer.append_witness(JubJubScalar::from(7u64));
                let pt =
                    dusk_jubjub::GENERATOR_EXTENDED * JubJubScalar::from(7u64);
                let z = composer.append_point(pt)?;
                let tf_z = TorsionFreeWitnessPoint::new_unchecked(z);
                composer.append_public(42u64);
                while composer.constraints() < 2500 {
                    let r =
                        composer.gate_mul(Constraint::new().mult(1).a(a).b(b));
                    composer.assert_equal(x, r);
                    composer.append_logic_and::<63>(a, b);
                    composer.append_logic_xor::<63>(a, b);
                    composer.component_range_bits::<64>(a);
                    composer.component_mul_generator(
                        y,
                        dusk_jubjub::GENERATOR_EXTENDED,
                    )?;
                    composer.component_mul_point(y, tf_z);
                }
                Ok(())
            }
        }

        let mut rng = StdRng::seed_from_u64(0x9070_cafe);
        let pp = PublicParameters::setup(1 << 13, &mut rng).unwrap();
        let (prover, verifier) =
            Compiler::compile::<AllWidgetsCircuit>(&pp, b"gpu-quot-test")
                .unwrap();

        // 1) End-to-end prove + verify exercises all 6 widgets + PI on GPU.
        let (proof, pi) = prover
            .prove(&mut StdRng::seed_from_u64(7), &AllWidgetsCircuit)
            .unwrap();
        verifier.verify(&proof, &pi).unwrap();

        // 2) Direct bit-for-bit GPU vs CPU comparison for both `beta != 0`
        // (factored path) and `beta == 0` (unfactored fallback), with and
        // without public inputs.
        let domain = EvaluationDomain::new(prover.constraints).unwrap();
        let qdomain = EvaluationDomain::new(8 * domain.size()).unwrap();
        let n = domain.size();
        assert!(n >= GPU_NTT_MIN_LEN);

        let v_h = &prover.prover_key.v_h_coset_8n().evals;
        let van_inv: [BlsScalar; 8] =
            core::array::from_fn(|k| v_h[k].invert().unwrap());
        let cache = FastProverCache::new(
            &domain,
            &qdomain,
            &prover.prover_key,
            &van_inv,
        );

        let rand_poly = |len: usize, r: &mut StdRng| {
            Polynomial::from_coefficients_vec(
                (0..len).map(|_| BlsScalar::random(&mut *r)).collect(),
            )
        };
        let z_p = rand_poly(n + 3, &mut rng);
        let a_p = rand_poly(n + 2, &mut rng);
        let b_p = rand_poly(n + 2, &mut rng);
        let c_p = rand_poly(n + 2, &mut rng);
        let d_p = rand_poly(n + 2, &mut rng);
        let pi_p = rand_poly(n, &mut rng);
        let zero_pi = Polynomial::zero();

        for (beta, pi_ref) in [
            (BlsScalar::random(&mut rng), &pi_p),
            (BlsScalar::zero(), &zero_pi),
        ] {
            let ch = (
                BlsScalar::random(&mut rng),
                beta,
                BlsScalar::random(&mut rng),
                BlsScalar::random(&mut rng),
                BlsScalar::random(&mut rng),
                BlsScalar::random(&mut rng),
                BlsScalar::random(&mut rng),
            );
            let gpu_res = cache.compute_split_quotient_raw_iffts_gpu(
                &domain,
                &z_p,
                (&a_p, &b_p, &c_p, &d_p),
                pi_ref,
                &ch,
            );
            let cpu_res = cache.compute_split_quotient_raw_iffts_cpu(
                &domain,
                &z_p,
                (&a_p, &b_p, &c_p, &d_p),
                pi_ref,
                &ch,
            );
            assert_eq!(
                gpu_res,
                cpu_res,
                "GPU full quotient IFFTs mismatch CPU (beta_zero={})",
                beta == BlsScalar::zero()
            );
        }
    }
}
