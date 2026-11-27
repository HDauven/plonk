# `fast-plonk` Benchmarks & Architecture Notes

All measurements below compare upstream `dusk-network/plonk` `master` (`750d38d`) against `fast-plonk` (`HDauven/plonk`, branch `fast-plonk`) on identical Dusk PLONK V3 circuits and bit-for-bit identical proofs.

## Test Environment

- **CPU**: Intel Core Ultra 9 275HX (22 cores: 8 P-cores + 14 E-cores, x86_64 with ADX + BMI2)
- **GPU**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (Blackwell SM 12.0, 12 GB GDDR7, Driver `591.86`, optional `--features gpu-cuda`)
- **Profile**: `--profile bench` (`opt-level = 3`, `lto = true`, `codegen-units = 1`, `debug = false`)
- **Circuit**: `BenchmarkCircuit` (JubJub scalar multiplication + arithmetic/range/logic gates, 32 public inputs, sizes `2^12` through `2^16`)
- **Command**:
  ```bash
  cargo run --profile bench --example profile_plonk
  # With native CUDA GPU backend:
  cargo run --profile bench --features gpu-cuda --example profile_plonk
  # With per-stage prover breakdown:
  PLONK_STAGE_PROFILE=1 cargo run --profile bench --features gpu-cuda --example profile_plonk
  ```

---

## 1. End-to-End Proving Latency

| Circuit Size | Gates | `master` (`750d38d`) | Phase 1 (`blst` XYZZ) | Phase 2 (CPU Batched-Affine) | Phase 3 (`gpu-cuda`, RTX 5070 Ti) | GPU Speedup vs `master` |
|---|---:|---:|---:|---:|---:|---:|
| `2^12` | 4,096 | `161.53 ms` | `50.58 ms` | `44.36 ms` *(min `41.24 ms`)* | **`18.02 ms`** *(min `16.26 ms`)* | **8.96×** *(9.93× min)* |
| `2^13` | 8,192 | `326.13 ms` | `76.31 ms` | `72.90 ms` *(min `67.66 ms`)* | **`24.33 ms`** *(min `22.80 ms`)* | **13.40×** *(14.30× min)* |
| `2^14` | 16,384 | `647.60 ms` | `144.16 ms` | `120.12 ms` *(min `119.22 ms`)* | **`33.93 ms`** *(min `33.14 ms`)* | **19.09×** *(19.54× min)* |
| `2^15` | 32,768 | `1,074.70 ms` | `253.42 ms` | `231.37 ms` *(min `218.48 ms`)* | **`59.67 ms`** *(min `55.85 ms`)* | **18.01×** *(19.24× min)* |
| `2^16` | 65,536 | `2,073.60 ms` | `476.86 ms` | `440.97 ms` *(min `418.90 ms`)* | **`90.63 ms`** *(min `90.58 ms`)* | **22.88×** *(22.89× min)* |

### Stage-by-Stage Prover Breakdown at `2^16` (65,536 gates)

Captured via `PLONK_STAGE_PROFILE=1`:

| Stage | CPU `fast-plonk` (~`432 ms`) | GPU `gpu-cuda` (~`91 ms`) | GPU vs CPU | Primary Acceleration |
|---|---:|---:|---:|---|
| `composer` | `12.3 ms` | `12.8 ms` | `1.0×` | CPU batched JubJub inversions & zero/unit selector fast paths |
| `wire_ifft + wire_msm` (4×) | `31.0 ms` (`11.3 + 19.7`) | **`18.9 ms`** | **1.6×** | Fused GPU 4× IFFT + on-GPU-scan 4× Pippenger MSM (overlapped with CPU `FastFr -> BlsScalar`; skips 3.25 s CPU Lagrange G1 IFFT) |
| `perm + z_ifft + z_msm` (1×) | `53.4 ms` (`13.9 + 39.5`) | **`11.8 ms`** | **4.5×** | `FastFr` ADX 2-pass parallel prefix-product + fused GPU `z` IFFT + GPU MSM overlapped with CPU conversion |
| `quot + quot_msm` (8-coset + 4×) | `247.9 ms` (`121.2 + 126.7`) | **`26.2 ms`** | **9.5×** | GPU `pi` IFFT + full-VRAM 40× coset NTT + `quotient_eval_wave` (all 6 widgets + PI + perm) + 8× IFFT + `quotient_radix8_combine` + fused 4× GPU MSM overlapped with CPU conversion |
| `r4 + r5` (lin + 2× MSM) | `84.4 ms` | **`18.5 ms`** | **4.6×** | Fused `r(X) + W_z(X)` + 2-pass parallel block Ruffini directly in `FastFr` (zero `to_bls` round-trips) + 2× GPU MSM |
| **Total** | **`~432 ms`** | **`~90.6 ms`** | **~4.9×** | **22.9× vs `master` (`2,074 ms`)** |

---

## 2. Standalone MSM: `blst` XYZZ vs Batched-Affine Pippenger

Measured head-to-head on random BLS12-381 G1 points and 255-bit scalars (`1×` single polynomial and `4×` batched polynomials, matching `z_poly` and `t_0..t_3` commitments):

| Size | 1× `blst` XYZZ | 1× Batched-Affine | 1× Speedup | 4× `blst` XYZZ | 4× Batched-Affine | 4× Speedup |
|---|---:|---:|---:|---:|---:|---:|
| `2^12` | `7.03 ms` | **`6.58 ms`** | **1.07×** | `21.45 ms` | **`16.83 ms`** (`w=10`) | **1.27×** |
| `2^13` | `9.56 ms` | **`9.10 ms`** | **1.05×** | `33.47 ms` | **`33.11 ms`** (`w=11`) | **1.01×** |
| `2^14` | `16.91 ms` | **`15.11 ms`** | **1.12×** | `56.64 ms` | **`50.21 ms`** (`w=12`) | **1.13×** |
| `2^15` | `29.54 ms` | **`24.19 ms`** | **1.22×** | `98.10 ms` | **`87.47 ms`** (`w=12`) | **1.12×** |
| `2^16` | `57.04 ms` | **`46.19 ms`** | **1.23×** | `207.95 ms` | **`142.06 ms`** (`w=13`) | **1.46×** |

### Why Batched-Affine Beats `blst` XYZZ Here
1. **Lower field-op cost per bucket add**: Montgomery batching (`1` `blst_fp_eucl_inverse` amortized across up to `256` distinct buckets) costs `~5.4M + 1S` per addition vs `blst`'s XYZZ mixed addition (`7M + 2S`).
2. **Half the bucket footprint**: Affine buckets occupy `96 B` vs `192 B` for XYZZ, allowing `w = 13` (`4,096` buckets = `384 KiB`) to remain L2-resident across P- and E-cores.
3. **Edge-case guards**:
   - Narrow top partial windows (`eff_bits < 10`, e.g., `256 % 12 = 4`) collapse batch sizes and stall on serial inversions, so they are automatically dispatched to `blst_p1s_tile_pippenger`.
   - Repeated/skewed witness scalars (common in Lagrange wire commitments) spill into a zero-inversion Jacobian overflow accumulator whenever the deferred queue narrows below `32` distinct buckets.

---

## 3. Verification & Proof Deserialization

Verification cost is $O(1)$ in circuit size (measured with 32 public inputs):

| Operation | `master` | Phase 1 (`blst` `wbits`) | Phase 2 (Comb + GLV + Pipelined) | Speedup vs `master` |
|---|---:|---:|---:|---:|
| **`Proof::from_bytes`** | `~870 µs` | `224.0 µs` | **`207.7 µs`** *(min `196.5 µs`)* | **4.2×** |
| **Verify single (32 PI)** | `~2,550 µs` | `887.5 µs` idle / `1,016 µs` loaded | **`942.3 µs`** *(~800 µs idle)* | **2.7× – 3.2×** |
| **Verify batch 16 (32 PI)** | `~40.8 ms` *(16×)* | `2.52 ms` (`157.5 µs/p`) | **`2.34 ms`** (**`146.2 µs/p`**) | **17.4×** |
| **Verify batch 32 (32 PI)** | `~81.6 ms` *(32×)* | `3.54 ms` (`110.6 µs/p`) | **`2.86 ms`** (**`89.3 µs/p`**) | **28.6×** |
| **Verify batch 64 (32 PI)** | `~163.2 ms` *(64×)* | `4.66 ms` (`72.8 µs/p`) | **`4.01 ms`** (**`62.6 µs/p`**) | **40.7×** |

### Single-Verify Component Breakdown (`blst` backend)

| Component | Phase 1 | Phase 2 | Notes |
|---|---:|---:|---|
| Static 16-pt MSM | `265–321 µs` (`blst_p1s_mult_wbits`) | **`~218 µs`** (serial) | 4-tooth × 2-group × 8-pt variable-time Comb (`192 KiB`), no constant-time linear table scans |
| Dynamic 11-pt + `left` MSM | `~260–420 µs` | **`~349 µs`** (2-way par with static) | 128-bit GLV Straus (`w = 5`), zero constant-time `wbits` fallback |
| 2-pair Pairing (`ml1 + ml2 + finalverify`) | `~470–555 µs` | **`~355 µs` exposed** | `ml1(w_z + u·w_zw, x·H)` (~115 µs) is pipelined concurrently with the right MSM; serial `ml2 + finalverify` avoids sub-ms Rayon wakeup jitter |

---

## 4. Feature & Compatibility Matrix

Every commit on `fast-plonk` is verified across all supported feature configurations:

```bash
# Default (std + blst-backend) + all optional features
cargo clippy --all-features --features rkyv/size_32 --all-targets -- -D warnings
cargo test --release
cargo test --release --all-features

# Pure-Rust multi-threaded fallback (no blst C/assembly)
cargo clippy --no-default-features --features std,legacy-proving --lib --examples --benches -- -D warnings
cargo test --release --no-default-features --features std --lib --test composer --test batch_verification

# no_std + alloc and bare no_std
cargo clippy --no-default-features --features alloc -- -D warnings
cargo clippy --no-default-features -- -D warnings
cargo test --release --no-default-features --features alloc --test composer --test batch_verification

# Docs, benchmarks, and examples
cargo rustdoc --lib -- --html-in-header katex-header.html -D warnings
cargo rustdoc --lib -- --document-private-items -D warnings
cargo bench --no-run
cargo run --release --example circuit
```
