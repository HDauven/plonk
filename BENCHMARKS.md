# `fast-plonk` Benchmarks & Architecture Notes

All measurements below compare upstream `dusk-network/plonk` `master` (`750d38d`) against `fast-plonk` (`HDauven/plonk`, branch `fast-plonk`) on identical Dusk PLONK V3 circuits and bit-for-bit identical proofs.

## Test Environment

- **CPU**: Intel Core Ultra 9 275HX (22 cores: 8 P-cores + 14 E-cores, x86_64 with ADX + BMI2)
- **Profile**: `--profile bench` (`opt-level = 3`, `lto = true`, `codegen-units = 1`, `debug = false`)
- **Circuit**: `BenchmarkCircuit` (JubJub scalar multiplication + arithmetic/range/logic gates, 32 public inputs, sizes `2^12` through `2^16`)
- **Command**:
  ```bash
  cargo run --profile bench --example profile_plonk
  # With per-stage prover breakdown:
  PLONK_STAGE_PROFILE=1 cargo run --profile bench --example profile_plonk
  ```

---

## 1. End-to-End Proving Latency

| Circuit Size | Gates | `master` (`750d38d`) | Phase 1 (`blst` XYZZ) | Phase 2 (`fast-plonk`, Batched-Affine) | Speedup vs `master` |
|---|---:|---:|---:|---:|---:|
| `2^12` | 4,096 | `161.53 ms` | `50.58 ms` | **`44.36 ms`** *(min `41.24 ms`)* | **3.64×** *(3.92× min)* |
| `2^13` | 8,192 | `326.13 ms` | `76.31 ms` | **`72.90 ms`** *(min `67.66 ms`)* | **4.47×** *(4.82× min)* |
| `2^14` | 16,384 | `647.60 ms` | `144.16 ms` | **`120.12 ms`** *(min `119.22 ms`)* | **5.39×** *(5.43× min)* |
| `2^15` | 32,768 | `1,074.70 ms` | `253.42 ms` | **`231.37 ms`** *(min `218.48 ms`)* | **4.64×** *(4.92× min)* |
| `2^16` | 65,536 | `2,073.60 ms` | `476.86 ms` | **`440.97 ms`** *(min `418.90 ms`)* | **4.70×** *(4.95× min)* |

### Stage-by-Stage Prover Breakdown at `2^16` (65,536 gates)

Captured via `PLONK_STAGE_PROFILE=1` (typical clean run ~`432 ms`):

| Stage | Time | Share | Primary Acceleration |
|---|---:|---:|---|
| `composer` | `12.3 ms` | 2.9% | Batched JubJub inversions (`504→1`, `256→1`), cached generator multiples, zero/unit selector fast paths |
| `wire_ifft` | `11.3 ms` | 2.6% | Radix-4 DIT IFFT with block parallelism |
| `wire_msm` | `19.7 ms` | 4.6% | Precomputed Lagrange SRS (`g1_ifft_blst`), zero skipping, `±1..=±15` direct folding, 16/64-bit buckets, skewed-bucket Jacobian overflow |
| `perm + z_ifft` | `13.9 ms` | 3.2% | Parallel prefix-product permutation vector + `blst_fr_eucl_inverse` batch inversion |
| `z_msm` | `39.5 ms` | 9.2% | Batched-Affine Pippenger (`w = 13`) |
| `quot` (8-coset) | `121.2 ms` | 28.1% | Fused 8-coset `FastFr` engine (`x86_64` `mulx`/`adcx`/`adox` assembly) |
| `quot_msm` (4×) | `126.7 ms` | 29.3% | Batched-Affine Pippenger (`w = 13`, 4 polynomials tiled flat across Rayon) |
| `r4 + r5` (2× MSM + lin) | `84.4 ms` | 19.5% | Fused linearization + Horner evaluation + Batched-Affine Pippenger |
| **Total** | **`~432 ms`** | **100%** | **4.8× vs `master` (`2,074 ms`)** |

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
