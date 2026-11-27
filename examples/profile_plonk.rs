// Stage-by-stage performance profiler for fast-plonk proving and verification.

use std::hint::black_box;
use std::time::Instant;

use dusk_bytes::Serializable;
use dusk_plonk::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Debug, Clone, Copy)]
struct BenchCircuit<const DEGREE: usize> {
    a: BlsScalar,
    b: BlsScalar,
    x: BlsScalar,
    y: JubJubScalar,
    z: JubJubExtended,
}

impl<const DEGREE: usize> Default for BenchCircuit<DEGREE> {
    fn default() -> Self {
        Self {
            a: BlsScalar::from(2u64),
            b: BlsScalar::from(3u64),
            x: BlsScalar::from(6u64),
            y: JubJubScalar::from(7u64),
            z: dusk_jubjub::GENERATOR_EXTENDED * JubJubScalar::from(7u64),
        }
    }
}

impl<const DEGREE: usize> Circuit for BenchCircuit<DEGREE> {
    fn circuit(&self, composer: &mut Composer) -> Result<(), Error> {
        let w_a = composer.append_witness(self.a);
        let w_b = composer.append_witness(self.b);
        let w_x = composer.append_witness(self.x);
        let w_y = composer.append_witness(self.y);
        let w_z = composer.append_point(self.z)?;

        let mut diff = 0;
        let mut prev = composer.constraints();

        while prev + diff < DEGREE {
            let r_w =
                composer.gate_mul(Constraint::new().mult(1).a(w_a).b(w_b));

            composer.append_constant(15);
            composer.append_constant_point(self.z)?;

            composer.assert_equal(w_x, r_w);
            composer.assert_equal_point(w_z, w_z);

            composer.gate_add(Constraint::new().left(1).right(1).a(w_a).b(w_b));

            let tf_z = TorsionFreeWitnessPoint::new_unchecked(w_z);
            composer.component_add_point(tf_z, tf_z);
            composer.append_logic_and::<127>(w_a, w_b);
            composer.append_logic_xor::<127>(w_a, w_b);
            composer.component_boolean(Composer::ONE);
            composer.component_decomposition::<254>(w_a);
            composer.component_mul_generator(
                w_y,
                dusk_jubjub::GENERATOR_EXTENDED,
            )?;
            composer.component_mul_point(w_y, tf_z);
            composer.component_range_bits::<256>(w_a);
            composer.component_select(Composer::ONE, w_a, w_b);
            composer.component_select_identity(Composer::ONE, tf_z);
            composer.component_select_one(Composer::ONE, w_a);
            composer.component_select_point(Composer::ONE, w_z, w_z);
            composer.component_select_zero(Composer::ONE, w_a);

            diff = composer.constraints() - prev;
            prev = composer.constraints();
        }

        Ok(())
    }
}

#[derive(Default)]
struct VerificationCircuit;

impl Circuit for VerificationCircuit {
    fn circuit(&self, composer: &mut Composer) -> Result<(), Error> {
        for input in 1..=32u64 {
            composer.append_public(input);
        }
        Ok(())
    }
}

fn median(mut times: Vec<f64>) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn bench_prove<const DEGREE: usize>(pp: &PublicParameters, iters: usize) {
    let label = b"dusk-network";
    let (prover, verifier) =
        Compiler::compile::<BenchCircuit<DEGREE>>(pp, label)
            .expect("compile failed");
    let circuit = BenchCircuit::<DEGREE>::default();

    // Warmup + correctness check
    let mut rng = StdRng::seed_from_u64(0x1234_5678);
    let (proof, pi) = prover.prove(&mut rng, &circuit).expect("prove failed");
    verifier.verify(&proof, &pi).expect("verify failed");

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let res = black_box(prover.prove(&mut rng, &circuit)).unwrap();
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
        black_box(res);
    }
    let pow = DEGREE.trailing_zeros();
    println!(
        "Prove 2^{pow:2} ({DEGREE:6} gates): median = {:8.2} ms  (min = {:8.2} ms)",
        median(samples.clone()),
        samples.iter().copied().fold(f64::INFINITY, f64::min),
    );
}

fn main() {
    println!(
        "Threads: {} (RAYON_NUM_THREADS={:?})",
        rayon::current_num_threads(),
        std::env::var("RAYON_NUM_THREADS").ok()
    );

    let pp = PublicParameters::setup(1 << 16, &mut StdRng::seed_from_u64(42))
        .expect("setup failed");

    bench_prove::<{ 1 << 12 }>(&pp, 7);
    bench_prove::<{ 1 << 13 }>(&pp, 7);
    bench_prove::<{ 1 << 14 }>(&pp, 5);
    bench_prove::<{ 1 << 15 }>(&pp, 5);
    bench_prove::<{ 1 << 16 }>(&pp, 3);

    // Verification benchmarks (32 public inputs, matching benches/verification.rs)
    let pp_v = PublicParameters::setup(1 << 6, &mut StdRng::seed_from_u64(99))
        .expect("setup failed");
    let (prover_v, verifier_v) =
        Compiler::compile::<VerificationCircuit>(&pp_v, b"dusk-network")
            .expect("compile failed");
    let mut rng = StdRng::seed_from_u64(777);
    let pairs: Vec<(Proof, Vec<BlsScalar>)> = (0..64)
        .map(|_| {
            prover_v
                .prove(&mut rng, &VerificationCircuit)
                .expect("prove failed")
        })
        .collect();
    let proof_bytes = pairs[0].0.to_bytes();

    // Warmup
    for _ in 0..10 {
        verifier_v.verify(&pairs[0].0, &pairs[0].1).unwrap();
        let _ = Proof::from_bytes(&proof_bytes).unwrap();
    }

    let mut deser_samples = Vec::new();
    for _ in 0..9 {
        let t0 = Instant::now();
        for _ in 0..50 {
            let p = Proof::from_bytes(black_box(&proof_bytes)).unwrap();
            black_box(p);
        }
        deser_samples.push(t0.elapsed().as_secs_f64() * 1e6 / 50.0);
    }
    println!(
        "Proof::from_bytes:            median = {:8.1} us",
        median(deser_samples)
    );

    let mut ver_samples = Vec::new();
    for _ in 0..9 {
        let t0 = Instant::now();
        for _ in 0..50 {
            verifier_v
                .verify(black_box(&pairs[0].0), black_box(&pairs[0].1))
                .unwrap();
        }
        ver_samples.push(t0.elapsed().as_secs_f64() * 1e6 / 50.0);
    }
    println!(
        "Verify single (32 PI):        median = {:8.1} us",
        median(ver_samples)
    );

    for k in [16usize, 32, 64, 256, 1024, 4096] {
        let items: Vec<(&Proof, &[BlsScalar])> = (0..k)
            .map(|i| {
                let (p, pi) = &pairs[i % pairs.len()];
                (p, pi.as_slice())
            })
            .collect();
        verifier_v.verify_batch(&items).unwrap();

        let inner_iters = if k <= 64 {
            10
        } else if k <= 256 {
            5
        } else {
            3
        };
        let outer_iters = if k <= 1024 { 7 } else { 5 };
        let mut samples = Vec::new();
        for _ in 0..outer_iters {
            let t0 = Instant::now();
            for _ in 0..inner_iters {
                verifier_v.verify_batch(black_box(&items)).unwrap();
            }
            samples.push(t0.elapsed().as_secs_f64() * 1e3 / inner_iters as f64);
        }
        let med = median(samples);
        println!(
            "Verify batch {k:4} (32 PI):    median = {:8.2} ms  ({:6.1} us/proof)",
            med,
            med * 1e3 / k as f64
        );
    }

    for k in [64usize, 256, 1024, 4096] {
        let batch_bytes: Vec<[u8; Proof::SIZE]> = (0..k)
            .map(|i| pairs[i % pairs.len()].0.to_bytes())
            .collect();
        let _ = Proof::from_bytes_batch(&batch_bytes).unwrap();

        let inner_iters = if k <= 256 { 5 } else { 3 };
        let mut samples = Vec::new();
        for _ in 0..7 {
            let t0 = Instant::now();
            for _ in 0..inner_iters {
                let decoded =
                    Proof::from_bytes_batch(black_box(&batch_bytes)).unwrap();
                black_box(decoded);
            }
            samples.push(t0.elapsed().as_secs_f64() * 1e3 / inner_iters as f64);
        }
        let med = median(samples);
        println!(
            "from_bytes_batch {k:4}:        median = {:8.2} ms  ({:6.1} us/proof)",
            med,
            med * 1e3 / k as f64
        );
    }
}
