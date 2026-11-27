// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! Fast BLS12-381 scalar field (`Fr`) arithmetic and radix-4 FFT kernels using
//! inline `x86_64` ADX/BMI2 (`mulx`/`adcx`/`adox`) CIOS Montgomery
//! multiplication.
//!
//! Representation matches `dusk_bls12_381::BlsScalar` bit-for-bit: four
//! little-endian `u64` limbs in Montgomery form (`a * 2^256 mod r`), canonical
//! in `[0, r)`.

use alloc::vec::Vec;

use dusk_bls12_381::BlsScalar;

/// BLS12-381 scalar field modulus `r`:
/// `0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001`.
pub(crate) const MODULUS: [u64; 4] = [
    0xffff_ffff_0000_0001,
    0x53bd_a402_fffe_5bfe,
    0x3339_d808_09a1_d805,
    0x73ed_a753_299d_7d48,
];

/// `-r^{-1} mod 2^64`.
const INV: u64 = 0xffff_fffe_ffff_ffff;

/// `1` in Montgomery form (`R = 2^256 mod r`).
pub(crate) const ONE_MONT: [u64; 4] = [
    0x0000_0001_ffff_fffe,
    0x5884_b7fa_0003_4802,
    0x998c_4fef_ecbc_4ff5,
    0x1824_b159_acc5_056f,
];

/// BLS12-381 scalar field element in Montgomery form, bit-identical to
/// `dusk_bls12_381::BlsScalar`.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
pub(crate) struct FastFr(pub(crate) [u64; 4]);

impl FastFr {
    pub(crate) const ZERO: Self = Self([0, 0, 0, 0]);
    pub(crate) const ONE: Self = Self(ONE_MONT);

    #[inline(always)]
    pub(crate) fn from_bls(s: &BlsScalar) -> Self {
        Self(*s.internal_repr())
    }

    #[inline(always)]
    pub(crate) fn is_zero(&self) -> bool {
        (self.0[0] | self.0[1] | self.0[2] | self.0[3]) == 0
    }

    /// Convert back to canonical `BlsScalar` safely via Montgomery reduction
    /// + `BlsScalar::from_raw`.
    #[inline]
    pub(crate) fn to_bls(self) -> BlsScalar {
        let canonical = self.mont_reduce_to_canonical();
        BlsScalar::from_raw(canonical)
    }

    /// Reduce `self = a*R mod r` to canonical integer `a mod r` (`[u64; 4]`).
    #[inline]
    pub(crate) fn mont_reduce_to_canonical(self) -> [u64; 4] {
        #[cfg(target_arch = "x86_64")]
        if has_adx_bmi2() {
            // SAFETY: runtime checked ADX + BMI2.
            return unsafe { mul_adx_inner(self, Self([1, 0, 0, 0])).0 };
        }
        mul_portable(self, Self([1, 0, 0, 0])).0
    }

    #[inline(always)]
    pub(crate) fn add_portable(self, rhs: Self) -> Self {
        let (d0, c) = adc(self.0[0], rhs.0[0], 0);
        let (d1, c) = adc(self.0[1], rhs.0[1], c);
        let (d2, c) = adc(self.0[2], rhs.0[2], c);
        let (d3, _) = adc(self.0[3], rhs.0[3], c);
        sub_modulus_if_ge([d0, d1, d2, d3])
    }

    #[inline(always)]
    pub(crate) fn sub_portable(self, rhs: Self) -> Self {
        let (d0, b) = sbb(self.0[0], rhs.0[0], 0);
        let (d1, b) = sbb(self.0[1], rhs.0[1], b);
        let (d2, b) = sbb(self.0[2], rhs.0[2], b);
        let (d3, b) = sbb(self.0[3], rhs.0[3], b);
        if b != 0 {
            let (d0, c) = adc(d0, MODULUS[0], 0);
            let (d1, c) = adc(d1, MODULUS[1], c);
            let (d2, c) = adc(d2, MODULUS[2], c);
            let (d3, _) = adc(d3, MODULUS[3], c);
            Self([d0, d1, d2, d3])
        } else {
            Self([d0, d1, d2, d3])
        }
    }

    #[inline(always)]
    pub(crate) fn neg_portable(self) -> Self {
        if self.is_zero() {
            Self::ZERO
        } else {
            let (d0, b) = sbb(MODULUS[0], self.0[0], 0);
            let (d1, b) = sbb(MODULUS[1], self.0[1], b);
            let (d2, b) = sbb(MODULUS[2], self.0[2], b);
            let (d3, _) = sbb(MODULUS[3], self.0[3], b);
            Self([d0, d1, d2, d3])
        }
    }
}

#[inline(always)]
fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let sum = (a as u128) + (b as u128) + (carry as u128);
    (sum as u64, (sum >> 64) as u64)
}

#[inline(always)]
fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let diff = (a as u128).wrapping_sub((b as u128) + (borrow as u128));
    (diff as u64, (diff >> 127) as u64)
}

#[inline(always)]
fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let prod = (a as u128) + (b as u128) * (c as u128) + (carry as u128);
    (prod as u64, (prod >> 64) as u64)
}

#[inline(always)]
fn sub_modulus_if_ge(d: [u64; 4]) -> FastFr {
    let (s0, b) = sbb(d[0], MODULUS[0], 0);
    let (s1, b) = sbb(d[1], MODULUS[1], b);
    let (s2, b) = sbb(d[2], MODULUS[2], b);
    let (s3, b) = sbb(d[3], MODULUS[3], b);
    if b == 0 {
        FastFr([s0, s1, s2, s3])
    } else {
        FastFr(d)
    }
}

/// Portable CIOS Montgomery multiplication (`a * b * R^{-1} mod r`).
#[inline]
pub(crate) fn mul_portable(a: FastFr, b: FastFr) -> FastFr {
    let mut t = [0u64; 4];
    for &bi in &b.0 {
        let (t0, c) = mac(t[0], a.0[0], bi, 0);
        let (t1, c) = mac(t[1], a.0[1], bi, c);
        let (t2, c) = mac(t[2], a.0[2], bi, c);
        let (t3, t4) = mac(t[3], a.0[3], bi, c);

        let m = t0.wrapping_mul(INV);
        let (_, c) = mac(t0, m, MODULUS[0], 0);
        let (t0, c) = mac(t1, m, MODULUS[1], c);
        let (t1, c) = mac(t2, m, MODULUS[2], c);
        let (t2, c) = mac(t3, m, MODULUS[3], c);
        let (t3, _) = adc(t4, 0, c);
        t = [t0, t1, t2, t3];
    }
    sub_modulus_if_ge(t)
}

/// Detect x86_64 ADX + BMI2 support once.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn has_adx_bmi2() -> bool {
    std::arch::is_x86_feature_detected!("adx")
        && std::arch::is_x86_feature_detected!("bmi2")
}

/// Inline x86_64 ADX + BMI2 CIOS Montgomery multiplication by reference,
/// avoiding 32-byte stack copies when operands reside in slices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "adx,bmi2")]
#[inline]
pub(crate) unsafe fn mul_adx_ref(a: &FastFr, b: &FastFr) -> FastFr {
    let mut r0: u64;
    let mut r1: u64;
    let mut r2: u64;
    let mut r3: u64;

    // SAFETY: Caller guarantees ADX + BMI2 CPU features. All pointers point to
    // live 4-`u64` arrays (`a.0`, `b.0`, `MODULUS`).
    unsafe {
        core::arch::asm!(
            // ---- Round 0: t = a * b[0] ----
            "mov rdx, [{b} + 0]",
            "mulx {t1}, {t0}, [{a} + 0]",
            "mulx {t2}, rax, [{a} + 8]",
            "add {t1}, rax",
            "mulx {t3}, rax, [{a} + 16]",
            "adc {t2}, rax",
            "mulx {t4}, rax, [{a} + 24]",
            "adc {t3}, rax",
            "adc {t4}, 0",

            // Reduce round 0: m = t0 * INV; t = (t + m * r) >> 64
            "mov rdx, {inv}",
            "imul rdx, {t0}",
            "xor eax, eax",
            "mulx rcx, rax, [{m} + 0]",
            "adcx {t0}, rax",
            "adox {t1}, rcx",
            "mulx rcx, rax, [{m} + 8]",
            "adcx {t1}, rax",
            "adox {t2}, rcx",
            "mulx rcx, rax, [{m} + 16]",
            "adcx {t2}, rax",
            "adox {t3}, rcx",
            "mulx rcx, rax, [{m} + 24]",
            "adcx {t3}, rax",
            "adox {t4}, rcx",
            "mov eax, 0",
            "adcx {t4}, rax",

            // ---- Round 1: t += a * b[1] ----
            "mov rdx, [{b} + 8]",
            "xor {t0:e}, {t0:e}",
            "mulx rcx, rax, [{a} + 0]",
            "adcx {t1}, rax",
            "adox {t2}, rcx",
            "mulx rcx, rax, [{a} + 8]",
            "adcx {t2}, rax",
            "adox {t3}, rcx",
            "mulx rcx, rax, [{a} + 16]",
            "adcx {t3}, rax",
            "adox {t4}, rcx",
            "mulx rcx, rax, [{a} + 24]",
            "adcx {t4}, rax",
            "adox {t0}, rcx",
            "mov eax, 0",
            "adcx {t0}, rax",

            // Reduce round 1
            "mov rdx, {inv}",
            "imul rdx, {t1}",
            "xor eax, eax",
            "mulx rcx, rax, [{m} + 0]",
            "adcx {t1}, rax",
            "adox {t2}, rcx",
            "mulx rcx, rax, [{m} + 8]",
            "adcx {t2}, rax",
            "adox {t3}, rcx",
            "mulx rcx, rax, [{m} + 16]",
            "adcx {t3}, rax",
            "adox {t4}, rcx",
            "mulx rcx, rax, [{m} + 24]",
            "adcx {t4}, rax",
            "adox {t0}, rcx",
            "mov eax, 0",
            "adcx {t0}, rax",

            // ---- Round 2: t += a * b[2] ----
            "mov rdx, [{b} + 16]",
            "xor {t1:e}, {t1:e}",
            "mulx rcx, rax, [{a} + 0]",
            "adcx {t2}, rax",
            "adox {t3}, rcx",
            "mulx rcx, rax, [{a} + 8]",
            "adcx {t3}, rax",
            "adox {t4}, rcx",
            "mulx rcx, rax, [{a} + 16]",
            "adcx {t4}, rax",
            "adox {t0}, rcx",
            "mulx rcx, rax, [{a} + 24]",
            "adcx {t0}, rax",
            "adox {t1}, rcx",
            "mov eax, 0",
            "adcx {t1}, rax",

            // Reduce round 2
            "mov rdx, {inv}",
            "imul rdx, {t2}",
            "xor eax, eax",
            "mulx rcx, rax, [{m} + 0]",
            "adcx {t2}, rax",
            "adox {t3}, rcx",
            "mulx rcx, rax, [{m} + 8]",
            "adcx {t3}, rax",
            "adox {t4}, rcx",
            "mulx rcx, rax, [{m} + 16]",
            "adcx {t4}, rax",
            "adox {t0}, rcx",
            "mulx rcx, rax, [{m} + 24]",
            "adcx {t0}, rax",
            "adox {t1}, rcx",
            "mov eax, 0",
            "adcx {t1}, rax",

            // ---- Round 3: t += a * b[3] ----
            "mov rdx, [{b} + 24]",
            "xor {t2:e}, {t2:e}",
            "mulx rcx, rax, [{a} + 0]",
            "adcx {t3}, rax",
            "adox {t4}, rcx",
            "mulx rcx, rax, [{a} + 8]",
            "adcx {t4}, rax",
            "adox {t0}, rcx",
            "mulx rcx, rax, [{a} + 16]",
            "adcx {t0}, rax",
            "adox {t1}, rcx",
            "mulx rcx, rax, [{a} + 24]",
            "adcx {t1}, rax",
            "adox {t2}, rcx",
            "mov eax, 0",
            "adcx {t2}, rax",

            // Reduce round 3
            "mov rdx, {inv}",
            "imul rdx, {t3}",
            "xor eax, eax",
            "mulx rcx, rax, [{m} + 0]",
            "adcx {t3}, rax",
            "adox {t4}, rcx",
            "mulx rcx, rax, [{m} + 8]",
            "adcx {t4}, rax",
            "adox {t0}, rcx",
            "mulx rcx, rax, [{m} + 16]",
            "adcx {t0}, rax",
            "adox {t1}, rcx",
            "mulx rcx, rax, [{m} + 24]",
            "adcx {t1}, rax",
            "adox {t2}, rcx",
            "mov eax, 0",
            "adcx {t2}, rax",

            // Result is now in (t4, t0, t1, t2) in [0, 2r).
            // Conditional subtract modulus r using rdx/rcx/rax/t3 as temps.
            "mov rax, {t4}",
            "mov rcx, {t0}",
            "mov rdx, {t1}",
            "mov {t3}, {t2}",
            "sub rax, [{m} + 0]",
            "sbb rcx, [{m} + 8]",
            "sbb rdx, [{m} + 16]",
            "sbb {t3}, [{m} + 24]",
            "cmovnc {t4}, rax",
            "cmovnc {t0}, rcx",
            "cmovnc {t1}, rdx",
            "cmovnc {t2}, {t3}",

            a = in(reg) a.0.as_ptr(),
            b = in(reg) b.0.as_ptr(),
            m = in(reg) MODULUS.as_ptr(),
            inv = in(reg) INV,
            t4 = out(reg) r0,
            t0 = out(reg) r1,
            t1 = out(reg) r2,
            t2 = out(reg) r3,
            t3 = out(reg) _,
            out("rax") _,
            out("rcx") _,
            out("rdx") _,
            options(pure, readonly, nostack),
        );
    }

    FastFr([r0, r1, r2, r3])
}

/// Inline x86_64 ADX + BMI2 CIOS Montgomery multiplication for BLS12-381 `Fr`.
///
/// Exploits `r < 2^255` (top bit of `r[3]` is 0), so no 5th-limb carry ever
/// overflows `u64`, and uses dual carry chains (`adcx` / `adox`) with `mulx`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "adx,bmi2")]
#[inline]
pub(crate) unsafe fn mul_adx_inner(a: FastFr, b: FastFr) -> FastFr {
    // SAFETY: Caller guarantees ADX + BMI2 CPU features.
    unsafe { mul_adx_ref(&a, &b) }
}

/// Register-only doubling `2 * a mod r` (zero stack spills).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "adx,bmi2")]
#[inline]
pub(crate) unsafe fn dbl_adx_inner(a: FastFr) -> FastFr {
    let mut r0 = a.0[0];
    let mut r1 = a.0[1];
    let mut r2 = a.0[2];
    let mut r3 = a.0[3];
    // SAFETY: pure register arithmetic reading `MODULUS`.
    unsafe {
        core::arch::asm!(
            "add {r0}, {r0}",
            "adc {r1}, {r1}",
            "adc {r2}, {r2}",
            "adc {r3}, {r3}",
            "mov {s0}, {r0}",
            "mov {s1}, {r1}",
            "mov {s2}, {r2}",
            "mov {s3}, {r3}",
            "sub {s0}, [{m} + 0]",
            "sbb {s1}, [{m} + 8]",
            "sbb {s2}, [{m} + 16]",
            "sbb {s3}, [{m} + 24]",
            "cmovnc {r0}, {s0}",
            "cmovnc {r1}, {s1}",
            "cmovnc {r2}, {s2}",
            "cmovnc {r3}, {s3}",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            s0 = out(reg) _,
            s1 = out(reg) _,
            s2 = out(reg) _,
            s3 = out(reg) _,
            m = in(reg) MODULUS.as_ptr(),
            options(pure, readonly, nostack),
        );
    }
    FastFr([r0, r1, r2, r3])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "adx,bmi2")]
#[inline]
pub(crate) unsafe fn add_adx_ref(a: FastFr, b: &FastFr) -> FastFr {
    let mut r0 = a.0[0];
    let mut r1 = a.0[1];
    let mut r2 = a.0[2];
    let mut r3 = a.0[3];
    // SAFETY: pure register arithmetic reading `b.0` and `MODULUS`.
    unsafe {
        core::arch::asm!(
            "add {r0}, [{b} + 0]",
            "adc {r1}, [{b} + 8]",
            "adc {r2}, [{b} + 16]",
            "adc {r3}, [{b} + 24]",
            "mov {s0}, {r0}",
            "mov {s1}, {r1}",
            "mov {s2}, {r2}",
            "mov {s3}, {r3}",
            "sub {s0}, [{m} + 0]",
            "sbb {s1}, [{m} + 8]",
            "sbb {s2}, [{m} + 16]",
            "sbb {s3}, [{m} + 24]",
            "cmovnc {r0}, {s0}",
            "cmovnc {r1}, {s1}",
            "cmovnc {r2}, {s2}",
            "cmovnc {r3}, {s3}",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            s0 = out(reg) _,
            s1 = out(reg) _,
            s2 = out(reg) _,
            s3 = out(reg) _,
            b = in(reg) b.0.as_ptr(),
            m = in(reg) MODULUS.as_ptr(),
            options(pure, readonly, nostack),
        );
    }
    FastFr([r0, r1, r2, r3])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "adx,bmi2")]
#[inline]
pub(crate) unsafe fn add_adx_inner(a: FastFr, b: FastFr) -> FastFr {
    unsafe { add_adx_ref(a, &b) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "adx,bmi2")]
#[inline]
pub(crate) unsafe fn sub_adx_ref(a: FastFr, b: &FastFr) -> FastFr {
    let mut r0 = a.0[0];
    let mut r1 = a.0[1];
    let mut r2 = a.0[2];
    let mut r3 = a.0[3];
    // SAFETY: pure register arithmetic reading `b.0` and `MODULUS`.
    unsafe {
        core::arch::asm!(
            "sub {r0}, [{b} + 0]",
            "sbb {r1}, [{b} + 8]",
            "sbb {r2}, [{b} + 16]",
            "sbb {r3}, [{b} + 24]",
            "mov {s0}, 0",
            "mov {s1}, 0",
            "mov {s2}, 0",
            "mov {s3}, 0",
            "cmovc {s0}, [{m} + 0]",
            "cmovc {s1}, [{m} + 8]",
            "cmovc {s2}, [{m} + 16]",
            "cmovc {s3}, [{m} + 24]",
            "add {r0}, {s0}",
            "adc {r1}, {s1}",
            "adc {r2}, {s2}",
            "adc {r3}, {s3}",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            s0 = out(reg) _,
            s1 = out(reg) _,
            s2 = out(reg) _,
            s3 = out(reg) _,
            b = in(reg) b.0.as_ptr(),
            m = in(reg) MODULUS.as_ptr(),
            options(pure, readonly, nostack),
        );
    }
    FastFr([r0, r1, r2, r3])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "adx,bmi2")]
#[inline]
pub(crate) unsafe fn sub_adx_inner(a: FastFr, b: FastFr) -> FastFr {
    unsafe { sub_adx_ref(a, &b) }
}

/// Bit-reverse `n` within `log_n` bits.
#[inline(always)]
pub(crate) fn bitrev(n: u32, log_n: u32) -> usize {
    if log_n == 0 {
        0
    } else {
        (n.reverse_bits() >> (32 - log_n)) as usize
    }
}

/// Convert a slice of `BlsScalar`s into `Vec<FastFr>` (zero-cost limb copy).
pub(crate) fn bls_slice_to_fast(slice: &[BlsScalar]) -> Vec<FastFr> {
    slice.iter().map(FastFr::from_bls).collect()
}

/// Serial in-place radix-4 DIT FFT over a bit-reversed `FastFr` slice using
/// precomputed `n/2` forward or inverse roots of unity (`roots[k] = omega^k`).
///
/// Under `x86_64` with ADX+BMI2, runs the entire FFT inside a single
/// `#[target_feature(enable = "adx,bmi2")]` kernel so every Montgomery
/// multiply and butterfly inlines with zero call overhead.
pub(crate) fn serial_fft_bitreversed_fast(
    a: &mut [FastFr],
    roots: &[FastFr],
    log_n: u32,
) {
    #[cfg(target_arch = "x86_64")]
    if has_adx_bmi2() {
        // SAFETY: runtime checked ADX + BMI2.
        unsafe { serial_fft_bitreversed_adx(a, roots, log_n) };
        return;
    }
    serial_fft_bitreversed_portable(a, roots, log_n);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "adx,bmi2")]
unsafe fn serial_fft_bitreversed_adx(
    a: &mut [FastFr],
    roots: &[FastFr],
    log_n: u32,
) {
    let n = a.len();
    debug_assert_eq!(n, 1usize << log_n);
    if log_n == 0 {
        return;
    }

    let mut stage = 0u32;
    let mut m = 1usize;

    // Opening radix-2 pass when `log_n` is odd so remaining stages are radix-4.
    if (log_n & 1) == 1 {
        let mut i = 0;
        while i < n {
            let u = a[i];
            let v = a[i + 1];
            a[i] = unsafe { add_adx_inner(u, v) };
            a[i + 1] = unsafe { sub_adx_inner(u, v) };
            i += 2;
        }
        stage = 1;
        m = 2;
    }

    // Quarter-turn root `omega^(n/4)`.
    let quarter = if n >= 4 { roots[n / 4] } else { FastFr::ONE };

    // Radix-4 DIT stages.
    while stage < log_n {
        let step = m << 2;
        let stride = n / step;

        if m == 1 {
            let mut i = 0;
            while i < n {
                let x0 = a[i];
                let x1 = a[i + 1];
                let x2 = a[i + 2];
                let x3 = a[i + 3];

                let t0 = unsafe { add_adx_inner(x0, x1) };
                let t1 = unsafe { sub_adx_inner(x0, x1) };
                let t2 = unsafe { add_adx_inner(x2, x3) };
                let t3 =
                    unsafe { mul_adx_inner(sub_adx_inner(x2, x3), quarter) };

                a[i] = unsafe { add_adx_inner(t0, t2) };
                a[i + 2] = unsafe { sub_adx_inner(t0, t2) };
                a[i + 1] = unsafe { add_adx_inner(t1, t3) };
                a[i + 3] = unsafe { sub_adx_inner(t1, t3) };
                i += 4;
            }
        } else if m == 2 {
            let w1 = roots[stride];
            let w2 = roots[2 * stride];
            let w3 = roots[3 * stride];
            let mut i = 0;
            while i < n {
                // j = 0: twiddles are (1, 1, 1)
                let x0 = a[i];
                let x1 = a[i + 2];
                let x2 = a[i + 4];
                let x3 = a[i + 6];
                let t0 = unsafe { add_adx_inner(x0, x1) };
                let t1 = unsafe { sub_adx_inner(x0, x1) };
                let t2 = unsafe { add_adx_inner(x2, x3) };
                let t3 =
                    unsafe { mul_adx_inner(sub_adx_inner(x2, x3), quarter) };
                a[i] = unsafe { add_adx_inner(t0, t2) };
                a[i + 4] = unsafe { sub_adx_inner(t0, t2) };
                a[i + 2] = unsafe { add_adx_inner(t1, t3) };
                a[i + 6] = unsafe { sub_adx_inner(t1, t3) };

                // j = 1: twiddles are (w2, w1, w3)
                let x0 = a[i + 1];
                let x1 = unsafe { mul_adx_inner(a[i + 3], w2) };
                let x2 = unsafe { mul_adx_inner(a[i + 5], w1) };
                let x3 = unsafe { mul_adx_inner(a[i + 7], w3) };
                let t0 = unsafe { add_adx_inner(x0, x1) };
                let t1 = unsafe { sub_adx_inner(x0, x1) };
                let t2 = unsafe { add_adx_inner(x2, x3) };
                let t3 =
                    unsafe { mul_adx_inner(sub_adx_inner(x2, x3), quarter) };
                a[i + 1] = unsafe { add_adx_inner(t0, t2) };
                a[i + 5] = unsafe { sub_adx_inner(t0, t2) };
                a[i + 3] = unsafe { add_adx_inner(t1, t3) };
                a[i + 7] = unsafe { sub_adx_inner(t1, t3) };
                i += 8;
            }
        } else {
            let mut k = 0;
            while k < n {
                // j = 0: twiddles are (1, 1, 1)
                let x0 = a[k];
                let x1 = a[k + m];
                let x2 = a[k + 2 * m];
                let x3 = a[k + 3 * m];
                let t0 = unsafe { add_adx_inner(x0, x1) };
                let t1 = unsafe { sub_adx_inner(x0, x1) };
                let t2 = unsafe { add_adx_inner(x2, x3) };
                let t3 =
                    unsafe { mul_adx_inner(sub_adx_inner(x2, x3), quarter) };
                a[k] = unsafe { add_adx_inner(t0, t2) };
                a[k + 2 * m] = unsafe { sub_adx_inner(t0, t2) };
                a[k + m] = unsafe { add_adx_inner(t1, t3) };
                a[k + 3 * m] = unsafe { sub_adx_inner(t1, t3) };

                let mut j = 1;
                while j < m {
                    let idx1 = j * stride;
                    let idx2 = idx1 << 1;
                    let idx3 = idx2 + idx1;
                    let w1 = &roots[idx1];
                    let w2 = &roots[idx2];
                    let w3_neg;
                    let w3 = if idx3 < (n >> 1) {
                        &roots[idx3]
                    } else {
                        // omega^(n/2 + r) = -omega^r
                        w3_neg = unsafe {
                            sub_adx_ref(FastFr::ZERO, &roots[idx3 - (n >> 1)])
                        };
                        &w3_neg
                    };

                    let x0 = a[k + j];
                    let x1 = unsafe { mul_adx_ref(&a[k + j + m], w2) };
                    let x2 = unsafe { mul_adx_ref(&a[k + j + 2 * m], w1) };
                    let x3 = unsafe { mul_adx_ref(&a[k + j + 3 * m], w3) };

                    let t0 = unsafe { add_adx_ref(x0, &x1) };
                    let t1 = unsafe { sub_adx_ref(x0, &x1) };
                    let t2 = unsafe { add_adx_ref(x2, &x3) };
                    let t3 = unsafe {
                        let d = sub_adx_ref(x2, &x3);
                        mul_adx_ref(&d, &quarter)
                    };

                    a[k + j] = unsafe { add_adx_ref(t0, &t2) };
                    a[k + j + 2 * m] = unsafe { sub_adx_ref(t0, &t2) };
                    a[k + j + m] = unsafe { add_adx_ref(t1, &t3) };
                    a[k + j + 3 * m] = unsafe { sub_adx_ref(t1, &t3) };
                    j += 1;
                }
                k += step;
            }
        }

        stage += 2;
        m <<= 2;
    }
}

fn serial_fft_bitreversed_portable(
    a: &mut [FastFr],
    roots: &[FastFr],
    log_n: u32,
) {
    let n = a.len();
    if log_n == 0 {
        return;
    }
    let mut stage = 0u32;
    let mut m = 1usize;
    if (log_n & 1) == 1 {
        for i in (0..n).step_by(2) {
            let u = a[i];
            let v = a[i + 1];
            a[i] = u.add_portable(v);
            a[i + 1] = u.sub_portable(v);
        }
        stage = 1;
        m = 2;
    }
    let quarter = if n >= 4 { roots[n / 4] } else { FastFr::ONE };
    while stage < log_n {
        let step = m << 2;
        let stride = n / step;
        for k in (0..n).step_by(step) {
            for j in 0..m {
                let idx1 = j * stride;
                let idx2 = idx1 << 1;
                let idx3 = idx2 + idx1;
                let w1 = roots[idx1];
                let w2 = roots[idx2];
                let w3 = if idx3 < (n >> 1) {
                    roots[idx3]
                } else {
                    roots[idx3 - (n >> 1)].neg_portable()
                };
                let x0 = a[k + j];
                let x1 = mul_portable(a[k + j + m], w2);
                let x2 = mul_portable(a[k + j + 2 * m], w1);
                let x3 = mul_portable(a[k + j + 3 * m], w3);
                let t0 = x0.add_portable(x1);
                let t1 = x0.sub_portable(x1);
                let t2 = x2.add_portable(x3);
                let t3 = mul_portable(x2.sub_portable(x3), quarter);
                a[k + j] = t0.add_portable(t2);
                a[k + j + 2 * m] = t0.sub_portable(t2);
                a[k + j + m] = t1.add_portable(t3);
                a[k + j + 3 * m] = t1.sub_portable(t3);
            }
        }
        stage += 2;
        m <<= 2;
    }
}

/// Scale a `FastFr` slice in place by `factor`.
pub(crate) fn scale_fast_in_place(a: &mut [FastFr], factor: FastFr) {
    #[cfg(target_arch = "x86_64")]
    if has_adx_bmi2() {
        // SAFETY: runtime checked ADX + BMI2.
        unsafe { scale_fast_adx(a, factor) };
        return;
    }
    for x in a.iter_mut() {
        *x = mul_portable(*x, factor);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "adx,bmi2")]
unsafe fn scale_fast_adx(a: &mut [FastFr], factor: FastFr) {
    for x in a.iter_mut() {
        *x = unsafe { mul_adx_inner(*x, factor) };
    }
}

/// Fold a polynomial `poly` (degree `< 2n`) onto coset `k` with shift powers
/// `coset_powers[j] = c_k^j` and `c_k_pow_n = c_k^n`, scatter bit-reversed
/// into an `n`-element buffer, and run the `n`-point `FastFr` FFT.
pub(crate) fn coset_n_fft_fast(
    poly: &[BlsScalar],
    n: usize,
    log_n: u32,
    c_k_pow_n: FastFr,
    coset_powers: &[FastFr],
    fft_roots: &[FastFr],
) -> Vec<FastFr> {
    let mut evals = vec![FastFr::ZERO; n];
    #[cfg(target_arch = "x86_64")]
    if has_adx_bmi2() {
        // SAFETY: runtime checked ADX + BMI2.
        unsafe {
            coset_fold_bitrev_adx(
                poly,
                &mut evals,
                n,
                log_n,
                c_k_pow_n,
                coset_powers,
            );
            serial_fft_bitreversed_adx(&mut evals, fft_roots, log_n);
        }
        return evals;
    }
    coset_fold_bitrev_portable(
        poly,
        &mut evals,
        n,
        log_n,
        c_k_pow_n,
        coset_powers,
    );
    serial_fft_bitreversed_portable(&mut evals, fft_roots, log_n);
    evals
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "adx,bmi2")]
unsafe fn coset_fold_bitrev_adx(
    poly: &[BlsScalar],
    out: &mut [FastFr],
    n: usize,
    log_n: u32,
    c_k_pow_n: FastFr,
    coset_powers: &[FastFr],
) {
    let len = poly.len();
    debug_assert!(
        len <= 2 * n,
        "coset_fold_bitrev_adx requires poly.len() <= 2 * n"
    );
    let low_len = len.min(n);
    let extra = len.saturating_sub(n);

    // Fold coefficients at indices `n..len` onto `0..extra`:
    // `f[j] = poly[j] + c_k^n * poly[n + j]`.
    for j in 0..extra.min(n) {
        let lo = FastFr::from_bls(&poly[j]);
        let hi = FastFr::from_bls(&poly[n + j]);
        let folded = unsafe { add_adx_inner(lo, mul_adx_inner(hi, c_k_pow_n)) };
        out[bitrev(j as u32, log_n)] =
            unsafe { mul_adx_inner(folded, coset_powers[j]) };
    }
    for j in extra.min(n)..low_len {
        let v = FastFr::from_bls(&poly[j]);
        out[bitrev(j as u32, log_n)] =
            unsafe { mul_adx_inner(v, coset_powers[j]) };
    }
}

fn coset_fold_bitrev_portable(
    poly: &[BlsScalar],
    out: &mut [FastFr],
    n: usize,
    log_n: u32,
    c_k_pow_n: FastFr,
    coset_powers: &[FastFr],
) {
    let len = poly.len();
    debug_assert!(
        len <= 2 * n,
        "coset_fold_bitrev_adx requires poly.len() <= 2 * n"
    );
    let low_len = len.min(n);
    let extra = len.saturating_sub(n);
    for j in 0..extra.min(n) {
        let lo = FastFr::from_bls(&poly[j]);
        let hi = FastFr::from_bls(&poly[n + j]);
        let folded = lo.add_portable(mul_portable(hi, c_k_pow_n));
        out[bitrev(j as u32, log_n)] = mul_portable(folded, coset_powers[j]);
    }
    for j in extra.min(n)..low_len {
        let v = FastFr::from_bls(&poly[j]);
        out[bitrev(j as u32, log_n)] = mul_portable(v, coset_powers[j]);
    }
}

/// Serial `n`-point IFFT over `FastFr` (bit-reverse copy + radix-4 FFT with
/// inverse roots + scale by `size_inv`).
pub(crate) fn serial_ifft_fast(
    evals: &[FastFr],
    ifft_roots: &[FastFr],
    size_inv: FastFr,
    log_n: u32,
) -> Vec<FastFr> {
    let n = 1usize << log_n;
    let mut coeffs = vec![FastFr::ZERO; n];
    for (k, &v) in evals.iter().enumerate() {
        coeffs[bitrev(k as u32, log_n)] = v;
    }
    serial_fft_bitreversed_fast(&mut coeffs, ifft_roots, log_n);
    scale_fast_in_place(&mut coeffs, size_inv);
    coeffs
}

/// Serial `n`-point IFFT from `&[BlsScalar]` directly to `Vec<BlsScalar>`
/// using the ADX/BMI2 `FastFr` kernel.
pub(crate) fn serial_ifft_bls_fast(
    evals: &[BlsScalar],
    ifft_roots: &[FastFr],
    size_inv: FastFr,
    log_n: u32,
) -> Vec<BlsScalar> {
    let n = 1usize << log_n;
    let mut coeffs = vec![FastFr::ZERO; n];
    for (k, v) in evals.iter().take(n).enumerate() {
        coeffs[bitrev(k as u32, log_n)] = FastFr::from_bls(v);
    }
    serial_fft_bitreversed_fast(&mut coeffs, ifft_roots, log_n);
    scale_fast_in_place(&mut coeffs, size_inv);
    coeffs.into_iter().map(FastFr::to_bls).collect()
}

/// Fast polynomial evaluation `sum_i coeffs[i] * point^i` using 4-way
/// instruction-level-parallel Horner in ADX/BMI2 `FastFr`.
pub(crate) fn poly_evaluate_fast(
    coeffs: &[BlsScalar],
    point: &BlsScalar,
) -> BlsScalar {
    if coeffs.is_empty() {
        return BlsScalar::zero();
    }
    #[cfg(target_arch = "x86_64")]
    if has_adx_bmi2() {
        // SAFETY: runtime checked ADX + BMI2.
        return unsafe { poly_evaluate_adx(coeffs, point) };
    }
    let mut result = BlsScalar::zero();
    for coeff in coeffs.iter().rev() {
        result *= point;
        result += coeff;
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "adx,bmi2")]
unsafe fn poly_evaluate_adx(
    coeffs: &[BlsScalar],
    point: &BlsScalar,
) -> BlsScalar {
    let x = FastFr::from_bls(point);
    let len = coeffs.len();
    if len < 8 {
        let mut acc = FastFr::ZERO;
        for c in coeffs.iter().rev() {
            let fc = FastFr::from_bls(c);
            acc = unsafe { add_adx_ref(mul_adx_ref(&acc, &x), &fc) };
        }
        return acc.to_bls();
    }

    let x2 = unsafe { mul_adx_ref(&x, &x) };
    let x3 = unsafe { mul_adx_ref(&x2, &x) };
    let x4 = unsafe { mul_adx_ref(&x2, &x2) };

    let mut a0 = FastFr::ZERO;
    let mut a1 = FastFr::ZERO;
    let mut a2 = FastFr::ZERO;
    let mut a3 = FastFr::ZERO;

    let top = len & !3;
    for idx in (top..len).rev() {
        let fc = FastFr::from_bls(&coeffs[idx]);
        match idx & 3 {
            0 => a0 = fc,
            1 => a1 = fc,
            2 => a2 = fc,
            _ => a3 = fc,
        }
    }
    let mut k = top;
    while k >= 4 {
        k -= 4;
        let c0 = FastFr::from_bls(&coeffs[k]);
        let c1 = FastFr::from_bls(&coeffs[k + 1]);
        let c2 = FastFr::from_bls(&coeffs[k + 2]);
        let c3 = FastFr::from_bls(&coeffs[k + 3]);
        a0 = unsafe { add_adx_ref(mul_adx_ref(&a0, &x4), &c0) };
        a1 = unsafe { add_adx_ref(mul_adx_ref(&a1, &x4), &c1) };
        a2 = unsafe { add_adx_ref(mul_adx_ref(&a2, &x4), &c2) };
        a3 = unsafe { add_adx_ref(mul_adx_ref(&a3, &x4), &c3) };
    }

    let t1 = unsafe { mul_adx_ref(&a1, &x) };
    let t2 = unsafe { mul_adx_ref(&a2, &x2) };
    let t3 = unsafe { mul_adx_ref(&a3, &x3) };
    let sum =
        unsafe { add_adx_ref(add_adx_ref(a0, &t1), &add_adx_ref(t2, &t3)) };
    sum.to_bls()
}

/// Fast Ruffini synthetic division of `coeffs` by `(X - z)` using ADX/BMI2
/// and 2-pass parallel block carry propagation for large polynomials.
pub(crate) fn poly_ruffini_fast(
    coeffs: &[BlsScalar],
    z: &BlsScalar,
) -> Vec<BlsScalar> {
    let quotient_len = coeffs.len().saturating_sub(1);
    if quotient_len == 0 {
        return Vec::new();
    }

    const PARALLEL_RUFFINI_MIN_LEN: usize = 1 << 12;
    const RUFFINI_BLOCK: usize = 2048;

    #[cfg(feature = "std")]
    if quotient_len >= PARALLEL_RUFFINI_MIN_LEN
        && rayon::current_num_threads() > 1
    {
        use rayon::prelude::*;

        let fz = FastFr::from_bls(z);
        let num_blocks = quotient_len.div_ceil(RUFFINI_BLOCK);
        let mut local_q = vec![FastFr::ZERO; quotient_len];

        // Pass 1: independent local Ruffini per block starting from carry 0.
        let block_exits: Vec<(FastFr, usize)> = local_q
            .par_chunks_mut(RUFFINI_BLOCK)
            .enumerate()
            .map(|(b, out_blk)| {
                let start = b * RUFFINI_BLOCK;
                let len = out_blk.len();
                let exit = poly_ruffini_local_fast(
                    &coeffs[start + 1..start + 1 + len],
                    fz,
                    out_blk,
                );
                (exit, len)
            })
            .collect();

        // Serial carry propagation across the ~32 blocks (< 1 µs).
        let z_pow_block =
            FastFr::from_bls(&z.pow_vartime(&[RUFFINI_BLOCK as u64, 0, 0, 0]));
        let mut incoming = vec![FastFr::ZERO; num_blocks];
        for b in (0..num_blocks - 1).rev() {
            let (exit_above, len_above) = block_exits[b + 1];
            let z_pow_above = if len_above == RUFFINI_BLOCK {
                z_pow_block
            } else {
                FastFr::from_bls(&z.pow_vartime(&[len_above as u64, 0, 0, 0]))
            };
            let carry_above = incoming[b + 1];
            let shifted = mul_fast(carry_above, z_pow_above);
            incoming[b] = add_fast(exit_above, shifted);
        }

        // Pass 2: add `incoming[b] * z^(end - i)` and convert `FastFr -> BlsScalar`.
        let mut quotient = vec![BlsScalar::zero(); quotient_len];
        quotient
            .par_chunks_mut(RUFFINI_BLOCK)
            .zip(local_q.par_chunks(RUFFINI_BLOCK))
            .enumerate()
            .for_each(|(b, (dst_blk, src_blk))| {
                poly_ruffini_apply_carry_fast(
                    src_blk,
                    incoming[b],
                    fz,
                    dst_blk,
                );
            });
        return quotient;
    }

    #[cfg(target_arch = "x86_64")]
    if has_adx_bmi2() {
        // SAFETY: runtime checked ADX + BMI2.
        return unsafe { poly_ruffini_adx(coeffs, z) };
    }
    let mut quotient = vec![BlsScalar::zero(); quotient_len];
    let mut value = coeffs[quotient_len];
    quotient[quotient_len - 1] = value;
    for i in (0..quotient_len - 1).rev() {
        value = coeffs[i + 1] + *z * value;
        quotient[i] = value;
    }
    quotient
}

#[inline]
fn mul_fast(a: FastFr, b: FastFr) -> FastFr {
    #[cfg(target_arch = "x86_64")]
    if has_adx_bmi2() {
        // SAFETY: runtime checked ADX + BMI2.
        return unsafe { mul_adx_inner(a, b) };
    }
    mul_portable(a, b)
}

#[inline]
fn add_fast(a: FastFr, b: FastFr) -> FastFr {
    #[cfg(target_arch = "x86_64")]
    if has_adx_bmi2() {
        // SAFETY: runtime checked ADX + BMI2.
        return unsafe { add_adx_inner(a, b) };
    }
    a.add_portable(b)
}

#[cfg(feature = "std")]
fn poly_ruffini_local_fast(
    src_plus_one: &[BlsScalar],
    fz: FastFr,
    out_blk: &mut [FastFr],
) -> FastFr {
    #[cfg(target_arch = "x86_64")]
    if has_adx_bmi2() {
        // SAFETY: runtime checked ADX + BMI2.
        return unsafe { poly_ruffini_local_adx(src_plus_one, fz, out_blk) };
    }
    let len = out_blk.len();
    let mut value = FastFr::ZERO;
    for i in (0..len).rev() {
        let c = FastFr::from_bls(&src_plus_one[i]);
        value = mul_portable(value, fz).add_portable(c);
        out_blk[i] = value;
    }
    value
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "adx,bmi2")]
unsafe fn poly_ruffini_local_adx(
    src_plus_one: &[BlsScalar],
    fz: FastFr,
    out_blk: &mut [FastFr],
) -> FastFr {
    let len = out_blk.len();
    let mut value = FastFr::ZERO;
    for i in (0..len).rev() {
        let c = FastFr::from_bls(&src_plus_one[i]);
        value = unsafe { add_adx_ref(mul_adx_ref(&value, &fz), &c) };
        out_blk[i] = value;
    }
    value
}

#[cfg(feature = "std")]
fn poly_ruffini_apply_carry_fast(
    src_blk: &[FastFr],
    incoming: FastFr,
    fz: FastFr,
    dst_blk: &mut [BlsScalar],
) {
    if incoming.is_zero() {
        for (d, s) in dst_blk.iter_mut().zip(src_blk.iter()) {
            *d = s.to_bls();
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if has_adx_bmi2() {
        // SAFETY: runtime checked ADX + BMI2.
        unsafe {
            poly_ruffini_apply_carry_adx(src_blk, incoming, fz, dst_blk);
        }
        return;
    }
    let len = src_blk.len();
    let mut carry = incoming;
    for i in (0..len).rev() {
        carry = mul_portable(carry, fz);
        dst_blk[i] = src_blk[i].add_portable(carry).to_bls();
    }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "adx,bmi2")]
unsafe fn poly_ruffini_apply_carry_adx(
    src_blk: &[FastFr],
    incoming: FastFr,
    fz: FastFr,
    dst_blk: &mut [BlsScalar],
) {
    let len = src_blk.len();
    let mut carry = incoming;
    for i in (0..len).rev() {
        carry = unsafe { mul_adx_ref(&carry, &fz) };
        let v = unsafe { add_adx_ref(src_blk[i], &carry) };
        dst_blk[i] = v.to_bls();
    }
}

/// Compute one chunk of permutation ratios using `FastFr` (ADX/BMI2 when
/// available), batch-invert denominators in-place, compute the chunk's local
/// exclusive prefix product (`local[0] = 1`), and return the total product of
/// the chunk.
#[cfg(feature = "std")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn perm_chunk_prefix_fast(
    wires: [&[BlsScalar]; 4],
    sigmas: [&[BlsScalar]; 4],
    start: usize,
    init_root: FastFr,
    group_gen: FastFr,
    beta: FastFr,
    beta_k1: FastFr,
    beta_k2: FastFr,
    beta_k3: FastFr,
    gamma: FastFr,
    out_chunk: &mut [FastFr],
    denoms: &mut Vec<FastFr>,
    prods: &mut Vec<FastFr>,
) -> FastFr {
    #[cfg(target_arch = "x86_64")]
    if has_adx_bmi2() {
        // SAFETY: runtime checked ADX + BMI2.
        return unsafe {
            perm_chunk_prefix_adx(
                wires, sigmas, start, init_root, group_gen, beta, beta_k1,
                beta_k2, beta_k3, gamma, out_chunk, denoms, prods,
            )
        };
    }
    perm_chunk_prefix_portable(
        wires, sigmas, start, init_root, group_gen, beta, beta_k1, beta_k2,
        beta_k3, gamma, out_chunk, denoms, prods,
    )
}

#[cfg(feature = "std")]
#[inline]
fn invert_fast_fr_nonzero(v: FastFr) -> FastFr {
    #[cfg(feature = "blst-backend")]
    {
        let mut blst_in = blst::blst_fr { l: v.0 };
        let mut blst_out = blst::blst_fr::default();
        // SAFETY: `v` is nonzero; `blst_fr` matches `FastFr` Montgomery limbs.
        unsafe {
            blst::blst_fr_eucl_inverse(&mut blst_out, &blst_in);
        }
        let _ = &mut blst_in;
        FastFr(blst_out.l)
    }
    #[cfg(not(feature = "blst-backend"))]
    {
        FastFr::from_bls(&v.to_bls().invert().expect("nonzero"))
    }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "adx,bmi2")]
#[allow(clippy::too_many_arguments)]
unsafe fn perm_chunk_prefix_adx(
    wires: [&[BlsScalar]; 4],
    sigmas: [&[BlsScalar]; 4],
    start: usize,
    mut root: FastFr,
    group_gen: FastFr,
    beta: FastFr,
    beta_k1: FastFr,
    beta_k2: FastFr,
    beta_k3: FastFr,
    gamma: FastFr,
    out_chunk: &mut [FastFr],
    denoms: &mut Vec<FastFr>,
    prods: &mut Vec<FastFr>,
) -> FastFr {
    let len = out_chunk.len();
    denoms.clear();
    prods.clear();
    let mut acc_prod = FastFr::ONE;

    for (offset, slot) in out_chunk.iter_mut().enumerate() {
        let i = start + offset;
        let a_g =
            unsafe { add_adx_ref(FastFr::from_bls(&wires[0][i]), &gamma) };
        let b_g =
            unsafe { add_adx_ref(FastFr::from_bls(&wires[1][i]), &gamma) };
        let c_g =
            unsafe { add_adx_ref(FastFr::from_bls(&wires[2][i]), &gamma) };
        let d_g =
            unsafe { add_adx_ref(FastFr::from_bls(&wires[3][i]), &gamma) };

        let n0 = unsafe { add_adx_ref(a_g, &mul_adx_ref(&beta, &root)) };
        let n1 = unsafe { add_adx_ref(b_g, &mul_adx_ref(&beta_k1, &root)) };
        let n2 = unsafe { add_adx_ref(c_g, &mul_adx_ref(&beta_k2, &root)) };
        let n3 = unsafe { add_adx_ref(d_g, &mul_adx_ref(&beta_k3, &root)) };
        let n01 = unsafe { mul_adx_ref(&n0, &n1) };
        let n23 = unsafe { mul_adx_ref(&n2, &n3) };
        *slot = unsafe { mul_adx_ref(&n01, &n23) };

        let s0 = FastFr::from_bls(&sigmas[0][i]);
        let s1 = FastFr::from_bls(&sigmas[1][i]);
        let s2 = FastFr::from_bls(&sigmas[2][i]);
        let s3 = FastFr::from_bls(&sigmas[3][i]);
        let d0 = unsafe { add_adx_ref(a_g, &mul_adx_ref(&beta, &s0)) };
        let d1 = unsafe { add_adx_ref(b_g, &mul_adx_ref(&beta, &s1)) };
        let d2 = unsafe { add_adx_ref(c_g, &mul_adx_ref(&beta, &s2)) };
        let d3 = unsafe { add_adx_ref(d_g, &mul_adx_ref(&beta, &s3)) };
        let d01 = unsafe { mul_adx_ref(&d0, &d1) };
        let d23 = unsafe { mul_adx_ref(&d2, &d3) };
        let den = unsafe { mul_adx_ref(&d01, &d23) };
        assert!(!den.is_zero(), "permutation denominator must be nonzero");
        denoms.push(den);
        prods.push(acc_prod);
        acc_prod = unsafe { mul_adx_ref(&acc_prod, &den) };
        root = unsafe { mul_adx_ref(&root, &group_gen) };
    }

    let mut inv_acc = invert_fast_fr_nonzero(acc_prod);
    for k in (0..len).rev() {
        let den = denoms[k];
        let den_inv = unsafe { mul_adx_ref(&inv_acc, &prods[k]) };
        inv_acc = unsafe { mul_adx_ref(&inv_acc, &den) };
        out_chunk[k] = unsafe { mul_adx_ref(&out_chunk[k], &den_inv) };
    }

    let mut prefix = FastFr::ONE;
    for slot in out_chunk.iter_mut() {
        let r = *slot;
        *slot = prefix;
        prefix = unsafe { mul_adx_ref(&prefix, &r) };
    }
    prefix
}

#[cfg(feature = "std")]
#[allow(clippy::too_many_arguments)]
fn perm_chunk_prefix_portable(
    wires: [&[BlsScalar]; 4],
    sigmas: [&[BlsScalar]; 4],
    start: usize,
    mut root: FastFr,
    group_gen: FastFr,
    beta: FastFr,
    beta_k1: FastFr,
    beta_k2: FastFr,
    beta_k3: FastFr,
    gamma: FastFr,
    out_chunk: &mut [FastFr],
    denoms: &mut Vec<FastFr>,
    prods: &mut Vec<FastFr>,
) -> FastFr {
    let len = out_chunk.len();
    denoms.clear();
    prods.clear();
    let mut acc_prod = FastFr::ONE;

    for (offset, slot) in out_chunk.iter_mut().enumerate() {
        let i = start + offset;
        let a_g = FastFr::from_bls(&wires[0][i]).add_portable(gamma);
        let b_g = FastFr::from_bls(&wires[1][i]).add_portable(gamma);
        let c_g = FastFr::from_bls(&wires[2][i]).add_portable(gamma);
        let d_g = FastFr::from_bls(&wires[3][i]).add_portable(gamma);

        let n0 = a_g.add_portable(mul_portable(beta, root));
        let n1 = b_g.add_portable(mul_portable(beta_k1, root));
        let n2 = c_g.add_portable(mul_portable(beta_k2, root));
        let n3 = d_g.add_portable(mul_portable(beta_k3, root));
        *slot = mul_portable(mul_portable(n0, n1), mul_portable(n2, n3));

        let d0 = a_g
            .add_portable(mul_portable(beta, FastFr::from_bls(&sigmas[0][i])));
        let d1 = b_g
            .add_portable(mul_portable(beta, FastFr::from_bls(&sigmas[1][i])));
        let d2 = c_g
            .add_portable(mul_portable(beta, FastFr::from_bls(&sigmas[2][i])));
        let d3 = d_g
            .add_portable(mul_portable(beta, FastFr::from_bls(&sigmas[3][i])));
        let den = mul_portable(mul_portable(d0, d1), mul_portable(d2, d3));
        assert!(!den.is_zero(), "permutation denominator must be nonzero");
        denoms.push(den);
        prods.push(acc_prod);
        acc_prod = mul_portable(acc_prod, den);
        root = mul_portable(root, group_gen);
    }

    let mut inv_acc = invert_fast_fr_nonzero(acc_prod);
    for k in (0..len).rev() {
        let den = denoms[k];
        let den_inv = mul_portable(inv_acc, prods[k]);
        inv_acc = mul_portable(inv_acc, den);
        out_chunk[k] = mul_portable(out_chunk[k], den_inv);
    }

    let mut prefix = FastFr::ONE;
    for slot in out_chunk.iter_mut() {
        let r = *slot;
        *slot = prefix;
        prefix = mul_portable(prefix, r);
    }
    prefix
}

/// Multiply each `FastFr` in `src` by `scale` (skipping the mul when
/// `scale == ONE`) and write canonical `BlsScalar` outputs into `dst`.
#[cfg(feature = "std")]
pub(crate) fn scale_fast_fr_to_bls_chunk(
    src: &[FastFr],
    scale: FastFr,
    dst: &mut [BlsScalar],
) {
    if scale == FastFr::ONE {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = s.to_bls();
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if has_adx_bmi2() {
        // SAFETY: runtime checked ADX + BMI2.
        unsafe { scale_fast_fr_to_bls_adx(src, scale, dst) };
        return;
    }
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = mul_portable(*s, scale).to_bls();
    }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "adx,bmi2")]
unsafe fn scale_fast_fr_to_bls_adx(
    src: &[FastFr],
    scale: FastFr,
    dst: &mut [BlsScalar],
) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = unsafe { mul_adx_ref(s, &scale) }.to_bls();
    }
}

/// Fused linear combination + 2-pass parallel block Ruffini division producing
/// `Vec<FastFr>` directly (zero intermediate `FastFr -> BlsScalar` Montgomery
/// round-trips when feeding the result straight into GPU MSM).
#[cfg(feature = "gpu-cuda")]
pub(crate) fn poly_lincomb_and_ruffini_fast_fr(
    terms: &[(BlsScalar, &[BlsScalar])],
    z: &BlsScalar,
) -> Vec<FastFr> {
    use rayon::prelude::*;

    let max_len = terms
        .iter()
        .filter(|(s, p)| *s != BlsScalar::zero() && !p.is_empty())
        .map(|(_, p)| p.len())
        .max()
        .unwrap_or(0);
    let quotient_len = max_len.saturating_sub(1);
    if quotient_len == 0 {
        return Vec::new();
    }

    // 1. Compute coefficients[1..max_len] directly into `Vec<FastFr>` (index 0
    // is discarded by Ruffini division by `X - z`).
    let mut coeffs = vec![FastFr::ZERO; quotient_len];
    const CHUNK: usize = 512;
    let fast_terms: Vec<(FastFr, &[BlsScalar])> = terms
        .iter()
        .filter_map(|&(s, p)| {
            let fs = FastFr::from_bls(&s);
            (!fs.is_zero() && p.len() > 1).then_some((fs, &p[1..]))
        })
        .collect();
    if quotient_len >= (1 << 12) && rayon::current_num_threads() > 1 {
        coeffs.par_chunks_mut(CHUNK).enumerate().for_each(
            |(chunk_idx, out_chunk)| {
                poly_lincomb_chunk_fast_fr(
                    &fast_terms,
                    chunk_idx * CHUNK,
                    out_chunk,
                );
            },
        );
    } else {
        poly_lincomb_chunk_fast_fr(&fast_terms, 0, &mut coeffs);
    }

    // 2. Parallel block Ruffini in-place on `Vec<FastFr>` (zero `to_bls`!).
    let fz = FastFr::from_bls(z);
    const RUFFINI_BLOCK: usize = 2048;
    if quotient_len >= (1 << 12) && rayon::current_num_threads() > 1 {
        let num_blocks = quotient_len.div_ceil(RUFFINI_BLOCK);
        let block_exits: Vec<(FastFr, usize)> = coeffs
            .par_chunks_mut(RUFFINI_BLOCK)
            .map(|blk| (poly_ruffini_inplace_local_fast_fr(blk, fz), blk.len()))
            .collect();

        let z_pow_block =
            FastFr::from_bls(&z.pow_vartime(&[RUFFINI_BLOCK as u64, 0, 0, 0]));
        let mut incoming = vec![FastFr::ZERO; num_blocks];
        for b in (0..num_blocks - 1).rev() {
            let (exit_above, len_above) = block_exits[b + 1];
            let z_pow_above = if len_above == RUFFINI_BLOCK {
                z_pow_block
            } else {
                FastFr::from_bls(&z.pow_vartime(&[len_above as u64, 0, 0, 0]))
            };
            let shifted = mul_fast(incoming[b + 1], z_pow_above);
            incoming[b] = add_fast(exit_above, shifted);
        }

        coeffs.par_chunks_mut(RUFFINI_BLOCK).enumerate().for_each(
            |(b, blk)| {
                let inc = incoming[b];
                if !inc.is_zero() {
                    poly_ruffini_apply_carry_inplace_fast_fr(blk, inc, fz);
                }
            },
        );
    } else {
        poly_ruffini_inplace_local_fast_fr(&mut coeffs, fz);
    }

    coeffs
}

#[cfg(feature = "gpu-cuda")]
fn poly_lincomb_chunk_fast_fr(
    terms: &[(FastFr, &[BlsScalar])],
    start: usize,
    out_chunk: &mut [FastFr],
) {
    #[cfg(target_arch = "x86_64")]
    if has_adx_bmi2() {
        // SAFETY: runtime checked ADX + BMI2.
        unsafe { poly_lincomb_chunk_fast_fr_adx(terms, start, out_chunk) };
        return;
    }
    let len = out_chunk.len();
    for &(fs, poly) in terms {
        if start >= poly.len() {
            continue;
        }
        let end = (start + len).min(poly.len());
        for (dst, src) in
            out_chunk[..end - start].iter_mut().zip(&poly[start..end])
        {
            *dst = dst.add_portable(mul_portable(FastFr::from_bls(src), fs));
        }
    }
}

#[cfg(all(feature = "gpu-cuda", target_arch = "x86_64"))]
#[target_feature(enable = "adx,bmi2")]
unsafe fn poly_lincomb_chunk_fast_fr_adx(
    terms: &[(FastFr, &[BlsScalar])],
    start: usize,
    out_chunk: &mut [FastFr],
) {
    let len = out_chunk.len();
    for &(fs, poly) in terms {
        if start >= poly.len() {
            continue;
        }
        let end = (start + len).min(poly.len());
        for (dst, src) in
            out_chunk[..end - start].iter_mut().zip(&poly[start..end])
        {
            let fsrc = FastFr::from_bls(src);
            let prod = unsafe { mul_adx_ref(&fsrc, &fs) };
            *dst = unsafe { add_adx_ref(*dst, &prod) };
        }
    }
}

#[cfg(feature = "gpu-cuda")]
fn poly_ruffini_inplace_local_fast_fr(
    blk: &mut [FastFr],
    fz: FastFr,
) -> FastFr {
    #[cfg(target_arch = "x86_64")]
    if has_adx_bmi2() {
        // SAFETY: runtime checked ADX + BMI2.
        return unsafe { poly_ruffini_inplace_local_fast_fr_adx(blk, fz) };
    }
    let mut value = FastFr::ZERO;
    for slot in blk.iter_mut().rev() {
        value = mul_portable(value, fz).add_portable(*slot);
        *slot = value;
    }
    value
}

#[cfg(all(feature = "gpu-cuda", target_arch = "x86_64"))]
#[target_feature(enable = "adx,bmi2")]
unsafe fn poly_ruffini_inplace_local_fast_fr_adx(
    blk: &mut [FastFr],
    fz: FastFr,
) -> FastFr {
    let mut value = FastFr::ZERO;
    for slot in blk.iter_mut().rev() {
        value = unsafe { add_adx_ref(mul_adx_ref(&value, &fz), slot) };
        *slot = value;
    }
    value
}

#[cfg(feature = "gpu-cuda")]
fn poly_ruffini_apply_carry_inplace_fast_fr(
    blk: &mut [FastFr],
    incoming: FastFr,
    fz: FastFr,
) {
    #[cfg(target_arch = "x86_64")]
    if has_adx_bmi2() {
        // SAFETY: runtime checked ADX + BMI2.
        unsafe {
            poly_ruffini_apply_carry_inplace_fast_fr_adx(blk, incoming, fz);
        }
        return;
    }
    let mut carry = incoming;
    for slot in blk.iter_mut().rev() {
        carry = mul_portable(carry, fz);
        *slot = slot.add_portable(carry);
    }
}

#[cfg(all(feature = "gpu-cuda", target_arch = "x86_64"))]
#[target_feature(enable = "adx,bmi2")]
unsafe fn poly_ruffini_apply_carry_inplace_fast_fr_adx(
    blk: &mut [FastFr],
    incoming: FastFr,
    fz: FastFr,
) {
    let mut carry = incoming;
    for slot in blk.iter_mut().rev() {
        carry = unsafe { mul_adx_ref(&carry, &fz) };
        *slot = unsafe { add_adx_ref(*slot, &carry) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "adx,bmi2")]
unsafe fn poly_ruffini_adx(
    coeffs: &[BlsScalar],
    z: &BlsScalar,
) -> Vec<BlsScalar> {
    let quotient_len = coeffs.len().saturating_sub(1);
    let mut quotient = vec![BlsScalar::zero(); quotient_len];
    let fz = FastFr::from_bls(z);
    let mut value = FastFr::from_bls(&coeffs[quotient_len]);
    quotient[quotient_len - 1] = coeffs[quotient_len];
    for i in (0..quotient_len - 1).rev() {
        let c = FastFr::from_bls(&coeffs[i + 1]);
        value = unsafe { add_adx_ref(mul_adx_ref(&value, &fz), &c) };
        quotient[i] = value.to_bls();
    }
    quotient
}

/// Accumulate `out_chunk[j] = sum_k scalar_k * poly_k[start + j]` using ADX/BMI2.
pub(crate) fn poly_lincomb_chunk_fast(
    terms: &[(BlsScalar, &[BlsScalar])],
    start: usize,
    out_chunk: &mut [BlsScalar],
) {
    #[cfg(target_arch = "x86_64")]
    if has_adx_bmi2() {
        // SAFETY: runtime checked ADX + BMI2.
        unsafe { poly_lincomb_chunk_adx(terms, start, out_chunk) };
        return;
    }
    for (scalar, poly) in terms {
        if *scalar == BlsScalar::zero() || start >= poly.len() {
            continue;
        }
        let end = (start + out_chunk.len()).min(poly.len());
        for (dst, src) in
            out_chunk[..end - start].iter_mut().zip(&poly[start..end])
        {
            *dst += *src * scalar;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "adx,bmi2")]
unsafe fn poly_lincomb_chunk_adx(
    terms: &[(BlsScalar, &[BlsScalar])],
    start: usize,
    out_chunk: &mut [BlsScalar],
) {
    let len = out_chunk.len();
    let mut acc: Vec<FastFr> = out_chunk.iter().map(FastFr::from_bls).collect();
    for (scalar, poly) in terms {
        let fs = FastFr::from_bls(scalar);
        if fs.is_zero() || start >= poly.len() {
            continue;
        }
        let end = (start + len).min(poly.len());
        for (dst, src) in acc[..end - start].iter_mut().zip(&poly[start..end]) {
            let fsrc = FastFr::from_bls(src);
            let prod = unsafe { mul_adx_ref(&fsrc, &fs) };
            *dst = unsafe { add_adx_ref(*dst, &prod) };
        }
    }
    for (dst, val) in out_chunk.iter_mut().zip(acc) {
        *dst = val.to_bls();
    }
}

#[cfg(test)]
mod tests {
    use ff::Field;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::fft::EvaluationDomain;

    #[test]
    fn fast_fr_arithmetic_matches_bls_scalar() {
        let mut rng = StdRng::seed_from_u64(0xface_b00c);
        let edges = [
            BlsScalar::zero(),
            BlsScalar::one(),
            -BlsScalar::one(),
            BlsScalar::from(2u64),
            -BlsScalar::from(2u64),
            BlsScalar::from(u64::MAX),
            BlsScalar::from(2u64).pow(&[254, 0, 0, 0]),
            BlsScalar::from(2u64).pow(&[254, 0, 0, 0]) - BlsScalar::one(),
            dusk_bls12_381::GENERATOR,
            dusk_bls12_381::ROOT_OF_UNITY,
        ];
        let mut vals: Vec<BlsScalar> = edges.to_vec();
        for _ in 0..200 {
            vals.push(BlsScalar::random(&mut rng));
        }

        for &a in &vals {
            let fa = FastFr::from_bls(&a);
            assert_eq!(fa.to_bls(), a);
            for &b in &edges {
                let fb = FastFr::from_bls(&b);
                assert_eq!(mul_portable(fa, fb).to_bls(), a * b);
                assert_eq!(fa.add_portable(fb).to_bls(), a + b);
                assert_eq!(fa.sub_portable(fb).to_bls(), a - b);
                #[cfg(target_arch = "x86_64")]
                if has_adx_bmi2() {
                    unsafe {
                        assert_eq!(mul_adx_inner(fa, fb).to_bls(), a * b);
                        assert_eq!(add_adx_inner(fa, fb).to_bls(), a + b);
                        assert_eq!(sub_adx_inner(fa, fb).to_bls(), a - b);
                    }
                }
            }
        }
    }

    #[test]
    fn fast_fr_fft_and_ifft_match_domain() {
        let mut rng = StdRng::seed_from_u64(0xdead_1234);
        for log_n in 0u32..=12 {
            let n = 1usize << log_n;
            let domain = EvaluationDomain::new(n).unwrap();
            let input: Vec<BlsScalar> =
                (0..n).map(|_| BlsScalar::random(&mut rng)).collect();
            let expected_fft = domain.fft(&input);
            let expected_ifft = domain.ifft(&input);

            let fft_roots = bls_slice_to_fast(&domain.precompute_fft_roots());
            let ifft_roots = bls_slice_to_fast(&domain.precompute_ifft_roots());

            let mut buf = vec![FastFr::ZERO; n];
            for (k, v) in input.iter().enumerate() {
                buf[bitrev(k as u32, log_n)] = FastFr::from_bls(v);
            }
            serial_fft_bitreversed_fast(&mut buf, &fft_roots, log_n);
            let got_fft: Vec<BlsScalar> =
                buf.into_iter().map(FastFr::to_bls).collect();
            assert_eq!(got_fft, expected_fft, "fft mismatch log_n={log_n}");

            let got_ifft = serial_ifft_bls_fast(
                &input,
                &ifft_roots,
                FastFr::from_bls(&domain.size_inv),
                log_n,
            );
            assert_eq!(got_ifft, expected_ifft, "ifft mismatch log_n={log_n}");
        }
    }

    #[test]
    fn fast_fr_poly_eval_ruffini_lincomb_match_reference() {
        let mut rng = StdRng::seed_from_u64(0x5150_9001);
        for len in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 13, 64, 131] {
            let coeffs: Vec<BlsScalar> =
                (0..len).map(|_| BlsScalar::random(&mut rng)).collect();
            let z = BlsScalar::random(&mut rng);

            let mut expected_eval = BlsScalar::zero();
            for c in coeffs.iter().rev() {
                expected_eval *= z;
                expected_eval += c;
            }
            assert_eq!(
                poly_evaluate_fast(&coeffs, &z),
                expected_eval,
                "len={len}"
            );

            let got_ruf = poly_ruffini_fast(&coeffs, &z);
            let mut expected_ruf = Vec::with_capacity(len.saturating_sub(1));
            if len > 1 {
                let mut val = coeffs[len - 1];
                expected_ruf.push(val);
                for c in coeffs[1..len - 1].iter().rev() {
                    val = *c + z * val;
                    expected_ruf.push(val);
                }
                expected_ruf.reverse();
            }
            assert_eq!(got_ruf, expected_ruf, "ruffini len={len}");
        }

        let p0: Vec<BlsScalar> =
            (0..97).map(|_| BlsScalar::random(&mut rng)).collect();
        let p1: Vec<BlsScalar> =
            (0..53).map(|_| BlsScalar::random(&mut rng)).collect();
        let s0 = BlsScalar::random(&mut rng);
        let s1 = BlsScalar::random(&mut rng);
        let mut got = vec![BlsScalar::zero(); 97];
        poly_lincomb_chunk_fast(
            &[(s0, &p0), (BlsScalar::zero(), &p0), (s1, &p1)],
            0,
            &mut got,
        );
        let mut expected = vec![BlsScalar::zero(); 97];
        for i in 0..97 {
            expected[i] += p0[i] * s0;
            if i < p1.len() {
                expected[i] += p1[i] * s1;
            }
        }
        assert_eq!(got, expected);
    }
}
