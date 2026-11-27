// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! Programmatic PTX code generator for BLS12-381 `Fp` (384-bit), `Fr`
//! (256-bit), register-resident XYZZ elliptic-curve group arithmetic, and GPU
//! Pippenger MSM kernels.
//!
//! Emits portable `.version 7.5` / `.target sm_70` PTX with **zero `.local`
//! stack frames and zero device function-call overhead** so the NVIDIA driver
//! JIT-compiles register-resident native SASS for Volta, Turing, Ampere, Ada,
//! Hopper, and Blackwell (SM 12.0) GPUs with zero `nvcc` dependency.

use core::fmt::Write as _;

/// BLS12-381 base field modulus `p` (little-endian `[u64; 6]`).
pub(crate) const FP_P: [u64; 6] = [
    0xb9fe_ffff_ffff_aaab,
    0x1eab_fffe_b153_ffff,
    0x6730_d2a0_f6b0_f624,
    0x6477_4b84_f385_12bf,
    0x4b1b_a7b6_434b_acd7,
    0x1a01_11ea_397f_e69a,
];

/// `-p^{-1} mod 2^64` for Montgomery CIOS reduction in `Fp`.
pub(crate) const FP_INV: u64 = 0x89f3_fffc_fffc_fffd;

/// `1` in BLS12-381 `Fp` Montgomery form (`2^384 mod p`).
pub(crate) const FP_ONE_MONT: [u64; 6] = [
    0x7609_0000_0002_fffd,
    0xebf4_000b_c40c_0002,
    0x5f48_9857_53c7_58ba,
    0x77ce_5853_7052_5745,
    0x5c07_1a97_a256_ec6d,
    0x15f6_5ec3_fa80_e493,
];

/// BLS12-381 scalar field modulus `r` (little-endian `[u64; 4]`).
pub(crate) const FR_R: [u64; 4] = crate::fast_fr::MODULUS;

/// `-r^{-1} mod 2^64` for Montgomery CIOS reduction in `Fr`.
pub(crate) const FR_INV: u64 = 0xffff_fffe_ffff_ffff;

const _: () = {
    assert!(FP_P[0].wrapping_mul(FP_INV) == u64::MAX);
    assert!(FR_R[0].wrapping_mul(FR_INV) == u64::MAX);
};

/// Declare scratch registers required by all inline `emit_fp_*` helpers.
fn emit_fp_scratch_decls(out: &mut String) {
    let _ = writeln!(
        out,
        "    .reg .u64 %mt<7>, %ms<6>, %mm, %mz0, %mborrow, %mor0, %mor1, %mor2;\n    .reg .pred %mge, %misz;"
    );
}

fn emit_fp_load(
    out: &mut String,
    space: &str,
    dst: &str,
    ptr: &str,
    off: usize,
) {
    for i in 0..6 {
        let _ = writeln!(
            out,
            "    ld.{space}.u64 %{dst}{i}, [%{ptr} + {}];",
            off + i * 8
        );
    }
}

fn emit_fp_store(
    out: &mut String,
    space: &str,
    ptr: &str,
    off: usize,
    src: &str,
) {
    for i in 0..6 {
        let _ = writeln!(
            out,
            "    st.{space}.u64 [%{ptr} + {}], %{src}{i};",
            off + i * 8
        );
    }
}

fn emit_fp_copy(out: &mut String, dst: &str, src: &str) {
    for i in 0..6 {
        let _ = writeln!(out, "    mov.u64 %{dst}{i}, %{src}{i};");
    }
}

fn emit_fp_set_zero(out: &mut String, dst: &str) {
    for i in 0..6 {
        let _ = writeln!(out, "    mov.u64 %{dst}{i}, 0;");
    }
}

fn emit_fp_set_one(out: &mut String, dst: &str) {
    for (i, limb) in FP_ONE_MONT.iter().enumerate() {
        let _ = writeln!(out, "    mov.u64 %{dst}{i}, 0x{limb:016x};");
    }
}

fn emit_fp_is_zero(out: &mut String, pred: &str, a: &str) {
    let _ = writeln!(
        out,
        "    or.b64 %mor0, %{a}0, %{a}1;\n    \
         or.b64 %mor1, %{a}2, %{a}3;\n    \
         or.b64 %mor2, %{a}4, %{a}5;\n    \
         or.b64 %mor0, %mor0, %mor1;\n    \
         or.b64 %mor0, %mor0, %mor2;\n    \
         setp.eq.u64 %{pred}, %mor0, 0;"
    );
}

/// Inline aliasing-safe `dst = (a + b) mod p` in registers.
fn emit_fp_add(out: &mut String, dst: &str, a: &str, b: &str) {
    let p = FP_P;
    let _ = writeln!(
        out,
        "    mov.u64 %mz0, 0;\n    \
         add.cc.u64  %mt0, %{a}0, %{b}0;\n    \
         addc.cc.u64 %mt1, %{a}1, %{b}1;\n    \
         addc.cc.u64 %mt2, %{a}2, %{b}2;\n    \
         addc.cc.u64 %mt3, %{a}3, %{b}3;\n    \
         addc.cc.u64 %mt4, %{a}4, %{b}4;\n    \
         addc.u64    %mt5, %{a}5, %{b}5;\n    \
         sub.cc.u64  %ms0, %mt0, 0x{:016x};\n    \
         subc.cc.u64 %ms1, %mt1, 0x{:016x};\n    \
         subc.cc.u64 %ms2, %mt2, 0x{:016x};\n    \
         subc.cc.u64 %ms3, %mt3, 0x{:016x};\n    \
         subc.cc.u64 %ms4, %mt4, 0x{:016x};\n    \
         subc.cc.u64 %ms5, %mt5, 0x{:016x};\n    \
         subc.u64 %mborrow, %mz0, %mz0;\n    \
         setp.eq.u64 %mge, %mborrow, 0;\n    \
         selp.u64 %{dst}0, %ms0, %mt0, %mge;\n    \
         selp.u64 %{dst}1, %ms1, %mt1, %mge;\n    \
         selp.u64 %{dst}2, %ms2, %mt2, %mge;\n    \
         selp.u64 %{dst}3, %ms3, %mt3, %mge;\n    \
         selp.u64 %{dst}4, %ms4, %mt4, %mge;\n    \
         selp.u64 %{dst}5, %ms5, %mt5, %mge;",
        p[0], p[1], p[2], p[3], p[4], p[5]
    );
}

/// Inline aliasing-safe `dst = (a - b) mod p` in registers.
fn emit_fp_sub(out: &mut String, dst: &str, a: &str, b: &str) {
    let p = FP_P;
    let _ = writeln!(
        out,
        "    mov.u64 %mz0, 0;\n    \
         sub.cc.u64  %mt0, %{a}0, %{b}0;\n    \
         subc.cc.u64 %mt1, %{a}1, %{b}1;\n    \
         subc.cc.u64 %mt2, %{a}2, %{b}2;\n    \
         subc.cc.u64 %mt3, %{a}3, %{b}3;\n    \
         subc.cc.u64 %mt4, %{a}4, %{b}4;\n    \
         subc.cc.u64 %mt5, %{a}5, %{b}5;\n    \
         subc.u64 %mborrow, %mz0, %mz0;\n    \
         add.cc.u64  %ms0, %mt0, 0x{:016x};\n    \
         addc.cc.u64 %ms1, %mt1, 0x{:016x};\n    \
         addc.cc.u64 %ms2, %mt2, 0x{:016x};\n    \
         addc.cc.u64 %ms3, %mt3, 0x{:016x};\n    \
         addc.cc.u64 %ms4, %mt4, 0x{:016x};\n    \
         addc.u64    %ms5, %mt5, 0x{:016x};\n    \
         setp.eq.u64 %mge, %mborrow, 0;\n    \
         selp.u64 %{dst}0, %mt0, %ms0, %mge;\n    \
         selp.u64 %{dst}1, %mt1, %ms1, %mge;\n    \
         selp.u64 %{dst}2, %mt2, %ms2, %mge;\n    \
         selp.u64 %{dst}3, %mt3, %ms3, %mge;\n    \
         selp.u64 %{dst}4, %mt4, %ms4, %mge;\n    \
         selp.u64 %{dst}5, %mt5, %ms5, %mge;",
        p[0], p[1], p[2], p[3], p[4], p[5]
    );
}

/// Inline aliasing-safe `dst = (-a) mod p` in registers.
fn emit_fp_neg(out: &mut String, dst: &str, a: &str) {
    let p = FP_P;
    emit_fp_is_zero(out, "misz", a);
    let _ = writeln!(
        out,
        "    sub.cc.u64  %ms0, 0x{:016x}, %{a}0;\n    \
         subc.cc.u64 %ms1, 0x{:016x}, %{a}1;\n    \
         subc.cc.u64 %ms2, 0x{:016x}, %{a}2;\n    \
         subc.cc.u64 %ms3, 0x{:016x}, %{a}3;\n    \
         subc.cc.u64 %ms4, 0x{:016x}, %{a}4;\n    \
         subc.u64    %ms5, 0x{:016x}, %{a}5;\n    \
         selp.u64 %{dst}0, 0, %ms0, %misz;\n    \
         selp.u64 %{dst}1, 0, %ms1, %misz;\n    \
         selp.u64 %{dst}2, 0, %ms2, %misz;\n    \
         selp.u64 %{dst}3, 0, %ms3, %misz;\n    \
         selp.u64 %{dst}4, 0, %ms4, %misz;\n    \
         selp.u64 %{dst}5, 0, %ms5, %misz;",
        p[0], p[1], p[2], p[3], p[4], p[5]
    );
}

/// Inline aliasing-safe 6-limb CIOS Montgomery `dst = a * b * R^-1 mod p` in
/// registers (`dst` may freely alias `a` and/or `b`).
fn emit_fp_mul(out: &mut String, dst: &str, a: &str, b: &str) {
    let p = FP_P;
    let _ = writeln!(
        out,
        "    mov.u64 %mz0, 0;\n    \
         mov.u64 %mt0, 0; mov.u64 %mt1, 0; mov.u64 %mt2, 0;\n    \
         mov.u64 %mt3, 0; mov.u64 %mt4, 0; mov.u64 %mt5, 0; mov.u64 %mt6, 0;"
    );
    for i in 0..6 {
        let _ = writeln!(
            out,
            "    mad.lo.cc.u64  %mt0, %{a}{i}, %{b}0, %mt0;\n    \
             madc.lo.cc.u64 %mt1, %{a}{i}, %{b}1, %mt1;\n    \
             madc.lo.cc.u64 %mt2, %{a}{i}, %{b}2, %mt2;\n    \
             madc.lo.cc.u64 %mt3, %{a}{i}, %{b}3, %mt3;\n    \
             madc.lo.cc.u64 %mt4, %{a}{i}, %{b}4, %mt4;\n    \
             madc.lo.cc.u64 %mt5, %{a}{i}, %{b}5, %mt5;\n    \
             addc.u64       %mt6, %mt6, 0;\n    \
             mad.hi.cc.u64  %mt1, %{a}{i}, %{b}0, %mt1;\n    \
             madc.hi.cc.u64 %mt2, %{a}{i}, %{b}1, %mt2;\n    \
             madc.hi.cc.u64 %mt3, %{a}{i}, %{b}2, %mt3;\n    \
             madc.hi.cc.u64 %mt4, %{a}{i}, %{b}3, %mt4;\n    \
             madc.hi.cc.u64 %mt5, %{a}{i}, %{b}4, %mt5;\n    \
             madc.hi.u64    %mt6, %{a}{i}, %{b}5, %mt6;\n    \
             mul.lo.u64     %mm,  %mt0, 0x{:016x};\n    \
             mad.lo.cc.u64  %mt0, %mm, 0x{:016x}, %mt0;\n    \
             madc.lo.cc.u64 %mt1, %mm, 0x{:016x}, %mt1;\n    \
             madc.lo.cc.u64 %mt2, %mm, 0x{:016x}, %mt2;\n    \
             madc.lo.cc.u64 %mt3, %mm, 0x{:016x}, %mt3;\n    \
             madc.lo.cc.u64 %mt4, %mm, 0x{:016x}, %mt4;\n    \
             madc.lo.cc.u64 %mt5, %mm, 0x{:016x}, %mt5;\n    \
             addc.u64       %mt6, %mt6, 0;\n    \
             mad.hi.cc.u64  %mt0, %mm, 0x{:016x}, %mt1;\n    \
             madc.hi.cc.u64 %mt1, %mm, 0x{:016x}, %mt2;\n    \
             madc.hi.cc.u64 %mt2, %mm, 0x{:016x}, %mt3;\n    \
             madc.hi.cc.u64 %mt3, %mm, 0x{:016x}, %mt4;\n    \
             madc.hi.cc.u64 %mt4, %mm, 0x{:016x}, %mt5;\n    \
             madc.hi.u64    %mt5, %mm, 0x{:016x}, %mt6;\n    \
             mov.u64        %mt6, 0;",
            FP_INV,
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            p[5],
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            p[5],
        );
    }
    let _ = writeln!(
        out,
        "    sub.cc.u64  %ms0, %mt0, 0x{:016x};\n    \
         subc.cc.u64 %ms1, %mt1, 0x{:016x};\n    \
         subc.cc.u64 %ms2, %mt2, 0x{:016x};\n    \
         subc.cc.u64 %ms3, %mt3, 0x{:016x};\n    \
         subc.cc.u64 %ms4, %mt4, 0x{:016x};\n    \
         subc.cc.u64 %ms5, %mt5, 0x{:016x};\n    \
         subc.u64 %mborrow, %mz0, %mz0;\n    \
         setp.eq.u64 %mge, %mborrow, 0;\n    \
         selp.u64 %{dst}0, %ms0, %mt0, %mge;\n    \
         selp.u64 %{dst}1, %ms1, %mt1, %mge;\n    \
         selp.u64 %{dst}2, %ms2, %mt2, %mge;\n    \
         selp.u64 %{dst}3, %ms3, %mt3, %mge;\n    \
         selp.u64 %{dst}4, %ms4, %mt4, %mge;\n    \
         selp.u64 %{dst}5, %ms5, %mt5, %mge;",
        p[0], p[1], p[2], p[3], p[4], p[5]
    );
}

/// Emit a register-only `.func fp_mul_val` (`.reg` state-space parameters,
/// 0 bytes `.local`, 0 pointers) used by `msm_merge_split_buckets` and
/// `msm_reduce_windows` to keep PTX compact so `ptxas` JIT-compiles in ~0.2s.
fn emit_fp_mul_val_func(out: &mut String) {
    let _ = writeln!(
        out,
        r#"
.func (
    .reg .u64 %r0, .reg .u64 %r1, .reg .u64 %r2,
    .reg .u64 %r3, .reg .u64 %r4, .reg .u64 %r5
) fp_mul_val(
    .reg .u64 %a0, .reg .u64 %a1, .reg .u64 %a2,
    .reg .u64 %a3, .reg .u64 %a4, .reg .u64 %a5,
    .reg .u64 %b0, .reg .u64 %b1, .reg .u64 %b2,
    .reg .u64 %b3, .reg .u64 %b4, .reg .u64 %b5
)
{{"#
    );
    emit_fp_scratch_decls(out);
    emit_fp_mul(out, "r", "a", "b");
    let _ = writeln!(out, "    ret;\n}}");
}

/// Call `fp_mul_val` passing 6-limb registers by value (0 `.local` memory).
fn emit_fp_mul_call(out: &mut String, dst: &str, a: &str, b: &str) {
    let _ = writeln!(
        out,
        "    call (%{dst}0, %{dst}1, %{dst}2, %{dst}3, %{dst}4, %{dst}5), \
         fp_mul_val, (\
         %{a}0, %{a}1, %{a}2, %{a}3, %{a}4, %{a}5, \
         %{b}0, %{b}1, %{b}2, %{b}3, %{b}4, %{b}5);"
    );
}

/// Inline in-place XYZZ doubling (`dbl-2008-s-1` for `a = 0`) using registers
/// `ax, ay, azz, azzz` and 5 temp `Fp` register banks `ta..te`.
fn emit_xyzz_dbl_inline(
    out: &mut String,
    lbl: &str,
    ax: &str,
    ay: &str,
    azz: &str,
    azzz: &str,
    use_call: bool,
) {
    let mul = if use_call {
        emit_fp_mul_call
    } else {
        emit_fp_mul
    };
    emit_fp_is_zero(out, "pz0", azz);
    let _ = writeln!(out, "    @%pz0 bra L_{lbl}_DONE;");
    // ta = U = 2 * Y
    emit_fp_add(out, "ta", ay, ay);
    // tb = V = U^2
    mul(out, "tb", "ta", "ta");
    // tc = W = U * V
    mul(out, "tc", "ta", "tb");
    // td = S = X * V
    mul(out, "td", ax, "tb");
    // te = M = 3 * X^2
    mul(out, "ta", ax, ax);
    emit_fp_add(out, "te", "ta", "ta");
    emit_fp_add(out, "te", "te", "ta");
    // X3 = M^2 - 2 * S
    mul(out, "ta", "te", "te");
    emit_fp_add(out, ax, "td", "td");
    emit_fp_sub(out, ax, "ta", ax);
    // Y3 = M * (S - X3) - W * Y
    emit_fp_sub(out, "ta", "td", ax);
    mul(out, "ta", "te", "ta");
    mul(out, "td", "tc", ay);
    emit_fp_sub(out, ay, "ta", "td");
    // ZZ3 = V * ZZ, ZZZ3 = W * ZZZ
    mul(out, azz, "tb", azz);
    mul(out, azzz, "tc", azzz);
    let _ = writeln!(out, "L_{lbl}_DONE:");
}

/// Inline in-place XYZZ + Affine mixed addition (`madd-2008-s` with complete
/// infinity, cancellation, and doubling handling) in registers.
fn emit_xyzz_add_affine_inline(
    out: &mut String,
    lbl: &str,
    ax: &str,
    ay: &str,
    azz: &str,
    azzz: &str,
    bx: &str,
    by: &str,
) {
    // If aff == (0, 0) (infinity), skip.
    emit_fp_is_zero(out, "pz0", bx);
    emit_fp_is_zero(out, "pz1", by);
    let _ = writeln!(
        out,
        "    and.pred %pz0, %pz0, %pz1;\n    @%pz0 bra L_{lbl}_DONE;"
    );

    // If acc == inf (azz == 0), copy aff with ZZ = ZZZ = 1_mont.
    emit_fp_is_zero(out, "pz0", azz);
    let _ = writeln!(out, "    @!%pz0 bra L_{lbl}_NONINF;");
    emit_fp_copy(out, ax, bx);
    emit_fp_copy(out, ay, by);
    emit_fp_set_one(out, azz);
    emit_fp_set_one(out, azzz);
    let _ = writeln!(out, "    bra L_{lbl}_DONE;");

    let _ = writeln!(out, "L_{lbl}_NONINF:");
    // ta = U2 = bx * azz, tb = S2 = by * azzz
    emit_fp_mul(out, "ta", bx, azz);
    emit_fp_mul(out, "tb", by, azzz);
    // ta = P = U2 - ax, tb = R = S2 - ay
    emit_fp_sub(out, "ta", "ta", ax);
    emit_fp_sub(out, "tb", "tb", ay);

    emit_fp_is_zero(out, "pz0", "ta");
    let _ = writeln!(out, "    @!%pz0 bra L_{lbl}_GEN;");

    // P == 0: R != 0 -> cancellation (inf), R == 0 -> doubling.
    emit_fp_is_zero(out, "pz1", "tb");
    let _ = writeln!(out, "    @!%pz1 bra L_{lbl}_CANCEL;");
    emit_xyzz_dbl_inline(out, &format!("{lbl}_DBL"), ax, ay, azz, azzz, true);
    let _ = writeln!(out, "    bra L_{lbl}_DONE;");

    let _ = writeln!(out, "L_{lbl}_CANCEL:");
    emit_fp_set_zero(out, ax);
    emit_fp_set_zero(out, ay);
    emit_fp_set_zero(out, azz);
    emit_fp_set_zero(out, azzz);
    let _ = writeln!(out, "    bra L_{lbl}_DONE;");

    let _ = writeln!(out, "L_{lbl}_GEN:");
    // tc = PP = P^2, td = PPP = P * PP, te = Q = ax * PP
    emit_fp_mul(out, "tc", "ta", "ta");
    emit_fp_mul(out, "td", "ta", "tc");
    emit_fp_mul(out, "te", ax, "tc");
    // X3 = R^2 - PPP - 2 * Q
    emit_fp_mul(out, "ta", "tb", "tb");
    emit_fp_sub(out, "ta", "ta", "td");
    emit_fp_add(out, ax, "te", "te");
    emit_fp_sub(out, ax, "ta", ax);
    // Y3 = R * (Q - X3) - ay * PPP
    emit_fp_sub(out, "ta", "te", ax);
    emit_fp_mul(out, "ta", "tb", "ta");
    emit_fp_mul(out, "te", ay, "td");
    emit_fp_sub(out, ay, "ta", "te");
    // ZZ3 = azz * PP, ZZZ3 = azzz * PPP
    emit_fp_mul(out, azz, azz, "tc");
    emit_fp_mul(out, azzz, azzz, "td");

    let _ = writeln!(out, "L_{lbl}_DONE:");
}

/// Inline in-place XYZZ + XYZZ addition (`add-2008-s` with complete infinity,
/// cancellation, and doubling handling) using `fp_mul_val` calls to keep PTX
/// compact and JIT compilation instantaneous.
///
/// Note: overwrites `bx` and `by` as dead scratch in the general branch.
#[allow(clippy::too_many_arguments)]
fn emit_xyzz_add_inline(
    out: &mut String,
    lbl: &str,
    ax: &str,
    ay: &str,
    azz: &str,
    azzz: &str,
    bx: &str,
    by: &str,
    bzz: &str,
    bzzz: &str,
) {
    // If b == inf (bzz == 0), skip.
    emit_fp_is_zero(out, "pz0", bzz);
    let _ = writeln!(out, "    @%pz0 bra L_{lbl}_DONE;");

    // If a == inf (azz == 0), copy b -> a.
    emit_fp_is_zero(out, "pz0", azz);
    let _ = writeln!(out, "    @!%pz0 bra L_{lbl}_NONINF;");
    emit_fp_copy(out, ax, bx);
    emit_fp_copy(out, ay, by);
    emit_fp_copy(out, azz, bzz);
    emit_fp_copy(out, azzz, bzzz);
    let _ = writeln!(out, "    bra L_{lbl}_DONE;");

    let _ = writeln!(out, "L_{lbl}_NONINF:");
    // ta = U1 = ax * bzz, tb = U2 = bx * azz
    emit_fp_mul_call(out, "ta", ax, bzz);
    emit_fp_mul_call(out, "tb", bx, azz);
    // tc = S1 = ay * bzzz, td = S2 = by * azzz
    emit_fp_mul_call(out, "tc", ay, bzzz);
    emit_fp_mul_call(out, "td", by, azzz);
    // tb = P = U2 - U1, td = R = S2 - S1
    emit_fp_sub(out, "tb", "tb", "ta");
    emit_fp_sub(out, "td", "td", "tc");

    emit_fp_is_zero(out, "pz0", "tb");
    let _ = writeln!(out, "    @!%pz0 bra L_{lbl}_GEN;");

    // P == 0: R != 0 -> cancellation (inf), R == 0 -> doubling.
    emit_fp_is_zero(out, "pz1", "td");
    let _ = writeln!(out, "    @!%pz1 bra L_{lbl}_CANCEL;");
    emit_xyzz_dbl_inline(out, &format!("{lbl}_DBL"), ax, ay, azz, azzz, true);
    let _ = writeln!(out, "    bra L_{lbl}_DONE;");

    let _ = writeln!(out, "L_{lbl}_CANCEL:");
    emit_fp_set_zero(out, ax);
    emit_fp_set_zero(out, ay);
    emit_fp_set_zero(out, azz);
    emit_fp_set_zero(out, azzz);
    let _ = writeln!(out, "    bra L_{lbl}_DONE;");

    let _ = writeln!(out, "L_{lbl}_GEN:");
    // te = PP = P^2, tb = PPP = P * PP, ta = Q = U1 * PP
    emit_fp_mul_call(out, "te", "tb", "tb");
    emit_fp_mul_call(out, "tb", "tb", "te");
    emit_fp_mul_call(out, "ta", "ta", "te");
    // X3 = R^2 - PPP - 2 * Q (reuse dead bx, by as scratch)
    emit_fp_mul_call(out, bx, "td", "td");
    emit_fp_sub(out, bx, bx, "tb");
    emit_fp_add(out, by, "ta", "ta");
    emit_fp_sub(out, ax, bx, by);
    // Y3 = R * (Q - X3) - S1 * PPP
    emit_fp_sub(out, bx, "ta", ax);
    emit_fp_mul_call(out, bx, "td", bx);
    emit_fp_mul_call(out, by, "tc", "tb");
    emit_fp_sub(out, ay, bx, by);
    // ZZ3 = azz * bzz * PP, ZZZ3 = azzz * bzzz * PPP
    emit_fp_mul_call(out, azz, azz, bzz);
    emit_fp_mul_call(out, azz, azz, "te");
    emit_fp_mul_call(out, azzz, azzz, bzzz);
    emit_fp_mul_call(out, azzz, azzz, "tb");

    let _ = writeln!(out, "L_{lbl}_DONE:");
}

/// Emit inline 4-limb `Fr` Montgomery reduction by `1` on `%sc0..%sc3`.
fn emit_fr_from_mont_inline(out: &mut String) {
    let r = FR_R;
    let _ = writeln!(out, "    mov.u64 %fz0, 0;\n    mov.u64 %fc4, 0;");
    for _ in 0..4 {
        let _ = writeln!(
            out,
            "    mul.lo.u64 %fm, %sc0, 0x{:016x};\n    \
             mad.lo.cc.u64 %sc0, %fm, 0x{:016x}, %sc0;\n    \
             madc.lo.cc.u64 %sc1, %fm, 0x{:016x}, %sc1;\n    \
             madc.lo.cc.u64 %sc2, %fm, 0x{:016x}, %sc2;\n    \
             madc.lo.cc.u64 %sc3, %fm, 0x{:016x}, %sc3;\n    \
             addc.u64 %fc4, %fc4, 0;\n    \
             mad.hi.cc.u64 %sc0, %fm, 0x{:016x}, %sc1;\n    \
             madc.hi.cc.u64 %sc1, %fm, 0x{:016x}, %sc2;\n    \
             madc.hi.cc.u64 %sc2, %fm, 0x{:016x}, %sc3;\n    \
             madc.hi.u64 %sc3, %fm, 0x{:016x}, %fc4;\n    \
             mov.u64 %fc4, 0;",
            FR_INV, r[0], r[1], r[2], r[3], r[0], r[1], r[2], r[3],
        );
    }
    let _ = writeln!(
        out,
        "    sub.cc.u64 %fs0, %sc0, 0x{:016x};\n    \
         subc.cc.u64 %fs1, %sc1, 0x{:016x};\n    \
         subc.cc.u64 %fs2, %sc2, 0x{:016x};\n    \
         subc.cc.u64 %fs3, %sc3, 0x{:016x};\n    \
         subc.u64 %fborrow, %fz0, %fz0;\n    \
         setp.eq.u64 %fge, %fborrow, 0;\n    \
         selp.u64 %sc0, %fs0, %sc0, %fge;\n    \
         selp.u64 %sc1, %fs1, %sc1, %fge;\n    \
         selp.u64 %sc2, %fs2, %sc2, %fge;\n    \
         selp.u64 %sc3, %fs3, %sc3, %fge;",
        r[0], r[1], r[2], r[3]
    );
}

/// Emit inline branchless register-only Booth digit extraction from `%sc0..%sc3`
/// at bit offset `%bit0` with window width `%w`, setting `%mag` and `%neg`.
fn emit_booth_digit_inline(out: &mut String, lbl: &str) {
    let _ = writeln!(
        out,
        r#"    setp.eq.u32 %bp0, %bit0, 0;
    @%bp0 bra L_{lbl}_BZERO;
    sub.u32 %bstart, %bit0, 1;
    shr.u32 %blimb, %bstart, 6;
    and.b32 %boff, %bstart, 63;
    setp.eq.u32 %bp0, %blimb, 0;
    setp.eq.u32 %bp1, %blimb, 1;
    setp.eq.u32 %bp2, %blimb, 2;
    setp.eq.u32 %bp3, %blimb, 3;
    mov.u64 %blo, 0;
    selp.u64 %blo, %sc0, %blo, %bp0;
    selp.u64 %blo, %sc1, %blo, %bp1;
    selp.u64 %blo, %sc2, %blo, %bp2;
    selp.u64 %blo, %sc3, %blo, %bp3;
    mov.u64 %bhi, 0;
    selp.u64 %bhi, %sc1, %bhi, %bp0;
    selp.u64 %bhi, %sc2, %bhi, %bp1;
    selp.u64 %bhi, %sc3, %bhi, %bp2;
    shr.u64 %bsh, %blo, %boff;
    setp.ne.u32 %bp0, %boff, 0;
    @!%bp0 bra L_{lbl}_BSHIFTED;
    sub.u32 %brem, 64, %boff;
    shl.b64 %bhi, %bhi, %brem;
    or.b64 %bsh, %bsh, %bhi;
L_{lbl}_BSHIFTED:
    cvt.u32.u64 %braw, %bsh;
    and.b32 %braw, %braw, %bmask;
    bra L_{lbl}_BDECODE;
L_{lbl}_BZERO:
    cvt.u32.u64 %braw, %sc0;
    shl.b32 %braw, %braw, 1;
    and.b32 %braw, %braw, %bmask;
L_{lbl}_BDECODE:
    shr.u32 %neg, %braw, %w;
    and.b32 %neg, %neg, 1;
    setp.ne.u32 %bp0, %neg, 0;
    add.u32 %bpos_m, %braw, 1;
    shr.u32 %bpos_m, %bpos_m, 1;
    sub.u32 %bneg_m, %btwow1, %braw;
    shr.u32 %bneg_m, %bneg_m, 1;
    selp.u32 %mag, %bneg_m, %bpos_m, %bp0;"#
    );
}

/// Emit the 4 register-resident GPU Pippenger MSM kernels plus unit-test
/// helper kernel (`test_fp_ops`).
fn emit_msm_kernels(out: &mut String) {
    // Kernel 1: msm_count_buckets (0 bytes .local, 0 calls)
    let _ = writeln!(
        out,
        r#"
.visible .entry msm_count_buckets(
    .param .u64 p_scalars_mont,
    .param .u64 p_scalars_std,
    .param .u64 p_counts,
    .param .u32 p_msm_len,
    .param .u32 p_total_scalars,
    .param .u32 p_w,
    .param .u32 p_ny,
    .param .u32 p_num_buckets
)
{{
    .reg .u32 %r_tid, %r_ctaid, %r_ntid, %total, %msm_len, %w, %w1, %bmask, %btwow1, %ny, %nb, %poly, %wi, %bit0, %mag, %neg, %bidx, %old;
    .reg .u32 %bstart, %blimb, %boff, %brem, %braw, %bpos_m, %bneg_m;
    .reg .u64 %sm_ptr, %ss_ptr, %cnt_ptr, %off, %addr, %sc<4>, %orv, %fz0, %fc4, %fm, %fs<4>, %fborrow, %blo, %bhi, %bsh;
    .reg .pred %oob, %is_z, %fge, %loop_p, %has_mag, %bp<4>;

    mov.u32 %r_ctaid, %ctaid.x;
    mov.u32 %r_ntid, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %r_ctaid, %r_ntid, %r_tid;

    ld.param.u32 %total, [p_total_scalars];
    setp.ge.u32 %oob, %r_tid, %total;
    @%oob ret;

    ld.param.u64 %sm_ptr, [p_scalars_mont];
    ld.param.u64 %ss_ptr, [p_scalars_std];
    ld.param.u64 %cnt_ptr, [p_counts];
    ld.param.u32 %msm_len, [p_msm_len];
    ld.param.u32 %w, [p_w];
    ld.param.u32 %ny, [p_ny];
    ld.param.u32 %nb, [p_num_buckets];

    mul.wide.u32 %off, %r_tid, 32;
    add.u64 %sm_ptr, %sm_ptr, %off;
    add.u64 %ss_ptr, %ss_ptr, %off;

    ld.global.u64 %sc0, [%sm_ptr + 0];
    ld.global.u64 %sc1, [%sm_ptr + 8];
    ld.global.u64 %sc2, [%sm_ptr + 16];
    ld.global.u64 %sc3, [%sm_ptr + 24];
    or.b64 %orv, %sc0, %sc1;
    or.b64 %orv, %orv, %sc2;
    or.b64 %orv, %orv, %sc3;
    setp.eq.u64 %is_z, %orv, 0;
    @!%is_z bra L_CNT_NONZERO;
    mov.u64 %fz0, 0;
    st.global.u64 [%ss_ptr + 0], %fz0;
    st.global.u64 [%ss_ptr + 8], %fz0;
    st.global.u64 [%ss_ptr + 16], %fz0;
    st.global.u64 [%ss_ptr + 24], %fz0;
    ret;

L_CNT_NONZERO:"#
    );
    emit_fr_from_mont_inline(out);
    let _ = writeln!(
        out,
        r#"    st.global.u64 [%ss_ptr + 0], %sc0;
    st.global.u64 [%ss_ptr + 8], %sc1;
    st.global.u64 [%ss_ptr + 16], %sc2;
    st.global.u64 [%ss_ptr + 24], %sc3;

    add.u32 %w1, %w, 1;
    mov.u32 %btwow1, 1;
    shl.b32 %btwow1, %btwow1, %w1;
    sub.u32 %bmask, %btwow1, 1;

    div.u32 %poly, %r_tid, %msm_len;
    mov.u32 %wi, 0;
    mov.u32 %bit0, 0;
L_CNT_LOOP:
    setp.ge.u32 %loop_p, %wi, %ny;
    @%loop_p ret;"#
    );
    emit_booth_digit_inline(out, "CNT");
    let _ = writeln!(
        out,
        r#"    setp.ne.u32 %has_mag, %mag, 0;
    @!%has_mag bra L_CNT_NEXT;
    mad.lo.u32 %bidx, %poly, %ny, %wi;
    mad.lo.u32 %bidx, %bidx, %nb, %mag;
    sub.u32 %bidx, %bidx, 1;
    mul.wide.u32 %off, %bidx, 4;
    add.u64 %addr, %cnt_ptr, %off;
    atom.global.add.u32 %old, [%addr], 1;
L_CNT_NEXT:
    add.u32 %wi, %wi, 1;
    add.u32 %bit0, %bit0, %w;
    bra L_CNT_LOOP;
}}"#
    );

    // Kernel 2: msm_scatter_indices (0 bytes .local, 0 calls)
    let _ = writeln!(
        out,
        r#"
.visible .entry msm_scatter_indices(
    .param .u64 p_scalars_std,
    .param .u64 p_cursors,
    .param .u64 p_sorted,
    .param .u32 p_msm_len,
    .param .u32 p_total_scalars,
    .param .u32 p_w,
    .param .u32 p_ny,
    .param .u32 p_num_buckets
)
{{
    .reg .u32 %r_tid, %r_ctaid, %r_ntid, %total, %msm_len, %w, %w1, %bmask, %btwow1, %ny, %nb, %poly, %pidx, %wi, %bit0, %mag, %neg, %bidx, %pos, %val;
    .reg .u32 %bstart, %blimb, %boff, %brem, %braw, %bpos_m, %bneg_m;
    .reg .u64 %ss_ptr, %cur_ptr, %srt_ptr, %off, %addr, %sc<4>, %orv, %blo, %bhi, %bsh;
    .reg .pred %oob, %is_z, %loop_p, %has_mag, %bp<4>;

    mov.u32 %r_ctaid, %ctaid.x;
    mov.u32 %r_ntid, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %r_ctaid, %r_ntid, %r_tid;

    ld.param.u32 %total, [p_total_scalars];
    setp.ge.u32 %oob, %r_tid, %total;
    @%oob ret;

    ld.param.u64 %ss_ptr, [p_scalars_std];
    ld.param.u64 %cur_ptr, [p_cursors];
    ld.param.u64 %srt_ptr, [p_sorted];
    ld.param.u32 %msm_len, [p_msm_len];
    ld.param.u32 %w, [p_w];
    ld.param.u32 %ny, [p_ny];
    ld.param.u32 %nb, [p_num_buckets];

    mul.wide.u32 %off, %r_tid, 32;
    add.u64 %ss_ptr, %ss_ptr, %off;
    ld.global.u64 %sc0, [%ss_ptr + 0];
    ld.global.u64 %sc1, [%ss_ptr + 8];
    ld.global.u64 %sc2, [%ss_ptr + 16];
    ld.global.u64 %sc3, [%ss_ptr + 24];
    or.b64 %orv, %sc0, %sc1;
    or.b64 %orv, %orv, %sc2;
    or.b64 %orv, %orv, %sc3;
    setp.eq.u64 %is_z, %orv, 0;
    @%is_z ret;

    add.u32 %w1, %w, 1;
    mov.u32 %btwow1, 1;
    shl.b32 %btwow1, %btwow1, %w1;
    sub.u32 %bmask, %btwow1, 1;

    div.u32 %poly, %r_tid, %msm_len;
    rem.u32 %pidx, %r_tid, %msm_len;
    mov.u32 %wi, 0;
    mov.u32 %bit0, 0;
L_SCT_LOOP:
    setp.ge.u32 %loop_p, %wi, %ny;
    @%loop_p ret;"#
    );
    emit_booth_digit_inline(out, "SCT");
    let _ = writeln!(
        out,
        r#"    setp.ne.u32 %has_mag, %mag, 0;
    @!%has_mag bra L_SCT_NEXT;
    shl.b32 %neg, %neg, 31;
    or.b32 %val, %pidx, %neg;
    mad.lo.u32 %bidx, %poly, %ny, %wi;
    mad.lo.u32 %bidx, %bidx, %nb, %mag;
    sub.u32 %bidx, %bidx, 1;
    mul.wide.u32 %off, %bidx, 4;
    add.u64 %addr, %cur_ptr, %off;
    atom.global.add.u32 %pos, [%addr], 1;
    mul.wide.u32 %off, %pos, 4;
    add.u64 %addr, %srt_ptr, %off;
    st.global.u32 [%addr], %val;
L_SCT_NEXT:
    add.u32 %wi, %wi, 1;
    add.u32 %bit0, %bit0, %w;
    bra L_SCT_LOOP;
}}"#
    );

    // Kernel 2a: msm_scan_blocks — per-block reduction of (entries, tasks, merges, tmps).
    let _ = writeln!(
        out,
        r#"
.visible .entry msm_scan_blocks(
    .param .u64 p_counts,
    .param .u64 p_block_totals,
    .param .u32 p_total_buckets,
    .param .u32 p_max_chunk
)
{{
    .shared .align 4 .b8 sh_sb[4096];
    .reg .u32 %t, %bid, %b, %tb, %mc, %c, %nt, %nm, %ntmp, %tmp, %stride, %peer;
    .reg .u32 %pe, %ptk, %pmg, %ptp;
    .reg .u64 %counts, %btot, %off, %addr, %sh_base, %sh_p, %sh_q;
    .reg .pred %p_oob, %p0, %p1, %p_act, %p_loop, %is_t0;

    mov.u32 %t, %tid.x;
    mov.u32 %bid, %ctaid.x;
    shl.b32 %b, %bid, 8;
    add.u32 %b, %b, %t;

    ld.param.u64 %counts, [p_counts];
    ld.param.u64 %btot, [p_block_totals];
    ld.param.u32 %tb, [p_total_buckets];
    ld.param.u32 %mc, [p_max_chunk];

    mov.u32 %c, 0;
    mov.u32 %nt, 0;
    mov.u32 %nm, 0;
    mov.u32 %ntmp, 0;
    setp.ge.u32 %p_oob, %b, %tb;
    @%p_oob bra L_SB_STOR;
    mul.wide.u32 %off, %b, 4;
    add.u64 %addr, %counts, %off;
    ld.global.u32 %c, [%addr];
    setp.eq.u32 %p0, %c, 0;
    @%p0 bra L_SB_STOR;
    setp.le.u32 %p1, %c, %mc;
    @!%p1 bra L_SB_SPLIT;
    mov.u32 %nt, 1;
    bra L_SB_STOR;
L_SB_SPLIT:
    add.u32 %tmp, %c, %mc;
    sub.u32 %tmp, %tmp, 1;
    div.u32 %nt, %tmp, %mc;
    mov.u32 %nm, 1;
    mov.u32 %ntmp, %nt;
L_SB_STOR:
    mov.u64 %sh_base, sh_sb;
    mul.wide.u32 %off, %t, 4;
    add.u64 %sh_p, %sh_base, %off;
    st.shared.u32 [%sh_p + 0], %c;
    st.shared.u32 [%sh_p + 1024], %nt;
    st.shared.u32 [%sh_p + 2048], %nm;
    st.shared.u32 [%sh_p + 3072], %ntmp;
    bar.sync 0;

    mov.u32 %stride, 128;
L_SB_RED:
    setp.eq.u32 %p_loop, %stride, 0;
    @%p_loop bra L_SB_OUT;
    setp.lt.u32 %p_act, %t, %stride;
    @!%p_act bra L_SB_SYNC;
    add.u32 %peer, %t, %stride;
    mul.wide.u32 %off, %peer, 4;
    add.u64 %sh_q, %sh_base, %off;
    ld.shared.u32 %pe, [%sh_q + 0];
    ld.shared.u32 %ptk, [%sh_q + 1024];
    ld.shared.u32 %pmg, [%sh_q + 2048];
    ld.shared.u32 %ptp, [%sh_q + 3072];
    add.u32 %c, %c, %pe;
    add.u32 %nt, %nt, %ptk;
    add.u32 %nm, %nm, %pmg;
    add.u32 %ntmp, %ntmp, %ptp;
    st.shared.u32 [%sh_p + 0], %c;
    st.shared.u32 [%sh_p + 1024], %nt;
    st.shared.u32 [%sh_p + 2048], %nm;
    st.shared.u32 [%sh_p + 3072], %ntmp;
L_SB_SYNC:
    bar.sync 0;
    shr.u32 %stride, %stride, 1;
    bra L_SB_RED;

L_SB_OUT:
    setp.eq.u32 %is_t0, %t, 0;
    @!%is_t0 ret;
    mul.wide.u32 %off, %bid, 16;
    add.u64 %addr, %btot, %off;
    st.global.u32 [%addr + 0], %c;
    st.global.u32 [%addr + 4], %nt;
    st.global.u32 [%addr + 8], %nm;
    st.global.u32 [%addr + 12], %ntmp;
    ret;
}}

// Kernel 2b: msm_scan_block_totals - single-block tiled exclusive scan over block totals.
.visible .entry msm_scan_block_totals(
    .param .u64 p_block_totals,
    .param .u64 p_block_offsets,
    .param .u64 p_totals_out,
    .param .u32 p_num_blocks,
    .param .u32 p_total_buckets
)
{{
    .shared .align 4 .b8 sh_bt[4096];
    .reg .u32 %t, %nb, %tb, %tile, %idx, %step, %peer;
    .reg .u32 %re, %rtk, %rmg, %rtp;
    .reg .u32 %ve, %vtk, %vmg, %vtp;
    .reg .u32 %se, %stk, %smg, %stp;
    .reg .u32 %pe, %ptk, %pmg, %ptp;
    .reg .u64 %btot, %boff, %tout, %off, %addr, %sh_base, %sh_p, %sh_q;
    .reg .pred %p_done, %p_in, %p_step, %p_ge, %is_t0;

    mov.u32 %t, %tid.x;
    ld.param.u64 %btot, [p_block_totals];
    ld.param.u64 %boff, [p_block_offsets];
    ld.param.u64 %tout, [p_totals_out];
    ld.param.u32 %nb, [p_num_blocks];
    ld.param.u32 %tb, [p_total_buckets];

    mov.u64 %sh_base, sh_bt;
    mul.wide.u32 %off, %t, 4;
    add.u64 %sh_p, %sh_base, %off;

    mov.u32 %re, 0;
    mov.u32 %rtk, 0;
    mov.u32 %rmg, 0;
    mov.u32 %rtp, %tb;
    mov.u32 %tile, 0;

L_BT_TILE:
    setp.ge.u32 %p_done, %tile, %nb;
    @%p_done bra L_BT_FIN;

    add.u32 %idx, %tile, %t;
    mov.u32 %ve, 0;
    mov.u32 %vtk, 0;
    mov.u32 %vmg, 0;
    mov.u32 %vtp, 0;
    setp.lt.u32 %p_in, %idx, %nb;
    @!%p_in bra L_BT_LD_DONE;
    mul.wide.u32 %off, %idx, 16;
    add.u64 %addr, %btot, %off;
    ld.global.u32 %ve, [%addr + 0];
    ld.global.u32 %vtk, [%addr + 4];
    ld.global.u32 %vmg, [%addr + 8];
    ld.global.u32 %vtp, [%addr + 12];
L_BT_LD_DONE:
    mov.u32 %se, %ve;
    mov.u32 %stk, %vtk;
    mov.u32 %smg, %vmg;
    mov.u32 %stp, %vtp;
    st.shared.u32 [%sh_p + 0], %se;
    st.shared.u32 [%sh_p + 1024], %stk;
    st.shared.u32 [%sh_p + 2048], %smg;
    st.shared.u32 [%sh_p + 3072], %stp;
    bar.sync 0;

    mov.u32 %step, 1;
L_BT_HS:
    setp.ge.u32 %p_step, %step, 256;
    @%p_step bra L_BT_WR;
    mov.u32 %pe, 0;
    mov.u32 %ptk, 0;
    mov.u32 %pmg, 0;
    mov.u32 %ptp, 0;
    setp.ge.u32 %p_ge, %t, %step;
    @!%p_ge bra L_BT_HS_SYNC;
    sub.u32 %peer, %t, %step;
    mul.wide.u32 %off, %peer, 4;
    add.u64 %sh_q, %sh_base, %off;
    ld.shared.u32 %pe, [%sh_q + 0];
    ld.shared.u32 %ptk, [%sh_q + 1024];
    ld.shared.u32 %pmg, [%sh_q + 2048];
    ld.shared.u32 %ptp, [%sh_q + 3072];
L_BT_HS_SYNC:
    bar.sync 0;
    add.u32 %se, %se, %pe;
    add.u32 %stk, %stk, %ptk;
    add.u32 %smg, %smg, %pmg;
    add.u32 %stp, %stp, %ptp;
    st.shared.u32 [%sh_p + 0], %se;
    st.shared.u32 [%sh_p + 1024], %stk;
    st.shared.u32 [%sh_p + 2048], %smg;
    st.shared.u32 [%sh_p + 3072], %stp;
    bar.sync 0;
    shl.b32 %step, %step, 1;
    bra L_BT_HS;

L_BT_WR:
    @!%p_in bra L_BT_ADV;
    sub.u32 %pe, %se, %ve;
    sub.u32 %ptk, %stk, %vtk;
    sub.u32 %pmg, %smg, %vmg;
    sub.u32 %ptp, %stp, %vtp;
    add.u32 %pe, %pe, %re;
    add.u32 %ptk, %ptk, %rtk;
    add.u32 %pmg, %pmg, %rmg;
    add.u32 %ptp, %ptp, %rtp;
    mul.wide.u32 %off, %idx, 16;
    add.u64 %addr, %boff, %off;
    st.global.u32 [%addr + 0], %pe;
    st.global.u32 [%addr + 4], %ptk;
    st.global.u32 [%addr + 8], %pmg;
    st.global.u32 [%addr + 12], %ptp;
L_BT_ADV:
    ld.shared.u32 %pe, [%sh_base + 1020];
    ld.shared.u32 %ptk, [%sh_base + 2044];
    ld.shared.u32 %pmg, [%sh_base + 3068];
    ld.shared.u32 %ptp, [%sh_base + 4092];
    add.u32 %re, %re, %pe;
    add.u32 %rtk, %rtk, %ptk;
    add.u32 %rmg, %rmg, %pmg;
    add.u32 %rtp, %rtp, %ptp;
    bar.sync 0;
    add.u32 %tile, %tile, 256;
    bra L_BT_TILE;

L_BT_FIN:
    setp.eq.u32 %is_t0, %t, 0;
    @!%is_t0 ret;
    st.global.u32 [%tout + 0], %re;
    st.global.u32 [%tout + 4], %rtk;
    st.global.u32 [%tout + 8], %rmg;
    st.global.u32 [%tout + 12], %rtp;
    ret;
}}

// Kernel 2c: msm_build_tasks - writes cursors, tasks, merges, and zeros empty
// bucket `zz` limbs directly in VRAM (eliminating full buckets memset!).
.visible .entry msm_build_tasks(
    .param .u64 p_counts,
    .param .u64 p_block_offsets,
    .param .u64 p_cursors,
    .param .u64 p_tasks,
    .param .u64 p_merges,
    .param .u64 p_buckets,
    .param .u32 p_total_buckets,
    .param .u32 p_max_chunk
)
{{
    .shared .align 4 .b8 sh_mk[4096];
    .reg .u32 %t, %bid, %b, %tb, %mc, %c, %nt, %nm, %ntmp, %tmp, %step, %peer;
    .reg .u32 %se, %stk, %smg, %stp, %pe, %ptk, %pmg, %ptp;
    .reg .u32 %start, %end_b, %t_off, %m_off, %tmp_off, %k, %sk, %ek, %dk;
    .reg .u64 %counts, %boff, %cursors, %tasks, %merges, %buckets;
    .reg .u64 %off, %addr, %sh_base, %sh_p, %sh_q;
    .reg .pred %p_oob, %p0, %p1, %p_step, %p_ge, %p_kdone;

    mov.u32 %t, %tid.x;
    mov.u32 %bid, %ctaid.x;
    shl.b32 %b, %bid, 8;
    add.u32 %b, %b, %t;

    ld.param.u64 %counts, [p_counts];
    ld.param.u64 %boff, [p_block_offsets];
    ld.param.u64 %cursors, [p_cursors];
    ld.param.u64 %tasks, [p_tasks];
    ld.param.u64 %merges, [p_merges];
    ld.param.u64 %buckets, [p_buckets];
    ld.param.u32 %tb, [p_total_buckets];
    ld.param.u32 %mc, [p_max_chunk];

    mov.u32 %c, 0;
    mov.u32 %nt, 0;
    mov.u32 %nm, 0;
    mov.u32 %ntmp, 0;
    setp.ge.u32 %p_oob, %b, %tb;
    @%p_oob bra L_MK_STOR;
    mul.wide.u32 %off, %b, 4;
    add.u64 %addr, %counts, %off;
    ld.global.u32 %c, [%addr];
    setp.eq.u32 %p0, %c, 0;
    @%p0 bra L_MK_STOR;
    setp.le.u32 %p1, %c, %mc;
    @!%p1 bra L_MK_SPLIT;
    mov.u32 %nt, 1;
    bra L_MK_STOR;
L_MK_SPLIT:
    add.u32 %tmp, %c, %mc;
    sub.u32 %tmp, %tmp, 1;
    div.u32 %nt, %tmp, %mc;
    mov.u32 %nm, 1;
    mov.u32 %ntmp, %nt;
L_MK_STOR:
    mov.u64 %sh_base, sh_mk;
    mul.wide.u32 %off, %t, 4;
    add.u64 %sh_p, %sh_base, %off;
    mov.u32 %se, %c;
    mov.u32 %stk, %nt;
    mov.u32 %smg, %nm;
    mov.u32 %stp, %ntmp;
    st.shared.u32 [%sh_p + 0], %se;
    st.shared.u32 [%sh_p + 1024], %stk;
    st.shared.u32 [%sh_p + 2048], %smg;
    st.shared.u32 [%sh_p + 3072], %stp;
    bar.sync 0;

    mov.u32 %step, 1;
L_MK_HS:
    setp.ge.u32 %p_step, %step, 256;
    @%p_step bra L_MK_EMIT;
    mov.u32 %pe, 0;
    mov.u32 %ptk, 0;
    mov.u32 %pmg, 0;
    mov.u32 %ptp, 0;
    setp.ge.u32 %p_ge, %t, %step;
    @!%p_ge bra L_MK_HS_SYNC;
    sub.u32 %peer, %t, %step;
    mul.wide.u32 %off, %peer, 4;
    add.u64 %sh_q, %sh_base, %off;
    ld.shared.u32 %pe, [%sh_q + 0];
    ld.shared.u32 %ptk, [%sh_q + 1024];
    ld.shared.u32 %pmg, [%sh_q + 2048];
    ld.shared.u32 %ptp, [%sh_q + 3072];
L_MK_HS_SYNC:
    bar.sync 0;
    add.u32 %se, %se, %pe;
    add.u32 %stk, %stk, %ptk;
    add.u32 %smg, %smg, %pmg;
    add.u32 %stp, %stp, %ptp;
    st.shared.u32 [%sh_p + 0], %se;
    st.shared.u32 [%sh_p + 1024], %stk;
    st.shared.u32 [%sh_p + 2048], %smg;
    st.shared.u32 [%sh_p + 3072], %stp;
    bar.sync 0;
    shl.b32 %step, %step, 1;
    bra L_MK_HS;

L_MK_EMIT:
    @%p_oob ret;
    mul.wide.u32 %off, %bid, 16;
    add.u64 %addr, %boff, %off;
    ld.global.u32 %pe, [%addr + 0];
    ld.global.u32 %ptk, [%addr + 4];
    ld.global.u32 %pmg, [%addr + 8];
    ld.global.u32 %ptp, [%addr + 12];

    sub.u32 %start, %se, %c;
    add.u32 %start, %start, %pe;
    sub.u32 %t_off, %stk, %nt;
    add.u32 %t_off, %t_off, %ptk;
    sub.u32 %m_off, %smg, %nm;
    add.u32 %m_off, %m_off, %pmg;
    sub.u32 %tmp_off, %stp, %ntmp;
    add.u32 %tmp_off, %tmp_off, %ptp;

    mul.wide.u32 %off, %b, 4;
    add.u64 %addr, %cursors, %off;
    st.global.u32 [%addr], %start;

    setp.eq.u32 %p0, %c, 0;
    @!%p0 bra L_MK_NONZERO;
    mul.wide.u32 %off, %b, 192;
    add.u64 %addr, %buckets, %off;
    st.global.u64 [%addr + 96], 0;
    st.global.u64 [%addr + 104], 0;
    st.global.u64 [%addr + 112], 0;
    st.global.u64 [%addr + 120], 0;
    st.global.u64 [%addr + 128], 0;
    st.global.u64 [%addr + 136], 0;
    ret;

L_MK_NONZERO:
    add.u32 %end_b, %start, %c;
    setp.le.u32 %p1, %c, %mc;
    @!%p1 bra L_MK_MULTI;
    mul.wide.u32 %off, %t_off, 12;
    add.u64 %addr, %tasks, %off;
    st.global.u32 [%addr + 0], %start;
    st.global.u32 [%addr + 4], %end_b;
    st.global.u32 [%addr + 8], %b;
    ret;

L_MK_MULTI:
    mul.wide.u32 %off, %m_off, 12;
    add.u64 %addr, %merges, %off;
    st.global.u32 [%addr + 0], %b;
    st.global.u32 [%addr + 4], %tmp_off;
    st.global.u32 [%addr + 8], %nt;

    mov.u32 %k, 0;
    mov.u32 %sk, %start;
L_MK_LOOP:
    setp.ge.u32 %p_kdone, %k, %nt;
    @%p_kdone ret;
    add.u32 %ek, %sk, %mc;
    min.u32 %ek, %ek, %end_b;
    add.u32 %dk, %tmp_off, %k;
    add.u32 %tmp, %t_off, %k;
    mul.wide.u32 %off, %tmp, 12;
    add.u64 %addr, %tasks, %off;
    st.global.u32 [%addr + 0], %sk;
    st.global.u32 [%addr + 4], %ek;
    st.global.u32 [%addr + 8], %dk;
    mov.u32 %sk, %ek;
    add.u32 %k, %k, 1;
    bra L_MK_LOOP;
}}"#
    );

    // Kernel 3: msm_accumulate_tasks (100% register-resident, load-balanced
    // sub-bucket chunking so no thread ever exceeds MAX_CHUNK point additions).
    // Each task is 12 bytes: `{ start: u32, end: u32, dst: u32 }`.
    let _ = writeln!(
        out,
        r#"
.visible .entry msm_accumulate_tasks(
    .param .u64 p_bases,
    .param .u64 p_tasks,
    .param .u64 p_sorted,
    .param .u64 p_buckets,
    .param .u32 p_num_tasks
)
{{
    .reg .u32 %tid_g, %r_ctaid, %r_ntid, %num_t, %pos, %end, %dst, %ent, %pidx, %neg;
    .reg .u64 %bases, %tasks, %sorted, %buckets, %off, %addr, %dst_addr;
    .reg .u64 %ax<6>, %ay<6>, %azz<6>, %azzz<6>, %bx<6>, %by<6>;
    .reg .u64 %ta<6>, %tb<6>, %tc<6>, %td<6>, %te<6>;
    .reg .pred %oob, %done, %is_neg, %pz0, %pz1;"#
    );
    emit_fp_scratch_decls(out);
    let _ = writeln!(
        out,
        r#"
    mov.u32 %r_ctaid, %ctaid.x;
    mov.u32 %r_ntid, %ntid.x;
    mov.u32 %tid_g, %tid.x;
    mad.lo.u32 %tid_g, %r_ctaid, %r_ntid, %tid_g;

    ld.param.u32 %num_t, [p_num_tasks];
    setp.ge.u32 %oob, %tid_g, %num_t;
    @%oob ret;

    ld.param.u64 %bases, [p_bases];
    ld.param.u64 %tasks, [p_tasks];
    ld.param.u64 %sorted, [p_sorted];
    ld.param.u64 %buckets, [p_buckets];

    mul.wide.u32 %off, %tid_g, 12;
    add.u64 %addr, %tasks, %off;
    ld.global.u32 %pos, [%addr + 0];
    ld.global.u32 %end, [%addr + 4];
    ld.global.u32 %dst, [%addr + 8];

    mul.wide.u32 %off, %dst, 192;
    add.u64 %dst_addr, %buckets, %off;"#
    );
    emit_fp_set_zero(out, "ax");
    emit_fp_set_zero(out, "ay");
    emit_fp_set_zero(out, "azz");
    emit_fp_set_zero(out, "azzz");
    let _ = writeln!(
        out,
        r#"
L_ACC_LOOP:
    setp.ge.u32 %done, %pos, %end;
    @%done bra L_ACC_STORE;
    mul.wide.u32 %off, %pos, 4;
    add.u64 %addr, %sorted, %off;
    ld.global.u32 %ent, [%addr];
    and.b32 %pidx, %ent, 0x7fffffff;
    shr.u32 %neg, %ent, 31;

    mul.wide.u32 %off, %pidx, 96;
    add.u64 %addr, %bases, %off;"#
    );
    emit_fp_load(out, "global", "bx", "addr", 0);
    emit_fp_load(out, "global", "by", "addr", 48);
    let _ = writeln!(
        out,
        "    setp.ne.u32 %is_neg, %neg, 0;\n    @!%is_neg bra L_ACC_ADD;"
    );
    emit_fp_neg(out, "by", "by");
    let _ = writeln!(out, "L_ACC_ADD:");
    emit_xyzz_add_affine_inline(
        out, "ACC_MADD", "ax", "ay", "azz", "azzz", "bx", "by",
    );
    let _ = writeln!(
        out,
        "    add.u32 %pos, %pos, 1;\n    bra L_ACC_LOOP;\n\nL_ACC_STORE:"
    );
    emit_fp_store(out, "global", "dst_addr", 0, "ax");
    emit_fp_store(out, "global", "dst_addr", 48, "ay");
    emit_fp_store(out, "global", "dst_addr", 96, "azz");
    emit_fp_store(out, "global", "dst_addr", 144, "azzz");
    let _ = writeln!(out, "    ret;\n}}");

    // Kernel 3b: msm_merge_split_buckets (1 warp = 32 threads per split bucket).
    // Each merge descriptor is 12 bytes: `{ dst_bucket: u32, tmp_start: u32, tmp_count: u32 }`.
    let _ = writeln!(
        out,
        r#"
.visible .entry msm_merge_split_buckets(
    .param .u64 p_buckets,
    .param .u64 p_merges
)
{{
    .shared .align 8 .b8 sh_mrg[6144];
    .reg .u32 %m_idx, %lane, %dst_b, %tmp_s, %tmp_c, %k, %slot, %stride, %peer;
    .reg .u64 %buckets, %merges, %off, %addr, %sh_base, %sh_p;
    .reg .u64 %ax<6>, %ay<6>, %azz<6>, %azzz<6>;
    .reg .u64 %bx<6>, %by<6>, %bzz<6>, %bzzz<6>;
    .reg .u64 %ta<6>, %tb<6>, %tc<6>, %td<6>, %te<6>;
    .reg .pred %p_loop, %p_act, %is_t0, %pz0, %pz1;"#
    );
    emit_fp_scratch_decls(out);
    let _ = writeln!(
        out,
        r#"
    mov.u32 %m_idx, %ctaid.x;
    mov.u32 %lane, %tid.x;
    ld.param.u64 %buckets, [p_buckets];
    ld.param.u64 %merges, [p_merges];

    mul.wide.u32 %off, %m_idx, 12;
    add.u64 %addr, %merges, %off;
    ld.global.u32 %dst_b, [%addr + 0];
    ld.global.u32 %tmp_s, [%addr + 4];
    ld.global.u32 %tmp_c, [%addr + 8];"#
    );
    emit_fp_set_zero(out, "ax");
    emit_fp_set_zero(out, "ay");
    emit_fp_set_zero(out, "azz");
    emit_fp_set_zero(out, "azzz");
    let _ = writeln!(
        out,
        r#"
    mov.u32 %k, %lane;
L_MRG_STRIDE:
    setp.ge.u32 %p_loop, %k, %tmp_c;
    @%p_loop bra L_MRG_TREE;
    add.u32 %slot, %tmp_s, %k;
    mul.wide.u32 %off, %slot, 192;
    add.u64 %addr, %buckets, %off;"#
    );
    emit_fp_load(out, "global", "bx", "addr", 0);
    emit_fp_load(out, "global", "by", "addr", 48);
    emit_fp_load(out, "global", "bzz", "addr", 96);
    emit_fp_load(out, "global", "bzzz", "addr", 144);
    emit_xyzz_add_inline(
        out, "MRG_ACC", "ax", "ay", "azz", "azzz", "bx", "by", "bzz", "bzzz",
    );
    let _ = writeln!(out, "    add.u32 %k, %k, 32;\n    bra L_MRG_STRIDE;");

    let _ = writeln!(
        out,
        r#"
L_MRG_TREE:
    mov.u64 %sh_base, sh_mrg;
    mul.wide.u32 %off, %lane, 192;
    add.u64 %sh_p, %sh_base, %off;"#
    );
    emit_fp_store(out, "shared", "sh_p", 0, "ax");
    emit_fp_store(out, "shared", "sh_p", 48, "ay");
    emit_fp_store(out, "shared", "sh_p", 96, "azz");
    emit_fp_store(out, "shared", "sh_p", 144, "azzz");
    let _ = writeln!(
        out,
        r#"    bar.sync 0;

    mov.u32 %stride, 16;
L_MRG_RED:
    setp.eq.u32 %p_loop, %stride, 0;
    @%p_loop bra L_MRG_WRITE;
    setp.lt.u32 %p_act, %lane, %stride;
    @!%p_act bra L_MRG_SYNC;
    add.u32 %peer, %lane, %stride;
    mul.wide.u32 %off, %peer, 192;
    add.u64 %addr, %sh_base, %off;"#
    );
    emit_fp_load(out, "shared", "bx", "addr", 0);
    emit_fp_load(out, "shared", "by", "addr", 48);
    emit_fp_load(out, "shared", "bzz", "addr", 96);
    emit_fp_load(out, "shared", "bzzz", "addr", 144);
    emit_xyzz_add_inline(
        out, "MRG_T", "ax", "ay", "azz", "azzz", "bx", "by", "bzz", "bzzz",
    );
    emit_fp_store(out, "shared", "sh_p", 0, "ax");
    emit_fp_store(out, "shared", "sh_p", 48, "ay");
    emit_fp_store(out, "shared", "sh_p", 96, "azz");
    emit_fp_store(out, "shared", "sh_p", 144, "azzz");
    let _ = writeln!(
        out,
        r#"L_MRG_SYNC:
    bar.sync 0;
    shr.u32 %stride, %stride, 1;
    bra L_MRG_RED;

L_MRG_WRITE:
    setp.eq.u32 %is_t0, %lane, 0;
    @!%is_t0 ret;
    mul.wide.u32 %off, %dst_b, 192;
    add.u64 %addr, %buckets, %off;"#
    );
    emit_fp_store(out, "global", "addr", 0, "ax");
    emit_fp_store(out, "global", "addr", 48, "ay");
    emit_fp_store(out, "global", "addr", 96, "azz");
    emit_fp_store(out, "global", "addr", 144, "azzz");
    let _ = writeln!(out, "    ret;\n}}");

    // Kernel 4: msm_reduce_windows (64 threads per window, 24 KB shared mem =
    // 2 blocks/SM, parallel 6-level tree reduction for both sum and weighted
    // running sum!).
    let _ = writeln!(
        out,
        r#"
.visible .entry msm_reduce_windows(
    .param .u64 p_buckets,
    .param .u64 p_window_sums,
    .param .u32 p_num_buckets,
    .param .u32 p_log_chunk
)
{{
    .shared .align 8 .b8 sh_sum[12288];
    .shared .align 8 .b8 sh_run[12288];
    .reg .u32 %w_idx, %lane, %nb, %chunk, %log_c, %base_b, %k, %idx, %d, %sh_off, %stride, %peer, %lvl, %two_l;
    .reg .u64 %buckets, %wsums, %off, %b_addr, %out_addr, %sh_s, %sh_r, %sh_p, %sh_p2;
    .reg .u64 %ax<6>, %ay<6>, %azz<6>, %azzz<6>;
    .reg .u64 %bx<6>, %by<6>, %bzz<6>, %bzzz<6>;
    .reg .u64 %ta<6>, %tb<6>, %tc<6>, %td<6>, %te<6>;
    .reg .pred %p_loop, %p_act, %is_t0, %pz0, %pz1;"#
    );
    emit_fp_scratch_decls(out);
    let _ = writeln!(
        out,
        r#"
    mov.u32 %w_idx, %ctaid.x;
    mov.u32 %lane, %tid.x;
    ld.param.u64 %buckets, [p_buckets];
    ld.param.u64 %wsums, [p_window_sums];
    ld.param.u32 %nb, [p_num_buckets];
    ld.param.u32 %log_c, [p_log_chunk];

    shr.u32 %chunk, %nb, 6;
    mad.lo.u32 %base_b, %w_idx, %nb, 0;
    mad.lo.u32 %base_b, %lane, %chunk, %base_b;

    mul.lo.u32 %sh_off, %lane, 192;
    cvt.u64.u32 %off, %sh_off;
    mov.u64 %sh_s, sh_sum;
    mov.u64 %sh_r, sh_run;
    add.u64 %sh_p, %sh_s, %off;"#
    );
    emit_fp_set_zero(out, "ax");
    emit_fp_store(out, "shared", "sh_p", 0, "ax");
    emit_fp_store(out, "shared", "sh_p", 48, "ax");
    emit_fp_store(out, "shared", "sh_p", 96, "ax");
    emit_fp_store(out, "shared", "sh_p", 144, "ax");
    let _ = writeln!(out, "    add.u64 %sh_p, %sh_r, %off;");
    emit_fp_store(out, "shared", "sh_p", 0, "ax");
    emit_fp_store(out, "shared", "sh_p", 48, "ax");
    emit_fp_store(out, "shared", "sh_p", 96, "ax");
    emit_fp_store(out, "shared", "sh_p", 144, "ax");

    // Phase 1: each of the 64 threads computes local triangle sum over `chunk` buckets.
    let _ = writeln!(
        out,
        r#"
    mov.u32 %k, %chunk;
L_RED_CHUNK:
    setp.eq.u32 %p_loop, %k, 0;
    @%p_loop bra L_RED_P2A;
    sub.u32 %k, %k, 1;
    add.u32 %idx, %base_b, %k;
    mul.wide.u32 %off, %idx, 192;
    add.u64 %b_addr, %buckets, %off;

    cvt.u64.u32 %off, %sh_off;
    add.u64 %sh_p, %sh_r, %off;"#
    );
    emit_fp_load(out, "shared", "ax", "sh_p", 0);
    emit_fp_load(out, "shared", "ay", "sh_p", 48);
    emit_fp_load(out, "shared", "azz", "sh_p", 96);
    emit_fp_load(out, "shared", "azzz", "sh_p", 144);
    emit_fp_load(out, "global", "bx", "b_addr", 0);
    emit_fp_load(out, "global", "by", "b_addr", 48);
    emit_fp_load(out, "global", "bzz", "b_addr", 96);
    emit_fp_load(out, "global", "bzzz", "b_addr", 144);
    emit_xyzz_add_inline(
        out, "RED_RUN", "ax", "ay", "azz", "azzz", "bx", "by", "bzz", "bzzz",
    );
    emit_fp_store(out, "shared", "sh_p", 0, "ax");
    emit_fp_store(out, "shared", "sh_p", 48, "ay");
    emit_fp_store(out, "shared", "sh_p", 96, "azz");
    emit_fp_store(out, "shared", "sh_p", 144, "azzz");

    emit_fp_copy(out, "bx", "ax");
    emit_fp_copy(out, "by", "ay");
    emit_fp_copy(out, "bzz", "azz");
    emit_fp_copy(out, "bzzz", "azzz");
    let _ = writeln!(
        out,
        "    cvt.u64.u32 %off, %sh_off;\n    add.u64 %sh_p, %sh_s, %off;"
    );
    emit_fp_load(out, "shared", "ax", "sh_p", 0);
    emit_fp_load(out, "shared", "ay", "sh_p", 48);
    emit_fp_load(out, "shared", "azz", "sh_p", 96);
    emit_fp_load(out, "shared", "azzz", "sh_p", 144);
    emit_xyzz_add_inline(
        out, "RED_SUM", "ax", "ay", "azz", "azzz", "bx", "by", "bzz", "bzzz",
    );
    emit_fp_store(out, "shared", "sh_p", 0, "ax");
    emit_fp_store(out, "shared", "sh_p", 48, "ay");
    emit_fp_store(out, "shared", "sh_p", 96, "azz");
    emit_fp_store(out, "shared", "sh_p", 144, "azzz");
    let _ = writeln!(out, "    bra L_RED_CHUNK;");

    // Phase 2A: parallel 6-step binary tree reduction of `sh_sum[0..64]` -> `sh_sum[0]`.
    // Save `sh_sum[0]` to `wsums[w_idx]` temporarily so we can reuse `sh_sum` as `sh_w`!
    let _ = writeln!(
        out,
        r#"
L_RED_P2A:
    bar.sync 0;
    mov.u32 %stride, 32;
L_RED_STREE:
    setp.eq.u32 %p_loop, %stride, 0;
    @%p_loop bra L_RED_SAVE_S;
    setp.lt.u32 %p_act, %lane, %stride;
    @!%p_act bra L_RED_SSYNC;
    cvt.u64.u32 %off, %sh_off;
    add.u64 %sh_p, %sh_s, %off;
    add.u32 %peer, %lane, %stride;
    mul.wide.u32 %off, %peer, 192;
    add.u64 %sh_p2, %sh_s, %off;"#
    );
    emit_fp_load(out, "shared", "ax", "sh_p", 0);
    emit_fp_load(out, "shared", "ay", "sh_p", 48);
    emit_fp_load(out, "shared", "azz", "sh_p", 96);
    emit_fp_load(out, "shared", "azzz", "sh_p", 144);
    emit_fp_load(out, "shared", "bx", "sh_p2", 0);
    emit_fp_load(out, "shared", "by", "sh_p2", 48);
    emit_fp_load(out, "shared", "bzz", "sh_p2", 96);
    emit_fp_load(out, "shared", "bzzz", "sh_p2", 144);
    emit_xyzz_add_inline(
        out, "RED_ST", "ax", "ay", "azz", "azzz", "bx", "by", "bzz", "bzzz",
    );
    emit_fp_store(out, "shared", "sh_p", 0, "ax");
    emit_fp_store(out, "shared", "sh_p", 48, "ay");
    emit_fp_store(out, "shared", "sh_p", 96, "azz");
    emit_fp_store(out, "shared", "sh_p", 144, "azzz");
    let _ = writeln!(
        out,
        r#"L_RED_SSYNC:
    bar.sync 0;
    shr.u32 %stride, %stride, 1;
    bra L_RED_STREE;

L_RED_SAVE_S:
    setp.eq.u32 %is_t0, %lane, 0;
    @!%is_t0 bra L_RED_P2B_INIT;
    mul.wide.u32 %off, %w_idx, 192;
    add.u64 %out_addr, %wsums, %off;"#
    );
    emit_fp_load(out, "shared", "ax", "sh_s", 0);
    emit_fp_load(out, "shared", "ay", "sh_s", 48);
    emit_fp_load(out, "shared", "azz", "sh_s", 96);
    emit_fp_load(out, "shared", "azzz", "sh_s", 144);
    emit_fp_store(out, "global", "out_addr", 0, "ax");
    emit_fp_store(out, "global", "out_addr", 48, "ay");
    emit_fp_store(out, "global", "out_addr", 96, "azz");
    emit_fp_store(out, "global", "out_addr", 144, "azzz");

    // Phase 2B: parallel 6-level pair-merge for weighted sum
    // `W_all = sum(t=1..63) t * sh_run[t]` reusing `sh_sum` as `sh_w`!
    // Level 0 (`stride = 32`): for `t < 32`:
    //   w_new[t] = sh_run[2*t + 1], r_new[t] = sh_run[2*t] + sh_run[2*t + 1].
    let _ = writeln!(
        out,
        r#"L_RED_P2B_INIT:
    bar.sync 0;
    setp.lt.u32 %p_act, %lane, 32;
    @!%p_act bra L_RED_L0_SYNC;
    shl.b32 %two_l, %lane, 1;
    mul.wide.u32 %off, %two_l, 192;
    add.u64 %sh_p, %sh_r, %off;
    add.u64 %sh_p2, %sh_p, 192;"#
    );
    emit_fp_load(out, "shared", "ax", "sh_p", 0);
    emit_fp_load(out, "shared", "ay", "sh_p", 48);
    emit_fp_load(out, "shared", "azz", "sh_p", 96);
    emit_fp_load(out, "shared", "azzz", "sh_p", 144);
    emit_fp_load(out, "shared", "bx", "sh_p2", 0);
    emit_fp_load(out, "shared", "by", "sh_p2", 48);
    emit_fp_load(out, "shared", "bzz", "sh_p2", 96);
    emit_fp_load(out, "shared", "bzzz", "sh_p2", 144);
    // Write `sh_w[lane] = R_R` (in `bx..bzzz`) BEFORE `xyzz_add` clobbers `bx, by`!
    let _ = writeln!(
        out,
        "    cvt.u64.u32 %off, %sh_off;\n    add.u64 %sh_p, %sh_s, %off;"
    );
    emit_fp_store(out, "shared", "sh_p", 0, "bx");
    emit_fp_store(out, "shared", "sh_p", 48, "by");
    emit_fp_store(out, "shared", "sh_p", 96, "bzz");
    emit_fp_store(out, "shared", "sh_p", 144, "bzzz");
    emit_xyzz_add_inline(
        out, "RED_L0", "ax", "ay", "azz", "azzz", "bx", "by", "bzz", "bzzz",
    );
    let _ = writeln!(out, "L_RED_L0_SYNC:\n    bar.sync 0;");
    let _ = writeln!(
        out,
        "    @!%p_act bra L_RED_WTREE_START;\n    cvt.u64.u32 %off, %sh_off;\n    add.u64 %sh_p, %sh_r, %off;"
    );
    emit_fp_store(out, "shared", "sh_p", 0, "ax");
    emit_fp_store(out, "shared", "sh_p", 48, "ay");
    emit_fp_store(out, "shared", "sh_p", 96, "azz");
    emit_fp_store(out, "shared", "sh_p", 144, "azzz");

    // Levels `lvl = 1..=5` (`stride = 16, 8, 4, 2, 1`):
    //   W_new = sh_w[2t] + sh_w[2t+1] + 2^lvl * sh_run[2t+1]
    //   R_new = sh_run[2t] + sh_run[2t+1]
    let _ = writeln!(
        out,
        r#"L_RED_WTREE_START:
    bar.sync 0;
    mov.u32 %stride, 16;
    mov.u32 %lvl, 1;
L_RED_WTREE:
    setp.eq.u32 %p_loop, %stride, 0;
    @%p_loop bra L_RED_SCALE;
    setp.lt.u32 %p_act, %lane, %stride;
    @!%p_act bra L_RED_WSYNC1;

    // 1) Compute `2^lvl * R_R` in `ax..azzz` (from `sh_run[2*lane + 1]`).
    shl.b32 %two_l, %lane, 1;
    mul.wide.u32 %off, %two_l, 192;
    add.u64 %sh_p2, %sh_r, %off;
    add.u64 %sh_p2, %sh_p2, 192;"#
    );
    emit_fp_load(out, "shared", "ax", "sh_p2", 0);
    emit_fp_load(out, "shared", "ay", "sh_p2", 48);
    emit_fp_load(out, "shared", "azz", "sh_p2", 96);
    emit_fp_load(out, "shared", "azzz", "sh_p2", 144);
    let _ = writeln!(
        out,
        r#"    mov.u32 %d, 0;
L_RED_WDBL:
    setp.ge.u32 %p_loop, %d, %lvl;
    @%p_loop bra L_RED_WADD;
"#
    );
    emit_xyzz_dbl_inline(out, "RED_WL_DBL", "ax", "ay", "azz", "azzz", true);
    let _ = writeln!(out, "    add.u32 %d, %d, 1;\n    bra L_RED_WDBL;");
    let _ = writeln!(
        out,
        r#"L_RED_WADD:
    // Add `sh_w[2*lane]` and `sh_w[2*lane + 1]` into `ax..azzz` -> `W_new`.
    mul.wide.u32 %off, %two_l, 192;
    add.u64 %sh_p, %sh_s, %off;"#
    );
    emit_fp_load(out, "shared", "bx", "sh_p", 0);
    emit_fp_load(out, "shared", "by", "sh_p", 48);
    emit_fp_load(out, "shared", "bzz", "sh_p", 96);
    emit_fp_load(out, "shared", "bzzz", "sh_p", 144);
    emit_xyzz_add_inline(
        out, "RED_WA1", "ax", "ay", "azz", "azzz", "bx", "by", "bzz", "bzzz",
    );
    let _ = writeln!(out, "    add.u64 %sh_p2, %sh_p, 192;");
    emit_fp_load(out, "shared", "bx", "sh_p2", 0);
    emit_fp_load(out, "shared", "by", "sh_p2", 48);
    emit_fp_load(out, "shared", "bzz", "sh_p2", 96);
    emit_fp_load(out, "shared", "bzzz", "sh_p2", 144);
    emit_xyzz_add_inline(
        out, "RED_WA2", "ax", "ay", "azz", "azzz", "bx", "by", "bzz", "bzzz",
    );
    // Sync before overwriting `sh_s[lane]` (safe: all active lanes already read `sh_s[2t], sh_s[2t+1]`).
    let _ = writeln!(out, "L_RED_WSYNC1:\n    bar.sync 0;");
    let _ = writeln!(
        out,
        "    @!%p_act bra L_RED_WSYNC2;\n    cvt.u64.u32 %off, %sh_off;\n    add.u64 %sh_p, %sh_s, %off;"
    );
    emit_fp_store(out, "shared", "sh_p", 0, "ax");
    emit_fp_store(out, "shared", "sh_p", 48, "ay");
    emit_fp_store(out, "shared", "sh_p", 96, "azz");
    emit_fp_store(out, "shared", "sh_p", 144, "azzz");

    // 2) Compute `R_new = sh_run[2*lane] + sh_run[2*lane + 1]` in `ax..azzz`.
    let _ = writeln!(
        out,
        "    mul.wide.u32 %off, %two_l, 192;\n    add.u64 %sh_p, %sh_r, %off;\n    add.u64 %sh_p2, %sh_p, 192;"
    );
    emit_fp_load(out, "shared", "ax", "sh_p", 0);
    emit_fp_load(out, "shared", "ay", "sh_p", 48);
    emit_fp_load(out, "shared", "azz", "sh_p", 96);
    emit_fp_load(out, "shared", "azzz", "sh_p", 144);
    emit_fp_load(out, "shared", "bx", "sh_p2", 0);
    emit_fp_load(out, "shared", "by", "sh_p2", 48);
    emit_fp_load(out, "shared", "bzz", "sh_p2", 96);
    emit_fp_load(out, "shared", "bzzz", "sh_p2", 144);
    emit_xyzz_add_inline(
        out, "RED_RA", "ax", "ay", "azz", "azzz", "bx", "by", "bzz", "bzzz",
    );
    let _ = writeln!(out, "L_RED_WSYNC2:\n    bar.sync 0;");
    let _ = writeln!(
        out,
        "    @!%p_act bra L_RED_WSYNC3;\n    cvt.u64.u32 %off, %sh_off;\n    add.u64 %sh_p, %sh_r, %off;"
    );
    emit_fp_store(out, "shared", "sh_p", 0, "ax");
    emit_fp_store(out, "shared", "sh_p", 48, "ay");
    emit_fp_store(out, "shared", "sh_p", 96, "azz");
    emit_fp_store(out, "shared", "sh_p", 144, "azzz");
    let _ = writeln!(
        out,
        r#"L_RED_WSYNC3:
    bar.sync 0;
    shr.u32 %stride, %stride, 1;
    add.u32 %lvl, %lvl, 1;
    bra L_RED_WTREE;

    // Phase 3: thread 0 scales `sh_w[0]` by `chunk = 2^log_c` and adds saved `S_all`.
L_RED_SCALE:
    setp.eq.u32 %is_t0, %lane, 0;
    @!%is_t0 ret;"#
    );
    emit_fp_load(out, "shared", "ax", "sh_s", 0);
    emit_fp_load(out, "shared", "ay", "sh_s", 48);
    emit_fp_load(out, "shared", "azz", "sh_s", 96);
    emit_fp_load(out, "shared", "azzz", "sh_s", 144);
    let _ = writeln!(
        out,
        r#"    mov.u32 %d, 0;
L_RED_DBL:
    setp.ge.u32 %p_loop, %d, %log_c;
    @%p_loop bra L_RED_FINISH;"#
    );
    emit_xyzz_dbl_inline(out, "RED_SCALE_DBL", "ax", "ay", "azz", "azzz", true);
    let _ = writeln!(out, "    add.u32 %d, %d, 1;\n    bra L_RED_DBL;");

    let _ = writeln!(
        out,
        "L_RED_FINISH:\n    mul.wide.u32 %off, %w_idx, 192;\n    add.u64 %out_addr, %wsums, %off;"
    );
    emit_fp_load(out, "global", "bx", "out_addr", 0);
    emit_fp_load(out, "global", "by", "out_addr", 48);
    emit_fp_load(out, "global", "bzz", "out_addr", 96);
    emit_fp_load(out, "global", "bzzz", "out_addr", 144);
    emit_xyzz_add_inline(
        out, "RED_FIN", "ax", "ay", "azz", "azzz", "bx", "by", "bzz", "bzzz",
    );
    emit_fp_store(out, "global", "out_addr", 0, "ax");
    emit_fp_store(out, "global", "out_addr", 48, "ay");
    emit_fp_store(out, "global", "out_addr", 96, "azz");
    emit_fp_store(out, "global", "out_addr", 144, "azzz");
    let _ = writeln!(out, "    ret;\n}}");

    // Unit-test kernel: verify inline `fp_mul`, `fp_add`, `fp_sub`, `fp_neg`.
    let _ = writeln!(
        out,
        r#"
.visible .entry test_fp_ops(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_out_mul,
    .param .u64 p_out_add,
    .param .u64 p_out_sub,
    .param .u64 p_out_neg,
    .param .u32 p_n
)
{{
    .reg .u32 %r_tid, %n;
    .reg .u64 %pa, %pb, %pm, %pad, %ps, %pn, %off;
    .reg .u64 %a<6>, %b<6>, %res<6>;
    .reg .pred %oob;"#
    );
    emit_fp_scratch_decls(out);
    let _ = writeln!(
        out,
        r#"    mov.u32 %r_tid, %tid.x;
    ld.param.u32 %n, [p_n];
    setp.ge.u32 %oob, %r_tid, %n;
    @%oob ret;
    ld.param.u64 %pa, [p_a];
    ld.param.u64 %pb, [p_b];
    ld.param.u64 %pm, [p_out_mul];
    ld.param.u64 %pad, [p_out_add];
    ld.param.u64 %ps, [p_out_sub];
    ld.param.u64 %pn, [p_out_neg];
    mul.wide.u32 %off, %r_tid, 48;
    add.u64 %pa, %pa, %off;
    add.u64 %pb, %pb, %off;
    add.u64 %pm, %pm, %off;
    add.u64 %pad, %pad, %off;
    add.u64 %ps, %ps, %off;
    add.u64 %pn, %pn, %off;"#
    );
    emit_fp_load(out, "global", "a", "pa", 0);
    emit_fp_load(out, "global", "b", "pb", 0);
    emit_fp_mul(out, "res", "a", "b");
    emit_fp_store(out, "global", "pm", 0, "res");
    emit_fp_add(out, "res", "a", "b");
    emit_fp_store(out, "global", "pad", 0, "res");
    emit_fp_sub(out, "res", "a", "b");
    emit_fp_store(out, "global", "ps", 0, "res");
    emit_fp_neg(out, "res", "a");
    emit_fp_store(out, "global", "pn", 0, "res");
    let _ = writeln!(out, "    ret;\n}}");
}

/// Declare scratch registers required by inline 4-limb `Fr` arithmetic helpers.
fn emit_fr_scratch_decls(out: &mut String) {
    let _ = writeln!(
        out,
        "    .reg .u64 %ft<5>, %fs<4>, %fm, %fz0, %fborrow;\n    .reg .pred %fge;"
    );
}

fn emit_fr_load(
    out: &mut String,
    space: &str,
    dst: &str,
    ptr: &str,
    off: usize,
) {
    for i in 0..4 {
        let _ = writeln!(
            out,
            "    ld.{space}.u64 %{dst}{i}, [%{ptr} + {}];",
            off + i * 8
        );
    }
}

fn emit_fr_store(
    out: &mut String,
    space: &str,
    ptr: &str,
    off: usize,
    src: &str,
) {
    for i in 0..4 {
        let _ = writeln!(
            out,
            "    st.{space}.u64 [%{ptr} + {}], %{src}{i};",
            off + i * 8
        );
    }
}

/// Inline aliasing-safe `dst = (a + b) mod r` in 4 `u64` registers.
fn emit_fr_add(out: &mut String, dst: &str, a: &str, b: &str) {
    let r = FR_R;
    let _ = writeln!(
        out,
        "    mov.u64 %fz0, 0;\n    \
         add.cc.u64  %ft0, %{a}0, %{b}0;\n    \
         addc.cc.u64 %ft1, %{a}1, %{b}1;\n    \
         addc.cc.u64 %ft2, %{a}2, %{b}2;\n    \
         addc.u64    %ft3, %{a}3, %{b}3;\n    \
         sub.cc.u64  %fs0, %ft0, 0x{:016x};\n    \
         subc.cc.u64 %fs1, %ft1, 0x{:016x};\n    \
         subc.cc.u64 %fs2, %ft2, 0x{:016x};\n    \
         subc.cc.u64 %fs3, %ft3, 0x{:016x};\n    \
         subc.u64 %fborrow, %fz0, %fz0;\n    \
         setp.eq.u64 %fge, %fborrow, 0;\n    \
         selp.u64 %{dst}0, %fs0, %ft0, %fge;\n    \
         selp.u64 %{dst}1, %fs1, %ft1, %fge;\n    \
         selp.u64 %{dst}2, %fs2, %ft2, %fge;\n    \
         selp.u64 %{dst}3, %fs3, %ft3, %fge;",
        r[0], r[1], r[2], r[3]
    );
}

/// Inline aliasing-safe `dst = (a - b) mod r` in 4 `u64` registers.
fn emit_fr_sub(out: &mut String, dst: &str, a: &str, b: &str) {
    let r = FR_R;
    let _ = writeln!(
        out,
        "    mov.u64 %fz0, 0;\n    \
         sub.cc.u64  %ft0, %{a}0, %{b}0;\n    \
         subc.cc.u64 %ft1, %{a}1, %{b}1;\n    \
         subc.cc.u64 %ft2, %{a}2, %{b}2;\n    \
         subc.cc.u64 %ft3, %{a}3, %{b}3;\n    \
         subc.u64 %fborrow, %fz0, %fz0;\n    \
         add.cc.u64  %fs0, %ft0, 0x{:016x};\n    \
         addc.cc.u64 %fs1, %ft1, 0x{:016x};\n    \
         addc.cc.u64 %fs2, %ft2, 0x{:016x};\n    \
         addc.u64    %fs3, %ft3, 0x{:016x};\n    \
         setp.eq.u64 %fge, %fborrow, 0;\n    \
         selp.u64 %{dst}0, %ft0, %fs0, %fge;\n    \
         selp.u64 %{dst}1, %ft1, %fs1, %fge;\n    \
         selp.u64 %{dst}2, %ft2, %fs2, %fge;\n    \
         selp.u64 %{dst}3, %ft3, %fs3, %fge;",
        r[0], r[1], r[2], r[3]
    );
}

/// Inline aliasing-safe 4-limb CIOS Montgomery `dst = a * b * R^-1 mod r` in
/// registers (`dst` may freely alias `a` and/or `b`).
fn emit_fr_mul(out: &mut String, dst: &str, a: &str, b: &str) {
    let r = FR_R;
    let _ = writeln!(
        out,
        "    mov.u64 %fz0, 0;\n    \
         mov.u64 %ft0, 0; mov.u64 %ft1, 0; mov.u64 %ft2, 0;\n    \
         mov.u64 %ft3, 0; mov.u64 %ft4, 0;"
    );
    for i in 0..4 {
        let _ = writeln!(
            out,
            "    mad.lo.cc.u64  %ft0, %{a}{i}, %{b}0, %ft0;\n    \
             madc.lo.cc.u64 %ft1, %{a}{i}, %{b}1, %ft1;\n    \
             madc.lo.cc.u64 %ft2, %{a}{i}, %{b}2, %ft2;\n    \
             madc.lo.cc.u64 %ft3, %{a}{i}, %{b}3, %ft3;\n    \
             addc.u64       %ft4, %ft4, 0;\n    \
             mad.hi.cc.u64  %ft1, %{a}{i}, %{b}0, %ft1;\n    \
             madc.hi.cc.u64 %ft2, %{a}{i}, %{b}1, %ft2;\n    \
             madc.hi.cc.u64 %ft3, %{a}{i}, %{b}2, %ft3;\n    \
             madc.hi.u64    %ft4, %{a}{i}, %{b}3, %ft4;\n    \
             mul.lo.u64     %fm,  %ft0, 0x{:016x};\n    \
             mad.lo.cc.u64  %ft0, %fm, 0x{:016x}, %ft0;\n    \
             madc.lo.cc.u64 %ft1, %fm, 0x{:016x}, %ft1;\n    \
             madc.lo.cc.u64 %ft2, %fm, 0x{:016x}, %ft2;\n    \
             madc.lo.cc.u64 %ft3, %fm, 0x{:016x}, %ft3;\n    \
             addc.u64       %ft4, %ft4, 0;\n    \
             mad.hi.cc.u64  %ft0, %fm, 0x{:016x}, %ft1;\n    \
             madc.hi.cc.u64 %ft1, %fm, 0x{:016x}, %ft2;\n    \
             madc.hi.cc.u64 %ft2, %fm, 0x{:016x}, %ft3;\n    \
             madc.hi.u64    %ft3, %fm, 0x{:016x}, %ft4;\n    \
             mov.u64        %ft4, 0;",
            FR_INV, r[0], r[1], r[2], r[3], r[0], r[1], r[2], r[3]
        );
    }
    let _ = writeln!(
        out,
        "    sub.cc.u64  %fs0, %ft0, 0x{:016x};\n    \
         subc.cc.u64 %fs1, %ft1, 0x{:016x};\n    \
         subc.cc.u64 %fs2, %ft2, 0x{:016x};\n    \
         subc.cc.u64 %fs3, %ft3, 0x{:016x};\n    \
         subc.u64 %fborrow, %fz0, %fz0;\n    \
         setp.eq.u64 %fge, %fborrow, 0;\n    \
         selp.u64 %{dst}0, %fs0, %ft0, %fge;\n    \
         selp.u64 %{dst}1, %fs1, %ft1, %fge;\n    \
         selp.u64 %{dst}2, %fs2, %ft2, %fge;\n    \
         selp.u64 %{dst}3, %fs3, %ft3, %fge;",
        r[0], r[1], r[2], r[3]
    );
}

/// Emit the 8-stage shared-memory Cooley-Tukey DIT loop (`stage = 0..8`)
/// operating on `sh[0..256]` (8 KB) with 128 threads/block.
fn emit_shared_8_stages(out: &mut String, lbl: &str) {
    let _ = writeln!(
        out,
        r#"    mov.u32 %st, 0;
L_{lbl}_ST:
    setp.ge.u32 %p_done, %st, 8;
    @%p_done bra L_{lbl}_STORE;
    mov.u32 %m, 1;
    shl.b32 %m, %m, %st;
    sub.u32 %mask, %m, 1;
    and.b32 %jj, %lane, %mask;
    shr.u32 %grp, %lane, %st;
    add.u32 %st1, %st, 1;
    shl.b32 %i0, %grp, %st1;
    add.u32 %i0, %i0, %jj;
    add.u32 %i1, %i0, %m;
    sub.u32 %rsh, %log_n, %st1;
    shl.b32 %ridx, %jj, %rsh;

    mul.wide.u32 %off, %ridx, 32;
    add.u64 %addr, %roots, %off;"#
    );
    emit_fr_load(out, "global", "rw", "addr", 0);
    let _ = writeln!(
        out,
        "    mul.wide.u32 %off, %i0, 32;\n    add.u64 %shp0, %sh_base, %off;\n    \
         mul.wide.u32 %off, %i1, 32;\n    add.u64 %shp1, %sh_base, %off;"
    );
    emit_fr_load(out, "shared", "ru", "shp0", 0);
    emit_fr_load(out, "shared", "rv", "shp1", 0);
    emit_fr_mul(out, "rv", "rv", "rw");
    emit_fr_add(out, "roa", "ru", "rv");
    emit_fr_sub(out, "rob", "ru", "rv");
    emit_fr_store(out, "shared", "shp0", 0, "roa");
    emit_fr_store(out, "shared", "shp1", 0, "rob");
    let _ = writeln!(
        out,
        "    bar.sync 0;\n    add.u32 %st, %st, 1;\n    bra L_{lbl}_ST;"
    );
}

/// Emit the 3 batched BLS12-381 `Fr` NTT kernels:
/// 1. `ntt_coset_fold_stage8`: fused bit-reverse + coset fold (`lo + hi*c_k^n`)
///    + `coset_powers` multiply + first 8 shared-memory radix-2 stages.
/// 2. `ntt_bitrev_stage8`: fused bit-reverse + first 8 shared-memory radix-2
///    stages (for batched IFFTs).
/// 3. `ntt_radix2_stage`: global radix-2 butterfly stage (`stage >= 8`) with
///    optional fused `size_inv` scaling on the final stage.
fn emit_ntt_kernels(out: &mut String) {
    // Kernel 1: ntt_coset_fold_stage8
    let _ = writeln!(
        out,
        r#"
.visible .entry ntt_coset_fold_stage8(
    .param .u64 p_src,
    .param .u64 p_coset_powers,
    .param .u64 p_ck_pown,
    .param .u64 p_roots,
    .param .u64 p_out,
    .param .u32 p_n,
    .param .u32 p_log_n,
    .param .u32 p_src_stride,
    .param .u32 p_num_polys
)
{{
    .shared .align 8 .b8 sh_ntt[8192];
    .reg .u32 %blk, %lane, %n, %log_n, %stride, %npolys, %tiles_per_poly, %out_idx, %tile, %base;
    .reg .u32 %local_k, %p_idx, %rev_sh, %half, %slot, %pos, %j, %jn, %sidx, %cidx;
    .reg .u32 %st, %st1, %m, %mask, %jj, %grp, %i0, %i1, %rsh, %ridx;
    .reg .u64 %src, %cpows, %ckp, %roots, %out_p, %sh_base, %shp0, %shp1, %off, %addr;
    .reg .u64 %ru<4>, %rv<4>, %rw<4>, %roa<4>, %rob<4>, %rck<4>;
    .reg .pred %p_hi, %p_done;"#
    );
    emit_fr_scratch_decls(out);
    let _ = writeln!(
        out,
        r#"
    mov.u32 %blk, %ctaid.x;
    mov.u32 %lane, %tid.x;
    ld.param.u64 %src, [p_src];
    ld.param.u64 %cpows, [p_coset_powers];
    ld.param.u64 %ckp, [p_ck_pown];
    ld.param.u64 %roots, [p_roots];
    ld.param.u64 %out_p, [p_out];
    ld.param.u32 %n, [p_n];
    ld.param.u32 %log_n, [p_log_n];
    ld.param.u32 %stride, [p_src_stride];
    ld.param.u32 %npolys, [p_num_polys];

    shr.u32 %tiles_per_poly, %n, 8;
    div.u32 %out_idx, %blk, %tiles_per_poly;
    rem.u32 %tile, %blk, %tiles_per_poly;
    shl.b32 %base, %tile, 8;
    div.u32 %local_k, %out_idx, %npolys;
    rem.u32 %p_idx, %out_idx, %npolys;
    sub.u32 %rev_sh, 32, %log_n;

    mul.wide.u32 %off, %local_k, 32;
    add.u64 %addr, %ckp, %off;"#
    );
    emit_fr_load(out, "global", "rck", "addr", 0);
    let _ = writeln!(
        out,
        r#"    mov.u64 %sh_base, sh_ntt;
    mov.u32 %half, 0;
L_CF_LOAD:
    setp.ge.u32 %p_done, %half, 2;
    @%p_done bra L_CF_SYNC;
    shl.b32 %slot, %half, 7;
    add.u32 %slot, %slot, %lane;
    add.u32 %pos, %base, %slot;
    brev.b32 %j, %pos;
    shr.u32 %j, %j, %rev_sh;

    mad.lo.u32 %sidx, %p_idx, %stride, %j;
    mul.wide.u32 %off, %sidx, 32;
    add.u64 %addr, %src, %off;"#
    );
    emit_fr_load(out, "global", "ru", "addr", 0);
    let _ = writeln!(
        out,
        r#"    add.u32 %jn, %j, %n;
    setp.lt.u32 %p_hi, %jn, %stride;
    @!%p_hi bra L_CF_NOHI;
    mad.lo.u32 %sidx, %p_idx, %stride, %jn;
    mul.wide.u32 %off, %sidx, 32;
    add.u64 %addr, %src, %off;"#
    );
    emit_fr_load(out, "global", "rv", "addr", 0);
    emit_fr_mul(out, "rv", "rv", "rck");
    emit_fr_add(out, "ru", "ru", "rv");
    let _ = writeln!(
        out,
        r#"L_CF_NOHI:
    mad.lo.u32 %cidx, %local_k, %n, %j;
    mul.wide.u32 %off, %cidx, 32;
    add.u64 %addr, %cpows, %off;"#
    );
    emit_fr_load(out, "global", "rw", "addr", 0);
    emit_fr_mul(out, "ru", "ru", "rw");
    let _ = writeln!(
        out,
        "    mul.wide.u32 %off, %slot, 32;\n    add.u64 %shp0, %sh_base, %off;"
    );
    emit_fr_store(out, "shared", "shp0", 0, "ru");
    let _ = writeln!(
        out,
        "    add.u32 %half, %half, 1;\n    bra L_CF_LOAD;\nL_CF_SYNC:\n    bar.sync 0;"
    );
    emit_shared_8_stages(out, "CF");
    let _ = writeln!(
        out,
        r#"L_CF_STORE:
    mov.u32 %half, 0;
L_CF_ST_LOOP:
    setp.ge.u32 %p_done, %half, 2;
    @%p_done ret;
    shl.b32 %slot, %half, 7;
    add.u32 %slot, %slot, %lane;
    mul.wide.u32 %off, %slot, 32;
    add.u64 %shp0, %sh_base, %off;"#
    );
    emit_fr_load(out, "shared", "ru", "shp0", 0);
    let _ = writeln!(
        out,
        "    add.u32 %pos, %base, %slot;\n    \
         mad.lo.u32 %sidx, %out_idx, %n, %pos;\n    \
         mul.wide.u32 %off, %sidx, 32;\n    \
         add.u64 %addr, %out_p, %off;"
    );
    emit_fr_store(out, "global", "addr", 0, "ru");
    let _ = writeln!(
        out,
        "    add.u32 %half, %half, 1;\n    bra L_CF_ST_LOOP;\n}}"
    );

    // Kernel 2: ntt_bitrev_stage8 (fused bit-reverse + first 8 shared-mem stages)
    let _ = writeln!(
        out,
        r#"
.visible .entry ntt_bitrev_stage8(
    .param .u64 p_src,
    .param .u64 p_roots,
    .param .u64 p_out,
    .param .u32 p_n,
    .param .u32 p_log_n
)
{{
    .shared .align 8 .b8 sh_ntt[8192];
    .reg .u32 %blk, %lane, %n, %log_n, %tiles_per_poly, %out_idx, %tile, %base, %rev_sh;
    .reg .u32 %half, %slot, %pos, %j, %sidx;
    .reg .u32 %st, %st1, %m, %mask, %jj, %grp, %i0, %i1, %rsh, %ridx;
    .reg .u64 %src, %roots, %out_p, %sh_base, %shp0, %shp1, %off, %addr;
    .reg .u64 %ru<4>, %rv<4>, %rw<4>, %roa<4>, %rob<4>;
    .reg .pred %p_done;"#
    );
    emit_fr_scratch_decls(out);
    let _ = writeln!(
        out,
        r#"
    mov.u32 %blk, %ctaid.x;
    mov.u32 %lane, %tid.x;
    ld.param.u64 %src, [p_src];
    ld.param.u64 %roots, [p_roots];
    ld.param.u64 %out_p, [p_out];
    ld.param.u32 %n, [p_n];
    ld.param.u32 %log_n, [p_log_n];

    shr.u32 %tiles_per_poly, %n, 8;
    div.u32 %out_idx, %blk, %tiles_per_poly;
    rem.u32 %tile, %blk, %tiles_per_poly;
    shl.b32 %base, %tile, 8;
    sub.u32 %rev_sh, 32, %log_n;

    mov.u64 %sh_base, sh_ntt;
    mov.u32 %half, 0;
L_BR_LOAD:
    setp.ge.u32 %p_done, %half, 2;
    @%p_done bra L_BR_SYNC;
    shl.b32 %slot, %half, 7;
    add.u32 %slot, %slot, %lane;
    add.u32 %pos, %base, %slot;
    brev.b32 %j, %pos;
    shr.u32 %j, %j, %rev_sh;
    mad.lo.u32 %sidx, %out_idx, %n, %j;
    mul.wide.u32 %off, %sidx, 32;
    add.u64 %addr, %src, %off;"#
    );
    emit_fr_load(out, "global", "ru", "addr", 0);
    let _ = writeln!(
        out,
        "    mul.wide.u32 %off, %slot, 32;\n    add.u64 %shp0, %sh_base, %off;"
    );
    emit_fr_store(out, "shared", "shp0", 0, "ru");
    let _ = writeln!(
        out,
        "    add.u32 %half, %half, 1;\n    bra L_BR_LOAD;\nL_BR_SYNC:\n    bar.sync 0;"
    );
    emit_shared_8_stages(out, "BR");
    let _ = writeln!(
        out,
        r#"L_BR_STORE:
    mov.u32 %half, 0;
L_BR_ST_LOOP:
    setp.ge.u32 %p_done, %half, 2;
    @%p_done ret;
    shl.b32 %slot, %half, 7;
    add.u32 %slot, %slot, %lane;
    mul.wide.u32 %off, %slot, 32;
    add.u64 %shp0, %sh_base, %off;"#
    );
    emit_fr_load(out, "shared", "ru", "shp0", 0);
    let _ = writeln!(
        out,
        "    add.u32 %pos, %base, %slot;\n    \
         mad.lo.u32 %sidx, %out_idx, %n, %pos;\n    \
         mul.wide.u32 %off, %sidx, 32;\n    \
         add.u64 %addr, %out_p, %off;"
    );
    emit_fr_store(out, "global", "addr", 0, "ru");
    let _ = writeln!(
        out,
        "    add.u32 %half, %half, 1;\n    bra L_BR_ST_LOOP;\n}}"
    );

    // Kernel 3: ntt_radix2_stage (global stages >= 8, optional fused final scale)
    let _ = writeln!(
        out,
        r#"
.visible .entry ntt_radix2_stage(
    .param .u64 p_buf,
    .param .u64 p_roots,
    .param .u64 p_scale,
    .param .u32 p_n,
    .param .u32 p_log_n,
    .param .u32 p_stage,
    .param .u32 p_total_bf
)
{{
    .reg .u32 %tid_g, %r_ctaid, %r_ntid, %n, %log_n, %stage, %stage1, %total_bf;
    .reg .u32 %half_log, %half_mask, %poly, %k, %m, %m_mask, %j, %grp, %i0, %i1, %rsh, %ridx;
    .reg .u64 %buf, %roots, %scale_p, %off, %addr0, %addr1, %raddr;
    .reg .u64 %ru<4>, %rv<4>, %rw<4>, %roa<4>, %rob<4>, %rsc<4>;
    .reg .pred %oob, %has_scale;"#
    );
    emit_fr_scratch_decls(out);
    let _ = writeln!(
        out,
        r#"
    mov.u32 %r_ctaid, %ctaid.x;
    mov.u32 %r_ntid, %ntid.x;
    mov.u32 %tid_g, %tid.x;
    mad.lo.u32 %tid_g, %r_ctaid, %r_ntid, %tid_g;

    ld.param.u32 %total_bf, [p_total_bf];
    setp.ge.u32 %oob, %tid_g, %total_bf;
    @%oob ret;

    ld.param.u64 %buf, [p_buf];
    ld.param.u64 %roots, [p_roots];
    ld.param.u64 %scale_p, [p_scale];
    ld.param.u32 %n, [p_n];
    ld.param.u32 %log_n, [p_log_n];
    ld.param.u32 %stage, [p_stage];

    sub.u32 %half_log, %log_n, 1;
    shr.u32 %poly, %tid_g, %half_log;
    shr.u32 %half_mask, %n, 1;
    sub.u32 %half_mask, %half_mask, 1;
    and.b32 %k, %tid_g, %half_mask;

    mov.u32 %m, 1;
    shl.b32 %m, %m, %stage;
    sub.u32 %m_mask, %m, 1;
    and.b32 %j, %k, %m_mask;
    shr.u32 %grp, %k, %stage;
    add.u32 %stage1, %stage, 1;
    shl.b32 %i0, %grp, %stage1;
    add.u32 %i0, %i0, %j;
    shl.b32 %poly, %poly, %log_n;
    add.u32 %i0, %i0, %poly;
    add.u32 %i1, %i0, %m;

    sub.u32 %rsh, %log_n, %stage1;
    shl.b32 %ridx, %j, %rsh;
    mul.wide.u32 %off, %ridx, 32;
    add.u64 %raddr, %roots, %off;
    mul.wide.u32 %off, %i0, 32;
    add.u64 %addr0, %buf, %off;
    mul.wide.u32 %off, %i1, 32;
    add.u64 %addr1, %buf, %off;"#
    );
    emit_fr_load(out, "global", "ru", "addr0", 0);
    emit_fr_load(out, "global", "rv", "addr1", 0);
    emit_fr_load(out, "global", "rw", "raddr", 0);
    emit_fr_mul(out, "rv", "rv", "rw");
    emit_fr_add(out, "roa", "ru", "rv");
    emit_fr_sub(out, "rob", "ru", "rv");
    let _ = writeln!(
        out,
        "    setp.ne.u64 %has_scale, %scale_p, 0;\n    @!%has_scale bra L_R2_STORE;"
    );
    emit_fr_load(out, "global", "rsc", "scale_p", 0);
    emit_fr_mul(out, "roa", "roa", "rsc");
    emit_fr_mul(out, "rob", "rob", "rsc");
    let _ = writeln!(out, "L_R2_STORE:");
    emit_fr_store(out, "global", "addr0", 0, "roa");
    emit_fr_store(out, "global", "addr1", 0, "rob");
    let _ = writeln!(out, "    ret;\n}}");
}

/// Emit a register-only `.func fr_mul_val` (`.reg` state-space parameters,
/// 0 bytes `.local`, 0 pointers) used by `quotient_eval_wave` so `ptxas`
/// compiles the full 6-widget quotient kernel in `< 0.1s`.
fn emit_fr_mul_val_func(out: &mut String) {
    let _ = writeln!(
        out,
        r#"
.func (
    .reg .u64 %r0, .reg .u64 %r1, .reg .u64 %r2, .reg .u64 %r3
) fr_mul_val(
    .reg .u64 %a0, .reg .u64 %a1, .reg .u64 %a2, .reg .u64 %a3,
    .reg .u64 %b0, .reg .u64 %b1, .reg .u64 %b2, .reg .u64 %b3
)
{{"#
    );
    emit_fr_scratch_decls(out);
    emit_fr_mul(out, "r", "a", "b");
    let _ = writeln!(out, "    ret;\n}}");
}

fn emit_fr_mul_call(out: &mut String, dst: &str, a: &str, b: &str) {
    let _ = writeln!(
        out,
        "    call (%{dst}0, %{dst}1, %{dst}2, %{dst}3), fr_mul_val, (\
         %{a}0, %{a}1, %{a}2, %{a}3, %{b}0, %{b}1, %{b}2, %{b}3);"
    );
}

fn emit_fr_is_zero(out: &mut String, pred: &str, a: &str) {
    let _ = writeln!(
        out,
        "    or.b64 %ft0, %{a}0, %{a}1;\n    \
         or.b64 %ft1, %{a}2, %{a}3;\n    \
         or.b64 %ft0, %ft0, %ft1;\n    \
         setp.eq.u64 %{pred}, %ft0, 0;"
    );
}

fn emit_fr_set_zero(out: &mut String, dst: &str) {
    for i in 0..4 {
        let _ = writeln!(out, "    mov.u64 %{dst}{i}, 0;");
    }
}

/// Load challenge `idx` from `p_ch` (`[FastFr; 44]`) into register bank `dst`.
fn emit_ch_load(out: &mut String, dst: &str, idx: usize) {
    emit_fr_load(out, "global", dst, "ch_p", idx * 32);
}

/// Load selector `sel_idx` (`0..17`) at flat offset `%sel_off` (`(k*n + j) * 32`)
/// into register bank `dst` using `%addr` as scratch.
fn emit_sel_load(out: &mut String, dst: &str, sel_idx: usize) {
    let _ = writeln!(
        out,
        "    ld.global.u64 %addr, [%sel_ptrs + {}];\n    \
         add.u64 %addr, %addr, %sel_off;",
        sel_idx * 8
    );
    emit_fr_load(out, "global", dst, "addr", 0);
}

/// `dst = delta(f) = (f*(f - 3)) * (f*(f - 3) + 2)` using `tmp` as scratch.
fn emit_fr_delta(out: &mut String, dst: &str, f: &str, tmp: &str) {
    emit_ch_load(out, tmp, 15); // three
    emit_fr_sub(out, tmp, f, tmp);
    emit_fr_mul_call(out, dst, f, tmp);
    emit_ch_load(out, tmp, 14); // two
    emit_fr_add(out, tmp, dst, tmp);
    emit_fr_mul_call(out, dst, dst, tmp);
}

/// Emit `quotient_eval_wave`: evaluates ALL 6 Dusk PLONK widgets (arithmetic,
/// range, logic, fixed-base ECC, variable-base ECC, permutation + L_1) and
/// divides by `v_H` for 4 cosets (`4 * n` points in parallel) directly in VRAM!
fn emit_quotient_eval_kernel(out: &mut String) {
    let _ = writeln!(
        out,
        r#"
.visible .entry quotient_eval_wave(
    .param .u64 p_wave_evals,
    .param .u64 p_sel_ptrs,
    .param .u64 p_ch,
    .param .u64 p_q_out,
    .param .u32 p_n,
    .param .u32 p_log_n,
    .param .u32 p_k_base,
    .param .u32 p_num_polys
)
{{
    .reg .u32 %tid_g, %r_ctaid, %r_ntid, %n, %log_n, %k_base, %npolys, %total_pts;
    .reg .u32 %local_k, %coset_k, %j, %jw, %mask, %base_p, %idx;
    .reg .u64 %wev, %sel_ptrs, %ch_p, %q_out, %off, %sel_off, %addr, %sp, %has_ib;
    .reg .u64 %ra<4>, %rb<4>, %rc<4>, %rd<4>, %raw<4>, %rbw<4>, %rdw<4>;
    .reg .u64 %rt<4>, %rab<4>, %rx<4>, %ry<4>, %rz<4>, %rp<4>, %rq<4>, %rs<4>, %ru<4>, %rv<4>, %rw<4>;
    .reg .pred %oob, %pz, %p_has;"#
    );
    emit_fr_scratch_decls(out);
    let _ = writeln!(
        out,
        r#"
    mov.u32 %r_ctaid, %ctaid.x;
    mov.u32 %r_ntid, %ntid.x;
    mov.u32 %tid_g, %tid.x;
    mad.lo.u32 %tid_g, %r_ctaid, %r_ntid, %tid_g;

    ld.param.u32 %n, [p_n];
    shl.b32 %total_pts, %n, 2;
    setp.ge.u32 %oob, %tid_g, %total_pts;
    @%oob ret;

    ld.param.u64 %wev, [p_wave_evals];
    ld.param.u64 %sel_ptrs, [p_sel_ptrs];
    ld.param.u64 %ch_p, [p_ch];
    ld.param.u64 %q_out, [p_q_out];
    ld.param.u32 %log_n, [p_log_n];
    ld.param.u32 %k_base, [p_k_base];
    ld.param.u32 %npolys, [p_num_polys];

    shr.u32 %local_k, %tid_g, %log_n;
    add.u32 %coset_k, %k_base, %local_k;
    sub.u32 %mask, %n, 1;
    and.b32 %j, %tid_g, %mask;
    add.u32 %jw, %j, 1;
    and.b32 %jw, %jw, %mask;

    // Flat byte offset into `8 * n` selector arrays: `(coset_k * n + j) * 32`.
    shl.b32 %idx, %coset_k, %log_n;
    add.u32 %idx, %idx, %j;
    mul.wide.u32 %sel_off, %idx, 32;

    // Load wire evaluations `a, b, c, d` at `j` and `a_w, b_w, d_w` at `j_w`.
    // Layout in `wev`: polynomial `(local_k * npolys + p)` is `[z, a, b, c, d, (pi)]`.
    mul.lo.u32 %base_p, %local_k, %npolys;"#
    );

    // Helper closure to load wave poly `(base_p + p_off)` at index reg (`%j` or `%jw`)
    let load_wave_poly =
        |out: &mut String, dst: &str, p_off: u32, idx_reg: &str| {
            let _ = writeln!(
                out,
                "    add.u32 %idx, %base_p, {p_off};\n    \
             shl.b32 %idx, %idx, %log_n;\n    \
             add.u32 %idx, %idx, %{idx_reg};\n    \
             mul.wide.u32 %off, %idx, 32;\n    \
             add.u64 %addr, %wev, %off;"
            );
            emit_fr_load(out, "global", dst, "addr", 0);
        };

    load_wave_poly(out, "ra", 1, "j");
    load_wave_poly(out, "rb", 2, "j");
    load_wave_poly(out, "rc", 3, "j");
    load_wave_poly(out, "rd", 4, "j");
    load_wave_poly(out, "raw", 1, "jw");
    load_wave_poly(out, "rbw", 2, "jw");
    load_wave_poly(out, "rdw", 4, "jw");

    emit_fr_set_zero(out, "rt");
    emit_fr_mul_call(out, "rab", "ra", "rb");

    // 1. Arithmetic widget (`q_arith` = sel[6])
    emit_sel_load(out, "rp", 6);
    emit_fr_is_zero(out, "pz", "rp");
    let _ = writeln!(out, "    @%pz bra L_Q_RANGE;");
    emit_sel_load(out, "rx", 0); // q_m
    emit_fr_mul_call(out, "rx", "rab", "rx");
    emit_sel_load(out, "ry", 1); // q_l
    emit_fr_mul_call(out, "ry", "ra", "ry");
    emit_fr_add(out, "rx", "rx", "ry");
    emit_sel_load(out, "ry", 2); // q_r
    emit_fr_mul_call(out, "ry", "rb", "ry");
    emit_fr_add(out, "rx", "rx", "ry");
    emit_sel_load(out, "ry", 3); // q_o
    emit_fr_mul_call(out, "ry", "rc", "ry");
    emit_fr_add(out, "rx", "rx", "ry");
    emit_sel_load(out, "ry", 4); // q_f
    emit_fr_mul_call(out, "ry", "rd", "ry");
    emit_fr_add(out, "rx", "rx", "ry");
    emit_sel_load(out, "ry", 5); // q_c
    emit_fr_add(out, "rx", "rx", "ry");
    emit_fr_mul_call(out, "rx", "rx", "rp");
    emit_fr_add(out, "rt", "rt", "rx");

    // 2. Range widget (`q_range` = sel[7], optional)
    let _ = writeln!(
        out,
        "L_Q_RANGE:\n    ld.global.u64 %sp, [%sel_ptrs + 56];\n    \
         setp.eq.u64 %pz, %sp, 0;\n    @%pz bra L_Q_LOGIC;\n    \
         add.u64 %addr, %sp, %sel_off;"
    );
    emit_fr_load(out, "global", "rp", "addr", 0);
    emit_fr_is_zero(out, "pz", "rp");
    let _ = writeln!(out, "    @%pz bra L_Q_LOGIC;");
    // b1 = delta(c - 4*d)
    emit_fr_add(out, "ry", "rd", "rd");
    emit_fr_add(out, "ry", "ry", "ry");
    emit_fr_sub(out, "ry", "rc", "ry");
    emit_fr_delta(out, "rx", "ry", "rz");
    // b2 = delta(b - 4*c) * range_k1
    emit_fr_add(out, "ry", "rc", "rc");
    emit_fr_add(out, "ry", "ry", "ry");
    emit_fr_sub(out, "ry", "rb", "ry");
    emit_fr_delta(out, "ry", "ry", "rz");
    emit_ch_load(out, "rz", 17);
    emit_fr_mul_call(out, "ry", "ry", "rz");
    emit_fr_add(out, "rx", "rx", "ry");
    // b3 = delta(a - 4*b) * range_k2
    emit_fr_add(out, "ry", "rb", "rb");
    emit_fr_add(out, "ry", "ry", "ry");
    emit_fr_sub(out, "ry", "ra", "ry");
    emit_fr_delta(out, "ry", "ry", "rz");
    emit_ch_load(out, "rz", 18);
    emit_fr_mul_call(out, "ry", "ry", "rz");
    emit_fr_add(out, "rx", "rx", "ry");
    // b4 = delta(d_w - 4*a) * range_k3
    emit_fr_add(out, "ry", "ra", "ra");
    emit_fr_add(out, "ry", "ry", "ry");
    emit_fr_sub(out, "ry", "rdw", "ry");
    emit_fr_delta(out, "ry", "ry", "rz");
    emit_ch_load(out, "rz", 19);
    emit_fr_mul_call(out, "ry", "ry", "rz");
    emit_fr_add(out, "rx", "rx", "ry");
    // t_1 += rsum * q_range * range_ch
    emit_fr_mul_call(out, "rx", "rx", "rp");
    emit_ch_load(out, "rz", 16);
    emit_fr_mul_call(out, "rx", "rx", "rz");
    emit_fr_add(out, "rt", "rt", "rx");

    // 3. Logic widget (`q_logic` = sel[8], optional)
    let _ = writeln!(
        out,
        "L_Q_LOGIC:\n    ld.global.u64 %sp, [%sel_ptrs + 64];\n    \
         setp.eq.u64 %pz, %sp, 0;\n    @%pz bra L_Q_FIXED;\n    \
         add.u64 %addr, %sp, %sel_off;"
    );
    emit_fr_load(out, "global", "rw", "addr", 0); // q_logic in %rw
    emit_fr_is_zero(out, "pz", "rw");
    let _ = writeln!(out, "    @%pz bra L_Q_FIXED;");
    // rp = la = a_w - 4*a, rq = lb = b_w - 4*b, rs = ld = d_w - 4*d
    emit_fr_add(out, "ry", "ra", "ra");
    emit_fr_add(out, "ry", "ry", "ry");
    emit_fr_sub(out, "rp", "raw", "ry");
    emit_fr_add(out, "ry", "rb", "rb");
    emit_fr_add(out, "ry", "ry", "ry");
    emit_fr_sub(out, "rq", "rbw", "ry");
    emit_fr_add(out, "ry", "rd", "rd");
    emit_fr_add(out, "ry", "ry", "ry");
    emit_fr_sub(out, "rs", "rdw", "ry");
    // rx = c0 + c1 + c2 + c3
    emit_fr_delta(out, "rx", "rp", "rz"); // c0
    emit_fr_delta(out, "ry", "rq", "rz");
    emit_ch_load(out, "rz", 21); // logic_k1
    emit_fr_mul_call(out, "ry", "ry", "rz");
    emit_fr_add(out, "rx", "rx", "ry");
    emit_fr_delta(out, "ry", "rs", "rz");
    emit_ch_load(out, "rz", 22); // logic_k2
    emit_fr_mul_call(out, "ry", "ry", "rz");
    emit_fr_add(out, "rx", "rx", "ry");
    // ru = la * lb; c3 = (c - la_lb) * logic_k3
    emit_fr_mul_call(out, "ru", "rp", "rq");
    emit_fr_sub(out, "ry", "rc", "ru");
    emit_ch_load(out, "rz", 23); // logic_k3
    emit_fr_mul_call(out, "ry", "ry", "rz");
    emit_fr_add(out, "rx", "rx", "ry");
    // delta_xor_and:
    // rv = apb = la + lb; rp = apb_sq = apb * apb; rq = sq_sum = apb_sq - 2*la_lb
    emit_fr_add(out, "rv", "rp", "rq");
    emit_fr_mul_call(out, "rp", "rv", "rv");
    emit_fr_add(out, "ru", "ru", "ru");
    emit_fr_sub(out, "rq", "rp", "ru"); // sq_sum
    // rq = eighteen_sq_sum = 2 * (8 * sq_sum + sq_sum)
    emit_fr_add(out, "ru", "rq", "rq");
    emit_fr_add(out, "ru", "ru", "ru");
    emit_fr_add(out, "ru", "ru", "ru");
    emit_fr_add(out, "rq", "ru", "rq");
    emit_fr_add(out, "rq", "rq", "rq");
    // rv = three_apb = 2*apb + apb
    emit_fr_add(out, "ru", "rv", "rv");
    emit_fr_add(out, "rv", "ru", "rv");
    // ru = nine_apb = 2*three_apb + three_apb; ry = eighteen_apb = 2*nine_apb
    emit_fr_add(out, "ru", "rv", "rv");
    emit_fr_add(out, "ru", "ru", "rv");
    emit_fr_add(out, "ry", "ru", "ru"); // eighteen_apb
    // rp = eighty_one_apb = 3 * (eighteen_apb + nine_apb)
    emit_fr_add(out, "rp", "ry", "ru");
    emit_fr_add(out, "rz", "rp", "rp");
    emit_fr_add(out, "rp", "rz", "rp");
    // inner0 = 4*c - eighteen_apb + 81
    emit_fr_add(out, "ru", "rc", "rc");
    emit_fr_add(out, "ru", "ru", "ru");
    emit_fr_sub(out, "ru", "ru", "ry");
    emit_ch_load(out, "rz", 25); // 81
    emit_fr_add(out, "ru", "ru", "rz");
    // inner1 = c * inner0 + eighteen_sq_sum
    emit_fr_mul_call(out, "ru", "rc", "ru");
    emit_fr_add(out, "ru", "ru", "rq");
    // f_inner = inner1 - eighty_one_apb + 83
    emit_fr_sub(out, "ru", "ru", "rp");
    emit_ch_load(out, "rz", 26); // 83
    emit_fr_add(out, "ru", "ru", "rz");
    // f_val = c * f_inner; 2*f_val in ru
    emit_fr_mul_call(out, "ru", "rc", "ru");
    emit_fr_add(out, "ru", "ru", "ru");
    // rp = three_c = 3*ld; rq = nine_c = 3*three_c
    emit_fr_add(out, "rp", "rs", "rs");
    emit_fr_add(out, "rp", "rp", "rs");
    emit_fr_add(out, "rq", "rp", "rp");
    emit_fr_add(out, "rq", "rq", "rp");
    // e_val = (three_apb + three_c) - 2*f_val (in ry)
    emit_fr_add(out, "ry", "rv", "rp");
    emit_fr_sub(out, "ry", "ry", "ru");
    // b_val = q_c * (nine_c - three_apb) (in rz)
    emit_fr_sub(out, "rz", "rq", "rv");
    emit_sel_load(out, "rp", 5); // q_c
    emit_fr_mul_call(out, "rz", "rp", "rz");
    // c4 = (b_val + e_val) * logic_k4
    emit_fr_add(out, "ry", "rz", "ry");
    emit_ch_load(out, "rz", 24); // logic_k4
    emit_fr_mul_call(out, "ry", "ry", "rz");
    // t_1 += (l_sum * q_logic) * logic_ch
    emit_fr_add(out, "rx", "rx", "ry");
    emit_fr_mul_call(out, "rx", "rw", "rx");
    emit_ch_load(out, "rz", 20); // logic_ch
    emit_fr_mul_call(out, "rx", "rx", "rz");
    emit_fr_add(out, "rt", "rt", "rx");

    // 4. Fixed-base ECC widget (`q_fixed` = sel[9], optional)
    let _ = writeln!(
        out,
        "L_Q_FIXED:\n    ld.global.u64 %sp, [%sel_ptrs + 72];\n    \
         setp.eq.u64 %pz, %sp, 0;\n    @%pz bra L_Q_VAR;\n    \
         add.u64 %addr, %sp, %sel_off;"
    );
    emit_fr_load(out, "global", "rw", "addr", 0); // q_fixed
    emit_fr_is_zero(out, "pz", "rw");
    let _ = writeln!(out, "    @%pz bra L_Q_VAR;");
    // rp = bit = d_w - 2*d; rq = bit^2; rx = bit * (bit^2 - 1)
    emit_fr_add(out, "ry", "rd", "rd");
    emit_fr_sub(out, "rp", "rdw", "ry");
    emit_fr_mul_call(out, "rq", "rp", "rp");
    emit_ch_load(out, "rv", 13); // one
    emit_fr_sub(out, "ry", "rq", "rv");
    emit_fr_mul_call(out, "rx", "rp", "ry");
    // rs = y_alpha = bit_sq * (q_r - 1) + 1
    emit_sel_load(out, "ry", 2); // q_r
    emit_fr_sub(out, "ry", "ry", "rv");
    emit_fr_mul_call(out, "rs", "rq", "ry");
    emit_fr_add(out, "rs", "rs", "rv");
    // ru = x_alpha = bit * q_l
    emit_sel_load(out, "ry", 1); // q_l
    emit_fr_mul_call(out, "ru", "rp", "ry");
    // xy_cons = (bit * q_c - c) * fixed_k1
    emit_sel_load(out, "ry", 5); // q_c
    emit_fr_mul_call(out, "ry", "rp", "ry");
    emit_fr_sub(out, "ry", "ry", "rc");
    emit_ch_load(out, "rz", 28); // fixed_k1
    emit_fr_mul_call(out, "ry", "ry", "rz");
    emit_fr_add(out, "rx", "rx", "ry");
    // rv = d_term = (c * ab) * edwards_d
    emit_fr_mul_call(out, "rv", "rc", "rab");
    emit_ch_load(out, "rz", 31); // edwards_d
    emit_fr_mul_call(out, "rv", "rv", "rz");
    // x_cons = ((a_w + a_w * d_term) - (a * y_alpha + b * x_alpha)) * fixed_k2
    emit_fr_mul_call(out, "ry", "raw", "rv");
    emit_fr_add(out, "ry", "raw", "ry");
    emit_fr_mul_call(out, "rz", "ra", "rs");
    emit_fr_mul_call(out, "rp", "rb", "ru");
    emit_fr_add(out, "rz", "rz", "rp");
    emit_fr_sub(out, "ry", "ry", "rz");
    emit_ch_load(out, "rz", 29); // fixed_k2
    emit_fr_mul_call(out, "ry", "ry", "rz");
    emit_fr_add(out, "rx", "rx", "ry");
    // y_cons = ((b_w - b_w * d_term) - (b * y_alpha + a * x_alpha)) * fixed_k3
    emit_fr_mul_call(out, "ry", "rbw", "rv");
    emit_fr_sub(out, "ry", "rbw", "ry");
    emit_fr_mul_call(out, "rz", "rb", "rs");
    emit_fr_mul_call(out, "rp", "ra", "ru");
    emit_fr_add(out, "rz", "rz", "rp");
    emit_fr_sub(out, "ry", "ry", "rz");
    emit_ch_load(out, "rz", 30); // fixed_k3
    emit_fr_mul_call(out, "ry", "ry", "rz");
    emit_fr_add(out, "rx", "rx", "ry");
    // t_1 += id * q_fixed * fixed_ch
    emit_fr_mul_call(out, "rx", "rx", "rw");
    emit_ch_load(out, "rz", 27); // fixed_ch
    emit_fr_mul_call(out, "rx", "rx", "rz");
    emit_fr_add(out, "rt", "rt", "rx");

    // 5. Variable-base ECC widget (`q_var` = sel[10], optional)
    let _ = writeln!(
        out,
        "L_Q_VAR:\n    ld.global.u64 %sp, [%sel_ptrs + 80];\n    \
         setp.eq.u64 %pz, %sp, 0;\n    @%pz bra L_Q_PI;\n    \
         add.u64 %addr, %sp, %sel_off;"
    );
    emit_fr_load(out, "global", "rw", "addr", 0); // q_var
    emit_fr_is_zero(out, "pz", "rw");
    let _ = writeln!(out, "    @%pz bra L_Q_PI;");
    // rx = xy_cons = a * d - d_w
    emit_fr_mul_call(out, "rx", "ra", "rd");
    emit_fr_sub(out, "rx", "rx", "rdw");
    // rp = y1_x2 = b * c; rq = y1_y2 = b * d; rs = x1_x2 = a * c
    emit_fr_mul_call(out, "rp", "rb", "rc");
    emit_fr_mul_call(out, "rq", "rb", "rd");
    emit_fr_mul_call(out, "rs", "ra", "rc");
    // ru = d_prod = (edwards_d * d_w) * y1_x2
    emit_ch_load(out, "rz", 31); // edwards_d
    emit_fr_mul_call(out, "ru", "rz", "rdw");
    emit_fr_mul_call(out, "ru", "ru", "rp");
    // x3_cons = ((d_w + y1_x2) - (a_w + a_w * d_prod)) * var_k1
    emit_fr_add(out, "ry", "rdw", "rp");
    emit_fr_mul_call(out, "rz", "raw", "ru");
    emit_fr_add(out, "rz", "raw", "rz");
    emit_fr_sub(out, "ry", "ry", "rz");
    emit_ch_load(out, "rz", 33); // var_k1
    emit_fr_mul_call(out, "ry", "ry", "rz");
    emit_fr_add(out, "rx", "rx", "ry");
    // y3_cons = ((y1_y2 + x1_x2) - (b_w - b_w * d_prod)) * var_k2
    emit_fr_add(out, "ry", "rq", "rs");
    emit_fr_mul_call(out, "rz", "rbw", "ru");
    emit_fr_sub(out, "rz", "rbw", "rz");
    emit_fr_sub(out, "ry", "ry", "rz");
    emit_ch_load(out, "rz", 34); // var_k2
    emit_fr_mul_call(out, "ry", "ry", "rz");
    emit_fr_add(out, "rx", "rx", "ry");
    // t_1 += id * q_var * var_ch
    emit_fr_mul_call(out, "rx", "rx", "rw");
    emit_ch_load(out, "rz", 32); // var_ch
    emit_fr_mul_call(out, "rx", "rx", "rz");
    emit_fr_add(out, "rt", "rt", "rx");

    // 6. Public inputs (`pi_ev`, present iff `npolys == 6`)
    let _ = writeln!(
        out,
        "L_Q_PI:\n    setp.eq.u32 %p_has, %npolys, 6;\n    @!%p_has bra L_Q_PERM;"
    );
    load_wave_poly(out, "rx", 5, "j");
    emit_fr_add(out, "rt", "rt", "rx");

    // 7. Permutation widget + L_1 check (`raw, rbw, rdw, rab` are now free!)
    let _ = writeln!(out, "L_Q_PERM:");
    load_wave_poly(out, "raw", 0, "j"); // z_j
    load_wave_poly(out, "rbw", 0, "jw"); // z_jw
    emit_sel_load(out, "rdw", 15); // x = linear_evals[k*n + j]

    let _ = writeln!(
        out,
        "    ld.global.u64 %has_ib, [%ch_p + {}];\n    \
         setp.eq.u64 %pz, %has_ib, 0;\n    @%pz bra L_Q_PERM_NOINV;",
        35 * 32
    );
    // Factored beta path (`has_inv_beta == true`)
    emit_ch_load(out, "rp", 7); // inv_beta
    emit_ch_load(out, "rq", 8); // gamma_over_beta
    emit_fr_mul_call(out, "ra", "ra", "rp");
    emit_fr_add(out, "ra", "ra", "rq"); // a_gb
    emit_fr_mul_call(out, "rb", "rb", "rp");
    emit_fr_add(out, "rb", "rb", "rq"); // b_gb
    emit_fr_mul_call(out, "rc", "rc", "rp");
    emit_fr_add(out, "rc", "rc", "rq"); // c_gb
    emit_fr_mul_call(out, "rd", "rd", "rp");
    emit_fr_add(out, "rd", "rd", "rq"); // d_gb
    // id_term = (a_gb + x)(b_gb + k1*x)(c_gb + k2*x)(d_gb + k3*x) * z_j
    emit_fr_add(out, "rx", "ra", "rdw");
    emit_ch_load(out, "rz", 10); // k1
    emit_fr_mul_call(out, "ry", "rz", "rdw");
    emit_fr_add(out, "ry", "rb", "ry");
    emit_fr_mul_call(out, "rx", "rx", "ry");
    emit_ch_load(out, "rz", 11); // k2
    emit_fr_mul_call(out, "ry", "rz", "rdw");
    emit_fr_add(out, "ry", "rc", "ry");
    emit_ch_load(out, "rz", 12); // k3
    emit_fr_mul_call(out, "ru", "rz", "rdw");
    emit_fr_add(out, "ru", "rd", "ru");
    emit_fr_mul_call(out, "ry", "ry", "ru");
    emit_fr_mul_call(out, "rx", "rx", "ry");
    emit_fr_mul_call(out, "rx", "rx", "raw"); // id_term
    // copy_term = (a_gb + s1)(b_gb + s2)(c_gb + s3)(d_gb + s4) * z_jw
    emit_sel_load(out, "rz", 11);
    emit_fr_add(out, "ry", "ra", "rz");
    emit_sel_load(out, "rz", 12);
    emit_fr_add(out, "ru", "rb", "rz");
    emit_fr_mul_call(out, "ry", "ry", "ru");
    emit_sel_load(out, "rz", 13);
    emit_fr_add(out, "rv", "rc", "rz");
    emit_sel_load(out, "rz", 14);
    emit_fr_add(out, "ru", "rd", "rz");
    emit_fr_mul_call(out, "rv", "rv", "ru");
    emit_fr_mul_call(out, "ry", "ry", "rv");
    emit_fr_mul_call(out, "ry", "ry", "rbw"); // copy_term
    emit_fr_sub(out, "rx", "rx", "ry");
    emit_ch_load(out, "rz", 9); // beta4_alpha
    emit_fr_mul_call(out, "rx", "rx", "rz");
    let _ = writeln!(out, "    bra L_Q_L1;");

    // Unfactored beta == 0 fallback path
    let _ = writeln!(out, "L_Q_PERM_NOINV:");
    emit_ch_load(out, "rq", 6); // gamma
    emit_fr_add(out, "ra", "ra", "rq");
    emit_fr_add(out, "rb", "rb", "rq");
    emit_fr_add(out, "rc", "rc", "rq");
    emit_fr_add(out, "rd", "rd", "rq");
    emit_ch_load(out, "rp", 2); // beta
    emit_fr_mul_call(out, "ry", "rp", "rdw");
    emit_fr_add(out, "rx", "ra", "ry");
    emit_ch_load(out, "rz", 3); // beta_k1
    emit_fr_mul_call(out, "ry", "rz", "rdw");
    emit_fr_add(out, "ry", "rb", "ry");
    emit_fr_mul_call(out, "rx", "rx", "ry");
    emit_ch_load(out, "rz", 4); // beta_k2
    emit_fr_mul_call(out, "ry", "rz", "rdw");
    emit_fr_add(out, "ry", "rc", "ry");
    emit_ch_load(out, "rz", 5); // beta_k3
    emit_fr_mul_call(out, "ru", "rz", "rdw");
    emit_fr_add(out, "ru", "rd", "ru");
    emit_fr_mul_call(out, "ry", "ry", "ru");
    emit_fr_mul_call(out, "rx", "rx", "ry");
    emit_fr_mul_call(out, "rx", "rx", "raw");
    emit_sel_load(out, "rz", 11);
    emit_fr_mul_call(out, "rz", "rp", "rz");
    emit_fr_add(out, "ry", "ra", "rz");
    emit_sel_load(out, "rz", 12);
    emit_fr_mul_call(out, "rz", "rp", "rz");
    emit_fr_add(out, "ru", "rb", "rz");
    emit_fr_mul_call(out, "ry", "ry", "ru");
    emit_sel_load(out, "rz", 13);
    emit_fr_mul_call(out, "rz", "rp", "rz");
    emit_fr_add(out, "rv", "rc", "rz");
    emit_sel_load(out, "rz", 14);
    emit_fr_mul_call(out, "rz", "rp", "rz");
    emit_fr_add(out, "ru", "rd", "rz");
    emit_fr_mul_call(out, "rv", "rv", "ru");
    emit_fr_mul_call(out, "ry", "ry", "rv");
    emit_fr_mul_call(out, "ry", "ry", "rbw");
    emit_fr_sub(out, "rx", "rx", "ry");
    emit_ch_load(out, "rz", 0); // alpha
    emit_fr_mul_call(out, "rx", "rx", "rz");

    // 8. L_1 check + vanishing_inv[coset_k] multiply + store to `q_out[coset_k*n + j]`
    let _ = writeln!(out, "L_Q_L1:");
    emit_sel_load(out, "ry", 16); // l1_evals
    emit_ch_load(out, "rz", 1); // alpha_sq
    emit_fr_mul_call(out, "ry", "ry", "rz");
    emit_ch_load(out, "rz", 13); // one
    emit_fr_sub(out, "rz", "raw", "rz"); // z_j - 1
    emit_fr_mul_call(out, "ry", "rz", "ry");
    emit_fr_add(out, "rx", "rx", "ry");
    emit_fr_add(out, "rt", "rt", "rx");
    // Load vanishing_inv[coset_k] from ch_p + (36 + coset_k) * 32
    let _ = writeln!(
        out,
        "    add.u32 %idx, %coset_k, 36;\n    \
         mul.wide.u32 %off, %idx, 32;\n    \
         add.u64 %addr, %ch_p, %off;"
    );
    emit_fr_load(out, "global", "ry", "addr", 0);
    emit_fr_mul_call(out, "rt", "rt", "ry");
    let _ = writeln!(out, "    add.u64 %addr, %q_out, %sel_off;");
    emit_fr_store(out, "global", "addr", 0, "rt");
    let _ = writeln!(out, "    ret;\n}}");

    // Kernel: quotient_radix8_combine — performs the 8-point inverse DFT +
    // coset inv_twiddle + gen_inv_t_scaled + gen_inv_sn directly in VRAM,
    // writing `[t_low, t_mid, t_high, t_fourth[0..n], extra_v4[0..16]]`
    // (halving DtoH bandwidth from 16 MB -> 8 MB at 2^16!) and setting atomic
    // flags bit 0 (`v7 != 0`, unsatisfied) or bit 1 (`extra_tail` overflow).
    let _ = writeln!(
        out,
        r#"
.visible .entry quotient_radix8_combine(
    .param .u64 p_iffts,
    .param .u64 p_inv_twiddles,
    .param .u64 p_gen_inv_t,
    .param .u64 p_consts,
    .param .u64 p_t_out,
    .param .u64 p_flags,
    .param .u32 p_n
)
{{
    .reg .u32 %t, %r_ctaid, %r_ntid, %n, %idx, %dummy;
    .reg .u64 %iffts, %inv_tw, %gen_t, %consts, %t_out, %flags, %off, %addr;
    .reg .u64 %ra<4>, %rb<4>, %rc<4>, %rd<4>;
    .reg .u64 %eva<4>, %evb<4>, %evc<4>, %evd<4>;
    .reg .u64 %oda<4>, %odb<4>, %odc<4>, %odd<4>;
    .reg .u64 %rw<4>, %rgt<4>, %rv<4>, %rsn<4>;
    .reg .pred %oob, %pz, %p_lt16;"#
    );
    emit_fr_scratch_decls(out);
    let _ = writeln!(
        out,
        r#"
    mov.u32 %r_ctaid, %ctaid.x;
    mov.u32 %r_ntid, %ntid.x;
    mov.u32 %t, %tid.x;
    mad.lo.u32 %t, %r_ctaid, %r_ntid, %t;

    ld.param.u32 %n, [p_n];
    setp.ge.u32 %oob, %t, %n;
    @%oob ret;

    ld.param.u64 %iffts, [p_iffts];
    ld.param.u64 %inv_tw, [p_inv_twiddles];
    ld.param.u64 %gen_t, [p_gen_inv_t];
    ld.param.u64 %consts, [p_consts];
    ld.param.u64 %t_out, [p_t_out];
    ld.param.u64 %flags, [p_flags];"#
    );

    let emit_u_load = |out: &mut String, dst: &str, k: u32| {
        if k == 0 {
            let _ = writeln!(
                out,
                "    mul.wide.u32 %off, %t, 32;\n    add.u64 %addr, %iffts, %off;"
            );
            emit_fr_load(out, "global", dst, "addr", 0);
        } else {
            let _ = writeln!(
                out,
                "    mad.lo.u32 %idx, {k}, %n, %t;\n    mul.wide.u32 %off, %idx, 32;\n    add.u64 %addr, %iffts, %off;"
            );
            emit_fr_load(out, "global", dst, "addr", 0);
            let tw_k = k - 1;
            let _ = writeln!(
                out,
                "    mad.lo.u32 %idx, {tw_k}, %n, %t;\n    mul.wide.u32 %off, %idx, 32;\n    add.u64 %addr, %inv_tw, %off;"
            );
            emit_fr_load(out, "global", "rv", "addr", 0);
            emit_fr_mul_call(out, dst, dst, "rv");
        }
    };

    // Load w2 from consts[1] (+32)
    emit_fr_load(out, "global", "rw", "consts", 32);

    // Even 4-point DFT: u0, u4, u2, u6 -> eva, evb, evc, evd (ev0..ev3)
    emit_u_load(out, "ra", 0);
    emit_u_load(out, "rb", 4);
    emit_fr_add(out, "rc", "ra", "rb"); // e0
    emit_fr_sub(out, "rd", "ra", "rb"); // e1
    emit_u_load(out, "ra", 2);
    emit_u_load(out, "rb", 6);
    emit_fr_add(out, "eva", "ra", "rb"); // e2 (temp)
    emit_fr_sub(out, "evb", "ra", "rb");
    emit_fr_mul_call(out, "evb", "evb", "rw"); // e3 = (u2 - u6) * w2
    emit_fr_sub(out, "evc", "rc", "eva"); // ev2 = e0 - e2
    emit_fr_add(out, "eva", "rc", "eva"); // ev0 = e0 + e2
    emit_fr_sub(out, "evd", "rd", "evb"); // ev3 = e1 - e3
    emit_fr_add(out, "evb", "rd", "evb"); // ev1 = e1 + e3

    // Odd 4-point DFT: u1, u5, u3, u7 -> oda, odb, odc, odd (od0..od3)
    emit_u_load(out, "ra", 1);
    emit_u_load(out, "rb", 5);
    emit_fr_add(out, "rc", "ra", "rb"); // o0
    emit_fr_sub(out, "rd", "ra", "rb"); // o1
    emit_u_load(out, "ra", 3);
    emit_u_load(out, "rb", 7);
    emit_fr_add(out, "oda", "ra", "rb"); // o2 (temp)
    emit_fr_sub(out, "odb", "ra", "rb");
    emit_fr_mul_call(out, "odb", "odb", "rw"); // o3 = (u3 - u7) * w2
    emit_fr_sub(out, "odc", "rc", "oda");
    emit_fr_mul_call(out, "odc", "odc", "rw"); // od2 = (o0 - o2) * w2
    emit_fr_add(out, "oda", "rc", "oda"); // od0 = o0 + o2
    emit_fr_add(out, "ra", "rd", "odb"); // o1 + o3
    emit_fr_sub(out, "rb", "rd", "odb"); // o1 - o3
    emit_fr_load(out, "global", "rw", "consts", 0); // w1
    emit_fr_mul_call(out, "odb", "ra", "rw"); // od1 = (o1 + o3) * w1
    emit_fr_load(out, "global", "rw", "consts", 64); // w3
    emit_fr_mul_call(out, "odd", "rb", "rw"); // od3 = (o1 - o3) * w3

    // Load gen_inv_t_scaled[t]
    let _ = writeln!(
        out,
        "    mul.wide.u32 %off, %t, 32;\n    add.u64 %addr, %gen_t, %off;"
    );
    emit_fr_load(out, "global", "rgt", "addr", 0);

    // Emit v0..v3 scaled by gt * sn[0..4] to t_out[s*n + t]
    for (s, (ev, od)) in [
        ("eva", "oda"),
        ("evb", "odb"),
        ("evc", "odc"),
        ("evd", "odd"),
    ]
    .into_iter()
    .enumerate()
    {
        let sn_off = (3 + s) * 32;
        emit_fr_add(out, "rv", ev, od);
        emit_fr_mul_call(out, "rv", "rv", "rgt");
        emit_fr_load(out, "global", "rsn", "consts", sn_off);
        emit_fr_mul_call(out, "rv", "rv", "rsn");
        let _ = writeln!(
            out,
            "    mad.lo.u32 %idx, {s}, %n, %t;\n    mul.wide.u32 %off, %idx, 32;\n    add.u64 %addr, %t_out, %off;"
        );
        emit_fr_store(out, "global", "addr", 0, "rv");
    }

    // v4 = ev0 - od0 -> store scaled into t_out[4*n + t] if t < 16, else flag overflow if nonzero
    emit_fr_sub(out, "rv", "eva", "oda");
    let _ = writeln!(
        out,
        "    setp.lt.u32 %p_lt16, %t, 16;\n    @!%p_lt16 bra L_R8_V4_HI;"
    );
    emit_fr_mul_call(out, "rv", "rv", "rgt");
    emit_fr_load(out, "global", "rsn", "consts", 7 * 32); // sn[4]
    emit_fr_mul_call(out, "rv", "rv", "rsn");
    let _ = writeln!(
        out,
        "    mad.lo.u32 %idx, 4, %n, %t;\n    mul.wide.u32 %off, %idx, 32;\n    add.u64 %addr, %t_out, %off;"
    );
    emit_fr_store(out, "global", "addr", 0, "rv");
    let _ = writeln!(out, "    bra L_R8_V5;\nL_R8_V4_HI:");
    emit_fr_is_zero(out, "pz", "rv");
    let _ = writeln!(
        out,
        "    @%pz bra L_R8_V5;\n    atom.global.or.b32 %dummy, [%flags], 2;"
    );

    // v5 = ev1 - od1, v6 = ev2 - od2 -> flag overflow (bit 1) if nonzero
    let _ = writeln!(out, "L_R8_V5:");
    emit_fr_sub(out, "rv", "evb", "odb");
    emit_fr_is_zero(out, "pz", "rv");
    let _ = writeln!(
        out,
        "    @%pz bra L_R8_V6;\n    atom.global.or.b32 %dummy, [%flags], 2;"
    );
    let _ = writeln!(out, "L_R8_V6:");
    emit_fr_sub(out, "rv", "evc", "odc");
    emit_fr_is_zero(out, "pz", "rv");
    let _ = writeln!(
        out,
        "    @%pz bra L_R8_V7;\n    atom.global.or.b32 %dummy, [%flags], 2;"
    );

    // v7 = ev3 - od3 -> flag unsatisfied (bit 0) if nonzero
    let _ = writeln!(out, "L_R8_V7:");
    emit_fr_sub(out, "rv", "evd", "odd");
    emit_fr_is_zero(out, "pz", "rv");
    let _ = writeln!(
        out,
        "    @%pz ret;\n    atom.global.or.b32 %dummy, [%flags], 1;\n    ret;\n}}"
    );
}

/// Emit `g1_decompress_and_subgroup_check`:
/// Parallel BLS12-381 compressed G1 decoding (48-byte big-endian Zcash format)
/// with canonicality check (`x < p`), square root `y = (x^3 + 4)^((p+1)/4)`,
/// lexicographic sign selection, and exact prime-order G1 subgroup verification
/// `[z^2]P == -phi(P)` (`|z| = 0xd201000000010000`), outputting 96-byte
/// `blst_p1_affine` Montgomery limbs and a `u8` status (`0 = ok`, `1 = invalid`).
fn emit_g1_decompress_kernel(out: &mut String) {
    const FP_FOUR_MONT: [u64; 6] = [
        0xaa27_0000_000c_fff3,
        0x53cc_0032_fc34_000a,
        0x478f_e97a_6b0a_807f,
        0xb1d3_7ebe_e6ba_24d7,
        0x8ec9_733b_bf78_ab2f,
        0x09d6_4551_3d83_de7e,
    ];
    const FP_R2_MONT: [u64; 6] = [
        0xf4df_1f34_1c34_1746,
        0x0a76_e6a6_09d1_04f1,
        0x8de5_476c_4c95_b6d5,
        0x67eb_88a9_939d_83c0,
        0x9a79_3e85_b519_952d,
        0x1198_8fe5_92ca_e3aa,
    ];
    const FP_HALF_P: [u64; 6] = [
        0xdcff_7fff_ffff_d555,
        0x0f55_ffff_58a9_ffff,
        0xb398_6950_7b58_7b12,
        0xb23b_a5c2_79c2_895f,
        0x258d_d3db_21a5_d66b,
        0x0d00_88f5_1cbf_f34d,
    ];
    let p = FP_P;

    let _ = writeln!(
        out,
        r#"
.visible .entry g1_decompress_and_subgroup_check(
    .param .u64 p_in,
    .param .u64 p_params,
    .param .u64 p_out,
    .param .u64 p_status,
    .param .u32 p_count
)
{{
    .reg .u32 %tid_g, %r_ctaid, %r_ntid, %count, %u_lo, %u_hi, %flags, %sort_bit;
    .reg .u32 %wi, %bit, %li, %sh, %dig, %tmp32;
    .reg .u64 %in_p, %prm_p, %out_p, %st_p, %off, %addr, %wtmp, %ebits, %zabs;
    .reg .u64 %px<6>, %py<6>, %ax<6>, %ay<6>, %azz<6>, %azzz<6>;
    .reg .u64 %bx<6>, %by<6>, %qx<6>, %qy<6>, %qzz<6>, %qzzz<6>;
    .reg .u64 %ta<6>, %tb<6>, %tc<6>, %td<6>, %te<6>;
    .reg .pred %oob, %pz0, %pz1, %p_inf, %p_comp, %p_lex, %p_sort, %p_flip, %p_loop, %p_d1, %p_d2, %p_d3, %p_add;"#
    );
    emit_fp_scratch_decls(out);
    let _ = writeln!(
        out,
        r#"
    mov.u32 %r_ctaid, %ctaid.x;
    mov.u32 %r_ntid, %ntid.x;
    mov.u32 %tid_g, %tid.x;
    mad.lo.u32 %tid_g, %r_ctaid, %r_ntid, %tid_g;

    ld.param.u32 %count, [p_count];
    setp.ge.u32 %oob, %tid_g, %count;
    @%oob ret;

    ld.param.u64 %in_p, [p_in];
    ld.param.u64 %prm_p, [p_params];
    ld.param.u64 %out_p, [p_out];
    ld.param.u64 %st_p, [p_status];

    // 1. Load 48 big-endian bytes into little-endian limbs px0..px5.
    mul.wide.u32 %off, %tid_g, 48;
    add.u64 %addr, %in_p, %off;"#
    );
    for i in 0..6usize {
        let byte_off = (5 - i) * 8;
        let _ = writeln!(
            out,
            "    ld.global.u64 %wtmp, [%addr + {byte_off}];\n    \
             mov.b64 {{%u_lo, %u_hi}}, %wtmp;\n    \
             prmt.b32 %u_lo, %u_lo, 0, 0x0123;\n    \
             prmt.b32 %u_hi, %u_hi, 0, 0x0123;\n    \
             mov.b64 %px{i}, {{%u_hi, %u_lo}};"
        );
    }
    let _ = writeln!(
        out,
        r#"
    // Extract top 3 flag bits from px5: bit2=comp(0x80), bit1=inf(0x40), bit0=sort(0x20).
    shr.u64 %wtmp, %px5, 61;
    cvt.u32.u64 %flags, %wtmp;
    and.b64 %px5, %px5, 0x1fffffffffffffff;

    // Compression bit must be set.
    and.b32 %tmp32, %flags, 4;
    setp.eq.u32 %p_comp, %tmp32, 0;
    @%p_comp bra L_DEC_INVALID;

    // Check infinity flag.
    and.b32 %tmp32, %flags, 2;
    setp.ne.u32 %p_inf, %tmp32, 0;
    @!%p_inf bra L_DEC_FINITE;

    // Infinity requires sort bit == 0 and all 381 payload bits == 0.
    and.b32 %tmp32, %flags, 1;
    setp.ne.u32 %pz0, %tmp32, 0;
    @%pz0 bra L_DEC_INVALID;"#
    );
    emit_fp_is_zero(out, "pz0", "px");
    let _ = writeln!(
        out,
        "    @!%pz0 bra L_DEC_INVALID;\n    \
         mul.wide.u32 %off, %tid_g, 96;\n    \
         add.u64 %addr, %out_p, %off;"
    );
    emit_fp_set_zero(out, "px");
    emit_fp_store(out, "global", "addr", 0, "px");
    emit_fp_store(out, "global", "addr", 48, "px");
    let _ = writeln!(
        out,
        "    cvt.u64.u32 %off, %tid_g;\n    \
         add.u64 %addr, %st_p, %off;\n    \
         st.global.u8 [%addr], 0;\n    \
         ret;\n\nL_DEC_FINITE:\n    \
         and.b32 %sort_bit, %flags, 1;"
    );

    // Canonical check: px < p (must borrow when computing px - p).
    let _ = writeln!(
        out,
        "    mov.u64 %mz0, 0;\n    \
         sub.cc.u64  %mt0, %px0, 0x{:016x};\n    \
         subc.cc.u64 %mt1, %px1, 0x{:016x};\n    \
         subc.cc.u64 %mt2, %px2, 0x{:016x};\n    \
         subc.cc.u64 %mt3, %px3, 0x{:016x};\n    \
         subc.cc.u64 %mt4, %px4, 0x{:016x};\n    \
         subc.cc.u64 %mt5, %px5, 0x{:016x};\n    \
         subc.u64    %mborrow, %mz0, %mz0;\n    \
         setp.eq.u64 %pz0, %mborrow, 0;\n    \
         @%pz0 bra L_DEC_INVALID;",
        p[0], p[1], p[2], p[3], p[4], p[5]
    );

    // 2. Convert x to Montgomery form and compute rhs = x^3 + 4 in tc.
    for (i, &limb) in FP_R2_MONT.iter().enumerate() {
        let _ = writeln!(out, "    mov.u64 %ta{i}, 0x{limb:016x};");
    }
    emit_fp_mul_call(out, "px", "px", "ta");
    emit_fp_mul_call(out, "ta", "px", "px");
    emit_fp_mul_call(out, "ta", "ta", "px");
    for (i, &limb) in FP_FOUR_MONT.iter().enumerate() {
        let _ = writeln!(out, "    mov.u64 %tb{i}, 0x{limb:016x};");
    }
    emit_fp_add(out, "tc", "ta", "tb");

    // 3. Compute y = rhs^((p + 1) / 4) via 2-bit register window (tc=rhs, td=rhs^2, te=rhs^3).
    emit_fp_mul_call(out, "td", "tc", "tc");
    emit_fp_mul_call(out, "te", "td", "tc");
    emit_fp_set_one(out, "py");
    let _ = writeln!(
        out,
        r#"
    mov.u32 %wi, 190;
L_DEC_SQRT:
    setp.eq.u32 %p_loop, %wi, 0;
    @%p_loop bra L_DEC_SQRT_CHK;
    sub.u32 %wi, %wi, 1;"#
    );
    emit_fp_mul_call(out, "py", "py", "py");
    emit_fp_mul_call(out, "py", "py", "py");
    let _ = writeln!(
        out,
        r#"    shl.b32 %bit, %wi, 1;
    shr.u32 %li, %bit, 6;
    and.b32 %sh, %bit, 63;
    mul.wide.u32 %off, %li, 8;
    add.u64 %addr, %prm_p, %off;
    ld.global.u64 %ebits, [%addr];
    shr.u64 %ebits, %ebits, %sh;
    cvt.u32.u64 %dig, %ebits;
    and.b32 %dig, %dig, 3;
    setp.eq.u32 %p_d1, %dig, 1;
    @!%p_d1 bra L_DEC_D2;"#
    );
    emit_fp_mul_call(out, "py", "py", "tc");
    let _ = writeln!(
        out,
        "    bra L_DEC_SQRT;\nL_DEC_D2:\n    \
         setp.eq.u32 %p_d2, %dig, 2;\n    \
         @!%p_d2 bra L_DEC_D3;"
    );
    emit_fp_mul_call(out, "py", "py", "td");
    let _ = writeln!(
        out,
        "    bra L_DEC_SQRT;\nL_DEC_D3:\n    \
         setp.eq.u32 %p_d3, %dig, 3;\n    \
         @!%p_d3 bra L_DEC_SQRT;"
    );
    emit_fp_mul_call(out, "py", "py", "te");
    let _ = writeln!(out, "    bra L_DEC_SQRT;\n\nL_DEC_SQRT_CHK:");

    // 4. Verify y^2 == rhs.
    emit_fp_mul_call(out, "ta", "py", "py");
    let _ = writeln!(
        out,
        "    xor.b64 %mt0, %ta0, %tc0;\n    \
         xor.b64 %mt1, %ta1, %tc1;\n    \
         xor.b64 %mt2, %ta2, %tc2;\n    \
         xor.b64 %mt3, %ta3, %tc3;\n    \
         xor.b64 %mt4, %ta4, %tc4;\n    \
         xor.b64 %mt5, %ta5, %tc5;\n    \
         or.b64 %mt0, %mt0, %mt1;\n    \
         or.b64 %mt2, %mt2, %mt3;\n    \
         or.b64 %mt4, %mt4, %mt5;\n    \
         or.b64 %mt0, %mt0, %mt2;\n    \
         or.b64 %mt0, %mt0, %mt4;\n    \
         setp.ne.u64 %pz0, %mt0, 0;\n    \
         @%pz0 bra L_DEC_INVALID;"
    );

    // 5. Lexicographic sign check: convert y to canonical via fp_mul(py, 1),
    // compare against (p - 1) / 2, and negate py if (lex_larger != sort_bit).
    let _ = writeln!(
        out,
        "    mov.u64 %tb0, 1;\n    \
         mov.u64 %tb1, 0;\n    \
         mov.u64 %tb2, 0;\n    \
         mov.u64 %tb3, 0;\n    \
         mov.u64 %tb4, 0;\n    \
         mov.u64 %tb5, 0;"
    );
    emit_fp_mul_call(out, "ta", "py", "tb");
    for (i, &limb) in FP_HALF_P.iter().enumerate() {
        let _ = writeln!(out, "    mov.u64 %tb{i}, 0x{limb:016x};");
    }
    let _ = writeln!(
        out,
        "    mov.u64 %mz0, 0;\n    \
         sub.cc.u64  %mt0, %tb0, %ta0;\n    \
         subc.cc.u64 %mt1, %tb1, %ta1;\n    \
         subc.cc.u64 %mt2, %tb2, %ta2;\n    \
         subc.cc.u64 %mt3, %tb3, %ta3;\n    \
         subc.cc.u64 %mt4, %tb4, %ta4;\n    \
         subc.cc.u64 %mt5, %tb5, %ta5;\n    \
         subc.u64    %mborrow, %mz0, %mz0;\n    \
         setp.ne.u64 %p_lex, %mborrow, 0;\n    \
         setp.ne.u32 %p_sort, %sort_bit, 0;\n    \
         xor.pred %p_flip, %p_lex, %p_sort;\n    \
         @!%p_flip bra L_DEC_SG;"
    );
    emit_fp_neg(out, "py", "py");

    // 6. Subgroup check: [z^2]P == -phi(P) = (beta * px, -py),
    // where |z| = 0xd201000000010000 (bit 63 set, 63 doublings + 5 adds twice).
    let _ = writeln!(out, "\nL_DEC_SG:");
    emit_fp_copy(out, "ax", "px");
    emit_fp_copy(out, "ay", "py");
    emit_fp_set_one(out, "azz");
    emit_fp_set_one(out, "azzz");
    let _ = writeln!(
        out,
        r#"    mov.u64 %zabs, 0xd201000000010000;
    mov.u32 %bit, 63;
L_DEC_SG1:
    setp.eq.u32 %p_loop, %bit, 0;
    @%p_loop bra L_DEC_SG2_INIT;
    sub.u32 %bit, %bit, 1;"#
    );
    emit_xyzz_dbl_inline(out, "SG1_DBL", "ax", "ay", "azz", "azzz", true);
    let _ = writeln!(
        out,
        "    shr.u64 %wtmp, %zabs, %bit;\n    \
         and.b64 %wtmp, %wtmp, 1;\n    \
         setp.ne.u64 %p_add, %wtmp, 0;\n    \
         @!%p_add bra L_DEC_SG1;"
    );
    emit_xyzz_add_affine_inline(
        out, "SG1_ADD", "ax", "ay", "azz", "azzz", "px", "py",
    );
    let _ = writeln!(out, "    bra L_DEC_SG1;\n\nL_DEC_SG2_INIT:");

    // Save Q = [|z|]P in qx..qzzz and compute R = [|z|]Q = [z^2]P.
    emit_fp_copy(out, "qx", "ax");
    emit_fp_copy(out, "qy", "ay");
    emit_fp_copy(out, "qzz", "azz");
    emit_fp_copy(out, "qzzz", "azzz");
    let _ = writeln!(
        out,
        r#"    mov.u32 %bit, 63;
L_DEC_SG2:
    setp.eq.u32 %p_loop, %bit, 0;
    @%p_loop bra L_DEC_SG_CHK;
    sub.u32 %bit, %bit, 1;"#
    );
    emit_xyzz_dbl_inline(out, "SG2_DBL", "ax", "ay", "azz", "azzz", true);
    let _ = writeln!(
        out,
        "    shr.u64 %wtmp, %zabs, %bit;\n    \
         and.b64 %wtmp, %wtmp, 1;\n    \
         setp.ne.u64 %p_add, %wtmp, 0;\n    \
         @!%p_add bra L_DEC_SG2;"
    );
    emit_fp_copy(out, "bx", "qx");
    emit_fp_copy(out, "by", "qy");
    emit_xyzz_add_inline(
        out, "SG2_ADD", "ax", "ay", "azz", "azzz", "bx", "by", "qzz", "qzzz",
    );
    let _ = writeln!(out, "    bra L_DEC_SG2;\n\nL_DEC_SG_CHK:");

    // R = (ax / azz, ay / azzz) must be finite and equal (beta * px, -py).
    emit_fp_is_zero(out, "pz0", "azz");
    let _ = writeln!(out, "    @%pz0 bra L_DEC_INVALID;");
    emit_fp_load(out, "global", "ta", "prm_p", 48); // beta_mont
    emit_fp_mul_call(out, "tb", "px", "ta");
    emit_fp_mul_call(out, "tb", "tb", "azz");
    let _ = writeln!(
        out,
        "    xor.b64 %mt0, %ax0, %tb0;\n    \
         xor.b64 %mt1, %ax1, %tb1;\n    \
         xor.b64 %mt2, %ax2, %tb2;\n    \
         xor.b64 %mt3, %ax3, %tb3;\n    \
         xor.b64 %mt4, %ax4, %tb4;\n    \
         xor.b64 %mt5, %ax5, %tb5;\n    \
         or.b64 %mt0, %mt0, %mt1;\n    \
         or.b64 %mt2, %mt2, %mt3;\n    \
         or.b64 %mt4, %mt4, %mt5;\n    \
         or.b64 %mt0, %mt0, %mt2;\n    \
         or.b64 %mt0, %mt0, %mt4;\n    \
         setp.ne.u64 %pz0, %mt0, 0;\n    \
         @%pz0 bra L_DEC_INVALID;"
    );
    emit_fp_neg(out, "ta", "py");
    emit_fp_mul_call(out, "tb", "ta", "azzz");
    let _ = writeln!(
        out,
        "    xor.b64 %mt0, %ay0, %tb0;\n    \
         xor.b64 %mt1, %ay1, %tb1;\n    \
         xor.b64 %mt2, %ay2, %tb2;\n    \
         xor.b64 %mt3, %ay3, %tb3;\n    \
         xor.b64 %mt4, %ay4, %tb4;\n    \
         xor.b64 %mt5, %ay5, %tb5;\n    \
         or.b64 %mt0, %mt0, %mt1;\n    \
         or.b64 %mt2, %mt2, %mt3;\n    \
         or.b64 %mt4, %mt4, %mt5;\n    \
         or.b64 %mt0, %mt0, %mt2;\n    \
         or.b64 %mt0, %mt0, %mt4;\n    \
         setp.ne.u64 %pz0, %mt0, 0;\n    \
         @%pz0 bra L_DEC_INVALID;"
    );

    // 7. Store validated Montgomery affine point (px, py) and status = 0.
    let _ = writeln!(
        out,
        "    mul.wide.u32 %off, %tid_g, 96;\n    \
         add.u64 %addr, %out_p, %off;"
    );
    emit_fp_store(out, "global", "addr", 0, "px");
    emit_fp_store(out, "global", "addr", 48, "py");
    let _ = writeln!(
        out,
        "    cvt.u64.u32 %off, %tid_g;\n    \
         add.u64 %addr, %st_p, %off;\n    \
         st.global.u8 [%addr], 0;\n    \
         ret;\n\nL_DEC_INVALID:\n    \
         cvt.u64.u32 %off, %tid_g;\n    \
         add.u64 %addr, %st_p, %off;\n    \
         st.global.u8 [%addr], 1;\n    \
         ret;\n}}"
    );
}

/// Generate the complete PTX source module for BLS12-381 GPU MSM & NTT kernels.
pub(crate) fn generate_msm_ptx() -> String {
    let mut out = String::with_capacity(96 * 1024);
    let _ = writeln!(out, ".version 7.5\n.target sm_70\n.address_size 64\n");
    emit_fp_mul_val_func(&mut out);
    emit_fr_mul_val_func(&mut out);
    emit_msm_kernels(&mut out);
    emit_ntt_kernels(&mut out);
    emit_quotient_eval_kernel(&mut out);
    emit_g1_decompress_kernel(&mut out);
    out
}
