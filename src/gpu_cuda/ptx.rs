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

/// Generate the complete PTX source module for BLS12-381 GPU MSM & NTT kernels.
pub(crate) fn generate_msm_ptx() -> String {
    let mut out = String::with_capacity(64 * 1024);
    let _ = writeln!(out, ".version 7.5\n.target sm_70\n.address_size 64\n");
    emit_fp_mul_val_func(&mut out);
    emit_msm_kernels(&mut out);
    emit_ntt_kernels(&mut out);
    out
}
