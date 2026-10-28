//! End-to-end evaluator tests.
//!
//! Small synthetic kernels (1-2 threads) validate each mechanism cheaply;
//! the Harris reduction kernels from the paper benchmarks (64-128 threads,
//! ~100 instructions each) validate real equivalence checking.

use std::collections::BTreeSet;
use std::path::PathBuf;

use volta_analysis::driver::{EquivOutcome, analyze_kernel, check_output_equivalence};
use volta_analysis::{AnalysisError, LowerError};

/// Check equivalence along the reference run's output arrays.
fn check_equiv(a: &AnalysisOutput, b: &AnalysisOutput) -> EquivOutcome {
    let arrays: Vec<String> = a.outputs.iter().map(|(n, _)| n.clone()).collect();
    check_output_equivalence(a, b, &arrays).unwrap()
}
use volta_analysis::eval::{
    AnalysisConfig, AnalysisOutput, ArrayDef, ArrayKind, EvalError, ParamValue,
};
use volta_frontend::ascii::AsAscii;
use volta_frontend::ast::Module;
use volta_frontend::parse::Parser;

fn parse(src: &str) -> Module {
    let ascii = src.as_bytes().as_ascii_slice().expect("ascii source");
    Parser::new(ascii)
        .parse_module()
        .unwrap_or_else(|e| panic!("parse error: {:?}", e.error))
}

/// The volta_bench paper-benchmark kernel tree (PTX + CUDA sources).
const KERNELS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../volta_bench/kernels");

fn parse_file(rel: &str) -> Module {
    let path = PathBuf::from(KERNELS_DIR).join(rel);
    let src =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    parse(&src)
}

const HEADER: &str = ".version 8.0\n.target sm_80\n.address_size 64\n\n";

fn wrap(body: &str) -> String {
    format!("{}{}", HEADER, body)
}

/// in/out f32 arrays at fixed bases.
fn in_out_config(threads: u32, len: u64) -> AnalysisConfig {
    let mut config = AnalysisConfig::new((threads, 1, 1));
    config.arrays = vec![
        ArrayDef {
            name: "in".to_string(),
            base: 0x10000,
            elem_width: 4,
            len,
            kind: ArrayKind::Input,
        },
        ArrayDef {
            name: "out".to_string(),
            base: 0x20000,
            elem_width: 4,
            len,
            kind: ArrayKind::Output,
        },
    ];
    config.params = vec![
        ParamValue::ArrayPtr("in".to_string()),
        ParamValue::ArrayPtr("out".to_string()),
    ];
    config
}

fn display_output(output: &AnalysisOutput, array: &str, index: u64) -> String {
    let (_, elems) = output
        .outputs
        .iter()
        .find(|(n, _)| n == array)
        .expect("output array");
    let (_, e) = elems
        .iter()
        .find(|(i, _)| *i == index)
        .expect("element written");
    output.arena.display_expr(*e)
}

// =========================================================================
// Synthetic kernels: one mechanism each
// =========================================================================

/// Two threads write the same shared address without synchronization.
#[test]
fn test_shared_write_write_race() {
    let src = wrap(
        ".visible .entry k()
{
    .reg .b32 %r<3>;
    .shared .align 4 .b8 sdata[8];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, sdata;
    st.shared.u32 [%r2], %r1;
    ret;
}
",
    );
    let module = parse(&src);
    let config = AnalysisConfig::new((2, 1, 1));
    let err = analyze_kernel(&module, None, config).unwrap_err();
    assert!(
        matches!(err, AnalysisError::Eval(EvalError::DataRace { .. })),
        "expected data race, got: {}",
        err
    );
}

/// Neighbor exchange through shared memory, correctly synchronized:
/// out[tid] = in[tid ^ 1].
const SWAP_BODY: &str = ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<3>;
    .reg .b32 %r<7>;
    .reg .b64 %rd<7>;
    .shared .align 4 .b8 sdata[8];

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    mov.u32 %r1, %tid.x;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd3, %r2;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    mov.u32 %r3, sdata;
    add.s32 %r4, %r3, %r2;
    st.shared.f32 [%r4], %f1;
    BARRIER
    xor.b32 %r5, %r1, 1;
    shl.b32 %r5, %r5, 2;
    add.s32 %r6, %r3, %r5;
    ld.shared.f32 %f2, [%r6];
    add.s64 %rd5, %rd2, %rd3;
    st.global.f32 [%rd5], %f2;
    ret;
}
";

#[test]
fn test_shared_exchange_with_barrier() {
    let src = wrap(&SWAP_BODY.replace("BARRIER", "bar.sync 0;"));
    let module = parse(&src);
    let output = analyze_kernel(&module, None, in_out_config(2, 2)).unwrap();
    assert_eq!(display_output(&output, "out", 0), "in[1]");
    assert_eq!(display_output(&output, "out", 1), "in[0]");
    assert_eq!(output.stats.block_syncs, 2);
}

#[test]
fn test_shared_exchange_without_barrier_races() {
    let src = wrap(&SWAP_BODY.replace("BARRIER", ""));
    let module = parse(&src);
    let err = analyze_kernel(&module, None, in_out_config(2, 2)).unwrap_err();
    assert!(
        matches!(err, AnalysisError::Eval(EvalError::DataRace { .. })),
        "expected data race, got: {}",
        err
    );
}

/// A store through the `.extern .shared` window must not be visible through
/// a static `.shared` variable: the CUDA ABI places the dynamic segment
/// after all static allocations, so `buf` and `lut` are disjoint. The `lut`
/// read is therefore undefined and surfaces as `UndefinedOutput` when it
/// reaches the output array (with the two aliased at shared offset 0 it
/// would instead read back the 42 stored through `buf`).
#[test]
fn test_extern_shared_store_invisible_through_static_shared() {
    let src = wrap(
        ".extern .shared .align 16 .b8 buf[];
.visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<3>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<2>;
    .shared .align 4 .f32 lut[64];

    ld.param.u64 %rd1, [k_param_1];
    mov.u32 %r1, buf;
    mov.f32 %f1, 0f42280000;
    st.shared.f32 [%r1], %f1;
    mov.u32 %r2, lut;
    ld.shared.f32 %f2, [%r2];
    st.global.f32 [%rd1], %f2;
    ret;
}
",
    );
    let module = parse(&src);
    let mut config = in_out_config(1, 1);
    config.dynamic_shared_bytes = 16;
    let err = analyze_kernel(&module, None, config).unwrap_err();
    assert!(
        matches!(err, AnalysisError::Eval(EvalError::UndefinedOutput { .. })),
        "expected undefined output (buf and lut must not alias), got: {}",
        err
    );
}

/// Unsynchronized accesses to the extern window from one thread and to a
/// static `.shared` variable from another are not a race: the regions are
/// disjoint. With the extern window wrongly aliased at offset 0 this
/// reported a false write-write race.
#[test]
fn test_extern_vs_static_shared_no_false_race() {
    let src = wrap(
        ".extern .shared .align 16 .b8 buf[];
.visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .pred %p<3>;
    .reg .f32 %f<4>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<2>;
    .shared .align 4 .f32 lut[64];

    ld.param.u64 %rd1, [k_param_1];
    mov.u32 %r1, %tid.x;
    setp.eq.s32 %p1, %r1, 0;
    setp.eq.s32 %p2, %r1, 1;
    mov.u32 %r2, buf;
    mov.f32 %f1, 0f42280000;
@%p1 st.shared.f32 [%r2], %f1;
    mov.u32 %r3, lut;
    mov.f32 %f2, 0f40E00000;
@%p2 st.shared.f32 [%r3], %f2;
@%p2 ld.shared.f32 %f3, [%r3];
@%p2 st.global.f32 [%rd1], %f3;
    ret;
}
",
    );
    let module = parse(&src);
    let mut config = in_out_config(2, 1);
    config.dynamic_shared_bytes = 16;
    let output = analyze_kernel(&module, None, config)
        .expect("disjoint extern/static shared accesses must not race");
    assert_eq!(display_output(&output, "out", 0), "7");
}

/// Extern-only kernels are unchanged by extern-window placement: the window
/// still starts at shared offset 0, a store/load roundtrip works, the
/// missing-size launch error still fires, and dynamic_shared_bytes still
/// bounds the window.
#[test]
fn test_extern_only_shared_placement_and_bounds() {
    let body = ".extern .shared .align 16 .b8 buf[];
.visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<3>;
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;

    ld.param.u64 %rd1, [k_param_1];
    mov.u32 %r1, buf;
    mov.f32 %f1, 0f42280000;
    st.shared.f32 [%r1OFFSET], %f1;
    ld.shared.f32 %f2, [%r1];
    st.global.f32 [%rd1], %f2;
    ret;
}
";
    // In-bounds roundtrip at buf[0].
    let src = wrap(&body.replace("OFFSET", ""));
    let module = parse(&src);
    let mut config = in_out_config(1, 1);
    config.dynamic_shared_bytes = 16;
    let output = analyze_kernel(&module, None, config).expect("in-bounds extern access");
    assert_eq!(display_output(&output, "out", 0), "42");

    // Without dynamic_shared_bytes the launch is rejected.
    let module = parse(&src);
    let err = analyze_kernel(&module, None, in_out_config(1, 1)).unwrap_err();
    assert!(
        matches!(
            &err,
            AnalysisError::Eval(EvalError::Config { message })
                if message.contains("dynamic_shared_bytes")
        ),
        "expected missing dynamic_shared_bytes error, got: {}",
        err
    );

    // A store ending one byte past the window end is out of bounds.
    let src = wrap(&body.replace("OFFSET", "+16"));
    let module = parse(&src);
    let mut config = in_out_config(1, 1);
    config.dynamic_shared_bytes = 16;
    let err = analyze_kernel(&module, None, config).unwrap_err();
    assert!(
        matches!(err, AnalysisError::Eval(EvalError::OutOfBounds { .. })),
        "expected out-of-bounds, got: {}",
        err
    );
}

/// Threads waiting on different barrier ids never fire: deadlock.
#[test]
fn test_mismatched_barriers_deadlock() {
    let src = wrap(
        ".visible .entry k()
{
    .reg .pred %p<2>;
    .reg .b32 %r<2>;

    mov.u32 %r1, %tid.x;
    setp.eq.s32 %p1, %r1, 0;
    @%p1 bra $L1;
    bar.sync 0;
    bra $L2;
$L1:
    bar.sync 1;
$L2:
    ret;
}
",
    );
    let module = parse(&src);
    let err = analyze_kernel(&module, None, AnalysisConfig::new((2, 1, 1))).unwrap_err();
    assert!(
        matches!(err, AnalysisError::Eval(EvalError::Deadlock { .. })),
        "expected deadlock, got: {}",
        err
    );
}

/// A thread that exits counts as having arrived at the barrier (paper's
/// Sync rule allows `return`), so this does NOT deadlock.
#[test]
fn test_exited_thread_releases_barrier() {
    let src = wrap(
        ".visible .entry k()
{
    .reg .pred %p<2>;
    .reg .b32 %r<2>;

    mov.u32 %r1, %tid.x;
    setp.eq.s32 %p1, %r1, 0;
    @%p1 ret;
    bar.sync 0;
    ret;
}
",
    );
    let module = parse(&src);
    analyze_kernel(&module, None, AnalysisConfig::new((2, 1, 1))).unwrap();
}

/// An uninitialized shared read is tolerated during execution (the paper's
/// race example depends on it), but an output computed from one is an error.
#[test]
fn test_uninitialized_shared_read_flows_to_output() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;
    .shared .align 4 .b8 sdata[8];

    ld.param.u64 %rd2, [k_param_1];
    mov.u32 %r1, sdata;
    ld.shared.u32 %r2, [%r1];
    st.global.u32 [%rd2], %r2;
    ret;
}
",
    );
    let module = parse(&src);
    let err = analyze_kernel(&module, None, in_out_config(1, 1)).unwrap_err();
    assert!(
        matches!(err, AnalysisError::Eval(EvalError::UndefinedOutput { .. })),
        "expected undefined output, got: {}",
        err
    );
}

#[test]
fn test_out_of_bounds_shared_access() {
    let src = wrap(
        ".visible .entry k()
{
    .reg .b32 %r<3>;
    .shared .align 4 .b8 sdata[8];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, sdata;
    st.shared.u32 [%r2+64], %r1;
    ret;
}
",
    );
    let module = parse(&src);
    let err = analyze_kernel(&module, None, AnalysisConfig::new((1, 1, 1))).unwrap_err();
    assert!(
        matches!(err, AnalysisError::Eval(EvalError::OutOfBounds { .. })),
        "expected out-of-bounds, got: {}",
        err
    );
}

// =========================================================================
// Natural alignment (PTX ISA 6.4.1: "The address must be naturally aligned
// to a multiple of the access size"; misaligned accesses are undefined
// behavior on hardware, so the evaluator rejects them)
// =========================================================================

/// A 4-byte load at an address 2 mod 4 is misaligned.
#[test]
fn test_misaligned_scalar_load() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<2>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    ld.global.f32 %f1, [%rd1+2];
    st.global.f32 [%rd2], %f1;
    ret;
}
",
    );
    let module = parse(&src);
    let err = analyze_kernel(&module, None, in_out_config(1, 4)).unwrap_err();
    assert!(
        matches!(
            err,
            AnalysisError::Eval(EvalError::Misaligned { required: 4, .. })
        ),
        "expected misaligned access, got: {}",
        err
    );
}

/// A 4-byte store at an address 2 mod 4 is misaligned (write side of the
/// same chokepoint).
#[test]
fn test_misaligned_scalar_store() {
    let src = wrap(
        ".visible .entry k()
{
    .reg .b32 %r<3>;
    .shared .align 4 .b8 sdata[16];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, sdata;
    st.shared.u32 [%r2+2], %r1;
    ret;
}
",
    );
    let module = parse(&src);
    let err = analyze_kernel(&module, None, AnalysisConfig::new((1, 1, 1))).unwrap_err();
    assert!(
        matches!(
            err,
            AnalysisError::Eval(EvalError::Misaligned { required: 4, .. })
        ),
        "expected misaligned access, got: {}",
        err
    );
}

/// 2-byte accesses at addresses 2 mod 4 are *naturally aligned* (natural
/// alignment is relative to the access size, not any fixed width): the
/// ordinary f16 access pattern must keep working.
#[test]
fn test_aligned_u16_access_at_2_mod_4() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .b16 %rs<2>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    ld.global.u16 %rs1, [%rd1+2];
    st.global.u16 [%rd2+2], %rs1;
    ret;
}
",
    );
    let module = parse(&src);
    let mut config = AnalysisConfig::new((1, 1, 1));
    config.arrays = vec![
        ArrayDef {
            name: "in".to_string(),
            base: 0x10000,
            elem_width: 2,
            len: 4,
            kind: ArrayKind::Input,
        },
        ArrayDef {
            name: "out".to_string(),
            base: 0x20000,
            elem_width: 2,
            len: 4,
            kind: ArrayKind::Output,
        },
    ];
    config.params = vec![
        ParamValue::ArrayPtr("in".to_string()),
        ParamValue::ArrayPtr("out".to_string()),
    ];
    let output = analyze_kernel(&module, None, config).expect("aligned u16 access");
    assert_eq!(display_output(&output, "out", 1), "in[1]");
}

/// The access size of a vector load is the *total* bytes accessed
/// (ld.v4.f32 is one 16-byte access), so an address 4 mod 16 is misaligned
/// even though every element taken alone is 4-aligned.
#[test]
fn test_misaligned_vector_load() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<5>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd1+4];
    st.global.f32 [%rd2], %f1;
    ret;
}
",
    );
    let module = parse(&src);
    let err = analyze_kernel(&module, None, in_out_config(1, 8)).unwrap_err();
    assert!(
        matches!(
            err,
            AnalysisError::Eval(EvalError::Misaligned { required: 16, .. })
        ),
        "expected misaligned vector access, got: {}",
        err
    );
}

/// The same vector load at a 16-byte boundary is legal.
#[test]
fn test_aligned_vector_load() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<5>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd1];
    st.global.f32 [%rd2], %f4;
    ret;
}
",
    );
    let module = parse(&src);
    let output = analyze_kernel(&module, None, in_out_config(1, 8)).expect("aligned v4 load");
    assert_eq!(display_output(&output, "out", 0), "in[3]");
}

/// One full-warp ldmatrix.x4 body; each lane supplies the row address
/// `sdata + tid*16 + extra` (lane i*8+r owns row r of matrix i).
fn ldmatrix_body(extra: u32) -> String {
    format!(
        ".visible .entry k()
{{
    .reg .b32 %r<10>;
    .shared .align 16 .b8 sdata[1024];

    mov.u32 %r1, %tid.x;
    shl.b32 %r2, %r1, 4;
    mov.u32 %r3, sdata;
    add.s32 %r4, %r3, %r2;
    add.s32 %r4, %r4, {};
    ldmatrix.sync.aligned.x4.m8n8.shared.b16 {{%r5, %r6, %r7, %r8}}, [%r4];
    ret;
}}
",
        extra
    )
}

/// ldmatrix row addresses are legal at 16-byte boundaries: each 8x8 b16
/// matrix row is fetched by four lanes as one 16-byte access.
#[test]
fn test_aligned_ldmatrix() {
    let module = parse(&wrap(&ldmatrix_body(0)));
    analyze_kernel(&module, None, AnalysisConfig::new((32, 1, 1))).expect("aligned ldmatrix");
}

/// A row address 8 mod 16 violates ldmatrix's 16-byte row alignment
/// (PTX ISA 9.7.14.5.15) even though each lane's own 4-byte read would be
/// 4-aligned.
#[test]
fn test_misaligned_ldmatrix_row_address() {
    let module = parse(&wrap(&ldmatrix_body(8)));
    let err = analyze_kernel(&module, None, AnalysisConfig::new((32, 1, 1))).unwrap_err();
    assert!(
        matches!(
            err,
            AnalysisError::Eval(EvalError::Misaligned { required: 16, .. })
        ),
        "expected misaligned ldmatrix row, got: {}",
        err
    );
}

/// One full-warp wmma.load.a body with base `sdata + extra` and the given
/// stride (in f16 elements).
fn wmma_load_body(extra: u32, stride: u32) -> String {
    format!(
        ".visible .entry k()
{{
    .reg .b32 %r<12>;
    .shared .align 32 .b8 sdata[2048];

    mov.u32 %r1, sdata;
    add.s32 %r1, %r1, {};
    mov.u32 %r2, {};
    wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 \
         {{%r3, %r4, %r5, %r6, %r7, %r8, %r9, %r10}}, [%r1], %r2;
    ret;
}}
",
        extra, stride
    )
}

/// One full-warp wmma.store.d body with base `sdata + extra` and the given
/// stride (in f32 elements).
fn wmma_store_body(extra: u32, stride: u32) -> String {
    format!(
        ".visible .entry k()
{{
    .reg .b32 %r<3>;
    .reg .f32 %f<9>;
    .shared .align 32 .b8 sdata[4096];

    mov.f32 %f1, 0f3F800000;
    mov.f32 %f2, 0f3F800000;
    mov.f32 %f3, 0f3F800000;
    mov.f32 %f4, 0f3F800000;
    mov.f32 %f5, 0f3F800000;
    mov.f32 %f6, 0f3F800000;
    mov.f32 %f7, 0f3F800000;
    mov.f32 %f8, 0f3F800000;
    mov.u32 %r1, sdata;
    add.s32 %r1, %r1, {};
    mov.u32 %r2, {};
    wmma.store.d.sync.aligned.row.m16n16k16.shared.f32 \
         [%r1], {{%f1, %f2, %f3, %f4, %f5, %f6, %f7, %f8}}, %r2;
    ret;
}}
",
        extra, stride
    )
}

/// wmma with a 32-byte-aligned base and the default stride (16 elements)
/// is legal, for both the f16 a-fragment load and the f32 d-fragment store.
#[test]
fn test_aligned_wmma_load_and_store() {
    let module = parse(&wrap(&wmma_load_body(0, 16)));
    analyze_kernel(&module, None, AnalysisConfig::new((32, 1, 1))).expect("aligned wmma.load");

    let module = parse(&wrap(&wmma_store_body(0, 16)));
    analyze_kernel(&module, None, AnalysisConfig::new((32, 1, 1))).expect("aligned wmma.store");
}

/// A stride *larger* than the default is explicitly allowed (a submatrix
/// of a larger matrix) as long as it keeps rows aligned.
#[test]
fn test_wmma_stride_above_default_is_legal() {
    let module = parse(&wrap(&wmma_load_body(0, 32)));
    analyze_kernel(&module, None, AnalysisConfig::new((32, 1, 1))).expect("stride-32 wmma.load");
}

/// nvcc's bank-conflict skew (`__shared__ half tile[..][16 + 8]`, the
/// corpus's Conv2D-opt) yields a 24-element f16 stride: 48 bytes, not a
/// multiple of the 32-byte fragment size but 16-byte aligned, which is
/// the contract nvcc actually compiles against. It must stay legal.
#[test]
fn test_wmma_skewed_stride_is_legal() {
    let module = parse(&wrap(&wmma_load_body(0, 24)));
    analyze_kernel(&module, None, AnalysisConfig::new((32, 1, 1))).expect("stride-24 wmma.load");
}

/// A wmma base address 8 mod 32 violates the 32-byte fragment alignment
/// (PTX ISA 9.7.14.4.2) even though each f16 element read is 2-aligned.
#[test]
fn test_wmma_load_base_misaligned() {
    let module = parse(&wrap(&wmma_load_body(8, 16)));
    let err = analyze_kernel(&module, None, AnalysisConfig::new((32, 1, 1))).unwrap_err();
    assert!(
        matches!(
            err,
            AnalysisError::Eval(EvalError::Misaligned { required: 32, .. })
        ),
        "expected misaligned wmma base, got: {}",
        err
    );
}

/// The store side enforces the same base alignment.
#[test]
fn test_wmma_store_base_misaligned() {
    let module = parse(&wrap(&wmma_store_body(8, 16)));
    let err = analyze_kernel(&module, None, AnalysisConfig::new((32, 1, 1))).unwrap_err();
    assert!(
        matches!(
            err,
            AnalysisError::Eval(EvalError::Misaligned { required: 32, .. })
        ),
        "expected misaligned wmma store base, got: {}",
        err
    );
}

/// A stride below the leading dimension (16 elements for m16n16k16) is
/// undefined behavior by itself (PTX ISA 9.7.14.4.3).
#[test]
fn test_wmma_stride_below_default() {
    let module = parse(&wrap(&wmma_load_body(0, 8)));
    let err = analyze_kernel(&module, None, AnalysisConfig::new((32, 1, 1))).unwrap_err();
    assert!(
        matches!(
            err,
            AnalysisError::Eval(EvalError::WmmaStrideTooSmall {
                stride: 8,
                minimum: 16,
                ..
            })
        ),
        "expected wmma stride below default, got: {}",
        err
    );
}

/// A stride that is legal in *size* but whose byte pitch breaks the
/// 16-byte stride granularity leaves rows past the first misaligned:
/// 20 f16 elements = 40 bytes.
#[test]
fn test_wmma_stride_misaligned() {
    let module = parse(&wrap(&wmma_load_body(0, 20)));
    let err = analyze_kernel(&module, None, AnalysisConfig::new((32, 1, 1))).unwrap_err();
    assert!(
        matches!(
            err,
            AnalysisError::Eval(EvalError::Misaligned { required: 16, .. })
        ),
        "expected misaligned wmma stride, got: {}",
        err
    );
}

/// The issue #4 reproducer, parametrized: thread 0 fills `sdata[0..16)`
/// with one word and `sdata[16..32)` with another, then every lane loads
/// an 8x8 b16 row through `ldmatrix.x1 ... [%r2+imm]` with
/// `%r2 = sdata + reg`, and thread 0 exports lane 0's word. The lowering
/// used to drop `imm`.
fn ldmatrix_offset_body(reg: i64, imm: i64) -> String {
    format!(
        ".visible .entry k(
    .param .u64 out_param
)
{{
    .reg .pred %p<2>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<3>;
    .shared .align 16 .b8 sdata[32];

    mov.u32 %r1, %tid.x;
    setp.eq.s32 %p1, %r1, 0;
    mov.u32 %r2, sdata;
    @%p1 mov.u32 %r3, 111111111;
    @%p1 st.shared.u32 [%r2], %r3;
    @%p1 mov.u32 %r4, 222222222;
    @%p1 st.shared.u32 [%r2+16], %r4;
    bar.sync 0;
    add.s32 %r2, %r2, {};
    ldmatrix.sync.aligned.x1.m8n8.shared.b16 {{%r5}}, [%r2+{}];
    @%p1 ld.param.u64 %rd1, [out_param];
    @%p1 cvta.to.global.u64 %rd1, %rd1;
    @%p1 st.global.u32 [%rd1], %r5;
    ret;
}}
",
        reg, imm
    )
}

/// `ldmatrix` applies the immediate of its `[reg + imm]` address operand:
/// positive, nvcc's `+-N` negative spelling, and absent.
#[test]
fn test_ldmatrix_applies_address_offset() {
    for (reg, imm, expected) in [
        (0, 16, "222222222"),
        (16, -16, "111111111"),
        (0, 0, "111111111"),
        (16, 0, "222222222"),
    ] {
        let module = parse(&wrap(&ldmatrix_offset_body(reg, imm)));
        let mut config = out_only_config(4, 1);
        config.block_dim = (32, 1, 1);
        let out = analyze_kernel(&module, None, config)
            .unwrap_or_else(|e| panic!("ldmatrix [%r2+{}] with %r2 = sdata+{}: {}", imm, reg, e));
        assert_eq!(
            display_output(&out, "out", 0),
            expected,
            "ldmatrix [%r2+{}] with %r2 = sdata+{}",
            imm,
            reg
        );
    }
}

/// `wmma.store` applies the immediate of its address operand: a d-fragment
/// stored at `[out+64]` with the default stride lands on elements
/// 16..272 of `out`, not 0..256.
#[test]
fn test_wmma_store_applies_address_offset() {
    let module = parse(&wrap(
        ".visible .entry k(
    .param .u64 out_param
)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;
    .reg .f32 %f<9>;

    ld.param.u64 %rd1, [out_param];
    cvta.to.global.u64 %rd1, %rd1;
    mov.f32 %f1, 0f3F800000;
    mov.f32 %f2, 0f3F800000;
    mov.f32 %f3, 0f3F800000;
    mov.f32 %f4, 0f3F800000;
    mov.f32 %f5, 0f3F800000;
    mov.f32 %f6, 0f3F800000;
    mov.f32 %f7, 0f3F800000;
    mov.f32 %f8, 0f3F800000;
    mov.u32 %r1, 16;
    wmma.store.d.sync.aligned.row.m16n16k16.global.f32 \
         [%rd1+64], {%f1, %f2, %f3, %f4, %f5, %f6, %f7, %f8}, %r1;
    ret;
}
",
    ));
    let mut config = out_only_config(4, 272);
    config.block_dim = (32, 1, 1);
    let out = analyze_kernel(&module, None, config).expect("wmma.store with offset");
    let (_, elems) = out
        .outputs
        .iter()
        .find(|(n, _)| n == "out")
        .expect("output array");
    let written: Vec<u64> = elems.iter().map(|(i, _)| *i).collect();
    assert_eq!(written, (16..272).collect::<Vec<u64>>());
}

/// `wmma.load` applies the immediate of its address operand: a c-fragment
/// loaded from `[in+64]` with the default stride is exactly `in[16..272)`,
/// one element per fragment register across the warp.
#[test]
fn test_wmma_load_applies_address_offset() {
    let module = parse(&wrap(
        ".visible .entry k(
    .param .u64 in_param,
    .param .u64 out_param
)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<5>;
    .reg .f32 %f<9>;

    ld.param.u64 %rd1, [in_param];
    ld.param.u64 %rd2, [out_param];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    mov.u32 %r1, 16;
    wmma.load.c.sync.aligned.row.m16n16k16.global.f32 \
         {%f1, %f2, %f3, %f4, %f5, %f6, %f7, %f8}, [%rd1+64], %r1;
    mov.u32 %r2, %tid.x;
    mul.wide.u32 %rd3, %r2, 32;
    add.s64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f1;
    st.global.f32 [%rd4+4], %f2;
    st.global.f32 [%rd4+8], %f3;
    st.global.f32 [%rd4+12], %f4;
    st.global.f32 [%rd4+16], %f5;
    st.global.f32 [%rd4+20], %f6;
    st.global.f32 [%rd4+24], %f7;
    st.global.f32 [%rd4+28], %f8;
    ret;
}
",
    ));
    let out = analyze_kernel(&module, None, in_out_config(32, 272)).expect("wmma.load with offset");
    let (_, elems) = out
        .outputs
        .iter()
        .find(|(n, _)| n == "out")
        .expect("output array");
    assert_eq!(elems.len(), 256);
    let loaded: BTreeSet<String> = elems
        .iter()
        .map(|(_, e)| out.arena.display_expr(*e))
        .collect();
    let expected: BTreeSet<String> = (16..272).map(|i| format!("in[{}]", i)).collect();
    assert_eq!(loaded, expected);
}

/// An array base that is not a multiple of the element width is rejected
/// when the analysis is configured, not per access.
#[test]
fn test_config_rejects_misaligned_array_base() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0
)
{
    .reg .b64 %rd<2>;

    ld.param.u64 %rd1, [k_param_0];
    ret;
}
",
    );
    let module = parse(&src);
    let mut config = AnalysisConfig::new((1, 1, 1));
    config.arrays = vec![ArrayDef {
        name: "in".to_string(),
        base: 2,
        elem_width: 4,
        len: 4,
        kind: ArrayKind::Input,
    }];
    config.params = vec![ParamValue::ArrayPtr("in".to_string())];
    let err = analyze_kernel(&module, None, config).unwrap_err();
    assert!(
        matches!(err, AnalysisError::Eval(EvalError::Config { .. })),
        "expected config error, got: {}",
        err
    );
}

/// nvcc's u16 magic-number division (Conv2D-opt's index cache): the
/// immediate -17873 is the u16 constant 47663 = ceil(2^19/11); operands of
/// `mul.wide.u16` must be reinterpreted as unsigned, not consumed at their
/// producer's signedness.
#[test]
fn test_mul_wide_u16_magic_division() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .b16 %rs<5>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_1];
    mov.u16 %rs1, 14;
    mul.wide.u16 %r1, %rs1, -17873;
    shr.u32 %r2, %r1, 19;
    cvt.u16.u32 %rs2, %r2;
    mul.lo.s16 %rs3, %rs2, 11;
    sub.s16 %rs4, %rs1, %rs3;
    cvt.u32.u16 %r3, %rs4;
    st.global.u32 [%rd1], %r2;
    st.global.u32 [%rd1+4], %r3;
    ret;
}
",
    );
    let module = parse(&src);
    let output = analyze_kernel(&module, None, in_out_config(1, 2)).expect("analysis");
    // 14 / 11 = 1, 14 % 11 = 3.
    assert_eq!(display_output(&output, "out", 0), "1");
    assert_eq!(display_output(&output, "out", 1), "3");
}

/// `mul.hi.u32` must zero-extend a canonically-signed operand: the high
/// half of 0xFFFFFFFF * 2 is 1, not -1's sign fill.
#[test]
fn test_mul_hi_u32_negative_operand() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_1];
    mov.u32 %r1, -1;
    mul.hi.u32 %r2, %r1, 2;
    st.global.u32 [%rd1], %r2;
    ret;
}
",
    );
    let module = parse(&src);
    let output = analyze_kernel(&module, None, in_out_config(1, 1)).expect("analysis");
    assert_eq!(display_output(&output, "out", 0), "1");
}

/// Output-only config: one `out` array of `len` `elem_width`-byte
/// elements, passed as the kernel's single parameter.
fn out_only_config(elem_width: u64, len: u64) -> AnalysisConfig {
    let mut config = AnalysisConfig::new((1, 1, 1));
    config.arrays = vec![ArrayDef {
        name: "out".to_string(),
        base: 0x20000,
        elem_width,
        len,
        kind: ArrayKind::Output,
    }];
    config.params = vec![ParamValue::ArrayPtr("out".to_string())];
    config
}

/// A byte-element global scratch array (kind Input, so it is never
/// compared as an output) plus a u32 `out` array.
fn u8_scratch_config() -> AnalysisConfig {
    let mut config = AnalysisConfig::new((1, 1, 1));
    config.arrays = vec![
        ArrayDef {
            name: "scratch".to_string(),
            base: 0x30000,
            elem_width: 1,
            len: 8,
            kind: ArrayKind::Input,
        },
        ArrayDef {
            name: "out".to_string(),
            base: 0x20000,
            elem_width: 4,
            len: 2,
            kind: ArrayKind::Output,
        },
    ];
    config.params = vec![
        ParamValue::ArrayPtr("scratch".to_string()),
        ParamValue::ArrayPtr("out".to_string()),
    ];
    config
}

/// Value-boundary canonicalization: `mov.u32 %r, -1` and `not.b32 %r, 0`
/// leave the same hardware value in the register, so both kernels must
/// export the same canonical constant (this was a false DIFF: the mov
/// side exported -1, the not side 4294967295).
#[test]
fn test_mov_neg1_equivalent_to_not_zero() {
    let mov_kernel = wrap(
        ".visible .entry k(
    .param .u64 k_param_0
)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;

    ld.param.u64 %rd1, [k_param_0];
    mov.u32 %r1, -1;
    st.global.u32 [%rd1], %r1;
    ret;
}
",
    );
    let not_kernel = wrap(
        ".visible .entry k(
    .param .u64 k_param_0
)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;

    ld.param.u64 %rd1, [k_param_0];
    not.b32 %r1, 0;
    st.global.u32 [%rd1], %r1;
    ret;
}
",
    );
    let a = analyze_kernel(&parse(&mov_kernel), None, out_only_config(4, 1)).unwrap();
    let b = analyze_kernel(&parse(&not_kernel), None, out_only_config(4, 1)).unwrap();
    assert_eq!(display_output(&a, "out", 0), "4294967295");
    assert_eq!(display_output(&b, "out", 0), "4294967295");
    assert!(matches!(check_equiv(&a, &b), EquivOutcome::Equivalent));
}

/// Value-boundary canonicalization at setp and selp: comparing a
/// symbolic value against the immediate -1 at s32 and against a
/// not.b32-computed 4294967295 must build identical comparison nodes
/// (the computed operand is reinterpreted at the instruction type), and
/// likewise the selp arms `-1` and the computed 4294967295 at b32.
/// Previously both instructions used raw operands, so this pair was a
/// false DIFF: one side's expressions carried -1 where the other's
/// carried 4294967295.
#[test]
fn test_setp_selp_canonicalize_operands() {
    let literal = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    ld.global.u32 %r1, [%rd1];
    setp.eq.s32 %p1, %r1, -1;
    selp.b32 %r2, -1, 0, %p1;
    st.global.u32 [%rd2], %r2;
    ret;
}
",
    );
    let computed = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    ld.global.u32 %r1, [%rd1];
    not.b32 %r3, 0;
    setp.eq.s32 %p1, %r1, %r3;
    selp.b32 %r2, %r3, 0, %p1;
    st.global.u32 [%rd2], %r2;
    ret;
}
",
    );
    let a = analyze_kernel(&parse(&literal), None, in_out_config(1, 1)).unwrap();
    let b = analyze_kernel(&parse(&computed), None, in_out_config(1, 1)).unwrap();
    let da = display_output(&a, "out", 0);
    let db = display_output(&b, "out", 0);
    assert_eq!(da, db);
    assert!(da.contains("== -1"), "compare canonicalized at s32: {}", da);
    assert!(
        da.contains("? 4294967295"),
        "select arm canonicalized at b32: {}",
        da
    );
    assert!(matches!(check_equiv(&a, &b), EquivOutcome::Equivalent));
}

/// `st.u8` keeps only the low byte of the stored value (hardware chops
/// 300 to 44); `ld.u8` zero-extends it back.
#[test]
fn test_store_u8_truncates_to_low_bits() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    mov.u32 %r1, 44;
    add.s32 %r2, %r1, 256;
    st.global.u8 [%rd1], %r2;
    ld.global.u8 %r3, [%rd1];
    st.global.u32 [%rd2], %r3;
    ret;
}
",
    );
    let output = analyze_kernel(&parse(&src), None, u8_scratch_config()).unwrap();
    assert_eq!(display_output(&output, "out", 0), "44");
}

/// `ld.s8` sign-extends the memory byte per the load type: the byte 0xFF
/// reads back as -1, and storing that at 32-bit width exports the
/// canonical unsigned pattern 4294967295 - the same constant a kernel
/// that stores -1 directly exports.
#[test]
fn test_load_s8_sign_extends() {
    let via_byte = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    mov.u32 %r1, 255;
    st.global.u8 [%rd1], %r1;
    ld.global.s8 %r2, [%rd1];
    st.global.s32 [%rd2], %r2;
    ret;
}
",
    );
    let direct = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd2, [k_param_1];
    mov.u32 %r1, -1;
    st.global.u32 [%rd2], %r1;
    ret;
}
",
    );
    let a = analyze_kernel(&parse(&via_byte), None, u8_scratch_config()).unwrap();
    let b = analyze_kernel(&parse(&direct), None, u8_scratch_config()).unwrap();
    assert_eq!(display_output(&a, "out", 0), "4294967295");
    assert_eq!(display_output(&b, "out", 0), "4294967295");
    assert!(matches!(check_equiv(&a, &b), EquivOutcome::Equivalent));
}

/// cvt reads its source at the *source* format (ISA Table 15: extension
/// follows the source format): `and.b32` leaves the unsigned-canonical
/// 4294967288 in the register, and `cvt.s64.s32` must reinterpret it as
/// the s32 -8 before widening; mirrored, `cvt.u64.u32` of a `sub.s32`
/// result -8 must widen the u32 reading 4294967288.
#[test]
fn test_cvt_reinterprets_at_source_format() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0
)
{
    .reg .b32 %r<5>;
    .reg .b64 %rd<4>;

    ld.param.u64 %rd1, [k_param_0];
    mov.u32 %r1, -5;
    and.b32 %r2, %r1, -4;
    cvt.s64.s32 %rd2, %r2;
    st.global.s64 [%rd1], %rd2;
    mov.u32 %r3, 3;
    sub.s32 %r4, %r3, 11;
    cvt.u64.u32 %rd3, %r4;
    st.global.u64 [%rd1+8], %rd3;
    ret;
}
",
    );
    let output = analyze_kernel(&parse(&src), None, out_only_config(8, 2)).unwrap();
    // (-5 & -4) at 32 bits is 0xFFFFFFF8; sign-extended through s32: -8.
    assert_eq!(display_output(&output, "out", 0), "-8");
    // 3 - 11 = -8; zero-extended through u32: 4294967288.
    assert_eq!(display_output(&output, "out", 1), "4294967288");
}

/// A symbolic value stored below its source register's width would need
/// a truncation node we do not model: loud error, not silent nonsense.
#[test]
fn test_symbolic_sub_register_store_rejected() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    ld.global.u32 %r1, [%rd1];
    st.global.u8 [%rd2], %r1;
    ret;
}
",
    );
    let err = analyze_kernel(&parse(&src), None, in_out_config(1, 1)).unwrap_err();
    assert!(
        matches!(
            &err,
            AnalysisError::Eval(EvalError::Unsupported { what, .. })
                if what.contains("symbolic value stored")
        ),
        "expected symbolic sub-register store rejection, got: {}",
        err
    );
}

/// A symbolic scalar loaded at a type narrower than the destination
/// register would need an extension node we do not model: loud error.
#[test]
fn test_symbolic_sub_register_load_rejected() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    ld.global.s8 %r1, [%rd1];
    st.global.u32 [%rd2], %r1;
    ret;
}
",
    );
    // The byte array serves as the kernel's input here: loading one of
    // its symbolic elements at s8 into a 32-bit register would need an
    // extension.
    let err = analyze_kernel(&parse(&src), None, u8_scratch_config()).unwrap_err();
    assert!(
        matches!(
            &err,
            AnalysisError::Eval(EvalError::Unsupported { what, .. })
                if what.contains("symbolic value loaded")
        ),
        "expected symbolic sub-register load rejection, got: {}",
        err
    );
}

/// A packed f16x2 pair fills a 32-bit register; storing it at a
/// sub-4-byte width would stuff the whole two-half value into a 2-byte
/// granule (a shape the memory model never anticipates): loud error,
/// not silent nonsense. The pair is built through the sanctioned path
/// (two adjacent 2-byte stores read back at 4 bytes).
#[test]
fn test_pair_sub_width_store_rejected() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .b16 %rs<3>;
    .reg .b32 %r<2>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    mov.u16 %rs1, 1;
    mov.u16 %rs2, 2;
    st.global.u16 [%rd1], %rs1;
    st.global.u16 [%rd1+2], %rs2;
    ld.global.u32 %r1, [%rd1];
    st.global.u16 [%rd1+4], %r1;
    ret;
}
",
    );
    let err = analyze_kernel(&parse(&src), None, u8_scratch_config()).unwrap_err();
    assert!(
        matches!(
            &err,
            AnalysisError::Eval(EvalError::Unsupported { what, .. })
                if what.contains("pair stored")
        ),
        "expected pair sub-width store rejection, got: {}",
        err
    );
}

/// Branching on input data violates structured-CTA.
#[test]
fn test_symbolic_branch_rejected() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<2>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.global.f32 %f1, [%rd1];
    setp.gt.f32 %p1, %f1, 0f00000000;
    @%p1 bra $L1;
$L1:
    ret;
}
",
    );
    let module = parse(&src);
    let err = analyze_kernel(&module, None, in_out_config(1, 1)).unwrap_err();
    assert!(
        matches!(err, AnalysisError::Eval(EvalError::NotConcrete { .. })),
        "expected structured-CTA violation, got: {}",
        err
    );
}

/// The `__symexpf` callseq idiom becomes symbolic exp.
#[test]
fn test_symexpf_callseq() {
    let src = wrap(
        ".extern .func  (.param .b32 func_retval0) __symexpf
(
    .param .b32 __symexpf_param_0
)
;

.visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<3>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    ld.global.f32 %f1, [%rd1];
    { // callseq 0, 0
    .reg .b32 temp_param_reg;
    .param .b32 param0;
    st.param.f32 [param0+0], %f1;
    .param .b32 retval0;
    call.uni (retval0),
    __symexpf,
    (
    param0
    );
    ld.param.f32 %f2, [retval0+0];
    } // callseq 0
    st.global.f32 [%rd2], %f2;
    ret;
}
",
    );
    let module = parse(&src);
    let output = analyze_kernel(&module, None, in_out_config(1, 1)).unwrap();
    assert_eq!(display_output(&output, "out", 0), "exp(in[0])");
}

/// shfl.sync.idx exchanges lane values (2 lanes, mask 0x3):
/// out[tid] = in[tid ^ 1].
#[test]
fn test_shfl_sync_idx_exchange() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<3>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<7>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    mov.u32 %r1, %tid.x;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd3, %r2;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    xor.b32 %r3, %r1, 1;
    mov.u32 %r4, 31;
    mov.u32 %r5, 3;
    shfl.sync.idx.b32 %f2, %f1, %r3, %r4, %r5;
    add.s64 %rd5, %rd2, %rd3;
    st.global.f32 [%rd5], %f2;
    ret;
}
",
    );
    let module = parse(&src);
    let output = analyze_kernel(&module, None, in_out_config(2, 2)).unwrap();
    assert_eq!(display_output(&output, "out", 0), "in[1]");
    assert_eq!(display_output(&output, "out", 1), "in[0]");
    // Warp syncs count once per fired group.
    assert_eq!(output.stats.warp_syncs, 1);
}

/// Float `.sat` clamps the result to [0, 1]; concrete operands fold all
/// the way to the clamped constant.
#[test]
fn test_add_sat_f32_concrete_clamps() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0
)
{
    .reg .f32 %f<5>;
    .reg .b64 %rd<2>;

    ld.param.u64 %rd1, [k_param_0];
    cvta.to.global.u64 %rd1, %rd1;
    mov.f32 %f1, 0f3F400000;
    add.sat.f32 %f2, %f1, %f1;
    st.global.f32 [%rd1], %f2;
    mov.f32 %f3, 0fBF400000;
    add.sat.f32 %f4, %f3, 0f3E800000;
    st.global.f32 [%rd1+4], %f4;
    ret;
}
",
    );
    let module = parse(&src);
    let mut config = AnalysisConfig::new((1, 1, 1));
    config.arrays = vec![ArrayDef {
        name: "out".to_string(),
        base: 0x20000,
        elem_width: 4,
        len: 2,
        kind: ArrayKind::Output,
    }];
    config.params = vec![ParamValue::ArrayPtr("out".to_string())];
    let output = analyze_kernel(&module, None, config).unwrap();
    // 0.75 + 0.75 = 1.5 saturates to 1.0.
    assert_eq!(display_output(&output, "out", 0), "1");
    // -0.75 + 0.25 = -0.5 saturates to 0.0.
    assert_eq!(display_output(&output, "out", 1), "0");
}

/// f16 input/output arrays alongside an f32 input (elem_width 2 out).
fn f32_in_f16_out_config(threads: u32, len: u64) -> AnalysisConfig {
    let mut config = AnalysisConfig::new((threads, 1, 1));
    config.arrays = vec![
        ArrayDef {
            name: "in".to_string(),
            base: 0x10000,
            elem_width: 4,
            len,
            kind: ArrayKind::Input,
        },
        ArrayDef {
            name: "out".to_string(),
            base: 0x20000,
            elem_width: 2,
            len,
            kind: ArrayKind::Output,
        },
    ];
    config.params = vec![
        ParamValue::ArrayPtr("in".to_string()),
        ParamValue::ArrayPtr("out".to_string()),
    ];
    config
}

/// nvcc's fused-ReLU epilogue `cvt.rn.relu.f16.f32` is the exact value
/// transformation max(x, 0) on a symbolic input.
#[test]
fn test_cvt_relu_symbolic_is_max() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .b16 %rs<2>;
    .reg .f32 %f<2>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    ld.global.f32 %f1, [%rd1];
    cvt.rn.relu.f16.f32 %rs1, %f1;
    st.global.u16 [%rd2], %rs1;
    ret;
}
",
    );
    let module = parse(&src);
    let output = analyze_kernel(&module, None, f32_in_f16_out_config(1, 1)).unwrap();
    assert_eq!(display_output(&output, "out", 0), "max(in[0], 0)");
}

/// ReLU via the fused cvt epilogue is equivalent to ReLU via an explicit
/// max.f32 against zero (with flipped operand order): the clamp and the
/// max are the same real function, and canon flattens max atoms.
#[test]
fn test_relu_epilogue_equivalent_to_explicit_max() {
    const PROLOGUE: &str = ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .b16 %rs<2>;
    .reg .f32 %f<4>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<7>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    mov.u32 %r1, %tid.x;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd3, %r2;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    shl.b32 %r3, %r1, 1;
    cvt.u64.u32 %rd5, %r3;
    add.s64 %rd6, %rd2, %rd5;
";
    let fused = wrap(&format!(
        "{}    cvt.rn.relu.f16.f32 %rs1, %f1;
    st.global.u16 [%rd6], %rs1;
    ret;
}}
",
        PROLOGUE
    ));
    let explicit = wrap(&format!(
        "{}    mov.f32 %f2, 0f00000000;
    max.f32 %f3, %f2, %f1;
    cvt.rn.f16.f32 %rs1, %f3;
    st.global.u16 [%rd6], %rs1;
    ret;
}}
",
        PROLOGUE
    ));
    let a = analyze_kernel(&parse(&fused), None, f32_in_f16_out_config(2, 2)).unwrap();
    let b = analyze_kernel(&parse(&explicit), None, f32_in_f16_out_config(2, 2)).unwrap();
    assert!(matches!(check_equiv(&a, &b), EquivOutcome::Equivalent));
}

// =========================================================================
// Paper kernels: Harris reductions
// =========================================================================

/// Config for the reduction kernels: int arrays, `n` input elements.
fn reduction_config(threads: u32, n: u64) -> AnalysisConfig {
    let mut config = AnalysisConfig::new((threads, 1, 1));
    config.arrays = vec![
        ArrayDef {
            name: "in".to_string(),
            base: 0x10000,
            elem_width: 4,
            len: n,
            kind: ArrayKind::Input,
        },
        ArrayDef {
            name: "out".to_string(),
            base: 0x20000,
            elem_width: 4,
            len: 1,
            kind: ArrayKind::Output,
        },
    ];
    config.params = vec![
        ParamValue::ArrayPtr("in".to_string()),
        ParamValue::ArrayPtr("out".to_string()),
    ];
    config
}

fn run_reduction(
    file: &str,
    kernel: &str,
    threads: u32,
    n: u64,
    dynamic_shared: u64,
) -> AnalysisOutput {
    let module = parse_file(file);
    let mut config = reduction_config(threads, n);
    config.dynamic_shared_bytes = dynamic_shared;
    analyze_kernel(&module, Some(kernel), config)
        .unwrap_or_else(|e| panic!("{} failed: {}", file, e))
}

fn red1() -> AnalysisOutput {
    run_reduction(
        "01_reduction/Red-1.ptx",
        "_Z17reduce1024_1blockPKiPi",
        128,
        128,
        0,
    )
}

#[test]
fn test_red1_race_free() {
    let output = red1();
    assert_eq!(output.outputs.len(), 1);
    assert!(output.stats.block_syncs > 0);
}

#[test]
fn test_red1_self_equivalence() {
    let a = red1();
    let b = red1();
    assert!(matches!(check_equiv(&a, &b), EquivOutcome::Equivalent));
}

#[test]
fn test_red1_red2_equivalent() {
    let a = red1();
    let b = run_reduction("01_reduction/Red-2.ptx", "_Z7reduce2PiS_", 128, 128, 0);
    assert!(matches!(check_equiv(&a, &b), EquivOutcome::Equivalent));
}

#[test]
fn test_red1_red3_equivalent() {
    let a = red1();
    // Red-3 uses extern (dynamic) shared memory: 128 ints.
    let b = run_reduction("01_reduction/Red-3.ptx", "_Z7reduce3PiS_", 128, 128, 512);
    assert!(matches!(check_equiv(&a, &b), EquivOutcome::Equivalent));
}

#[test]
fn test_red1_red4_equivalent() {
    let a = red1();
    // Red-4 adds `in[i] + in[i + 64]` on load and reduces the low half of
    // sdata. Threads 64..128 read `in[128..192)`, but those values only land
    // in the unused half of sdata, so the sum still covers exactly
    // `in[0..128)`. The input array must span 192 elements to keep the loads
    // in bounds.
    let b = run_reduction("01_reduction/Red-4.ptx", "_Z7reduce0PiS_", 128, 192, 0);
    assert!(matches!(check_equiv(&a, &b), EquivOutcome::Equivalent));
}

/// A deliberately wrong reduction (dropping one element) must be caught.
#[test]
fn test_red1_wrong_input_not_equivalent() {
    let a = red1();
    // Same kernel, but with only 127 of the 128 inputs shared: rename the
    // input array so its symbols differ.
    let module = parse_file("01_reduction/Red-1.ptx");
    let mut config = reduction_config(128, 128);
    config.arrays[0].name = "other".to_string();
    config.params[0] = ParamValue::ArrayPtr("other".to_string());
    let b = analyze_kernel(&module, Some("_Z17reduce1024_1blockPKiPi"), config).unwrap();
    assert!(matches!(
        check_equiv(&a, &b),
        EquivOutcome::NotEquivalent { .. }
    ));
}

// =========================================================================
// Exact real-constant folding: the fold algebra and canon's algebra are
// one algebra, so the same real-model value folds to the same constant
// regardless of which instructions computed it.
// =========================================================================

/// One-input/one-output f32 config over `in`/`out` (single thread).
fn one_elem_config() -> AnalysisConfig {
    in_out_config(1, 1)
}

/// `div.rn.f32 d, 1.0, 3.0` and `rcp.approx.f32 d, 3.0` now fold to the
/// same exact rational 1/3, so kernels using either spelling are
/// equivalent. Under the old f64 folds, the division folded to
/// rational-of-fl(1/3) while rcp stayed symbolic and canonicalized to
/// exactly 1/3 - the same value under two different constants.
#[test]
fn test_div_by_three_equivalent_to_rcp_of_three() {
    let div_src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<5>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    ld.global.f32 %f1, [%rd1];
    div.rn.f32 %f2, 0f3F800000, 0f40400000;
    mul.f32 %f3, %f1, %f2;
    st.global.f32 [%rd2], %f3;
    ret;
}
",
    );
    let rcp_src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<5>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    ld.global.f32 %f1, [%rd1];
    mov.f32 %f2, 0f40400000;
    rcp.approx.f32 %f3, %f2;
    mul.f32 %f4, %f1, %f3;
    st.global.f32 [%rd2], %f4;
    ret;
}
",
    );
    let a = analyze_kernel(&parse(&div_src), None, one_elem_config()).unwrap();
    let b = analyze_kernel(&parse(&rcp_src), None, one_elem_config()).unwrap();
    // Both fold to the exact rational 1/3.
    assert_eq!(display_output(&a, "out", 0), "(in[0] * 1/3)");
    assert_eq!(display_output(&b, "out", 0), "(in[0] * 1/3)");
    assert!(matches!(check_equiv(&a, &b), EquivOutcome::Equivalent));
}

/// `sum / N` vs the strength-reduced `t = 1.0 / N; sum * t` at concrete
/// N = 768: 1/768 is not dyadic, so the old fold rounded `t` to fl(1/768)
/// and reported a false DIFF; the exact fold makes them equivalent.
#[test]
fn test_div_by_n_equivalent_to_mul_by_computed_reciprocal() {
    let div_src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<4>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    ld.global.f32 %f1, [%rd1];
    div.rn.f32 %f2, %f1, 0f44400000;
    st.global.f32 [%rd2], %f2;
    ret;
}
",
    );
    let recip_src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<5>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    ld.global.f32 %f1, [%rd1];
    div.rn.f32 %f2, 0f3F800000, 0f44400000;
    mul.f32 %f3, %f1, %f2;
    st.global.f32 [%rd2], %f3;
    ret;
}
",
    );
    let a = analyze_kernel(&parse(&div_src), None, one_elem_config()).unwrap();
    let b = analyze_kernel(&parse(&recip_src), None, one_elem_config()).unwrap();
    assert!(matches!(check_equiv(&a, &b), EquivOutcome::Equivalent));
}

/// f64 in/out config for the double-precision constant tests.
fn f64_out_config() -> AnalysisConfig {
    let mut config = AnalysisConfig::new((1, 1, 1));
    config.arrays = vec![ArrayDef {
        name: "out".to_string(),
        base: 0x20000,
        elem_width: 8,
        len: 1,
        kind: ArrayKind::Output,
    }];
    config.params = vec![ParamValue::ArrayPtr("out".to_string())];
    config
}

/// Concrete fma folds exactly as mul-then-add: one algebra, no fused
/// rounding (the old f64 folds could disagree between the two spellings).
#[test]
fn test_concrete_fma_equivalent_to_mul_add() {
    let fma_src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0
)
{
    .reg .f64 %fd<5>;
    .reg .b64 %rd<2>;

    ld.param.u64 %rd1, [k_param_0];
    cvta.to.global.u64 %rd1, %rd1;
    mov.f64 %fd1, 0d3FB999999999999A;
    mov.f64 %fd2, 0d3FC999999999999A;
    mov.f64 %fd3, 0d3FD3333333333333;
    fma.rn.f64 %fd4, %fd1, %fd2, %fd3;
    st.global.f64 [%rd1], %fd4;
    ret;
}
",
    );
    let mul_add_src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0
)
{
    .reg .f64 %fd<6>;
    .reg .b64 %rd<2>;

    ld.param.u64 %rd1, [k_param_0];
    cvta.to.global.u64 %rd1, %rd1;
    mov.f64 %fd1, 0d3FB999999999999A;
    mov.f64 %fd2, 0d3FC999999999999A;
    mov.f64 %fd3, 0d3FD3333333333333;
    mul.f64 %fd4, %fd1, %fd2;
    add.f64 %fd5, %fd4, %fd3;
    st.global.f64 [%rd1], %fd5;
    ret;
}
",
    );
    let a = analyze_kernel(&parse(&fma_src), None, f64_out_config()).unwrap();
    let b = analyze_kernel(&parse(&mul_add_src), None, f64_out_config()).unwrap();
    assert!(matches!(check_equiv(&a, &b), EquivOutcome::Equivalent));
}

/// The model-faithful direction: a kernel that adds 0.1 + 0.2 at runtime
/// computes the exact rational sum, which is NOT the pre-rounded literal
/// 0d3FD3333333333334 (the f64 the hardware would produce). The old f64
/// fold rounded the runtime sum and wrongly equated the two.
#[test]
fn test_runtime_sum_differs_from_prerounded_literal() {
    let sum_src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0
)
{
    .reg .f64 %fd<4>;
    .reg .b64 %rd<2>;

    ld.param.u64 %rd1, [k_param_0];
    cvta.to.global.u64 %rd1, %rd1;
    mov.f64 %fd1, 0d3FB999999999999A;
    mov.f64 %fd2, 0d3FC999999999999A;
    add.f64 %fd3, %fd1, %fd2;
    st.global.f64 [%rd1], %fd3;
    ret;
}
",
    );
    let literal_src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0
)
{
    .reg .f64 %fd<2>;
    .reg .b64 %rd<2>;

    ld.param.u64 %rd1, [k_param_0];
    cvta.to.global.u64 %rd1, %rd1;
    mov.f64 %fd1, 0d3FD3333333333334;
    st.global.f64 [%rd1], %fd1;
    ret;
}
",
    );
    let a = analyze_kernel(&parse(&sum_src), None, f64_out_config()).unwrap();
    let b = analyze_kernel(&parse(&literal_src), None, f64_out_config()).unwrap();
    assert!(matches!(
        check_equiv(&a, &b),
        EquivOutcome::NotEquivalent { .. }
    ));
}

/// A NaN literal (0f7FC00000) is rejected at lowering: NaN denotes no
/// real number, so it cannot enter the analysis model.
#[test]
fn test_nan_literal_rejected_at_lowering() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0
)
{
    .reg .f32 %f<2>;
    .reg .b64 %rd<2>;

    ld.param.u64 %rd1, [k_param_0];
    cvta.to.global.u64 %rd1, %rd1;
    mov.f32 %f1, 0f7FC00000;
    st.global.f32 [%rd1], %f1;
    ret;
}
",
    );
    let mut config = AnalysisConfig::new((1, 1, 1));
    config.arrays = vec![ArrayDef {
        name: "out".to_string(),
        base: 0x20000,
        elem_width: 4,
        len: 1,
        kind: ArrayKind::Output,
    }];
    config.params = vec![ParamValue::ArrayPtr("out".to_string())];
    let err = analyze_kernel(&parse(&src), None, config).unwrap_err();
    assert!(
        matches!(err, AnalysisError::Lower(_)),
        "expected lowering rejection of the NaN literal, got: {}",
        err
    );
    assert!(
        err.to_string().contains("NaN literal"),
        "unexpected message: {}",
        err
    );
}

/// A NaN float parameter is a config validation error.
#[test]
fn test_nan_float_param_rejected() {
    let src = wrap(
        ".visible .entry k(
    .param .f32 k_param_0
)
{
    ret;
}
",
    );
    let mut config = AnalysisConfig::new((1, 1, 1));
    config.params = vec![ParamValue::Float(f64::NAN)];
    let err = analyze_kernel(&parse(&src), None, config).unwrap_err();
    assert!(
        matches!(&err, AnalysisError::Eval(EvalError::Config { message }) if message.contains("NaN")),
        "expected NaN config validation error, got: {}",
        err
    );
}

/// `0.0 / 0.0` no longer mints a NaN: the division stays a symbolic node
/// and the decision procedure errors loudly on the formally-zero
/// denominator when the element is checked.
#[test]
fn test_zero_over_zero_is_a_loud_equivalence_error() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0
)
{
    .reg .f32 %f<3>;
    .reg .b64 %rd<2>;

    ld.param.u64 %rd1, [k_param_0];
    cvta.to.global.u64 %rd1, %rd1;
    div.rn.f32 %f2, 0f00000000, 0f00000000;
    st.global.f32 [%rd1], %f2;
    ret;
}
",
    );
    let mut config = AnalysisConfig::new((1, 1, 1));
    config.arrays = vec![ArrayDef {
        name: "out".to_string(),
        base: 0x20000,
        elem_width: 4,
        len: 1,
        kind: ArrayKind::Output,
    }];
    config.params = vec![ParamValue::ArrayPtr("out".to_string())];
    let a = analyze_kernel(&parse(&src), None, config.clone()).unwrap();
    // The output expression is the unfolded division, not a constant.
    assert_eq!(display_output(&a, "out", 0), "(0 / 0)");
    let b = analyze_kernel(&parse(&src), None, config).unwrap();
    let arrays: Vec<String> = a.outputs.iter().map(|(n, _)| n.clone()).collect();
    let err = check_output_equivalence(&a, &b, &arrays).unwrap_err();
    assert!(
        err.to_string().contains("division"),
        "expected a division-by-zero equivalence error, got: {}",
        err
    );
}

// =========================================================================
// The paper's Sync semantics at warp level: exited lanes arrive at every
// warp op and are included in the chi-clear (Sync'/syncMem over the full
// mask set I, return-threads included).
// =========================================================================

/// Tail-exit warp reduction: threads 16..31 exit immediately, and the 16
/// survivors run a full-mask `shfl.sync.down` tree reduction. The groups
/// fire because exited lanes count as arrived (paper's Sync rule; the ISA
/// says the same for shfl.sync: "wait until all non-exited threads
/// corresponding to membermask"); previously data ops required every mask
/// lane live and this idiom was a false Deadlock. The guarded adds discard
/// the Undefined values sourced from exited lanes 16..23, exactly as real
/// kernels discard the out-of-range contributions, so the result equals a
/// sequential sum of in[0..16].
#[test]
fn test_tail_exit_warp_shfl_reduction() {
    let mut body = String::from(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .pred %p<7>;
    .reg .f32 %f<7>;
    .reg .b32 %r<13>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    mov.u32 %r1, %tid.x;
    setp.gt.u32 %p1, %r1, 15;
@%p1 ret;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd3, %r2;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    mov.u32 %r3, 31;
    mov.u32 %r4, -1;
",
    );
    for (round, offset) in [8u32, 4, 2, 1].into_iter().enumerate() {
        let (w, p, r) = (round + 2, round + 2, round + 5);
        body.push_str(&format!(
            "    mov.u32 %r{r}, {offset};
    shfl.sync.down.b32 %f{w}, %f1, %r{r}, %r3, %r4;
    add.u32 %r{r2}, %r1, {offset};
    setp.lt.u32 %p{p}, %r{r2}, 16;
@%p{p} add.f32 %f1, %f1, %f{w};
",
            r2 = round + 9,
        ));
    }
    body.push_str(
        "    setp.ne.u32 %p6, %r1, 0;
@%p6 ret;
    st.global.f32 [%rd2], %f1;
    ret;
}
",
    );
    let module = parse(&wrap(&body));
    let a = analyze_kernel(&module, None, in_out_config(32, 16)).unwrap();
    assert_eq!(a.stats.warp_syncs, 4);

    // Reference: one thread sums in[0..16] sequentially.
    let mut ref_body = String::from(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<3>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    ld.global.f32 %f1, [%rd1];
",
    );
    for i in 1..16 {
        ref_body.push_str(&format!(
            "    ld.global.f32 %f2, [%rd1+{}];
    add.f32 %f1, %f1, %f2;
",
            i * 4
        ));
    }
    ref_body.push_str(
        "    st.global.f32 [%rd2], %f1;
    ret;
}
",
    );
    let b = analyze_kernel(&parse(&wrap(&ref_body)), None, in_out_config(1, 16)).unwrap();
    assert!(matches!(check_equiv(&a, &b), EquivOutcome::Equivalent));
}

/// A shfl value sourced from an exited (in-mask) lane is Undefined, per the
/// ISA: "results are undefined if a thread sources a register from an
/// inactive thread". Storing it directly to an output array surfaces as
/// UndefinedOutput; the group itself fires (no Deadlock).
#[test]
fn test_shfl_from_exited_lane_is_undefined() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<3>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    mov.u32 %r1, %tid.x;
    setp.gt.u32 %p1, %r1, 15;
@%p1 ret;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd3, %r2;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    mov.u32 %r3, 31;
    mov.u32 %r4, -1;
    mov.u32 %r5, 8;
    shfl.sync.down.b32 %f2, %f1, %r5, %r3, %r4;
    add.s64 %rd5, %rd2, %rd3;
    st.global.f32 [%rd5], %f2;
    ret;
}
",
    );
    let module = parse(&src);
    // Lanes 8..15 source exited lanes 16..23 and store the Undefined result.
    let err = analyze_kernel(&module, None, in_out_config(32, 16)).unwrap_err();
    assert!(
        matches!(err, AnalysisError::Eval(EvalError::UndefinedOutput { .. })),
        "expected undefined output from exited source lane, got: {}",
        err
    );
}

/// The verified false race: a lane reads shared memory and exits; the
/// survivor bar.warp.syncs on the full mask and writes the same byte. The
/// paper's syncMem clears pending sets over the whole I including the
/// returned lane, so this is NOT a race (it previously was reported as
/// one, because only live members joined the chi sync set).
#[test]
fn test_bar_warp_sync_clears_chi_for_exited_reader() {
    let src = wrap(
        ".visible .entry k()
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .shared .align 4 .b8 sdata[8];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, sdata;
    setp.eq.u32 %p1, %r1, 1;
@%p1 ld.shared.u32 %r3, [%r2];
@%p1 ret;
    bar.warp.sync 3;
    st.shared.u32 [%r2], %r1;
    ret;
}
",
    );
    let module = parse(&src);
    analyze_kernel(&module, None, AnalysisConfig::new((2, 1, 1)))
        .expect("read-then-exit / sync / write must not race under the paper's syncMem");
}

/// Mirrored variant: a lane writes shared memory and exits; the survivor
/// syncs and reads the byte back. No race, and the exited lane's value is
/// visible (syncMem clears ordering state, not memory).
#[test]
fn test_bar_warp_sync_clears_chi_for_exited_writer() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<2>;
    .shared .align 4 .b8 sdata[8];

    ld.param.u64 %rd1, [k_param_1];
    cvta.to.global.u64 %rd1, %rd1;
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, sdata;
    setp.eq.u32 %p1, %r1, 1;
    mov.u32 %r3, 42;
@%p1 st.shared.u32 [%r2], %r3;
@%p1 ret;
    bar.warp.sync 3;
    ld.shared.u32 %r4, [%r2];
    st.global.u32 [%rd1], %r4;
    ret;
}
",
    );
    let module = parse(&src);
    let output = analyze_kernel(&module, None, in_out_config(2, 1))
        .expect("write-then-exit / sync / read must not race under the paper's syncMem");
    assert_eq!(display_output(&output, "out", 0), "42");
}

/// bar.warp.sync end-to-end (previously UnsupportedInstruction at
/// lowering): the neighbor exchange synchronized by a warp barrier instead
/// of a CTA barrier.
#[test]
fn test_bar_warp_sync_neighbor_exchange() {
    let src = wrap(&SWAP_BODY.replace("BARRIER", "bar.warp.sync 3;"));
    let module = parse(&src);
    let output = analyze_kernel(&module, None, in_out_config(2, 2)).unwrap();
    assert_eq!(display_output(&output, "out", 0), "in[1]");
    assert_eq!(display_output(&output, "out", 1), "in[0]");
    assert_eq!(output.stats.block_syncs, 0);
    assert_eq!(output.stats.warp_syncs, 1);
}

/// `barrier.sync{.aligned} a` (no thread count) is the same full-CTA
/// barrier as `bar.sync a` (ISA: "bar{.cta}.sync is equivalent to
/// barrier{.cta}.sync.aligned").
#[test]
fn test_barrier_sync_is_bar_sync() {
    for form in ["barrier.sync 0;", "barrier.cta.sync.aligned 0;"] {
        let src = wrap(&SWAP_BODY.replace("BARRIER", form));
        let module = parse(&src);
        let output = analyze_kernel(&module, None, in_out_config(2, 2))
            .unwrap_or_else(|e| panic!("{} failed: {}", form, e));
        assert_eq!(display_output(&output, "out", 0), "in[1]");
        assert_eq!(display_output(&output, "out", 1), "in[0]");
        assert_eq!(output.stats.block_syncs, 2);
    }
}

// =========================================================================
// activemask: exact for the converged case (existing, non-exited lanes)
// =========================================================================

/// activemask kernel: every thread records its mask, with a CTA barrier
/// after the query so no thread has exited when any thread queries (each
/// thread blocks rather than exits, making the all-alive answer
/// deterministic under round-robin).
fn activemask_body(barrier: &str) -> String {
    format!(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;

    ld.param.u64 %rd1, [k_param_1];
    cvta.to.global.u64 %rd1, %rd1;
    mov.u32 %r2, %tid.x;
    {barrier}
    activemask.b32 %r1;
    shl.b32 %r3, %r2, 2;
    cvt.u64.u32 %rd2, %r3;
    add.s64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r1;
    ret;
}}
"
    )
}

/// All 32 lanes alive: every thread reads 0xFFFFFFFF. The barrier comes
/// *after* activemask here, so each thread queries while all others are
/// alive (blocked or not-yet-run, never exited).
#[test]
fn test_activemask_all_alive_full_warp() {
    let body = activemask_body("").replace(
        "activemask.b32 %r1;",
        "activemask.b32 %r1;\n    bar.sync 0;",
    );
    let module = parse(&wrap(&body));
    let output = analyze_kernel(&module, None, in_out_config(32, 32)).unwrap();
    for i in 0..32 {
        assert_eq!(display_output(&output, "out", i), "4294967295");
    }
}

/// A 16-thread block: warp lanes 16..31 do not exist and contribute 0, so
/// every thread reads 0xFFFF (again with the barrier after the query, so
/// all queries happen before any exit).
#[test]
fn test_activemask_16_thread_block() {
    let body = activemask_body("").replace(
        "activemask.b32 %r1;",
        "activemask.b32 %r1;\n    bar.sync 0;",
    );
    let module = parse(&wrap(&body));
    let output = analyze_kernel(&module, None, in_out_config(16, 16)).unwrap();
    for i in 0..16 {
        assert_eq!(display_output(&output, "out", i), "65535");
    }
}

/// Tail-exit: threads 16..31 exit before the barrier; survivors query
/// after it. Only thread 0's value is asserted: under round-robin each
/// survivor also observes the *earlier survivors'* exits (thread 1 sees
/// thread 0 gone, and so on), just as hardware activemask depends on
/// execution timing - the model deliberately promises nothing about
/// exit ordering beyond the executing thread's own observation point.
#[test]
fn test_activemask_after_tail_exit() {
    let mut body = activemask_body("bar.sync 0;");
    body = body.replace(
        "    mov.u32 %r2, %tid.x;\n",
        "    mov.u32 %r2, %tid.x;\n    setp.gt.u32 %p1, %r2, 15;\n@%p1 ret;\n",
    );
    let module = parse(&wrap(&body));
    let output = analyze_kernel(&module, None, in_out_config(32, 32)).unwrap();
    assert_eq!(display_output(&output, "out", 0), "65535");
}

// =========================================================================
// Bounds hardening: ownership containment, whole-vector footprints, and
// overflow-proof address arithmetic (all release-active)
// =========================================================================

/// The paper's own §6.2 pattern: `__shared__ int A[48]` read at
/// `A[tid]` by 64 threads - threads 48..63 read out of bounds and must be
/// caught (hardware happens to tolerate this; the model must not).
#[test]
fn test_shared_array_overflow_read_is_out_of_bounds() {
    let src = wrap(
        ".visible .entry k()
{
    .reg .b32 %r<6>;
    .shared .align 4 .b8 A[192];

    mov.u32 %r1, %tid.x;
    shl.b32 %r2, %r1, 2;
    mov.u32 %r3, A;
    add.s32 %r4, %r3, %r2;
    ld.shared.u32 %r5, [%r4];
    ret;
}
",
    );
    let module = parse(&src);
    let err = analyze_kernel(&module, None, AnalysisConfig::new((64, 1, 1))).unwrap_err();
    assert!(
        matches!(
            err,
            AnalysisError::Eval(EvalError::OutOfBounds { addr: 192, .. })
        ),
        "expected out-of-bounds at A+192, got: {}",
        err
    );
}

/// Cross-array overflow into an adjacent shared array: a v4 (16-byte)
/// load whose first byte lies in A but whose tail runs into the adjacent
/// B. The whole-footprint ownership check makes this a loud OutOfBounds;
/// the per-element checks alone each passed inside a different array and
/// silently read the neighbor.
#[test]
fn test_vector_straddling_adjacent_shared_arrays() {
    let src = wrap(
        ".visible .entry k()
{
    .reg .b32 %r<7>;
    .shared .align 4 .b8 A[40];
    .shared .align 4 .b8 B[64];

    mov.u32 %r1, A;
    ld.shared.v4.u32 {%r2, %r3, %r4, %r5}, [%r1+32];
    ret;
}
",
    );
    let module = parse(&src);
    let err = analyze_kernel(&module, None, AnalysisConfig::new((1, 1, 1))).unwrap_err();
    assert!(
        matches!(
            err,
            AnalysisError::Eval(EvalError::OutOfBounds {
                addr: 32,
                width: 16,
                ..
            })
        ),
        "expected the v4 straddle to be out of bounds, got: {}",
        err
    );
}

/// `x[tid - 1]` at tid = 0 with the array based at address 0: the index
/// wraps to 0xFFFF_FFFF_FFFF_FFFC. The additive bounds check wrapped with
/// it in release builds and silently accepted the access (yielding a
/// silent Undefined); the subtraction-form check rejects it in every
/// build profile. The test suite runs in release, where this must hold.
#[test]
fn test_negative_index_wrap_is_out_of_bounds() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<2>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    mov.u32 %r1, %tid.x;
    add.s32 %r2, %r1, -1;
    cvt.s64.s32 %rd3, %r2;
    shl.b64 %rd3, %rd3, 2;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    st.global.f32 [%rd2], %f1;
    ret;
}
",
    );
    let module = parse(&src);
    let mut config = in_out_config(1, 4);
    config.arrays[0].base = 0; // x based at 0: base + (-4) wraps
    let err = analyze_kernel(&module, None, config).unwrap_err();
    assert!(
        matches!(
            err,
            AnalysisError::Eval(EvalError::OutOfBounds { addr, width: 4, .. })
                if addr == u64::MAX - 3
        ),
        "expected the wrapped address to be out of bounds, got: {}",
        err
    );
}

/// Aligned in-bounds accesses at region edges still pass: the last scalar
/// element of a shared array (exact fit against the region end, with an
/// adjacent array following) and an exact-fit v4 footprint ending at the
/// region end.
#[test]
fn test_region_edge_exact_fit_accesses_pass() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .b32 %r<13>;
    .reg .b64 %rd<2>;
    .shared .align 4 .b8 A[48];
    .shared .align 4 .b8 B[64];

    ld.param.u64 %rd1, [k_param_1];
    cvta.to.global.u64 %rd1, %rd1;
    mov.u32 %r1, A;
    mov.u32 %r2, 1;
    mov.u32 %r3, 2;
    mov.u32 %r4, 3;
    mov.u32 %r5, 4;
    st.shared.v4.u32 [%r1+32], {%r2, %r3, %r4, %r5};
    mov.u32 %r6, 7;
    st.shared.u32 [%r1+44], %r6;
    ld.shared.u32 %r7, [%r1+44];
    ld.shared.v4.u32 {%r8, %r9, %r10, %r11}, [%r1+32];
    mov.u32 %r12, B;
    st.shared.u32 [%r12], %r7;
    ld.shared.u32 %r7, [%r12];
    st.global.u32 [%rd1], %r7;
    st.global.u32 [%rd1+4], %r8;
    ret;
}
",
    );
    let module = parse(&src);
    let output = analyze_kernel(&module, None, in_out_config(1, 2))
        .expect("exact-fit edge accesses must stay in bounds");
    // The scalar store at A+44 overwrote the v4's last element; the v4
    // read back yields {1, 2, 3, 7}.
    assert_eq!(display_output(&output, "out", 0), "7");
    assert_eq!(display_output(&output, "out", 1), "1");
}

/// A valid array in the upper half of the address space: based at
/// 0x7FFF_FFFF_FFFF_FFF0, the +16 access lands at 2^63 - crossing the
/// i64 sign boundary with no u64 wrap. Effective addresses are u64
/// arithmetic mod 2^64 (as on hardware), so this is simply in bounds;
/// the previous i64 checked sum falsely rejected it as an overflow.
#[test]
fn test_upper_half_address_space_access_in_bounds() {
    let src = wrap(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{
    .reg .f32 %f<2>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    ld.global.f32 %f1, [%rd1+16];
    st.global.f32 [%rd2], %f1;
    ret;
}
",
    );
    let module = parse(&src);
    let mut config = in_out_config(1, 8);
    config.arrays[0].base = 0x7FFF_FFFF_FFFF_FFF0;
    let output = analyze_kernel(&module, None, config)
        .expect("crossing 2^63 without wrapping u64 is in bounds");
    assert_eq!(display_output(&output, "out", 0), "in[4]");
}

// =========================================================================
// Cross-family region overlap: config arrays vs the module-global window
// =========================================================================

/// A config array overlapping the reserved module-global window is a loud
/// setup error, not a silent shadowing: both families land in the one
/// global region list, whose disjointness `check_bounds` relies on.
#[test]
fn test_array_overlapping_module_global_window_rejected() {
    const MODULE_GLOBAL_BASE: u64 = 0x7000_0000_0000_0000;
    let src = wrap(
        ".global .align 4 .b32 g;
.visible .entry k()
{
    ret;
}
",
    );

    // The probe scenario: an array based exactly at the module-global
    // base, covering the global `g`.
    let module = parse(&src);
    let mut config = AnalysisConfig::new((1, 1, 1));
    config.arrays.push(ArrayDef {
        name: "in".to_string(),
        base: MODULE_GLOBAL_BASE,
        elem_width: 4,
        len: 4,
        kind: ArrayKind::Input,
    });
    let err = analyze_kernel(&module, None, config.clone()).unwrap_err();
    assert!(
        matches!(
            &err,
            AnalysisError::Eval(EvalError::Config { message })
                if message.contains("module-global")
        ),
        "expected a module-global overlap config error, got: {}",
        err
    );

    // Range overlap, not just base membership: an array based below the
    // window whose tail reaches into it is rejected too.
    config.arrays[0].base = MODULE_GLOBAL_BASE - 8;
    let err = analyze_kernel(&module, None, config.clone()).unwrap_err();
    assert!(
        matches!(
            &err,
            AnalysisError::Eval(EvalError::Config { message })
                if message.contains("module-global")
        ),
        "expected a module-global overlap config error, got: {}",
        err
    );

    // Half-open boundary: an array ending exactly at the window base does
    // not overlap and is accepted.
    config.arrays[0].base = MODULE_GLOBAL_BASE - 16;
    analyze_kernel(&module, None, config).expect("an array ending at the window base is disjoint");
}

// =========================================================================
// Hostile declaration sizes: checked packing arithmetic
// =========================================================================

/// A shared declaration whose byte size overflows u64 (here via a
/// two-dimensional element count: 2^31 * 2^31 u32 elements = 2^64 bytes)
/// is rejected loudly at lowering; release-wrapping would silently give
/// the variable a tiny size and mismodel every access to it.
#[test]
fn test_hostile_shared_declaration_size_rejected() {
    let src = wrap(
        ".visible .entry k()
{
    .shared .align 4 .u32 A[2147483648][2147483648];

    ret;
}
",
    );
    let module = parse(&src);
    let err = analyze_kernel(&module, None, AnalysisConfig::new((1, 1, 1))).unwrap_err();
    assert!(
        matches!(
            &err,
            AnalysisError::Lower(LowerError::VariableSizeOverflow { what }) if what.contains("A")
        ),
        "expected a variable-size overflow error, got: {}",
        err
    );
}

// =========================================================================
// cvta restriction: only cvta.to.global (the corpus form) is accepted
// =========================================================================

fn cvta_kernel(form: &str) -> String {
    wrap(&format!(
        ".visible .entry k(
    .param .u64 k_param_0,
    .param .u64 k_param_1
)
{{
    .reg .b32 %r<2>;
    .reg .b64 %rd<4>;
    .shared .align 4 .b8 sdata[16];

    ld.param.u64 %rd1, [k_param_1];
    mov.u32 %r1, sdata;
    cvt.u64.u32 %rd2, %r1;
    {form}
    ret;
}}
"
    ))
}

/// The corpus's one cvta form is accepted as the identity.
#[test]
fn test_cvta_to_global_accepted() {
    let module = parse(&cvta_kernel("cvta.to.global.u64 %rd3, %rd1;"));
    analyze_kernel(&module, None, in_out_config(1, 1)).expect("cvta.to.global is the corpus form");
}

/// Every other cvta form is rejected by name, with the reason split by
/// why: `cvta.global` (to-generic over global) is identity-compatible and
/// rejected only because the corpus never uses it, while the shared/local
/// forms would mint or bless addresses in the wrong space (generic
/// addressing has no per-space windows in the model).
#[test]
fn test_cvta_other_forms_rejected() {
    for (form, expect, reason_fragment) in [
        ("cvta.global.u64 %rd3, %rd1;", "cvta.global", "corpus"),
        (
            "cvta.shared.u64 %rd3, %rd2;",
            "cvta.shared",
            "per-space windows",
        ),
        (
            "cvta.to.shared.u64 %rd3, %rd1;",
            "cvta.to.shared",
            "per-space windows",
        ),
    ] {
        let module = parse(&cvta_kernel(form));
        let err = analyze_kernel(&module, None, in_out_config(1, 1)).unwrap_err();
        match &err {
            AnalysisError::Lower(LowerError::UnsupportedInstruction {
                instruction,
                reason,
            }) => {
                assert_eq!(instruction, expect, "wrong instruction name for {}", form);
                assert!(
                    reason.as_deref().unwrap_or("").contains(reason_fragment),
                    "reason for {} should mention '{}', got: {:?}",
                    form,
                    reason_fragment,
                    reason
                );
            }
            other => panic!(
                "expected UnsupportedInstruction for {}, got: {}",
                form, other
            ),
        }
    }
}
