//! PTX Abstract Syntax Tree
//!
//! This module defines the AST types for representing parsed PTX programs.

use crate::ascii::{AsciiChar, AsciiSliceExt, AsciiString};
use crate::instr::InstrKind;
use crate::lex::DottedIdent;

// Re-export Span from volta_common
pub use volta_common::Span;

// =============================================================================
// FromAscii Trait
// =============================================================================

/// Trait for types that can be parsed from an ASCII string slice.
/// Used by ModifierParser to generically parse instruction modifiers.
pub trait FromAscii: Sized {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self>;
}

// =============================================================================
// Module Structure
// =============================================================================

/// A complete PTX module
#[derive(Debug, Clone)]
pub struct Module {
    pub version: Version,
    pub target: Target,
    pub address_size: Option<AddressSize>,
    pub items: Vec<TopLevelItem>,
}

/// PTX version directive: `.version major.minor`
#[derive(Debug, Clone, Copy)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
}

/// Target architecture: `.target sm_XX`
#[derive(Debug, Clone)]
pub struct Target {
    pub archs: Vec<Arch>,
    pub texmode: Option<Texmode>,
    pub debug: bool,
    pub map_f64_to_f32: bool,
}

/// NVIDIA GPU architecture (SM version)
// Don't reorder the variants; it affects `Ord` implementation
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Arch {
    // Tesla (sm_1x)
    Sm10,
    Sm11,
    Sm12,
    Sm13,
    // Fermi (sm_2x)
    Sm20,
    // Kepler (sm_3x)
    Sm30,
    Sm32,
    Sm35,
    Sm37,
    // Maxwell (sm_5x)
    Sm50,
    Sm52,
    Sm53,
    // Pascal (sm_6x)
    Sm60,
    Sm61,
    Sm62,
    // Volta/Turing (sm_7x)
    Sm70,
    Sm72,
    Sm75,
    // Ampere (sm_8x)
    Sm80,
    Sm86,
    Sm87,
    Sm88,
    Sm89,
    // Hopper (sm_9x)
    Sm90,
    Sm90a,
    // Blackwell (sm_10x)
    Sm100,
    Sm100f,
    Sm100a,
    Sm103,
    Sm103f,
    Sm103a,
    // sm_11x
    Sm110,
    Sm110f,
    Sm110a,
    // sm_12x
    Sm120,
    Sm120f,
    Sm120a,
    Sm121,
    Sm121f,
    Sm121a,
}

impl FromAscii for Arch {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            // Tesla
            b"sm_10" | b"compute_10" => Arch::Sm10,
            b"sm_11" | b"compute_11" => Arch::Sm11,
            b"sm_12" | b"compute_12" => Arch::Sm12,
            b"sm_13" | b"compute_13" => Arch::Sm13,
            // Fermi
            b"sm_20" | b"compute_20" => Arch::Sm20,
            // Kepler
            b"sm_30" | b"compute_30" => Arch::Sm30,
            b"sm_32" | b"compute_32" => Arch::Sm32,
            b"sm_35" | b"compute_35" => Arch::Sm35,
            b"sm_37" | b"compute_37" => Arch::Sm37,
            // Maxwell
            b"sm_50" | b"compute_50" => Arch::Sm50,
            b"sm_52" | b"compute_52" => Arch::Sm52,
            b"sm_53" | b"compute_53" => Arch::Sm53,
            // Pascal
            b"sm_60" | b"compute_60" => Arch::Sm60,
            b"sm_61" | b"compute_61" => Arch::Sm61,
            b"sm_62" | b"compute_62" => Arch::Sm62,
            // Volta/Turing
            b"sm_70" | b"compute_70" => Arch::Sm70,
            b"sm_72" | b"compute_72" => Arch::Sm72,
            b"sm_75" | b"compute_75" => Arch::Sm75,
            // Ampere
            b"sm_80" | b"compute_80" => Arch::Sm80,
            b"sm_86" | b"compute_86" => Arch::Sm86,
            b"sm_87" | b"compute_87" => Arch::Sm87,
            b"sm_88" | b"compute_88" => Arch::Sm88,
            b"sm_89" | b"compute_89" => Arch::Sm89,
            // Hopper
            b"sm_90" | b"compute_90" => Arch::Sm90,
            b"sm_90a" | b"compute_90a" => Arch::Sm90a,
            // Blackwell
            b"sm_100" | b"compute_100" => Arch::Sm100,
            b"sm_100f" | b"compute_100f" => Arch::Sm100f,
            b"sm_100a" | b"compute_100a" => Arch::Sm100a,
            // sm_101 renamed to sm_110 in PTX ISA 9.0
            b"sm_101" | b"compute_101" => Arch::Sm110,
            b"sm_101f" | b"compute_101f" => Arch::Sm110f,
            b"sm_101a" | b"compute_101a" => Arch::Sm110a,
            b"sm_103" | b"compute_103" => Arch::Sm103,
            b"sm_103f" | b"compute_103f" => Arch::Sm103f,
            b"sm_103a" | b"compute_103a" => Arch::Sm103a,
            // sm_11x
            b"sm_110" | b"compute_110" => Arch::Sm110,
            b"sm_110f" | b"compute_110f" => Arch::Sm110f,
            b"sm_110a" | b"compute_110a" => Arch::Sm110a,
            // sm_12x
            b"sm_120" | b"compute_120" => Arch::Sm120,
            b"sm_120f" | b"compute_120f" => Arch::Sm120f,
            b"sm_120a" | b"compute_120a" => Arch::Sm120a,
            b"sm_121" | b"compute_121" => Arch::Sm121,
            b"sm_121f" | b"compute_121f" => Arch::Sm121f,
            b"sm_121a" | b"compute_121a" => Arch::Sm121a,
            _ => return None,
        })
    }
}

impl Arch {
    pub fn as_str(&self) -> &str {
        match self {
            Arch::Sm10 => "sm_10",
            Arch::Sm11 => "sm_11",
            Arch::Sm12 => "sm_12",
            Arch::Sm13 => "sm_13",
            Arch::Sm20 => "sm_20",
            Arch::Sm30 => "sm_30",
            Arch::Sm32 => "sm_32",
            Arch::Sm35 => "sm_35",
            Arch::Sm37 => "sm_37",
            Arch::Sm50 => "sm_50",
            Arch::Sm52 => "sm_52",
            Arch::Sm53 => "sm_53",
            Arch::Sm60 => "sm_60",
            Arch::Sm61 => "sm_61",
            Arch::Sm62 => "sm_62",
            Arch::Sm70 => "sm_70",
            Arch::Sm72 => "sm_72",
            Arch::Sm75 => "sm_75",
            Arch::Sm80 => "sm_80",
            Arch::Sm86 => "sm_86",
            Arch::Sm87 => "sm_87",
            Arch::Sm88 => "sm_88",
            Arch::Sm89 => "sm_89",
            Arch::Sm90 => "sm_90",
            Arch::Sm90a => "sm_90a",
            Arch::Sm100 => "sm_100",
            Arch::Sm100f => "sm_100f",
            Arch::Sm100a => "sm_100a",
            Arch::Sm103 => "sm_103",
            Arch::Sm103f => "sm_103f",
            Arch::Sm103a => "sm_103a",
            Arch::Sm110 => "sm_110",
            Arch::Sm110f => "sm_110f",
            Arch::Sm110a => "sm_110a",
            Arch::Sm120 => "sm_120",
            Arch::Sm120f => "sm_120f",
            Arch::Sm120a => "sm_120a",
            Arch::Sm121 => "sm_121",
            Arch::Sm121f => "sm_121f",
            Arch::Sm121a => "sm_121a",
        }
    }
}

/// Texture mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Texmode {
    Unified,
    Independent,
}

impl FromAscii for Texmode {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"texmode_unified" => Texmode::Unified,
            b"texmode_independent" => Texmode::Independent,
            _ => return None,
        })
    }
}

impl Texmode {
    pub fn as_str(&self) -> &str {
        match self {
            Texmode::Unified => "texmode_unified",
            Texmode::Independent => "texmode_independent",
        }
    }
}

/// Address size: `.address_size 32` or `.address_size 64`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressSize {
    Bits32,
    Bits64,
}

/// Top-level items in a PTX module
#[derive(Debug, Clone)]
pub enum TopLevelItem {
    /// Variable declaration at module scope
    Variable(VarDecl),
    /// Function definition
    Function(Function),
    /// Kernel entry point
    Entry(Function),
    /// File directive for debug info
    File(FileDirective),
    /// Other directives
    Directive(Directive),
}

// =============================================================================
// Types
// =============================================================================

/// Fundamental PTX types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    // Signed integers
    S8,
    S16,
    S32,
    S64,
    // Unsigned integers
    U8,
    U16,
    U32,
    U64,
    // Packed integers
    S16x2,
    U16x2,
    // Floating point
    F16,
    F16x2,
    Bf16,
    Bf16x2,
    F32,
    F32x2,
    F64,
    Tf32,
    // Bit-size types
    B8,
    B16,
    B32,
    B64,
    B128,
    B1024,
    // Predicate
    Pred,
}

impl ScalarType {
    /// Size in bits
    pub fn bits(&self) -> u32 {
        match self {
            ScalarType::S8 | ScalarType::U8 | ScalarType::B8 => 8,
            ScalarType::S16
            | ScalarType::U16
            | ScalarType::F16
            | ScalarType::Bf16
            | ScalarType::B16 => 16,
            ScalarType::S32
            | ScalarType::U32
            | ScalarType::F32
            | ScalarType::F16x2
            | ScalarType::Bf16x2
            | ScalarType::S16x2
            | ScalarType::U16x2
            | ScalarType::Tf32
            | ScalarType::B32 => 32,
            ScalarType::S64
            | ScalarType::U64
            | ScalarType::F64
            | ScalarType::F32x2
            | ScalarType::B64 => 64,
            ScalarType::B128 => 128,
            ScalarType::B1024 => 1024,
            ScalarType::Pred => 1,
        }
    }
}

impl FromAscii for ScalarType {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"s8" => ScalarType::S8,
            b"s16" => ScalarType::S16,
            b"s32" => ScalarType::S32,
            b"s64" => ScalarType::S64,
            b"u8" => ScalarType::U8,
            b"u16" => ScalarType::U16,
            b"u32" => ScalarType::U32,
            b"u64" => ScalarType::U64,
            b"s16x2" => ScalarType::S16x2,
            b"u16x2" => ScalarType::U16x2,
            b"f16" => ScalarType::F16,
            b"f16x2" => ScalarType::F16x2,
            b"bf16" => ScalarType::Bf16,
            b"bf16x2" => ScalarType::Bf16x2,
            b"f32" => ScalarType::F32,
            b"f32x2" => ScalarType::F32x2,
            b"f64" => ScalarType::F64,
            b"tf32" => ScalarType::Tf32,
            b"b8" => ScalarType::B8,
            b"b16" => ScalarType::B16,
            b"b32" => ScalarType::B32,
            b"b64" => ScalarType::B64,
            b"b128" => ScalarType::B128,
            b"b1024" => ScalarType::B1024,
            b"pred" => ScalarType::Pred,
            _ => return None,
        })
    }
}

/// Vector width modifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecWidth {
    V2,
    V4,
    V8,
}

impl FromAscii for VecWidth {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"v2" => VecWidth::V2,
            b"v4" => VecWidth::V4,
            b"v8" => VecWidth::V8,
            _ => return None,
        })
    }
}

impl VecWidth {
    pub fn count(&self) -> u32 {
        match self {
            VecWidth::V2 => 2,
            VecWidth::V4 => 4,
            VecWidth::V8 => 8,
        }
    }
}

/// A type with optional vector width
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Type {
    pub vec: Option<VecWidth>,
    pub scalar: ScalarType,
}

/// A pointer type attribute
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ptr {
    pub space: Option<PtrSpace>,
    pub align: Option<u32>,
}

/// Possible pointer spaces
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtrSpace {
    Const,
    Global,
    Local,
    Shared,
}

impl FromAscii for PtrSpace {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"const" => PtrSpace::Const,
            b"global" => PtrSpace::Global,
            b"local" => PtrSpace::Local,
            b"shared" => PtrSpace::Shared,
            _ => return None,
        })
    }
}

// =============================================================================
// State Spaces
// =============================================================================

/// PTX state spaces
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateSpace {
    /// Register (`.reg`)
    Reg,
    /// Special register (`.sreg`)
    Sreg,
    /// Constant memory (`.const`)
    Const,
    /// Global memory (`.global`)
    Global,
    /// Local memory (`.local`)
    Local,
    /// Parameter (`.param`)
    Param,
    /// Shared memory (`.shared`)
    Shared,
    /// Texture memory (`.tex`) - deprecated
    Tex,
}

impl FromAscii for StateSpace {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"reg" => StateSpace::Reg,
            b"sreg" => StateSpace::Sreg,
            b"const" => StateSpace::Const,
            b"global" => StateSpace::Global,
            b"local" => StateSpace::Local,
            b"param" => StateSpace::Param,
            b"shared" => StateSpace::Shared,
            b"tex" => StateSpace::Tex,
            _ => return None,
        })
    }
}

/// Sub-qualifier for shared/param state spaces
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateSpaceQualifier {
    /// `::cta` - CTA-local shared memory
    Cta,
    /// `::cluster` - Cluster-wide shared memory
    Cluster,
    /// `::entry` - Kernel entry parameter
    Entry,
    /// `::func` - Device function parameter
    Func,
}

// =============================================================================
// Linkage and Visibility
// =============================================================================

/// Linkage specifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Linkage {
    #[default]
    None,
    /// `.extern` - external linkage
    Extern,
    /// `.visible` - visible to other modules
    Visible,
    /// `.weak` - weak linkage
    Weak,
}

impl FromAscii for Linkage {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"extern" => Linkage::Extern,
            b"visible" => Linkage::Visible,
            b"weak" => Linkage::Weak,
            _ => return None,
        })
    }
}

// =============================================================================
// Variable Declarations
// =============================================================================

/// A variable declaration
#[derive(Debug, Clone)]
pub struct VarDecl {
    pub span: Span,
    pub linkage: Linkage,
    pub space: StateSpace,
    pub space_qualifier: Option<StateSpaceQualifier>,
    pub align: Option<u32>,
    pub ty: Type,
    pub name: AsciiString,
    /// For parameterized names like `%r<100>`, this is the count
    pub param_count: Option<u32>,
    /// Array dimensions, empty for scalars
    pub array_dims: Vec<Option<u32>>, // None means unsized `[]`
    /// Optional initializer
    pub init: Option<Initializer>,
}

/// Variable initializer
#[derive(Debug, Clone)]
pub enum Initializer {
    /// Single value
    Scalar(InitValue),
    /// Aggregate initializer `{ ... }`
    Aggregate(Vec<Initializer>),
}

/// A value in an initializer
#[derive(Debug, Clone)]
pub enum InitValue {
    /// Integer literal
    Int(i64),
    /// Unsigned integer literal
    UInt(u64),
    /// Floating point literal
    Float(f64),
    /// Symbol reference
    Symbol(AsciiString),
    /// Symbol + offset
    SymbolOffset(AsciiString, i64),
    /// Generic address: `generic(symbol)`
    Generic(AsciiString),
}

// =============================================================================
// Functions
// =============================================================================

/// Performance-tuning directives for kernel functions
#[derive(Debug, Clone, Default)]
pub struct PerformanceDirectives {
    /// Maximum number of registers per thread (.maxnreg)
    pub max_nreg: Option<u32>,
    /// Maximum thread block dimensions (.maxntid nx, ny, nz)
    pub max_ntid: Option<(u32, Option<u32>, Option<u32>)>,
    /// Required thread block dimensions (.reqntid nx, ny, nz)
    pub req_ntid: Option<(u32, Option<u32>, Option<u32>)>,
    /// Minimum CTAs per SM (.minnctapersm)
    pub min_ncta_per_sm: Option<u32>,
    /// Maximum CTAs per SM (.maxnctapersm) - deprecated
    pub max_ncta_per_sm: Option<u32>,
    /// Function does not return (.noreturn)
    pub noreturn: bool,
    /// Pragma strings
    pub pragmas: Vec<AsciiString>,
}

/// A function or entry point definition. Whether this is a `.entry` kernel
/// or a `.func` device function is encoded by the enclosing `TopLevelItem`
/// variant (`Entry` vs `Function`).
#[derive(Debug, Clone)]
pub struct Function {
    pub linkage: Linkage,
    /// Return parameters (for `.func`)
    pub return_params: Vec<Parameter>,
    /// Function name
    pub name: AsciiString,
    /// Span of the function name
    pub name_span: Span,
    /// Input parameters
    pub params: Vec<Parameter>,
    /// Performance-tuning directives
    pub perf_directives: PerformanceDirectives,
    /// Function body (None for declarations)
    pub body: Option<FunctionBody>,
}

/// A function parameter
#[derive(Debug, Clone)]
pub struct Parameter {
    pub span: Span,
    pub space: StateSpace,
    pub align: Option<u32>,
    pub ty: Type,
    /// Optional pointer attribute for pointer parameters
    pub ptr: Option<Ptr>,
    pub name: AsciiString,
    /// Array dimensions for byte-array parameters
    pub array_dims: Vec<u32>,
}

/// Function body
#[derive(Debug, Clone)]
pub struct FunctionBody {
    pub statements: Vec<Statement>,
}

// =============================================================================
// Statements
// =============================================================================

/// A label definition
#[derive(Debug, Clone)]
pub struct Label {
    pub span: Span,
    pub name: AsciiString,
}

/// A statement in a function body
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)] // parser AST: boxing would complicate construction for little gain
pub enum Statement {
    /// Label definition
    Label(Label),
    /// Variable declaration (local)
    Variable(VarDecl),
    /// Instruction
    Instruction(Instruction),
    /// Block of statements `{ ... }`
    Block(Vec<Statement>),
    /// Directive (e.g., `.pragma`, `.loc`)
    Directive(Directive),
}

// =============================================================================
// Instructions
// =============================================================================

/// A PTX instruction with optional predicate
#[derive(Debug, Clone)]
pub struct Instruction {
    pub span: Span,
    /// Guard predicate: `@p` or `@!p`
    pub predicate: Option<Predicate>,
    /// The instruction itself
    pub op: InstructionOp,
}

/// Predicate guard for an instruction
#[derive(Debug, Clone)]
pub struct Predicate {
    pub negated: bool,
    pub reg: AsciiString,
}

/// The instruction operation
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)] // parser AST: boxing would complicate construction for little gain
pub enum InstructionOp {
    /// Parsed instruction with all modifiers resolved
    Parsed(ParsedInstruction),
    /// Unparsed instruction (for incremental development)
    /// Contains the opcode and raw modifiers
    Unparsed {
        kind: InstrKind,
        modifiers: Vec<DottedIdent>,
        operands: Vec<Operand>,
    },
}

/// A fully parsed instruction with strongly-typed modifiers
#[derive(Debug, Clone)]
pub enum ParsedInstruction {
    // =========================================================================
    // Integer Arithmetic Instructions (Blocks 1-11)
    // =========================================================================
    Add(AddInstr),
    Sub(SubInstr),
    Mul(MulInstr),
    Mad(MadInstr),
    Mul24(Mul24Instr),
    Mad24(Mad24Instr),
    Sad(SadInstr),
    Div(DivInstr),
    Rem(RemInstr),
    Abs(AbsInstr),
    Neg(NegInstr),
    Min(MinInstr),
    Max(MaxInstr),

    // =========================================================================
    // Bit Manipulation Instructions (Blocks 14-24)
    // =========================================================================
    Popc(PopcInstr),
    Clz(ClzInstr),
    Bfind(BfindInstr),
    Fns(FnsInstr),
    Brev(BrevInstr),
    Bfe(BfeInstr),
    Bfi(BfiInstr),
    Szext(SzextInstr),
    Bmsk(BmskInstr),
    Dp4a(Dp4aInstr),
    Dp2a(Dp2aInstr),

    // =========================================================================
    // Extended-Precision Integer Arithmetic (Blocks 25-30)
    // =========================================================================
    AddCc(AddCcInstr),
    Addc(AddcInstr),
    SubCc(SubCcInstr),
    Subc(SubcInstr),
    MadCc(MadCcInstr),
    Madc(MadcInstr),

    // =========================================================================
    // Floating-Point Instructions (Blocks 31-52)
    // =========================================================================
    Testp(TestpInstr),
    Copysign(CopysignInstr),
    Fma(FmaInstr),
    Rcp(RcpInstr),
    Sqrt(SqrtInstr),
    Rsqrt(RsqrtInstr),
    Sin(SinInstr),
    Cos(CosInstr),
    Lg2(Lg2Instr),
    Ex2(Ex2Instr),
    Tanh(TanhInstr),

    // =========================================================================
    // Comparison and Selection Instructions (Blocks 66-71)
    // =========================================================================
    Set(SetInstr),
    Setp(SetpInstr),
    Selp(SelpInstr),
    Slct(SlctInstr),

    // =========================================================================
    // Logic and Shift Instructions (Blocks 72-85)
    // =========================================================================
    And(LogicInstr),
    Or(LogicInstr),
    Xor(LogicInstr),
    Not(NotInstr),
    Cnot(CnotInstr),
    Lop3(Lop3Instr),
    Shf(ShfInstr),
    Shl(ShiftInstr),
    Shr(ShiftInstr),

    // =========================================================================
    // Data Movement Instructions (Blocks 81-102)
    // =========================================================================
    Mov(MovInstr),
    Shfl(ShflInstr),
    ShflSync(ShflSyncInstr),
    Prmt(PrmtInstr),
    Ld(LdInstr),
    Ldu(LduInstr),
    St(StInstr),
    Prefetch(PrefetchInstr),
    Cvt(CvtInstr),
    Cvta(CvtaInstr),
    Isspacep(IsspacepInstr),
    Mapa(MapaInstr),
    Getctarank(GetctarankInstr),

    // =========================================================================
    // Control Flow Instructions (Blocks 127-131)
    // =========================================================================
    Bra(BraInstr),
    BrxIdx(BrxIdxInstr),
    Call(CallInstr),
    Ret(RetInstr),
    Exit,

    // =========================================================================
    // Synchronization Instructions (Blocks 132-145)
    // =========================================================================
    Bar(BarInstr),
    Barrier(BarrierInstr),
    BarWarpSync(BarWarpSyncInstr),
    BarrierCluster(BarrierClusterInstr),
    Membar(MembarInstr),
    Fence(FenceInstr),
    Atom(AtomInstr),
    Red(RedInstr),
    Vote(VoteInstr),
    VoteSync(VoteSyncInstr),
    MatchSync(MatchSyncInstr),
    Activemask(ActivemaskInstr),
    ReduxSync(ReduxSyncInstr),
    Griddepcontrol(GriddepcontrolInstr),
    ElectSync(ElectSyncInstr),

    // =========================================================================
    // Mbarrier Instructions (Blocks 146-154)
    // =========================================================================
    MbarrierInit(MbarrierInitInstr),
    MbarrierInval(MbarrierInvalInstr),
    MbarrierExpectTx(MbarrierExpectTxInstr),
    MbarrierCompleteTx(MbarrierCompleteTxInstr),
    MbarrierArrive(MbarrierArriveInstr),
    MbarrierArriveDrop(MbarrierArriveDropInstr),
    MbarrierTestWait(MbarrierTestWaitInstr),
    MbarrierTryWait(MbarrierTryWaitInstr),
    MbarrierPendingCount(MbarrierPendingCountInstr),

    // =========================================================================
    // Stack Instructions (Blocks 183-185)
    // =========================================================================
    Stacksave(StacksaveInstr),
    Stackrestore(StackrestoreInstr),
    Alloca(AllocaInstr),

    // =========================================================================
    // Miscellaneous Instructions (Blocks 194-198)
    // =========================================================================
    Brkpt,
    Nanosleep(NanosleepInstr),
    Pmevent(PmeventInstr),
    Trap,
    Setmaxnreg(SetmaxnregInstr),

    // Other - catch-all for complex instructions (matrix ops, texture, etc.)
    Other {
        kind: InstrKind,
        modifiers: Vec<DottedIdent>,
        operands: Vec<Operand>,
    },
}

// =============================================================================
// Constant Expressions
// =============================================================================

/// A constant expression (used in initializers, array bounds, address offsets)
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Integer literal (signed)
    IntLitS(i64),
    /// Integer literal (unsigned)
    IntLitU(u64),
    /// Identifier (e.g., symbol name)
    Ident(AsciiString),
    /// Unary expression
    Unary(UnaryOp, Box<Expr>),
    /// Binary expression
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
    /// Ternary conditional: cond ? then : else
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
}

/// Unary operators for constant expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Unary plus (+)
    Pos,
    /// Unary minus (-)
    Neg,
    /// Logical not (!)
    Not,
    /// Bitwise complement (~)
    BitNot,
    /// Cast to signed 64-bit (.s64)
    CastS64,
    /// Cast to unsigned 64-bit (.u64)
    CastU64,
}

/// Binary operators for constant expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    // Shifts
    Shl,
    Shr,
    // Comparison
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    // Bitwise
    BitAnd,
    BitXor,
    BitOr,
    // Logical
    And,
    Or,
}

// =============================================================================
// Operands
// =============================================================================

/// An instruction operand
#[derive(Debug, Clone)]
pub enum Operand {
    /// Identifier: register (`%r0`), shared memory (`buf_shmem`), label, etc.
    /// The actual meaning is resolved during lowering based on declarations.
    Ident(AsciiString),
    /// Immediate integer
    ImmInt(i64),
    /// Immediate unsigned integer
    ImmUInt(u64),
    /// Immediate float
    ImmFloat(f64),
    /// Symbol reference
    Symbol(AsciiString),
    /// Constant expression (for complex immediates)
    Expr(Box<Expr>),
    /// Memory address: `[addr]`
    Address(Address),
    /// Vector register with component: `%v.x`
    VectorElement(AsciiString, VectorComponent),
    /// Vector operand: `{%r0, %r1, %r2, %r3}`
    Vector(Vec<Operand>),
    /// Underscore (sink/don't care)
    Underscore,
    /// Predicate with optional negation for setp: `!p`
    PredicateOperand { negated: bool, name: AsciiString },
    /// Two predicates separated by `|` for setp: `p|q`
    PredicatePair(AsciiString, AsciiString),
}

/// Canonical vector component selector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonVectorComponent {
    X,
    Y,
    Z,
    W,
}

/// Vector component selector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorComponent {
    X,
    Y,
    Z,
    W,
    R,
    G,
    B,
    A,
}

impl FromAscii for VectorComponent {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"x" => VectorComponent::X,
            b"y" => VectorComponent::Y,
            b"z" => VectorComponent::Z,
            b"w" => VectorComponent::W,
            b"r" => VectorComponent::R,
            b"g" => VectorComponent::G,
            b"b" => VectorComponent::B,
            b"a" => VectorComponent::A,
            _ => return None,
        })
    }
}

impl VectorComponent {
    pub fn canonicalize(self) -> CanonVectorComponent {
        match self {
            // Keep XYZW as is
            VectorComponent::X => CanonVectorComponent::X,
            VectorComponent::Y => CanonVectorComponent::Y,
            VectorComponent::Z => CanonVectorComponent::Z,
            VectorComponent::W => CanonVectorComponent::W,
            // Map RGBA to XYZW
            VectorComponent::R => CanonVectorComponent::X,
            VectorComponent::G => CanonVectorComponent::Y,
            VectorComponent::B => CanonVectorComponent::Z,
            VectorComponent::A => CanonVectorComponent::W,
        }
    }
}

/// Memory address operand
#[derive(Debug, Clone)]
pub struct Address {
    /// Base: register or symbol
    pub base: AddressBase,
    /// Offset expression (can be negative)
    pub offset: Option<Box<Expr>>,
}

/// Base of an address
#[derive(Debug, Clone)]
pub enum AddressBase {
    /// Register base: `[%rd0]`
    Register(AsciiString),
    /// Symbol base: `[arr]`
    Symbol(AsciiString),
    /// Immediate address: `[0x1000]` (only for .local)
    Immediate(i64),
}

// =============================================================================
// Instruction-Specific Structures
// =============================================================================

// --- Arithmetic Instructions ---

/// Add instruction with type-specific forms
///
/// Integer (Block 1):
/// - `add.type d, a, b` where type = u16, u32, u64, s16, s64, u16x2, s16x2
/// - `add{.sat}.s32 d, a, b` - sat only for s32
///
/// Float (Block 33):
/// - `add{.rnd}{.ftz}{.sat}.f32 d, a, b`
/// - `add{.rnd}{.ftz}.f32x2 d, a, b` (no sat)
/// - `add{.rnd}.f64 d, a, b` (no ftz, no sat)
///
/// Half (Block 53):
/// - `add{.rnd}{.ftz}{.sat}.f16 d, a, b`
/// - `add{.rnd}{.ftz}{.sat}.f16x2 d, a, b`
/// - `add{.rnd}.bf16 d, a, b` (no ftz, no sat)
/// - `add{.rnd}.bf16x2 d, a, b`
///
/// Mixed precision (Block 63):
/// - `add{.rnd}{.sat}.f32.atype d, a, c` where atype = f16/bf16 (no ftz)
#[derive(Debug, Clone)]
pub enum AddInstr {
    /// Integer add: `add.type d, a, b` (no modifiers)
    Integer {
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Integer add with saturation: `add{.sat}.s32 d, a, b`
    IntegerSat {
        sat: bool,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Float32: `add{.rnd}{.ftz}{.sat}.f32 d, a, b`
    Float32 {
        rnd: Option<FpRound>,
        ftz: bool,
        sat: bool,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Float32x2: `add{.rnd}{.ftz}.f32x2 d, a, b` (no sat)
    Float32x2 {
        rnd: Option<FpRound>,
        ftz: bool,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Float64: `add{.rnd}.f64 d, a, b` (no ftz, no sat)
    Float64 {
        rnd: Option<FpRound>,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Half f16/f16x2: `add{.rnd}{.ftz}{.sat}.type d, a, b`
    HalfF16 {
        rnd: Option<FpRound>,
        ftz: bool,
        sat: bool,
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Half bf16/bf16x2: `add{.rnd}.type d, a, b` (no ftz, no sat)
    HalfBf16 {
        rnd: Option<FpRound>,
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Mixed precision: `add{.rnd}{.sat}.f32.src_type d, a, b`
    MixedPrecision {
        rnd: Option<FpRound>,
        sat: bool,
        /// Source type (f16 or bf16)
        src_type: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
}

/// Sub instruction with type-specific forms
///
/// Integer (Block 2):
/// - `sub.type d, a, b` where type = u16, u32, u64, s16, s64
/// - `sub{.sat}.s32 d, a, b` - sat only for s32
///
/// Float (Block 34):
/// - `sub{.rnd}{.ftz}{.sat}.f32 d, a, b`
/// - `sub{.rnd}{.ftz}.f32x2 d, a, b` (no sat)
/// - `sub{.rnd}.f64 d, a, b` (no ftz, no sat)
///
/// Half (Block 54):
/// - `sub{.rnd}{.ftz}{.sat}.f16 d, a, b`
/// - `sub{.rnd}{.ftz}{.sat}.f16x2 d, a, b`
/// - `sub{.rnd}.bf16 d, a, b` (no ftz, no sat)
/// - `sub{.rnd}.bf16x2 d, a, b`
///
/// Mixed precision (Block 64):
/// - `sub{.rnd}{.sat}.f32.atype d, a, c` where atype = f16/bf16 (no ftz)
#[derive(Debug, Clone)]
pub enum SubInstr {
    /// Integer sub: `sub.type d, a, b` (no modifiers)
    Integer {
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Integer sub with saturation: `sub{.sat}.s32 d, a, b`
    IntegerSat {
        sat: bool,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Float32: `sub{.rnd}{.ftz}{.sat}.f32 d, a, b`
    Float32 {
        rnd: Option<FpRound>,
        ftz: bool,
        sat: bool,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Float32x2: `sub{.rnd}{.ftz}.f32x2 d, a, b` (no sat)
    Float32x2 {
        rnd: Option<FpRound>,
        ftz: bool,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Float64: `sub{.rnd}.f64 d, a, b` (no ftz, no sat)
    Float64 {
        rnd: Option<FpRound>,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Half f16/f16x2: `sub{.rnd}{.ftz}{.sat}.type d, a, b`
    HalfF16 {
        rnd: Option<FpRound>,
        ftz: bool,
        sat: bool,
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Half bf16/bf16x2: `sub{.rnd}.type d, a, b` (no ftz, no sat)
    HalfBf16 {
        rnd: Option<FpRound>,
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Mixed precision: `sub{.rnd}{.sat}.f32.src_type d, a, b`
    MixedPrecision {
        rnd: Option<FpRound>,
        sat: bool,
        /// Source type (f16 or bf16)
        src_type: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
}

/// Multiply instruction with mutually exclusive forms:
/// - Block 3 (integer): `mul.mode.type d, a, b` where mode is hi/lo/wide
/// - Block 35 (float): `mul{.rnd}{.ftz}{.sat}.f32/f64 d, a, b`
#[derive(Debug, Clone)]
pub enum MulInstr {
    /// Integer multiply: `mul.mode.type d, a, b`
    Integer {
        mode: MulMode, // required for integer
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Floating-point multiply: `mul{.rnd}{.ftz}{.sat}.type d, a, b`
    Float {
        rnd: Option<FpRound>,
        ftz: bool,
        sat: bool,
        ty: ScalarType, // f32, f32x2, f64
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
}

/// Multiply mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MulMode {
    Hi,
    Lo,
    Wide,
}

impl FromAscii for MulMode {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"hi" => MulMode::Hi,
            b"lo" => MulMode::Lo,
            b"wide" => MulMode::Wide,
            _ => return None,
        })
    }
}

/// Multiply-add instruction with mutually exclusive forms:
/// - Block 4 (integer): `mad.mode.type d, a, b, c` or `mad.hi.sat.s32 d, a, b, c`
/// - Block 37 (float): `mad{.ftz}{.sat}.f32` (legacy) or `mad.rnd{.ftz}{.sat}.f32/f64`
#[derive(Debug, Clone)]
pub enum MadInstr {
    /// Integer multiply-add: `mad.mode.type d, a, b, c`
    Integer {
        mode: MulMode, // required for integer
        sat: bool,     // only valid with .hi.s32
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
        src_c: Operand,
    },
    /// Floating-point multiply-add: `mad{.rnd}{.ftz}{.sat}.type d, a, b, c`
    Float {
        rnd: Option<FpRound>, // None for legacy sm_1x mode
        ftz: bool,
        sat: bool,
        ty: ScalarType, // f32 or f64
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
        src_c: Operand,
    },
}

/// Divide instruction with mutually exclusive forms:
/// - Block 8 (integer): `div.type d, a, b`
/// - Block 38 (float): `div.approx{.ftz}.f32`, `div.full{.ftz}.f32`, `div.rnd{.ftz}.f32/f64`
#[derive(Debug, Clone)]
pub enum DivInstr {
    /// Integer division: `div.type d, a, b`
    Integer {
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Fast approximate division: `div.approx{.ftz}.f32 d, a, b`
    Approx {
        ftz: bool,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Full-range approximate division: `div.full{.ftz}.f32 d, a, b`
    Full {
        ftz: bool,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// IEEE 754 compliant division: `div.rnd{.ftz}.f32 d, a, b` or `div.rnd.f64 d, a, b`
    Ieee {
        rnd: FpRound,
        ftz: bool,
        ty: ScalarType, // f32 or f64
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
}

/// Remainder: `rem.type d, a, b`
#[derive(Debug, Clone)]
pub struct RemInstr {
    pub ty: ScalarType,
    pub dst: Operand,
    pub src_a: Operand,
    pub src_b: Operand,
}

/// Absolute value instruction with type-specific forms
///
/// Integer (Block 10):
/// - `abs.type d, a` where type = s16, s32, s64 (no ftz)
///
/// Float (Block 39):
/// - `abs{.ftz}.f32 d, a`
/// - `abs.f64 d, a` (no ftz)
///
/// Half (Block 58):
/// - `abs{.ftz}.f16 d, a`
/// - `abs{.ftz}.f16x2 d, a`
/// - `abs.bf16 d, a` (no ftz)
/// - `abs.bf16x2 d, a` (no ftz)
#[derive(Debug, Clone)]
pub enum AbsInstr {
    /// Integer abs: `abs.type d, a` (no ftz)
    Integer {
        ty: ScalarType,
        dst: Operand,
        src: Operand,
    },
    /// Float32 abs: `abs{.ftz}.f32 d, a`
    Float32 {
        ftz: bool,
        dst: Operand,
        src: Operand,
    },
    /// Float64 abs: `abs.f64 d, a` (no ftz)
    Float64 { dst: Operand, src: Operand },
    /// Half f16/f16x2: `abs{.ftz}.type d, a`
    HalfF16 {
        ftz: bool,
        ty: ScalarType,
        dst: Operand,
        src: Operand,
    },
    /// Half bf16/bf16x2: `abs.type d, a` (no ftz)
    HalfBf16 {
        ty: ScalarType,
        dst: Operand,
        src: Operand,
    },
}

/// Negate instruction with type-specific forms
///
/// Integer (Block 11):
/// - `neg.type d, a` where type = s16, s32, s64 (no ftz)
///
/// Float (Block 40):
/// - `neg{.ftz}.f32 d, a`
/// - `neg.f64 d, a` (no ftz)
///
/// Half (Block 57):
/// - `neg{.ftz}.f16 d, a`
/// - `neg{.ftz}.f16x2 d, a`
/// - `neg.bf16 d, a` (no ftz)
/// - `neg.bf16x2 d, a` (no ftz)
#[derive(Debug, Clone)]
pub enum NegInstr {
    /// Integer neg: `neg.type d, a` (no ftz)
    Integer {
        ty: ScalarType,
        dst: Operand,
        src: Operand,
    },
    /// Float32 neg: `neg{.ftz}.f32 d, a`
    Float32 {
        ftz: bool,
        dst: Operand,
        src: Operand,
    },
    /// Float64 neg: `neg.f64 d, a` (no ftz)
    Float64 { dst: Operand, src: Operand },
    /// Half f16/f16x2: `neg{.ftz}.type d, a`
    HalfF16 {
        ftz: bool,
        ty: ScalarType,
        dst: Operand,
        src: Operand,
    },
    /// Half bf16/bf16x2: `neg.type d, a` (no ftz)
    HalfBf16 {
        ty: ScalarType,
        dst: Operand,
        src: Operand,
    },
}

/// Min instruction with type-specific forms
///
/// Integer (Block 12):
/// - `min.atype d, a, b` where atype = u16, u32, u64, u16x2, s16, s64
/// - `min{.relu}.btype d, a, b` where btype = s16x2, s32
///
/// Float (Blocks 41, 59):
/// - `min{.ftz}{.NaN}{.xorsign.abs}.f32 d, a, b`
/// - `min{.ftz}{.NaN}{.abs}.f32 d, a, b, c`
/// - `min.f64 d, a, b`
/// - `min{.ftz}{.NaN}{.xorsign.abs}.f16 d, a, b` (ftz for f16/f16x2 only)
/// - `min{.NaN}{.xorsign.abs}.bf16 d, a, b`
#[derive(Debug, Clone)]
pub enum MinInstr {
    /// Integer min: `min.type d, a, b` (no modifiers)
    Integer {
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Integer min with relu: `min{.relu}.type d, a, b` (only s16x2, s32)
    IntegerRelu {
        relu: bool,
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Float32 2-operand: `min{.ftz}{.NaN}{.xorsign.abs}.f32 d, a, b`
    Float32 {
        ftz: bool,
        nan: bool,
        /// xorsign.abs modifier (xorsign requires abs)
        xorsign_abs: bool,
        /// abs without xorsign
        abs: bool,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Float32 3-operand: `min{.ftz}{.NaN}{.abs}.f32 d, a, b, c`
    Float32Acc {
        ftz: bool,
        nan: bool,
        abs: bool,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
        src_c: Operand,
    },
    /// Float64: `min.f64 d, a, b` (no modifiers)
    Float64 {
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Half precision (f16/f16x2): `min{.ftz}{.NaN}{.xorsign.abs}.type d, a, b`
    HalfF16 {
        ftz: bool,
        nan: bool,
        xorsign_abs: bool,
        abs: bool,
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Half precision (bf16/bf16x2): `min{.NaN}{.xorsign.abs}.type d, a, b` (no ftz)
    HalfBf16 {
        nan: bool,
        xorsign_abs: bool,
        abs: bool,
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
}

/// Max instruction with type-specific forms
///
/// Integer (Block 13):
/// - `max.atype d, a, b` where atype = u16, u32, u64, u16x2, s16, s64
/// - `max{.relu}.btype d, a, b` where btype = s16x2, s32
///
/// Float (Blocks 42, 60):
/// - `max{.ftz}{.NaN}{.xorsign.abs}.f32 d, a, b`
/// - `max{.ftz}{.NaN}{.abs}.f32 d, a, b, c`
/// - `max.f64 d, a, b`
/// - `max{.ftz}{.NaN}{.xorsign.abs}.f16 d, a, b` (ftz for f16/f16x2 only)
/// - `max{.NaN}{.xorsign.abs}.bf16 d, a, b`
#[derive(Debug, Clone)]
pub enum MaxInstr {
    /// Integer max: `max.type d, a, b` (no modifiers)
    Integer {
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Integer max with relu: `max{.relu}.type d, a, b` (only s16x2, s32)
    IntegerRelu {
        relu: bool,
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Float32 2-operand: `max{.ftz}{.NaN}{.xorsign.abs}.f32 d, a, b`
    Float32 {
        ftz: bool,
        nan: bool,
        /// xorsign.abs modifier (xorsign requires abs)
        xorsign_abs: bool,
        /// abs without xorsign
        abs: bool,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Float32 3-operand: `max{.ftz}{.NaN}{.abs}.f32 d, a, b, c`
    Float32Acc {
        ftz: bool,
        nan: bool,
        abs: bool,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
        src_c: Operand,
    },
    /// Float64: `max.f64 d, a, b` (no modifiers)
    Float64 {
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Half precision (f16/f16x2): `max{.ftz}{.NaN}{.xorsign.abs}.type d, a, b`
    HalfF16 {
        ftz: bool,
        nan: bool,
        xorsign_abs: bool,
        abs: bool,
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Half precision (bf16/bf16x2): `max{.NaN}{.xorsign.abs}.type d, a, b` (no ftz)
    HalfBf16 {
        nan: bool,
        xorsign_abs: bool,
        abs: bool,
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
}

// --- Floating-Point Instructions ---

/// Floating-point rounding mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpRound {
    /// Round to nearest even
    Rn,
    /// Round toward zero
    Rz,
    /// Round toward negative infinity
    Rm,
    /// Round toward positive infinity
    Rp,
}

impl FromAscii for FpRound {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"rn" => FpRound::Rn,
            b"rz" => FpRound::Rz,
            b"rm" => FpRound::Rm,
            b"rp" => FpRound::Rp,
            _ => return None,
        })
    }
}

/// Integer rounding mode (for cvt)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntRound {
    Rni,
    Rzi,
    Rmi,
    Rpi,
}

impl FromAscii for IntRound {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"rni" => IntRound::Rni,
            b"rzi" => IntRound::Rzi,
            b"rmi" => IntRound::Rmi,
            b"rpi" => IntRound::Rpi,
            _ => return None,
        })
    }
}

/// FMA instruction with type-specific forms
///
/// Float (Block 36):
/// - `fma.rnd{.ftz}{.sat}.f32 d, a, b, c`
/// - `fma.rnd{.ftz}.f32x2 d, a, b, c` (no sat)
/// - `fma.rnd.f64 d, a, b, c` (no ftz, no sat)
///
/// Half (Block 56):
/// - `fma.rnd{.ftz}{.sat}.f16 d, a, b, c` (sat mode)
/// - `fma.rnd{.ftz}.relu.f16 d, a, b, c` (relu mode - mutually exclusive with sat)
/// - `fma.rnd{.relu}.bf16 d, a, b, c` (no ftz, relu optional)
/// - `fma.rnd.oob{.relu}.type d, a, b, c` (oob mode)
///
/// Mixed precision (Block 65):
/// - `fma.rnd{.sat}.f32.abtype d, a, b, c` where abtype = f16, bf16 (no ftz)
#[derive(Debug, Clone)]
pub enum FmaInstr {
    /// Float32: `fma.rnd{.ftz}{.sat}.f32 d, a, b, c`
    Float32 {
        rnd: FpRound,
        ftz: bool,
        sat: bool,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
        src_c: Operand,
    },
    /// Float32x2: `fma.rnd{.ftz}.f32x2 d, a, b, c` (no sat)
    Float32x2 {
        rnd: FpRound,
        ftz: bool,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
        src_c: Operand,
    },
    /// Float64: `fma.rnd.f64 d, a, b, c` (no ftz, no sat)
    Float64 {
        rnd: FpRound,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
        src_c: Operand,
    },
    /// Half f16/f16x2 with sat: `fma.rnd{.ftz}{.sat}.type d, a, b, c`
    HalfF16Sat {
        rnd: FpRound,
        ftz: bool,
        sat: bool,
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
        src_c: Operand,
    },
    /// Half f16/f16x2 with relu: `fma.rnd{.ftz}.relu.type d, a, b, c`
    HalfF16Relu {
        rnd: FpRound,
        ftz: bool,
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
        src_c: Operand,
    },
    /// Half bf16/bf16x2: `fma.rnd{.relu}.type d, a, b, c` (no ftz, no sat)
    HalfBf16 {
        rnd: FpRound,
        relu: bool,
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
        src_c: Operand,
    },
    /// OOB mode: `fma.rnd.oob{.relu}.type d, a, b, c` (half types)
    Oob {
        rnd: FpRound,
        relu: bool,
        ty: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
        src_c: Operand,
    },
    /// Mixed precision: `fma.rnd{.sat}.f32.src_type d, a, b, c` (no ftz)
    MixedPrecision {
        rnd: FpRound,
        sat: bool,
        /// Source type (f16 or bf16)
        src_type: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
        src_c: Operand,
    },
}

/// Reciprocal instruction with mutually exclusive forms:
/// - Block 43: `rcp.approx{.ftz}.f32`, `rcp.rnd{.ftz}.f32`, `rcp.rnd.f64`
/// - Block 44: `rcp.approx.ftz.f64`
#[derive(Debug, Clone)]
pub enum RcpInstr {
    /// Fast approximate reciprocal: `rcp.approx{.ftz}.f32 d, a` or `rcp.approx.ftz.f64 d, a`
    Approx {
        ftz: bool,
        ty: ScalarType, // f32 or f64
        dst: Operand,
        src: Operand,
    },
    /// IEEE 754 compliant reciprocal: `rcp.rnd{.ftz}.f32 d, a` or `rcp.rnd.f64 d, a`
    Ieee {
        rnd: FpRound,
        ftz: bool,
        ty: ScalarType, // f32 or f64
        dst: Operand,
        src: Operand,
    },
}

/// Square root instruction with mutually exclusive forms:
/// - Block 45: `sqrt.approx{.ftz}.f32`, `sqrt.rnd{.ftz}.f32`, `sqrt.rnd.f64`
#[derive(Debug, Clone)]
pub enum SqrtInstr {
    /// Fast approximate square root: `sqrt.approx{.ftz}.f32 d, a`
    Approx {
        ftz: bool,
        dst: Operand,
        src: Operand,
    },
    /// IEEE 754 compliant square root: `sqrt.rnd{.ftz}.f32 d, a` or `sqrt.rnd.f64 d, a`
    Ieee {
        rnd: FpRound,
        ftz: bool,
        ty: ScalarType, // f32 or f64
        dst: Operand,
        src: Operand,
    },
}

// --- Comparison and Selection ---

/// Comparison operator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    // Ordered comparisons
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // Unsigned aliases
    Lo,
    Ls,
    Hi,
    Hs,
    // Unordered comparisons (float)
    Equ,
    Neu,
    Ltu,
    Leu,
    Gtu,
    Geu,
    // Special (float)
    Num,
    Nan,
}

impl FromAscii for CmpOp {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"eq" => CmpOp::Eq,
            b"ne" => CmpOp::Ne,
            b"lt" => CmpOp::Lt,
            b"le" => CmpOp::Le,
            b"gt" => CmpOp::Gt,
            b"ge" => CmpOp::Ge,
            b"lo" => CmpOp::Lo,
            b"ls" => CmpOp::Ls,
            b"hi" => CmpOp::Hi,
            b"hs" => CmpOp::Hs,
            b"equ" => CmpOp::Equ,
            b"neu" => CmpOp::Neu,
            b"ltu" => CmpOp::Ltu,
            b"leu" => CmpOp::Leu,
            b"gtu" => CmpOp::Gtu,
            b"geu" => CmpOp::Geu,
            b"num" => CmpOp::Num,
            b"nan" => CmpOp::Nan,
            _ => return None,
        })
    }
}

/// Boolean operator for setp
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    And,
    Or,
    Xor,
}

impl FromAscii for BoolOp {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"and" => BoolOp::And,
            b"or" => BoolOp::Or,
            b"xor" => BoolOp::Xor,
            _ => return None,
        })
    }
}

/// Set instruction with mutually exclusive forms
///
/// Simple form (Block 66):
/// - `set.CmpOp{.ftz}.dtype.stype d, a, b`
///
/// With boolean operation (Block 66):
/// - `set.CmpOp.BoolOp{.ftz}.dtype.stype d, a, b, {!}c`
#[derive(Debug, Clone)]
pub enum SetInstr {
    /// Simple set: `set.CmpOp{.ftz}.dtype.stype d, a, b`
    Simple {
        cmp_op: CmpOp,
        ftz: bool,
        dst_type: ScalarType,
        src_type: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
    /// Set with boolean operation: `set.CmpOp.BoolOp{.ftz}.dtype.stype d, a, b, {!}c`
    WithBoolOp {
        cmp_op: CmpOp,
        bool_op: BoolOp,
        ftz: bool,
        dst_type: ScalarType,
        src_type: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
        src_c: Operand,
    },
}

/// Setp instruction with mutually exclusive forms
///
/// Simple form (Block 67):
/// - `setp.CmpOp{.ftz}.type p[|q], a, b`
///
/// With boolean operation (Block 67):
/// - `setp.CmpOp.BoolOp{.ftz}.type p[|q], a, b, {!}c`
#[derive(Debug, Clone)]
pub enum SetpInstr {
    /// Simple setp: `setp.CmpOp{.ftz}.type p[|q], a, b`
    Simple {
        cmp_op: CmpOp,
        ftz: bool,
        ty: ScalarType,
        dst_p: Operand,
        dst_q: Option<Operand>,
        src_a: Operand,
        src_b: Operand,
    },
    /// Setp with boolean operation: `setp.CmpOp.BoolOp{.ftz}.type p[|q], a, b, {!}c`
    WithBoolOp {
        cmp_op: CmpOp,
        bool_op: BoolOp,
        ftz: bool,
        ty: ScalarType,
        dst_p: Operand,
        dst_q: Option<Operand>,
        src_a: Operand,
        src_b: Operand,
        src_c: Operand,
    },
}

/// Selp: `selp.type d, a, b, c`
#[derive(Debug, Clone)]
pub struct SelpInstr {
    pub ty: ScalarType,
    pub dst: Operand,
    pub src_a: Operand,
    pub src_b: Operand,
    pub src_c: Operand,
}

/// Slct instruction with type-specific forms
///
/// Integer form (Block 69):
/// - `slct.dtype.s32 d, a, b, c` (no ftz)
///
/// Float form (Block 69):
/// - `slct{.ftz}.dtype.f32 d, a, b, c`
#[derive(Debug, Clone)]
pub enum SlctInstr {
    /// Integer slct: `slct.dtype.s32 d, a, b, c` (no ftz)
    Integer {
        dst_type: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
        src_c: Operand,
    },
    /// Float slct: `slct{.ftz}.dtype.f32 d, a, b, c`
    Float {
        ftz: bool,
        dst_type: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
        src_c: Operand,
    },
}

// --- Logic and Shift ---

/// Logic instruction: `and/or/xor.type d, a, b`
#[derive(Debug, Clone)]
pub struct LogicInstr {
    pub ty: ScalarType,
    pub dst: Operand,
    pub src_a: Operand,
    pub src_b: Operand,
}

/// Not: `not.type d, a`
#[derive(Debug, Clone)]
pub struct NotInstr {
    pub ty: ScalarType,
    pub dst: Operand,
    pub src: Operand,
}

/// Shift: `shl/shr.type d, a, b`
#[derive(Debug, Clone)]
pub struct ShiftInstr {
    pub ty: ScalarType,
    pub dst: Operand,
    pub src_a: Operand,
    pub src_b: Operand,
}

// --- Data Movement ---

/// Mov: `mov.type d, a`
#[derive(Debug, Clone)]
pub struct MovInstr {
    pub ty: ScalarType,
    pub dst: Operand,
    pub src: Operand,
}

/// Memory semantics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemSemantics {
    #[default]
    Weak,
    Volatile,
    Relaxed,
    Acquire,
    Release,
}

impl FromAscii for MemSemantics {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"weak" => MemSemantics::Weak,
            b"volatile" => MemSemantics::Volatile,
            b"relaxed" => MemSemantics::Relaxed,
            b"acquire" => MemSemantics::Acquire,
            b"release" => MemSemantics::Release,
            _ => return None,
        })
    }
}

/// Memory scope
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemScope {
    Cta,
    Cluster,
    Gpu,
    Sys,
}

impl FromAscii for MemScope {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"cta" => MemScope::Cta,
            b"cluster" => MemScope::Cluster,
            b"gpu" => MemScope::Gpu,
            b"sys" => MemScope::Sys,
            _ => return None,
        })
    }
}

/// Cache operation for load instructions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheOp {
    Ca, // Cache at all levels (load)
    Cg, // Cache at global level (load/store)
    Cs, // Cache streaming (load/store)
    Lu, // Last use (load)
    Cv, // Don't cache, volatile (load)
    Wb, // Write-back (store)
    Wt, // Write-through (store)
}

impl FromAscii for CacheOp {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"ca" => CacheOp::Ca,
            b"cg" => CacheOp::Cg,
            b"cs" => CacheOp::Cs,
            b"lu" => CacheOp::Lu,
            b"cv" => CacheOp::Cv,
            b"wb" => CacheOp::Wb,
            b"wt" => CacheOp::Wt,
            _ => return None,
        })
    }
}

/// Load instruction: `ld{.sem}{.scope}{.space}{.cop}{.vec}.type d, [a]`
/// (Blocks 86-87). Cache performance-hint qualifiers (eviction priority,
/// prefetch size) are consumed and dropped at parse time.
#[derive(Debug, Clone)]
pub struct LdInstr {
    pub semantics: MemSemantics,
    pub scope: Option<MemScope>,
    pub space: Option<StateSpace>,
    pub space_qualifier: Option<StateSpaceQualifier>,
    pub cache_op: Option<CacheOp>,
    pub vec: Option<VecWidth>,
    /// Non-coherent load (ld.global.nc)
    pub nc: bool,
    /// MMIO load (ld.mmio)
    pub mmio: bool,
    /// Unified addressing hint
    pub unified: bool,
    pub ty: ScalarType,
    pub dst: Operand,
    pub addr: Operand,
}

/// Store instruction: `st{.sem}{.scope}{.space}{.cop}{.vec}.type [a], b`
/// (Block 89). Cache performance-hint qualifiers (eviction priority,
/// prefetch size) are consumed and dropped at parse time.
#[derive(Debug, Clone)]
pub struct StInstr {
    pub semantics: MemSemantics,
    pub scope: Option<MemScope>,
    pub space: Option<StateSpace>,
    pub space_qualifier: Option<StateSpaceQualifier>,
    pub cache_op: Option<CacheOp>,
    pub vec: Option<VecWidth>,
    /// MMIO store (st.mmio)
    pub mmio: bool,
    pub ty: ScalarType,
    pub addr: Operand,
    pub src: Operand,
}

/// Rounding mode for cvt instruction (mutually exclusive)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CvtRounding {
    /// Integer rounding: rni, rzi, rmi, rpi
    Integer(IntRound),
    /// FP rounding: rn, rz, rm, rp
    Float(FpRound),
    /// Stochastic rounding: rs
    Stochastic,
    /// Round to nearest away: rna (for tf32)
    Rna,
}

/// Cvt instruction with mutually exclusive forms:
/// - Block 99: Standard conversions with various rounding modes
/// - Block 100: Pack conversions `cvt.pack.sat.type d, a, b`
#[derive(Debug, Clone)]
pub enum CvtInstr {
    /// Standard conversion: `cvt{.rnd}{.ftz}{.sat}{.relu}{.satfinite}.dtype.atype d, a`
    Standard {
        rnd: Option<CvtRounding>,
        ftz: bool,
        sat: bool,
        relu: bool,
        satfinite: bool,
        dst_type: ScalarType,
        src_type: ScalarType,
        dst: Operand,
        src: Operand,
    },
    /// Pack conversion: `cvt.pack.sat.dtype.atype d, a, b` (Block 100)
    Pack {
        sat: bool,
        dst_type: ScalarType,
        src_type: ScalarType,
        dst: Operand,
        src_a: Operand,
        src_b: Operand,
    },
}

/// Cvta: `cvta{.to}.space.size d, a`
#[derive(Debug, Clone)]
pub struct CvtaInstr {
    pub to_generic: bool,
    pub space: StateSpace,
    pub ty: ScalarType,
    pub dst: Operand,
    pub src: Operand,
}

// --- Control Flow ---

/// Branch: `bra{.uni} target`
#[derive(Debug, Clone)]
pub struct BraInstr {
    pub uniform: bool,
    pub target: Operand,
}

/// Call: `call{.uni} (ret), func, (args)`
#[derive(Debug, Clone)]
pub struct CallInstr {
    pub uniform: bool,
    pub return_operands: Vec<Operand>,
    pub target: Operand,
    pub arguments: Vec<Operand>,
}

// --- Synchronization ---

/// Bar: `bar.sync/bar.arrive/bar.red`
#[derive(Debug, Clone)]
pub struct BarInstr {
    pub mode: BarMode,
    pub operands: Vec<Operand>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarMode {
    Sync,
    Arrive,
    Red,
}

/// Membar: `membar.level`
#[derive(Debug, Clone)]
pub struct MembarInstr {
    pub level: MemScope,
}

/// Atomic operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomOp {
    And,
    Or,
    Xor,
    Cas,
    Exch,
    Add,
    Inc,
    Dec,
    Min,
    Max,
}

impl FromAscii for AtomOp {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"and" => AtomOp::And,
            b"or" => AtomOp::Or,
            b"xor" => AtomOp::Xor,
            b"cas" => AtomOp::Cas,
            b"exch" => AtomOp::Exch,
            b"add" => AtomOp::Add,
            b"inc" => AtomOp::Inc,
            b"dec" => AtomOp::Dec,
            b"min" => AtomOp::Min,
            b"max" => AtomOp::Max,
            _ => return None,
        })
    }
}

/// Atom: `atom{.sem}{.scope}{.space}.op.type d, [a], b{, c}`
#[derive(Debug, Clone)]
pub struct AtomInstr {
    pub semantics: MemSemantics,
    pub scope: Option<MemScope>,
    pub space: Option<StateSpace>,
    pub op: AtomOp,
    pub ty: ScalarType,
    pub dst: Operand,
    pub addr: Operand,
    pub src_b: Operand,
    pub src_c: Option<Operand>,
}

// =============================================================================
// Additional Integer Arithmetic Instructions
// =============================================================================

/// mul24: `mul24.mode.type d, a, b` (Block 5)
#[derive(Debug, Clone)]
pub struct Mul24Instr {
    pub mode: MulMode,
    pub ty: ScalarType,
    pub dst: Operand,
    pub src_a: Operand,
    pub src_b: Operand,
}

/// mad24: `mad24.mode.type d, a, b, c` (Block 6)
#[derive(Debug, Clone)]
pub struct Mad24Instr {
    pub mode: MulMode,
    pub sat: bool,
    pub ty: ScalarType,
    pub dst: Operand,
    pub src_a: Operand,
    pub src_b: Operand,
    pub src_c: Operand,
}

/// sad: `sad.type d, a, b, c` (Block 7)
#[derive(Debug, Clone)]
pub struct SadInstr {
    pub ty: ScalarType,
    pub dst: Operand,
    pub src_a: Operand,
    pub src_b: Operand,
    pub src_c: Operand,
}

// =============================================================================
// Bit Manipulation Instructions
// =============================================================================

/// popc: `popc.type d, a` (Block 14)
#[derive(Debug, Clone)]
pub struct PopcInstr {
    pub ty: ScalarType,
    pub dst: Operand,
    pub src: Operand,
}

/// clz: `clz.type d, a` (Block 15)
#[derive(Debug, Clone)]
pub struct ClzInstr {
    pub ty: ScalarType,
    pub dst: Operand,
    pub src: Operand,
}

/// bfind: `bfind{.shiftamt}.type d, a` (Block 16)
#[derive(Debug, Clone)]
pub struct BfindInstr {
    pub shiftamt: bool,
    pub ty: ScalarType,
    pub dst: Operand,
    pub src: Operand,
}

/// fns: `fns.b32 d, mask, base, offset` (Block 17)
#[derive(Debug, Clone)]
pub struct FnsInstr {
    pub dst: Operand,
    pub mask: Operand,
    pub base: Operand,
    pub offset: Operand,
}

/// brev: `brev.type d, a` (Block 18)
#[derive(Debug, Clone)]
pub struct BrevInstr {
    pub ty: ScalarType,
    pub dst: Operand,
    pub src: Operand,
}

/// bfe: `bfe.type d, a, b, c` (Block 19)
#[derive(Debug, Clone)]
pub struct BfeInstr {
    pub ty: ScalarType,
    pub dst: Operand,
    pub src_a: Operand,
    pub start: Operand,
    pub len: Operand,
}

/// bfi: `bfi.type f, a, b, c, d` (Block 20)
#[derive(Debug, Clone)]
pub struct BfiInstr {
    pub ty: ScalarType,
    pub dst: Operand,
    pub src_a: Operand,
    pub src_b: Operand,
    pub start: Operand,
    pub len: Operand,
}

/// Clamp/wrap mode for szext and bmsk
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClampWrapMode {
    Clamp,
    Wrap,
}

impl FromAscii for ClampWrapMode {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"clamp" => ClampWrapMode::Clamp,
            b"wrap" => ClampWrapMode::Wrap,
            _ => return None,
        })
    }
}

/// szext: `szext.mode.type d, a, b` (Block 21)
#[derive(Debug, Clone)]
pub struct SzextInstr {
    pub mode: ClampWrapMode,
    pub ty: ScalarType,
    pub dst: Operand,
    pub src: Operand,
    pub pos: Operand,
}

/// bmsk: `bmsk.mode.b32 d, a, b` (Block 22)
#[derive(Debug, Clone)]
pub struct BmskInstr {
    pub mode: ClampWrapMode,
    pub dst: Operand,
    pub start: Operand,
    pub len: Operand,
}

/// dp4a: `dp4a.atype.btype d, a, b, c` (Block 23)
#[derive(Debug, Clone)]
pub struct Dp4aInstr {
    pub atype: ScalarType,
    pub btype: ScalarType,
    pub dst: Operand,
    pub src_a: Operand,
    pub src_b: Operand,
    pub src_c: Operand,
}

/// dp2a: `dp2a.mode.atype.btype d, a, b, c` (Block 24)
#[derive(Debug, Clone)]
pub struct Dp2aInstr {
    pub mode: MulMode,
    pub atype: ScalarType,
    pub btype: ScalarType,
    pub dst: Operand,
    pub src_a: Operand,
    pub src_b: Operand,
    pub src_c: Operand,
}

// =============================================================================
// Extended-Precision Integer Arithmetic
// =============================================================================

/// add.cc: `add.cc.type d, a, b` (Block 25)
#[derive(Debug, Clone)]
pub struct AddCcInstr {
    pub ty: ScalarType,
    pub dst: Operand,
    pub src_a: Operand,
    pub src_b: Operand,
}

/// addc: `addc{.cc}.type d, a, b` (Block 26)
#[derive(Debug, Clone)]
pub struct AddcInstr {
    pub cc: bool,
    pub ty: ScalarType,
    pub dst: Operand,
    pub src_a: Operand,
    pub src_b: Operand,
}

/// sub.cc: `sub.cc.type d, a, b` (Block 27)
#[derive(Debug, Clone)]
pub struct SubCcInstr {
    pub ty: ScalarType,
    pub dst: Operand,
    pub src_a: Operand,
    pub src_b: Operand,
}

/// subc: `subc{.cc}.type d, a, b` (Block 28)
#[derive(Debug, Clone)]
pub struct SubcInstr {
    pub cc: bool,
    pub ty: ScalarType,
    pub dst: Operand,
    pub src_a: Operand,
    pub src_b: Operand,
}

/// mad.cc: `mad{.hi,.lo}.cc.type d, a, b, c` (Block 29)
#[derive(Debug, Clone)]
pub struct MadCcInstr {
    pub mode: Option<MulMode>,
    pub ty: ScalarType,
    pub dst: Operand,
    pub src_a: Operand,
    pub src_b: Operand,
    pub src_c: Operand,
}

/// madc: `madc{.hi,.lo}{.cc}.type d, a, b, c` (Block 30)
#[derive(Debug, Clone)]
pub struct MadcInstr {
    pub mode: Option<MulMode>,
    pub cc: bool,
    pub ty: ScalarType,
    pub dst: Operand,
    pub src_a: Operand,
    pub src_b: Operand,
    pub src_c: Operand,
}

// =============================================================================
// Additional Floating-Point Instructions
// =============================================================================

/// Testp operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestpOp {
    Finite,
    Infinite,
    Number,
    NotANumber,
    Normal,
    Subnormal,
}

impl FromAscii for TestpOp {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"finite" => TestpOp::Finite,
            b"infinite" => TestpOp::Infinite,
            b"number" => TestpOp::Number,
            b"notanumber" => TestpOp::NotANumber,
            b"normal" => TestpOp::Normal,
            b"subnormal" => TestpOp::Subnormal,
            _ => return None,
        })
    }
}

/// testp: `testp.op.type p, a` (Block 31)
#[derive(Debug, Clone)]
pub struct TestpInstr {
    pub op: TestpOp,
    pub ty: ScalarType,
    pub dst: Operand,
    pub src: Operand,
}

/// copysign: `copysign.type d, a, b` (Block 32)
#[derive(Debug, Clone)]
pub struct CopysignInstr {
    pub ty: ScalarType,
    pub dst: Operand,
    pub magnitude: Operand,
    pub sign: Operand,
}

/// rsqrt: `rsqrt.approx{.ftz}.type d, a` (Block 46)
#[derive(Debug, Clone)]
pub struct RsqrtInstr {
    pub approx: bool,
    pub ftz: bool,
    pub ty: ScalarType,
    pub dst: Operand,
    pub src: Operand,
}

/// sin: `sin.approx{.ftz}.f32 d, a` (Block 48)
#[derive(Debug, Clone)]
pub struct SinInstr {
    pub ftz: bool,
    pub dst: Operand,
    pub src: Operand,
}

/// cos: `cos.approx{.ftz}.f32 d, a` (Block 49)
#[derive(Debug, Clone)]
pub struct CosInstr {
    pub ftz: bool,
    pub dst: Operand,
    pub src: Operand,
}

/// lg2: `lg2.approx{.ftz}.f32 d, a` (Block 50)
#[derive(Debug, Clone)]
pub struct Lg2Instr {
    pub ftz: bool,
    pub dst: Operand,
    pub src: Operand,
}

/// ex2 instruction with type-specific ftz rules
///
/// Float32 (Block 51):
/// - `ex2.approx{.ftz}.f32 d, a` - ftz optional
///
/// Half (Block 62):
/// - `ex2.approx.atype d, a` where atype = f16, f16x2 (NO ftz)
/// - `ex2.approx.ftz.btype d, a` where btype = bf16, bf16x2 (ftz REQUIRED)
#[derive(Debug, Clone)]
pub enum Ex2Instr {
    /// Float32: `ex2.approx{.ftz}.f32 d, a` - ftz optional
    Float32 {
        ftz: bool,
        dst: Operand,
        src: Operand,
    },
    /// Half f16/f16x2: `ex2.approx.type d, a` - NO ftz allowed
    HalfF16 {
        ty: ScalarType,
        dst: Operand,
        src: Operand,
    },
    /// Half bf16/bf16x2: `ex2.approx.ftz.type d, a` - ftz REQUIRED (implicit)
    HalfBf16 {
        ty: ScalarType,
        dst: Operand,
        src: Operand,
    },
}

/// tanh: `tanh.approx.type d, a` (Blocks 52, 61)
#[derive(Debug, Clone)]
pub struct TanhInstr {
    pub ty: ScalarType,
    pub dst: Operand,
    pub src: Operand,
}

// =============================================================================
// Logic and Shift Instructions
// =============================================================================

/// cnot: `cnot.type d, a` (Block 76)
#[derive(Debug, Clone)]
pub struct CnotInstr {
    pub ty: ScalarType,
    pub dst: Operand,
    pub src: Operand,
}

/// lop3: `lop3.b32 d, a, b, c, immLut` (Block 77)
#[derive(Debug, Clone)]
pub struct Lop3Instr {
    pub bool_op: Option<BoolOp>,
    pub dst: Operand,
    pub dst_pred: Option<Operand>,
    pub src_a: Operand,
    pub src_b: Operand,
    pub src_c: Operand,
    pub lut: Operand,
    pub pred_q: Option<Operand>,
}

/// Shift direction for shf
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftDir {
    Left,
    Right,
}

impl FromAscii for ShiftDir {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"l" => ShiftDir::Left,
            b"r" => ShiftDir::Right,
            _ => return None,
        })
    }
}

/// shf: `shf.dir.mode.b32 d, a, b, c` (Block 78)
#[derive(Debug, Clone)]
pub struct ShfInstr {
    pub dir: ShiftDir,
    pub mode: ClampWrapMode,
    pub dst: Operand,
    pub lo: Operand,
    pub hi: Operand,
    pub shift: Operand,
}

// =============================================================================
// Data Movement Instructions
// =============================================================================

/// Shuffle mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShflMode {
    Up,
    Down,
    Bfly,
    Idx,
}

impl FromAscii for ShflMode {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"up" => ShflMode::Up,
            b"down" => ShflMode::Down,
            b"bfly" => ShflMode::Bfly,
            b"idx" => ShflMode::Idx,
            _ => return None,
        })
    }
}

/// shfl: `shfl.mode.b32 d[|p], a, b, c` (Block 83, deprecated)
#[derive(Debug, Clone)]
pub struct ShflInstr {
    pub mode: ShflMode,
    pub dst: Operand,
    pub dst_pred: Option<Operand>,
    pub src: Operand,
    pub src_b: Operand,
    pub src_c: Operand,
}

/// shfl.sync: `shfl.sync.mode.b32 d[|p], a, b, c, membermask` (Block 84)
#[derive(Debug, Clone)]
pub struct ShflSyncInstr {
    pub mode: ShflMode,
    pub dst: Operand,
    pub dst_pred: Option<Operand>,
    pub src: Operand,
    pub src_b: Operand,
    pub src_c: Operand,
    pub membermask: Operand,
}

/// Prmt mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrmtMode {
    None,
    F4e,
    B4e,
    Rc8,
    Ecl,
    Ecr,
    Rc16,
}

impl FromAscii for PrmtMode {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"f4e" => PrmtMode::F4e,
            b"b4e" => PrmtMode::B4e,
            b"rc8" => PrmtMode::Rc8,
            b"ecl" => PrmtMode::Ecl,
            b"ecr" => PrmtMode::Ecr,
            b"rc16" => PrmtMode::Rc16,
            _ => return None,
        })
    }
}

/// prmt: `prmt.b32{.mode} d, a, b, c` (Block 85)
#[derive(Debug, Clone)]
pub struct PrmtInstr {
    pub mode: PrmtMode,
    pub dst: Operand,
    pub src_a: Operand,
    pub src_b: Operand,
    pub selector: Operand,
}

/// ldu: `ldu{.ss}{.vec}.type d, [a]` (Block 88)
#[derive(Debug, Clone)]
pub struct LduInstr {
    pub space: Option<StateSpace>,
    pub vec: Option<VecWidth>,
    pub ty: ScalarType,
    pub dst: Operand,
    pub addr: Operand,
}

/// Prefetch level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchLevel {
    L1,
    L2,
}

impl FromAscii for PrefetchLevel {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"L1" => PrefetchLevel::L1,
            b"L2" => PrefetchLevel::L2,
            _ => return None,
        })
    }
}

/// prefetch: `prefetch{.space}.level [a]` (Block 93)
#[derive(Debug, Clone)]
pub struct PrefetchInstr {
    pub space: Option<StateSpace>,
    pub level: PrefetchLevel,
    pub addr: Operand,
}

/// isspacep: `isspacep.space p, a` (Block 97)
#[derive(Debug, Clone)]
pub struct IsspacepInstr {
    pub space: StateSpace,
    pub space_qualifier: Option<StateSpaceQualifier>,
    pub dst: Operand,
    pub src: Operand,
}

/// mapa: `mapa{.space}.type d, a, b` (Block 101)
#[derive(Debug, Clone)]
pub struct MapaInstr {
    pub space: Option<StateSpace>,
    pub space_qualifier: Option<StateSpaceQualifier>,
    pub ty: ScalarType,
    pub dst: Operand,
    pub src: Operand,
    pub cta: Operand,
}

/// getctarank: `getctarank{.space}.type d, a` (Block 102)
#[derive(Debug, Clone)]
pub struct GetctarankInstr {
    pub space: Option<StateSpace>,
    pub space_qualifier: Option<StateSpaceQualifier>,
    pub ty: ScalarType,
    pub dst: Operand,
    pub src: Operand,
}

// =============================================================================
// Control Flow Instructions
// =============================================================================

/// brx.idx: `brx.idx{.uni} index, tlist` (Block 128)
#[derive(Debug, Clone)]
pub struct BrxIdxInstr {
    pub uniform: bool,
    pub index: Operand,
    pub target_list: Operand,
}

/// ret: `ret{.uni}` (Block 130)
#[derive(Debug, Clone)]
pub struct RetInstr {
    pub uniform: bool,
}

// =============================================================================
// Synchronization Instructions
// =============================================================================

/// barrier: `barrier{.cta}.sync{.aligned} a{, b}` (Block 132)
#[derive(Debug, Clone)]
pub struct BarrierInstr {
    pub cta: bool,
    pub mode: BarMode,
    pub aligned: bool,
    pub operands: Vec<Operand>,
}

/// bar.warp.sync: `bar.warp.sync membermask` (Block 133)
#[derive(Debug, Clone)]
pub struct BarWarpSyncInstr {
    pub membermask: Operand,
}

/// barrier.cluster: `barrier.cluster.arrive{.sem}{.aligned}` (Block 134)
#[derive(Debug, Clone)]
pub struct BarrierClusterInstr {
    pub arrive: bool,
    pub sem: Option<MemSemantics>,
    pub aligned: bool,
}

/// Fence semantic
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FenceSem {
    Sc,
    AcqRel,
    Acquire,
    Release,
}

impl FromAscii for FenceSem {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"sc" => FenceSem::Sc,
            b"acq_rel" => FenceSem::AcqRel,
            b"acquire" => FenceSem::Acquire,
            b"release" => FenceSem::Release,
            _ => return None,
        })
    }
}

/// fence: `fence{.sem}.scope` (Block 135)
#[derive(Debug, Clone)]
pub struct FenceInstr {
    pub sem: Option<FenceSem>,
    pub scope: Option<MemScope>,
    pub proxy: bool,
}

/// Reduction operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedOp {
    And,
    Or,
    Xor,
    Add,
    Inc,
    Dec,
    Min,
    Max,
}

impl FromAscii for RedOp {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"and" => RedOp::And,
            b"or" => RedOp::Or,
            b"xor" => RedOp::Xor,
            b"add" => RedOp::Add,
            b"inc" => RedOp::Inc,
            b"dec" => RedOp::Dec,
            b"min" => RedOp::Min,
            b"max" => RedOp::Max,
            _ => return None,
        })
    }
}

/// red: `red{.sem}{.scope}{.space}.op.type [a], b` (Block 137)
#[derive(Debug, Clone)]
pub struct RedInstr {
    pub semantics: MemSemantics,
    pub scope: Option<MemScope>,
    pub space: Option<StateSpace>,
    pub op: RedOp,
    pub ty: ScalarType,
    pub addr: Operand,
    pub src: Operand,
}

/// Vote mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoteMode {
    All,
    Any,
    Uni,
    Ballot,
}

impl FromAscii for VoteMode {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"all" => VoteMode::All,
            b"any" => VoteMode::Any,
            b"uni" => VoteMode::Uni,
            b"ballot" => VoteMode::Ballot,
            _ => return None,
        })
    }
}

/// vote: `vote.mode.pred d, {!}a` (Block 139, deprecated)
#[derive(Debug, Clone)]
pub struct VoteInstr {
    pub mode: VoteMode,
    pub dst: Operand,
    pub src: Operand,
}

/// vote.sync: `vote.sync.mode.pred d, {!}a, membermask` (Block 140)
#[derive(Debug, Clone)]
pub struct VoteSyncInstr {
    pub mode: VoteMode,
    pub dst: Operand,
    pub src: Operand,
    pub membermask: Operand,
}

/// Match mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchMode {
    Any,
    All,
}

impl FromAscii for MatchMode {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"any" => MatchMode::Any,
            b"all" => MatchMode::All,
            _ => return None,
        })
    }
}

/// match.sync: `match.mode.sync.type d[|p], a, membermask` (Block 141)
#[derive(Debug, Clone)]
pub struct MatchSyncInstr {
    pub mode: MatchMode,
    pub ty: ScalarType,
    pub dst: Operand,
    pub dst_pred: Option<Operand>,
    pub src: Operand,
    pub membermask: Operand,
}

/// activemask: `activemask.b32 d` (Block 142)
#[derive(Debug, Clone)]
pub struct ActivemaskInstr {
    pub dst: Operand,
}

/// redux.sync: `redux.sync.op.type dst, src, membermask` (Block 143)
#[derive(Debug, Clone)]
pub struct ReduxSyncInstr {
    pub op: RedOp,
    pub ty: ScalarType,
    pub abs: bool,
    pub nan: bool,
    pub dst: Operand,
    pub src: Operand,
    pub membermask: Operand,
}

/// Griddepcontrol action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GriddepcontrolAction {
    LaunchDependents,
    Wait,
}

impl FromAscii for GriddepcontrolAction {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"launch_dependents" => GriddepcontrolAction::LaunchDependents,
            b"wait" => GriddepcontrolAction::Wait,
            _ => return None,
        })
    }
}

/// griddepcontrol: `griddepcontrol.action` (Block 144)
#[derive(Debug, Clone)]
pub struct GriddepcontrolInstr {
    pub action: GriddepcontrolAction,
}

/// elect.sync: `elect.sync d|p, membermask` (Block 145)
#[derive(Debug, Clone)]
pub struct ElectSyncInstr {
    pub dst: Operand,
    pub dst_pred: Operand,
    pub membermask: Operand,
}

// =============================================================================
// Mbarrier Instructions
// =============================================================================

/// mbarrier.init: `mbarrier.init{.shared{::cta}}.b64 [addr], count` (Block 146)
#[derive(Debug, Clone)]
pub struct MbarrierInitInstr {
    pub space: Option<StateSpace>,
    pub space_qualifier: Option<StateSpaceQualifier>,
    pub addr: Operand,
    pub count: Operand,
}

/// mbarrier.inval: `mbarrier.inval{.shared{::cta}}.b64 [addr]` (Block 147)
#[derive(Debug, Clone)]
pub struct MbarrierInvalInstr {
    pub space: Option<StateSpace>,
    pub space_qualifier: Option<StateSpaceQualifier>,
    pub addr: Operand,
}

/// mbarrier.expect_tx: `mbarrier.expect_tx{.sem.scope}{.space}.b64 [addr], txCount` (Block 148)
#[derive(Debug, Clone)]
pub struct MbarrierExpectTxInstr {
    pub sem: Option<MemSemantics>,
    pub scope: Option<MemScope>,
    pub space: Option<StateSpace>,
    pub space_qualifier: Option<StateSpaceQualifier>,
    pub addr: Operand,
    pub tx_count: Operand,
}

/// mbarrier.complete_tx: `mbarrier.complete_tx{.sem.scope}{.space}.b64 [addr], txCount` (Block 149)
#[derive(Debug, Clone)]
pub struct MbarrierCompleteTxInstr {
    pub sem: Option<MemSemantics>,
    pub scope: Option<MemScope>,
    pub space: Option<StateSpace>,
    pub space_qualifier: Option<StateSpaceQualifier>,
    pub addr: Operand,
    pub tx_count: Operand,
}

/// mbarrier.arrive: `mbarrier.arrive{.sem.scope}{.shared{::cta}}.b64 state, [addr]{, count}` (Block 150)
#[derive(Debug, Clone)]
pub struct MbarrierArriveInstr {
    pub sem: Option<MemSemantics>,
    pub scope: Option<MemScope>,
    pub space: Option<StateSpace>,
    pub space_qualifier: Option<StateSpaceQualifier>,
    pub expect_tx: bool,
    pub no_complete: bool,
    pub state: Operand,
    pub addr: Operand,
    pub count: Option<Operand>,
}

/// mbarrier.arrive_drop: similar to mbarrier.arrive (Block 151)
#[derive(Debug, Clone)]
pub struct MbarrierArriveDropInstr {
    pub sem: Option<MemSemantics>,
    pub scope: Option<MemScope>,
    pub space: Option<StateSpace>,
    pub space_qualifier: Option<StateSpaceQualifier>,
    pub expect_tx: bool,
    pub no_complete: bool,
    pub state: Operand,
    pub addr: Operand,
    pub count: Option<Operand>,
}

/// mbarrier.test_wait: `mbarrier.test_wait{.sem.scope}{.shared{::cta}}.b64 waitComplete, [addr], state` (Block 153)
#[derive(Debug, Clone)]
pub struct MbarrierTestWaitInstr {
    pub parity: bool,
    pub sem: Option<MemSemantics>,
    pub scope: Option<MemScope>,
    pub space: Option<StateSpace>,
    pub space_qualifier: Option<StateSpaceQualifier>,
    pub wait_complete: Operand,
    pub addr: Operand,
    pub state_or_parity: Operand,
}

/// mbarrier.try_wait: similar to test_wait but with optional suspend hint (Block 153)
#[derive(Debug, Clone)]
pub struct MbarrierTryWaitInstr {
    pub parity: bool,
    pub sem: Option<MemSemantics>,
    pub scope: Option<MemScope>,
    pub space: Option<StateSpace>,
    pub space_qualifier: Option<StateSpaceQualifier>,
    pub wait_complete: Operand,
    pub addr: Operand,
    pub state_or_parity: Operand,
    pub suspend_hint: Option<Operand>,
}

/// mbarrier.pending_count: `mbarrier.pending_count.b64 count, state` (Block 154)
#[derive(Debug, Clone)]
pub struct MbarrierPendingCountInstr {
    pub count: Operand,
    pub state: Operand,
}

// =============================================================================
// Stack Instructions
// =============================================================================

/// stacksave: `stacksave.type d` (Block 183)
#[derive(Debug, Clone)]
pub struct StacksaveInstr {
    pub ty: ScalarType,
    pub dst: Operand,
}

/// stackrestore: `stackrestore.type a` (Block 184)
#[derive(Debug, Clone)]
pub struct StackrestoreInstr {
    pub ty: ScalarType,
    pub src: Operand,
}

/// alloca: `alloca.type ptr, size{, immAlign}` (Block 185)
#[derive(Debug, Clone)]
pub struct AllocaInstr {
    pub ty: ScalarType,
    pub ptr: Operand,
    pub size: Operand,
    pub align: Option<Operand>,
}

// =============================================================================
// Miscellaneous Instructions
// =============================================================================

/// nanosleep: `nanosleep.u32 t` (Block 195)
#[derive(Debug, Clone)]
pub struct NanosleepInstr {
    pub duration: Operand,
}

/// pmevent: `pmevent a` or `pmevent.mask a` (Block 196)
#[derive(Debug, Clone)]
pub struct PmeventInstr {
    pub mask: bool,
    pub event: Operand,
}

/// Setmaxnreg action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetmaxnregAction {
    Inc,
    Dec,
}

impl FromAscii for SetmaxnregAction {
    fn from_ascii(s: &[AsciiChar]) -> Option<Self> {
        Some(match s.as_bytes() {
            b"inc" => SetmaxnregAction::Inc,
            b"dec" => SetmaxnregAction::Dec,
            _ => return None,
        })
    }
}

/// setmaxnreg: `setmaxnreg.action.sync.aligned.u32 imm-reg-count` (Block 198)
#[derive(Debug, Clone)]
pub struct SetmaxnregInstr {
    pub action: SetmaxnregAction,
    pub reg_count: Operand,
}

// =============================================================================
// Directives
// =============================================================================

/// File directive for debug info: `.file filenum "filename"`
#[derive(Debug, Clone)]
pub struct FileDirective {
    pub file_num: u32,
    pub filename: AsciiString,
    pub size: Option<u64>,
    pub timestamp: Option<u64>,
}

/// Generic directive
#[derive(Debug, Clone)]
pub struct Directive {
    pub span: Span,
    pub name: AsciiString,
    pub arguments: Vec<crate::lex::Token>,
}
