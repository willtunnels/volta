//! Lowering pass: AST → LoweredProgram
//!
//! This module implements the lowering pass that transforms the parsed AST into
//! a form suitable for abstract interpretation. The lowering:
//!
//! 1. Collects register declarations and builds a symbol table
//! 2. Resolves register references to indices
//! 3. Resolves labels to instruction PCs
//! 4. Performs type checking on operands
//! 5. Converts complex instruction variants to a uniform representation

use std::collections::HashMap;

use volta_common::Span;
use volta_frontend::ascii::AsciiSliceExt;
use volta_frontend::ast::{
    self, AbsInstr, AddInstr, Address, AddressBase, BarMode, BraInstr, CallInstr,
    CmpOp as AstCmpOp, CvtInstr, CvtRounding, DivInstr, FmaInstr, FromAscii, Function,
    FunctionBody, Instruction, InstructionOp, LdInstr, MadInstr, MaxInstr, MemSemantics, MinInstr,
    MulInstr, MulMode, NegInstr, Operand as AstOperand, ParsedInstruction, ScalarType, SetpInstr,
    ShflMode as AstShflMode, ShflSyncInstr, StInstr, StateSpace, Statement, SubInstr, VarDecl,
    VecWidth,
};
use volta_frontend::instr::InstrKind;
use volta_frontend::instr_parse::{is_cache_perf_hint, parse_instruction};
use volta_frontend::lex::DottedIdent;

use id_collections::{Id, IdVec};

use crate::lower_error::{LowerError, LowerResult};
use crate::lowered::{
    BinOp, Clamp, CmpOp, InstrId, LoweredInstr, LoweredProgram, MemSpace, MembarScope,
    MulMode as LoweredMulMode, Operand, Predicate, ShflMode, UnaryOp,
};
use crate::source_map::SourceMapBuilder;
use crate::symbols::{RegId, SpecialRegKind, SymbolTable};
use crate::tensor_core::{MmaLayout, MmaOperand, MmaShape};
use crate::types::{ScalarTypeExt, TypeCompatibility, check_type_compatibility};

// =============================================================================
// Loud Rejection Helpers
// =============================================================================

/// Build the loud rejection for a modifier or instruction form the model
/// does not implement. Policy: lowering must never drop a semantic modifier
/// silently; anything unmodeled errors here, naming the modifier.
fn unsupported(instruction: impl Into<String>, reason: impl Into<String>) -> LowerError {
    LowerError::UnsupportedInstruction {
        instruction: instruction.into(),
        reason: Some(reason.into()),
    }
}

/// Reject packed-SIMD (multi-lane) types in scalar arithmetic lowering.
/// `BinOp`/`UnaryOp`/`Fma` evaluate one lane, so lowering a packed type
/// through them would compute a single 32/64-bit lane (or integer-add float
/// bit patterns) silently.
fn check_not_packed(ty: ScalarType, instruction: &str) -> LowerResult<()> {
    match ty {
        ScalarType::U16x2
        | ScalarType::S16x2
        | ScalarType::F16x2
        | ScalarType::Bf16x2
        | ScalarType::F32x2 => Err(unsupported(
            instruction,
            format!("packed SIMD arithmetic on {:?}", ty),
        )),
        _ => Ok(()),
    }
}

/// Scalar (single-lane) floating-point types - the domain of the modeled
/// float value clamps (`Clamp`). Excludes the packed float types and tf32.
fn is_scalar_float(ty: ScalarType) -> bool {
    matches!(
        ty,
        ScalarType::F16 | ScalarType::Bf16 | ScalarType::F32 | ScalarType::F64
    )
}

// =============================================================================
// Type-Aware Operand Resolution
// =============================================================================

/// An operand resolved with its type information (for type checking)
#[derive(Debug, Clone)]
pub struct ResolvedOperand {
    /// The lowered operand
    pub operand: Operand,
    /// The declared type of the operand (None for immediates, which are polymorphic)
    pub ty: Option<ScalarType>,
    /// The name of the operand (for error messages)
    pub name: Option<String>,
}

impl ResolvedOperand {
    /// Create a resolved operand for a register
    fn register(operand: Operand, ty: ScalarType, name: String) -> Self {
        Self {
            operand,
            ty: Some(ty),
            name: Some(name),
        }
    }

    /// Create a resolved operand for an immediate (type is polymorphic)
    fn immediate(operand: Operand) -> Self {
        Self {
            operand,
            ty: None,
            name: None,
        }
    }

    /// Create a resolved operand for a special register
    fn special_reg(operand: Operand, ty: ScalarType, name: String) -> Self {
        Self {
            operand,
            ty: Some(ty),
            name: Some(name),
        }
    }
}

/// Resolved destination register with type information
#[derive(Debug, Clone)]
pub struct ResolvedDst {
    /// The register ID
    pub reg: RegId,
    /// The declared type
    pub ty: ScalarType,
    /// The register name (for error messages)
    pub name: String,
}

/// State of a block-scope `.param` variable used by the nvcc callseq idiom:
///
/// ```text
/// .param .b32 param0;
/// st.param.f32 [param0+0], %f1;      // -> Stored(%f1)
/// .param .b32 retval0;
/// call.uni (retval0), __symexpf, (param0);   // -> retval0 = PendingExp(%f1)
/// ld.param.f32 %f2, [retval0+0];     // -> emit %f2 = exp(%f1)
/// ```
///
/// No instructions are emitted for the `.param` traffic itself; the call
/// collapses to a single `UnaryOp::Exp` emitted at the consuming `ld.param`.
#[derive(Debug, Clone, Copy)]
enum LocalParamSlot {
    /// Declared but not yet written
    Empty,
    /// Holds the operand last stored via `st.param`
    Stored(Operand),
    /// Holds the pending result of `__symexpf` applied to the operand
    PendingExp(Operand),
}

/// Context for the lowering pass
pub struct LoweringContext {
    /// Symbol table being built
    symbols: SymbolTable,
    /// Source map builder for tracking spans
    source_map_builder: SourceMapBuilder,
    /// Lowered instructions
    instructions: Vec<LoweredInstr>,
    /// Predicates for each instruction
    predicates: Vec<Option<Predicate>>,
    /// Pending labels (label name → will point to next instruction)
    pending_labels: Vec<String>,
    /// Pending label spans (to be associated with the next instruction)
    pending_label_spans: Vec<Span>,
    /// Forward references to resolve (PC, label name)
    forward_refs: Vec<(InstrId, String)>,
    /// Current instruction span (set before lowering each instruction)
    current_span: Option<Span>,
    /// Block-scope `.param` variables (callseq idiom); flat because the
    /// blocks re-declare them before each use
    local_params: HashMap<String, LocalParamSlot>,
}

impl LoweringContext {
    pub fn new() -> Self {
        Self {
            symbols: SymbolTable::new(),
            source_map_builder: SourceMapBuilder::new(),
            instructions: Vec::new(),
            predicates: Vec::new(),
            pending_labels: Vec::new(),
            pending_label_spans: Vec::new(),
            forward_refs: Vec::new(),
            current_span: None,
            local_params: HashMap::new(),
        }
    }

    /// Get current PC (next instruction index)
    fn current_pc(&self) -> InstrId {
        InstrId::from_index(self.instructions.len() as u32)
    }

    /// Emit an instruction
    fn emit(&mut self, instr: LoweredInstr, predicate: Option<Predicate>) -> LowerResult<()> {
        // Resolve pending labels to this PC
        let pc = self.current_pc();
        for label in self.pending_labels.drain(..) {
            self.symbols.declare_label(&label, pc)?;
        }

        // Record instruction span
        self.source_map_builder
            .record_instruction(pc, self.current_span);

        // Record any pending label spans
        for label_span in self.pending_label_spans.drain(..) {
            self.source_map_builder.record_pending_label(label_span);
        }

        self.instructions.push(instr);
        self.predicates.push(predicate);
        Ok(())
    }

    /// Record a label to be resolved to the next instruction
    fn record_label(&mut self, name: &str, span: Option<Span>) {
        self.pending_labels.push(name.to_string());
        if let Some(s) = span {
            self.pending_label_spans.push(s);
        }
    }

    /// Record a forward reference to a label
    fn record_forward_ref(&mut self, label: &str) {
        self.forward_refs
            .push((self.current_pc(), label.to_string()));
    }

    /// Resolve all forward references, patching branch targets.
    fn resolve_forward_refs(&mut self) -> LowerResult<()> {
        for (pc, label) in &self.forward_refs {
            let Some(target) = self.symbols.resolve_label(label) else {
                return Err(LowerError::UndefinedLabel {
                    name: label.clone(),
                });
            };
            if let Some(LoweredInstr::Bra { target: t }) =
                self.instructions.get_mut(pc.to_index() as usize)
            {
                *t = target;
            }
        }
        Ok(())
    }

    /// Resolve a register operand to RegId
    fn resolve_register(&self, name: &str) -> LowerResult<RegId> {
        // Look up register by exact name (including any % prefix)
        self.symbols.resolve_register(name).ok_or_else(|| {
            let suggestions = self.symbols.find_similar_registers(name);
            LowerError::UndefinedRegister {
                name: name.to_string(),
                suggestions,
            }
        })
    }

    /// Resolve a memory-variable symbol (shared, local, or module-global) to
    /// its address operand. Shared and local variables live in their own
    /// per-space address spaces, so their "address" is the space-relative
    /// offset assigned by the symbol table; module globals get absolute
    /// addresses in a reserved region.
    fn resolve_mem_symbol(&self, name: &str) -> Option<Operand> {
        if let Some(info) = self.symbols.get_shared_var(name) {
            return Some(Operand::ImmU64(info.offset));
        }
        if let Some(info) = self.symbols.get_local_var(name) {
            return Some(Operand::ImmU64(info.offset));
        }
        if let Some(info) = self.symbols.get_global_var(name) {
            return Some(Operand::ImmU64(info.addr));
        }
        None
    }

    /// Resolve an operand (register, immediate, or special register)
    fn resolve_operand(&self, op: &AstOperand) -> LowerResult<Operand> {
        match op {
            AstOperand::Ident(name) => {
                let name_str = name.to_string();
                // PTX predefined constant: warp size
                if name_str == "WARP_SZ" {
                    return Ok(Operand::ImmI64(32));
                }
                // Check for special register first (e.g., %tid.x, %ntid.x)
                if let Some(kind) = SpecialRegKind::from_name(&name_str) {
                    return Ok(Operand::SpecialReg(kind));
                }
                // Check if it's a declared register
                if let Some(reg_id) = self.symbols.resolve_register(&name_str) {
                    return Ok(Operand::Reg(reg_id));
                }
                // Check if it's a shared/local/global memory symbol
                if let Some(op) = self.resolve_mem_symbol(&name_str) {
                    return Ok(op);
                }
                // Not found - return error with suggestions
                let suggestions = self.symbols.find_similar_registers(&name_str);
                Err(LowerError::UndefinedRegister {
                    name: name_str,
                    suggestions,
                })
            }
            AstOperand::ImmInt(val) => Ok(Operand::ImmI64(*val)),
            AstOperand::ImmUInt(val) => Ok(Operand::ImmU64(*val)),
            AstOperand::ImmFloat(val) => {
                // A NaN bit pattern (e.g. 0f7FC00000) denotes no real
                // number: reject it here, the analysis model's ingestion
                // point for float literals. The infinities pass through
                // (running-max/min seeds like 0fFF800000 are how real
                // kernels start their reductions).
                if val.is_nan() {
                    return Err(LowerError::NanLiteral);
                }
                Ok(Operand::ImmF64(*val))
            }
            AstOperand::Symbol(name) => {
                let name_str = name.to_string();
                // Check if it's a shared/local/global memory symbol
                if let Some(op) = self.resolve_mem_symbol(&name_str) {
                    return Ok(op);
                }
                // Unknown symbol
                Err(LowerError::UnsupportedInstruction {
                    instruction: format!("symbol reference: {}", name),
                    reason: Some(format!(
                        "Unknown symbol '{}' - not a shared, local, or global variable",
                        name
                    )),
                })
            }
            AstOperand::Underscore => {
                // Underscore means "don't care" - we create a dummy register
                // This is used for predicates we want to discard
                Err(LowerError::UnsupportedInstruction {
                    instruction: "underscore operand".to_string(),
                    reason: Some("Underscore operands not yet implemented".to_string()),
                })
            }
            AstOperand::Address(addr) => {
                // A memory operand `[base + offset]` has no single-operand
                // value, and resolving just its base would silently drop
                // the offset (how `ldmatrix ... [%r2+16]` once read `[%r2]`,
                // issue #4). Memory instructions go through
                // `resolve_mem_operand`; anything else reaching here is a
                // lowering bug or malformed PTX.
                Err(LowerError::UnsupportedInstruction {
                    instruction: "address operand".to_string(),
                    reason: Some(format!(
                        "memory operand {:?} in a value position (memory \
                         instructions resolve it as base + offset)",
                        addr
                    )),
                })
            }
            AstOperand::Vector(_ops) => {
                // Vector operand - not handled inline, should be handled by instruction
                Err(LowerError::UnsupportedInstruction {
                    instruction: "vector operand".to_string(),
                    reason: Some(
                        "Vector operands should be handled by the instruction".to_string(),
                    ),
                })
            }
            AstOperand::PredicateOperand {
                negated: _,
                name: _,
            } => Err(LowerError::UnsupportedInstruction {
                instruction: "predicate operand".to_string(),
                reason: Some("Use resolve_predicate_operand for predicates".to_string()),
            }),
            AstOperand::PredicatePair(_, _) => Err(LowerError::UnsupportedInstruction {
                instruction: "predicate pair".to_string(),
                reason: Some("Predicate pairs not yet implemented".to_string()),
            }),
            AstOperand::VectorElement(name, component) => {
                // Check if this is actually a special register like %tid.x
                // The parser might mis-parse %tid.x as a VectorElement
                let full_name = format!(
                    "{}.{}",
                    name,
                    match component.canonicalize() {
                        ast::CanonVectorComponent::X => "x",
                        ast::CanonVectorComponent::Y => "y",
                        ast::CanonVectorComponent::Z => "z",
                        ast::CanonVectorComponent::W => "w",
                    }
                );
                if let Some(kind) = SpecialRegKind::from_name(&full_name) {
                    return Ok(Operand::SpecialReg(kind));
                }
                // Otherwise it's a real vector element which we don't support yet
                Err(LowerError::UnsupportedInstruction {
                    instruction: "vector element".to_string(),
                    reason: Some("Vector elements not yet implemented".to_string()),
                })
            }
            AstOperand::Expr(_) => Err(LowerError::UnsupportedInstruction {
                instruction: "expression operand".to_string(),
                reason: Some("Expression operands not yet implemented".to_string()),
            }),
        }
    }

    /// Resolve an operand with type information (for type checking)
    fn resolve_operand_typed(&self, op: &AstOperand) -> LowerResult<ResolvedOperand> {
        match op {
            AstOperand::Ident(name) => {
                let name_str = name.to_string();
                // PTX predefined constant: warp size (polymorphic immediate)
                if name_str == "WARP_SZ" {
                    return Ok(ResolvedOperand::immediate(Operand::ImmI64(32)));
                }
                // Check for special register first
                if let Some(kind) = SpecialRegKind::from_name(&name_str) {
                    let ty = kind.ty();
                    return Ok(ResolvedOperand::special_reg(
                        Operand::SpecialReg(kind),
                        ty,
                        name_str,
                    ));
                }
                // Check if it's a declared register
                if let Some(info) = self.symbols.get_register(&name_str) {
                    return Ok(ResolvedOperand::register(
                        Operand::Reg(info.id),
                        info.declared_type,
                        name_str,
                    ));
                }
                // Check if it's a shared/local/global memory symbol. The
                // resulting address is a constant, so treat it like an
                // immediate (polymorphic - `mov.u32` takes shared addresses,
                // `mov.u64` takes local depot addresses).
                if let Some(op) = self.resolve_mem_symbol(&name_str) {
                    return Ok(ResolvedOperand::immediate(op));
                }
                // Not found - return error
                let suggestions = self.symbols.find_similar_registers(&name_str);
                Err(LowerError::UndefinedRegister {
                    name: name_str,
                    suggestions,
                })
            }
            AstOperand::ImmInt(_) | AstOperand::ImmUInt(_) | AstOperand::ImmFloat(_) => {
                // Immediates are polymorphic - their type is determined by context
                Ok(ResolvedOperand::immediate(self.resolve_operand(op)?))
            }
            AstOperand::VectorElement(name, component) => {
                // Check if this is actually a special register like %tid.x
                let full_name = format!(
                    "{}.{}",
                    name,
                    match component.canonicalize() {
                        ast::CanonVectorComponent::X => "x",
                        ast::CanonVectorComponent::Y => "y",
                        ast::CanonVectorComponent::Z => "z",
                        ast::CanonVectorComponent::W => "w",
                    }
                );
                if let Some(kind) = SpecialRegKind::from_name(&full_name) {
                    let ty = kind.ty();
                    return Ok(ResolvedOperand::special_reg(
                        Operand::SpecialReg(kind),
                        ty,
                        full_name,
                    ));
                }
                // Otherwise delegate to resolve_operand (which will error)
                Ok(ResolvedOperand::immediate(self.resolve_operand(op)?))
            }
            _ => {
                // For other operand types, delegate to resolve_operand
                // These are typically address operands or other special cases
                Ok(ResolvedOperand::immediate(self.resolve_operand(op)?))
            }
        }
    }

    /// Resolve a destination register with type information (for type checking)
    fn resolve_dst_typed(&self, op: &AstOperand) -> LowerResult<ResolvedDst> {
        match op {
            AstOperand::Ident(name) => {
                let name_str = name.to_string();
                // Check if it's a special register (not allowed as destination)
                if SpecialRegKind::from_name(&name_str).is_some() {
                    return Err(LowerError::SpecialRegAsDestination {
                        instruction: "destination".to_string(),
                        register: name_str,
                    });
                }
                // Look up the register
                if let Some(info) = self.symbols.get_register(&name_str) {
                    return Ok(ResolvedDst {
                        reg: info.id,
                        ty: info.declared_type,
                        name: name_str,
                    });
                }
                // Not found
                let suggestions = self.symbols.find_similar_registers(&name_str);
                Err(LowerError::UndefinedRegister {
                    name: name_str,
                    suggestions,
                })
            }
            _ => Err(LowerError::InvalidOperand {
                instruction: "instruction".to_string(),
                operand: format!("{:?}", op),
                reason: "destination must be a register",
            }),
        }
    }

    /// Resolve an address operand to base operand
    fn resolve_address(&self, addr: &Address) -> LowerResult<Operand> {
        match &addr.base {
            AddressBase::Register(name) => {
                let name_str = name.to_string();
                let reg_id = self.resolve_register(&name_str)?;
                Ok(Operand::Reg(reg_id))
            }
            AddressBase::Symbol(name) => {
                // Shared, local, or module-global variables used as a base
                let name_str = name.to_string();
                if let Some(op) = self.resolve_mem_symbol(&name_str) {
                    Ok(op)
                } else {
                    Err(LowerError::UndefinedSymbol { name: name_str })
                }
            }
            AddressBase::Immediate(val) => Ok(Operand::ImmI64(*val)),
        }
    }

    /// The constant byte offset of an address operand `[base + offset]`
    /// (0 when absent). PTX address offsets are immediates, so an offset
    /// that does not evaluate to a constant (an identifier, an operator
    /// `eval_const_expr` does not model, a division by zero) is rejected
    /// rather than silently read as `+0`.
    fn get_address_offset(&self, addr: &Address) -> LowerResult<i64> {
        match &addr.offset {
            Some(expr) => Self::eval_const_expr(expr).ok_or_else(|| LowerError::InvalidOperand {
                instruction: "address operand".to_string(),
                operand: format!("{:?}", expr),
                reason: "address offset must be a constant expression",
            }),
            None => Ok(0),
        }
    }

    /// Resolve a memory operand to `(base, offset)`: `[base + offset]`
    /// splits into its resolved base and constant byte offset; a bare
    /// register or immediate is a base with offset 0. Every memory
    /// instruction resolves its address through here, so no lowering can
    /// drop the offset (`resolve_operand` rejects address operands).
    fn resolve_mem_operand(&self, op: &AstOperand) -> LowerResult<(Operand, i64)> {
        match op {
            AstOperand::Address(addr) => {
                Ok((self.resolve_address(addr)?, self.get_address_offset(addr)?))
            }
            _ => Ok((self.resolve_operand(op)?, 0)),
        }
    }

    /// Try to evaluate a constant expression
    fn eval_const_expr(expr: &ast::Expr) -> Option<i64> {
        match expr {
            ast::Expr::IntLitS(v) => Some(*v),
            ast::Expr::IntLitU(v) => Some(*v as i64),
            ast::Expr::Binary(lhs, op, rhs) => {
                let l = Self::eval_const_expr(lhs)?;
                let r = Self::eval_const_expr(rhs)?;
                Some(match op {
                    ast::BinaryOp::Add => l.wrapping_add(r),
                    ast::BinaryOp::Sub => l.wrapping_sub(r),
                    ast::BinaryOp::Mul => l.wrapping_mul(r),
                    ast::BinaryOp::Div => l.checked_div(r)?,
                    ast::BinaryOp::Shl => l.wrapping_shl(r as u32),
                    ast::BinaryOp::Shr => l.wrapping_shr(r as u32),
                    _ => return None,
                })
            }
            ast::Expr::Unary(op, inner) => {
                let v = Self::eval_const_expr(inner)?;
                Some(match op {
                    ast::UnaryOp::Neg => v.wrapping_neg(),
                    ast::UnaryOp::Pos => v,
                    _ => return None,
                })
            }
            _ => None,
        }
    }

    /// Resolve destination register (must not be a special register)
    fn resolve_dst(&self, op: &AstOperand) -> LowerResult<RegId> {
        match op {
            AstOperand::Ident(name) => {
                let name_str = name.to_string();
                // Check if it's a special register
                if SpecialRegKind::from_name(&name_str).is_some() {
                    return Err(LowerError::SpecialRegAsDestination {
                        instruction: "destination".to_string(),
                        register: name_str,
                    });
                }
                self.resolve_register(&name_str)
            }
            _ => Err(LowerError::InvalidOperand {
                instruction: "instruction".to_string(),
                operand: format!("{:?}", op),
                reason: "destination must be a register",
            }),
        }
    }

    /// Resolve a vector of destination registers
    fn resolve_dst_vector(&self, op: &AstOperand) -> LowerResult<Vec<RegId>> {
        match op {
            AstOperand::Vector(regs) => {
                let mut result = Vec::with_capacity(regs.len());
                for reg_op in regs {
                    match reg_op {
                        AstOperand::Ident(name) => {
                            let name_str = name.to_string();
                            result.push(self.resolve_register(&name_str)?);
                        }
                        _ => {
                            return Err(LowerError::InvalidOperand {
                                instruction: "vector load".to_string(),
                                operand: format!("{:?}", reg_op),
                                reason: "vector element must be a register",
                            });
                        }
                    }
                }
                Ok(result)
            }
            AstOperand::Ident(name) => {
                // Single register - return as single-element vector
                let name_str = name.to_string();
                Ok(vec![self.resolve_register(&name_str)?])
            }
            _ => Err(LowerError::InvalidOperand {
                instruction: "vector load".to_string(),
                operand: format!("{:?}", op),
                reason: "destination must be a register or vector of registers",
            }),
        }
    }

    /// Resolve a predicate guard
    fn resolve_predicate(&self, pred: &ast::Predicate) -> LowerResult<Predicate> {
        let name_str = pred.reg.to_string();
        let reg = self.resolve_register(&name_str)?;
        Ok(Predicate {
            reg,
            negated: pred.negated,
        })
    }

    /// Convert AST memory space to lowered memory space. Generic (spaceless)
    /// accesses and unmodeled spaces are rejected: memory spaces have
    /// separate address spaces here, so silently defaulting to global would
    /// read/write the wrong memory.
    fn convert_space(&self, space: Option<StateSpace>, instruction: &str) -> LowerResult<MemSpace> {
        match space {
            Some(StateSpace::Global) => Ok(MemSpace::Global),
            Some(StateSpace::Shared) => Ok(MemSpace::Shared),
            Some(StateSpace::Local) => Ok(MemSpace::Local),
            Some(StateSpace::Param) => Ok(MemSpace::Param),
            Some(StateSpace::Const) => Ok(MemSpace::Const),
            Some(other) => Err(unsupported(
                instruction,
                format!("state space {:?} is not modeled", other),
            )),
            None => Err(unsupported(
                instruction,
                "generic (spaceless) memory access - the state space must be explicit",
            )),
        }
    }

    /// Convert AST comparison operator to lowered comparison operator
    fn convert_cmp_op(&self, op: AstCmpOp) -> CmpOp {
        match op {
            AstCmpOp::Eq => CmpOp::Eq,
            AstCmpOp::Ne => CmpOp::Ne,
            AstCmpOp::Lt => CmpOp::Lt,
            AstCmpOp::Le => CmpOp::Le,
            AstCmpOp::Gt => CmpOp::Gt,
            AstCmpOp::Ge => CmpOp::Ge,
            AstCmpOp::Lo => CmpOp::Lo,
            AstCmpOp::Ls => CmpOp::Ls,
            AstCmpOp::Hi => CmpOp::Hi,
            AstCmpOp::Hs => CmpOp::Hs,
            AstCmpOp::Equ => CmpOp::Equ,
            AstCmpOp::Neu => CmpOp::Neu,
            AstCmpOp::Ltu => CmpOp::Ltu,
            AstCmpOp::Leu => CmpOp::Leu,
            AstCmpOp::Gtu => CmpOp::Gtu,
            AstCmpOp::Geu => CmpOp::Geu,
            AstCmpOp::Num => CmpOp::Num,
            AstCmpOp::Nan => CmpOp::Nan,
        }
    }

    /// Convert AST shuffle mode to lowered shuffle mode
    fn convert_shfl_mode(&self, mode: AstShflMode) -> ShflMode {
        match mode {
            AstShflMode::Up => ShflMode::Up,
            AstShflMode::Down => ShflMode::Down,
            AstShflMode::Bfly => ShflMode::Bfly,
            AstShflMode::Idx => ShflMode::Idx,
        }
    }

    /// Convert AST mul mode to lowered mul mode
    fn convert_mul_mode(&self, mode: MulMode) -> LoweredMulMode {
        match mode {
            MulMode::Hi => LoweredMulMode::Hi,
            MulMode::Lo => LoweredMulMode::Lo,
            MulMode::Wide => LoweredMulMode::Wide,
        }
    }

    // =========================================================================
    // Type Checking Helpers
    // =========================================================================

    /// Check that an operand's type is compatible with the instruction's expected type.
    /// Per PTX 9.4:
    /// - Bit-types (.bX) are compatible with any type of same size
    /// - Signed/unsigned integers of same size are compatible
    /// - Float types must match exactly (no float<->int mixing)
    fn check_operand_type(
        &self,
        resolved: &ResolvedOperand,
        expected: ScalarType,
        instruction: &str,
    ) -> LowerResult<()> {
        // Immediates are polymorphic - they take on the instruction's type
        let Some(actual) = resolved.ty else {
            return Ok(());
        };

        match check_type_compatibility(actual, expected) {
            TypeCompatibility::Exact | TypeCompatibility::Compatible => Ok(()),
            TypeCompatibility::Incompatible { reason: _ } => {
                let name = resolved
                    .name
                    .clone()
                    .unwrap_or_else(|| "operand".to_string());
                Err(LowerError::TypeMismatch {
                    register: name,
                    declared_type: actual,
                    used_as: expected,
                    instruction: instruction.to_string(),
                    hint: self.type_mismatch_hint(actual, expected),
                })
            }
        }
    }

    /// Check that a destination register's type is compatible with the instruction's type.
    fn check_dst_type(
        &self,
        dst: &ResolvedDst,
        expected: ScalarType,
        instruction: &str,
    ) -> LowerResult<()> {
        match check_type_compatibility(dst.ty, expected) {
            TypeCompatibility::Exact | TypeCompatibility::Compatible => Ok(()),
            TypeCompatibility::Incompatible { reason: _ } => Err(LowerError::TypeMismatch {
                register: dst.name.clone(),
                declared_type: dst.ty,
                used_as: expected,
                instruction: instruction.to_string(),
                hint: self.type_mismatch_hint(dst.ty, expected),
            }),
        }
    }

    /// Relaxed type checking for ld/st/cvt instructions (PTX 9.4.1).
    /// Allows operands to be wider than the instruction type.
    /// The value will be truncated (store) or extended (load) as needed.
    fn check_operand_type_relaxed(
        &self,
        resolved: &ResolvedOperand,
        instr_ty: ScalarType,
        instruction: &str,
    ) -> LowerResult<()> {
        // Immediates are polymorphic
        let Some(actual) = resolved.ty else {
            return Ok(());
        };

        let actual_bits = actual.bits();
        let instr_bits = instr_ty.bits();

        // Operand can be wider than instruction type (will be truncated/extended)
        if actual_bits >= instr_bits {
            // Still check type category compatibility (no float<->int mixing)
            // unless one is a bit-type
            if actual.is_bits_type() || instr_ty.is_bits_type() {
                return Ok(());
            }
            // Float<->int mixing is still invalid
            if actual.is_float() != instr_ty.is_float() {
                let name = resolved
                    .name
                    .clone()
                    .unwrap_or_else(|| "operand".to_string());
                return Err(LowerError::TypeMismatch {
                    register: name,
                    declared_type: actual,
                    used_as: instr_ty,
                    instruction: instruction.to_string(),
                    hint: "Cannot mix float and integer types; use cvt for conversion".to_string(),
                });
            }
            return Ok(());
        }

        // Operand is narrower than instruction type - not allowed even with relaxed rules
        let name = resolved
            .name
            .clone()
            .unwrap_or_else(|| "operand".to_string());
        Err(LowerError::TypeMismatch {
            register: name,
            declared_type: actual,
            used_as: instr_ty,
            instruction: instruction.to_string(),
            hint: format!(
                "Operand is {} bits but instruction requires at least {} bits",
                actual_bits, instr_bits
            ),
        })
    }

    /// Relaxed type checking for destination registers (ld instructions).
    /// Destination can be wider than instruction type (value will be extended).
    fn check_dst_type_relaxed(
        &self,
        dst: &ResolvedDst,
        instr_ty: ScalarType,
        instruction: &str,
    ) -> LowerResult<()> {
        let dst_bits = dst.ty.bits();
        let instr_bits = instr_ty.bits();

        // Destination can be wider (value will be extended)
        if dst_bits >= instr_bits {
            // Check type category compatibility
            if dst.ty.is_bits_type() || instr_ty.is_bits_type() {
                return Ok(());
            }
            if dst.ty.is_float() != instr_ty.is_float() {
                return Err(LowerError::TypeMismatch {
                    register: dst.name.clone(),
                    declared_type: dst.ty,
                    used_as: instr_ty,
                    instruction: instruction.to_string(),
                    hint: "Cannot mix float and integer types; use cvt for conversion".to_string(),
                });
            }
            return Ok(());
        }

        // Destination is narrower - not allowed
        Err(LowerError::TypeMismatch {
            register: dst.name.clone(),
            declared_type: dst.ty,
            used_as: instr_ty,
            instruction: instruction.to_string(),
            hint: format!(
                "Destination register is {} bits but instruction produces {} bits",
                dst_bits, instr_bits
            ),
        })
    }

    /// Generate a helpful hint for type mismatch errors
    fn type_mismatch_hint(&self, actual: ScalarType, expected: ScalarType) -> String {
        // Float<->int mismatch
        if actual.is_float() && expected.is_integer() {
            return format!(
                "Use cvt.{}.{} to convert float to integer",
                crate::types::format_scalar_type(expected),
                crate::types::format_scalar_type(actual)
            );
        }
        if actual.is_integer() && expected.is_float() {
            return format!(
                "Use cvt.{}.{} to convert integer to float",
                crate::types::format_scalar_type(expected),
                crate::types::format_scalar_type(actual)
            );
        }

        // Size mismatch
        if actual.bits() != expected.bits() {
            return format!(
                "Type size mismatch: {} is {} bits, expected {} bits",
                crate::types::format_scalar_type(actual),
                actual.bits(),
                expected.bits()
            );
        }

        "Declare register with compatible type".to_string()
    }
}

impl Default for LoweringContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Lower a PTX function to a LoweredProgram
///
/// `module_vars` should contain module-level variable declarations (e.g., extern shared memory).
pub fn lower_function(func: &Function, module_vars: &[VarDecl]) -> LowerResult<LoweredProgram> {
    let body = func.body.as_ref().ok_or(LowerError::NoFunctionBody {
        name: func.name.to_string(),
    })?;

    let mut ctx = LoweringContext::new();

    // First: collect module-level declarations (e.g., extern shared memory)
    for var in module_vars {
        collect_var_decl(&mut ctx, var)?;
    }

    // Second: collect function-level declarations
    collect_declarations(&mut ctx, &func.params, body)?;

    // Place the extern-shared window now that every static `.shared`
    // declaration is known: the CUDA ABI bases the dynamic segment after
    // all static allocations. This must precede `lower_body`, which is the
    // first point that resolves shared symbols to addresses
    // (`resolve_mem_symbol`).
    ctx.symbols.finalize_shared_layout()?;

    // Third: lower instructions
    lower_body(&mut ctx, body)?;

    // Resolve forward references
    ctx.resolve_forward_refs()?;

    Ok(LoweredProgram {
        instructions: IdVec::from_vec(ctx.instructions),
        predicates: IdVec::from_vec(ctx.predicates),
        symbols: ctx.symbols,
        source_map: ctx.source_map_builder.build(),
        entry_pc: InstrId::from_index(0),
    })
}

/// Collect all declarations (registers, labels, shared memory)
fn collect_declarations(
    ctx: &mut LoweringContext,
    params: &[ast::Parameter],
    body: &FunctionBody,
) -> LowerResult<()> {
    // Collect kernel parameters
    for param in params {
        let ty = param.ty.scalar;
        let size_bytes = ty.size_bytes() as u64;
        ctx.symbols
            .declare_param(&param.name.to_string(), ty, size_bytes, param.ptr)?;
    }

    // Collect declarations from body
    for stmt in &body.statements {
        collect_stmt_declarations(ctx, stmt)?;
    }

    Ok(())
}

/// Collect declarations from a statement
fn collect_stmt_declarations(ctx: &mut LoweringContext, stmt: &Statement) -> LowerResult<()> {
    match stmt {
        Statement::Variable(var_decl) => {
            collect_var_decl(ctx, var_decl)?;
        }
        Statement::Block(stmts) => {
            for s in stmts {
                collect_stmt_declarations(ctx, s)?;
            }
        }
        Statement::Label(_) | Statement::Instruction(_) | Statement::Directive(_) => {
            // These don't introduce declarations
        }
    }
    Ok(())
}

/// Collect a variable declaration
fn collect_var_decl(ctx: &mut LoweringContext, var: &VarDecl) -> LowerResult<()> {
    let name = var.name.to_string();
    let ty = var.ty.scalar;
    let count = var.param_count.unwrap_or(1);

    // Checked product (release-active): each dimension fits u32, but three
    // or more hostile dimensions can wrap the u64 product, and a wrapped
    // element count would silently pass the packer's own checked
    // byte-size arithmetic.
    let num_elements: u64 = var
        .array_dims
        .iter()
        .filter_map(|d| *d)
        .try_fold(1u64, |acc, d| acc.checked_mul(d as u64))
        .ok_or_else(|| LowerError::VariableSizeOverflow {
            what: format!("variable '{}' (array dimensions overflow)", name),
        })?;
    let num_elements = if num_elements == 0 { 1 } else { num_elements };
    let alignment = var.align.unwrap_or(ty.size_bytes() as u32) as u64;

    match var.space {
        StateSpace::Reg => {
            ctx.symbols.declare_register(&name, ty, count)?;
        }
        StateSpace::Shared => {
            let is_extern = matches!(var.linkage, ast::Linkage::Extern);
            ctx.symbols
                .declare_shared(&name, ty, num_elements, is_extern, alignment)?;
        }
        StateSpace::Param => {
            // Block-scope `.param` slots used by the callseq idiom. These are
            // virtual: no memory is allocated and no instructions touch them.
            ctx.local_params.insert(name, LocalParamSlot::Empty);
        }
        StateSpace::Local => {
            // Function-scope local memory (e.g. the __local_depot stack array)
            ctx.symbols
                .declare_local(&name, ty, num_elements, alignment)?;
        }
        StateSpace::Global => {
            // Module-scope variable (e.g. set by the host via
            // cudaMemcpyToSymbol); the driver binds its value by name.
            ctx.symbols
                .declare_global_var(&name, ty, num_elements, alignment)?;
        }
        _ => {
            // Other state spaces (const, tex) - handle as needed
        }
    }

    Ok(())
}

/// Lower the function body
fn lower_body(ctx: &mut LoweringContext, body: &FunctionBody) -> LowerResult<()> {
    for stmt in &body.statements {
        lower_statement(ctx, stmt)?;
    }
    Ok(())
}

/// Lower a statement
fn lower_statement(ctx: &mut LoweringContext, stmt: &Statement) -> LowerResult<()> {
    match stmt {
        Statement::Label(label) => {
            ctx.record_label(&label.name.to_string(), Some(label.span));
        }
        Statement::Instruction(instr) => {
            // Set current span for error reporting
            ctx.current_span = Some(instr.span);
            lower_instruction(ctx, instr)?;
        }
        Statement::Block(stmts) => {
            for s in stmts {
                lower_statement(ctx, s)?;
            }
        }
        Statement::Variable(_) => {
            // Already handled in first pass
        }
        Statement::Directive(_) => {
            // Directives are ignored for now
        }
    }
    Ok(())
}

/// Lower an instruction
fn lower_instruction(ctx: &mut LoweringContext, instr: &Instruction) -> LowerResult<()> {
    // Resolve predicate if present
    let predicate = match &instr.predicate {
        Some(p) => Some(ctx.resolve_predicate(p)?),
        None => None,
    };

    match &instr.op {
        InstructionOp::Parsed(parsed) => {
            lower_parsed_instruction(ctx, parsed, predicate)?;
        }
        InstructionOp::Unparsed {
            kind,
            modifiers,
            operands,
        } => {
            // Try to parse the unparsed instruction into a strongly-typed form
            match parse_instruction(*kind, modifiers.clone(), operands.clone()) {
                Ok(parsed) => {
                    lower_parsed_instruction(ctx, &parsed, predicate)?;
                }
                Err(e) => {
                    // If parsing fails, report the error
                    Err(LowerError::UnsupportedInstruction {
                        instruction: format!("{:?}", kind),
                        reason: Some(format!("Instruction parsing failed: {:?}", e)),
                    })?;
                }
            }
        }
    }

    Ok(())
}

/// Lower a parsed instruction
fn lower_parsed_instruction(
    ctx: &mut LoweringContext,
    instr: &ParsedInstruction,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    match instr {
        // =========================================================================
        // Arithmetic - Add
        // =========================================================================
        ParsedInstruction::Add(add_instr) => {
            lower_add(ctx, add_instr, predicate)?;
        }

        // =========================================================================
        // Arithmetic - Sub
        // =========================================================================
        ParsedInstruction::Sub(sub) => {
            lower_sub(ctx, sub, predicate)?;
        }

        // =========================================================================
        // Arithmetic - Mul
        // =========================================================================
        ParsedInstruction::Mul(mul) => {
            lower_mul(ctx, mul, predicate)?;
        }

        // =========================================================================
        // Arithmetic - Mad (multiply-add)
        // =========================================================================
        ParsedInstruction::Mad(mad) => {
            lower_mad(ctx, mad, predicate)?;
        }

        // =========================================================================
        // Arithmetic - FMA (fused multiply-add)
        // =========================================================================
        ParsedInstruction::Fma(fma) => {
            lower_fma(ctx, fma, predicate)?;
        }

        // =========================================================================
        // Arithmetic - Div
        // =========================================================================
        ParsedInstruction::Div(div) => {
            lower_div(ctx, div, predicate)?;
        }

        // =========================================================================
        // Arithmetic - Rem
        // =========================================================================
        ParsedInstruction::Rem(rem) => {
            let dst_typed = ctx.resolve_dst_typed(&rem.dst)?;
            let src_a_typed = ctx.resolve_operand_typed(&rem.src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(&rem.src_b)?;

            ctx.check_dst_type(&dst_typed, rem.ty, "rem")?;
            ctx.check_operand_type(&src_a_typed, rem.ty, "rem")?;
            ctx.check_operand_type(&src_b_typed, rem.ty, "rem")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Rem,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: rem.ty,
                    clamp: None,
                },
                predicate,
            )?;
        }

        // =========================================================================
        // Arithmetic - Neg
        // =========================================================================
        ParsedInstruction::Neg(neg) => {
            lower_neg(ctx, neg, predicate)?;
        }

        // =========================================================================
        // Float unary - Rcp / Sqrt / Rsqrt
        // =========================================================================
        ParsedInstruction::Rcp(rcp) => {
            // rnd/ftz/approx are precision controls: floats are reals here,
            // so every reciprocal is exact and they change nothing.
            let (ty, dst, src) = match rcp {
                ast::RcpInstr::Approx {
                    ftz: _ftz,
                    ty,
                    dst,
                    src,
                } => (*ty, dst, src),
                ast::RcpInstr::Ieee {
                    rnd: _rnd,
                    ftz: _ftz,
                    ty,
                    dst,
                    src,
                } => (*ty, dst, src),
            };
            lower_float_unary(ctx, UnaryOp::Rcp, "rcp", ty, dst, src, predicate)?;
        }
        ParsedInstruction::Sqrt(sqrt) => {
            // rnd/ftz/approx are precision controls: no effect over the reals.
            let (ty, dst, src) = match sqrt {
                ast::SqrtInstr::Approx {
                    ftz: _ftz,
                    dst,
                    src,
                } => (ScalarType::F32, dst, src),
                ast::SqrtInstr::Ieee {
                    rnd: _rnd,
                    ftz: _ftz,
                    ty,
                    dst,
                    src,
                } => (*ty, dst, src),
            };
            lower_float_unary(ctx, UnaryOp::Sqrt, "sqrt", ty, dst, src, predicate)?;
        }
        ParsedInstruction::Rsqrt(rsqrt) => {
            // approx/ftz are precision controls: no effect over the reals.
            let ast::RsqrtInstr {
                approx: _approx,
                ftz: _ftz,
                ty,
                dst,
                src,
            } = rsqrt;
            lower_float_unary(ctx, UnaryOp::Rsqrt, "rsqrt", *ty, dst, src, predicate)?;
        }

        // =========================================================================
        // Arithmetic - Abs
        // =========================================================================
        ParsedInstruction::Abs(abs) => {
            lower_abs(ctx, abs, predicate)?;
        }

        // =========================================================================
        // Arithmetic - Min/Max
        // =========================================================================
        ParsedInstruction::Min(min) => {
            lower_min(ctx, min, predicate)?;
        }
        ParsedInstruction::Max(max) => {
            lower_max(ctx, max, predicate)?;
        }

        // =========================================================================
        // Logic - And/Or/Xor
        // =========================================================================
        ParsedInstruction::And(logic)
        | ParsedInstruction::Or(logic)
        | ParsedInstruction::Xor(logic) => {
            let (op, instr_name) = match instr {
                ParsedInstruction::And(_) => (BinOp::And, "and"),
                ParsedInstruction::Or(_) => (BinOp::Or, "or"),
                ParsedInstruction::Xor(_) => (BinOp::Xor, "xor"),
                _ => unreachable!(),
            };
            let dst_typed = ctx.resolve_dst_typed(&logic.dst)?;
            let src_a_typed = ctx.resolve_operand_typed(&logic.src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(&logic.src_b)?;

            ctx.check_dst_type(&dst_typed, logic.ty, instr_name)?;
            ctx.check_operand_type(&src_a_typed, logic.ty, instr_name)?;
            ctx.check_operand_type(&src_b_typed, logic.ty, instr_name)?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: logic.ty,
                    clamp: None,
                },
                predicate,
            )?;
        }

        // =========================================================================
        // Logic - Not
        // =========================================================================
        ParsedInstruction::Not(not) => {
            let dst_typed = ctx.resolve_dst_typed(&not.dst)?;
            let src_typed = ctx.resolve_operand_typed(&not.src)?;

            ctx.check_dst_type(&dst_typed, not.ty, "not")?;
            ctx.check_operand_type(&src_typed, not.ty, "not")?;

            ctx.emit(
                LoweredInstr::UnaryOp {
                    op: UnaryOp::Not,
                    dst: dst_typed.reg,
                    src: src_typed.operand,
                    ty: not.ty,
                },
                predicate,
            )?;
        }

        // =========================================================================
        // Shift - Shl/Shr
        // =========================================================================
        ParsedInstruction::Shl(shift) | ParsedInstruction::Shr(shift) => {
            let (op, instr_name) = match instr {
                ParsedInstruction::Shl(_) => (BinOp::Shl, "shl"),
                ParsedInstruction::Shr(_) => (BinOp::Shr, "shr"),
                _ => unreachable!(),
            };
            let dst_typed = ctx.resolve_dst_typed(&shift.dst)?;
            let src_a_typed = ctx.resolve_operand_typed(&shift.src_a)?;
            // For shift, src_b is the shift amount - typically u32, but we check against instruction type
            let src_b_typed = ctx.resolve_operand_typed(&shift.src_b)?;

            ctx.check_dst_type(&dst_typed, shift.ty, instr_name)?;
            ctx.check_operand_type(&src_a_typed, shift.ty, instr_name)?;
            // Shift amount is typically smaller, but we allow same type or compatible
            ctx.check_operand_type(&src_b_typed, shift.ty, instr_name)?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: shift.ty,
                    clamp: None,
                },
                predicate,
            )?;
        }

        // =========================================================================
        // Bit Field Insert - Bfi
        // =========================================================================
        ParsedInstruction::Bfi(bfi) => {
            let dst_typed = ctx.resolve_dst_typed(&bfi.dst)?;
            let src_a_typed = ctx.resolve_operand_typed(&bfi.src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(&bfi.src_b)?;
            // start and len are typically u32 immediates or registers
            let start = ctx.resolve_operand(&bfi.start)?;
            let len = ctx.resolve_operand(&bfi.len)?;

            ctx.check_dst_type(&dst_typed, bfi.ty, "bfi")?;
            ctx.check_operand_type(&src_a_typed, bfi.ty, "bfi")?;
            ctx.check_operand_type(&src_b_typed, bfi.ty, "bfi")?;

            ctx.emit(
                LoweredInstr::Bfi {
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    start,
                    len,
                    ty: bfi.ty,
                },
                predicate,
            )?;
        }

        // =========================================================================
        // Comparison - Setp
        // =========================================================================
        ParsedInstruction::Setp(setp) => {
            lower_setp(ctx, setp, predicate)?;
        }

        // =========================================================================
        // Selection - Selp
        // =========================================================================
        ParsedInstruction::Selp(selp) => {
            let dst_typed = ctx.resolve_dst_typed(&selp.dst)?;
            let src_a_typed = ctx.resolve_operand_typed(&selp.src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(&selp.src_b)?;
            let pred_typed = ctx.resolve_operand_typed(&selp.src_c)?;

            ctx.check_dst_type(&dst_typed, selp.ty, "selp")?;
            ctx.check_operand_type(&src_a_typed, selp.ty, "selp")?;
            ctx.check_operand_type(&src_b_typed, selp.ty, "selp")?;
            // Predicate operand must be pred type
            ctx.check_operand_type(&pred_typed, ScalarType::Pred, "selp")?;

            ctx.emit(
                LoweredInstr::Selp {
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    pred: pred_typed.operand,
                    ty: selp.ty,
                },
                predicate,
            )?;
        }

        // =========================================================================
        // Data Movement - Mov
        // =========================================================================
        ParsedInstruction::Mov(mov) => {
            match &mov.src {
                AstOperand::Vector(elements) if elements.len() == 2 => {
                    // Vector-to-scalar pack: mov.bN dst, {lo, hi}
                    // PTX ISA 9.7.9.4 semantics: d = lo | (hi << w)
                    // where w = type_bits / 2
                    let dst_typed = ctx.resolve_dst_typed(&mov.dst)?;
                    let dst = dst_typed.reg;
                    let elem_width = mov.ty.bits() / 2;

                    let lo = ctx.resolve_operand(&elements[0])?;
                    let hi = ctx.resolve_operand(&elements[1])?;

                    // Step 1: dst = hi << w
                    ctx.emit(
                        LoweredInstr::BinOp {
                            op: BinOp::Shl,
                            dst,
                            src_a: hi,
                            src_b: Operand::ImmI64(elem_width as i64),
                            ty: mov.ty,
                            clamp: None,
                        },
                        predicate,
                    )?;

                    // Step 2: dst = dst | lo
                    ctx.emit(
                        LoweredInstr::BinOp {
                            op: BinOp::Or,
                            dst,
                            src_a: Operand::Reg(dst),
                            src_b: lo,
                            ty: mov.ty,
                            clamp: None,
                        },
                        predicate,
                    )?;
                }
                AstOperand::Vector(elements) => {
                    return Err(LowerError::UnsupportedInstruction {
                        instruction: format!(
                            "mov.{:?} (vector pack with {} elements)",
                            mov.ty,
                            elements.len()
                        ),
                        reason: Some("only 2-element vector pack is supported".to_string()),
                    });
                }
                _ => {
                    // Normal scalar mov
                    let dst_typed = ctx.resolve_dst_typed(&mov.dst)?;
                    let src_typed = ctx.resolve_operand_typed(&mov.src)?;

                    ctx.check_dst_type(&dst_typed, mov.ty, "mov")?;
                    ctx.check_operand_type(&src_typed, mov.ty, "mov")?;

                    ctx.emit(
                        LoweredInstr::Mov {
                            dst: dst_typed.reg,
                            src: src_typed.operand,
                            ty: mov.ty,
                        },
                        predicate,
                    )?;
                }
            }
        }

        // =========================================================================
        // Data Movement - Load
        // =========================================================================
        ParsedInstruction::Ld(ld) => {
            lower_load(ctx, ld, predicate)?;
        }

        // =========================================================================
        // Data Movement - Store
        // =========================================================================
        ParsedInstruction::St(st) => {
            lower_store(ctx, st, predicate)?;
        }

        // =========================================================================
        // Data Movement - Cvt (type conversion)
        // =========================================================================
        ParsedInstruction::Cvt(cvt) => {
            lower_cvt(ctx, cvt, predicate)?;
        }

        // =========================================================================
        // Data Movement - Cvta (address conversion)
        // =========================================================================
        ParsedInstruction::Cvta(cvta) => {
            let ast::CvtaInstr {
                to_generic,
                space,
                ty: _ty, // pointer width; the identity holds at either width
                dst,
                src,
            } = cvta;
            // Only `cvta.to.global` is accepted, as the identity: global
            // addresses are absolute u64s here and the generic window over
            // global is identity-mapped, so generic->global is exact (the
            // corpus's one cvta form: param pointers into global arrays).
            // Every other form is rejected loudly, for two distinct
            // reasons:
            // - `cvta.global` (the to-generic direction over global) is
            //   identity-compatible for the same reason, but the corpus
            //   never emits it, so it is rejected purely to keep the
            //   modeled instruction surface minimal.
            // - The remaining forms have no faithful model: generic
            //   addressing has no per-space windows here, so a generic
            //   address minted from a shared/local/const address
            //   (`cvta.<space>`) would be an absolute address in the
            //   *wrong* space, and `cvta.to.<other-space>` would bless an
            //   arbitrary value as a shared/local address. Spaceless
            //   ld/st is already rejected (`convert_space`), so no
            //   accepted instruction can consume a generic address
            //   derived from shared/local - rejecting the producers
            //   closes the loop.
            if *to_generic || *space != StateSpace::Global {
                let form = format!(
                    "cvta{}.{}",
                    if *to_generic { "" } else { ".to" },
                    format!("{:?}", space).to_lowercase()
                );
                let reason = if *space == StateSpace::Global {
                    // Necessarily the to-generic direction here.
                    "cvta.global (global -> generic) would also be the identity, \
                     but the corpus never uses it; only cvta.to.global is modeled"
                } else {
                    "only cvta.to.global is modeled (as the identity); generic \
                     addresses have no per-space windows in this model"
                };
                return Err(LowerError::UnsupportedInstruction {
                    instruction: form,
                    reason: Some(reason.to_string()),
                });
            }
            let dst = ctx.resolve_dst(dst)?;
            let src = ctx.resolve_operand(src)?;
            ctx.emit(
                LoweredInstr::Cvta {
                    dst,
                    src,
                    space: MemSpace::Global,
                },
                predicate,
            )?;
        }

        // =========================================================================
        // Warp Shuffle - ShflSync
        // =========================================================================
        ParsedInstruction::ShflSync(shfl) => {
            lower_shfl_sync(ctx, shfl, predicate)?;
        }

        // =========================================================================
        // Control Flow - Branch
        // =========================================================================
        ParsedInstruction::Bra(bra) => {
            lower_branch(ctx, bra, predicate)?;
        }

        // =========================================================================
        // Control Flow - Return/Exit
        // =========================================================================
        ParsedInstruction::Ret(ast::RetInstr {
            // .uni is a divergence hint; it does not change what ret does.
            uniform: _uniform,
        }) => {
            ctx.emit(LoweredInstr::Ret, predicate)?;
        }
        ParsedInstruction::Exit => {
            ctx.emit(LoweredInstr::Exit, predicate)?;
        }

        // =========================================================================
        // Synchronization - Bar
        // =========================================================================
        ParsedInstruction::Bar(bar) => {
            lower_bar(ctx, bar.mode, &bar.operands, predicate)?;
        }

        // `barrier{.cta}.sync{.aligned} a` without a thread count is the
        // same full-CTA barrier as `bar.sync a`: the ISA states
        // "bar{.cta}.sync is equivalent to barrier{.cta}.sync.aligned",
        // and dropping `.aligned` only drops the compile-time promise that
        // all threads reach the same textual barrier - the runtime
        // semantics the evaluator models are identical. The bar path's
        // restrictions (immediate id 0-15, no thread-count operand, no
        // .arrive/.red) apply unchanged.
        ParsedInstruction::Barrier(ast::BarrierInstr {
            // .cta is the default scope, and .aligned only adds the
            // compile-time all-threads-reach-this-barrier promise (see
            // above); neither changes the runtime semantics modeled here.
            cta: _cta,
            aligned: _aligned,
            mode,
            operands,
        }) => {
            lower_bar(ctx, *mode, operands, predicate)?;
        }

        // =========================================================================
        // Synchronization - bar.warp.sync
        // =========================================================================
        ParsedInstruction::BarWarpSync(bws) => {
            // The membermask lowers exactly like shfl.sync's: any operand
            // form, required concrete at evaluation time.
            let mask = ctx.resolve_operand(&bws.membermask)?;
            ctx.emit(LoweredInstr::BarWarpSync { mask }, predicate)?;
        }

        // =========================================================================
        // Synchronization - Membar
        // =========================================================================
        ParsedInstruction::Membar(membar) => {
            let scope = match membar.level {
                ast::MemScope::Cta => MembarScope::Cta,
                ast::MemScope::Cluster => MembarScope::Cta, // Treat cluster as CTA
                ast::MemScope::Gpu => MembarScope::Gpu,
                ast::MemScope::Sys => MembarScope::Sys,
            };
            ctx.emit(LoweredInstr::Membar { scope }, predicate)?;
        }

        // =========================================================================
        // Function call (callseq idiom for __symexpf)
        // =========================================================================
        ParsedInstruction::Call(call) => {
            lower_call(ctx, call, predicate)?;
        }

        // =========================================================================
        // Warp query - activemask
        // =========================================================================
        ParsedInstruction::Activemask(am) => {
            let dst = ctx.resolve_dst(&am.dst)?;
            ctx.emit(LoweredInstr::Activemask { dst }, predicate)?;
        }

        // =========================================================================
        // Other - NOP placeholder
        // =========================================================================
        ParsedInstruction::Brkpt => {
            ctx.emit(LoweredInstr::Nop, predicate)?;
        }
        ParsedInstruction::Trap => {
            // Reaching a trap during evaluation is an analysis error
            ctx.emit(LoweredInstr::Trap, predicate)?;
        }

        // =========================================================================
        // Other - ld.global.nc (non-coherent global load)
        // Treat as regular global load - we don't model cache coherence
        // =========================================================================
        ParsedInstruction::Other {
            kind: InstrKind::LdGlobalNc,
            modifiers,
            operands,
        } => {
            lower_ld_global_nc(ctx, modifiers, operands, predicate)?;
        }

        // =========================================================================
        // Tensor Core - ldmatrix.sync
        // =========================================================================
        ParsedInstruction::Other {
            kind: InstrKind::Ldmatrix,
            modifiers,
            operands,
        } => {
            lower_ldmatrix(ctx, modifiers, operands, predicate)?;
        }

        // =========================================================================
        // Tensor Core - mma.sync
        // =========================================================================
        ParsedInstruction::Other {
            kind: InstrKind::Mma,
            modifiers,
            operands,
        } => {
            lower_mma(ctx, modifiers, operands, predicate)?;
        }

        // =========================================================================
        // Tensor Core - wmma.load
        // =========================================================================
        ParsedInstruction::Other {
            kind: InstrKind::WmmaLoad,
            modifiers,
            operands,
        } => {
            lower_wmma_load(ctx, modifiers, operands, predicate)?;
        }

        // =========================================================================
        // Tensor Core - wmma.store
        // =========================================================================
        ParsedInstruction::Other {
            kind: InstrKind::WmmaStore,
            modifiers,
            operands,
        } => {
            lower_wmma_store(ctx, modifiers, operands, predicate)?;
        }

        // =========================================================================
        // Tensor Core - wmma.mma
        // =========================================================================
        ParsedInstruction::Other {
            kind: InstrKind::WmmaMma,
            modifiers,
            operands,
        } => {
            lower_wmma_mma(ctx, modifiers, operands, predicate)?;
        }

        // =========================================================================
        // Unsupported instructions
        // =========================================================================
        _ => {
            return Err(LowerError::UnsupportedInstruction {
                instruction: format!("{:?}", instr),
                reason: Some("Instruction not yet implemented".to_string()),
            });
        }
    }

    Ok(())
}

// =========================================================================
// Instruction Lowering Helpers
// =========================================================================

fn lower_add(
    ctx: &mut LoweringContext,
    add: &AddInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    match add {
        AddInstr::Integer {
            ty,
            dst,
            src_a,
            src_b,
        } => {
            check_not_packed(*ty, "add")?;

            // Resolve with type information
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            // Type check against instruction type
            ctx.check_dst_type(&dst_typed, *ty, "add")?;
            ctx.check_operand_type(&src_a_typed, *ty, "add")?;
            ctx.check_operand_type(&src_b_typed, *ty, "add")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Add,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: *ty,
                    clamp: None,
                },
                predicate,
            )?;
        }
        AddInstr::IntegerSat {
            sat,
            dst,
            src_a,
            src_b,
        } => {
            if *sat {
                return Err(unsupported("add.sat.s32", ".sat (saturating integer add)"));
            }
            let ty = ScalarType::S32;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, ty, "add.sat")?;
            ctx.check_operand_type(&src_a_typed, ty, "add.sat")?;
            ctx.check_operand_type(&src_b_typed, ty, "add.sat")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Add,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty,
                    clamp: None,
                },
                predicate,
            )?;
        }
        AddInstr::Float32 {
            // Rounding mode and subnormal flushing don't apply: floats are
            // reals here, so every add is exact. `.sat` is a value clamp
            // (exact over the reals) threaded through as `Clamp::Sat`.
            rnd: _rnd,
            ftz: _ftz,
            sat,
            dst,
            src_a,
            src_b,
        } => {
            let ty = ScalarType::F32;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, ty, "add.f32")?;
            ctx.check_operand_type(&src_a_typed, ty, "add.f32")?;
            ctx.check_operand_type(&src_b_typed, ty, "add.f32")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Add,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty,
                    clamp: sat.then_some(Clamp::Sat),
                },
                predicate,
            )?;
        }
        AddInstr::Float32x2 { .. } => {
            return Err(unsupported("add.f32x2", "packed SIMD arithmetic on F32x2"));
        }
        AddInstr::Float64 {
            // Rounding mode doesn't apply over the reals.
            rnd: _rnd,
            dst,
            src_a,
            src_b,
        } => {
            let ty = ScalarType::F64;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, ty, "add.f64")?;
            ctx.check_operand_type(&src_a_typed, ty, "add.f64")?;
            ctx.check_operand_type(&src_b_typed, ty, "add.f64")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Add,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty,
                    clamp: None,
                },
                predicate,
            )?;
        }
        AddInstr::HalfF16 {
            // Rounding mode and subnormal flushing don't apply over the
            // reals; `.sat` is an exact value clamp.
            rnd: _rnd,
            ftz: _ftz,
            sat,
            ty,
            dst,
            src_a,
            src_b,
        } => {
            check_not_packed(*ty, "add")?;
            lower_half_binop(
                ctx,
                BinOp::Add,
                "add.f16",
                *ty,
                sat.then_some(Clamp::Sat),
                dst,
                src_a,
                src_b,
                predicate,
            )?;
        }
        AddInstr::HalfBf16 {
            // Rounding mode doesn't apply over the reals.
            rnd: _rnd,
            ty,
            dst,
            src_a,
            src_b,
        } => {
            check_not_packed(*ty, "add")?;
            lower_half_binop(
                ctx,
                BinOp::Add,
                "add.bf16",
                *ty,
                None,
                dst,
                src_a,
                src_b,
                predicate,
            )?;
        }
        AddInstr::MixedPrecision {
            // Rounding mode doesn't apply over the reals.
            rnd: _rnd,
            sat,
            src_type,
            dst,
            src_a,
            src_b,
        } => {
            if *sat {
                return Err(unsupported("add.f32 (mixed)", ".sat modifier"));
            }
            check_not_packed(*src_type, "add (mixed)")?;

            // Mixed precision add: f32 result from half (f16/bf16) inputs
            let dst_ty = ScalarType::F32;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            // Destination must be f32-compatible; sources match the half type
            ctx.check_dst_type(&dst_typed, dst_ty, "add.f32 (mixed)")?;
            ctx.check_operand_type(&src_a_typed, *src_type, "add.f32 (mixed)")?;
            ctx.check_operand_type(&src_b_typed, *src_type, "add.f32 (mixed)")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Add,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: dst_ty,
                    clamp: None,
                },
                predicate,
            )?;
        }
    }
    Ok(())
}

/// Shared lowering for scalar half-precision (f16/bf16) binops.
#[allow(clippy::too_many_arguments)] // thin emission helper; mirrors the instruction shape
fn lower_half_binop(
    ctx: &mut LoweringContext,
    op: BinOp,
    instr_name: &str,
    ty: ScalarType,
    clamp: Option<Clamp>,
    dst: &AstOperand,
    src_a: &AstOperand,
    src_b: &AstOperand,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let dst_typed = ctx.resolve_dst_typed(dst)?;
    let src_a_typed = ctx.resolve_operand_typed(src_a)?;
    let src_b_typed = ctx.resolve_operand_typed(src_b)?;

    ctx.check_dst_type(&dst_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_a_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_b_typed, ty, instr_name)?;

    ctx.emit(
        LoweredInstr::BinOp {
            op,
            dst: dst_typed.reg,
            src_a: src_a_typed.operand,
            src_b: src_b_typed.operand,
            ty,
            clamp,
        },
        predicate,
    )?;
    Ok(())
}

fn lower_sub(
    ctx: &mut LoweringContext,
    sub: &SubInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    match sub {
        SubInstr::Integer {
            ty,
            dst,
            src_a,
            src_b,
        } => {
            check_not_packed(*ty, "sub")?;

            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, *ty, "sub")?;
            ctx.check_operand_type(&src_a_typed, *ty, "sub")?;
            ctx.check_operand_type(&src_b_typed, *ty, "sub")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Sub,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: *ty,
                    clamp: None,
                },
                predicate,
            )?;
        }
        SubInstr::IntegerSat {
            sat,
            dst,
            src_a,
            src_b,
        } => {
            if *sat {
                return Err(unsupported("sub.sat.s32", ".sat (saturating integer sub)"));
            }
            let ty = ScalarType::S32;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, ty, "sub.sat")?;
            ctx.check_operand_type(&src_a_typed, ty, "sub.sat")?;
            ctx.check_operand_type(&src_b_typed, ty, "sub.sat")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Sub,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty,
                    clamp: None,
                },
                predicate,
            )?;
        }
        SubInstr::Float32 {
            // Rounding mode and subnormal flushing don't apply over the reals.
            rnd: _rnd,
            ftz: _ftz,
            sat,
            dst,
            src_a,
            src_b,
        } => {
            let ty = ScalarType::F32;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, ty, "sub.f32")?;
            ctx.check_operand_type(&src_a_typed, ty, "sub.f32")?;
            ctx.check_operand_type(&src_b_typed, ty, "sub.f32")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Sub,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty,
                    clamp: sat.then_some(Clamp::Sat),
                },
                predicate,
            )?;
        }
        SubInstr::Float32x2 { .. } => {
            return Err(unsupported("sub.f32x2", "packed SIMD arithmetic on F32x2"));
        }
        SubInstr::Float64 {
            // Rounding mode doesn't apply over the reals.
            rnd: _rnd,
            dst,
            src_a,
            src_b,
        } => {
            let ty = ScalarType::F64;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, ty, "sub.f64")?;
            ctx.check_operand_type(&src_a_typed, ty, "sub.f64")?;
            ctx.check_operand_type(&src_b_typed, ty, "sub.f64")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Sub,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty,
                    clamp: None,
                },
                predicate,
            )?;
        }
        SubInstr::HalfF16 {
            // Rounding mode and subnormal flushing don't apply over the
            // reals; `.sat` is an exact value clamp.
            rnd: _rnd,
            ftz: _ftz,
            sat,
            ty,
            dst,
            src_a,
            src_b,
        } => {
            check_not_packed(*ty, "sub")?;
            lower_half_binop(
                ctx,
                BinOp::Sub,
                "sub.f16",
                *ty,
                sat.then_some(Clamp::Sat),
                dst,
                src_a,
                src_b,
                predicate,
            )?;
        }
        SubInstr::HalfBf16 {
            // Rounding mode doesn't apply over the reals.
            rnd: _rnd,
            ty,
            dst,
            src_a,
            src_b,
        } => {
            check_not_packed(*ty, "sub")?;
            lower_half_binop(
                ctx,
                BinOp::Sub,
                "sub.bf16",
                *ty,
                None,
                dst,
                src_a,
                src_b,
                predicate,
            )?;
        }
        SubInstr::MixedPrecision {
            // Rounding mode doesn't apply over the reals.
            rnd: _rnd,
            sat,
            src_type,
            dst,
            src_a,
            src_b,
        } => {
            if *sat {
                return Err(unsupported("sub.f32 (mixed)", ".sat modifier"));
            }
            check_not_packed(*src_type, "sub (mixed)")?;

            // Mixed precision sub: f32 result from half (f16/bf16) inputs
            let dst_ty = ScalarType::F32;
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, dst_ty, "sub.f32 (mixed)")?;
            ctx.check_operand_type(&src_a_typed, *src_type, "sub.f32 (mixed)")?;
            ctx.check_operand_type(&src_b_typed, *src_type, "sub.f32 (mixed)")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Sub,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: dst_ty,
                    clamp: None,
                },
                predicate,
            )?;
        }
    }
    Ok(())
}

fn lower_mul(
    ctx: &mut LoweringContext,
    mul: &MulInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    match mul {
        MulInstr::Integer {
            mode,
            ty,
            dst,
            src_a,
            src_b,
        } => {
            check_not_packed(*ty, "mul")?;

            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            match mode {
                MulMode::Wide => {
                    // For mul.wide, sources are type `ty`, but destination is 2x wider
                    // e.g., mul.wide.s32 produces s64 result
                    let dst_ty = ty.widen().unwrap_or(*ty);
                    ctx.check_dst_type(&dst_typed, dst_ty, "mul.wide")?;
                    ctx.check_operand_type(&src_a_typed, *ty, "mul.wide")?;
                    ctx.check_operand_type(&src_b_typed, *ty, "mul.wide")?;

                    ctx.emit(
                        LoweredInstr::MulWide {
                            dst: dst_typed.reg,
                            src_a: src_a_typed.operand,
                            src_b: src_b_typed.operand,
                            src_ty: *ty,
                        },
                        predicate,
                    )?;
                }
                MulMode::Hi => {
                    ctx.check_dst_type(&dst_typed, *ty, "mul.hi")?;
                    ctx.check_operand_type(&src_a_typed, *ty, "mul.hi")?;
                    ctx.check_operand_type(&src_b_typed, *ty, "mul.hi")?;

                    ctx.emit(
                        LoweredInstr::MulHi {
                            dst: dst_typed.reg,
                            src_a: src_a_typed.operand,
                            src_b: src_b_typed.operand,
                            ty: *ty,
                        },
                        predicate,
                    )?;
                }
                MulMode::Lo => {
                    ctx.check_dst_type(&dst_typed, *ty, "mul.lo")?;
                    ctx.check_operand_type(&src_a_typed, *ty, "mul.lo")?;
                    ctx.check_operand_type(&src_b_typed, *ty, "mul.lo")?;

                    ctx.emit(
                        LoweredInstr::BinOp {
                            op: BinOp::Mul,
                            dst: dst_typed.reg,
                            src_a: src_a_typed.operand,
                            src_b: src_b_typed.operand,
                            ty: *ty,
                            clamp: None,
                        },
                        predicate,
                    )?;
                }
            }
        }
        MulInstr::Float {
            // Rounding mode and subnormal flushing don't apply over the
            // reals; `.sat` is an exact value clamp.
            rnd: _rnd,
            ftz: _ftz,
            sat,
            ty,
            dst,
            src_a,
            src_b,
        } => {
            // The ISA allows `.sat` on mul only for .f32 and .f16/.f16x2
            // (the instruction parser already enforces this; this guard is
            // fail-closed insurance for the variant's other types).
            if *sat && !matches!(ty, ScalarType::F32 | ScalarType::F16 | ScalarType::F16x2) {
                return Err(unsupported(
                    "mul.f",
                    format!(".sat modifier on {:?} (invalid PTX)", ty),
                ));
            }
            check_not_packed(*ty, "mul")?;

            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            ctx.check_dst_type(&dst_typed, *ty, "mul.f")?;
            ctx.check_operand_type(&src_a_typed, *ty, "mul.f")?;
            ctx.check_operand_type(&src_b_typed, *ty, "mul.f")?;

            ctx.emit(
                LoweredInstr::BinOp {
                    op: BinOp::Mul,
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: *ty,
                    clamp: sat.then_some(Clamp::Sat),
                },
                predicate,
            )?;
        }
    }
    Ok(())
}

fn lower_mad(
    ctx: &mut LoweringContext,
    mad: &MadInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    match mad {
        MadInstr::Integer {
            mode,
            sat,
            ty,
            dst,
            src_a,
            src_b,
            src_c,
        } => {
            if *sat {
                return Err(unsupported(
                    "mad.hi.sat.s32",
                    ".sat (saturating integer mad)",
                ));
            }
            check_not_packed(*ty, "mad")?;

            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;
            let src_c_typed = ctx.resolve_operand_typed(src_c)?;

            let instr_name = match mode {
                MulMode::Hi => "mad.hi",
                MulMode::Lo => "mad.lo",
                MulMode::Wide => "mad.wide",
            };

            // For mad.wide, destination is 2x wider
            let dst_ty = if matches!(mode, MulMode::Wide) {
                ty.widen().unwrap_or(*ty)
            } else {
                *ty
            };

            ctx.check_dst_type(&dst_typed, dst_ty, instr_name)?;
            ctx.check_operand_type(&src_a_typed, *ty, instr_name)?;
            ctx.check_operand_type(&src_b_typed, *ty, instr_name)?;
            // src_c is the accumulator, same type as destination
            ctx.check_operand_type(&src_c_typed, dst_ty, instr_name)?;

            ctx.emit(
                LoweredInstr::Mad {
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    src_c: src_c_typed.operand,
                    ty: *ty,
                    mode: ctx.convert_mul_mode(*mode),
                },
                predicate,
            )?;
        }
        MadInstr::Float {
            // Rounding mode and subnormal flushing don't apply over the reals.
            rnd: _rnd,
            ftz: _ftz,
            sat,
            ty,
            dst,
            src_a,
            src_b,
            src_c,
        } => {
            if *sat {
                return Err(unsupported("mad.f", ".sat modifier"));
            }
            check_not_packed(*ty, "mad")?;

            let dst_typed = ctx.resolve_dst_typed(dst)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;
            let src_c_typed = ctx.resolve_operand_typed(src_c)?;

            ctx.check_dst_type(&dst_typed, *ty, "mad.f")?;
            ctx.check_operand_type(&src_a_typed, *ty, "mad.f")?;
            ctx.check_operand_type(&src_b_typed, *ty, "mad.f")?;
            ctx.check_operand_type(&src_c_typed, *ty, "mad.f")?;

            ctx.emit(
                LoweredInstr::Fma {
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    src_c: src_c_typed.operand,
                    ty: *ty,
                    clamp: None,
                },
                predicate,
            )?;
        }
    }
    Ok(())
}

fn lower_fma(
    ctx: &mut LoweringContext,
    fma: &FmaInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    // In every accepted arm the rounding mode (and ftz, where present) is
    // ignorable: floats are reals here, so the fused multiply-add is exact.
    // `.sat`/`.relu` are exact value clamps, threaded through as `Clamp`.
    let (ty, src_ty, instr_name, clamp, dst, src_a, src_b, src_c) = match fma {
        FmaInstr::Float32 {
            rnd: _rnd,
            ftz: _ftz,
            sat,
            dst,
            src_a,
            src_b,
            src_c,
        } => {
            let ty = ScalarType::F32;
            let clamp = sat.then_some(Clamp::Sat);
            (ty, ty, "fma.rn.f32", clamp, dst, src_a, src_b, src_c)
        }
        FmaInstr::Float32x2 { .. } => {
            return Err(unsupported(
                "fma.rn.f32x2",
                "packed SIMD arithmetic on F32x2",
            ));
        }
        FmaInstr::Float64 {
            rnd: _rnd,
            dst,
            src_a,
            src_b,
            src_c,
        } => {
            let ty = ScalarType::F64;
            (ty, ty, "fma.rn.f64", None, dst, src_a, src_b, src_c)
        }
        FmaInstr::HalfF16Sat {
            rnd: _rnd,
            ftz: _ftz,
            sat,
            ty,
            dst,
            src_a,
            src_b,
            src_c,
        } => {
            check_not_packed(*ty, "fma")?;
            let clamp = sat.then_some(Clamp::Sat);
            (*ty, *ty, "fma.rn.f16", clamp, dst, src_a, src_b, src_c)
        }
        FmaInstr::HalfF16Relu {
            rnd: _rnd,
            ftz: _ftz,
            ty,
            dst,
            src_a,
            src_b,
            src_c,
        } => {
            check_not_packed(*ty, "fma")?;
            let clamp = Some(Clamp::Relu);
            (*ty, *ty, "fma.rn.relu.f16", clamp, dst, src_a, src_b, src_c)
        }
        FmaInstr::HalfBf16 {
            rnd: _rnd,
            relu,
            ty,
            dst,
            src_a,
            src_b,
            src_c,
        } => {
            if *relu {
                return Err(unsupported("fma.rn.bf16", ".relu modifier"));
            }
            check_not_packed(*ty, "fma")?;
            (*ty, *ty, "fma.rn.bf16", None, dst, src_a, src_b, src_c)
        }
        FmaInstr::Oob { .. } => {
            return Err(unsupported("fma.rn.oob", ".oob modifier"));
        }
        FmaInstr::MixedPrecision {
            rnd: _rnd,
            sat,
            src_type,
            dst,
            src_a,
            src_b,
            src_c,
        } => {
            if *sat {
                return Err(unsupported("fma.rn.f32 (mixed)", ".sat modifier"));
            }
            check_not_packed(*src_type, "fma (mixed)")?;
            (
                ScalarType::F32,
                *src_type,
                "fma.rn.f32 (mixed)",
                None,
                dst,
                src_a,
                src_b,
                src_c,
            )
        }
    };

    let dst_typed = ctx.resolve_dst_typed(dst)?;
    let src_a_typed = ctx.resolve_operand_typed(src_a)?;
    let src_b_typed = ctx.resolve_operand_typed(src_b)?;
    let src_c_typed = ctx.resolve_operand_typed(src_c)?;

    // For mixed precision FMA, the multiplicands are half precision while
    // the accumulator and destination are f32; otherwise all agree with ty.
    ctx.check_dst_type(&dst_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_a_typed, src_ty, instr_name)?;
    ctx.check_operand_type(&src_b_typed, src_ty, instr_name)?;
    ctx.check_operand_type(&src_c_typed, ty, instr_name)?;

    ctx.emit(
        LoweredInstr::Fma {
            dst: dst_typed.reg,
            src_a: src_a_typed.operand,
            src_b: src_b_typed.operand,
            src_c: src_c_typed.operand,
            ty,
            clamp,
        },
        predicate,
    )?;
    Ok(())
}

fn lower_div(
    ctx: &mut LoweringContext,
    div: &DivInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    // rnd/ftz and the approx/full/ieee precision split are ignorable: floats
    // are reals here, so every division is exact.
    let (ty, instr_name, dst, src_a, src_b) = match div {
        DivInstr::Integer {
            ty,
            dst,
            src_a,
            src_b,
        } => (*ty, "div", dst, src_a, src_b),
        DivInstr::Approx {
            ftz: _ftz,
            dst,
            src_a,
            src_b,
        } => (ScalarType::F32, "div.approx.f32", dst, src_a, src_b),
        DivInstr::Full {
            ftz: _ftz,
            dst,
            src_a,
            src_b,
        } => (ScalarType::F32, "div.full.f32", dst, src_a, src_b),
        DivInstr::Ieee {
            rnd: _rnd,
            ftz: _ftz,
            ty,
            dst,
            src_a,
            src_b,
        } => (*ty, "div.rn", dst, src_a, src_b),
    };

    let dst_typed = ctx.resolve_dst_typed(dst)?;
    let src_a_typed = ctx.resolve_operand_typed(src_a)?;
    let src_b_typed = ctx.resolve_operand_typed(src_b)?;

    ctx.check_dst_type(&dst_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_a_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_b_typed, ty, instr_name)?;

    ctx.emit(
        LoweredInstr::BinOp {
            op: BinOp::Div,
            dst: dst_typed.reg,
            src_a: src_a_typed.operand,
            src_b: src_b_typed.operand,
            ty,
            clamp: None,
        },
        predicate,
    )?;
    Ok(())
}

fn lower_neg(
    ctx: &mut LoweringContext,
    neg: &NegInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    // ftz is ignorable: subnormal flushing doesn't apply over the reals.
    let (ty, instr_name, dst, src) = match neg {
        NegInstr::Integer { ty, dst, src } => (*ty, "neg", dst, src),
        NegInstr::Float32 {
            ftz: _ftz,
            dst,
            src,
        } => (ScalarType::F32, "neg.f32", dst, src),
        NegInstr::Float64 { dst, src } => (ScalarType::F64, "neg.f64", dst, src),
        NegInstr::HalfF16 {
            ftz: _ftz,
            ty,
            dst,
            src,
        } => (*ty, "neg.f16", dst, src),
        NegInstr::HalfBf16 { ty, dst, src } => (*ty, "neg.bf16", dst, src),
    };
    check_not_packed(ty, instr_name)?;

    let dst_typed = ctx.resolve_dst_typed(dst)?;
    let src_typed = ctx.resolve_operand_typed(src)?;

    ctx.check_dst_type(&dst_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_typed, ty, instr_name)?;

    ctx.emit(
        LoweredInstr::UnaryOp {
            op: UnaryOp::Neg,
            dst: dst_typed.reg,
            src: src_typed.operand,
            ty,
        },
        predicate,
    )?;
    Ok(())
}

fn lower_abs(
    ctx: &mut LoweringContext,
    abs: &AbsInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    // ftz is ignorable: subnormal flushing doesn't apply over the reals.
    let (ty, instr_name, dst, src) = match abs {
        AbsInstr::Integer { ty, dst, src } => (*ty, "abs", dst, src),
        AbsInstr::Float32 {
            ftz: _ftz,
            dst,
            src,
        } => (ScalarType::F32, "abs.f32", dst, src),
        AbsInstr::Float64 { dst, src } => (ScalarType::F64, "abs.f64", dst, src),
        AbsInstr::HalfF16 {
            ftz: _ftz,
            ty,
            dst,
            src,
        } => (*ty, "abs.f16", dst, src),
        AbsInstr::HalfBf16 { ty, dst, src } => (*ty, "abs.bf16", dst, src),
    };
    check_not_packed(ty, instr_name)?;

    let dst_typed = ctx.resolve_dst_typed(dst)?;
    let src_typed = ctx.resolve_operand_typed(src)?;

    ctx.check_dst_type(&dst_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_typed, ty, instr_name)?;

    ctx.emit(
        LoweredInstr::UnaryOp {
            op: UnaryOp::Abs,
            dst: dst_typed.reg,
            src: src_typed.operand,
            ty,
        },
        predicate,
    )?;
    Ok(())
}

/// Shared lowering for float unary ops (rcp, sqrt, rsqrt, ...)
fn lower_float_unary(
    ctx: &mut LoweringContext,
    op: UnaryOp,
    instr_name: &str,
    ty: ScalarType,
    dst: &AstOperand,
    src: &AstOperand,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let dst_typed = ctx.resolve_dst_typed(dst)?;
    let src_typed = ctx.resolve_operand_typed(src)?;

    ctx.check_dst_type(&dst_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_typed, ty, instr_name)?;

    ctx.emit(
        LoweredInstr::UnaryOp {
            op,
            dst: dst_typed.reg,
            src: src_typed.operand,
            ty,
        },
        predicate,
    )?;
    Ok(())
}

/// Reject the float min/max modifiers we do not model. `.NaN` changes the
/// NaN-propagation contract (meaningless over the reals but a semantic claim
/// nonetheless), `.xorsign.abs`/`.abs` change the computed value outright.
fn reject_minmax_modifiers(
    instruction: &str,
    nan: bool,
    xorsign_abs: bool,
    abs: bool,
) -> LowerResult<()> {
    if nan {
        return Err(unsupported(instruction, ".NaN modifier"));
    }
    if xorsign_abs {
        return Err(unsupported(instruction, ".xorsign.abs modifier"));
    }
    if abs {
        return Err(unsupported(instruction, ".abs modifier"));
    }
    Ok(())
}

fn lower_min(
    ctx: &mut LoweringContext,
    min: &MinInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let (ty, instr_name, dst, src_a, src_b) = match min {
        MinInstr::Integer {
            ty,
            dst,
            src_a,
            src_b,
        } => (*ty, "min", dst, src_a, src_b),
        MinInstr::IntegerRelu {
            relu,
            ty,
            dst,
            src_a,
            src_b,
        } => {
            if *relu {
                return Err(unsupported("min", ".relu modifier"));
            }
            (*ty, "min", dst, src_a, src_b)
        }
        MinInstr::Float32 {
            // ftz is ignorable: subnormal flushing doesn't apply over the reals.
            ftz: _ftz,
            nan,
            xorsign_abs,
            abs,
            dst,
            src_a,
            src_b,
        } => {
            reject_minmax_modifiers("min.f32", *nan, *xorsign_abs, *abs)?;
            (ScalarType::F32, "min.f32", dst, src_a, src_b)
        }
        MinInstr::Float32Acc { .. } => {
            return Err(unsupported("min.f32", "3-input min"));
        }
        MinInstr::Float64 { dst, src_a, src_b } => (ScalarType::F64, "min.f64", dst, src_a, src_b),
        MinInstr::HalfF16 {
            // ftz is ignorable: subnormal flushing doesn't apply over the reals.
            ftz: _ftz,
            nan,
            xorsign_abs,
            abs,
            ty,
            dst,
            src_a,
            src_b,
        } => {
            reject_minmax_modifiers("min.f16", *nan, *xorsign_abs, *abs)?;
            (*ty, "min.f16", dst, src_a, src_b)
        }
        MinInstr::HalfBf16 {
            nan,
            xorsign_abs,
            abs,
            ty,
            dst,
            src_a,
            src_b,
        } => {
            reject_minmax_modifiers("min.bf16", *nan, *xorsign_abs, *abs)?;
            (*ty, "min.bf16", dst, src_a, src_b)
        }
    };
    check_not_packed(ty, instr_name)?;

    let dst_typed = ctx.resolve_dst_typed(dst)?;
    let src_a_typed = ctx.resolve_operand_typed(src_a)?;
    let src_b_typed = ctx.resolve_operand_typed(src_b)?;

    ctx.check_dst_type(&dst_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_a_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_b_typed, ty, instr_name)?;

    ctx.emit(
        LoweredInstr::BinOp {
            op: BinOp::Min,
            dst: dst_typed.reg,
            src_a: src_a_typed.operand,
            src_b: src_b_typed.operand,
            ty,
            clamp: None,
        },
        predicate,
    )?;
    Ok(())
}

fn lower_max(
    ctx: &mut LoweringContext,
    max: &MaxInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let (ty, instr_name, dst, src_a, src_b) = match max {
        MaxInstr::Integer {
            ty,
            dst,
            src_a,
            src_b,
        } => (*ty, "max", dst, src_a, src_b),
        MaxInstr::IntegerRelu {
            relu,
            ty,
            dst,
            src_a,
            src_b,
        } => {
            if *relu {
                return Err(unsupported("max", ".relu modifier"));
            }
            (*ty, "max", dst, src_a, src_b)
        }
        MaxInstr::Float32 {
            // ftz is ignorable: subnormal flushing doesn't apply over the reals.
            ftz: _ftz,
            nan,
            xorsign_abs,
            abs,
            dst,
            src_a,
            src_b,
        } => {
            reject_minmax_modifiers("max.f32", *nan, *xorsign_abs, *abs)?;
            (ScalarType::F32, "max.f32", dst, src_a, src_b)
        }
        MaxInstr::Float32Acc { .. } => {
            return Err(unsupported("max.f32", "3-input max"));
        }
        MaxInstr::Float64 { dst, src_a, src_b } => (ScalarType::F64, "max.f64", dst, src_a, src_b),
        MaxInstr::HalfF16 {
            // ftz is ignorable: subnormal flushing doesn't apply over the reals.
            ftz: _ftz,
            nan,
            xorsign_abs,
            abs,
            ty,
            dst,
            src_a,
            src_b,
        } => {
            reject_minmax_modifiers("max.f16", *nan, *xorsign_abs, *abs)?;
            (*ty, "max.f16", dst, src_a, src_b)
        }
        MaxInstr::HalfBf16 {
            nan,
            xorsign_abs,
            abs,
            ty,
            dst,
            src_a,
            src_b,
        } => {
            reject_minmax_modifiers("max.bf16", *nan, *xorsign_abs, *abs)?;
            (*ty, "max.bf16", dst, src_a, src_b)
        }
    };
    check_not_packed(ty, instr_name)?;

    let dst_typed = ctx.resolve_dst_typed(dst)?;
    let src_a_typed = ctx.resolve_operand_typed(src_a)?;
    let src_b_typed = ctx.resolve_operand_typed(src_b)?;

    ctx.check_dst_type(&dst_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_a_typed, ty, instr_name)?;
    ctx.check_operand_type(&src_b_typed, ty, instr_name)?;

    ctx.emit(
        LoweredInstr::BinOp {
            op: BinOp::Max,
            dst: dst_typed.reg,
            src_a: src_a_typed.operand,
            src_b: src_b_typed.operand,
            ty,
            clamp: None,
        },
        predicate,
    )?;
    Ok(())
}

fn lower_setp(
    ctx: &mut LoweringContext,
    setp: &SetpInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    match setp {
        SetpInstr::Simple {
            cmp_op,
            // ftz is ignorable: subnormal flushing doesn't apply over the reals.
            ftz: _ftz,
            ty,
            dst_p,
            dst_q,
            src_a,
            src_b,
        } => {
            if dst_q.is_some() {
                return Err(unsupported("setp", "dual predicate destination p|q"));
            }

            // Destination is a predicate register
            let dst_typed = ctx.resolve_dst_typed(dst_p)?;
            let src_a_typed = ctx.resolve_operand_typed(src_a)?;
            let src_b_typed = ctx.resolve_operand_typed(src_b)?;

            // Check destination is pred type
            ctx.check_dst_type(&dst_typed, ScalarType::Pred, "setp")?;
            // Check sources match instruction type
            ctx.check_operand_type(&src_a_typed, *ty, "setp")?;
            ctx.check_operand_type(&src_b_typed, *ty, "setp")?;

            ctx.emit(
                LoweredInstr::Setp {
                    cmp: ctx.convert_cmp_op(*cmp_op),
                    dst: dst_typed.reg,
                    src_a: src_a_typed.operand,
                    src_b: src_b_typed.operand,
                    ty: *ty,
                },
                predicate,
            )?;
        }
        SetpInstr::WithBoolOp { .. } => {
            return Err(unsupported(
                "setp",
                "boolean-combine modifier (.and/.or/.xor)",
            ));
        }
    }
    Ok(())
}

/// Check the ordering/system qualifiers shared by `ld` and `st`. `.weak` is
/// the default and is what the interpreter models. `.volatile` is more than
/// an optimization barrier: ISA §8.4.2 makes it equivalent to a relaxed
/// system-scope - i.e. morally strong - operation, so there are volatile
/// handshakes that are race-free under the spec. Volta deliberately treats
/// every access as weak (the paper's race model), which can only ADD
/// reported races on such programs, never hide one - the conservative
/// direction for a race checker. Everything else is an unmodeled
/// synchronization/system semantic.
fn check_mem_access_qualifiers(
    instruction: &str,
    semantics: MemSemantics,
    scope: Option<ast::MemScope>,
    space_qualifier: Option<ast::StateSpaceQualifier>,
    mmio: bool,
) -> LowerResult<()> {
    match semantics {
        MemSemantics::Weak | MemSemantics::Volatile => {}
        MemSemantics::Relaxed | MemSemantics::Acquire | MemSemantics::Release => {
            return Err(unsupported(
                instruction,
                format!(
                    "memory-ordering qualifier {:?} (atomic protocols are not modeled)",
                    semantics
                ),
            ));
        }
    }
    if let Some(scope) = scope {
        return Err(unsupported(
            instruction,
            format!("memory scope qualifier {:?}", scope),
        ));
    }
    if let Some(q) = space_qualifier {
        return Err(unsupported(
            instruction,
            format!("::-qualified state space ({:?})", q),
        ));
    }
    if mmio {
        return Err(unsupported(instruction, ".mmio modifier"));
    }
    Ok(())
}

/// Check that the `.vN` width modifier and the operand shape agree; the
/// emitted access width is taken from the operand's register list, so a
/// mismatch would silently access the wrong number of elements.
fn check_vec_arity(
    instruction: &str,
    vec: Option<VecWidth>,
    operand: &AstOperand,
) -> LowerResult<()> {
    let ok = match (vec, operand) {
        (Some(v), AstOperand::Vector(elements)) => v.count() as usize == elements.len(),
        (Some(_), _) => false,
        (None, AstOperand::Vector(_)) => false,
        (None, _) => true,
    };
    if ok {
        Ok(())
    } else {
        Err(LowerError::InvalidOperand {
            instruction: instruction.to_string(),
            operand: format!("{:?}", operand),
            reason: "the .vN width modifier must match the vector operand's register count",
        })
    }
}

fn lower_load(
    ctx: &mut LoweringContext,
    ld: &LdInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    // Destructure exhaustively so a new field cannot be dropped silently.
    let LdInstr {
        semantics,
        scope,
        space,
        space_qualifier,
        // Cache-behavior hints only; the loaded value is unaffected.
        cache_op: _cache_op,
        vec,
        // Non-coherent load: the memory consistency model does not apply
        // to ld.global.nc, so hardware may return stale data if the
        // kernel violates the read-only contract (nvcc emits .nc only for
        // const __restrict__ data). Volta returns the fresh value; a
        // kernel that breaks the contract is a garbage-in case.
        nc: _nc,
        mmio,
        unified,
        ty,
        dst,
        addr,
    } = ld;
    check_mem_access_qualifiers("ld", *semantics, *scope, *space_qualifier, *mmio)?;
    if *unified {
        return Err(unsupported("ld", ".unified modifier"));
    }

    // Param-space loads read either a kernel parameter or a block-scope
    // `.param` slot (the callseq idiom); both are handled symbolically
    // rather than as memory accesses.
    if *space == Some(StateSpace::Param) {
        return lower_param_load(ctx, ld, predicate);
    }

    // Resolve the address operand
    let (base, offset) = ctx.resolve_mem_operand(addr)?;

    let space = ctx.convert_space(*space, "ld")?;
    let instr_name = format!("ld.{:?}", ty);
    check_vec_arity("ld", *vec, dst)?;

    // Check if destination is a vector (e.g., {%f1, %f2, %f3, %f4})
    match dst {
        AstOperand::Vector(_) => {
            // Vector load - emit LoadVec
            // For vectors, we'd need to check each element, but for now just resolve
            let dst_regs = ctx.resolve_dst_vector(dst)?;
            ctx.emit(
                LoweredInstr::LoadVec {
                    dst: dst_regs,
                    space,
                    base,
                    offset,
                    ty: *ty,
                },
                predicate,
            )?;
        }
        _ => {
            // Single register load - use relaxed type checking (PTX 9.4.1)
            // Destination can be wider than instruction type (value is extended)
            let dst_typed = ctx.resolve_dst_typed(dst)?;
            ctx.check_dst_type_relaxed(&dst_typed, *ty, &instr_name)?;

            ctx.emit(
                LoweredInstr::Load {
                    dst: dst_typed.reg,
                    space,
                    base,
                    offset,
                    ty: *ty,
                },
                predicate,
            )?;
        }
    }
    Ok(())
}

fn lower_store(
    ctx: &mut LoweringContext,
    st: &StInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    // Destructure exhaustively so a new field cannot be dropped silently.
    let StInstr {
        semantics,
        scope,
        space,
        space_qualifier,
        // Cache-behavior hints only; the stored value is unaffected.
        cache_op: _cache_op,
        vec,
        mmio,
        ty,
        addr,
        src,
    } = st;
    check_mem_access_qualifiers("st", *semantics, *scope, *space_qualifier, *mmio)?;

    // Param-space stores write a block-scope `.param` slot (callseq idiom)
    if *space == Some(StateSpace::Param) {
        return lower_param_store(ctx, st, predicate);
    }

    // Resolve the address operand
    let (base, offset) = ctx.resolve_mem_operand(addr)?;

    let space = ctx.convert_space(*space, "st")?;
    let instr_name = format!("st.{:?}", ty);
    check_vec_arity("st", *vec, src)?;

    // Check if source is a vector (e.g., {%f1, %f2, %f3, %f4})
    if let AstOperand::Vector(_) = src {
        // Vector store - emit StoreVec
        // For vectors, we'd need to check each element, but for now just resolve
        let src_regs = ctx.resolve_dst_vector(src)?;
        ctx.emit(
            LoweredInstr::StoreVec {
                space,
                base,
                offset,
                src: src_regs,
                ty: *ty,
            },
            predicate,
        )?;
        return Ok(());
    }

    // Single register store - use relaxed type checking (PTX 9.4.1)
    // Source can be wider than instruction type (value is truncated)
    let src_typed = ctx.resolve_operand_typed(src)?;
    ctx.check_operand_type_relaxed(&src_typed, *ty, &instr_name)?;

    ctx.emit(
        LoweredInstr::Store {
            space,
            base,
            offset,
            src: src_typed.operand,
            ty: *ty,
        },
        predicate,
    )?;
    Ok(())
}

/// Extract the `(symbol, offset)` of a param-space address like `[param0+0]`.
fn param_addr_symbol(ctx: &LoweringContext, addr: &AstOperand) -> LowerResult<(String, i64)> {
    if let AstOperand::Address(addr) = addr
        && let AddressBase::Symbol(name) = &addr.base
    {
        return Ok((name.to_string(), ctx.get_address_offset(addr)?));
    }
    Err(LowerError::UnsupportedInstruction {
        instruction: "param-space access".to_string(),
        reason: Some("param accesses must use a symbol base address".to_string()),
    })
}

/// Lower `ld.param`: kernel parameters become `LoadParam`; block-scope
/// `.param` slots (callseq idiom) collapse to the deferred call result.
fn lower_param_load(
    ctx: &mut LoweringContext,
    ld: &LdInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let (name, offset) = param_addr_symbol(ctx, &ld.addr)?;
    let dst = ctx.resolve_dst(&ld.dst)?;

    // Block-scope `.param` slot (callseq idiom)?
    if let Some(slot) = ctx.local_params.get(&name).copied() {
        if offset != 0 {
            return Err(LowerError::UnsupportedInstruction {
                instruction: format!("ld.param [{}+{}]", name, offset),
                reason: Some(".param slots are scalar; nonzero offsets unsupported".to_string()),
            });
        }
        if predicate.is_some() {
            return Err(LowerError::UnsupportedInstruction {
                instruction: format!("ld.param [{}]", name),
                reason: Some("predicated callseq accesses are not supported".to_string()),
            });
        }
        return match slot {
            LocalParamSlot::PendingExp(src) => ctx.emit(
                LoweredInstr::UnaryOp {
                    op: UnaryOp::Exp,
                    dst,
                    src,
                    ty: ld.ty,
                },
                None,
            ),
            LocalParamSlot::Stored(src) => ctx.emit(
                LoweredInstr::Mov {
                    dst,
                    src,
                    ty: ld.ty,
                },
                None,
            ),
            LocalParamSlot::Empty => Err(LowerError::UnsupportedInstruction {
                instruction: format!("ld.param [{}]", name),
                reason: Some("read of a .param slot that was never written".to_string()),
            }),
        };
    }

    // Kernel parameter
    let Some(info) = ctx.symbols.get_param(&name) else {
        return Err(LowerError::UndefinedSymbol { name });
    };
    let param_id = info.id;
    if offset != 0 {
        return Err(LowerError::UnsupportedInstruction {
            instruction: format!("ld.param [{}+{}]", name, offset),
            reason: Some("param loads with nonzero offsets are not supported".to_string()),
        });
    }
    ctx.emit(LoweredInstr::LoadParam { dst, param_id }, predicate)
}

/// Lower `st.param`: records the stored operand in a block-scope `.param`
/// slot for the following `call`. Emits no instruction.
fn lower_param_store(
    ctx: &mut LoweringContext,
    st: &StInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    if predicate.is_some() {
        return Err(LowerError::UnsupportedInstruction {
            instruction: "st.param".to_string(),
            reason: Some("predicated callseq accesses are not supported".to_string()),
        });
    }
    let (name, offset) = param_addr_symbol(ctx, &st.addr)?;
    if offset != 0 {
        return Err(LowerError::UnsupportedInstruction {
            instruction: format!("st.param [{}+{}]", name, offset),
            reason: Some(".param slots are scalar; nonzero offsets unsupported".to_string()),
        });
    }
    if !ctx.local_params.contains_key(&name) {
        return Err(LowerError::UndefinedSymbol { name });
    }
    let src = ctx.resolve_operand(&st.src)?;
    ctx.local_params.insert(name, LocalParamSlot::Stored(src));
    Ok(())
}

/// Lower a direct call. Only `__symexpf` (the paper's hook for symbolic exp)
/// is supported. The call emits nothing; it marks the retval slot as the
/// pending exp of the argument, which the consuming `ld.param` materializes.
fn lower_call(
    ctx: &mut LoweringContext,
    call: &CallInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    // .uni is a divergence hint; it does not change what the call does.
    let CallInstr {
        uniform: _uniform,
        return_operands,
        target,
        arguments,
    } = call;
    let target_name = match target {
        AstOperand::Ident(name) => name.to_string(),
        other => {
            return Err(LowerError::UnsupportedInstruction {
                instruction: format!("call via {:?}", other),
                reason: Some("only direct calls are supported".to_string()),
            });
        }
    };
    if target_name != "__symexpf" {
        return Err(LowerError::UnsupportedInstruction {
            instruction: format!("call {}", target_name),
            reason: Some("only calls to the __symexpf symbolic-exp hook are supported".to_string()),
        });
    }
    if predicate.is_some() {
        return Err(LowerError::UnsupportedInstruction {
            instruction: format!("call {}", target_name),
            reason: Some("predicated calls are not supported".to_string()),
        });
    }

    let [arg] = arguments.as_slice() else {
        return Err(LowerError::UnsupportedInstruction {
            instruction: format!("call {}", target_name),
            reason: Some("__symexpf takes exactly one argument".to_string()),
        });
    };
    let [ret] = return_operands.as_slice() else {
        return Err(LowerError::UnsupportedInstruction {
            instruction: format!("call {}", target_name),
            reason: Some("__symexpf returns exactly one value".to_string()),
        });
    };
    let (AstOperand::Ident(arg_name), AstOperand::Ident(ret_name)) = (arg, ret) else {
        return Err(LowerError::UnsupportedInstruction {
            instruction: format!("call {}", target_name),
            reason: Some("call operands must be .param slot names".to_string()),
        });
    };

    let arg_name = arg_name.to_string();
    let src = match ctx.local_params.get(&arg_name) {
        Some(LocalParamSlot::Stored(op)) => *op,
        Some(_) => {
            return Err(LowerError::UnsupportedInstruction {
                instruction: format!("call {}", target_name),
                reason: Some(format!(
                    ".param slot '{}' was not written before the call",
                    arg_name
                )),
            });
        }
        None => return Err(LowerError::UndefinedSymbol { name: arg_name }),
    };

    let ret_name = ret_name.to_string();
    if !ctx.local_params.contains_key(&ret_name) {
        return Err(LowerError::UndefinedSymbol { name: ret_name });
    }
    ctx.local_params
        .insert(ret_name, LocalParamSlot::PendingExp(src));
    Ok(())
}

fn lower_cvt(
    ctx: &mut LoweringContext,
    cvt: &CvtInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    match cvt {
        CvtInstr::Standard {
            rnd,
            // ftz is ignorable: subnormal flushing doesn't apply over the reals.
            ftz: _ftz,
            sat,
            relu,
            satfinite,
            dst_type,
            src_type,
            dst,
            src,
        } => {
            if *satfinite {
                // .satfinite clamps to the destination format's finite range
                // - a different, format-dependent semantics from the [0,1]
                // value clamp; not modeled.
                return Err(unsupported("cvt", ".satfinite modifier"));
            }
            // `.sat`/`.relu` on float->float conversions are exact value
            // clamps over the reals, threaded through as `Clamp`.
            // Integer-destination `.sat` (clamp to MININT..MAXINT) is a
            // different operation and stays rejected, as does anything
            // outside the ISA's legal destination types.
            let clamp = match (*sat, *relu) {
                (false, false) => None,
                (true, true) => {
                    return Err(unsupported("cvt", ".sat and .relu together (invalid PTX)"));
                }
                (true, false) => {
                    // ISA: the float `.sat` clamp applies to .f16/.f32/.f64
                    // destinations only.
                    if !is_scalar_float(*src_type)
                        || !matches!(
                            dst_type,
                            ScalarType::F16 | ScalarType::F32 | ScalarType::F64
                        )
                    {
                        return Err(unsupported(
                            "cvt",
                            ".sat modifier outside a float->float form with .f16/.f32/.f64 \
                             destination (integer saturation is not modeled)",
                        ));
                    }
                    Some(Clamp::Sat)
                }
                (false, true) => {
                    // ISA: the scalar `.relu` forms have .f16/.bf16
                    // destinations only.
                    if !is_scalar_float(*src_type)
                        || !matches!(dst_type, ScalarType::F16 | ScalarType::Bf16)
                    {
                        return Err(unsupported(
                            "cvt",
                            ".relu modifier outside a float->float form with .f16/.bf16 \
                             destination",
                        ));
                    }
                    Some(Clamp::Relu)
                }
            };
            match rnd {
                // Float rounding modes are ignorable: floats are reals here,
                // and float<->float conversion is the identity (the paper's
                // documented abstraction).
                None | Some(CvtRounding::Float(_)) => {}
                Some(CvtRounding::Integer(_)) => {
                    return Err(unsupported(
                        "cvt",
                        "integer-rounding modifier (floor/ceil/trunc/rint not modeled)",
                    ));
                }
                Some(CvtRounding::Stochastic) => {
                    return Err(unsupported("cvt", ".rs stochastic-rounding modifier"));
                }
                Some(CvtRounding::Rna) => {
                    return Err(unsupported("cvt", ".rna rounding modifier"));
                }
            }

            let dst = ctx.resolve_dst(dst)?;
            let src = ctx.resolve_operand(src)?;
            ctx.emit(
                LoweredInstr::Cvt {
                    dst,
                    src,
                    dst_ty: *dst_type,
                    src_ty: *src_type,
                    clamp,
                },
                predicate,
            )?;
        }
        CvtInstr::Pack { .. } => {
            return Err(unsupported(
                "cvt.pack",
                "pack conversion (two-source packing)",
            ));
        }
    }
    Ok(())
}

fn lower_shfl_sync(
    ctx: &mut LoweringContext,
    shfl: &ShflSyncInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    // The destination may be written `d|p` where `p` receives the validity
    // predicate; the parser surfaces that as a PredicatePair operand.
    let (dst, pair_pred) = match &shfl.dst {
        AstOperand::PredicatePair(d, p) => (
            ctx.resolve_register(&d.to_string())?,
            Some(ctx.resolve_register(&p.to_string())?),
        ),
        other => (ctx.resolve_dst(other)?, None),
    };
    let dst_pred = match (&shfl.dst_pred, pair_pred) {
        (Some(op), None) => Some(ctx.resolve_dst(op)?),
        (None, p) => p,
        (Some(_), Some(_)) => {
            return Err(LowerError::UnsupportedInstruction {
                instruction: "shfl.sync".to_string(),
                reason: Some("conflicting predicate destinations".to_string()),
            });
        }
    };
    let src = ctx.resolve_operand(&shfl.src)?;
    let offset_or_lane = ctx.resolve_operand(&shfl.src_b)?;
    let clamp = ctx.resolve_operand(&shfl.src_c)?;
    let membermask = ctx.resolve_operand(&shfl.membermask)?;

    ctx.emit(
        LoweredInstr::ShflSync {
            mode: ctx.convert_shfl_mode(shfl.mode),
            dst,
            dst_pred,
            src,
            offset_or_lane,
            clamp,
            membermask,
        },
        predicate,
    )?;
    Ok(())
}

fn lower_branch(
    ctx: &mut LoweringContext,
    bra: &BraInstr,
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    // .uni is a divergence hint; it does not change where the branch goes.
    let BraInstr {
        uniform: _uniform,
        target,
    } = bra;

    // Get target label
    let target_name = match target {
        AstOperand::Symbol(name) => name.to_string(),
        AstOperand::Ident(name) => name.to_string(), // Labels are parsed as identifiers
        _ => {
            return Err(LowerError::InvalidBranchTarget {
                target: format!("{:?}", target),
            });
        }
    };

    // Try to resolve the label
    let target = match ctx.symbols.resolve_label(&target_name) {
        Some(pc) => pc,
        None => {
            // Forward reference - record it and emit placeholder
            ctx.record_forward_ref(&target_name);
            InstrId::from_index(0) // Will be patched later
        }
    };

    ctx.emit(LoweredInstr::Bra { target }, predicate)?;
    Ok(())
}

/// Lower `bar.sync a` / `barrier.sync{.aligned} a` (both callers pass their
/// mode and operand list; the two spellings share these semantics).
fn lower_bar(
    ctx: &mut LoweringContext,
    mode: BarMode,
    operands: &[AstOperand],
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    match mode {
        BarMode::Sync => {}
        // bar.arrive does not block the arriving thread; emitting a blocking
        // sync in its place would invent synchronization that isn't there.
        BarMode::Arrive => return Err(unsupported("bar.arrive", "non-blocking barrier arrival")),
        // bar.red also produces a reduction value in its destination.
        BarMode::Red => return Err(unsupported("bar.red", "reduction barrier")),
    }

    // The barrier id must be a concrete immediate: barriers are identified
    // per-id at evaluation time, so a register id cannot be resolved here.
    let barrier_id = match operands {
        [] => {
            return Err(unsupported(
                "bar.sync",
                "missing barrier id operand (an immediate 0-15 is required)",
            ));
        }
        [id, ..] => match id {
            AstOperand::ImmInt(v) if (0..=15).contains(v) => *v as u32,
            AstOperand::ImmUInt(v) if *v <= 15 => *v as u32,
            AstOperand::ImmInt(_) | AstOperand::ImmUInt(_) => {
                return Err(LowerError::InvalidOperand {
                    instruction: "bar.sync".to_string(),
                    operand: format!("{:?}", id),
                    reason: "barrier id must be in 0-15",
                });
            }
            other => {
                return Err(unsupported(
                    "bar.sync",
                    format!("register barrier id ({:?})", other),
                ));
            }
        },
    };

    match operands.len() {
        1 => ctx.emit(LoweredInstr::BarSync { barrier_id }, predicate)?,
        // The partial-CTA counted form synchronizes only `b` threads; the
        // evaluator's barrier rule is full-CTA, so lowering it as a plain
        // sync would be wrong. Rejected here rather than at evaluation.
        2 => {
            return Err(unsupported(
                "bar.sync",
                "thread-count operand (bar.sync a, b)",
            ));
        }
        _ => {
            return Err(LowerError::InvalidOperand {
                instruction: "bar.sync".to_string(),
                operand: format!("{:?}", operands),
                reason: "bar.sync takes one barrier-id operand",
            });
        }
    }
    Ok(())
}

/// Lower ld.global.nc (non-coherent global load)
/// Treat as regular global load - we don't model cache coherence
fn lower_ld_global_nc(
    ctx: &mut LoweringContext,
    modifiers: &[DottedIdent],
    operands: &[AstOperand],
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    // Exhaustive modifier handling: anything we don't recognize is rejected
    // by name rather than skipped.
    let mut elem_type: Option<ScalarType> = None;
    let mut vec: Option<VecWidth> = None;

    for modifier in modifiers {
        if let DottedIdent::Qualified(parts) = modifier {
            // Cache performance hints (eviction priority, prefetch size)
            // are accepted and ignored - the same whitelist as plain
            // ld/st. Anything else qualified (notably L2::cache_hint,
            // which requires a cache-policy operand we do not parse) is
            // rejected by name.
            if is_cache_perf_hint(parts) {
                continue;
            }
            return Err(unsupported(
                "ld.global.nc",
                format!("modifier .{}", modifier),
            ));
        }
        let mod_ascii = modifier.to_ascii_string();
        match mod_ascii.as_bytes() {
            // Cache-operation hints: cache behavior only.
            b"ca" | b"cg" | b"cs" | b"lu" | b"cv" => {}
            _ => {
                if let Some(v) = VecWidth::from_ascii(&mod_ascii) {
                    vec = Some(v);
                } else if let Some(ty) = ScalarType::from_ascii(&mod_ascii) {
                    elem_type = Some(ty);
                } else {
                    return Err(unsupported(
                        "ld.global.nc",
                        format!("modifier .{}", modifier),
                    ));
                }
            }
        }
    }

    let elem_type =
        elem_type.ok_or_else(|| unsupported("ld.global.nc", "missing type modifier"))?;

    // Operands: [destination, address]
    if operands.len() != 2 {
        return Err(LowerError::InvalidOperand {
            instruction: "ld.global.nc".to_string(),
            operand: format!("{:?}", operands),
            reason: "expected destination and address operands",
        });
    }

    let dst_operand = &operands[0];
    let addr_operand = &operands[1];
    check_vec_arity("ld.global.nc", vec, dst_operand)?;

    // Resolve address
    let (base, offset) = ctx.resolve_mem_operand(addr_operand)?;

    // Check if destination is a vector
    match dst_operand {
        AstOperand::Vector(_) => {
            // Vector load
            let dst_regs = ctx.resolve_dst_vector(dst_operand)?;
            ctx.emit(
                LoweredInstr::LoadVec {
                    dst: dst_regs,
                    space: MemSpace::Global,
                    base,
                    offset,
                    ty: elem_type,
                },
                predicate,
            )?;
        }
        _ => {
            // Single register load
            let dst = ctx.resolve_dst(dst_operand)?;
            ctx.emit(
                LoweredInstr::Load {
                    dst,
                    space: MemSpace::Global,
                    base,
                    offset,
                    ty: elem_type,
                },
                predicate,
            )?;
        }
    }

    Ok(())
}

// =========================================================================
// Tensor Core Lowering Helpers
// =========================================================================

/// Parse a `ScalarType` from a modifier string (e.g., "f16", "f32", "b16").
fn parse_scalar_type_modifier(modifier: &DottedIdent) -> Option<ScalarType> {
    let ascii = modifier.to_ascii_string();
    ScalarType::from_ascii(&ascii)
}

/// Lower `ldmatrix.sync.aligned[.trans].x{1,2,4}.m8n8.shared.b16`
fn lower_ldmatrix(
    ctx: &mut LoweringContext,
    modifiers: &[DottedIdent],
    operands: &[AstOperand],
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let mut trans = false;
    let mut num: Option<u32> = None;

    for modifier in modifiers {
        let s = modifier.to_string();
        match s.as_str() {
            // The modeled form: ldmatrix.sync.aligned.x{1,2,4}[.trans].m8n8.shared.b16
            "sync" | "aligned" | "shared" | "m8n8" | "b16" => {}
            "trans" => trans = true,
            "x1" => num = Some(1),
            "x2" => num = Some(2),
            "x4" => num = Some(4),
            // Everything else (m8n16, m16n16, b8, shared::cta, dst/src
            // format types, ...) selects an unmodeled fragment layout.
            other => {
                return Err(unsupported("ldmatrix", format!("modifier .{}", other)));
            }
        }
    }

    let num = num.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "ldmatrix".to_string(),
        reason: Some("missing x1/x2/x4 modifier".to_string()),
    })?;

    if operands.len() < 2 {
        return Err(LowerError::InvalidOperand {
            instruction: "ldmatrix".to_string(),
            operand: format!("{:?}", operands),
            reason: "expected destination vector and address operands",
        });
    }

    let dst = ctx.resolve_dst_vector(&operands[0])?;
    let (base, offset) = ctx.resolve_mem_operand(&operands[1])?;

    ctx.emit(
        LoweredInstr::Ldmatrix {
            dst,
            base,
            offset,
            num,
            trans,
        },
        predicate,
    )?;
    Ok(())
}

/// Lower `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
fn lower_mma(
    ctx: &mut LoweringContext,
    modifiers: &[DottedIdent],
    operands: &[AstOperand],
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let mut shape: Option<MmaShape> = None;
    let mut layouts: Vec<MmaLayout> = Vec::new();
    let mut types: Vec<ScalarType> = Vec::new();

    for modifier in modifiers {
        let s = modifier.to_string();
        if s == "sync" || s == "aligned" {
            // The modeled execution mode.
        } else if let Some(sh) = MmaShape::parse(&s) {
            shape = Some(sh);
        } else if let Some(layout) = MmaLayout::parse(&s) {
            layouts.push(layout);
        } else if let Some(ty) = parse_scalar_type_modifier(modifier) {
            types.push(ty);
        } else {
            // Everything else (.sp sparsity, .satfinite, b1 ops, ...) is an
            // unmodeled variant.
            return Err(unsupported("mma", format!("modifier .{}", s)));
        }
    }

    let shape = shape.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "mma".to_string(),
        reason: Some("missing shape modifier (e.g., m16n8k16)".to_string()),
    })?;

    if layouts.len() != 2 {
        return Err(LowerError::UnsupportedInstruction {
            instruction: "mma".to_string(),
            reason: Some(format!(
                "expected 2 layout modifiers (row/col), found {}",
                layouts.len()
            )),
        });
    }

    if types.len() != 4 {
        return Err(LowerError::UnsupportedInstruction {
            instruction: "mma".to_string(),
            reason: Some(format!(
                "expected 4 type modifiers (d_type, a_type, b_type, c_type), found {}",
                types.len()
            )),
        });
    }

    // The evaluator gathers f16 multiplicand fragments and f32 accumulators;
    // any other type combination would be executed with the wrong layout.
    if types
        != [
            ScalarType::F32,
            ScalarType::F16,
            ScalarType::F16,
            ScalarType::F32,
        ]
    {
        return Err(unsupported(
            "mma",
            format!(
                "type combination {:?} (only .f32.f16.f16.f32 is modeled)",
                types
            ),
        ));
    }

    if operands.len() < 4 {
        return Err(LowerError::InvalidOperand {
            instruction: "mma".to_string(),
            operand: format!("{:?}", operands),
            reason: "expected 4 vector operands (dst, src_a, src_b, src_c)",
        });
    }

    let dst = ctx.resolve_dst_vector(&operands[0])?;
    let src_a = ctx.resolve_dst_vector(&operands[1])?;
    let src_b = ctx.resolve_dst_vector(&operands[2])?;
    let src_c = ctx.resolve_dst_vector(&operands[3])?;

    let a_layout = layouts[0];
    let b_layout = layouts[1];
    // PTX syntax order: d_type, a_type, b_type, c_type
    let d_type = types[0];
    let a_type = types[1];
    let b_type = types[2];
    let c_type = types[3];

    ctx.emit(
        LoweredInstr::Mma {
            shape,
            dst,
            src_a,
            src_b,
            src_c,
            a_layout,
            b_layout,
            a_type,
            b_type,
            d_type,
            c_type,
        },
        predicate,
    )?;
    Ok(())
}

/// Lower `wmma.load.{a,b,c}.sync.aligned.{row,col}.m16n16k16[.shared].f16`
fn lower_wmma_load(
    ctx: &mut LoweringContext,
    modifiers: &[DottedIdent],
    operands: &[AstOperand],
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let mut operand_kind: Option<MmaOperand> = None;
    let mut layout: Option<MmaLayout> = None;
    let mut shape: Option<MmaShape> = None;
    let mut elem_type: Option<ScalarType> = None;
    let mut space: Option<MemSpace> = None;

    for modifier in modifiers {
        let s = modifier.to_string();
        if s == "sync" || s == "aligned" {
            // The modeled execution mode.
        } else if s == "shared" {
            space = Some(MemSpace::Shared);
        } else if s == "global" {
            space = Some(MemSpace::Global);
        } else if let Some(op) = MmaOperand::parse(&s) {
            operand_kind = Some(op);
        } else if let Some(l) = MmaLayout::parse(&s) {
            layout = Some(l);
        } else if let Some(sh) = MmaShape::parse(&s) {
            shape = Some(sh);
        } else if let Some(ty) = parse_scalar_type_modifier(modifier) {
            elem_type = Some(ty);
        } else {
            return Err(unsupported("wmma.load", format!("modifier .{}", s)));
        }
    }

    // Memory spaces have separate address spaces here, so a generic
    // (spaceless) wmma access cannot be resolved to the right memory.
    let space = space.ok_or_else(|| {
        unsupported(
            "wmma.load",
            "no explicit state space (generic addressing is not modeled)",
        )
    })?;

    let operand_kind = operand_kind.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "wmma.load".to_string(),
        reason: Some("missing operand modifier (a, b, or c)".to_string()),
    })?;

    let layout = layout.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "wmma.load".to_string(),
        reason: Some("missing layout modifier (row or col)".to_string()),
    })?;

    let shape = shape.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "wmma.load".to_string(),
        reason: Some("missing shape modifier (e.g., m16n16k16)".to_string()),
    })?;

    let elem_type = elem_type.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "wmma.load".to_string(),
        reason: Some("missing element type modifier (e.g., f16)".to_string()),
    })?;

    // The evaluator's fragment tables are fixed: f16 multiplicands (b16 raw
    // bits allowed) and f32 accumulators. Anything else would be read with
    // the wrong per-lane layout.
    match operand_kind {
        MmaOperand::A | MmaOperand::B => {
            if !matches!(elem_type, ScalarType::F16 | ScalarType::B16) {
                return Err(unsupported(
                    "wmma.load",
                    format!(
                        "a/b fragment element type {:?} (only f16 is modeled)",
                        elem_type
                    ),
                ));
            }
        }
        MmaOperand::C => {
            if elem_type != ScalarType::F32 {
                return Err(unsupported(
                    "wmma.load",
                    format!(
                        "accumulator element type {:?} (only .f32 accumulators are modeled)",
                        elem_type
                    ),
                ));
            }
        }
        MmaOperand::D => {
            return Err(unsupported("wmma.load", "operand .d (loads are a/b/c)"));
        }
    }

    if operands.len() < 3 {
        return Err(LowerError::InvalidOperand {
            instruction: "wmma.load".to_string(),
            operand: format!("{:?}", operands),
            reason: "expected destination vector, address, and stride operands",
        });
    }

    let dst = ctx.resolve_dst_vector(&operands[0])?;
    let (base, offset) = ctx.resolve_mem_operand(&operands[1])?;
    let stride = ctx.resolve_operand(&operands[2])?;

    ctx.emit(
        LoweredInstr::WmmaLoad {
            operand: operand_kind,
            shape,
            layout,
            dst,
            base,
            offset,
            stride,
            elem_type,
            space,
        },
        predicate,
    )?;
    Ok(())
}

/// Lower `wmma.store.d.sync.aligned.{row,col}.m16n16k16[.shared].f32`
fn lower_wmma_store(
    ctx: &mut LoweringContext,
    modifiers: &[DottedIdent],
    operands: &[AstOperand],
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let mut layout: Option<MmaLayout> = None;
    let mut shape: Option<MmaShape> = None;
    let mut elem_type: Option<ScalarType> = None;
    let mut space: Option<MemSpace> = None;

    for modifier in modifiers {
        let s = modifier.to_string();
        if s == "sync" || s == "aligned" || s == "d" {
            // The modeled execution mode; stores always write the d fragment.
        } else if s == "shared" {
            space = Some(MemSpace::Shared);
        } else if s == "global" {
            space = Some(MemSpace::Global);
        } else if let Some(l) = MmaLayout::parse(&s) {
            layout = Some(l);
        } else if let Some(sh) = MmaShape::parse(&s) {
            shape = Some(sh);
        } else if let Some(ty) = parse_scalar_type_modifier(modifier) {
            elem_type = Some(ty);
        } else {
            return Err(unsupported("wmma.store", format!("modifier .{}", s)));
        }
    }

    // Memory spaces have separate address spaces here, so a generic
    // (spaceless) wmma access cannot be resolved to the right memory.
    let space = space.ok_or_else(|| {
        unsupported(
            "wmma.store",
            "no explicit state space (generic addressing is not modeled)",
        )
    })?;

    let layout = layout.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "wmma.store".to_string(),
        reason: Some("missing layout modifier (row or col)".to_string()),
    })?;

    let shape = shape.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "wmma.store".to_string(),
        reason: Some("missing shape modifier (e.g., m16n16k16)".to_string()),
    })?;

    let elem_type = elem_type.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "wmma.store".to_string(),
        reason: Some("missing element type modifier (e.g., f32)".to_string()),
    })?;

    // The evaluator's d-fragment layout is the fixed f32 accumulator table;
    // an .f16 accumulator has a different (packed) layout.
    if elem_type != ScalarType::F32 {
        return Err(unsupported(
            "wmma.store",
            format!(
                "accumulator element type {:?} (only .f32 accumulators are modeled)",
                elem_type
            ),
        ));
    }

    if operands.len() < 3 {
        return Err(LowerError::InvalidOperand {
            instruction: "wmma.store".to_string(),
            operand: format!("{:?}", operands),
            reason: "expected address, source vector, and stride operands",
        });
    }

    let (base, offset) = ctx.resolve_mem_operand(&operands[0])?;
    let src = ctx.resolve_dst_vector(&operands[1])?;
    let stride = ctx.resolve_operand(&operands[2])?;

    ctx.emit(
        LoweredInstr::WmmaStore {
            shape,
            layout,
            src,
            base,
            offset,
            stride,
            elem_type,
            space,
        },
        predicate,
    )?;
    Ok(())
}

/// Lower `wmma.mma.sync.aligned.{row}.{row}.m16n16k16.f32.f32`
fn lower_wmma_mma(
    ctx: &mut LoweringContext,
    modifiers: &[DottedIdent],
    operands: &[AstOperand],
    predicate: Option<Predicate>,
) -> LowerResult<()> {
    let mut layouts: Vec<MmaLayout> = Vec::new();
    let mut shape: Option<MmaShape> = None;
    let mut types: Vec<ScalarType> = Vec::new();

    for modifier in modifiers {
        let s = modifier.to_string();
        if s == "sync" || s == "aligned" {
            // The modeled execution mode.
        } else if let Some(layout) = MmaLayout::parse(&s) {
            layouts.push(layout);
        } else if let Some(sh) = MmaShape::parse(&s) {
            shape = Some(sh);
        } else if let Some(ty) = parse_scalar_type_modifier(modifier) {
            types.push(ty);
        } else {
            return Err(unsupported("wmma.mma", format!("modifier .{}", s)));
        }
    }

    let shape = shape.ok_or_else(|| LowerError::UnsupportedInstruction {
        instruction: "wmma.mma".to_string(),
        reason: Some("missing shape modifier (e.g., m16n16k16)".to_string()),
    })?;

    if layouts.len() != 2 {
        return Err(LowerError::UnsupportedInstruction {
            instruction: "wmma.mma".to_string(),
            reason: Some(format!(
                "expected 2 layout modifiers (row/col), found {}",
                layouts.len()
            )),
        });
    }

    if types.len() != 2 {
        return Err(LowerError::UnsupportedInstruction {
            instruction: "wmma.mma".to_string(),
            reason: Some(format!(
                "expected 2 type modifiers (d_type, c_type), found {}",
                types.len()
            )),
        });
    }

    // The evaluator computes f16 multiplicands into f32 accumulators; the
    // alternate-float forms carry more/other type modifiers.
    if types != [ScalarType::F32, ScalarType::F32] {
        return Err(unsupported(
            "wmma.mma",
            format!("type combination {:?} (only .f32.f32 is modeled)", types),
        ));
    }

    if operands.len() < 4 {
        return Err(LowerError::InvalidOperand {
            instruction: "wmma.mma".to_string(),
            operand: format!("{:?}", operands),
            reason: "expected 4 vector operands (dst, src_a, src_b, src_c)",
        });
    }

    let dst = ctx.resolve_dst_vector(&operands[0])?;
    let src_a = ctx.resolve_dst_vector(&operands[1])?;
    let src_b = ctx.resolve_dst_vector(&operands[2])?;
    let src_c = ctx.resolve_dst_vector(&operands[3])?;

    let a_layout = layouts[0];
    let b_layout = layouts[1];
    let d_type = types[0];
    let c_type = types[1];

    ctx.emit(
        LoweredInstr::WmmaMma {
            shape,
            dst,
            src_a,
            src_b,
            src_c,
            a_layout,
            b_layout,
            d_type,
            c_type,
        },
        predicate,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use volta_frontend::ascii::AsciiString;

    /// Helper to convert &str to AsciiString for tests
    fn ascii(s: &str) -> AsciiString {
        AsciiString::try_from(s.to_string()).unwrap()
    }

    #[test]
    fn test_lowering_context_creation() {
        let ctx = LoweringContext::new();
        assert_eq!(ctx.current_pc(), InstrId::from_index(0));
        assert!(ctx.instructions.is_empty());
    }

    #[test]
    fn test_label_recording() {
        let mut ctx = LoweringContext::new();
        ctx.record_label("LOOP", Some(Span(0, 4)));
        ctx.emit(LoweredInstr::Nop, None).unwrap();

        assert_eq!(
            ctx.symbols.resolve_label("LOOP"),
            Some(InstrId::from_index(0))
        );
    }

    #[test]
    fn test_special_register_from_name() {
        // Test that special registers can be resolved from their names
        assert_eq!(
            SpecialRegKind::from_name("%tid.x"),
            Some(SpecialRegKind::TidX)
        );
        assert_eq!(
            SpecialRegKind::from_name("tid.x"),
            Some(SpecialRegKind::TidX)
        );
        assert_eq!(
            SpecialRegKind::from_name("%laneid"),
            Some(SpecialRegKind::LaneId)
        );
        assert!(SpecialRegKind::from_name("unknown").is_none());
    }

    // =========================================================================
    // Type Checking Tests
    // =========================================================================

    /// Helper to create a context with declared registers
    fn ctx_with_registers(regs: &[(&str, ScalarType)]) -> LoweringContext {
        let mut ctx = LoweringContext::new();
        for (name, ty) in regs {
            ctx.symbols.declare_register(name, *ty, 1).unwrap();
        }
        ctx
    }

    #[test]
    fn test_type_check_exact_match() {
        // Register declared as u32, used with add.u32 - should succeed
        let ctx = ctx_with_registers(&[("%r0", ScalarType::U32)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_dst_type(&dst, ScalarType::U32, "add.u32");
        assert!(result.is_ok());
    }

    #[test]
    fn test_type_check_signed_unsigned_compatible() {
        // Register declared as s32, used with add.u32 - should succeed (compatible)
        let ctx = ctx_with_registers(&[("%r0", ScalarType::S32)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_dst_type(&dst, ScalarType::U32, "add.u32");
        assert!(
            result.is_ok(),
            "s32 should be compatible with u32 instruction"
        );
    }

    #[test]
    fn test_type_check_bits_compatible_with_any() {
        // Register declared as b32, used with add.f32 - should succeed (bits compatible)
        let ctx = ctx_with_registers(&[("%r0", ScalarType::B32)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_dst_type(&dst, ScalarType::F32, "add.f32");
        assert!(
            result.is_ok(),
            "b32 should be compatible with f32 instruction"
        );
    }

    #[test]
    fn test_type_check_float_int_incompatible() {
        // Register declared as f32, used with add.u32 - should FAIL
        let ctx = ctx_with_registers(&[("%f0", ScalarType::F32)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%f0")))
            .unwrap();
        let result = ctx.check_dst_type(&dst, ScalarType::U32, "add.u32");
        assert!(
            result.is_err(),
            "f32 should NOT be compatible with u32 instruction"
        );

        // Verify error message contains useful information
        let err = result.unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("Type error"),
            "Error should mention type error"
        );
        assert!(msg.contains("%f0"), "Error should mention register name");
    }

    #[test]
    fn test_type_check_size_mismatch() {
        // Register declared as u64, used with add.u32 - should FAIL (size mismatch)
        let ctx = ctx_with_registers(&[("%r0", ScalarType::U64)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_dst_type(&dst, ScalarType::U32, "add.u32");
        assert!(
            result.is_err(),
            "u64 should NOT be compatible with u32 instruction (size mismatch)"
        );
    }

    #[test]
    fn test_type_check_immediates_are_polymorphic() {
        // Immediates should be compatible with any instruction type
        let ctx = ctx_with_registers(&[]);

        let imm = ctx.resolve_operand_typed(&AstOperand::ImmInt(42)).unwrap();

        // Should work with u32
        assert!(
            ctx.check_operand_type(&imm, ScalarType::U32, "add.u32")
                .is_ok()
        );
        // Should work with f32
        assert!(
            ctx.check_operand_type(&imm, ScalarType::F32, "add.f32")
                .is_ok()
        );
        // Should work with s64
        assert!(
            ctx.check_operand_type(&imm, ScalarType::S64, "add.s64")
                .is_ok()
        );
    }

    // =========================================================================
    // Relaxed Type Checking Tests (PTX 9.4.1)
    // =========================================================================

    #[test]
    fn test_relaxed_load_wider_dest_ok() {
        // ld.u8 into u32 register - should succeed (value is zero-extended)
        let ctx = ctx_with_registers(&[("%r0", ScalarType::U32)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_dst_type_relaxed(&dst, ScalarType::U8, "ld.u8");
        assert!(
            result.is_ok(),
            "Loading u8 into u32 register should be allowed"
        );
    }

    #[test]
    fn test_relaxed_load_same_size_ok() {
        // ld.u32 into u32 register - should succeed
        let ctx = ctx_with_registers(&[("%r0", ScalarType::U32)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_dst_type_relaxed(&dst, ScalarType::U32, "ld.u32");
        assert!(result.is_ok());
    }

    #[test]
    fn test_relaxed_load_narrower_dest_fails() {
        // ld.u32 into u8 register - should FAIL (dest too narrow)
        let ctx = ctx_with_registers(&[("%r0", ScalarType::U8)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_dst_type_relaxed(&dst, ScalarType::U32, "ld.u32");
        assert!(result.is_err(), "Loading u32 into u8 register should fail");
    }

    #[test]
    fn test_relaxed_load_float_int_still_incompatible() {
        // ld.f32 into u64 register - should FAIL (float/int mismatch)
        let ctx = ctx_with_registers(&[("%r0", ScalarType::U64)]);

        let dst = ctx
            .resolve_dst_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_dst_type_relaxed(&dst, ScalarType::F32, "ld.f32");
        assert!(
            result.is_err(),
            "Loading f32 into integer register should fail even with relaxed checking"
        );
    }

    #[test]
    fn test_relaxed_store_wider_source_ok() {
        // st.u8 from u32 register - should succeed (value is truncated)
        let ctx = ctx_with_registers(&[("%r0", ScalarType::U32)]);

        let src = ctx
            .resolve_operand_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_operand_type_relaxed(&src, ScalarType::U8, "st.u8");
        assert!(
            result.is_ok(),
            "Storing from u32 register to u8 should be allowed (truncation)"
        );
    }

    #[test]
    fn test_relaxed_store_narrower_source_fails() {
        // st.u32 from u8 register - should FAIL (source too narrow)
        let ctx = ctx_with_registers(&[("%r0", ScalarType::U8)]);

        let src = ctx
            .resolve_operand_typed(&AstOperand::Ident(ascii("%r0")))
            .unwrap();
        let result = ctx.check_operand_type_relaxed(&src, ScalarType::U32, "st.u32");
        assert!(
            result.is_err(),
            "Storing from u8 register to u32 should fail"
        );
    }

    // =========================================================================
    // Special Register Type Tests
    // =========================================================================

    #[test]
    fn test_special_reg_type_resolution() {
        let ctx = ctx_with_registers(&[]);

        // %tid.x is u32
        let tid = ctx
            .resolve_operand_typed(&AstOperand::Ident(ascii("%tid.x")))
            .unwrap();
        assert_eq!(tid.ty, Some(ScalarType::U32));

        // Should be compatible with u32 instruction
        assert!(
            ctx.check_operand_type(&tid, ScalarType::U32, "add.u32")
                .is_ok()
        );

        // Should NOT be compatible with f32 instruction
        assert!(
            ctx.check_operand_type(&tid, ScalarType::F32, "add.f32")
                .is_err()
        );
    }

    // =========================================================================
    // Hint Generation Tests
    // =========================================================================

    #[test]
    fn test_hint_for_float_to_int() {
        let ctx = ctx_with_registers(&[]);
        let hint = ctx.type_mismatch_hint(ScalarType::F32, ScalarType::U32);
        assert!(hint.contains("cvt"), "Hint should suggest cvt instruction");
    }

    #[test]
    fn test_hint_for_int_to_float() {
        let ctx = ctx_with_registers(&[]);
        let hint = ctx.type_mismatch_hint(ScalarType::S32, ScalarType::F32);
        assert!(hint.contains("cvt"), "Hint should suggest cvt instruction");
    }

    #[test]
    fn test_hint_for_size_mismatch() {
        let ctx = ctx_with_registers(&[]);
        let hint = ctx.type_mismatch_hint(ScalarType::U64, ScalarType::U32);
        assert!(
            hint.contains("bits") || hint.contains("size"),
            "Hint should mention size mismatch"
        );
    }

    // =========================================================================
    // Exhaustive-lowering policy tests: unmodeled modifiers/forms error
    // loudly (naming the modifier), and the modeled forms still lower.
    // =========================================================================

    // =========================================================================
    // Shared memory layout through lower_function: module-level externs are
    // collected before function-body statics, but the extern window must be
    // placed after all statics (CUDA ABI).
    // =========================================================================

    /// Parse `src` as a full module and lower its entry, passing module-level
    /// variable declarations through as the driver does.
    fn lower_module_src(src: &str) -> LoweredProgram {
        use volta_frontend::ascii::AsAscii;
        use volta_frontend::parse::Parser;

        let ascii = src.as_bytes().as_ascii_slice().expect("ascii source");
        let module = Parser::new(ascii)
            .parse_module()
            .unwrap_or_else(|e| panic!("parse error: {:?}", e.error));
        let vars: Vec<ast::VarDecl> = module
            .items
            .iter()
            .filter_map(|item| match item {
                ast::TopLevelItem::Variable(v) => Some(v.clone()),
                _ => None,
            })
            .collect();
        let func = module
            .items
            .iter()
            .find_map(|item| match item {
                ast::TopLevelItem::Entry(f) => Some(f),
                _ => None,
            })
            .expect("kernel not found");
        lower_function(func, &vars).expect("lowering failed")
    }

    #[test]
    fn test_extern_shared_window_follows_static_shared() {
        let program = lower_module_src(
            ".version 8.0\n.target sm_80\n.address_size 64\n\n\
             .extern .shared .align 16 .b8 buf[];\n\
             .visible .entry k()\n{\n\
             .reg .b32 %r<4>;\n\
             .shared .align 4 .f32 lut[64];\n\
             ret;\n}\n",
        );
        // The static packs from 0; the extern window is disjoint, after it.
        let lut = program.symbols.get_shared_var("lut").unwrap();
        assert_eq!((lut.offset, lut.size_bytes), (0, 256));
        let buf = program.symbols.get_shared_var("buf").unwrap();
        assert!(buf.is_extern);
        assert_eq!(buf.offset, 256);
        assert_eq!(program.symbols.extern_shared_base(), Some(256));
    }

    /// Extern-only modules (every extern-shared kernel in the corpus) keep
    /// the window at offset 0: behavior identical to before the fix.
    #[test]
    fn test_extern_only_shared_window_at_zero() {
        let program = lower_module_src(
            ".version 8.0\n.target sm_80\n.address_size 64\n\n\
             .extern .shared .align 16 .b8 buf[];\n\
             .visible .entry k()\n{\n\
             .reg .b32 %r<4>;\n\
             ret;\n}\n",
        );
        assert_eq!(program.symbols.get_shared_var("buf").unwrap().offset, 0);
        assert_eq!(program.symbols.extern_shared_base(), Some(0));
    }

    /// Parse and lower a kernel whose body is `body`, with a standard set of
    /// registers declared.
    fn lower_body(body: &str) -> LowerResult<LoweredProgram> {
        use volta_frontend::ascii::AsAscii;
        use volta_frontend::parse::Parser;

        let src = format!(
            ".version 8.0\n.target sm_80\n.address_size 64\n\n\
             .visible .entry k()\n{{\n\
             .reg .pred %p<4>;\n\
             .reg .b16 %rs<8>;\n\
             .reg .f32 %f<8>;\n\
             .reg .b32 %r<8>;\n\
             .reg .f64 %fd<4>;\n\
             .reg .b64 %rd<8>;\n\
             .shared .align 4 .b8 smem[64];\n\
             {}\n\
             ret;\n}}\n",
            body
        );
        let ascii = src.as_bytes().as_ascii_slice().expect("ascii source");
        let module = Parser::new(ascii)
            .parse_module()
            .unwrap_or_else(|e| panic!("parse error: {:?}", e.error));
        let func = module
            .items
            .iter()
            .find_map(|item| match item {
                ast::TopLevelItem::Entry(f) => Some(f),
                _ => None,
            })
            .expect("kernel not found");
        lower_function(func, &[])
    }

    /// Assert that `body` is rejected with an UnsupportedInstruction whose
    /// reason mentions `needle`.
    fn assert_rejected(body: &str, needle: &str) {
        match lower_body(body) {
            Err(LowerError::UnsupportedInstruction {
                instruction,
                reason,
            }) => {
                let text = format!("{} {}", instruction, reason.unwrap_or_default());
                assert!(
                    text.contains(needle),
                    "expected rejection of {:?} to mention {:?}, got {:?}",
                    body,
                    needle,
                    text
                );
            }
            Err(other) => panic!(
                "expected UnsupportedInstruction for {:?}, got {:?}",
                body, other
            ),
            Ok(_) => panic!("expected {:?} to be rejected, but it lowered", body),
        }
    }

    fn assert_lowers(body: &str) {
        if let Err(e) = lower_body(body) {
            panic!("expected {:?} to lower, got {:?}", body, e);
        }
    }

    /// Lower `body` (which must contain exactly one clamp-capable
    /// instruction - BinOp/Fma/Cvt) and return that instruction's clamp.
    fn lowered_clamp(body: &str) -> Option<Clamp> {
        let prog = lower_body(body)
            .unwrap_or_else(|e| panic!("expected {:?} to lower, got {:?}", body, e));
        let mut clamps = prog.instructions.values().filter_map(|instr| match instr {
            LoweredInstr::BinOp { clamp, .. }
            | LoweredInstr::Fma { clamp, .. }
            | LoweredInstr::Cvt { clamp, .. } => Some(*clamp),
            _ => None,
        });
        let clamp = clamps
            .next()
            .unwrap_or_else(|| panic!("no clamp-capable instruction lowered from {:?}", body));
        assert!(
            clamps.next().is_none(),
            "expected exactly one clamp-capable instruction in {:?}",
            body
        );
        clamp
    }

    #[test]
    fn test_reject_setp_bool_combine_and_dual_dest() {
        assert_rejected("setp.lt.and.s32 %p1, %r1, %r2, %p2;", "boolean-combine");
        assert_rejected("setp.lt.s32 %p1|%p2, %r1, %r2;", "dual predicate");
        assert_lowers("setp.lt.s32 %p1, %r1, %r2;");
        assert_lowers("setp.leu.f32 %p1, %f1, %f2;");
    }

    #[test]
    fn test_reject_integer_sat() {
        assert_rejected("add.sat.s32 %r1, %r2, %r3;", ".sat");
        assert_rejected("sub.sat.s32 %r1, %r2, %r3;", ".sat");
        assert_rejected("mad.hi.sat.s32 %r1, %r2, %r3, %r4;", ".sat");
        assert_lowers("add.s32 %r1, %r2, %r3;");
        assert_lowers("mad.lo.s32 %r1, %r2, %r3, %r4;");
    }

    #[test]
    fn test_float_sat_lowers_with_clamp() {
        // Float `.sat` is the exact value clamp min(max(x, 0), 1) over the
        // reals; f32 and scalar f16 forms carry it through lowering.
        assert_eq!(
            lowered_clamp("add.sat.f32 %f1, %f2, %f3;"),
            Some(Clamp::Sat)
        );
        assert_eq!(
            lowered_clamp("sub.rn.ftz.sat.f32 %f1, %f2, %f3;"),
            Some(Clamp::Sat)
        );
        assert_eq!(
            lowered_clamp("mul.sat.f32 %f1, %f2, %f3;"),
            Some(Clamp::Sat)
        );
        assert_eq!(
            lowered_clamp("fma.rn.sat.f32 %f1, %f2, %f3, %f4;"),
            Some(Clamp::Sat)
        );
        assert_eq!(
            lowered_clamp("add.rn.sat.f16 %rs1, %rs2, %rs3;"),
            Some(Clamp::Sat)
        );
        assert_eq!(
            lowered_clamp("sub.rn.sat.f16 %rs1, %rs2, %rs3;"),
            Some(Clamp::Sat)
        );
        assert_eq!(
            lowered_clamp("mul.rn.sat.f16 %rs1, %rs2, %rs3;"),
            Some(Clamp::Sat)
        );
        assert_eq!(
            lowered_clamp("fma.rn.sat.f16 %rs1, %rs2, %rs3, %rs4;"),
            Some(Clamp::Sat)
        );
        // Rounding modes and ftz alone are ignorable under the reals model,
        // and unclamped forms stay unclamped.
        assert_eq!(lowered_clamp("add.rn.ftz.f32 %f1, %f2, %f3;"), None);
        assert_eq!(lowered_clamp("mul.rn.f32 %f1, %f2, %f3;"), None);
        assert_eq!(lowered_clamp("fma.rn.ftz.f32 %f1, %f2, %f3, %f4;"), None);
        assert_eq!(lowered_clamp("mul.f16 %rs1, %rs2, %rs3;"), None);
    }

    #[test]
    fn test_float_sat_out_of_scope_forms_stay_rejected() {
        // mad.f and the mixed-precision (.f32.f16) arms are not modeled
        // with .sat; packed forms are rejected as packed SIMD.
        assert_rejected("mad.sat.f32 %f1, %f2, %f3, %f4;", ".sat");
        assert_rejected("add.rn.sat.f32.f16 %f1, %rs2, %rs3;", ".sat");
        assert_rejected("fma.rn.sat.f32.f16 %f1, %rs2, %rs3, %f4;", ".sat");
        assert_rejected("add.rn.sat.f16x2 %r1, %r2, %r3;", "packed SIMD");
        // `.sat` on .f64 add/mul/fma is illegal PTX (the ISA allows it on
        // .f32/.f16 only); the instruction parser rejects it loudly.
        assert_rejected("add.rn.sat.f64 %fd1, %fd2, %fd3;", "parsing failed");
        assert_rejected("mul.rn.sat.f64 %fd1, %fd2, %fd3;", "parsing failed");
        assert_rejected("fma.rn.sat.f64 %fd1, %fd2, %fd3, %fd4;", "parsing failed");
    }

    #[test]
    fn test_fma_half_relu_lowers_oob_rejected() {
        // fma.rn.relu.f16 is the exact value clamp max(x, 0).
        assert_eq!(
            lowered_clamp("fma.rn.relu.f16 %rs1, %rs2, %rs3, %rs4;"),
            Some(Clamp::Relu)
        );
        assert_eq!(lowered_clamp("fma.rn.f16 %rs1, %rs2, %rs3, %rs4;"), None);
        assert_rejected("fma.rn.oob.f16 %rs1, %rs2, %rs3, %rs4;", ".oob");
        assert_rejected("fma.rn.relu.bf16 %rs1, %rs2, %rs3, %rs4;", ".relu");
    }

    #[test]
    fn test_reject_mad_hi_wide_widths_still_lower() {
        // mad.hi at 64 bits lowers but is rejected at evaluation (interp.rs);
        // mad.hi at 32 bits is fully supported.
        assert_lowers("mad.hi.u32 %r1, %r2, %r3, %r4;");
    }

    #[test]
    fn test_reject_minmax_modifiers() {
        assert_rejected("min.NaN.f32 %f1, %f2, %f3;", ".NaN");
        assert_rejected("max.NaN.f32 %f1, %f2, %f3;", ".NaN");
        assert_rejected("max.xorsign.abs.f32 %f1, %f2, %f3;", ".xorsign.abs");
        assert_rejected("min.abs.f32 %f1, %f2, %f3;", ".abs");
        assert_rejected("max.ftz.f32 %f1, %f2, %f3, %f4;", "3-input");
        assert_rejected("min.relu.s32 %r1, %r2, %r3;", ".relu");
        assert_lowers("min.f32 %f1, %f2, %f3;");
        assert_lowers("max.s32 %r1, %r2, %r3;");
        assert_lowers("min.ftz.f32 %f1, %f2, %f3;");
    }

    #[test]
    fn test_cvt_float_sat_relu_lower_with_clamp() {
        // Float->float cvt `.sat`/`.relu` are exact value clamps; the
        // fused-ReLU epilogue nvcc emits is `cvt.rn.relu.f16.f32`.
        assert_eq!(
            lowered_clamp("cvt.rn.relu.f16.f32 %rs1, %f1;"),
            Some(Clamp::Relu)
        );
        assert_eq!(
            lowered_clamp("cvt.rn.relu.bf16.f32 %rs1, %f1;"),
            Some(Clamp::Relu)
        );
        assert_eq!(
            lowered_clamp("cvt.rn.sat.f16.f32 %rs1, %f1;"),
            Some(Clamp::Sat)
        );
        assert_eq!(
            lowered_clamp("cvt.rn.sat.f32.f64 %f1, %fd1;"),
            Some(Clamp::Sat)
        );
        assert_eq!(
            lowered_clamp("cvt.sat.f64.f32 %fd1, %f1;"),
            Some(Clamp::Sat)
        );
        assert_eq!(lowered_clamp("cvt.rn.f16.f32 %rs1, %f1;"), None);
    }

    #[test]
    fn test_reject_cvt_modifiers_keep_plain_conversions() {
        // Integer-destination saturation is a different operation (clamp to
        // MININT..MAXINT) and stays rejected, as do non-float sources.
        assert_rejected("cvt.sat.u32.f32 %r1, %f1;", ".sat");
        assert_rejected("cvt.rn.sat.f16.s32 %rs1, %r1;", ".sat");
        // .sat/.relu destinations outside the ISA's legal lists are invalid.
        assert_rejected("cvt.rn.sat.bf16.f32 %rs1, %f1;", ".sat");
        assert_rejected("cvt.rn.relu.f32.f16 %f1, %rs1;", ".relu");
        // .satfinite clamps to the destination's finite range: not modeled.
        assert_rejected("cvt.rn.satfinite.f16.f32 %rs1, %f1;", ".satfinite");
        // Integer rounding stays rejected, including under an accepted clamp.
        assert_rejected("cvt.rmi.f32.f32 %f1, %f2;", "integer-rounding");
        assert_rejected("cvt.rzi.s32.f32 %r1, %f1;", "integer-rounding");
        assert_rejected("cvt.rzi.sat.f32.f32 %f1, %f2;", "integer-rounding");
        // cvt.pack has its own InstrKind (CvtPack) with no lowering: it hits
        // the already-loud whole-instruction catch-all. The CvtInstr::Pack
        // rejection in lower_cvt is fail-closed insurance behind it.
        assert_rejected("cvt.pack.sat.u16.s32 %r1, %r2, %r3;", "CvtPack");
        // The corpus-hot conversions must keep lowering.
        assert_lowers("cvt.rn.f16.f32 %rs1, %f1;");
        assert_lowers("cvt.f32.f16 %f1, %rs1;");
        assert_lowers("cvt.u64.u32 %rd1, %r1;");
        assert_lowers("cvt.s64.s32 %rd1, %r1;");
        assert_lowers("cvt.u32.u64 %r1, %rd1;");
    }

    #[test]
    fn test_reject_packed_simd_arithmetic() {
        assert_rejected("add.u16x2 %r1, %r2, %r3;", "packed SIMD");
        assert_rejected("add.rn.f16x2 %r1, %r2, %r3;", "packed SIMD");
        assert_rejected("sub.rn.bf16x2 %r1, %r2, %r3;", "packed SIMD");
        assert_rejected("mul.rn.f16x2 %r1, %r2, %r3;", "packed SIMD");
        assert_rejected("min.s16x2 %r1, %r2, %r3;", "packed SIMD");
        assert_rejected("max.u16x2 %r1, %r2, %r3;", "packed SIMD");
        assert_rejected("neg.ftz.f16x2 %r1, %r2;", "packed SIMD");
        assert_rejected("fma.rn.f16x2 %r1, %r2, %r3, %r4;", "packed SIMD");
    }

    #[test]
    fn test_bar_rejections() {
        assert_rejected("bar.arrive 0, 64;", "bar.arrive");
        assert_rejected("bar.red 0;", "bar.red");
        // The full bar.red form fails earlier, at instruction parsing (the
        // .popc reduction op modifier is not parsed) - also loud.
        assert_rejected("bar.red.popc.u32 %r1, 0, %p1;", "parsing failed");
        assert_rejected("bar.sync %r1;", "register barrier id");
        assert_rejected("bar.sync;", "missing barrier id");
        assert_rejected("bar.sync 0, 64;", "thread-count");
        match lower_body("bar.sync 16;") {
            Err(LowerError::InvalidOperand { .. }) => {}
            other => panic!(
                "expected InvalidOperand for out-of-range id, got {:?}",
                other
            ),
        }
        assert_lowers("bar.sync 0;");
        assert_lowers("bar.sync 1;");
    }

    #[test]
    fn test_reject_generic_ld_st() {
        assert_rejected("ld.u32 %r1, [%rd1];", "generic");
        assert_rejected("st.u32 [%rd1], %r1;", "generic");
        assert_lowers("ld.global.u32 %r1, [%rd1];");
        assert_lowers("ld.volatile.shared.f32 %f1, [%r1];");
        assert_lowers("st.volatile.shared.u32 [%r1], %r2;");
        assert_lowers("ld.global.nc.f32 %f1, [%rd1];");
        assert_lowers("ld.global.nc.v4.u32 {%r1, %r2, %r3, %r4}, [%rd1];");
    }

    #[test]
    fn test_reject_ld_ordering_qualifiers() {
        assert_rejected("ld.acquire.gpu.global.u32 %r1, [%rd1];", "memory-ordering");
        assert_rejected("st.release.gpu.global.u32 [%rd1], %r1;", "memory-ordering");
    }

    #[test]
    fn test_cache_perf_hints_accepted_cache_hint_rejected() {
        // Eviction-priority and prefetch-size qualifiers are pure
        // performance hints with no extra operand; they are accepted and
        // ignored on ld, st, and ld.global.nc alike.
        assert_lowers("ld.global.L2::128B.f32 %f1, [%rd1];");
        assert_lowers("ld.global.L1::evict_last.L2::evict_first.u32 %r1, [%rd1];");
        assert_lowers("ld.global.L1::no_allocate.L2::256B.f32 %f1, [%rd1];");
        assert_lowers("st.global.L1::evict_first.f32 [%rd1], %f1;");
        assert_lowers("st.global.L2::evict_last.u32 [%rd1], %r1;");
        assert_lowers("ld.global.nc.L1::evict_unchanged.L2::64B.f32 %f1, [%rd1];");
        // L2::cache_hint requires a cache-policy operand we do not parse;
        // it must stay rejected on every path, not be skipped as a hint.
        assert_rejected(
            "ld.global.L2::cache_hint.f32 %f1, [%rd1];",
            "QualifiedModifier",
        );
        assert_rejected(
            "st.global.L2::cache_hint.f32 [%rd1], %f1;",
            "QualifiedModifier",
        );
        assert_rejected("ld.global.nc.L2::cache_hint.f32 %f1, [%rd1];", "cache_hint");
        // Spellings outside the spec's exact ld/st enumeration (L2 has no
        // no_allocate) stay rejected too.
        assert_rejected(
            "ld.global.L2::no_allocate.f32 %f1, [%rd1];",
            "QualifiedModifier",
        );
        assert_rejected(
            "ld.global.nc.L2::no_allocate.f32 %f1, [%rd1];",
            "no_allocate",
        );
    }

    #[test]
    fn test_ldmatrix_modifier_whitelist() {
        assert_rejected(
            "ldmatrix.sync.aligned.x4.m8n16.shared.b16 {%r1, %r2, %r3, %r4}, [%r5];",
            "m8n16",
        );
        assert_rejected(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b8 {%r1, %r2, %r3, %r4}, [%r5];",
            "b8",
        );
        assert_lowers("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r1, %r2, %r3, %r4}, [%r5];");
        assert_lowers("ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {%r1, %r2}, [%r5];");
    }

    #[test]
    fn test_wmma_requires_space_and_f32_accumulators() {
        // No state space modifier -> generic addressing, rejected.
        assert_rejected(
            "wmma.load.a.sync.aligned.row.m16n16k16.f16 \
             {%r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7}, [%r1], %r2;",
            "state space",
        );
        // .f16 accumulator fragments have a different layout.
        assert_rejected(
            "wmma.load.c.sync.aligned.row.m16n16k16.shared.f16 \
             {%r0, %r1, %r2, %r3}, [%r1], %r2;",
            "accumulator",
        );
        assert_rejected(
            "wmma.store.d.sync.aligned.row.m16n16k16.shared.f16 \
             [%r1], {%r0, %r1, %r2, %r3}, %r2;",
            "accumulator",
        );
        // The corpus forms still lower.
        assert_lowers(
            "wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 \
             {%r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7}, [%r1], %r2;",
        );
        assert_lowers(
            "wmma.store.d.sync.aligned.row.m16n16k16.shared.f32 \
             [%r1], {%f0, %f1, %f2, %f3, %f4, %f5, %f6, %f7}, %r2;",
        );
        assert_lowers(
            "wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 \
             {%f0, %f1, %f2, %f3, %f4, %f5, %f6, %f7}, \
             {%r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7}, \
             {%r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7}, \
             {%f0, %f1, %f2, %f3, %f4, %f5, %f6, %f7};",
        );
    }

    /// Lower `body` and return the byte offset of its one tensor-core
    /// memory instruction (ldmatrix / wmma.load / wmma.store), whose base
    /// must have resolved to a register.
    fn lowered_tensor_offset(body: &str) -> i64 {
        let prog = lower_body(body)
            .unwrap_or_else(|e| panic!("expected {:?} to lower, got {:?}", body, e));
        let mut offsets = prog.instructions.values().filter_map(|instr| match instr {
            LoweredInstr::Ldmatrix { base, offset, .. }
            | LoweredInstr::WmmaLoad { base, offset, .. }
            | LoweredInstr::WmmaStore { base, offset, .. } => {
                assert!(
                    matches!(base, Operand::Reg(_)),
                    "expected a register base in {:?}, got {:?}",
                    body,
                    base
                );
                Some(*offset)
            }
            _ => None,
        });
        let offset = offsets
            .next()
            .unwrap_or_else(|| panic!("no tensor-core memory instruction lowered from {:?}", body));
        assert!(
            offsets.next().is_none(),
            "expected exactly one tensor-core memory instruction in {:?}",
            body
        );
        offset
    }

    /// The immediate of a `[reg + imm]` address operand survives lowering
    /// for the tensor-core memory ops. It used to be dropped, so
    /// `ldmatrix ... [%r2+16]` loaded from `[%r2]` (issue #4).
    #[test]
    fn test_tensor_core_ops_keep_address_offset() {
        assert_eq!(
            lowered_tensor_offset(
                "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r1, %r2, %r3, %r4}, [%r5+16];"
            ),
            16
        );
        // nvcc spells negative immediates `+-N`.
        assert_eq!(
            lowered_tensor_offset("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%r1}, [%r5+-32];"),
            -32
        );
        assert_eq!(
            lowered_tensor_offset("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%r1}, [%r5];"),
            0
        );
        assert_eq!(
            lowered_tensor_offset(
                "wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 \
                 {%r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7}, [%rd1+64], %r2;"
            ),
            64
        );
        assert_eq!(
            lowered_tensor_offset(
                "wmma.store.d.sync.aligned.row.m16n16k16.shared.f32 \
                 [%rd1+128], {%f0, %f1, %f2, %f3, %f4, %f5, %f6, %f7}, %r2;"
            ),
            128
        );
    }

    /// An address offset that is not a constant is rejected, not silently
    /// read as `+0`.
    #[test]
    fn test_reject_non_constant_address_offset() {
        for body in [
            "ld.shared.u32 %r1, [%r2+bogus];",
            "st.shared.u32 [%r2+bogus], %r1;",
            "ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%r1}, [%r2+bogus];",
        ] {
            match lower_body(body) {
                Err(LowerError::InvalidOperand { reason, .. }) => {
                    assert!(reason.contains("constant"), "{:?}: {}", body, reason);
                }
                other => panic!("expected InvalidOperand for {:?}, got {:?}", body, other),
            }
        }
    }

    /// A memory operand where a value is expected is rejected instead of
    /// being read as its base register.
    #[test]
    fn test_reject_address_operand_in_value_position() {
        assert_rejected("add.u32 %r1, %r2, [%r3+4];", "address operand");
    }

    #[test]
    fn test_mma_type_combination_checked() {
        assert_rejected(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \
             {%f0, %f1, %f2, %f3}, {%r0, %r1, %r2, %r3}, {%r0, %r1}, {%f0, %f1, %f2, %f3};",
            "f32.f16.f16.f32",
        );
        assert_lowers(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
             {%f0, %f1, %f2, %f3}, {%r0, %r1, %r2, %r3}, {%r0, %r1}, {%f0, %f1, %f2, %f3};",
        );
    }

    #[test]
    fn test_vector_arity_must_match_modifier() {
        match lower_body("ld.global.v4.u32 {%r1, %r2}, [%rd1];") {
            Err(LowerError::InvalidOperand { .. }) => {}
            other => panic!(
                "expected InvalidOperand for v4/2-reg mismatch, got {:?}",
                other
            ),
        }
        assert_lowers("ld.global.v2.u32 {%r1, %r2}, [%rd1];");
        assert_lowers("st.shared.v4.u32 [%r1], {%r2, %r3, %r4, %r5};");
    }
}
