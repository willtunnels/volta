//! Symbol table for PTX lowering
//!
//! This module provides the symbol table that tracks register declarations,
//! labels, parameters, and shared memory allocations during lowering.

use std::collections::HashMap;
use std::fmt;

use id_collections::{IdVec, id_type};
use volta_frontend::ast::{Ptr, ScalarType};

use crate::lower_error::{LowerError, LowerResult};
use crate::lowered::InstrId;
use crate::types::{RegClass, RegCounts};

/// The kind of symbol in the PTX namespace
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Register,
    Parameter,
    Label,
    SharedVariable,
    LocalVariable,
    GlobalVariable,
}

impl fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Register => write!(f, "register"),
            Self::Parameter => write!(f, "parameter"),
            Self::Label => write!(f, "label"),
            Self::SharedVariable => write!(f, "shared variable"),
            Self::LocalVariable => write!(f, "local variable"),
            Self::GlobalVariable => write!(f, "global variable"),
        }
    }
}

/// Index of a register within its class
#[id_type]
pub struct RegIndex(pub u32);

impl fmt::Display for RegIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A resolved register reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegId {
    pub class: RegClass,
    pub index: RegIndex,
}

impl RegId {
    /// Create a new RegId
    pub fn new(class: RegClass, index: u32) -> Self {
        Self {
            class,
            index: RegIndex(index),
        }
    }
}

impl fmt::Display for RegId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}[{}]", self.class, self.index)
    }
}

/// Parameter index
#[id_type]
pub struct ParamId(pub u32);

impl fmt::Display for ParamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "param:{}", self.0)
    }
}

/// Information about a declared register
#[derive(Debug, Clone)]
pub struct RegInfo {
    pub id: RegId,
    pub name: String,
    pub declared_type: ScalarType,
}

/// Special register kinds (thread-dependent values known at runtime)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialRegKind {
    // Thread identification
    TidX,
    TidY,
    TidZ,
    NtidX,
    NtidY,
    NtidZ,
    CtaidX,
    CtaidY,
    CtaidZ,
    NctaidX,
    NctaidY,
    NctaidZ,
    // Warp-level
    LaneId,
    WarpId,
    NWarpId,
    LanemaskEq,
    LanemaskLt,
    LanemaskLe,
    LanemaskGt,
    LanemaskGe,
    // SM identification
    SmId,
    NSmId,
    // Grid
    GridId,
    // Clock
    Clock,
    Clock64,
    // Dynamic shared memory
    DynamicSmemSize,
}

impl SpecialRegKind {
    /// Get the type of this special register
    pub fn ty(&self) -> ScalarType {
        match self {
            // 32-bit unsigned
            Self::TidX
            | Self::TidY
            | Self::TidZ
            | Self::NtidX
            | Self::NtidY
            | Self::NtidZ
            | Self::CtaidX
            | Self::CtaidY
            | Self::CtaidZ
            | Self::NctaidX
            | Self::NctaidY
            | Self::NctaidZ
            | Self::LaneId
            | Self::WarpId
            | Self::NWarpId
            | Self::LanemaskEq
            | Self::LanemaskLt
            | Self::LanemaskLe
            | Self::LanemaskGt
            | Self::LanemaskGe
            | Self::Clock
            | Self::SmId
            | Self::NSmId
            | Self::GridId
            | Self::DynamicSmemSize => ScalarType::U32,
            // 64-bit unsigned
            Self::Clock64 => ScalarType::U64,
        }
    }

    /// Parse from PTX name (with or without % prefix)
    pub fn from_name(name: &str) -> Option<Self> {
        let name = name.strip_prefix('%').unwrap_or(name);
        Some(match name {
            "tid.x" => Self::TidX,
            "tid.y" => Self::TidY,
            "tid.z" => Self::TidZ,
            "ntid.x" => Self::NtidX,
            "ntid.y" => Self::NtidY,
            "ntid.z" => Self::NtidZ,
            "ctaid.x" => Self::CtaidX,
            "ctaid.y" => Self::CtaidY,
            "ctaid.z" => Self::CtaidZ,
            "nctaid.x" => Self::NctaidX,
            "nctaid.y" => Self::NctaidY,
            "nctaid.z" => Self::NctaidZ,
            "laneid" => Self::LaneId,
            "warpid" => Self::WarpId,
            "nwarpid" => Self::NWarpId,
            "lanemask_eq" => Self::LanemaskEq,
            "lanemask_lt" => Self::LanemaskLt,
            "lanemask_le" => Self::LanemaskLe,
            "lanemask_gt" => Self::LanemaskGt,
            "lanemask_ge" => Self::LanemaskGe,
            "clock" => Self::Clock,
            "clock64" => Self::Clock64,
            "smid" => Self::SmId,
            "nsmid" => Self::NSmId,
            "gridid" => Self::GridId,
            "dynamic_smem_size" => Self::DynamicSmemSize,
            _ => return None,
        })
    }

    /// Get the PTX name for this special register
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TidX => "%tid.x",
            Self::TidY => "%tid.y",
            Self::TidZ => "%tid.z",
            Self::NtidX => "%ntid.x",
            Self::NtidY => "%ntid.y",
            Self::NtidZ => "%ntid.z",
            Self::CtaidX => "%ctaid.x",
            Self::CtaidY => "%ctaid.y",
            Self::CtaidZ => "%ctaid.z",
            Self::NctaidX => "%nctaid.x",
            Self::NctaidY => "%nctaid.y",
            Self::NctaidZ => "%nctaid.z",
            Self::LaneId => "%laneid",
            Self::WarpId => "%warpid",
            Self::NWarpId => "%nwarpid",
            Self::LanemaskEq => "%lanemask_eq",
            Self::LanemaskLt => "%lanemask_lt",
            Self::LanemaskLe => "%lanemask_le",
            Self::LanemaskGt => "%lanemask_gt",
            Self::LanemaskGe => "%lanemask_ge",
            Self::Clock => "%clock",
            Self::Clock64 => "%clock64",
            Self::SmId => "%smid",
            Self::NSmId => "%nsmid",
            Self::GridId => "%gridid",
            Self::DynamicSmemSize => "%dynamic_smem_size",
        }
    }
}

/// Information about a kernel parameter
#[derive(Debug, Clone)]
pub struct ParamInfo {
    pub id: ParamId,
    pub name: String,
    pub ty: ScalarType,
    pub size_bytes: u64,
    /// The `.ptr` attribute when the kernel declares one (Triton does,
    /// nvcc does not): the contract `Interpreter::new` checks the config's
    /// binding against.
    pub ptr: Option<Ptr>,
}

/// Information about a shared memory allocation
#[derive(Debug, Clone)]
pub struct SharedMemInfo {
    pub name: String,
    pub offset: u64,
    pub size_bytes: u64,
    pub element_ty: ScalarType,
    pub is_extern: bool,
}

/// A `.extern .shared` declaration recorded during collection, before it can
/// be placed. All extern shared names alias the one dynamic-shared window,
/// and the CUDA ABI bases that window *after* every static `.shared`
/// allocation - which are not all known yet when the extern is declared
/// (module-level declarations are collected before function-body ones). So
/// declaration only records the request; `finalize_shared_layout` assigns
/// the offset.
#[derive(Debug, Clone)]
struct PendingExternShared {
    name: String,
    element_ty: ScalarType,
    /// Declared alignment, already raised to the element size.
    alignment: u64,
}

/// Placement state of the `.extern .shared` (dynamic shared) window.
#[derive(Debug, Clone)]
enum ExternSharedLayout {
    /// Declarations are still being collected; extern variables are recorded
    /// here, unplaced, and are absent from `shared_vars`.
    Collecting(Vec<PendingExternShared>),
    /// All static declarations are known; every extern variable is present
    /// in `shared_vars` at offset `extern_base`.
    Finalized { extern_base: u64 },
}

/// Information about a function-scope local memory allocation (e.g. the
/// `__local_depot` stack array nvcc emits for spilled locals).
#[derive(Debug, Clone)]
pub struct LocalMemInfo {
    pub name: String,
    pub offset: u64,
    pub size_bytes: u64,
    pub element_ty: ScalarType,
}

/// Information about a module-scope `.global` variable (e.g. values set by the
/// host via `cudaMemcpyToSymbol`). Assigned addresses in a reserved region so
/// they cannot collide with caller-chosen input array addresses.
#[derive(Debug, Clone)]
pub struct GlobalVarInfo {
    pub name: String,
    pub addr: u64,
    pub size_bytes: u64,
    pub element_ty: ScalarType,
}

/// Base address for module-scope `.global` variables. Analysis configs
/// must not place input/output arrays overlapping the window these occupy
/// (`[MODULE_GLOBAL_BASE, MODULE_GLOBAL_BASE + module_global_size())`);
/// `Interpreter::new` rejects configs that do.
pub const MODULE_GLOBAL_BASE: u64 = 0x7000_0000_0000_0000;

/// Round `offset` up to the next multiple of `alignment` (a power of two),
/// or `None` when the rounded offset overflows u64 (checked arithmetic,
/// release-active: packing must reject hostile sizes loudly, not wrap).
///
/// Natural alignment of every lowering-assigned address rests on this:
/// the `declare_*` functions raise the alignment to at least the element
/// size before calling, so shared/local packing and module-global
/// assignment (from the naturally-aligned `MODULE_GLOBAL_BASE`) always
/// yield naturally aligned, `.align`-honoring addresses.
fn align_up(offset: u64, alignment: u64) -> Option<u64> {
    // PTX requires `.align` values to be powers of two (and every scalar
    // element size is one); the bit trick below is wrong otherwise.
    debug_assert!(
        alignment.is_power_of_two(),
        "alignment {} is not a power of two",
        alignment
    );
    Some(offset.checked_add(alignment - 1)? & !(alignment - 1))
}

/// The loud rejection for a declaration whose byte size or packed
/// placement overflows the u64 address space. Every declare-time sum and
/// product below is checked against this (release-active): a hostile
/// declaration like `.shared .u32 A[0x4000000000000001]` would otherwise
/// wrap to byte size 4 and be silently mismodeled.
fn size_overflow(kind: &str, name: &str) -> LowerError {
    LowerError::VariableSizeOverflow {
        what: format!("{} '{}'", kind, name),
    }
}

/// Symbol table built during lowering
#[derive(Debug, Clone)]
pub struct SymbolTable {
    /// All declared names mapped to their kind (for duplicate checking).
    /// PTX uses a single namespace for all identifiers within a function.
    all_names: HashMap<String, SymbolKind>,

    /// Register name → info
    registers: HashMap<String, RegInfo>,
    /// Count of registers per class
    reg_counts: [u32; RegClass::COUNT],

    /// Parameter name → info
    params: HashMap<String, ParamInfo>,
    /// Parameters in declaration order
    params_ordered: IdVec<ParamId, ParamInfo>,

    /// Label name → instruction PC
    labels: HashMap<String, InstrId>,

    /// Shared memory allocations. Extern variables only appear here once
    /// `finalize_shared_layout` has placed them.
    shared_vars: HashMap<String, SharedMemInfo>,
    /// Total static shared memory size (packing cursor for statics). The
    /// dynamic extern window begins at `extern_shared_base()`, at or after
    /// this offset.
    shared_mem_size: u64,
    /// Has extern shared memory
    has_extern_shared: bool,
    /// Placement state of the `.extern .shared` window
    extern_layout: ExternSharedLayout,

    /// Function-scope local memory allocations (per-thread space)
    local_vars: HashMap<String, LocalMemInfo>,
    /// Total local memory size per thread
    local_mem_size: u64,

    /// Module-scope `.global` variables
    global_vars: HashMap<String, GlobalVarInfo>,
    /// Bytes of the reserved module-global region allocated so far
    module_global_size: u64,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            all_names: HashMap::new(),
            registers: HashMap::new(),
            reg_counts: [0; RegClass::COUNT],
            params: HashMap::new(),
            params_ordered: IdVec::new(),
            labels: HashMap::new(),
            shared_vars: HashMap::new(),
            shared_mem_size: 0,
            has_extern_shared: false,
            extern_layout: ExternSharedLayout::Collecting(Vec::new()),
            local_vars: HashMap::new(),
            local_mem_size: 0,
            global_vars: HashMap::new(),
            module_global_size: 0,
        }
    }

    /// Reserve a name in the global namespace.
    /// Returns an error if the name is already in use.
    fn reserve_name(&mut self, name: &str, kind: SymbolKind) -> LowerResult<()> {
        if let Some(&existing) = self.all_names.get(name) {
            return Err(LowerError::DuplicateName {
                name: name.to_string(),
                existing,
                attempted: kind,
            });
        }
        self.all_names.insert(name.to_string(), kind);
        Ok(())
    }

    /// Declare a register or range of registers
    ///
    /// For parameterized registers like `%r<100>`, this declares %r0 through %r99.
    /// Returns the RegId of the first register declared.
    pub fn declare_register(
        &mut self,
        base_name: &str,
        ty: ScalarType,
        count: u32,
    ) -> Result<RegId, LowerError> {
        let class = RegClass::from_scalar_type(ty);

        // PTX scopes declarations to blocks, but we flatten them. nvcc
        // re-declares helper registers like `temp_param_reg` in every callseq
        // block, so an identical single-register re-declaration is idempotent.
        if count == 1
            && let Some(existing) = self.registers.get(base_name)
            && existing.declared_type == ty
        {
            return Ok(existing.id);
        }

        let base_index = self.reg_counts[class as usize];

        // Parse the base name (strip angle brackets if present)
        let base = base_name.trim_end_matches(|c: char| c == '<' || c == '>' || c.is_ascii_digit());

        for i in 0..count {
            let name = if count == 1 {
                base_name.to_string()
            } else {
                format!("{}{}", base, i)
            };

            self.reserve_name(&name, SymbolKind::Register)?;

            let id = RegId::new(class, base_index + i);
            self.registers.insert(
                name.clone(),
                RegInfo {
                    id,
                    name,
                    declared_type: ty,
                },
            );
        }

        self.reg_counts[class as usize] += count;
        Ok(RegId::new(class, base_index))
    }

    /// Declare a kernel parameter
    /// Returns the ParamId of the declared parameter.
    pub fn declare_param(
        &mut self,
        name: &str,
        ty: ScalarType,
        size_bytes: u64,
        ptr: Option<Ptr>,
    ) -> LowerResult<ParamId> {
        self.reserve_name(name, SymbolKind::Parameter)?;

        let id = ParamId(self.params_ordered.len() as u32);
        let info = ParamInfo {
            id,
            name: name.to_string(),
            ty,
            size_bytes,
            ptr,
        };
        self.params.insert(name.to_string(), info.clone());
        let _ = self.params_ordered.push(info);
        Ok(id)
    }

    /// Declare a label at the given PC
    pub fn declare_label(&mut self, name: &str, pc: InstrId) -> LowerResult<()> {
        self.reserve_name(name, SymbolKind::Label)?;
        self.labels.insert(name.to_string(), pc);
        Ok(())
    }

    /// Declare shared memory variable.
    ///
    /// Static variables are packed at increasing offsets in declaration
    /// order. Extern (`.extern .shared`) variables are only *recorded* here:
    /// they all alias the one dynamic-shared window, which the CUDA ABI
    /// bases after every static allocation, so their offset cannot be known
    /// until `finalize_shared_layout` runs.
    pub fn declare_shared(
        &mut self,
        name: &str,
        element_ty: ScalarType,
        num_elements: u64,
        is_extern: bool,
        alignment: u64,
    ) -> LowerResult<()> {
        if matches!(self.extern_layout, ExternSharedLayout::Finalized { .. }) {
            panic!("declare_shared({name}) after finalize_shared_layout");
        }

        self.reserve_name(name, SymbolKind::SharedVariable)?;

        let elem_size = (element_ty.bits() as u64).div_ceil(8);
        let alignment = alignment.max(elem_size);

        if is_extern {
            self.has_extern_shared = true;
            let ExternSharedLayout::Collecting(pending) = &mut self.extern_layout else {
                unreachable!("state checked above");
            };
            pending.push(PendingExternShared {
                name: name.to_string(),
                element_ty,
                alignment,
            });
            return Ok(());
        }

        let size_bytes = num_elements
            .checked_mul(elem_size)
            .ok_or_else(|| size_overflow("shared variable", name))?;
        let offset = align_up(self.shared_mem_size, alignment)
            .ok_or_else(|| size_overflow("shared variable", name))?;
        let end = offset
            .checked_add(size_bytes)
            .ok_or_else(|| size_overflow("shared variable", name))?;

        self.shared_vars.insert(
            name.to_string(),
            SharedMemInfo {
                name: name.to_string(),
                offset,
                size_bytes,
                element_ty,
                is_extern: false,
            },
        );

        self.shared_mem_size = end;
        Ok(())
    }

    /// Place the `.extern .shared` window and enter the finalized state.
    ///
    /// All extern shared declarations alias one dynamic window that the CUDA
    /// ABI bases immediately after every static allocation, aligned to at
    /// least 16 bytes and to every extern's declared alignment. Must be
    /// called exactly once, after all shared declarations are collected and
    /// before any shared symbol is resolved to an address; declaring shared
    /// variables after this call panics. Errors when the window's aligned
    /// base overflows the address space (hostile static sizes).
    pub fn finalize_shared_layout(&mut self) -> LowerResult<()> {
        let pending = match &mut self.extern_layout {
            ExternSharedLayout::Collecting(pending) => std::mem::take(pending),
            ExternSharedLayout::Finalized { .. } => {
                panic!("finalize_shared_layout called twice")
            }
        };

        let alignment = pending.iter().fold(16, |a, e| a.max(e.alignment));
        let extern_base = align_up(self.shared_mem_size, alignment).ok_or_else(|| {
            LowerError::VariableSizeOverflow {
                what: format!(
                    ".extern .shared window (based after {:#x} static shared bytes)",
                    self.shared_mem_size
                ),
            }
        })?;

        for e in pending {
            self.shared_vars.insert(
                e.name.clone(),
                SharedMemInfo {
                    name: e.name,
                    offset: extern_base,
                    size_bytes: 0, // Size supplied at launch (dynamic shared bytes)
                    element_ty: e.element_ty,
                    is_extern: true,
                },
            );
        }

        self.extern_layout = ExternSharedLayout::Finalized { extern_base };
        Ok(())
    }

    /// Base offset of the dynamic (`.extern .shared`) window, or `None` if
    /// the kernel declares no extern shared memory. The window spans from
    /// this base for however many dynamic shared bytes the launch supplies.
    ///
    /// Panics if `finalize_shared_layout` has not run: before then no
    /// placement exists to report.
    pub fn extern_shared_base(&self) -> Option<u64> {
        match self.extern_layout {
            ExternSharedLayout::Finalized { extern_base } => {
                self.has_extern_shared.then_some(extern_base)
            }
            ExternSharedLayout::Collecting(_) => {
                panic!("extern_shared_base read before finalize_shared_layout")
            }
        }
    }

    /// Declare a function-scope local memory variable (per-thread space)
    pub fn declare_local(
        &mut self,
        name: &str,
        element_ty: ScalarType,
        num_elements: u64,
        alignment: u64,
    ) -> LowerResult<()> {
        self.reserve_name(name, SymbolKind::LocalVariable)?;

        let elem_size = (element_ty.bits() as u64).div_ceil(8);
        let alignment = alignment.max(elem_size).max(1);
        let size_bytes = num_elements
            .checked_mul(elem_size)
            .ok_or_else(|| size_overflow("local variable", name))?;
        let offset = align_up(self.local_mem_size, alignment)
            .ok_or_else(|| size_overflow("local variable", name))?;
        let end = offset
            .checked_add(size_bytes)
            .ok_or_else(|| size_overflow("local variable", name))?;

        self.local_vars.insert(
            name.to_string(),
            LocalMemInfo {
                name: name.to_string(),
                offset,
                size_bytes,
                element_ty,
            },
        );

        self.local_mem_size = end;
        Ok(())
    }

    /// Declare a module-scope `.global` variable, assigning it an address in
    /// the reserved module-global region.
    pub fn declare_global_var(
        &mut self,
        name: &str,
        element_ty: ScalarType,
        num_elements: u64,
        alignment: u64,
    ) -> LowerResult<()> {
        self.reserve_name(name, SymbolKind::GlobalVariable)?;

        let elem_size = (element_ty.bits() as u64).div_ceil(8);
        let alignment = alignment.max(elem_size).max(1);
        let size_bytes = num_elements
            .checked_mul(elem_size)
            .ok_or_else(|| size_overflow("global variable", name))?;
        let offset = align_up(self.module_global_size, alignment)
            .ok_or_else(|| size_overflow("global variable", name))?;
        let end = offset
            .checked_add(size_bytes)
            .ok_or_else(|| size_overflow("global variable", name))?;
        // The variable must also fit above the reserved base:
        // `MODULE_GLOBAL_BASE + end` is exactly the window end
        // `Interpreter::new` computes (and the region end `push_region`
        // re-checks), so its representability is established here once.
        if MODULE_GLOBAL_BASE.checked_add(end).is_none() {
            return Err(size_overflow("global variable", name));
        }

        self.global_vars.insert(
            name.to_string(),
            GlobalVarInfo {
                name: name.to_string(),
                addr: MODULE_GLOBAL_BASE + offset,
                size_bytes,
                element_ty,
            },
        );

        self.module_global_size = end;
        Ok(())
    }

    /// Look up a register, returning full info for type checking
    pub fn get_register(&self, name: &str) -> Option<&RegInfo> {
        self.registers.get(name)
    }

    /// Look up just the RegId
    pub fn resolve_register(&self, name: &str) -> Option<RegId> {
        self.registers.get(name).map(|info| info.id)
    }

    /// Check if name is a special register
    pub fn resolve_special_reg(&self, name: &str) -> Option<SpecialRegKind> {
        SpecialRegKind::from_name(name)
    }

    /// Look up a parameter
    pub fn get_param(&self, name: &str) -> Option<&ParamInfo> {
        self.params.get(name)
    }

    /// Look up a label
    pub fn resolve_label(&self, name: &str) -> Option<InstrId> {
        self.labels.get(name).copied()
    }

    /// Look up shared memory variable
    pub fn get_shared_var(&self, name: &str) -> Option<&SharedMemInfo> {
        self.shared_vars.get(name)
    }

    /// Iterate over all shared memory variables
    pub fn shared_vars(&self) -> impl Iterator<Item = &SharedMemInfo> {
        self.shared_vars.values()
    }

    /// Iterate over all function-scope local memory variables
    pub fn local_vars(&self) -> impl Iterator<Item = &LocalMemInfo> {
        self.local_vars.values()
    }

    /// Look up a function-scope local memory variable
    pub fn get_local_var(&self, name: &str) -> Option<&LocalMemInfo> {
        self.local_vars.get(name)
    }

    /// Total local memory size per thread
    pub fn local_mem_size(&self) -> u64 {
        self.local_mem_size
    }

    /// Look up a module-scope `.global` variable
    pub fn get_global_var(&self, name: &str) -> Option<&GlobalVarInfo> {
        self.global_vars.get(name)
    }

    /// Iterate over all module-scope `.global` variables
    pub fn global_vars(&self) -> impl Iterator<Item = &GlobalVarInfo> {
        self.global_vars.values()
    }

    /// Bytes of the reserved module-global window in use: the module-scope
    /// `.global` variables occupy `[MODULE_GLOBAL_BASE, MODULE_GLOBAL_BASE
    /// + module_global_size())`. The window end is representable in u64
    /// (`declare_global_var` checks it for every variable it places).
    pub fn module_global_size(&self) -> u64 {
        self.module_global_size
    }

    /// Get register name from ID (for error messages)
    pub fn register_name(&self, id: RegId) -> Option<&str> {
        self.registers
            .values()
            .find(|info| info.id == id)
            .map(|info| info.name.as_str())
    }

    /// Get register count for a class
    pub fn register_count(&self, class: RegClass) -> u32 {
        self.reg_counts[class as usize]
    }

    /// Get all register counts
    pub fn register_counts(&self) -> RegCounts {
        RegCounts::new(self.reg_counts)
    }

    /// Get parameters in order
    pub fn params(&self) -> &IdVec<ParamId, ParamInfo> {
        &self.params_ordered
    }

    /// Get parameter by ID
    pub fn get_param_by_id(&self, id: ParamId) -> Option<&ParamInfo> {
        self.params_ordered.get(id)
    }

    /// Get total *static* shared memory size. The dynamic (`.extern
    /// .shared`) window is not included: it begins at `extern_shared_base()`
    /// and its size is only known at launch (dynamic shared bytes), so any
    /// capacity accounting for a kernel with extern shared memory must use
    /// `extern_shared_base() + dynamic bytes` as the end of shared memory.
    pub fn shared_mem_size(&self) -> u64 {
        self.shared_mem_size
    }

    /// Check if kernel uses dynamic (extern) shared memory
    pub fn has_extern_shared(&self) -> bool {
        self.has_extern_shared
    }

    /// Find similar register names for error suggestions
    pub fn find_similar_registers(&self, name: &str) -> Vec<String> {
        self.registers
            .keys()
            .filter(|k| levenshtein_distance(k, name) <= 2)
            .take(3)
            .cloned()
            .collect()
    }

    /// Get all register names (for debugging)
    pub fn all_register_names(&self) -> impl Iterator<Item = &str> {
        self.registers.keys().map(|s| s.as_str())
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple Levenshtein distance for typo suggestions
fn levenshtein_distance(a: &str, b: &str) -> usize {
    if a == b {
        return 0;
    }
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }

    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let mut prev: Vec<usize> = (0..=b_chars.len()).collect();
    let mut curr = vec![0; b_chars.len() + 1];

    for (i, ca) in a_chars.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b_chars.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[b_chars.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_declaration() {
        let mut symbols = SymbolTable::new();
        symbols.declare_register("%r", ScalarType::U32, 10).unwrap();

        // Should have %r0 through %r9
        assert!(symbols.get_register("%r0").is_some());
        assert!(symbols.get_register("%r9").is_some());
        assert!(symbols.get_register("%r10").is_none());

        // All should be U32
        assert_eq!(
            symbols.get_register("%r5").unwrap().declared_type,
            ScalarType::U32
        );
    }

    #[test]
    fn test_single_register() {
        let mut symbols = SymbolTable::new();
        symbols
            .declare_register("%myReg", ScalarType::F32, 1)
            .unwrap();

        assert!(symbols.get_register("%myReg").is_some());
        assert_eq!(
            symbols.get_register("%myReg").unwrap().declared_type,
            ScalarType::F32
        );
    }

    #[test]
    fn test_duplicate_register_error() {
        let mut symbols = SymbolTable::new();
        symbols.declare_register("%r", ScalarType::U32, 5).unwrap();

        // Try to declare overlapping range
        let result = symbols.declare_register("%r", ScalarType::U32, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_special_registers() {
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
        assert_eq!(SpecialRegKind::from_name("%unknown"), None);
    }

    #[test]
    fn test_labels() {
        use id_collections::Id;
        let mut symbols = SymbolTable::new();
        symbols
            .declare_label("LOOP", InstrId::from_index(10))
            .unwrap();
        symbols
            .declare_label("END", InstrId::from_index(50))
            .unwrap();

        assert_eq!(symbols.resolve_label("LOOP"), Some(InstrId::from_index(10)));
        assert_eq!(symbols.resolve_label("END"), Some(InstrId::from_index(50)));
        assert_eq!(symbols.resolve_label("MISSING"), None);
    }

    #[test]
    fn test_duplicate_label_error() {
        use id_collections::Id;
        let mut symbols = SymbolTable::new();
        symbols
            .declare_label("LOOP", InstrId::from_index(10))
            .unwrap();

        // Try to declare same label again
        let result = symbols.declare_label("LOOP", InstrId::from_index(20));
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_param_error() {
        let mut symbols = SymbolTable::new();
        symbols
            .declare_param("ptr", ScalarType::U64, 8, None)
            .unwrap();

        // Try to declare same param again
        let result = symbols.declare_param("ptr", ScalarType::U64, 8, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_shared_error() {
        let mut symbols = SymbolTable::new();
        symbols
            .declare_shared("smem", ScalarType::F32, 256, false, 4)
            .unwrap();

        // Try to declare same shared var again
        let result = symbols.declare_shared("smem", ScalarType::F32, 128, false, 4);
        assert!(result.is_err());
    }

    // =========================================================================
    // Hostile declaration sizes: every packing sum/product is checked
    // =========================================================================

    /// The review's shape: `.shared .u32 A[0x4000000000000001]` has a byte
    /// size of 2^64 + 4, which release-wraps to 4. Each packer entry point
    /// must reject the overflow loudly instead (checked arithmetic is
    /// release-active).
    #[test]
    fn test_hostile_element_count_rejected_per_space() {
        let huge = 0x4000_0000_0000_0001; // * 4 bytes wraps u64 to 4
        let mut symbols = SymbolTable::new();
        assert!(matches!(
            symbols.declare_shared("a", ScalarType::U32, huge, false, 4),
            Err(LowerError::VariableSizeOverflow { .. })
        ));
        assert!(matches!(
            symbols.declare_local("b", ScalarType::U32, huge, 4),
            Err(LowerError::VariableSizeOverflow { .. })
        ));
        assert!(matches!(
            symbols.declare_global_var("c", ScalarType::U32, huge, 4),
            Err(LowerError::VariableSizeOverflow { .. })
        ));
    }

    /// Placement overflow: a size that multiplies fine can still push the
    /// packing cursor (or the module-global window end) past u64.
    #[test]
    fn test_hostile_placement_overflow_rejected() {
        // The second variable's aligned offset + size runs off the end.
        let mut symbols = SymbolTable::new();
        symbols
            .declare_shared("big", ScalarType::B8, u64::MAX - 8, false, 1)
            .unwrap();
        assert!(matches!(
            symbols.declare_shared("next", ScalarType::B8, 16, false, 1),
            Err(LowerError::VariableSizeOverflow { .. })
        ));

        // A module global whose size fits u64 but whose *address range*
        // `MODULE_GLOBAL_BASE + end` does not.
        let mut symbols = SymbolTable::new();
        assert!(matches!(
            symbols.declare_global_var("d", ScalarType::B8, 0x9000_0000_0000_0001, 1),
            Err(LowerError::VariableSizeOverflow { .. })
        ));

        // The extern window's aligned base overflows past hostile statics.
        let mut symbols = SymbolTable::new();
        symbols
            .declare_shared("big", ScalarType::B8, u64::MAX - 7, false, 1)
            .unwrap();
        symbols
            .declare_shared("dyn", ScalarType::B8, 1, true, 1)
            .unwrap();
        assert!(matches!(
            symbols.finalize_shared_layout(),
            Err(LowerError::VariableSizeOverflow { .. })
        ));
    }

    // =========================================================================
    // Shared memory layout: statics pack from 0, the extern window follows
    // =========================================================================

    /// The bug shape: an extern declared *before* the statics (module-level
    /// declarations are collected first) must still be placed after them.
    #[test]
    fn test_extern_shared_placed_after_statics() {
        let mut symbols = SymbolTable::new();
        symbols
            .declare_shared("buf", ScalarType::B8, 1, true, 16)
            .unwrap();
        symbols
            .declare_shared("lut", ScalarType::F32, 64, false, 4)
            .unwrap();
        symbols.finalize_shared_layout().unwrap();

        let lut = symbols.get_shared_var("lut").unwrap();
        assert_eq!((lut.offset, lut.size_bytes), (0, 256));
        let buf = symbols.get_shared_var("buf").unwrap();
        assert!(buf.is_extern);
        assert_eq!(buf.offset, 256);
        assert_eq!(symbols.extern_shared_base(), Some(256));
        assert_eq!(symbols.shared_mem_size(), 256);
    }

    /// With no statics the extern window stays at offset 0 (the pre-fix
    /// placement for every extern-only kernel in the corpus).
    #[test]
    fn test_extern_only_shared_base_zero() {
        let mut symbols = SymbolTable::new();
        symbols
            .declare_shared("buf", ScalarType::B8, 1, true, 16)
            .unwrap();
        symbols.finalize_shared_layout().unwrap();

        assert_eq!(symbols.get_shared_var("buf").unwrap().offset, 0);
        assert_eq!(symbols.extern_shared_base(), Some(0));
        assert_eq!(symbols.shared_mem_size(), 0);
    }

    /// All extern declarations alias one buffer: same base for every name,
    /// aligned to the largest declared alignment (at least 16).
    #[test]
    fn test_extern_shared_common_base_and_alignment() {
        let mut symbols = SymbolTable::new();
        symbols
            .declare_shared("a", ScalarType::B8, 1, true, 4)
            .unwrap();
        symbols
            .declare_shared("b", ScalarType::B8, 1, true, 128)
            .unwrap();
        // 20 bytes of statics; the window rounds up to the 128 boundary.
        symbols
            .declare_shared("lut", ScalarType::F32, 5, false, 4)
            .unwrap();
        symbols.finalize_shared_layout().unwrap();

        assert_eq!(symbols.extern_shared_base(), Some(128));
        assert_eq!(symbols.get_shared_var("a").unwrap().offset, 128);
        assert_eq!(symbols.get_shared_var("b").unwrap().offset, 128);
    }

    /// Even a byte-aligned extern window is based at a 16-byte boundary.
    #[test]
    fn test_extern_shared_base_min_16_alignment() {
        let mut symbols = SymbolTable::new();
        symbols
            .declare_shared("buf", ScalarType::B8, 1, true, 1)
            .unwrap();
        symbols
            .declare_shared("x", ScalarType::U32, 1, false, 4)
            .unwrap();
        symbols.finalize_shared_layout().unwrap();

        assert_eq!(symbols.extern_shared_base(), Some(16));
    }

    /// Packing is naturally aligned in every variable space: after an
    /// odd-sized `.b8` variable, wider variables still land on multiples of
    /// their element size (the `declare_*` functions raise the alignment to
    /// at least the element size before `align_up`). This is what lets the
    /// evaluator's per-access natural-alignment checks trust
    /// lowering-assigned shared/local/module-global addresses.
    #[test]
    fn test_var_packing_is_naturally_aligned() {
        let mut symbols = SymbolTable::new();
        symbols
            .declare_shared("pad", ScalarType::B8, 3, false, 1)
            .unwrap();
        symbols
            .declare_shared("halves", ScalarType::F16, 5, false, 2)
            .unwrap();
        symbols
            .declare_shared("words", ScalarType::F32, 2, false, 4)
            .unwrap();
        symbols.finalize_shared_layout().unwrap();
        // pad: [0, 3); halves: aligned up to 4, [4, 14); words: up to 16.
        assert_eq!(symbols.get_shared_var("pad").unwrap().offset, 0);
        assert_eq!(symbols.get_shared_var("halves").unwrap().offset, 4);
        assert_eq!(symbols.get_shared_var("words").unwrap().offset, 16);

        symbols
            .declare_local("depot", ScalarType::B8, 5, 1)
            .unwrap();
        symbols
            .declare_local("spill", ScalarType::F64, 1, 8)
            .unwrap();
        assert_eq!(symbols.get_local_var("depot").unwrap().offset, 0);
        assert_eq!(symbols.get_local_var("spill").unwrap().offset, 8);

        symbols
            .declare_global_var("flag", ScalarType::B8, 1, 1)
            .unwrap();
        symbols
            .declare_global_var("table", ScalarType::F64, 2, 8)
            .unwrap();
        // MODULE_GLOBAL_BASE is itself naturally aligned for any scalar
        // width, so the offsets' alignment carries over to the addresses.
        assert_eq!(MODULE_GLOBAL_BASE % 16, 0);
        assert_eq!(symbols.get_global_var("flag").unwrap().addr % 8, 0);
        assert_eq!(
            symbols.get_global_var("table").unwrap().addr,
            MODULE_GLOBAL_BASE + 8
        );
    }

    /// Static-only tables finalize to "no extern window".
    #[test]
    fn test_finalize_without_externs() {
        let mut symbols = SymbolTable::new();
        symbols
            .declare_shared("lut", ScalarType::F32, 64, false, 4)
            .unwrap();
        symbols.finalize_shared_layout().unwrap();

        assert_eq!(symbols.extern_shared_base(), None);
        assert_eq!(symbols.get_shared_var("lut").unwrap().offset, 0);
    }

    #[test]
    #[should_panic(expected = "after finalize_shared_layout")]
    fn test_declare_shared_after_finalize_panics() {
        let mut symbols = SymbolTable::new();
        symbols.finalize_shared_layout().unwrap();
        let _ = symbols.declare_shared("late", ScalarType::F32, 1, false, 4);
    }

    #[test]
    #[should_panic(expected = "before finalize_shared_layout")]
    fn test_extern_base_before_finalize_panics() {
        let symbols = SymbolTable::new();
        let _ = symbols.extern_shared_base();
    }

    #[test]
    fn test_levenshtein() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        assert_eq!(levenshtein_distance("hello", "hallo"), 1);
        assert_eq!(levenshtein_distance("hello", ""), 5);
        assert_eq!(levenshtein_distance("", "hello"), 5);
    }

    // =========================================================================
    // Cross-namespace conflict tests (PTX uses a single namespace)
    // =========================================================================

    #[test]
    fn test_label_conflicts_with_param() {
        use id_collections::Id;
        let mut symbols = SymbolTable::new();
        symbols
            .declare_param("foo", ScalarType::U64, 8, None)
            .unwrap();

        // Label should conflict with param
        let result = symbols.declare_label("foo", InstrId::from_index(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_label_conflicts_with_register() {
        use id_collections::Id;
        let mut symbols = SymbolTable::new();
        symbols.declare_register("foo", ScalarType::U32, 1).unwrap();

        // Label should conflict with register
        let result = symbols.declare_label("foo", InstrId::from_index(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_label_conflicts_with_shared() {
        use id_collections::Id;
        let mut symbols = SymbolTable::new();
        symbols
            .declare_shared("foo", ScalarType::U32, 256, false, 4)
            .unwrap();

        // Label should conflict with shared
        let result = symbols.declare_label("foo", InstrId::from_index(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_param_conflicts_with_register() {
        let mut symbols = SymbolTable::new();
        symbols.declare_register("foo", ScalarType::U32, 1).unwrap();

        // Param should conflict with register
        let result = symbols.declare_param("foo", ScalarType::U64, 8, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_shared_conflicts_with_param() {
        let mut symbols = SymbolTable::new();
        symbols
            .declare_param("foo", ScalarType::U64, 8, None)
            .unwrap();

        // Shared should conflict with param
        let result = symbols.declare_shared("foo", ScalarType::U32, 256, false, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_register_conflicts_with_label() {
        use id_collections::Id;
        let mut symbols = SymbolTable::new();
        symbols
            .declare_label("foo", InstrId::from_index(0))
            .unwrap();

        // Register should conflict with label
        let result = symbols.declare_register("foo", ScalarType::U32, 1);
        assert!(result.is_err());
    }
}
