//! Warp-cooperative operations: `bar.warp.sync`, `shfl.sync`, `ldmatrix`,
//! `mma.sync`, and the `wmma` family.
//!
//! Each of these is modeled as the paper's `sync I` for the participating
//! lane group, followed by the operation's cooperative data movement /
//! compute. The group fires only once every participating lane has arrived
//! at the same pc with the same mask (see `Interpreter::find_ready_warp_group`).
//!
//! Memory reads/writes performed by these ops are attributed to the exact
//! lane that owns each fragment element (per the PTX fragment tables in
//! `tensor_core`), so race checking stays byte- and thread-precise.

use volta_frontend::ast::ScalarType;

use crate::eval::error::{EvalError, EvalResult};
use crate::eval::interp::Interpreter;
use crate::eval::value::Value;
use crate::eval::{ThreadId, WARP_SIZE};
use crate::lowered::{InstrId, LoweredInstr, MemSpace, Operand, ShflMode};
use crate::symbolic::ExprId;
use crate::symbols::RegId;
use crate::tensor_core::{
    FragmentElement, MmaLayout, MmaOperand, MmaShape, m16n8k16_f16, m16n16k16_f16,
};
use crate::types::ScalarTypeExt;

/// A dense matrix of expressions being assembled from lane fragments.
struct Grid {
    cols: usize,
    cells: Vec<Option<ExprId>>,
}

impl Grid {
    fn new(rows: usize, cols: usize) -> Self {
        Self {
            cols,
            cells: vec![None; rows * cols],
        }
    }

    fn set(&mut self, row: u32, col: u32, e: ExprId) {
        self.cells[row as usize * self.cols + col as usize] = Some(e);
    }

    fn get(&self, row: u32, col: u32, pc: InstrId) -> EvalResult<ExprId> {
        self.cells[row as usize * self.cols + col as usize].ok_or_else(|| EvalError::Unsupported {
            pc,
            what: format!("incomplete fragment mapping at ({}, {})", row, col),
        })
    }
}

/// Element offset (in elements) within a matrix laid out in memory.
fn elem_offset(layout: MmaLayout, row: u32, col: u32, stride: u64) -> u64 {
    match layout {
        MmaLayout::Row => row as u64 * stride + col as u64,
        MmaLayout::Col => col as u64 * stride + row as u64,
    }
}

/// `ldmatrix` row alignment. PTX ISA 9.7.14.5.15: "a group of four
/// consecutive threads loads 16 bytes. The matrix addresses must be
/// naturally aligned accordingly" - every 8x8 b16 matrix row is one
/// 16-byte access, so each lane-supplied row address must be a multiple
/// of 16.
const LDMATRIX_ROW_BYTES: u64 = 16;

/// `wmma` base-address alignment for m16n16k16, the one supported shape:
/// the fragment size in bytes. PTX ISA 9.7.14.4.2 (Matrix Storage for
/// WMMA, "Address Alignment"): "The starting address of each instance of
/// the leading dimension (row or column) must be aligned with the size of
/// the corresponding fragment in bytes"; the section's example requires
/// "p is a multiple of 32". Every supported m16n16k16 fragment is 32
/// bytes (a/b are eight .f16x2 registers, c/d eight .f32 registers), and
/// the CUDA API documents the same base contract ("mptr must be a 256-bit
/// aligned pointer").
const WMMA_BASE_ALIGN_BYTES: u64 = 32;

/// `wmma` stride granularity: the stride in *bytes* must be a multiple of
/// 16. The ISA example goes further ("2*s is a multiple of 32", making
/// every leading-dimension instance fragment-aligned), but nvcc's own
/// emission does not honor that reading: bank-conflict-skewed tiles
/// (`__shared__ half tile[..][16 + 8]`, as in the corpus's Conv2D-opt)
/// compile to f16 strides of 24 elements = 48 bytes. What nvcc enforces
/// is the CUDA API's stride contract - ldm "must be a multiple of 8 for
/// __half element type or multiple of 4 for float element type", i.e. 16
/// bytes either way - which keeps every leading-dimension instance
/// aligned for the hardware's 16-byte row fetches. Requiring 32 here
/// would reject valid, hardware-correct nvcc output (verified on the
/// corpus: conv's f16 wmma ops all run at a 48-byte pitch from 32-aligned
/// bases).
const WMMA_STRIDE_ALIGN_BYTES: u64 = 16;

/// Minimum legal `wmma.load`/`wmma.store` stride, in matrix elements. PTX
/// ISA 9.7.14.4.3: the stride defaults to the matrix's leading dimension
/// and "specifying a value lower than the default value results in
/// undefined behavior"; larger values (a submatrix of a larger matrix) are
/// fine. For m16n16k16 the leading dimension is 16 elements for every
/// operand (a/b/c/d) and both layouts (the spec's default-stride table).
const WMMA_MIN_STRIDE_ELEMS: i64 = 16;

impl Interpreter<'_> {
    /// Execute a complete warp group blocked at `pc` with lane mask `mask`.
    /// `members` holds the *live* lanes (converged at `pc`); exited mask
    /// lanes arrive implicitly (see `find_ready_warp_group`).
    pub(in crate::eval) fn execute_warp_op(
        &mut self,
        pc: InstrId,
        mask: u32,
        members: &[ThreadId],
    ) -> EvalResult<()> {
        let instr = self
            .program
            .instruction(pc)
            .expect("warp group blocked at a valid pc")
            .clone();

        // The sync set is the paper's full I: every mask lane, exited lanes
        // included - `syncMem(I, X)` clears pending sets over the whole I,
        // and a lane that already returned is still in I. Syncing the live
        // members alone would leave an exited lane's pre-exit access
        // pending against the survivors and report a race the paper's
        // semantics does not have. Only live members execute and advance.
        // Every mask lane exists: `find_ready_warp_group` already rejected
        // masks naming lanes beyond the CTA, and the leader is always a
        // live member, so `members` is nonempty.
        let warp_base = (members[0].0 / WARP_SIZE) * WARP_SIZE;
        let group: Vec<ThreadId> = (0..WARP_SIZE)
            .filter(|lane| mask & (1u32 << lane) != 0)
            .map(|lane| ThreadId(warp_base + lane))
            .collect();

        // Preconditions run before the chi-clear and stats bump below, so
        // a rejected op provably mutates nothing (structural: every
        // `EvalError` aborts the whole analysis today, so this is not
        // observable, but rejection-implies-no-mutation should not rest
        // on that).
        self.check_warp_op_preconditions(pc, &instr, members)?;

        // Every warp-cooperative op is a synchronization point for its group
        // (the paper's `sync I`). The group's accesses are bracketed by
        // syncs: the sync *before* keeps the op's reads from racing with the
        // lanes' own pre-op writes (ldmatrix after st.shared), and the sync
        // *after* (below) keeps the op's writes from racing with the lanes'
        // post-op reads (wmma.store followed by per-lane ld.shared) - a
        // converged warp is synchronized on both sides of the op.
        self.stats.warp_syncs += 1;
        self.sync_warp_group(&group);

        match &instr {
            LoweredInstr::BarWarpSync { .. } => {}
            LoweredInstr::ShflSync {
                mode,
                dst,
                dst_pred,
                src,
                offset_or_lane,
                clamp,
                ..
            } => {
                self.exec_shfl_sync(
                    pc,
                    mask,
                    members,
                    *mode,
                    *dst,
                    *dst_pred,
                    src,
                    offset_or_lane,
                    clamp,
                )?;
            }
            LoweredInstr::Ldmatrix {
                dst,
                base,
                offset,
                num,
                ..
            } => {
                self.exec_ldmatrix(pc, members, dst, base, *offset, *num)?;
            }
            LoweredInstr::Mma { .. } => self.exec_mma(pc, members, &instr)?,
            LoweredInstr::WmmaLoad { .. } => self.exec_wmma_load(pc, members, &instr)?,
            LoweredInstr::WmmaStore { .. } => self.exec_wmma_store(pc, members, &instr)?,
            LoweredInstr::WmmaMma { .. } => self.exec_wmma_mma(pc, members, &instr)?,
            other => unreachable!("{:?} passed warp-op preconditions", other),
        }

        self.sync_warp_group(&group);
        self.advance_warp_group(members);
        Ok(())
    }

    /// Every precondition that rejects a warp op outright - the
    /// unknown-op arm, unsupported instruction forms, and the tensor-core
    /// ops' fully-live-warp requirement - checked from the instruction and
    /// the live-member list alone. `execute_warp_op` runs this *before*
    /// mutating any interpreter state (the group's chi-clear, the stats
    /// bump), so a rejection provably changes nothing. Errors that arise
    /// mid-execution (out-of-bounds, races, non-concrete operands, ...)
    /// necessarily come after the chi-clear; soundness there rests on
    /// every `EvalError` aborting the whole analysis.
    fn check_warp_op_preconditions(
        &self,
        pc: InstrId,
        instr: &LoweredInstr,
        members: &[ThreadId],
    ) -> EvalResult<()> {
        match instr {
            // Both handle exited lanes per-op (Undefined shfl source data,
            // arrived-at-sync semantics), so a partial warp is fine.
            LoweredInstr::BarWarpSync { .. } | LoweredInstr::ShflSync { .. } => Ok(()),
            LoweredInstr::Ldmatrix {
                dst, num, trans, ..
            } => {
                if *trans {
                    return Err(EvalError::Unsupported {
                        pc,
                        what: "ldmatrix .trans".to_string(),
                    });
                }
                // Covers the exited address-supplying lane in particular:
                // lane `i*8 + r` holds row r's address in a register, and
                // an exited lane's registers are not observable, so the
                // row's footprint cannot be modeled.
                self.require_live_warp(pc, members, "ldmatrix")?;
                if dst.len() != *num as usize {
                    return Err(EvalError::Unsupported {
                        pc,
                        what: format!("ldmatrix x{} with {} destination registers", num, dst.len()),
                    });
                }
                Ok(())
            }
            LoweredInstr::Mma {
                shape,
                a_layout,
                b_layout,
                ..
            } => {
                if *shape != MmaShape::new(16, 8, 16) {
                    return Err(EvalError::Unsupported {
                        pc,
                        what: format!("mma shape {}", shape),
                    });
                }
                if (*a_layout, *b_layout) != (MmaLayout::Row, MmaLayout::Col) {
                    return Err(EvalError::Unsupported {
                        pc,
                        what: "mma with non-row.col layouts".to_string(),
                    });
                }
                // Every lane owns a/b/c fragment elements the product needs.
                self.require_live_warp(pc, members, "mma.sync")
            }
            LoweredInstr::WmmaLoad { shape, .. }
            | LoweredInstr::WmmaStore { shape, .. }
            | LoweredInstr::WmmaMma { shape, .. } => self.check_wmma_shape(pc, members, *shape),
            other => Err(EvalError::Unsupported {
                pc,
                what: format!("warp-op dispatch for {:?}", other),
            }),
        }
    }

    /// Reject a tensor-core cooperative op whose warp has exited lanes.
    /// The ISA defines each of `ldmatrix`, `mma`, `wmma.load`,
    /// `wmma.store`, and `wmma.mma` as undefined "if any thread in the
    /// warp has exited" (their respective Descriptions, PTX ISA 9.1), so -
    /// as for misalignment - the undefined behavior is rejected loudly
    /// rather than silently modeled: an exited lane's addresses and
    /// fragment registers are unknowable, leaving no byte-precise access
    /// attribution. These ops block with the full-warp mask, and
    /// `find_ready_warp_group` already rejects masks naming lanes beyond
    /// the CTA, so fewer than 32 live members means exited lanes.
    fn require_live_warp(&self, pc: InstrId, members: &[ThreadId], what: &str) -> EvalResult<()> {
        if members.len() != WARP_SIZE as usize {
            return Err(EvalError::WarpMismatch {
                pc,
                reason: format!(
                    "{} requires all 32 warp lanes live, but only {} are \
                     (the PTX ISA defines the instruction as undefined if \
                     any thread in the warp has exited)",
                    what,
                    members.len()
                ),
            });
        }
        Ok(())
    }

    /// `shfl.sync`: exchange register values within the mask group,
    /// following the PTX ISA source-lane computation.
    #[allow(clippy::too_many_arguments)]
    fn exec_shfl_sync(
        &mut self,
        pc: InstrId,
        mask: u32,
        members: &[ThreadId],
        mode: ShflMode,
        dst: RegId,
        dst_pred: Option<RegId>,
        src: &Operand,
        offset_or_lane: &Operand,
        clamp: &Operand,
    ) -> EvalResult<()> {
        // Gather every lane's source value first.
        let mut lane_src: [Option<Value>; WARP_SIZE as usize] = [None; WARP_SIZE as usize];
        for &m in members {
            let lane = (m.0 % WARP_SIZE) as usize;
            lane_src[lane] = Some(self.operand_value(m, pc, src)?);
        }

        // Compute each lane's source lane per the PTX ISA pseudocode.
        let mut results: Vec<(ThreadId, bool, Value)> = Vec::with_capacity(members.len());
        for &m in members {
            let lane = (m.0 % WARP_SIZE) as i64;
            let b = self.concrete_operand(m, pc, offset_or_lane, "shfl.sync lane operand")?;
            let c = self.concrete_operand(m, pc, clamp, "shfl.sync clamp operand")?;
            let bval = b & 0x1f;
            let cval = c & 0x1f;
            let segmask = (c >> 8) & 0x1f;
            let max_lane = (lane & segmask) | (cval & !segmask);
            let min_lane = lane & segmask;
            let (j0, pval) = match mode {
                ShflMode::Up => {
                    let j = lane - bval;
                    (j, j >= max_lane)
                }
                ShflMode::Down => {
                    let j = lane + bval;
                    (j, j <= max_lane)
                }
                ShflMode::Bfly => {
                    let j = lane ^ bval;
                    (j, j <= max_lane)
                }
                ShflMode::Idx => {
                    let j = min_lane | (bval & !segmask);
                    (j, j <= max_lane)
                }
            };
            let j = if pval { j0 } else { lane };
            debug_assert!((0..WARP_SIZE as i64).contains(&j));
            if mask & (1u32 << j) == 0 {
                return Err(EvalError::WarpMismatch {
                    pc,
                    reason: format!(
                        "lane {} reads lane {} which is outside mask {:#010x}",
                        lane, j, mask
                    ),
                });
            }
            let value = match lane_src[j as usize] {
                Some(v) => v,
                // Source lane in the mask but exited: "results are
                // undefined if a thread sources a register from an
                // inactive thread" (shfl.sync Description) - the lazily
                // erroring `Undefined` is that exact model, failing only
                // if the value reaches an output or a concreteness point.
                // The ISA pseudocode sets `p = pval` from the pure lane
                // arithmetic without consulting the source's activity, so
                // the predicate keeps its computed value here (in
                // particular it is *true* for an in-range exited source,
                // unlike the out-of-segment case).
                None => Value::Scalar(self.arena.undefined()),
            };
            results.push((m, pval, value));
        }

        for (m, pval, value) in results {
            self.threads[m].regs.write(dst, value);
            if let Some(p) = dst_pred {
                let b = self.arena.bool_val(pval);
                self.threads[m].regs.write(p, Value::Scalar(b));
            }
        }
        Ok(())
    }

    /// `ldmatrix.sync.aligned.xN.m8n8.shared.b16`: cooperative load of N
    /// 8x8 b16 matrices. Lane `i*8 + r` supplies the address of row `r` of
    /// matrix `i` (its `base + offset`); lane `l` receives elements (row
    /// `l/4`, cols `(l%4)*2`, `(l%4)*2+1`) of each matrix as a packed pair.
    ///
    /// `check_warp_op_preconditions` already established: no `.trans`, all
    /// 32 lanes live, `dst.len() == num`.
    fn exec_ldmatrix(
        &mut self,
        pc: InstrId,
        members: &[ThreadId],
        dst: &[RegId],
        base: &Operand,
        offset: i64,
        num: u32,
    ) -> EvalResult<()> {
        // Row addresses come from the first num*8 lanes. Each row is loaded
        // by a group of four lanes as one 16-byte access, so every row
        // address must be 16-byte aligned (see `LDMATRIX_ROW_BYTES`).
        let mut row_addr = vec![[0u64; 8]; num as usize];
        for i in 0..num as usize {
            for r in 0..8 {
                let m = members[i * 8 + r];
                let a = self.concrete_operand(m, pc, base, "ldmatrix row address")?;
                let a = Self::offset_addr(a, offset);
                self.check_alignment(m, pc, MemSpace::Shared, a, LDMATRIX_ROW_BYTES)?;
                row_addr[i][r] = a;
            }
        }

        for &m in members {
            let lane = m.0 % WARP_SIZE;
            for (i, reg) in dst.iter().enumerate() {
                // Wrap-free in every profile: the row address passed the
                // 16-byte alignment check above, so it is at most
                // `u64::MAX - 15`, and the lane offset is at most 12.
                let byte = row_addr[i][(lane / 4) as usize] + (lane % 4) as u64 * 4;
                let v = self.mem_read(m, pc, MemSpace::Shared, byte, 4)?;
                self.threads[m].regs.write(*reg, v);
            }
        }
        Ok(())
    }

    /// `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
    ///
    /// `check_warp_op_preconditions` already established: m16n8k16 shape,
    /// row.col layouts, all 32 lanes live.
    fn exec_mma(
        &mut self,
        pc: InstrId,
        members: &[ThreadId],
        instr: &LoweredInstr,
    ) -> EvalResult<()> {
        let LoweredInstr::Mma {
            dst,
            src_a,
            src_b,
            src_c,
            ..
        } = instr
        else {
            unreachable!()
        };

        let mut a = Grid::new(16, 16);
        let mut b = Grid::new(16, 8);
        let mut c = Grid::new(16, 8);
        for &m in members {
            let lane = m.0 % WARP_SIZE;
            self.gather_f16_fragment(pc, m, src_a, &m16n8k16_f16::matrix_a(lane), &mut a)?;
            self.gather_f16_fragment(pc, m, src_b, &m16n8k16_f16::matrix_b(lane), &mut b)?;
            self.gather_f32_fragment(pc, m, src_c, &m16n8k16_f16::matrix_cd(lane), &mut c)?;
        }

        let d = self.matmul_acc(pc, &a, &b, &c, 16, 8, 16)?;

        for &m in members {
            let lane = m.0 % WARP_SIZE;
            for elem in m16n8k16_f16::matrix_cd(lane) {
                let e = d.get(elem.row, elem.col, pc)?;
                self.threads[m]
                    .regs
                    .write(dst[elem.reg_idx], Value::Scalar(e));
            }
        }
        Ok(())
    }

    /// `wmma.load.{a,b,c}.sync.aligned.{row,col}.m16n16k16{.shared}.{f16,f32}`.
    ///
    /// `check_warp_op_preconditions` already established the m16n16k16
    /// shape and all 32 lanes live (`check_wmma_shape`).
    fn exec_wmma_load(
        &mut self,
        pc: InstrId,
        members: &[ThreadId],
        instr: &LoweredInstr,
    ) -> EvalResult<()> {
        let LoweredInstr::WmmaLoad {
            operand,
            layout,
            dst,
            base,
            offset,
            stride,
            elem_type,
            space,
            ..
        } = instr
        else {
            unreachable!()
        };

        let base = self.uniform_concrete(pc, members, base, "wmma.load address")?;
        let base = Self::offset_addr(base, *offset);
        let stride = self.uniform_concrete(pc, members, stride, "wmma.load stride")?;
        let stride =
            self.check_wmma_addressing(pc, members[0], *space, base, stride, *elem_type)?;

        match operand {
            MmaOperand::A | MmaOperand::B => {
                if !matches!(elem_type, ScalarType::F16 | ScalarType::B16) {
                    return Err(EvalError::Unsupported {
                        pc,
                        what: format!("wmma.load a/b with element type {:?}", elem_type),
                    });
                }
                for &m in members {
                    let lane = m.0 % WARP_SIZE;
                    let elems = match operand {
                        MmaOperand::A => m16n16k16_f16::matrix_a_row(lane),
                        _ => m16n16k16_f16::matrix_b_row(lane),
                    };
                    let mut lo: Vec<Option<ExprId>> = vec![None; dst.len()];
                    let mut hi: Vec<Option<ExprId>> = vec![None; dst.len()];
                    for elem in &elems {
                        let off = elem_offset(*layout, elem.row, elem.col, stride);
                        let byte = base + off * 2;
                        let v = self.mem_read(m, pc, *space, byte, 2)?;
                        let Value::Scalar(e) = v else {
                            return Err(EvalError::ValueKindMismatch {
                                thread: m,
                                pc,
                                what: "wmma.load f16 element is not a scalar",
                            });
                        };
                        if elem.high_half == Some(true) {
                            hi[elem.reg_idx] = Some(e);
                        } else {
                            lo[elem.reg_idx] = Some(e);
                        }
                    }
                    for (r, reg) in dst.iter().enumerate() {
                        let (Some(l), Some(h)) = (lo[r], hi[r]) else {
                            return Err(EvalError::Unsupported {
                                pc,
                                what: "incomplete wmma fragment".to_string(),
                            });
                        };
                        self.threads[m].regs.write(*reg, Value::Pair(l, h));
                    }
                }
            }
            MmaOperand::C => {
                for &m in members {
                    let lane = m.0 % WARP_SIZE;
                    for elem in m16n16k16_f16::matrix_cd_f32(lane) {
                        let off = elem_offset(*layout, elem.row, elem.col, stride);
                        let byte = base + off * 4;
                        let v = self.mem_read(m, pc, *space, byte, 4)?;
                        let Value::Scalar(e) = v else {
                            return Err(EvalError::ValueKindMismatch {
                                thread: m,
                                pc,
                                what: "wmma.load f32 element is not a scalar",
                            });
                        };
                        self.threads[m]
                            .regs
                            .write(dst[elem.reg_idx], Value::Scalar(e));
                    }
                }
            }
            MmaOperand::D => {
                return Err(EvalError::Unsupported {
                    pc,
                    what: "wmma.load.d".to_string(),
                });
            }
        }
        Ok(())
    }

    /// `wmma.store.d.sync.aligned.{row,col}.m16n16k16{.shared}.f32`.
    ///
    /// `check_warp_op_preconditions` already established the m16n16k16
    /// shape and all 32 lanes live (`check_wmma_shape`).
    fn exec_wmma_store(
        &mut self,
        pc: InstrId,
        members: &[ThreadId],
        instr: &LoweredInstr,
    ) -> EvalResult<()> {
        let LoweredInstr::WmmaStore {
            layout,
            src,
            base,
            offset,
            stride,
            elem_type,
            space,
            ..
        } = instr
        else {
            unreachable!()
        };

        let base = self.uniform_concrete(pc, members, base, "wmma.store address")?;
        let base = Self::offset_addr(base, *offset);
        let stride = self.uniform_concrete(pc, members, stride, "wmma.store stride")?;
        let stride =
            self.check_wmma_addressing(pc, members[0], *space, base, stride, *elem_type)?;

        for &m in members {
            let lane = m.0 % WARP_SIZE;
            for elem in m16n16k16_f16::matrix_cd_f32(lane) {
                let v = self.read_reg(m, pc, src[elem.reg_idx])?;
                let off = elem_offset(*layout, elem.row, elem.col, stride);
                let byte = base + off * 4;
                self.mem_write(m, pc, *space, byte, 4, v)?;
            }
        }
        Ok(())
    }

    /// `wmma.mma.sync.aligned.{row,col}.{row,col}.m16n16k16.f32.f32`.
    ///
    /// The layouts describe how A/B were loaded; the fragments themselves
    /// are opaque, so the compute only needs the fragment position maps.
    /// `check_warp_op_preconditions` already established the m16n16k16
    /// shape and all 32 lanes live (`check_wmma_shape`).
    fn exec_wmma_mma(
        &mut self,
        pc: InstrId,
        members: &[ThreadId],
        instr: &LoweredInstr,
    ) -> EvalResult<()> {
        let LoweredInstr::WmmaMma {
            dst,
            src_a,
            src_b,
            src_c,
            ..
        } = instr
        else {
            unreachable!()
        };

        let mut a = Grid::new(16, 16);
        let mut b = Grid::new(16, 16);
        let mut c = Grid::new(16, 16);
        for &m in members {
            let lane = m.0 % WARP_SIZE;
            self.gather_f16_fragment(pc, m, src_a, &m16n16k16_f16::matrix_a_row(lane), &mut a)?;
            self.gather_f16_fragment(pc, m, src_b, &m16n16k16_f16::matrix_b_row(lane), &mut b)?;
            self.gather_f32_fragment(pc, m, src_c, &m16n16k16_f16::matrix_cd_f32(lane), &mut c)?;
        }

        let d = self.matmul_acc(pc, &a, &b, &c, 16, 16, 16)?;

        for &m in members {
            let lane = m.0 % WARP_SIZE;
            for elem in m16n16k16_f16::matrix_cd_f32(lane) {
                let e = d.get(elem.row, elem.col, pc)?;
                self.threads[m]
                    .regs
                    .write(dst[elem.reg_idx], Value::Scalar(e));
            }
        }
        Ok(())
    }

    // =====================================================================
    // Shared helpers
    // =====================================================================

    /// Precondition of every wmma op (called from
    /// `check_warp_op_preconditions`, before any state changes): the one
    /// supported m16n16k16 shape, with all 32 lanes live.
    fn check_wmma_shape(
        &self,
        pc: InstrId,
        members: &[ThreadId],
        shape: MmaShape,
    ) -> EvalResult<()> {
        if shape != MmaShape::new(16, 16, 16) {
            return Err(EvalError::Unsupported {
                pc,
                what: format!("wmma shape {}", shape),
            });
        }
        // Every lane owns fragment elements (and the loads/stores need the
        // whole warp's footprint).
        self.require_live_warp(pc, members, "wmma")?;
        Ok(())
    }

    /// Validate a `wmma.load`/`wmma.store` base address and stride against
    /// the m16n16k16 matrix-storage rules, returning the stride for offset
    /// arithmetic. The base must be fragment-aligned (see
    /// `WMMA_BASE_ALIGN_BYTES`); leading-dimension instance `r` starts at
    /// `base + r * elem_bytes * stride`, so the stride in bytes must keep
    /// every instance 16-byte aligned (see `WMMA_STRIDE_ALIGN_BYTES`); and
    /// a stride below the leading dimension is undefined behavior outright
    /// (see `WMMA_MIN_STRIDE_ELEMS`).
    fn check_wmma_addressing(
        &self,
        pc: InstrId,
        lead: ThreadId,
        space: MemSpace,
        base: u64,
        stride: i64,
        elem_type: ScalarType,
    ) -> EvalResult<u64> {
        if stride < WMMA_MIN_STRIDE_ELEMS {
            return Err(EvalError::WmmaStrideTooSmall {
                pc,
                stride,
                minimum: WMMA_MIN_STRIDE_ELEMS as u64,
            });
        }
        let stride = stride as u64;
        self.check_alignment(lead, pc, space, base, WMMA_BASE_ALIGN_BYTES)?;
        // The whole 16x16 fragment footprint must stay within the u64
        // address space (release-active, computed in u128 so the check
        // itself cannot overflow): the per-element sums below -
        // `base + (row*stride + col) * elem_bytes` with row/col <= 15 -
        // are then provably wrap-free, so no wrapped intermediate can
        // reach the bounds checker as an unrelated small address.
        let elem_bytes = elem_type.size_bytes() as u128;
        let max_off = 15u128 * stride as u128 + 15;
        if base as u128 + max_off * elem_bytes > u64::MAX as u128 {
            return Err(EvalError::Unsupported {
                pc,
                what: format!(
                    "wmma footprint from base {:#x} with stride {} overflows the address space",
                    base, stride
                ),
            });
        }
        let row_bytes = elem_type.size_bytes() as u64 * stride;
        if !row_bytes.is_multiple_of(WMMA_STRIDE_ALIGN_BYTES) {
            // With `base` aligned, the first misaligned leading-dimension
            // instance is the second one, at `base + row_bytes`.
            return Err(EvalError::Misaligned {
                thread: lead,
                pc,
                space,
                addr: base + row_bytes,
                required: WMMA_STRIDE_ALIGN_BYTES,
            });
        }
        Ok(stride)
    }

    /// Resolve an operand that must be concrete and identical on every lane.
    fn uniform_concrete(
        &mut self,
        pc: InstrId,
        members: &[ThreadId],
        op: &Operand,
        what: &'static str,
    ) -> EvalResult<i64> {
        let mut result: Option<i64> = None;
        for &m in members {
            let v = self.concrete_operand(m, pc, op, what)?;
            match result {
                None => result = Some(v),
                Some(prev) if prev == v => {}
                Some(prev) => {
                    return Err(EvalError::WarpMismatch {
                        pc,
                        reason: format!("{} differs across lanes ({} vs {})", what, prev, v),
                    });
                }
            }
        }
        result.ok_or(EvalError::WarpMismatch {
            pc,
            reason: "empty warp group".to_string(),
        })
    }

    /// Place one lane's packed-f16 fragment registers into a matrix grid.
    fn gather_f16_fragment(
        &mut self,
        pc: InstrId,
        m: ThreadId,
        regs: &[RegId],
        elems: &[FragmentElement],
        grid: &mut Grid,
    ) -> EvalResult<()> {
        for elem in elems {
            let v = self.read_reg(m, pc, regs[elem.reg_idx])?;
            let Value::Pair(lo, hi) = v else {
                return Err(EvalError::ValueKindMismatch {
                    thread: m,
                    pc,
                    what: "matrix fragment register does not hold a packed f16 pair",
                });
            };
            let e = if elem.high_half == Some(true) { hi } else { lo };
            grid.set(elem.row, elem.col, e);
        }
        Ok(())
    }

    /// Place one lane's f32 accumulator fragment registers into a grid.
    fn gather_f32_fragment(
        &mut self,
        pc: InstrId,
        m: ThreadId,
        regs: &[RegId],
        elems: &[FragmentElement],
        grid: &mut Grid,
    ) -> EvalResult<()> {
        for elem in elems {
            let v = self.read_reg(m, pc, regs[elem.reg_idx])?;
            let Value::Scalar(e) = v else {
                return Err(EvalError::ValueKindMismatch {
                    thread: m,
                    pc,
                    what: "accumulator fragment register holds a packed pair",
                });
            };
            grid.set(elem.row, elem.col, e);
        }
        Ok(())
    }

    /// `D = A * B + C` over the arena, as an fma chain per element.
    #[allow(clippy::too_many_arguments)] // internal helper; the args are the mma shape
    fn matmul_acc(
        &mut self,
        pc: InstrId,
        a: &Grid,
        b: &Grid,
        c: &Grid,
        m: u32,
        n: u32,
        k: u32,
    ) -> EvalResult<Grid> {
        let mut d = Grid::new(m as usize, n as usize);
        for i in 0..m {
            for j in 0..n {
                let mut acc = c.get(i, j, pc)?;
                for kk in 0..k {
                    let av = a.get(i, kk, pc)?;
                    let bv = b.get(kk, j, pc)?;
                    acc = self.arena.fma(av, bv, acc);
                }
                d.set(i, j, acc);
            }
        }
        Ok(d)
    }
}
