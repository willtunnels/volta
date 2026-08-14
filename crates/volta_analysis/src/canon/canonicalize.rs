//! Lowering evaluator expressions into canonical rationals.
//!
//! One bottom-up pass over the `ExprArena` DAG, memoized per `ExprId` (the
//! paper's "traverses each node exactly once"). Both kernels of a VC pair
//! canonicalize into the same session, distinguished by `Side`, so shared
//! structure interns identically across sides and elements.
//!
//! Memory discipline: results are interned (and memoized) only for nodes
//! with more than one parent in the expression DAG, plus roots and exp/atom
//! arguments. Single-use chain intermediates flow through as transient
//! owned vectors and are freed by scope - without this, a K-step
//! accumulator chain would permanently retain O(K^2) interned prefixes.

use crate::symbolic::{ExprArena, ExprId, ExprNode};

use super::arena::{Atom, PolyId, Rat, Term, UninterpOp};
use super::coeff::Coeff;
use super::ops::{PV, RatV};
use super::{CanonError, Session, Side};

/// Count each node's parents over the whole arena (a conservative
/// overapproximation of sharing among the nodes canonicalization will
/// visit). Depends only on the arena, so callers checking many elements
/// (or checking on several worker threads) can compute this once and
/// hand every session the same counts via
/// [`Session::provide_ref_counts`].
pub fn parent_counts(arena: &ExprArena) -> Vec<u32> {
    let n = arena.node_count();
    let mut counts = vec![0u32; n];
    for i in 0..n {
        arena.node(ExprId(i as u32)).for_each_child(|child| {
            counts[child.0 as usize] = counts[child.0 as usize].saturating_add(1);
        });
    }
    counts
}

impl Session {
    /// Canonicalize `e` (from the `side` kernel's arena) into an interned
    /// rational. Roots are always interned and memoized.
    pub fn canonicalize(
        &mut self,
        side: Side,
        arena: &ExprArena,
        e: ExprId,
    ) -> Result<Rat, CanonError> {
        self.ensure_ref_counts(side, arena);
        if let Some(&r) = self.expr_memo.get(&(side, e)) {
            return Ok(r);
        }
        let rv = self.canon_value(side, arena, e)?;
        let rat = self.ratv_intern(rv);
        self.expr_memo.insert((side, e), rat);
        Ok(rat)
    }

    /// Fill the side's parent counts if they aren't already provided
    /// (via [`Session::provide_ref_counts`]) or computed.
    fn ensure_ref_counts(&mut self, side: Side, arena: &ExprArena) {
        let counts = self.ref_counts.entry(side).or_default();
        if counts.len() >= arena.node_count() {
            return;
        }
        *counts = std::sync::Arc::new(parent_counts(arena));
    }

    fn is_shared(&self, side: Side, e: ExprId) -> bool {
        self.ref_counts
            .get(&side)
            .and_then(|c| c.get(e.0 as usize))
            .is_some_and(|&n| n >= 2)
    }

    /// Canonicalize to a possibly-transient rational value.
    fn canon_value(
        &mut self,
        side: Side,
        arena: &ExprArena,
        e: ExprId,
    ) -> Result<RatV, CanonError> {
        if let Some(&r) = self.expr_memo.get(&(side, e)) {
            return Ok(RatV::from_rat(r));
        }
        let rv = stacker::maybe_grow(64 * 1024, 8 * 1024 * 1024, || {
            self.canon_value_inner(side, arena, e)
        })?;
        // Shared nodes are interned + memoized so each is computed once;
        // single-use nodes pass their transient value straight to the parent.
        if self.is_shared(side, e) {
            let rat = self.ratv_intern(rv);
            self.expr_memo.insert((side, e), rat);
            return Ok(RatV::from_rat(rat));
        }
        Ok(rv)
    }

    fn canon_value_inner(
        &mut self,
        side: Side,
        arena: &ExprArena,
        e: ExprId,
    ) -> Result<RatV, CanonError> {
        let node = arena.node(e);
        match node {
            // Atoms
            ExprNode::IntConst(v) => {
                let p = self.arena.const_poly(Coeff::from_int(*v));
                Ok(RatV::from_rat(self.arena.rat_poly(p)))
            }
            ExprNode::RealConst(v) => {
                let p = self.arena.const_poly(Coeff::from_real(v)?);
                Ok(RatV::from_rat(self.arena.rat_poly(p)))
            }
            ExprNode::BoolConst(b) => {
                let p = self.arena.const_poly(Coeff::from_int(*b as i64));
                Ok(RatV::from_rat(self.arena.rat_poly(p)))
            }
            ExprNode::ParamSymbol(_) | ExprNode::InputElement { .. } | ExprNode::Symbol(_) => {
                let sym = node
                    .symbol_ref(arena)
                    .expect("symbolic atom must have a symbol_ref");
                let p = self.arena.symbol_poly(sym);
                Ok(RatV::from_rat(self.arena.rat_poly(p)))
            }

            // Arithmetic
            ExprNode::Add(a, b) => {
                let ra = self.canon_value(side, arena, *a)?;
                let rb = self.canon_value(side, arena, *b)?;
                self.rat_add_v(ra, rb)
            }
            ExprNode::Sub(a, b) => {
                let ra = self.canon_value(side, arena, *a)?;
                let rb = self.canon_value(side, arena, *b)?;
                self.rat_sub_v(ra, rb)
            }
            ExprNode::Mul(a, b) => {
                let ra = self.canon_value(side, arena, *a)?;
                let rb = self.canon_value(side, arena, *b)?;
                self.rat_mul_v(ra, rb)
            }
            ExprNode::Div(a, b) => {
                let ra = self.canon_value(side, arena, *a)?;
                let rb = self.canon_value(side, arena, *b)?;
                self.rat_div_v(ra, rb)
            }
            ExprNode::Neg(a) => {
                let ra = self.canon_value(side, arena, *a)?;
                self.rat_neg_v(ra)
            }
            ExprNode::Fma(a, b, c) => {
                let ra = self.canon_value(side, arena, *a)?;
                let rb = self.canon_value(side, arena, *b)?;
                let prod = self.rat_mul_v(ra, rb)?;
                let rc = self.canon_value(side, arena, *c)?;
                self.rat_add_v(prod, rc)
            }
            ExprNode::Rcp(a) => {
                let ra = self.canon_value(side, arena, *a)?;
                let one = RatV::from_rat(self.arena.rat_poly(self.arena.one));
                self.rat_div_v(one, ra)
            }

            // Exponential: a polynomial argument becomes a fused exp factor
            // (the argument itself must be interned - it is part of term
            // identity); a genuinely rational argument stays an opaque atom.
            ExprNode::Exp(a) => {
                let ra = self.canon_value(side, arena, *a)?;
                if self.pv_is_one(&ra.denom) {
                    let arg = self.pv_intern(ra.numer);
                    Ok(self.exp_poly(arg))
                } else {
                    let arg = self.ratv_intern(ra);
                    let p = self.arena.atom_poly(Atom::Uninterp {
                        op: UninterpOp::Exp,
                        args: vec![arg],
                    });
                    Ok(RatV::from_rat(self.arena.rat_poly(p)))
                }
            }

            // Max/Min: flattened, sorted, deduplicated opaque atoms.
            ExprNode::Max(a, b) => self.canonicalize_maxmin(side, arena, true, *a, *b),
            ExprNode::Min(a, b) => self.canonicalize_maxmin(side, arena, false, *a, *b),

            // Uninterpreted operations: sound, incomplete (equal only when
            // syntactically identical) - matching the paper's implementation.
            ExprNode::Rem(a, b) => self.uninterp2(side, arena, UninterpOp::Rem, *a, *b),
            ExprNode::Log(a) => self.uninterp1(side, arena, UninterpOp::Log, *a),
            ExprNode::Sqrt(a) => self.uninterp1(side, arena, UninterpOp::Sqrt, *a),
            ExprNode::Abs(a) => self.uninterp1(side, arena, UninterpOp::Abs, *a),
            ExprNode::BitAnd(a, b) => self.uninterp2(side, arena, UninterpOp::BitAnd, *a, *b),
            ExprNode::BitOr(a, b) => self.uninterp2(side, arena, UninterpOp::BitOr, *a, *b),
            ExprNode::BitXor(a, b) => self.uninterp2(side, arena, UninterpOp::BitXor, *a, *b),
            ExprNode::BitNot(a) => self.uninterp1(side, arena, UninterpOp::BitNot, *a),
            ExprNode::Shl(a, b) => self.uninterp2(side, arena, UninterpOp::Shl, *a, *b),
            ExprNode::Shr(a, b) => self.uninterp2(side, arena, UninterpOp::Shr, *a, *b),
            ExprNode::LShr(a, b) => self.uninterp2(side, arena, UninterpOp::LShr, *a, *b),
            ExprNode::Eq(a, b) => self.uninterp2(side, arena, UninterpOp::Eq, *a, *b),
            ExprNode::Ne(a, b) => self.uninterp2(side, arena, UninterpOp::Ne, *a, *b),
            ExprNode::Lt(a, b) => self.uninterp2(side, arena, UninterpOp::Lt, *a, *b),
            ExprNode::Le(a, b) => self.uninterp2(side, arena, UninterpOp::Le, *a, *b),
            ExprNode::Gt(a, b) => self.uninterp2(side, arena, UninterpOp::Gt, *a, *b),
            ExprNode::Ge(a, b) => self.uninterp2(side, arena, UninterpOp::Ge, *a, *b),
            ExprNode::And(a, b) => self.uninterp2(side, arena, UninterpOp::And, *a, *b),
            ExprNode::Or(a, b) => self.uninterp2(side, arena, UninterpOp::Or, *a, *b),
            ExprNode::Not(a) => self.uninterp1(side, arena, UninterpOp::Not, *a),
            ExprNode::Select(c, t, f) => {
                let rc = self.canon_value(side, arena, *c)?;
                let rc = self.ratv_intern(rc);
                let rt = self.canon_value(side, arena, *t)?;
                let rt = self.ratv_intern(rt);
                let rf = self.canon_value(side, arena, *f)?;
                let rf = self.ratv_intern(rf);
                let p = self.arena.atom_poly(Atom::Uninterp {
                    op: UninterpOp::Select,
                    args: vec![rc, rt, rf],
                });
                Ok(RatV::from_rat(self.arena.rat_poly(p)))
            }
            ExprNode::SymbolicRead { array, index } => {
                let ri = self.canon_value(side, arena, *index)?;
                let ri = self.ratv_intern(ri);
                let p = self.arena.atom_poly(Atom::Uninterp {
                    op: UninterpOp::ArrayRead(arena.string(*array).to_string()),
                    args: vec![ri],
                });
                Ok(RatV::from_rat(self.arena.rat_poly(p)))
            }

            // Conversions: identity over the reals.
            ExprNode::ToFloat(a) => self.canon_value(side, arena, *a),

            ExprNode::Discarded | ExprNode::Undefined => Err(CanonError::Undefined),
        }
    }

    /// A single fused exponential `e^{p}` as a rational.
    fn exp_poly(&mut self, p: PolyId) -> RatV {
        if p == self.arena.zero {
            return RatV::from_rat(self.arena.rat_poly(self.arena.one)); // e^0 = 1
        }
        let term = self.arena.terms.intern(Term {
            factors: Vec::new(),
            exp: Some(p),
        });
        RatV {
            numer: PV::Owned(vec![(term, Coeff::ONE)]),
            denom: PV::Id(self.arena.one),
        }
    }

    /// Canonicalize max/min: splice nested atoms of the same kind
    /// (`max(max(a,b),c) = max(a,b,c)`), sort, dedup (`max(x,x) = x`).
    fn canonicalize_maxmin(
        &mut self,
        side: Side,
        arena: &ExprArena,
        is_max: bool,
        a: ExprId,
        b: ExprId,
    ) -> Result<RatV, CanonError> {
        let ra = self.canon_value(side, arena, a)?;
        let ra = self.ratv_intern(ra);
        let rb = self.canon_value(side, arena, b)?;
        let rb = self.ratv_intern(rb);

        let mut args = Vec::new();
        for r in [ra, rb] {
            match self.as_bare_maxmin(r, is_max) {
                Some(inner) => args.extend(inner),
                None => args.push(r),
            }
        }
        args.sort_by_key(|r| (r.numer.0, r.denom.0));
        args.dedup();

        if args.len() == 1 {
            return Ok(RatV::from_rat(args[0]));
        }
        let p = self.arena.atom_poly(Atom::MaxMin { is_max, args });
        Ok(RatV::from_rat(self.arena.rat_poly(p)))
    }

    /// If `r` is exactly one bare max/min atom of the given kind, return its
    /// argument list (for flattening).
    fn as_bare_maxmin(&self, r: Rat, want_max: bool) -> Option<Vec<Rat>> {
        if r.denom != self.arena.one {
            return None;
        }
        let poly = self.arena.polys.get(r.numer);
        let [(term_id, coeff)] = poly.as_slice() else {
            return None;
        };
        if !coeff.is_one() {
            return None;
        }
        let term = self.arena.terms.get(*term_id);
        if term.exp.is_some() {
            return None;
        }
        let [(factor_id, 1)] = term.factors.as_slice() else {
            return None;
        };
        let super::arena::Factor::Atom(atom_id) = self.arena.factors.get(*factor_id) else {
            return None;
        };
        match self.arena.atoms.get(*atom_id) {
            Atom::MaxMin { is_max, args } if *is_max == want_max => Some(args.clone()),
            _ => None,
        }
    }

    fn uninterp1(
        &mut self,
        side: Side,
        arena: &ExprArena,
        op: UninterpOp,
        a: ExprId,
    ) -> Result<RatV, CanonError> {
        let ra = self.canon_value(side, arena, a)?;
        let ra = self.ratv_intern(ra);
        let p = self.arena.atom_poly(Atom::Uninterp { op, args: vec![ra] });
        Ok(RatV::from_rat(self.arena.rat_poly(p)))
    }

    fn uninterp2(
        &mut self,
        side: Side,
        arena: &ExprArena,
        op: UninterpOp,
        a: ExprId,
        b: ExprId,
    ) -> Result<RatV, CanonError> {
        let ra = self.canon_value(side, arena, a)?;
        let ra = self.ratv_intern(ra);
        let rb = self.canon_value(side, arena, b)?;
        let rb = self.ratv_intern(rb);
        let p = self.arena.atom_poly(Atom::Uninterp {
            op,
            args: vec![ra, rb],
        });
        Ok(RatV::from_rat(self.arena.rat_poly(p)))
    }
}
