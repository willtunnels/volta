//! Equivalence checking of symbolic expressions.
//!
//! A thin wrapper over the `canon` decision procedure (see
//! `docs/canonicalizer-plan.md`). Use `EquivSession` when checking many
//! elements of the same kernel pair: the session's intern tables and memos
//! make shared structure (score polynomials, softmax denominators, the
//! other 63 columns of the row) free after the first element.

use std::sync::Arc;

use crate::canon::{CanonError, Session, Side, parent_counts};
use crate::logging::info;
use crate::symbolic::{ExprArena, ExprId};

/// Errors from equivalence checking.
#[derive(Debug)]
pub enum EquivError {
    /// The decision procedure failed (undefined value, coefficient
    /// overflow, or an oversized VC).
    Canon(CanonError),
}

impl From<CanonError> for EquivError {
    fn from(e: CanonError) -> Self {
        Self::Canon(e)
    }
}

impl std::fmt::Display for EquivError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Canon(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for EquivError {}

pub type EquivResult<T> = Result<T, EquivError>;

/// A reusable equivalence-checking session for one kernel pair, fixed at
/// construction: session memos key on `(Side, ExprId)`, and an `ExprId`
/// is only meaningful within its own arena, so `check` takes element ids
/// only and cannot be handed a different pair.
///
/// Sessions self-limit their memory: when the intern tables grow past a
/// bound (chain-heavy VCs intern millions of intermediate terms per
/// element), the tables are dropped and rebuilt lazily from the still-live
/// `ExprArena`s. This trades some re-canonicalization of shared structure
/// for a hard cap on resident memory.
pub struct EquivSession<'a> {
    session: Session,
    recycle_terms: usize,
    arena1: &'a ExprArena,
    arena2: &'a ExprArena,
    /// Parent counts for the two arenas (see `canon::parent_counts`) -
    /// held here so recycling reuses them instead of rescanning the
    /// (possibly GiB-scale) arenas, and so parallel callers can hand
    /// every worker session the same computation.
    counts1: Arc<Vec<u32>>,
    counts2: Arc<Vec<u32>>,
}

/// Default recycle bound. Bytes per term are workload-dependent: polynomial
/// terms (matmul) run a few hundred bytes, exp-heavy attention terms average
/// 2-4 KB, so 4M terms retains roughly 1-16 GiB.
pub const DEFAULT_RECYCLE_TERMS: usize = 4_000_000;

impl<'a> EquivSession<'a> {
    pub fn new(arena1: &'a ExprArena, arena2: &'a ExprArena) -> Self {
        Self::with_recycle_terms(arena1, arena2, DEFAULT_RECYCLE_TERMS)
    }

    /// A session that recycles its intern tables once they exceed
    /// `recycle_terms` interned terms (`0` = never recycle). Lower values
    /// bound resident memory; each recycle re-canonicalizes structure that
    /// later elements would otherwise share.
    pub fn with_recycle_terms(
        arena1: &'a ExprArena,
        arena2: &'a ExprArena,
        recycle_terms: usize,
    ) -> Self {
        Self::with_shared_counts(
            arena1,
            arena2,
            recycle_terms,
            Arc::new(parent_counts(arena1)),
            Arc::new(parent_counts(arena2)),
        )
    }

    /// Like [`with_recycle_terms`](Self::with_recycle_terms), but with the
    /// arenas' parent counts (`canon::parent_counts`, in the same arena
    /// order) precomputed by the caller - for building many sessions over
    /// the same arena pair (e.g. one per parallel worker) without each
    /// rescanning the arenas or duplicating the count vectors.
    pub fn with_shared_counts(
        arena1: &'a ExprArena,
        arena2: &'a ExprArena,
        recycle_terms: usize,
        counts1: Arc<Vec<u32>>,
        counts2: Arc<Vec<u32>>,
    ) -> Self {
        assert_eq!(counts1.len(), arena1.node_count(), "counts1/arena1 mismatch");
        assert_eq!(counts2.len(), arena2.node_count(), "counts2/arena2 mismatch");
        Self {
            session: fresh_session(&counts1, &counts2),
            recycle_terms,
            arena1,
            arena2,
            counts1,
            counts2,
        }
    }

    /// Check whether `e1` (in the first arena) and `e2` (in the second)
    /// are equivalent over the reals.
    pub fn check(&mut self, e1: ExprId, e2: ExprId) -> EquivResult<bool> {
        if self.recycle_terms != 0 && self.session.interned_terms() > self.recycle_terms {
            info!(
                "recycling VC session at {} interned terms",
                self.session.interned_terms()
            );
            self.session = fresh_session(&self.counts1, &self.counts2);
        }
        Ok(self
            .session
            .check_equivalent(self.arena1, e1, self.arena2, e2)?)
    }
}

/// A new canon session with the pair's parent counts installed - the one
/// session constructor for both `EquivSession::with_shared_counts` and
/// every recycle, so recycling never rescans the arenas.
fn fresh_session(counts1: &Arc<Vec<u32>>, counts2: &Arc<Vec<u32>>) -> Session {
    let mut session = Session::new();
    session.provide_ref_counts(Side::Reference, Arc::clone(counts1));
    session.provide_ref_counts(Side::Optimized, Arc::clone(counts2));
    session
}

/// One-shot equivalence check (a fresh session per call; prefer
/// `EquivSession` in loops).
pub fn check_equivalent(
    arena1: &ExprArena,
    e1: ExprId,
    arena2: &ExprArena,
    e2: ExprId,
) -> EquivResult<bool> {
    EquivSession::new(arena1, arena2).check(e1, e2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_equivalence() {
        // (a + b) == (b + a)
        let mut arena = ExprArena::new();
        let a = arena.param_symbol("a");
        let b = arena.param_symbol("b");
        let e1 = arena.add(a, b);
        let e2 = arena.add(b, a);
        assert!(check_equivalent(&arena, e1, &arena, e2).unwrap());
    }

    #[test]
    fn test_distributivity() {
        // a * (b + c) == a*b + a*c
        let mut arena = ExprArena::new();
        let a = arena.param_symbol("a");
        let b = arena.param_symbol("b");
        let c = arena.param_symbol("c");
        let bc = arena.add(b, c);
        let e1 = arena.mul(a, bc);
        let ab = arena.mul(a, b);
        let ac = arena.mul(a, c);
        let e2 = arena.add(ab, ac);
        assert!(check_equivalent(&arena, e1, &arena, e2).unwrap());
    }

    #[test]
    fn test_not_equivalent() {
        let mut arena = ExprArena::new();
        let a = arena.param_symbol("a");
        let b = arena.param_symbol("b");
        assert!(!check_equivalent(&arena, a, &arena, b).unwrap());
    }

    #[test]
    fn test_reduction_pattern() {
        let mut arena = ExprArena::new();
        let i0 = arena.param_symbol("input_0");
        let i1 = arena.param_symbol("input_1");
        let i2 = arena.param_symbol("input_2");
        let i3 = arena.param_symbol("input_3");

        let t1 = arena.add(i3, i2);
        let t2 = arena.add(i1, i0);
        let e1 = arena.add(t1, t2);

        let t3 = arena.add(i3, i1);
        let t4 = arena.add(i2, i0);
        let e2 = arena.add(t3, t4);
        assert!(check_equivalent(&arena, e1, &arena, e2).unwrap());
    }

    #[test]
    fn test_exp_identity() {
        // exp(a) * exp(b) == exp(a + b)
        let mut arena = ExprArena::new();
        let a = arena.param_symbol("a");
        let b = arena.param_symbol("b");
        let ea = arena.exp(a);
        let eb = arena.exp(b);
        let e1 = arena.mul(ea, eb);
        let ab = arena.add(a, b);
        let e2 = arena.exp(ab);
        assert!(check_equivalent(&arena, e1, &arena, e2).unwrap());
    }

    #[test]
    fn test_fma_expansion() {
        // fma(a, b, c) == a*b + c
        let mut arena = ExprArena::new();
        let a = arena.param_symbol("a");
        let b = arena.param_symbol("b");
        let c = arena.param_symbol("c");
        let e1 = arena.fma(a, b, c);
        let ab = arena.mul(a, b);
        let e2 = arena.add(ab, c);
        assert!(check_equivalent(&arena, e1, &arena, e2).unwrap());
    }

    #[test]
    fn test_softmax_normalization_equivalence() {
        // exp(a)/(exp(a)+exp(b)) == exp(a-M)/(exp(a-M)+exp(b-M)), M = max(a,b)
        let mut arena = ExprArena::new();
        let a = arena.param_symbol("a");
        let b = arena.param_symbol("b");
        let m = arena.max(a, b);

        let ea = arena.exp(a);
        let eb = arena.exp(b);
        let d1 = arena.add(ea, eb);
        let e1 = arena.div(ea, d1);

        let am = arena.sub(a, m);
        let bm = arena.sub(b, m);
        let eam = arena.exp(am);
        let ebm = arena.exp(bm);
        let d2 = arena.add(eam, ebm);
        let e2 = arena.div(eam, d2);

        assert!(check_equivalent(&arena, e1, &arena, e2).unwrap());
    }

    #[test]
    fn test_session_reuse_across_elements() {
        // The same session checks several related identities.
        let mut arena = ExprArena::new();
        let sid = arena.intern_string("a");
        let tid = arena.intern_string("b");
        let elements: Vec<(ExprId, ExprId)> = (0..4)
            .map(|i| {
                let a = arena.input_element(sid, i);
                let b = arena.input_element(tid, i);
                (arena.add(a, b), arena.add(b, a))
            })
            .collect();
        let mut session = EquivSession::new(&arena, &arena);
        for (ab, ba) in elements {
            assert!(session.check(ab, ba).unwrap());
        }
    }

    #[test]
    fn test_undefined_error() {
        let mut arena = ExprArena::new();
        let a = arena.param_symbol("a");
        let u = arena.undefined();
        assert!(check_equivalent(&arena, a, &arena, u).is_err());
    }

    /// `with_shared_counts` (caller-precomputed parent counts, as the
    /// parallel driver builds per-worker sessions) decides exactly like
    /// the count-computing constructor, including across recycles
    /// (`recycle_terms = 1` forces a recycle before nearly every check,
    /// exercising the counts-preserving `fresh_session` path).
    #[test]
    fn shared_counts_sessions_match_default_sessions() {
        let mut arena = ExprArena::new();
        let sid = arena.intern_string("a");
        let tid = arena.intern_string("b");
        let cases: Vec<(ExprId, ExprId, bool)> = (0..4)
            .map(|i| {
                let a = arena.input_element(sid, i);
                let b = arena.input_element(tid, i);
                let ab = arena.add(a, b);
                let ba = arena.add(b, a);
                let bb = arena.add(b, b);
                if i % 2 == 0 {
                    (ab, ba, true)
                } else {
                    (ab, bb, false)
                }
            })
            .collect();

        let counts = Arc::new(parent_counts(&arena));
        let mut shared = EquivSession::with_shared_counts(
            &arena,
            &arena,
            1,
            Arc::clone(&counts),
            Arc::clone(&counts),
        );
        let mut default = EquivSession::with_recycle_terms(&arena, &arena, 1);
        for &(a, b, want) in &cases {
            assert_eq!(shared.check(a, b).unwrap(), want);
            assert_eq!(default.check(a, b).unwrap(), want);
        }
    }

    /// Regression: a named symbol whose string spells a machine symbol's
    /// rendered name (`"s{N}"`) used to intern to the same canon variable
    /// as machine `Symbol(N)` - proving two independent values
    /// "equivalent". The typed `SymbolRef` namespaces keep them apart.
    #[test]
    fn named_symbol_does_not_alias_machine_symbol() {
        use crate::symbolic::ExprNode;

        let mut arena = ExprArena::new();
        let machine = arena.symbol();
        let ExprNode::Symbol(sym) = *arena.node(machine) else {
            panic!("arena.symbol() must produce ExprNode::Symbol");
        };
        let named = arena.param_symbol(sym.to_string());
        assert!(
            !check_equivalent(&arena, machine, &arena, named).unwrap(),
            "machine Symbol({}) and NamedSymbol(\"{}\") are independent values",
            sym.0,
            sym
        );
        // And a named symbol still correlates with itself across arenas.
        let mut other = ExprArena::new();
        let named_other = other.param_symbol(sym.to_string());
        assert!(check_equivalent(&arena, named, &other, named_other).unwrap());
    }
}
