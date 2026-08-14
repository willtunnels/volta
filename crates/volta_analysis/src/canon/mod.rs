//! The decision procedure: canonicalization of evaluator expressions into
//! interned `Σ c·monomial·e^{poly}` rational forms, and equality over them
//! (see `docs/canonicalizer-plan.md`).
//!
//! A `Session` owns the intern tables and memo for one equivalence-checking
//! run (typically one benchmark: both kernels, all output elements), so
//! shared structure is canonicalized once and equal canonical objects have
//! equal ids everywhere.

mod arena;
mod canonicalize;
mod coeff;
mod ops;

use std::collections::HashMap;
use std::sync::Arc;

use crate::symbolic::{ExprArena, ExprId};

pub use arena::{CanonArena, Rat};
pub use canonicalize::parent_counts;
pub use coeff::{Coeff, CoeffError};

use arena::TermId;

/// Which kernel of a VC pair an expression comes from (memo key component;
/// the two kernels have independent `ExprArena`s with overlapping ids).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Reference,
    Optimized,
}

/// Errors from canonicalization or equality checking.
#[derive(Debug, Clone, PartialEq)]
pub enum CanonError {
    /// Coefficient arithmetic failed (overflow / non-finite constant).
    Coeff(CoeffError),
    /// A factor power exceeded u32.
    PowerOverflow,
    /// Division by a (symbolically) zero denominator.
    DivisionByZero,
    /// An `Undefined`/`Discarded` value reached the decision procedure.
    Undefined,
    /// The session's term-operation budget was exhausted.
    Budget { limit: u64 },
}

impl From<CoeffError> for CanonError {
    fn from(e: CoeffError) -> Self {
        Self::Coeff(e)
    }
}

impl std::fmt::Display for CanonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Coeff(e) => write!(f, "{}", e),
            Self::PowerOverflow => write!(f, "monomial power overflowed"),
            Self::DivisionByZero => write!(f, "division by a symbolically zero value"),
            Self::Undefined => write!(f, "cannot check equivalence of undefined values"),
            Self::Budget { limit } => {
                write!(f, "VC too large: exceeded {} term operations", limit)
            }
        }
    }
}

impl std::error::Error for CanonError {}

/// Default term-operation budget: enough for the paper's largest VCs
/// (FlashAttention canonicalization is ~10^8 term ops) with headroom, small
/// enough to fail within tens of seconds instead of hanging.
pub const DEFAULT_BUDGET: u64 = 4_000_000_000;

/// One canonicalization session: intern tables + memos + budget.
///
/// Memory discipline: only *shared* results are interned - values of
/// multi-parent expression nodes (per the ref-count prepass), exp/atom
/// arguments, and the roots handed to `canonicalize`. Everything along
/// single-use chains (fma accumulators, running rescales) stays a transient
/// owned vector freed by scope, so a K-step accumulator costs O(K) live
/// memory instead of retaining O(K^2) interned prefixes.
pub struct Session {
    pub(crate) arena: CanonArena,
    pub(crate) expr_memo: HashMap<(Side, ExprId), Rat>,
    pub(crate) term_mul_memo: HashMap<(TermId, TermId), TermId>,
    /// Per side: how many parents each `ExprId` has (over the whole
    /// arena). `Arc` so callers that build many sessions over the same
    /// arenas (recycling, parallel workers) can share one computation
    /// via [`Session::provide_ref_counts`]; computed lazily otherwise.
    pub(crate) ref_counts: HashMap<Side, Arc<Vec<u32>>>,
    pub(crate) ops: u64,
    pub(crate) budget: u64,
}

impl Session {
    pub fn new() -> Self {
        Self::with_budget(DEFAULT_BUDGET)
    }

    pub fn with_budget(budget: u64) -> Self {
        Self {
            arena: CanonArena::new(),
            expr_memo: HashMap::new(),
            term_mul_memo: HashMap::new(),
            ref_counts: HashMap::new(),
            ops: 0,
            budget,
        }
    }

    /// Term operations performed so far (diagnostics).
    pub fn ops_used(&self) -> u64 {
        self.ops
    }

    /// Install precomputed parent counts for one side's arena - the
    /// result of [`parent_counts`] over that exact arena. Counts affect
    /// only which intermediate results are interned (memory/time), never
    /// verdicts, but mismatched counts forfeit the sharing discipline -
    /// so the length is checked against the arena at first use
    /// (`ensure_ref_counts` recomputes if the counts are too short).
    pub fn provide_ref_counts(&mut self, side: Side, counts: Arc<Vec<u32>>) {
        self.ref_counts.insert(side, counts);
    }

    /// Number of interned terms - the dominant memory consumer. Callers
    /// checking many VC elements should recycle the session when this grows
    /// large (intermediate chain terms are interned per element and only
    /// partially shared across elements).
    pub fn interned_terms(&self) -> usize {
        self.arena.terms.len()
    }

    /// Canonicalize two expressions (possibly from different arenas) and
    /// decide their equality over the reals.
    pub fn check_equivalent(
        &mut self,
        arena_a: &ExprArena,
        a: ExprId,
        arena_b: &ExprArena,
        b: ExprId,
    ) -> Result<bool, CanonError> {
        let ra = self.canonicalize(Side::Reference, arena_a, a)?;
        let rb = self.canonicalize(Side::Optimized, arena_b, b)?;
        self.equivalent(ra, rb)
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::ExprArena;

    fn check(arena: &ExprArena, a: ExprId, b: ExprId) -> bool {
        Session::new().check_equivalent(arena, a, arena, b).unwrap()
    }

    #[test]
    fn test_commutativity_and_association() {
        let mut ar = ExprArena::new();
        let (a, b, c) = (
            ar.param_symbol("a"),
            ar.param_symbol("b"),
            ar.param_symbol("c"),
        );
        let ab = ar.add(a, b);
        let e1 = ar.add(ab, c);
        let bc = ar.add(b, c);
        let e2 = ar.add(c, ab);
        let e3 = ar.add(a, bc);
        assert!(check(&ar, e1, e2));
        assert!(check(&ar, e1, e3));
    }

    #[test]
    fn test_distribution() {
        let mut ar = ExprArena::new();
        let (a, b, c) = (
            ar.param_symbol("a"),
            ar.param_symbol("b"),
            ar.param_symbol("c"),
        );
        let bc = ar.add(b, c);
        let e1 = ar.mul(a, bc);
        let ab = ar.mul(a, b);
        let ac = ar.mul(a, c);
        let e2 = ar.add(ab, ac);
        assert!(check(&ar, e1, e2));
    }

    #[test]
    fn test_cancellation() {
        // a + b - a == b  (E + (-E) = 0)
        let mut ar = ExprArena::new();
        let (a, b) = (ar.param_symbol("a"), ar.param_symbol("b"));
        let ab = ar.add(a, b);
        let e1 = ar.sub(ab, a);
        assert!(check(&ar, e1, b));
    }

    #[test]
    fn test_not_equivalent() {
        let mut ar = ExprArena::new();
        let (a, b) = (ar.param_symbol("a"), ar.param_symbol("b"));
        assert!(!check(&ar, a, b));
        let two_a = ar.add(a, a);
        assert!(!check(&ar, two_a, a));
    }

    #[test]
    fn test_exp_fusion() {
        // e^a * e^b == e^{a+b}; and e^a * e^{-a} == 1
        let mut ar = ExprArena::new();
        let (a, b) = (ar.param_symbol("a"), ar.param_symbol("b"));
        let ea = ar.exp(a);
        let eb = ar.exp(b);
        let e1 = ar.mul(ea, eb);
        let apb = ar.add(a, b);
        let e2 = ar.exp(apb);
        assert!(check(&ar, e1, e2));

        let na = ar.neg(a);
        let ena = ar.exp(na);
        let prod = ar.mul(ea, ena);
        let one = ar.float_from_f64(1.0).unwrap();
        assert!(check(&ar, prod, one));
    }

    #[test]
    fn test_exact_float_coefficients() {
        // 0.125 * (a + a) == 0.25 * a, exactly.
        let mut ar = ExprArena::new();
        let a = ar.param_symbol("a");
        let aa = ar.add(a, a);
        let eighth = ar.float_from_f64(0.125).unwrap();
        let e1 = ar.mul(eighth, aa);
        let quarter = ar.float_from_f64(0.25).unwrap();
        let e2 = ar.mul(quarter, a);
        assert!(check(&ar, e1, e2));
    }

    #[test]
    fn test_max_rules() {
        // max(max(a,b),c) == max(a,max(b,c)); max(x,x) == x
        let mut ar = ExprArena::new();
        let (a, b, c) = (
            ar.param_symbol("a"),
            ar.param_symbol("b"),
            ar.param_symbol("c"),
        );
        let mab = ar.max(a, b);
        let e1 = ar.max(mab, c);
        let mbc = ar.max(b, c);
        let e2 = ar.max(a, mbc);
        assert!(check(&ar, e1, e2));

        let mxx = ar.max(a, a);
        assert!(check(&ar, mxx, a));
    }

    #[test]
    fn test_fraction_normalization() {
        // a / e^m == a * e^{-m} (monomial denominators fold away)
        let mut ar = ExprArena::new();
        let (a, m) = (ar.param_symbol("a"), ar.param_symbol("m"));
        let em = ar.exp(m);
        let e1 = ar.div(a, em);
        let nm = ar.neg(m);
        let enm = ar.exp(nm);
        let e2 = ar.mul(a, enm);
        assert!(check(&ar, e1, e2));
    }

    #[test]
    fn test_softmax_normalization() {
        // exp(a)/(exp(a)+exp(b)) == exp(a-M)/(exp(a-M)+exp(b-M)), M=max(a,b)
        // The monomial-quotient path: denominators differ by e^{-M}.
        let mut ar = ExprArena::new();
        let (a, b) = (ar.param_symbol("a"), ar.param_symbol("b"));
        let m = ar.max(a, b);

        let ea = ar.exp(a);
        let eb = ar.exp(b);
        let d1 = ar.add(ea, eb);
        let e1 = ar.div(ea, d1);

        let am = ar.sub(a, m);
        let bm = ar.sub(b, m);
        let eam = ar.exp(am);
        let ebm = ar.exp(bm);
        let d2 = ar.add(eam, ebm);
        let e2 = ar.div(eam, d2);

        assert!(check(&ar, e1, e2));

        // And the negative case: a different numerator is caught.
        let e3 = ar.div(ebm, d2);
        assert!(!check(&ar, e1, e3));
    }

    #[test]
    fn test_online_softmax_paper_example() {
        // The paper's Section 2 example at N = 4: naive softmax equals the
        // online (FlashAttention) formulation with running max and rescaled
        // denominator.
        let mut ar = ExprArena::new();
        let xs: Vec<ExprId> = (0..4)
            .map(|i| ar.param_symbol(format!("x[{}]", i)))
            .collect();

        // Naive: y_i = e^{x_i} / Σ e^{x_j}
        let exps: Vec<ExprId> = xs.iter().map(|&x| ar.exp(x)).collect();
        let mut d_naive = exps[0];
        for &e in &exps[1..] {
            d_naive = ar.add(d_naive, e);
        }

        // Online: running max m_j, d = d*e^{m_prev - m} + e^{x_j - m}
        let neg_inf = ar.float_from_f64(f64::NEG_INFINITY).unwrap();
        let zero = ar.float_from_f64(0.0).unwrap();
        let (mut m_prev, mut d_online) = (neg_inf, zero);
        for &x in &xs {
            let m = ar.max(m_prev, x);
            let dm = ar.sub(m_prev, m);
            let scale = ar.exp(dm);
            let xm = ar.sub(x, m);
            let e = ar.exp(xm);
            let scaled = ar.mul(d_online, scale);
            d_online = ar.add(scaled, e);
            m_prev = m;
        }

        // y_0 both ways.
        let y_naive = ar.div(exps[0], d_naive);
        let x0m = ar.sub(xs[0], m_prev);
        let e0m = ar.exp(x0m);
        let y_online = ar.div(e0m, d_online);

        assert!(check(&ar, y_naive, y_online));

        // Perturbed online result must be rejected.
        let x1m = ar.sub(xs[1], m_prev);
        let e1m = ar.exp(x1m);
        let y_wrong = ar.div(e1m, d_online);
        assert!(!check(&ar, y_naive, y_wrong));
    }

    #[test]
    fn test_budget_error() {
        let mut ar = ExprArena::new();
        let (a, b, c, d) = (
            ar.param_symbol("a"),
            ar.param_symbol("b"),
            ar.param_symbol("c"),
            ar.param_symbol("d"),
        );
        let ab = ar.add(a, b);
        let cd = ar.add(c, d);
        let p = ar.mul(ab, cd);
        let mut session = Session::with_budget(2);
        assert!(matches!(
            session.check_equivalent(&ar, p, &ar, p),
            Err(CanonError::Budget { .. })
        ));
    }

    #[test]
    fn test_undefined_error() {
        let mut ar = ExprArena::new();
        let a = ar.param_symbol("a");
        let u = ar.undefined();
        assert!(matches!(
            Session::new().check_equivalent(&ar, a, &ar, u),
            Err(CanonError::Undefined)
        ));
    }
}

#[cfg(test)]
mod chain_perf_tests {
    use super::*;
    use crate::symbolic::ExprArena;

    #[test]
    fn test_chain_ops_linear() {
        // A K-step fma accumulator chain must canonicalize in O(K) term ops.
        let mut ar = ExprArena::new();
        let k = 1000u32;
        let mut acc = ar.float_from_f64(0.0).unwrap();
        for i in 0..k {
            let a = ar.param_symbol(format!("A[{}]", i));
            let b = ar.param_symbol(format!("B[{}]", i));
            acc = ar.fma(a, b, acc);
        }
        let mut session = Session::new();
        let _ = session.canonicalize(Side::Reference, &ar, acc).unwrap();
        let ops = session.ops_used();
        assert!(
            ops < 20_000,
            "chain of {} steps took {} term ops (expected O(K))",
            k,
            ops
        );
    }
}
