# Mechanism-Aware Multi-Agent Search & Recommendation System

> **Course**: Multi-Agent Systems / Game Theoretic Design and Analysis of Agentic AI
> **Theme**: Theme 4 — Game Theoretic Design and Analysis of Agentic AI

A research-engineering project extending the **slate auction framework** from
*Mechanism Design for Large Language Models* (Duetting et al., WWW 2024 Best Paper)
into the **item-level search & recommendation** domain, where multiple scoring agents
with conflicting objectives collaborate via incentive-compatible auction aggregation.

---

## Problem Motivation

Traditional search and recommendation systems combine multiple ranking signals
(relevance, personalization, diversity, safety) through manually tuned weights.
This has three critical weaknesses:

1. **Opacity**: No principled explanation of why item A is ranked above item B
2. **No incentive structure**: Agents representing different objectives have no
   mechanism to express preference strength or resolve conflicts
3. **Manipulation vulnerability**: A dominant objective can override others silently

This project addresses all three by treating the ranking problem as a
**multi-agent slate auction** where each agent submits a score distribution over
candidates and a scalar bid, and a mechanism-aware aggregation rule produces the
final ranked slate with a full per-position influence audit trail.

---

## Core Idea: From Token Auction to Slate Auction

| Original (Duetting et al., 2024) | This Project |
|----------------------------------|--------------|
| Unit of decision: **next token** | Unit of decision: **ranked item / slate position** |
| Agents: LLM text generators | Agents: Relevance, Personalization, Diversity, Safety |
| Distribution: token probability | Distribution: item score distribution over candidate pool |
| Bid: scalar weight per token | Bid: scalar influence weight per slate position |
| Output: token sequence | Output: ranked slate (Top-K items) |
| Payment: per-token critical bid | Payment: per-position second-price influence cost |

The key theoretical bridge: **monotone aggregation + second-price payments**
hold when the decision unit changes from tokens to items, provided agents'
preferences satisfy the robust preference condition (Duetting et al., Thm. 1).

---

## Bid Mechanism and Incentive Design

> **Q (instructor feedback)**: *It is not clear how the bids submitted by agents
> appear in their value function. What prevents them from making potentially
> infinite bids? I think you perhaps have some budget constraint or a model where
> higher bids decrease value.*

### Value Function

Each agent `i` has an objective-specific **value function** over the aggregated
output distribution `q`:

```
V_i(q) = Σ_x  q(x) · p_i(x)      (inner product of q with agent i's score vector)
```

- Relevance agent:       `V_rel(q)  = E_{x~q}[BM25(x, query)]`
- Personalization agent: `V_per(q)  = E_{x~q}[affinity(x, user_history)]`
- Diversity agent:       `V_div(q)  = E_{x~q}[diversity_score(x, slate)]`
- Safety agent:          `V_saf(q)  = E_{x~q}[safety_score(x)]`

### Bid Enters Utility via the Aggregation Rule

Under **linear aggregation**, the influence weight of agent `i` is:

```
λ_i = b_i / Σ_j b_j
```

so the final distribution is:

```
q(x) = Σ_i λ_i · p_i(x)
```

Increasing `b_i` increases `λ_i`, which pulls `q` closer to `p_i`, which
increases `V_i(q)`. The bid thus enters the utility function *indirectly*
through `λ_i`.

### What Prevents Infinite Bidding: Second-Price Payment

Each agent pays the **critical bid** `π_i` — the minimum bid needed to achieve
its current influence share (computed by binary search, `src/mechanism/payment.py`):

```
π_i = inf { b : λ_i(b, b_{-i}) ≥ λ_i(b_i, b_{-i}) }
```

The full utility is:

```
U_i(b_i) = V_i(q(b_i, b_{-i}))  −  π_i(b_i, b_{-i})
```

**Why infinite bidding is suboptimal**:

- As `b_i → ∞`,  `λ_i → 1` (bounded by 1), so **marginal value gain → 0**
- Meanwhile `π_i` grows proportionally with `b_i`
- Therefore `U_i(b_i) → −∞` as `b_i → ∞`

This is the direct analogue of the Generalized Second-Price (GSP) auction
(Edelman et al., 2007): overbidding beyond your true valuation strictly
decreases net utility. Under this mechanism, truthful bidding `b_i = b_i*`
(where `b_i*` reflects true valuation weight) is a dominant strategy
(Duetting et al., Thm. 1; Vickrey, 1961).

### Budget Constraint (Practical Enforcement)

In our implementation, bids are further constrained by **normalization**:
`Σ b_i = 1`. Each agent is allocated an *influence budget* proportional to
its designated role weight (e.g., relevance=0.4, diversity=0.2). The dynamic
bid variant (Exp 1) adapts bids within a bounded range `[0.24, 0.56]` based
on query-specific confidence, ensuring the budget constraint is never violated.

---

## System Architecture

```
  User Query + Context
         │
         ▼
  Candidate Pool (BM25 recall, top-100 items)
         │
         ▼
  ┌──────────────────────────────────────────┐
  │           Scoring Agent Layer            │
  │                                          │
  │  Relevance Agent       bid b₁ = 0.40    │
  │  Personalization Agent bid b₂ = 0.30    │
  │  Diversity Agent       bid b₃ = 0.20    │
  │  Safety Agent          bid b₄ = 0.10    │
  │                                          │
  │  Each outputs: p_i ∈ Δ(Pool), b_i ∈ ℝ₊ │
  └──────────────┬───────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────┐
  │         Mechanism / Aggregation          │
  │                                          │
  │  Linear:     q = Σ λᵢpᵢ    [IC ✓]      │
  │  Log-linear: q ∝ Π pᵢ^λᵢ  [IC ✗]      │
  │                                          │
  │  λᵢ = bᵢ / Σbⱼ                         │
  │  πᵢ = critical bid (second-price)        │
  └──────────────┬───────────────────────────┘
                 │
                 ▼
  Ranked Slate (Top-10) + Influence Audit Log
```

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── run_pipeline.py                      # End-to-end demo (KuaiSAR, single query)
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/KuaiSAR_final/               # Raw CSVs (not committed)
│   └── processed/                       # Cached loader output (JSON)
├── results/
│   └── kuaisar/<timestamp>/
│       ├── run_meta.json
│       ├── exp1_ranking_quality.csv
│       ├── exp2_diversity_pareto.csv
│       ├── exp3_monotonicity.json
│       ├── exp4_manipulation.csv
│       ├── exp5_utility.csv
│       └── summary.json
├── src/
│   ├── agents/
│   │   ├── base_agent.py                # Abstract ScoringAgent interface
│   │   ├── relevance_agent.py           # BM25 scorer
│   │   ├── personalization_agent.py     # User history scorer
│   │   ├── diversity_agent.py           # MMR / category diversity scorer
│   │   └── safety_agent.py             # Rule-based safety filter
│   ├── mechanism/
│   │   ├── aggregation.py              # Linear & log-linear aggregation
│   │   ├── payment.py                  # Critical-bid (second-price) payment
│   │   ├── auction.py                  # Auction orchestrator → AuctionResult
│   │   └── utility.py                  # Agent value functions U_i = V_i − π_i
│   ├── pipeline/
│   │   └── candidate_pool.py           # BM25 candidate retrieval
│   ├── data/
│   │   └── kuaisar_loader.py           # KuaiSAR CSV loader + JSON cache
│   └── evaluation/
│       ├── metrics.py                  # NDCG@K, Recall@K, ILD, MRR, Coverage
│       └── manipulation_test.py        # Bid inflation stress test
└── experiments/
    └── run_kuaisar_experiments.py      # Full 5-experiment suite (saves to results/)
```

---

## Experimental Results (KuaiSAR Dataset)

> Dataset: 3,000 items · 50 queries · 36 categories · pool\_size=100

### Experiment 1 — Ranking Quality

| Method | NDCG@5 | NDCG@10 | Recall@10 | ILD@10 | MRR | **F-score** |
|--------|-------:|--------:|----------:|-------:|----:|------------:|
| single\_agent | 0.069 | 0.095 | 0.125 | 0.060 | 0.073 | 0.074 |
| fixed\_weight | 0.251 | 0.317 | 0.466 | 0.162 | 0.238 | 0.214 |
| **linear\_dynamic** | 0.228 | 0.299 | 0.456 | **0.262** | 0.219 | **0.279** |
| loglinear | **0.264** | **0.329** | **0.476** | 0.069 | **0.249** | 0.115 |

> F-score = 2·NDCG·ILD / (NDCG + ILD) — joint measure of relevance and diversity.

Key findings:
- Multi-agent (all variants) outperforms single-agent by **3×** on NDCG and **4×** on ILD
- `linear_dynamic` achieves the best F-score (+30% vs fixed\_weight, +143% vs loglinear)
- `loglinear`'s higher NDCG comes at a cost: ILD collapses to 0.069 ≈ single-agent (0.060), meaning diversity agent is effectively suppressed by the geometric mean
- `linear_dynamic` is 5.6% lower on NDCG but 62% higher on ILD — a favorable tradeoff

### Experiment 2 — Diversity–Relevance Pareto Frontier

| Diversity Bid | NDCG@10 | ILD@10 | F-score |
|--------------:|--------:|-------:|--------:|
| 0.00 | 0.326 | 0.092 | 0.144 |
| 0.20 ← default | 0.317 | 0.162 | 0.214 |
| **0.40 ← optimal** | **0.251** | **0.554** | **0.346** |
| 0.50 | 0.182 | 0.754 | 0.293 |
| 1.00 | 0.125 | 0.832 | 0.217 |

Changing `diversity_bid` from 0.20 → 0.40 improves F-score by **+61%** with no
model retraining — illustrating the auction's role as a principled control knob.

### Experiment 3 — Monotonicity Verification (300 trials)

| Mechanism | Violation Rate | Expected |
|-----------|---------------:|---------|
| Linear | **0.0%** | ~0% (Duetting et al. Thm. 1) |
| Log-linear | **100.0%** | >0% (adversarial borrowed-strength failure) |

The adversarial case: agent `i` has weak preference for item `x*` (score ≈ 0.4)
while all other agents have strong preference (score ≈ 0.9). Increasing `b_i`
reduces `λ_j` for `j ≠ i`, destroying the "borrowed strength" → `q(x*)` decreases
despite the higher bid. Linear aggregation is immune by algebra.

### Experiment 4 — Manipulation Stress Test

| k | Linear Gain | Linear Rank↑ | Loglinear Gain | Loglinear Rank↑ |
|--:|------------:|-------------:|---------------:|----------------:|
| 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5 | 0.200 | 0.016 | 0.000 | 0.094 |
| 10 | 0.200 | 0.016 | 0.000 | **0.592** |
| 20 | **0.200** | **0.016** | **0.800** | **0.652** |

> k = bid inflation multiplier. Rank↑ = normalized rank improvement of manipulator's items.

- **Linear**: gain saturates at 0.20 for k ≥ 5 — second-price property bounds the gain
- **Log-linear**: appears safe for k ≤ 9, then collapses at k = 10 (rank↑ = 59%) — threshold manipulation failure, more dangerous in practice

---

## References

1. Duetting P, Mirrokni V, Paes Leme R, Xu H, Zuo S. Mechanism design for large language models. *ACM Web Conference (WWW)* 2024. arXiv:2310.10826.
2. Vickrey W. Counterspeculation, auctions, and competitive sealed tenders. *Journal of Finance* 16(1), 1961.
3. Myerson R. Optimal auction design. *Mathematics of Operations Research* 6(1), 1981.
4. Edelman B, Ostrovsky M, Schwarz M. Internet advertising and the generalized second-price auction. *American Economic Review* 97(1), 2007.
5. Yang Y, Chai H, Shao S, et al. AgentNet: Decentralized evolutionary coordination for LLM-based multi-agent systems. arXiv:2504.00587. 2025.
6. Park C, Han S, Guo X, Ozdaglar AE, Zhang K, Kim JK. MAPoRL: Multi-agent post-co-training for collaborative LLMs with reinforcement learning. *ACL* 2025.
7. Park C, Chen Z, Ozdaglar A, Zhang K. Post-training LLMs as better decision-making agents: A regret-minimization approach. arXiv:2511.04393. 2025.
8. Gao Y, et al. KuaiSAR: A unified search and recommendation dataset. *CIKM* 2023.

---

## Installation & Reproduction

```bash
git clone https://github.com/SouthShore01/Mechanism-Aware-Multi-Agent-S-R-System.git
cd Mechanism-Aware-Multi-Agent-S-R-System
pip install -r requirements.txt
```

Download KuaiSAR data (Zenodo record 8181109):
```bash
wget https://zenodo.org/records/8181109/files/KuaiSAR.zip
unzip KuaiSAR.zip -d data/raw/
```

Run all experiments (results saved automatically to `results/kuaisar/<timestamp>/`):
```bash
python experiments/run_kuaisar_experiments.py --n_queries 50 --max_items 3000
```

Run individual experiments:
```bash
python experiments/run_kuaisar_experiments.py --exp 1        # ranking quality only
python experiments/run_kuaisar_experiments.py --exp 1,3,4    # subset
python experiments/run_kuaisar_experiments.py --exp 2 --div_steps 21  # finer Pareto sweep
```

End-to-end pipeline demo (single query with audit trace):
```bash
python run_pipeline.py --rule linear --n_queries 5
python run_pipeline.py --rule loglinear --n_queries 5
```
