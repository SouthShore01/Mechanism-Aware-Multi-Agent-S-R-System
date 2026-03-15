# Mechanism-Aware Multi-Agent Search & Recommendation System

> **Course**: Multi-Agent Systems / Game Theoretic Design and Analysis of Agentic AI
> **Theme**: Theme 4 — Game Theoretic Design and Analysis of Agentic AI
> **Mid-term Report Deadline**: March 24, 2026

A research-engineering project extending the **token auction framework** from
*Mechanism Design for Large Language Models* (WWW 2024 Best Paper) into the
**slate-level search & recommendation** domain, where multiple LLM-based agents
with conflicting objectives collaborate via auction-style aggregation.

---

## Problem Motivation

Traditional search and recommendation systems combine multiple ranking signals
(relevance, personalization, diversity, safety, business value) through manually
tuned weights. This approach has three critical weaknesses:

1. **Opacity**: No principled explanation of why item A is ranked above item B
2. **No incentive structure**: Agents representing different objectives have no
   mechanism to express preference strength or resolve conflicts
3. **Manipulation vulnerability**: A single dominant objective can override others
   silently without an audit trail

This project addresses these weaknesses by treating the ranking problem as a
**multi-agent auction** where each agent submits a distribution over candidates
and a scalar bid, and a mechanism-aware aggregation rule produces the final slate.

---

## Core Idea: From Token Auction to Slate Auction

| Original (Duetting et al., 2024) | This Project |
|----------------------------------|--------------|
| Unit of decision: **next token** | Unit of decision: **next item / position** |
| Agents: LLM advertisers | Agents: Relevance, Personalization, Diversity, Safety, Business |
| Distribution: token probability | Distribution: item score distribution over candidate pool |
| Bid: scalar weight per token | Bid: scalar influence weight per slate position |
| Output: token sequence | Output: ranked slate (search results / recommendation list) |
| Payment: per-token influence cost | Payment: per-position influence audit log |

The key theoretical bridge: **monotone aggregation + second-price-style payments**
still hold when the decision unit changes from tokens to items, as long as agents'
preferences satisfy the robust preference condition from the original paper.

---

## Paper Survey (4 Core Papers)

### Paper 1 — Theoretical Backbone
**Duetting P, Mirrokni V, Paes Leme R, Xu H, Zuo S.**
*Mechanism Design for Large Language Models.*
ACM Web Conference 2024 (Best Paper Award). arXiv:2310.10826.

- Proposes token-by-token auction for multi-LLM output aggregation
- Key results: monotonicity ↔ incentive compatibility; second-price rule under
  robust preferences; linear aggregation is monotone, log-linear is not
- **Role in this project**: provides the aggregation functions, payment design, and
  the theoretical condition (monotonicity) we test for in the S&R setting

### Paper 2 — Multi-Agent Architecture
**Yang Y, Chai H, Shao S et al.**
*AgentNet: Decentralized Evolutionary Coordination for LLM-Based Multi-Agent Systems.*
arXiv:2504.00587. 2025.

- Proposes decentralized, graph-structured agent coordination with dynamic routing
  and RAG-based memory sharing across agents
- **Role in this project**: informs the system architecture — each agent is a node
  in the coordination graph; contrast with our approach: AgentNet coordinates via
  communication, we coordinate via **incentive-compatible bidding**

### Paper 3 — Learning to Collaborate
**Park C, Han S, Guo X, Ozdaglar AE, Zhang K, Kim JK.**
*MAPoRL: Multi-Agent Post-Co-Training for Collaborative LLMs with Reinforcement Learning.*
ACL 2025 (pp. 30215–30248).

- Multi-agent RL post-training to make LLMs learn collaborative behavior
  rather than relying on prompt engineering alone
- **Role in this project**: establishes that coordination can be *learned*, not
  just rule-based; opens the future direction of learning agent bid strategies
  via RL; provides the contrast: mechanism design (rule-based) vs RL (learned)

### Paper 4 — Decision-Making & Regret
**Park C, Chen Z, Ozdaglar A, Zhang K.**
*Post-Training LLMs as Better Decision-Making Agents: A Regret-Minimization Approach.*
arXiv:2511.04393. 2025.

- Frames LLM post-training as regret minimization in online learning / game theory
- **Role in this project**: elevates the framing from "ranking quality" to
  "sequential decision quality under strategic pressure"; justifies the
  manipulation stress test experiments; connects to regret analysis of
  auction mechanisms under non-truthful bidding

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        User Query / Context                   │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    Candidate Generation Layer                  │
│  ┌─────────────────┐          ┌────────────────────────────┐  │
│  │  Retrieval Agent │          │  Recommendation Agent      │  │
│  │  (BM25 + Dense)  │          │  (Sequential RecBole)      │  │
│  └────────┬────────┘          └──────────────┬─────────────┘  │
│           └──────────────┬───────────────────┘                │
│                          ▼                                     │
│              Unified Candidate Pool  (top-K items)            │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                   Scoring Agent Layer                          │
│                                                                │
│   Agent 1: Relevance       → score distribution + bid b₁     │
│   Agent 2: Personalization → score distribution + bid b₂     │
│   Agent 3: Diversity       → score distribution + bid b₃     │
│   Agent 4: Safety          → score distribution + bid b₄     │
│  [Agent 5: Business/Sponsor→ score distribution + bid b₅]    │
│                                                                │
│  Each agent outputs: p_i ∈ Δ(CandidatePool), b_i ∈ ℝ₊       │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                  Mechanism / Aggregation Layer                 │
│                                                                │
│  Aggregation Rule (choose one):                               │
│    Linear:     q(x) = Σ λᵢ · pᵢ(x)    [MONOTONE ✓]         │
│    Log-linear: q(x) ∝ Π pᵢ(x)^λᵢ      [NOT monotone ✗]     │
│                                                                │
│  where λᵢ = bᵢ / Σⱼ bⱼ   (normalized bids)                  │
│                                                                │
│  Payment (second-price-style):                                │
│    tᵢ = (critical bid to maintain current influence)          │
│    Logged as: audit_trace[position][agent] = influence_cost   │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│               Final Ranked Slate  (Top-N items)               │
│               + Payment Audit Log per Position                 │
└──────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml                   # Default hyperparameters
│   └── experiment_configs/
│       ├── linear_agg.yaml
│       ├── loglinear_agg.yaml
│       └── fixed_weight.yaml
├── docs/
│   ├── report_outline.md              # 10-page midterm report outline
│   └── experiment_design.md           # Detailed experiment protocol
├── src/
│   ├── agents/
│   │   ├── base_agent.py              # Abstract ScoringAgent interface
│   │   ├── relevance_agent.py         # BM25 / embedding similarity scorer
│   │   ├── personalization_agent.py   # User history-based scorer
│   │   ├── diversity_agent.py         # MMR / coverage-based scorer
│   │   └── safety_agent.py            # Rule-based / classifier-based filter
│   ├── mechanism/
│   │   ├── aggregation.py             # Linear & log-linear aggregation rules
│   │   ├── payment.py                 # Critical-bid / second-price-style payment
│   │   └── auction.py                 # Main auction orchestrator (bid → slate)
│   ├── pipeline/
│   │   ├── retrieval.py               # BM25 + Faiss dense retrieval
│   │   ├── recommendation.py          # RecBole sequential model wrapper
│   │   └── candidate_pool.py          # Merge retrieval + rec candidates
│   ├── data/
│   │   ├── kuaisar_loader.py          # KuaiSAR / KuaiSAR-small data loader
│   │   └── preprocessing.py           # Feature extraction, query parsing
│   └── evaluation/
│       ├── metrics.py                 # NDCG@K, Recall@K, ILD (diversity)
│       ├── manipulation_test.py       # Strategic bid exaggeration stress test
│       └── audit_logger.py            # Per-position influence audit trace
├── experiments/
│   ├── run_baseline.py                # Fixed-weight score fusion baseline
│   ├── run_linear_agg.py              # Linear auction aggregation
│   ├── run_loglinear_agg.py           # Log-linear (monotonicity failure case)
│   └── run_manipulation_stress.py     # Strategic agent manipulation test
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_aggregation_analysis.ipynb
    └── 03_results_visualization.ipynb
```

---

## Framework & Tech Stack

| Component | Framework | Reason |
|-----------|-----------|--------|
| Agent orchestration | **LangGraph** | Stateful, auditable node-based workflow; supports DAG routing and per-node logging — better than AutoGen for controlled pipelines |
| Recommendation baseline | **RecBole** | Production-grade recsys framework with 70+ models; use SASRec or GRU4Rec as sequential baseline |
| Dense retrieval | **Faiss** | Standard ANN library for vector candidate recall; wrap with sentence-transformers |
| Sparse retrieval | **rank_bm25** | Lightweight BM25 for lexical recall |
| LLM scoring | **LangChain + OpenAI/Anthropic API** | LLM-in-the-loop scoring for relevance and safety agents |
| Data | **KuaiSAR-small** (Phase 1) → **KuaiSAR** (Phase 2) | Unified real-world search & recommendation dataset from Kuaishou |
| Experiment tracking | **MLflow** or **Weights & Biases** | Track aggregation variants and ablations |
| Visualization | **Plotly / Streamlit** | Interactive audit trace visualization for demo |

---

## Datasets

### Primary: KuaiSAR
- **Source**: [kuaisar.github.io](https://kuaisar.github.io) / Zenodo record 8181109
- **Scale**: 25,877 users · 6.89M items · 453K queries · 19.6M actions · 19 days
- **Key feature**: Records **user transitions between search and recommendation**,
  making it the only public dataset that supports unified S&R evaluation
- **Why it fits**: Provides both retrieval-style interactions (query → item) and
  recommendation-style interactions (history → item) in a single log, exactly
  matching our unified candidate pool design

### Auxiliary
| Dataset | Role |
|---------|------|
| **KuaiSAR-small** | Fast iteration / unit testing (10-day subset) |
| **KuaiRec** | Near-full observation matrix for offline counterfactual evaluation |
| **KuaiRand** | Random exposure intervention data for causal/A-B approximation |

---

## Experiment Design

### Experiment 1 — Ranking Quality (Main Result)

**Goal**: Compare final slate quality across aggregation methods.

| Method | Description |
|--------|-------------|
| `fixed_weight` | Static manually tuned weights per agent |
| `linear_agg` | Auction-style linear aggregation with normalized bids |
| `loglinear_agg` | Log-linear aggregation (expected to show manipulation failure) |
| `single_agent` | Ablation: only relevance agent, no multi-agent |

**Metrics**: NDCG@5, NDCG@10, Recall@20, MRR
**Data split**: Leave-last-out on KuaiSAR-small

---

### Experiment 2 — Diversity & Coverage

**Goal**: Show that mechanism-coordinated multi-agent outperforms single-objective
ranking on diversity without sacrificing relevance.

**Metrics**:
- ILD (Intra-List Diversity): average pairwise item distance in embedding space
- Coverage: fraction of item categories represented in top-10
- α-NDCG: diversity-aware NDCG

**Ablation**: vary bid of Diversity Agent from 0 → 1 with other bids fixed, observe
NDCG vs ILD Pareto frontier.

---

### Experiment 3 — Monotonicity Verification

**Goal**: Empirically verify the theoretical result that linear aggregation is
monotone (incentive-compatible) while log-linear is not.

**Protocol**:
1. Fix all agents' distributions; increase a single agent's bid by Δ
2. Measure change in that agent's influence score in the final distribution
3. Linear should show non-decreasing influence (monotone)
4. Log-linear should show non-monotone behavior at certain configurations

**Metric**: `influence_delta` = change in KL-divergence between aggregated
distribution and the perturbing agent's distribution

---

### Experiment 4 — Manipulation Stress Test (Key Research Contribution)

**Goal**: Simulate a strategic agent that exaggerates its bid to gain disproportionate
influence; measure system robustness.

**Protocol**:
1. Set one agent (e.g., Business/Sponsor) to submit inflated bid = true_value × k
   for k ∈ {1, 2, 5, 10}
2. Run linear vs log-linear aggregation
3. Measure: (a) manipulation gain = increase in that agent's influence beyond its
   true contribution; (b) ranking quality degradation for other metrics

**Expected result**:
- Linear aggregation: bounded manipulation gain (second-price property limits gain)
- Log-linear aggregation: unbounded manipulation, quality collapse

**Connection to Paper 4 (Regret-Minimization)**: strategic agent's regret
under both mechanisms can be analyzed as an online learning problem.

---

### Experiment 5 — Audit Trace Interpretability

**Goal**: Demonstrate that the mechanism produces human-readable, position-level
influence attribution.

**Output format** (per query, per result position):

```json
{
  "query": "短视频 搜索",
  "position": 1,
  "item_id": "video_3829",
  "agents": {
    "relevance":       {"bid": 0.40, "influence_share": 0.38, "payment": 0.12},
    "personalization": {"bid": 0.30, "influence_share": 0.29, "payment": 0.09},
    "diversity":       {"bid": 0.20, "influence_share": 0.21, "payment": 0.06},
    "safety":          {"bid": 0.10, "influence_share": 0.12, "payment": 0.04}
  },
  "aggregation_rule": "linear",
  "final_score": 0.847
}
```

---

## Development Roadmap

```
Phase 1 (Weeks 1-2): Data + Candidate Pipeline
  - Download KuaiSAR-small
  - Implement kuaisar_loader.py and preprocessing.py
  - Build BM25 retrieval + Faiss dense retrieval
  - Set up unified candidate pool

Phase 2 (Weeks 3-4): Agents + Mechanism
  - Implement 4 scoring agents (relevance, personalization, diversity, safety)
  - Implement linear_aggregation + payment in mechanism/
  - Wire into LangGraph workflow
  - Unit test: monotonicity check on toy data

Phase 3 (Weeks 5-6): Experiments 1-3
  - Run baseline comparisons (fixed-weight vs linear_agg)
  - Implement and run log-linear as failure case
  - Diversity and monotonicity experiments

Phase 4 (Weeks 7-8): Manipulation Test + Report
  - Experiment 4: strategic bid stress test
  - Audit trace visualization (Streamlit demo)
  - Write midterm report + assemble results tables
```

---

## Installation

```bash
git clone https://github.com/SouthShore01/Mechanism-Aware-Multi-Agent-S-R-System.git
cd Mechanism-Aware-Multi-Agent-S-R-System
pip install -r requirements.txt
```

Download KuaiSAR-small:
```bash
# From Zenodo (record 8181109)
# KuaiSAR-small = KuaiSAR.zip on Zenodo (10-day subset, 2023/5/22-5/31)
wget https://zenodo.org/records/8181109/files/KuaiSAR.zip
unzip KuaiSAR.zip -d data/raw/
```

Run baseline experiment:
```bash
python experiments/run_baseline.py --config configs/experiment_configs/fixed_weight.yaml
```

---

## Key Theoretical Results Referenced

| Theorem (Duetting et al.) | Implication for this Project |
|---------------------------|------------------------------|
| Monotonicity ↔ IC | Linear aggregation is incentive-compatible; log-linear is not |
| Second-price under robust preferences | Agents cannot gain by overbidding under linear rule |
| Welfare-maximizing rule = weighted avg | Our linear aggregation is optimal under KL-divergence preference |

**Extension we test**: Do these properties hold when the decision unit is an
*item slate position* rather than a *token*? Our Experiments 3 & 4 test this.

---

## Positioning for LLM Internship Applications

This project demonstrates:

- **System design**: multi-agent LLM orchestration with LangGraph, not just a chatbot demo
- **Research depth**: mechanism design theory applied to a real retrieval/ranking system
- **Experimental rigor**: ablations, stress tests, offline evaluation with real industry data
- **Auditability**: position-level influence attribution — a key concern in production recommendation systems
- **Breadth**: touches LLM agents + search/IR + recsys + game theory + RL (MAPoRL connection)

**Target roles**: LLM Application Engineer, Agent Systems Researcher, Search & Recommendation Engineer, Applied Scientist (Intern)

---

## References

1. Duetting P, Mirrokni V, Paes Leme R, Xu H, Zuo S. Mechanism design for large language models. WWW 2024. arXiv:2310.10826.
2. Yang Y, Chai H, Shao S, et al. AgentNet: Decentralized evolutionary coordination for LLM-based multi-agent systems. arXiv:2504.00587. 2025.
3. Park C, Han S, Guo X, Ozdaglar AE, Zhang K, Kim JK. MAPoRL: Multi-agent post-co-training for collaborative LLMs with reinforcement learning. ACL 2025.
4. Park C, Chen Z, Ozdaglar A, Zhang K. Post-training LLMs as better decision-making agents: A regret-minimization approach. arXiv:2511.04393. 2025.
5. He R, McAuley J. KuaiSAR: A unified search and recommendation dataset. CIKM 2023.
