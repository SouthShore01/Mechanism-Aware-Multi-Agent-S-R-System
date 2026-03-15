# Mid-Term Report Outline (10 pages, ICML single-column format)

**Title**: Mechanism-Aware Coordination for LLM-Based Multi-Agent Search and Recommendation Systems

**Deadline**: March 24, 2026, 11:59 PM ET

---

## Section 1 — Introduction (1 page)

**Goal**: Motivate the problem and state your contribution clearly.

- Para 1: Search and recommendation systems serve billions of users; modern systems
  must balance relevance, personalization, diversity, safety, and business objectives
  simultaneously. Traditional multi-objective ranking uses hand-tuned weights —
  opaque, unauditable, and vulnerable to silent objective dominance.

- Para 2: The rise of LLM-based agents opens a new design space. Instead of static
  weights, each objective can be represented as an **LLM agent** that expresses its
  preference as a distribution over candidates and a scalar bid. The question
  becomes: *how should these bids and distributions be aggregated in an
  incentive-compatible, interpretable way?*

- Para 3: We propose applying the **mechanism design framework** from Duetting et al.
  (2024) — originally designed for token-level LLM aggregation — to the problem of
  **slate-level search and recommendation**. Our key contribution is testing whether
  the theoretical properties (monotonicity, incentive compatibility, second-price
  payments) transfer to the item-ranking setting.

- Para 4: We survey four papers that together cover the mechanism design backbone,
  multi-agent architecture design, learning-to-collaborate approaches, and
  decision-making theory for LLM agents. We conclude with a proposed system design
  and experimental plan.

**Figures**: System overview figure (from README architecture)

---

## Section 2 — Problem Formulation (1 page)

**Goal**: Formally define the multi-agent S&R problem.

**2.1 Setting**
- Query q or user context u
- Candidate pool C = retrieval_candidates ∪ rec_candidates (|C| = K, typically 100-500)
- N agents, each with objective oᵢ (relevance, personalization, diversity, safety, business)
- Each agent i outputs: score distribution pᵢ ∈ Δ(C) and bid bᵢ ∈ ℝ₊
- Mechanism: aggregation function f(p₁,...,pₙ, b₁,...,bₙ) → q ∈ Δ(C)
- Final slate: top-L items sampled / ranked from q

**2.2 Desiderata**
- Incentive compatibility: agents cannot gain by misreporting bids
- Monotonicity: higher bid → weakly more influence in output
- Interpretability: output decomposable into per-agent influence shares
- Efficiency: ranking quality (NDCG, diversity) not worse than fixed-weight baseline

**2.3 Connection to Token Auction**
- Token auction: each step generates one token from aggregated LLM distribution
- Slate auction: each position generates one item from aggregated agent distribution
- The mapping is structural: Δ(vocabulary) → Δ(candidate_pool)

---

## Section 3 — Paper 1: Mechanism Design for LLMs (2 pages)

**Goal**: Deep dive on the core paper; extract exactly what transfers to your setting.

**3.1 Token Auction Framework**
- Model: n LLM agents, each represented as pᵢ(·|context), bid bᵢ
- Distribution aggregation function: f(p₁,...,pₙ, b) → q
- Payment rule: tᵢ(p, b) → ℝ

**3.2 Key Theoretical Results**
- Definition: monotone aggregation (agent i increases bid → q moves closer to pᵢ)
- Theorem: IC + IR ↔ monotone aggregation (under robust preferences)
- Theorem: any monotone aggregation admits second-price-style payments
- Concrete rules:
  - Linear: q(x) = Σ λᵢ pᵢ(x), λᵢ = bᵢ/Σbⱼ  [MONOTONE]
  - Log-linear: q(x) ∝ Π pᵢ(x)^λᵢ             [NOT monotone]

**3.3 What Transfers to Slate Setting**
- The aggregation functions are distribution-agnostic; apply directly over Δ(C)
- Monotonicity condition is domain-independent
- Payment design applies position-by-position

**3.4 What Doesn't Transfer (Open Questions)**
- Token-level: agents see partial sequence context at each step
- Slate-level: agents see full query but must score full candidate pool
- Position interdependence: item at position 1 affects utility of item at position 2

---

## Section 4 — Paper 2: AgentNet (1.5 pages)

**Goal**: Contrast coordination-by-communication (AgentNet) with coordination-by-incentive (our approach).

**4.1 AgentNet Overview**
- Decentralized multi-agent system with dynamic routing over agent graph
- Each agent: LLM + tool use + RAG memory + inter-agent message passing
- Evolutionary: agents can be added/removed; system adapts topology

**4.2 Coordination Architecture**
- Agent DAG: nodes = agents, edges = communication channels
- Message routing: based on agent capability and task decomposition
- Memory sharing: RAG-based cross-agent knowledge

**4.3 Contrast with Mechanism-Based Coordination**
| Dimension | AgentNet | This Project |
|-----------|----------|-------------|
| Coordination signal | Message passing | Bid + distribution submission |
| Conflict resolution | Consensus / voting | Auction mechanism |
| Interpretability | Agent communication logs | Per-position influence audit |
| Incentive structure | None (cooperative assumed) | Explicit (IC guarantee) |
| Strategic robustness | Not analyzed | Tested explicitly |

**4.4 Integration Point**
- We adopt AgentNet's **node-based agent graph** for our pipeline architecture
- We replace its communication-based aggregation with **auction-based aggregation**
- LangGraph implements the DAG; our mechanism module is the aggregation node

---

## Section 5 — Paper 3: MAPoRL (1 page)

**Goal**: Establish that mechanism design (rule-based) and RL co-training (learned) are complementary, not competing.

**5.1 MAPoRL Framework**
- Post-training phase: multiple LLM agents trained jointly via RL
- Reward: collaborative task success (not individual agent objective)
- Key insight: prompt-based multi-agent coordination leads to suboptimal collaboration;
  explicit RL training produces qualitatively better cooperative behavior

**5.2 Implication for This Project**
- MAPoRL shows that **bid strategies can be learned**, not just manually set
- Future direction: instead of agents submitting hand-crafted bids, train each
  agent's bidding policy via RL with mechanism-constrained reward
- Current project: use fixed bid policies; MAPoRL identifies the next research step

**5.3 Mechanism vs Learning Dichotomy**
| Approach | Mechanism Design | MAPoRL / RL |
|----------|-----------------|-------------|
| How coordination emerges | Auction rules | Post-training |
| Interpretability | High | Lower |
| Adaptation to new objectives | Reconfigure bids | Retrain |
| Formal guarantees | Yes (monotonicity, IC) | Empirical |

---

## Section 6 — Paper 4: Regret-Minimization for LLM Agents (1 page)

**Goal**: Frame the ranking problem as sequential decision-making; connect to regret analysis.

**6.1 Regret-Minimization Framework**
- LLMs as decision-making agents: at each step, choose action (item to rank) with
  goal of minimizing regret relative to best fixed policy in hindsight
- Post-training via regret signals produces agents with stronger decision quality
  and robustness to distributional shift

**6.2 Connection to Manipulation Stress Test**
- When a strategic agent inflates its bid by factor k, the other agents' effective
  influence decreases — they "regret" not inflating their own bids
- Under linear aggregation: regret is bounded (second-price prevents runaway gain)
- Under log-linear aggregation: regret is unbounded (no monotone payment exists)
- This is precisely our Experiment 4

**6.3 Broader Framing**
- This paper justifies treating the auction mechanism as an **online decision system**,
  not just a static ranker
- Opens direction: per-query regret tracking as an evaluation metric for mechanism
  robustness over time

---

## Section 7 — Proposed System (1.5 pages)

**Goal**: Describe your project proposal in detail.

**7.1 System Components** (reference README architecture figure)
- Candidate generation: BM25 + Faiss dense retrieval + RecBole sequential rec
- Agent layer: Relevance, Personalization, Diversity, Safety (4 agents, Phase 1)
- Mechanism layer: linear aggregation + critical-bid payment + audit logger
- Evaluation layer: NDCG, diversity, monotonicity check, manipulation test

**7.2 Dataset: KuaiSAR**
- 25K users, 6.9M items, 453K queries, 19.6M actions
- Unique: captures real user transitions between search and recommendation modes
- Evaluation: leave-last-out; separate search queries and rec sessions

**7.3 Experiment Plan** (summarize 5 experiments from README)
- Table: experiment, method variants, metrics, expected finding

**7.4 Minimum Viable Timeline**
- Phase 1 (wk 1-2): data pipeline
- Phase 2 (wk 3-4): agents + mechanism
- Phase 3 (wk 5-6): experiments 1-3
- Phase 4 (wk 7-8): manipulation test + report

---

## Section 8 — Discussion (0.5 pages)

**Goal**: Honestly assess limitations and future directions.

**Limitations**
- Offline evaluation cannot fully capture dynamic bidding behavior
- Agents currently use fixed bid policies (not learned)
- Position interdependence not fully modeled in current mechanism
- No real monetary payments; "influence cost" is a proxy

**Future Directions**
- RL-learned bidding policies (MAPoRL connection)
- Online A/B evaluation on live traffic
- Extension to conversational search (multi-turn slate)
- Formal proof of monotonicity for discrete item distributions

---

## Section 9 — Conclusion (0.5 pages)

- Restate the core idea: mechanism design framework from token auctions transfers
  meaningfully to slate-level S&R
- Summarize the 4-paper narrative arc:
  mechanism design theory → agent architecture → learned collaboration → decision quality
- State the research contribution: empirical verification of monotonicity transfer +
  strategic manipulation analysis in S&R setting
- State the engineering contribution: auditable, interpretable multi-agent ranking system

---

## References

Cite all 4 papers + KuaiSAR + any additional related work cited in body.
Target: ~10-15 references total.

---

## Writing Tips

- ICML single-column: ~500 words per page (excluding figures/tables)
- Each section should have at least one figure or table
- Use the problem formulation table in Section 2 as an anchor; refer back to it
- Make the "what transfers / what doesn't" framing explicit in Section 3 — this is
  where you show you actually understand the paper, not just summarize it
- The contrast tables in Sections 4 and 5 are what make a survey report stand out
  over a paper-by-paper description
