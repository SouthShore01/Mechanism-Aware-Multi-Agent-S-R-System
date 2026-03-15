"""
Main auction orchestrator: wires agents → aggregation → payment → ranked slate.

This is the central module of the system. Given a query and candidate pool,
it collects all agent outputs, runs the chosen aggregation rule, computes
payments, and returns a ranked slate with a full audit trace.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal, Optional
import numpy as np

from src.agents.base_agent import ScoringAgent, AgentOutput
from src.mechanism.aggregation import (
    linear_aggregation,
    loglinear_aggregation,
    get_influence_shares,
)
from src.mechanism.payment import compute_payments


AggregationRule = Literal["linear", "loglinear"]


@dataclass
class AuctionResult:
    """Full output of one auction run."""
    query: str
    ranked_items: List[Dict[str, Any]]          # Top-L items in ranked order
    aggregated_scores: np.ndarray               # Full distribution over candidates
    agent_outputs: List[AgentOutput]            # Raw agent distributions + bids
    influence_shares: Dict[str, float]          # Per-agent influence in final output
    payments: Dict[str, float]                  # Per-agent second-price-style payment
    aggregation_rule: AggregationRule
    audit_trace: List[Dict[str, Any]] = field(default_factory=list)  # Per-position log


class SlateAuction:
    """
    Runs the full slate auction for a query.

    Usage:
        auction = SlateAuction(agents=[rel_agent, pers_agent, div_agent, safety_agent])
        result = auction.run(query="短视频 搜索", candidates=[...], top_k=10)
        print(result.audit_trace)
    """

    def __init__(
        self,
        agents: List[ScoringAgent],
        aggregation_rule: AggregationRule = "linear",
        compute_payment: bool = True,
    ):
        self.agents = agents
        self.aggregation_rule = aggregation_rule
        self.compute_payment = compute_payment

    def run(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> AuctionResult:
        """
        Execute the full auction pipeline:
        1. Collect agent score distributions and bids
        2. Aggregate with chosen rule
        3. Compute payments (if enabled)
        4. Rank candidates by aggregated score
        5. Build audit trace per position
        """
        # Step 1: Collect agent outputs
        agent_outputs: List[AgentOutput] = [
            agent(query, candidates, user_context) for agent in self.agents
        ]

        # Step 2: Aggregate
        if self.aggregation_rule == "linear":
            aggregated = linear_aggregation(agent_outputs)
        else:
            aggregated = loglinear_aggregation(agent_outputs)

        # Step 3: Influence shares
        influence_shares = get_influence_shares(agent_outputs, aggregated)

        # Step 4: Payments
        payments = compute_payments(agent_outputs) if self.compute_payment else {}

        # Step 5: Rank candidates
        ranked_indices = np.argsort(aggregated)[::-1][:top_k]
        ranked_items = [candidates[i] for i in ranked_indices]

        # Step 6: Build per-position audit trace
        audit_trace = []
        for pos, (idx, item) in enumerate(zip(ranked_indices, ranked_items)):
            entry = {
                "position": pos + 1,
                "item_id": item.get("item_id", str(idx)),
                "aggregated_score": float(aggregated[idx]),
                "agents": {
                    ao.agent_name: {
                        "bid": ao.bid,
                        "score": float(ao.scores[idx]),
                        "influence_share": influence_shares[ao.agent_name],
                        "payment": payments.get(ao.agent_name, 0.0),
                    }
                    for ao in agent_outputs
                },
                "aggregation_rule": self.aggregation_rule,
            }
            audit_trace.append(entry)

        return AuctionResult(
            query=query,
            ranked_items=ranked_items,
            aggregated_scores=aggregated,
            agent_outputs=agent_outputs,
            influence_shares=influence_shares,
            payments=payments,
            aggregation_rule=self.aggregation_rule,
            audit_trace=audit_trace,
        )
