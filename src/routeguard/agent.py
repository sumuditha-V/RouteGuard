"""The LangGraph dispatch agent (M7).

Flow:  coordinator -> explanation -> recommendation -> critique -> (retry | finalize)

The golden rule: the LLM NEVER predicts. It receives the calibrated probability,
prediction, and SHAP drivers as READ-ONLY facts and only produces language
(explanation) and a proposed action (recommendation).

The Critique node is DETERMINISTIC CODE, not an LLM. The LLM proposes; code
validates. A recommendation that contradicts the SHAP drivers, restates the
probability, or is out of the allowed set is rejected no matter how the LLM
phrases it - which is exactly what makes the guardrail trustworthy and testable.

The LLM client is injected, so tests can pass a fake and run without an API key.
"""

import json
import os
from typing import Protocol, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from .config import PROJECT_ROOT, load_config

load_dotenv()

# --- which SHAP driver justifies which action (the deterministic guardrail table) ---
# Actions not listed (notify_customer, manual_review, no_action) need no specific driver.
ACTION_REQUIREMENTS = {
    "upgrade_shipping": {"delivery_window_days", "distance_km"},
    "change_warehouse": {"distance_km", "same_state"},
    # carrier is not a real Olist feature (see DATA_CARD); seller history is the proxy
    "choose_alternative_carrier": {"seller_hist_late_rate", "seller_hist_count"},
    "prioritize_dispatch": {"seller_hist_late_rate", "seller_hist_count"},
}


# ------------------------------- LLM client --------------------------------
class LLM(Protocol):
    def complete_text(self, system: str, user: str) -> str: ...
    def complete_json(self, system: str, user: str, schema: dict) -> dict: ...


class AnthropicLLM:
    """Real client. Reads ANTHROPIC_API_KEY from the environment (.env)."""

    def __init__(self):
        from anthropic import Anthropic

        cfg = load_config()["agent"]
        self.client = Anthropic()
        self.model = cfg["llm_model"]
        self.effort = cfg["effort"]
        self.max_tokens = cfg["max_tokens"]

    def _text(self, resp) -> str:
        return "".join(b.text for b in resp.content if b.type == "text")

    def complete_text(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
        )
        return self._text(resp)

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": schema}},
        )
        return json.loads(self._text(resp))


def _prompt(name: str) -> str:
    return (PROJECT_ROOT / "prompts" / f"{name}.md").read_text(encoding="utf-8")


# ------------------------------- graph state -------------------------------
class AgentState(TypedDict, total=False):
    order: dict                 # human-readable order details
    probability: float          # calibrated delay probability (FACT)
    prediction: int             # 1 = late, 0 = on time (FACT)
    threshold: float
    shap_drivers: list          # [{feature, value, shap, direction}, ...] (FACT)
    allowed_actions: list
    explanation: str
    recommendation: dict        # {action, rationale, justifying_feature}
    critique: dict              # {verdict, reason}
    retries: int
    final: dict


def _facts_block(state: AgentState) -> str:
    """Format the read-only facts the LLM nodes are allowed to use."""
    pct = round(state["probability"] * 100, 1)
    label = "LATE" if state["prediction"] == 1 else "ON TIME"
    lines = [f"Delay probability: {pct}% (FACT - do not change)",
             f"Prediction: {label} (threshold {state['threshold']})",
             "Top SHAP drivers:"]
    for d in state["shap_drivers"]:
        lines.append(f"  - {d['feature']} = {d['value']} "
                     f"(SHAP {d['shap']:+.3f}, {d['direction']})")
    lines.append(f"Order details: {json.dumps(state['order'])}")
    return "\n".join(lines)


# --------------------------------- nodes -----------------------------------
def coordinator(state: AgentState) -> dict:
    # basic input validation + init
    for key in ("probability", "prediction", "shap_drivers", "allowed_actions"):
        if key not in state:
            raise ValueError(f"coordinator: missing required input '{key}'")
    return {"retries": 0}


def make_explanation_node(llm: LLM):
    def explanation_worker(state: AgentState) -> dict:
        text = llm.complete_text(_prompt("explanation"), _facts_block(state))
        return {"explanation": text.strip()}
    return explanation_worker


def make_recommendation_node(llm: LLM):
    def recommendation_worker(state: AgentState) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": state["allowed_actions"]},
                "justifying_feature": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["action", "justifying_feature", "rationale"],
            "additionalProperties": False,
        }
        user = _facts_block(state)
        if state.get("critique") and state["critique"]["verdict"] == "reject":
            user += (f"\n\nPrevious attempt was REJECTED: "
                     f"{state['critique']['reason']}. Fix this specific problem.")
        rec = llm.complete_json(_prompt("recommendation"), user, schema)
        return {"recommendation": rec, "retries": state.get("retries", 0) + 1}
    return recommendation_worker


def critique(state: AgentState) -> dict:
    """DETERMINISTIC guardrail. Rejects recommendations that contradict the facts."""
    rec = state["recommendation"]
    action = rec.get("action")
    risk_features = {d["feature"] for d in state["shap_drivers"]
                     if d["direction"] == "increases_risk"}

    if action not in state["allowed_actions"]:
        return {"critique": {"verdict": "reject",
                             "reason": f"action '{action}' is not in the allowed set"}}
    if action == "no_action" and state["prediction"] == 1:
        return {"critique": {"verdict": "reject",
                             "reason": "order is predicted LATE but recommends no_action"}}

    required = ACTION_REQUIREMENTS.get(action)
    if required is not None:
        jf = rec.get("justifying_feature")
        if jf not in risk_features:
            return {"critique": {"verdict": "reject",
                                 "reason": f"justifying feature '{jf}' is not a SHAP "
                                           f"driver increasing risk for this order"}}
        if jf not in required:
            return {"critique": {"verdict": "reject",
                                 "reason": f"feature '{jf}' does not justify "
                                           f"action '{action}'"}}
    return {"critique": {"verdict": "approve", "reason": "passes all guardrail checks"}}


def _route(state: AgentState) -> str:
    max_retries = load_config()["agent"]["max_critique_retries"]
    if state["critique"]["verdict"] == "approve":
        return "finalize"
    if state.get("retries", 0) > max_retries:   # attempts exhausted
        return "finalize"
    return "retry"


def finalize(state: AgentState) -> dict:
    approved = state["critique"]["verdict"] == "approve"
    if approved:
        action = state["recommendation"]["action"]
        rationale = state["recommendation"]["rationale"]
    else:
        # guardrail never approved -> safe fallback
        action = "manual_review"
        rationale = (f"Automatic recommendation rejected by guardrail "
                     f"({state['critique']['reason']}). Escalated for manual review.")
    pct = round(state["probability"] * 100, 1)
    return {"final": {
        "delay_probability": state["probability"],
        "prediction": "late" if state["prediction"] == 1 else "on_time",
        "explanation": state.get("explanation", ""),
        "recommended_action": action,
        "reasoning": rationale,
        "confidence_statement": f"The ML model estimates a {pct}% chance of late delivery.",
        "guardrail": state["critique"]["verdict"],
    }}


def build_agent(llm: LLM):
    """Wire the LangGraph. `llm` is injected so tests can pass a fake."""
    g = StateGraph(AgentState)
    g.add_node("coordinator", coordinator)
    g.add_node("explanation", make_explanation_node(llm))
    g.add_node("recommendation", make_recommendation_node(llm))
    g.add_node("critique", critique)
    g.add_node("finalize", finalize)

    g.add_edge(START, "coordinator")
    g.add_edge("coordinator", "explanation")
    g.add_edge("explanation", "recommendation")
    g.add_edge("recommendation", "critique")
    g.add_conditional_edges("critique", _route,
                            {"retry": "recommendation", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


def run_agent(state: AgentState, llm: LLM = None) -> dict:
    """Convenience: build + invoke, returning the final response dict."""
    llm = llm or AnthropicLLM()
    result = build_agent(llm).invoke(state)
    return result["final"]
