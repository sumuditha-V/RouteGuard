"""Agent guardrail tests - the second headline correctness guarantee.

These prove the deterministic Critique node rejects recommendations that contradict
the SHAP explanation, no matter what the LLM proposes. A FakeLLM stands in for
Claude so the tests need no API key.
"""

from routeguard.agent import run_agent

DRIVERS = [
    {"feature": "delivery_window_days", "value": 3, "shap": 2.10, "direction": "increases_risk"},
    {"feature": "seller_hist_late_rate", "value": 0.09, "shap": 0.24, "direction": "increases_risk"},
    {"feature": "purchase_month", "value": 8, "shap": -0.11, "direction": "decreases_risk"},
]
ALLOWED = ["upgrade_shipping", "change_warehouse", "prioritize_dispatch",
           "notify_customer", "manual_review", "no_action"]


class FakeLLM:
    """Returns a fixed explanation and a caller-chosen recommendation every time."""
    def __init__(self, rec):
        self.rec = rec

    def complete_text(self, system, user):
        return "The order looks risky because of a very tight 3-day delivery window."

    def complete_json(self, system, user, schema):
        return dict(self.rec)


def _state(prediction=1):
    return {
        "order": {"order_id": "abc", "route": "SP->BA"},
        "probability": 0.78,
        "prediction": prediction,
        "threshold": 0.16,
        "shap_drivers": DRIVERS,
        "allowed_actions": ALLOWED,
    }


def test_recommendation_contradicting_shap_is_rejected():
    """Action justified by a feature that DECREASES risk must be rejected,
    and the agent must fall back to manual_review."""
    bad = {"action": "upgrade_shipping", "justifying_feature": "purchase_month",
           "rationale": "ship faster"}
    final = run_agent(_state(), llm=FakeLLM(bad))
    assert final["guardrail"] == "reject"
    assert final["recommended_action"] == "manual_review"


def test_feature_that_does_not_justify_action_is_rejected():
    """delivery_window_days increases risk, but it does not justify change_warehouse."""
    bad = {"action": "change_warehouse", "justifying_feature": "delivery_window_days",
           "rationale": "move warehouse"}
    final = run_agent(_state(), llm=FakeLLM(bad))
    assert final["guardrail"] == "reject"


def test_no_action_on_predicted_late_is_rejected():
    bad = {"action": "no_action", "justifying_feature": "delivery_window_days",
           "rationale": "do nothing"}
    final = run_agent(_state(prediction=1), llm=FakeLLM(bad))
    assert final["guardrail"] == "reject"


def test_valid_recommendation_is_approved():
    """A well-justified action passes the guardrail and is returned as-is."""
    good = {"action": "upgrade_shipping", "justifying_feature": "delivery_window_days",
            "rationale": "The 3-day window is very tight, so expedite shipping."}
    final = run_agent(_state(), llm=FakeLLM(good))
    assert final["guardrail"] == "approve"
    assert final["recommended_action"] == "upgrade_shipping"
    assert "78" in final["confidence_statement"]  # probability stated, not invented
