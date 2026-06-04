"""Unit tests for the OpenRouter-backed AI assistant core.

Covers the pure, Streamlit-free logic: the agent loop, the HTTP client (mocked),
and each tool's underlying logic. The Streamlit wiring in app.py is exercised by
the manual smoke test described in the design spec, not here.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import ai_assistant as ai


# ── fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def clean_df() -> pd.DataFrame:
    """Tiny stand-in for clean_data.csv with the columns the tools touch."""
    return pd.DataFrame(
        {
            "Shipping Mode": [
                "Same Day",
                "Same Day",
                "Standard Class",
                "Standard Class",
                "Standard Class",
            ],
            "Market": ["LATAM", "Europe", "LATAM", "Europe", "Europe"],
            "Order Region": ["South", "West", "South", "West", "West"],
            "Customer Segment": [
                "Consumer",
                "Consumer",
                "Home Office",
                "Consumer",
                "Corporate",
            ],
            "Department Name": ["Apparel", "Golf", "Apparel", "Golf", "Golf"],
            "Category Name": ["Shirts", "Cleats", "Shirts", "Cleats", "Cleats"],
            "Type": ["DEBIT", "TRANSFER", "DEBIT", "CASH", "DEBIT"],
            "Order Item Quantity": [1, 2, 1, 3, 1],
            "Sales": [100.0, 200.0, 150.0, 300.0, 120.0],
            "Order Item Discount Rate": [0.1, 0.0, 0.2, 0.05, 0.1],
            "Late_delivery_risk": [1, 0, 1, 1, 0],
        }
    )


class FakeClient:
    """Scripted OpenRouterClient: returns queued assistant messages in order."""

    def __init__(self, scripted_messages):
        self._queue = list(scripted_messages)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(list(messages))
        if not self._queue:
            raise AssertionError("FakeClient ran out of scripted responses")
        return self._queue.pop(0)


def _tool_call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


# ── run_agent ─────────────────────────────────────────────────────────────────
def test_run_agent_returns_text_when_no_tool_calls():
    client = FakeClient([{"role": "assistant", "content": "Hello there."}])
    text, actions = ai.run_agent(client, [{"role": "user", "content": "hi"}], {})
    assert text == "Hello there."
    assert actions == []


def test_run_agent_dispatches_single_tool_call_then_returns_text():
    client = FakeClient(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c1", "query_dataset", {"metric": "x"})],
            },
            {"role": "assistant", "content": "The late rate is 60%."},
        ]
    )
    seen = {}

    def query_dataset(**kwargs):
        seen.update(kwargs)
        return {"overall_late_rate": 0.6}

    text, actions = ai.run_agent(
        client,
        [{"role": "user", "content": "rate?"}],
        {"query_dataset": query_dataset},
    )
    assert text == "The late rate is 60%."
    assert seen == {"metric": "x"}
    assert len(actions) == 1
    assert "query_dataset" in actions[0]


def test_run_agent_chains_multiple_tool_calls():
    client = FakeClient(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c1", "set_inputs", {"market": "LATAM"})],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c2", "get_pred", {})],
            },
            {"role": "assistant", "content": "Done."},
        ]
    )
    impl = {"set_inputs": lambda **k: {"applied": k}, "get_pred": lambda: {"p": 0.5}}
    text, actions = ai.run_agent(client, [{"role": "user", "content": "go"}], impl)
    assert text == "Done."
    assert len(actions) == 2


def test_run_agent_feeds_tool_error_back_and_continues():
    client = FakeClient(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c1", "boom", {})],
            },
            {"role": "assistant", "content": "Recovered."},
        ]
    )

    def boom():
        raise ValueError("kaboom")

    text, actions = ai.run_agent(
        client, [{"role": "user", "content": "go"}], {"boom": boom}
    )
    assert text == "Recovered."
    # the error must have been passed back to the model as a tool result
    tool_msgs = [m for m in client.calls[-1] if m.get("role") == "tool"]
    assert any("kaboom" in m["content"] for m in tool_msgs)


def test_run_agent_handles_unknown_tool_name():
    client = FakeClient(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c1", "nope", {})],
            },
            {"role": "assistant", "content": "ok"},
        ]
    )
    text, _ = ai.run_agent(client, [{"role": "user", "content": "go"}], {})
    assert text == "ok"
    tool_msgs = [m for m in client.calls[-1] if m.get("role") == "tool"]
    assert any("unknown tool" in m["content"].lower() for m in tool_msgs)


def test_run_agent_stops_at_iteration_cap():
    # Always returns a tool call → would loop forever without the cap.
    looping = {
        "role": "assistant",
        "content": "still working",
        "tool_calls": [_tool_call("c", "noop", {})],
    }
    client = FakeClient([dict(looping) for _ in range(10)])
    text, actions = ai.run_agent(
        client,
        [{"role": "user", "content": "go"}],
        {"noop": lambda: {"ok": True}},
        max_iters=3,
    )
    assert len(client.calls) == 3
    assert any("max" in a.lower() for a in actions)


# ── tool logic: set_shipment_inputs ───────────────────────────────────────────
def test_set_inputs_accepts_valid_categorical(clean_df):
    out = ai.set_shipment_inputs_logic(clean_df, {"shipping_mode": "Same Day"})
    assert out["applied"]["Shipping Mode"] == "Same Day"
    assert not out["rejected"]


def test_set_inputs_rejects_unknown_categorical_with_options(clean_df):
    out = ai.set_shipment_inputs_logic(clean_df, {"market": "Atlantis"})
    assert "Market" not in out["applied"]
    assert "market" in out["rejected"]
    assert "LATAM" in out["rejected"]["market"]  # lists the valid options


def test_set_inputs_coerces_numeric_fields(clean_df):
    out = ai.set_shipment_inputs_logic(
        clean_df, {"item_quantity": 3, "sales": 250.5, "discount_rate": 0.2}
    )
    assert out["applied"]["Order Item Quantity"] == 3
    assert out["applied"]["Sales"] == 250.5
    assert out["applied"]["Order Item Discount Rate"] == 0.2


def test_set_inputs_ignores_unknown_field_name(clean_df):
    out = ai.set_shipment_inputs_logic(clean_df, {"not_a_field": "x"})
    assert not out["applied"]
    assert "not_a_field" in out["rejected"]


# ── tool logic: query_dataset ─────────────────────────────────────────────────
def test_query_overall_late_rate(clean_df):
    out = ai.query_dataset_logic(clean_df, "overall_late_rate")
    assert out["overall_late_rate"] == pytest.approx(3 / 5)


def test_query_late_rate_by_group(clean_df):
    out = ai.query_dataset_logic(clean_df, "late_rate_by", "Shipping Mode")
    assert out["late_rate_by"]["Same Day"] == pytest.approx(0.5)
    assert out["late_rate_by"]["Standard Class"] == pytest.approx(2 / 3)


def test_query_count_by_group(clean_df):
    out = ai.query_dataset_logic(clean_df, "count_by", "Market")
    assert out["count_by"]["Europe"] == 3
    assert out["count_by"]["LATAM"] == 2


def test_query_rejects_unknown_group_col(clean_df):
    out = ai.query_dataset_logic(clean_df, "late_rate_by", "Nonsense")
    assert "error" in out


def test_query_rejects_unknown_metric(clean_df):
    out = ai.query_dataset_logic(clean_df, "teleport")
    assert "error" in out


# ── tool logic: describe_prediction ───────────────────────────────────────────
def test_describe_prediction_none_returns_note():
    out = ai.describe_prediction_logic(None)
    assert "note" in out


def test_describe_prediction_passes_through_fields():
    pred = {"p_late": 0.72, "tier": "HIGH", "mode": "Same Day", "base_rate": 0.5}
    out = ai.describe_prediction_logic(pred)
    assert out["p_late"] == 0.72
    assert out["tier"] == "HIGH"


# ── tool logic: switch_model ──────────────────────────────────────────────────
def test_switch_model_accepts_available_file():
    out = ai.switch_model_logic("xgb_model.pkl", ["primary_model.pkl", "xgb_model.pkl"])
    assert out["applied"] == "xgb_model.pkl"


def test_switch_model_rejects_missing_file():
    out = ai.switch_model_logic("ghost.pkl", ["primary_model.pkl"])
    assert "error" in out
    assert "primary_model.pkl" in out["error"]  # lists what's available


# ── schemas ───────────────────────────────────────────────────────────────────
def test_tool_schemas_cover_the_four_tools():
    names = {s["function"]["name"] for s in ai.TOOL_SCHEMAS}
    assert names == {
        "set_shipment_inputs",
        "get_current_prediction",
        "query_dataset",
        "switch_model",
    }


def test_tool_schemas_are_well_formed():
    for schema in ai.TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"


# ── OpenRouterClient (mocked transport) ───────────────────────────────────────
class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.last_post = None

    def post(self, url, **kwargs):
        self.last_post = {"url": url, **kwargs}
        if self._exc is not None:
            raise self._exc
        return self._response


def test_client_chat_returns_assistant_message():
    payload = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    session = FakeSession(FakeResponse(200, payload))
    client = ai.OpenRouterClient("key", "deepseek/deepseek-v4-flash", session=session)
    msg = client.chat([{"role": "user", "content": "yo"}], tools=ai.TOOL_SCHEMAS)
    assert msg["content"] == "hi"
    sent = (
        json.loads(session.last_post["data"])
        if "data" in session.last_post
        else session.last_post["json"]
    )
    assert sent["model"] == "deepseek/deepseek-v4-flash"
    assert sent["tools"] == ai.TOOL_SCHEMAS


def test_client_chat_raises_on_non_200():
    session = FakeSession(FakeResponse(401, {"error": {"message": "bad key"}}))
    client = ai.OpenRouterClient("key", "m", session=session)
    with pytest.raises(ai.OpenRouterError) as exc:
        client.chat([{"role": "user", "content": "x"}])
    assert "bad key" in str(exc.value)


def test_client_chat_wraps_transport_errors():
    session = FakeSession(exc=RuntimeError("connection reset"))
    client = ai.OpenRouterClient("key", "m", session=session)
    with pytest.raises(ai.OpenRouterError):
        client.chat([{"role": "user", "content": "x"}])
