"""OpenRouter-backed AI assistant core for the Late-Delivery Risk app.

Streamlit-free by design: this module holds the HTTP client, the tool schemas,
the system prompt, the agent loop, and each tool's *pure* logic (operating on a
DataFrame / plain dicts). app.py wires these into ``st.session_state`` and the
loaded model. Keeping the logic here makes it unit-testable without a running app.

The honest framing the system prompt enforces: on this DataCo dataset the signal
concentrates in shipping mode, and the four ML models add only ~+0.01 ROC-AUC
over a per-mode base-rate lookup. The assistant must not overclaim model skill.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd

try:  # `requests` is the default transport; injectable for tests.
    import requests
except ImportError:  # pragma: no cover - requests is a declared dependency
    requests = None  # type: ignore[assignment]

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_TIMEOUT = 15

TARGET_COL = "Late_delivery_risk"

# Map the assistant-facing argument names to real clean_data.csv columns.
CATEGORICAL_FIELDS: dict[str, str] = {
    "shipping_mode": "Shipping Mode",
    "market": "Market",
    "order_region": "Order Region",
    "customer_segment": "Customer Segment",
    "department": "Department Name",
    "category": "Category Name",
    "payment_type": "Type",
}
NUMERIC_FIELDS: dict[str, str] = {
    "item_quantity": "Order Item Quantity",
    "sales": "Sales",
    "discount_rate": "Order Item Discount Rate",
}


class OpenRouterError(RuntimeError):
    """Raised for any failed OpenRouter call (HTTP error or transport failure)."""


class OpenRouterClient:
    """Thin OpenAI-compatible chat-completions client for OpenRouter."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        session: Any | None = None,
    ) -> None:
        # Strip stray whitespace/newlines from pasted credentials.
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url
        self.timeout = timeout
        self.session = session if session is not None else requests

    def _headers(self) -> dict[str, str]:
        """Build request headers, validating they are Latin-1 encodable.

        HTTP header values must be Latin-1. A key pasted with a non-ASCII
        character (commonly a "smart" em-dash from a formatted copy) would
        otherwise surface as a cryptic ``'latin-1' codec can't encode`` error —
        translate that into an actionable message instead.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json; charset=utf-8",
            "HTTP-Referer": "https://github.com/its-dreOwO/DAP",
            "X-Title": "DAP391m Late-Delivery Risk Predictor",
        }
        for name, value in headers.items():
            try:
                value.encode("latin-1")
            except UnicodeEncodeError:
                where = (
                    "your API key contains a non-ASCII character (e.g. a "
                    '"smart" em-dash from copy-paste) — re-paste it as plain text'
                    if name == "Authorization"
                    else f"request header {name!r} contains a non-ASCII character"
                )
                raise OpenRouterError(f"Cannot send request: {where}.") from None
        return headers

    def chat(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> dict[str, Any]:
        """Send one chat-completions request; return the first choice's message."""
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = self._headers()
        try:
            resp = self.session.post(
                self.base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001 - normalize all transport failures
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

        if resp.status_code != 200:
            raise OpenRouterError(
                f"OpenRouter returned {resp.status_code}: {resp.text}"
            )
        body = resp.json()
        try:
            return body["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise OpenRouterError(f"Unexpected OpenRouter response: {body}") from exc

    def stream_chat(self, messages: list[dict], tools: list[dict] | None = None):
        """Yield raw streaming chunk dicts (OpenAI delta format) via SSE."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = self._headers()
        try:
            resp = self.session.post(
                self.base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                timeout=self.timeout,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001 - normalize all transport failures
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

        if resp.status_code != 200:
            raise OpenRouterError(
                f"OpenRouter returned {resp.status_code}: {resp.text}"
            )
        yield from iter_sse_json(resp.iter_lines())


# ── SSE streaming helpers ─────────────────────────────────────────────────────
def iter_sse_json(lines: Any) -> Any:
    """Parse OpenRouter SSE ``data:`` lines into JSON dicts; stop at ``[DONE]``."""
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        try:
            yield json.loads(data)
        except ValueError:
            continue


def assemble_message_stream(chunks: Any):
    """Consume streaming chunk dicts, yielding text fragments as they arrive.

    Returns (via ``StopIteration.value``) the fully assembled assistant message,
    merging streamed tool-call fragments (each ``function.arguments`` delta is
    concatenated per tool-call index).
    """
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, str]] = {}
    for chunk in chunks:
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
            yield delta["content"]
        for call in delta.get("tool_calls") or []:
            idx = call.get("index", 0)
            slot = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if call.get("id"):
                slot["id"] = call["id"]
            fn = call.get("function") or {}
            if fn.get("name"):
                slot["name"] = fn["name"]
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts) or None,
    }
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": slot["id"],
                "type": "function",
                "function": {"name": slot["name"], "arguments": slot["arguments"]},
            }
            for _, slot in sorted(tool_calls.items())
        ]
    return message


# ── agent loop ────────────────────────────────────────────────────────────────
def _parse_arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def _dispatch_tool(
    name: str, args: dict, tools_impl: dict[str, Callable[..., Any]]
) -> tuple[Any, str]:
    """Run one tool, returning (result, human-readable action). Errors and
    unknown tools become an error result fed back to the model, not a crash."""
    if name not in tools_impl:
        result: Any = {"error": f"unknown tool: {name}"}
    else:
        try:
            result = tools_impl[name](**args)
        except Exception as exc:  # noqa: BLE001 - surface to the model
            result = {"error": str(exc)}
    return result, _describe_action(name, args, result)


def _tool_result_message(call: dict, result: Any) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call.get("id", ""),
        "content": json.dumps(result, default=str),
    }


def _last_assistant_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return ""


def run_agent(
    client: Any,
    messages: list[dict],
    tools_impl: dict[str, Callable[..., Any]],
    *,
    max_iters: int = 5,
) -> tuple[str, list[str]]:
    """Drive the tool-calling loop until the model returns plain text.

    Returns ``(final_text, actions)`` where ``actions`` is a human-readable list
    of the tool effects (for an "Actions taken" display). Tool exceptions and
    unknown tool names are fed back to the model as tool results so it can
    self-correct rather than crashing.
    """
    working = list(messages)
    actions: list[str] = []

    for _ in range(max_iters):
        message = client.chat(working, TOOL_SCHEMAS)
        working.append(message)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return (message.get("content") or "", actions)

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = _parse_arguments(fn.get("arguments"))
            result, action = _dispatch_tool(name, args, tools_impl)
            actions.append(action)
            working.append(_tool_result_message(call, result))

    actions.append("(stopped: reached max tool iterations)")
    return (_last_assistant_text(working), actions)


def run_agent_stream(
    client: Any,
    messages: list[dict],
    tools_impl: dict[str, Callable[..., Any]],
    *,
    max_iters: int = 5,
):
    """Streaming variant of :func:`run_agent`, yielding events for live UI.

    Yields dicts with a ``kind`` of:
    - ``"token"``  — a fragment of the final answer (stream into the chat bubble)
    - ``"tool_start"`` / ``"tool_end"`` — a tool is about to run / has finished
    - ``"final"`` — the last event, with the full ``text`` and ``actions`` list

    Requires ``client.stream_chat`` (SSE). Tool side effects happen between the
    start/end events, exactly as in the non-streaming loop.
    """
    working = list(messages)
    actions: list[str] = []

    for _ in range(max_iters):
        gen = assemble_message_stream(client.stream_chat(working, TOOL_SCHEMAS))
        message: dict[str, Any] = {}
        while True:
            try:
                fragment = next(gen)
            except StopIteration as stop:
                message = stop.value
                break
            yield {"kind": "token", "text": fragment}

        working.append(message)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            yield {
                "kind": "final",
                "text": message.get("content") or "",
                "actions": actions,
            }
            return

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = _parse_arguments(fn.get("arguments"))
            yield {"kind": "tool_start", "name": name, "args": args}
            result, action = _dispatch_tool(name, args, tools_impl)
            actions.append(action)
            yield {"kind": "tool_end", "name": name, "summary": action}
            working.append(_tool_result_message(call, result))

    actions.append("(stopped: reached max tool iterations)")
    yield {
        "kind": "final",
        "text": _last_assistant_text(working),
        "actions": actions,
    }


def _describe_action(name: str, args: dict, result: Any) -> str:
    summary = ""
    if isinstance(result, dict):
        if "summary" in result:
            summary = str(result["summary"])
        elif "error" in result:
            summary = f"error: {result['error']}"
    arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
    base = f"{name}({arg_str})" if arg_str else f"{name}()"
    return f"{base} — {summary}" if summary else base


# ── tool logic (pure; Streamlit-free) ─────────────────────────────────────────
def set_shipment_inputs_logic(clean_df: pd.DataFrame, changes: dict) -> dict:
    """Validate requested input changes against the real option lists.

    Returns ``{"applied": {col: value}, "rejected": {field: reason}, "summary"}``.
    Categoricals must match a value present in the data; unknown values are
    rejected with the valid options listed. Unknown field names are rejected.
    """
    applied: dict[str, Any] = {}
    rejected: dict[str, str] = {}

    for field, value in changes.items():
        if field in CATEGORICAL_FIELDS:
            col = CATEGORICAL_FIELDS[field]
            options = sorted(clean_df[col].dropna().astype(str).unique().tolist())
            if str(value) in options:
                applied[col] = str(value)
            else:
                shown = ", ".join(options[:12])
                rejected[field] = f"'{value}' is not valid. Options: {shown}"
        elif field in NUMERIC_FIELDS:
            col = NUMERIC_FIELDS[field]
            try:
                applied[col] = float(value) if "." in str(value) else int(value)
            except (TypeError, ValueError):
                rejected[field] = f"'{value}' is not a number"
        else:
            rejected[field] = f"unknown field '{field}'"

    parts = []
    if applied:
        parts.append("set " + ", ".join(f"{k}={v}" for k, v in applied.items()))
    if rejected:
        parts.append(f"rejected {len(rejected)}")
    return {"applied": applied, "rejected": rejected, "summary": "; ".join(parts)}


def query_dataset_logic(
    clean_df: pd.DataFrame, metric: str, group_col: str | None = None
) -> dict:
    """Compute a grounded statistic from the cleaned dataset."""
    if metric == "overall_late_rate":
        rate = float(clean_df[TARGET_COL].mean())
        return {"overall_late_rate": rate, "summary": f"overall late rate {rate:.1%}"}

    if metric in {"late_rate_by", "count_by"}:
        if not group_col or group_col not in clean_df.columns:
            cols = ", ".join(c for c in clean_df.columns if c != TARGET_COL)
            return {"error": f"unknown group_col '{group_col}'. Columns: {cols}"}
        if metric == "late_rate_by":
            grp = clean_df.groupby(group_col)[TARGET_COL].mean()
            table = {str(k): float(v) for k, v in grp.items()}
            return {"late_rate_by": table, "summary": f"late rate by {group_col}"}
        grp = clean_df.groupby(group_col).size()
        table = {str(k): int(v) for k, v in grp.items()}
        return {"count_by": table, "summary": f"counts by {group_col}"}

    return {"error": f"unknown metric '{metric}'"}


def describe_prediction_logic(last_prediction: dict | None) -> dict:
    """Return the current on-screen prediction, or a note if none has run."""
    if not last_prediction:
        return {
            "note": "No prediction has been run yet. Ask the user to click "
            "'Predict late-delivery risk' or change an input first."
        }
    return dict(last_prediction)


def switch_model_logic(model_file: str, available_files: list[str]) -> dict:
    """Validate a model swap against the model files present on disk."""
    if model_file in available_files:
        return {"applied": model_file, "summary": f"switched to {model_file}"}
    shown = ", ".join(available_files)
    return {"error": f"'{model_file}' not available. Choose from: {shown}"}


# ── tool schemas & system prompt ──────────────────────────────────────────────
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "set_shipment_inputs",
            "description": (
                "Change one or more shipment inputs in the app and re-run the "
                "prediction. Use to explore scenarios (e.g. compare shipping modes)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "shipping_mode": {
                        "type": "string",
                        "description": "Same Day / First Class / Second Class / "
                        "Standard Class — the dominant driver of late risk.",
                    },
                    "market": {"type": "string"},
                    "order_region": {"type": "string"},
                    "customer_segment": {"type": "string"},
                    "department": {"type": "string"},
                    "category": {"type": "string"},
                    "payment_type": {"type": "string"},
                    "item_quantity": {"type": "integer"},
                    "sales": {"type": "number"},
                    "discount_rate": {"type": "number"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_prediction",
            "description": (
                "Read the current on-screen prediction: P(late), risk tier, "
                "chosen shipping mode, that mode's historical base rate, the delta "
                "vs base, and the current inputs."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_dataset",
            "description": (
                "Look up a real statistic from the cleaned dataset to ground an "
                "answer in data rather than guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["late_rate_by", "count_by", "overall_late_rate"],
                    },
                    "group_col": {
                        "type": "string",
                        "description": "Column to group by, e.g. 'Shipping Mode', "
                        "'Market', 'Order Region'. Required for *_by metrics.",
                    },
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_model",
            "description": (
                "Swap the active trained model (e.g. primary_model.pkl vs "
                "xgb_model.pkl) and re-run, to compare predictions across models."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_file": {
                        "type": "string",
                        "description": "Filename of a model under model_outputs/.",
                    }
                },
                "required": ["model_file"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are an assistant embedded in a Streamlit app that predicts "
    "late-delivery risk for retail shipments using the DataCo Smart Supply Chain "
    "dataset (a simulated dataset). The target is Late_delivery_risk (1 = late).\n\n"
    "You can act on the app with four tools: set_shipment_inputs (change inputs "
    "and re-run), get_current_prediction (read what is on screen), query_dataset "
    "(real stats from the data), and switch_model (compare models).\n\n"
    "Important honesty constraints:\n"
    "- The signal concentrates almost entirely in Shipping Mode (faster modes "
    "have tight, routinely-missed SLAs). Late rates are roughly flat (~55%) "
    "across market, region, segment, and department.\n"
    "- A one-line per-mode base-rate lookup already scores ROC-AUC ~0.73; the ML "
    "models add only ~+0.01 ROC-AUC. Do NOT overstate the model's skill or imply "
    "non-mode inputs strongly change risk. Always ground claims with query_dataset "
    "or get_current_prediction rather than guessing.\n\n"
    "When asked to compare scenarios, use set_shipment_inputs for each and read "
    "the resulting prediction. Keep summaries concise and plain-language for a "
    "non-technical reader; report P(late), the risk tier, and how it compares to "
    "the shipping mode's base rate."
)
