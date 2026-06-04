# Streamlit AI Assistant — Design Spec

**Date:** 2026-06-04
**Project:** DAP391m — Late-Delivery Risk Predictor (Group 8)
**Feature:** Cloud LLM assistant embedded in `app/app.py`, powered by OpenRouter,
with tools to change inputs / parameters, query the dataset, switch the active
model, and summarize the current prediction.

## Goal

Add an AI assistant to the Streamlit app that users can converse with and that can
*act* on the app: change shipment inputs and re-run the prediction, look up real
dataset statistics, swap the active model, and explain/summarize the current
result in plain language. A one-click "Summarize this prediction" button provides
the same narration without a chat turn.

## Decisions (from brainstorming)

- **Interaction:** both a chat panel *and* a one-click summarize button.
- **AI tools:** all four — change sidebar inputs, read current prediction, query
  the dataset, switch the active model.
- **API key:** sidebar password text input (per session, never persisted).
- **Model:** default `deepseek/deepseek-v4-flash`, editable in a sidebar field
  (so a wrong slug can be corrected without a code change).
- **Structure:** new `app/ai_assistant.py` module (approach A).
- **HTTP:** `requests` directly against OpenRouter's OpenAI-compatible
  `/chat/completions` endpoint — no new SDK dependency (`requests==2.34.1` is
  already installed).

## Architecture

### New file: `app/ai_assistant.py`

Self-contained, no hard Streamlit dependency (operates on plain data + injected
tool callables, so it is unit-testable without a running app).

- **`OpenRouterClient`** — thin wrapper over `requests.post` to
  `https://openrouter.ai/api/v1/chat/completions`. Constructed with `api_key` +
  `model`. `chat(messages, tools)` sends the payload (15s timeout) and returns the
  parsed first choice. Sets `HTTP-Referer` / `X-Title` headers to the project
  name (OpenRouter convention). Raises a typed error on non-200 / timeout with the
  upstream message preserved.
- **`TOOL_SCHEMAS`** — the four function schemas in OpenAI tool format.
- **`SYSTEM_PROMPT`** — explains the app, the late-delivery domain, the honest
  base-rate caveat (ML adds only ~+0.01 ROC over a per-mode lookup, so the
  assistant must not overclaim model skill), and how/when to use each tool.
- **`run_agent(client, messages, tools_impl, max_iters=5)`** — the agent loop:
  1. Call `client.chat(messages, TOOL_SCHEMAS)`.
  2. If the response has `tool_calls`, dispatch each to `tools_impl[name](**args)`,
     append `role:"tool"` results, and loop.
  3. If no tool calls, return the final assistant text.
  4. Cap at `max_iters`; on cap, return the latest text plus a truncation note.
  Returns `(final_text, actions)` where `actions` is a list of human-readable
  strings describing tool effects (for the "Actions taken" display).

### `app/app.py` changes

- **State-backed sidebar widgets.** Each exposed input gets a `key=` so its value
  lives in `st.session_state` (e.g. `key="in_Shipping Mode"`). The
  `set_shipment_inputs` tool writes these keys.
- **Persisted prediction.** The prediction block writes its result
  (`p_late`, `tier`, `mode_rate`, chosen mode, current inputs, feature row as
  dict) into `st.session_state["last_prediction"]`.
- **Active model selection.** `resolve_model` keys off
  `st.session_state["active_model_file"]` (default `primary_model.pkl`) so
  `switch_model` can change it; cached model is cleared on switch.
- **Pending-mutation flag.** Tools that need a rerun to take effect
  (`set_shipment_inputs`, `switch_model`) set
  `st.session_state["rerun_pending"]=True`; after the agent turn completes and the
  chat renders, app.py calls `st.rerun()` once if the flag is set.
- **New "AI assistant" section** (below Prediction):
  - Sidebar: password field for the OpenRouter key + text field for the model
    (default `deepseek/deepseek-v4-flash`).
  - Chat panel: `st.chat_input` + `st.chat_message`, history in
    `st.session_state["chat_history"]`.
  - **"Summarize this prediction"** button — sends a fixed user message
    ("Summarize the current prediction for a non-technical reader") through the
    same `run_agent` path.

## Tools

Tool *implementations* live in `app.py` (they close over the loaded model and
`clean_df`); *schemas* live in `ai_assistant.py`. Implementations are injected
into `run_agent` as a `{name: callable}` dict.

1. **`set_shipment_inputs`** — args: any subset of `{shipping_mode, market,
   order_region, customer_segment, department, category, payment_type,
   item_quantity, sales, discount_rate}`. Validates categoricals against the live
   `clean_data.csv` option lists; unknown values are rejected with a message
   listing valid options (returned to the model, not raised). Writes valid values
   to the `in_*` session_state keys, sets `rerun_pending`. Returns the dict of
   applied changes.
2. **`get_current_prediction`** — no args. Returns `P(late)`, risk tier, chosen
   shipping mode, that mode's base rate, delta vs base, and current inputs from
   `last_prediction`. If none exists yet, returns a note to run a prediction first.
3. **`query_dataset`** — args: `metric` ∈ {`late_rate_by`, `count_by`,
   `overall_late_rate`}, optional `group_col` validated against real columns.
   Computes the stat live from `clean_df`; returns a compact JSON table.
4. **`switch_model`** — args: `model_file` restricted to existing
   `MODEL_CANDIDATES` filenames. Updates `active_model_file`, sets `rerun_pending`.
   Returns the new model label.

## Data flow

```
user types in chat  ──►  run_agent(client, history, tools_impl)
                              │
                              ├─ model emits tool_calls ──► dispatch to app.py impls
                              │     • set_shipment_inputs → writes in_* keys, rerun_pending
                              │     • get_current_prediction → reads last_prediction
                              │     • query_dataset → computes from clean_df
                              │     • switch_model → sets active_model_file, rerun_pending
                              │     └─ tool results appended, loop (≤5)
                              └─ final text ──► appended to chat_history, "Actions taken" shown
                              │
            if rerun_pending ──► st.rerun() ──► sidebar + prediction reflect changes
```

## Error handling

- **No API key:** AI section shows an inline note; rest of app works unchanged.
- **Network/HTTP/timeout (15s):** caught, shown as `st.error` in the chat;
  conversation state preserved for retry.
- **Bad model slug:** OpenRouter's error surfaced verbatim with a hint to check
  the model field (makes a wrong `deepseek/deepseek-v4-flash` slug diagnosable).
- **Tool errors** (invalid category, no prediction yet): returned to the model as
  the tool result so it self-corrects and explains, never crashes.
- **Iteration cap:** return latest text + truncation note.

## Security

- Key read only from the sidebar password field, held in `session_state` for the
  session, never written to disk or logged.
- Add `.streamlit/secrets.toml` to `.gitignore` defensively (not used, but
  prevents accidental key commits if a student adds one later).
- `HTTP-Referer` / `X-Title` headers carry only the project name.

## Testing (TDD)

New tests under `app/tests/` (no existing test dir; create it).

- **`run_agent`** with a stubbed client (no network): single tool call,
  multi-tool chain, iteration cap, tool that errors.
- **Each tool function** against a small in-memory DataFrame: valid set,
  unknown-category rejection, `query_dataset` math, `switch_model` guard.
- **`OpenRouterClient.chat`** with a mocked `requests.post`: payload shape,
  non-200 → typed error, timeout handling.
- **Manual smoke test:** run Streamlit, paste a key, try "compare Same Day vs
  Standard Class" and "why is this late?"; confirm inputs change and the summary
  reads correctly.

## Out of scope (YAGNI)

- Streaming responses.
- Multi-shipment / batch reasoning.
- Persisting chat history across sessions.
- Env-var / secrets-file key loading (sidebar input only, per decision).
- Tool access to retrain models or write files.

## Dependencies

- `requests` (already installed). No new packages.
