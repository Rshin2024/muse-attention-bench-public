"""Native inference transports for the trusted benchmark controller.

No network, authentication, or model invocation occurs on import. ``generate``
performs one billable request at most: it never retries or substitutes a model.
Credentials and opaque reasoning history stay on the trusted controller, never
inside actor sandboxes or public reports. Callers must enforce a process-level
trial deadline as well as the bounded socket timeouts used here.

Official request/response references (reviewed 7 November 2026):
OpenAI: https://developers.openai.com/api/docs/guides/migrate-to-responses
DeepSeek: https://api-docs.deepseek.com/api/create-response/
Claude: https://platform.claude.com/docs/en/api/messages/create
Gemini: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/start/get-started-with-gemini-4
GLM: https://docs.z.ai/api-reference/llm/chat-completion
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import time
from typing import Callable, Mapping
from urllib import error, request
from urllib.parse import quote, urlparse

MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_HOSTS = {"openai": {"api.openai.com"}, "anthropic": {"api.anthropic.com"},
          "deepseek": {"api.deepseek.com"}, "gemini": {"aiplatform.googleapis.com"},
          "glm": {"api.z.ai"}}
_PROTOCOLS = {"openai": "openai_responses", "deepseek": "openai_responses",
              "anthropic": "anthropic_messages", "gemini": "google_generate_content",
              "glm": "openai_chat_completions"}


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A native endpoint must not forward its Authorization header elsewhere.
        return None


class TransportError(RuntimeError):
    """A sanitized failure; never contains provider error bodies or credentials."""
    def __init__(self, kind: str, *, status_code=None, request_id=None,
                 retryable=False, cost_unknown=False, partial_result=None):
        self.kind, self.status_code, self.request_id = kind, status_code, request_id
        self.retryable, self.cost_unknown = bool(retryable), bool(cost_unknown)
        self.partial_result = partial_result
        super().__init__(f"Model transport failed: {kind}" + (f" (HTTP {status_code})" if status_code else ""))

    def as_dict(self):
        return {"type": type(self).__name__, "kind": self.kind,
                "status_code": self.status_code, "request_id": self.request_id,
                "retryable": self.retryable, "automatically_retried": False,
                "cost_unknown": self.cost_unknown,
                "partial_result": self.partial_result}


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _integer(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _sum_known(*values):
    return sum(values) if all(v is not None for v in values) else None


def normalize_usage(provider: str, response: dict) -> dict:
    """Preserve provider counters, with reasoning counted once in output cost."""
    raw = response.get("usageMetadata" if provider == "gemini" else "usage")
    raw = copy.deepcopy(raw) if isinstance(raw, dict) else {}
    cached = writes = reasoning = None
    if provider in {"openai", "deepseek"}:
        incoming, outgoing = _integer(raw.get("input_tokens")), _integer(raw.get("output_tokens"))
        details = raw.get("input_tokens_details") or {}
        cached = _integer(details.get("cached_tokens"))
        reasoning = _integer((raw.get("output_tokens_details") or {}).get("reasoning_tokens"))
        # GPT-5.6+ separately bills writes; an omitted counter is not zero cost.
        writes = _integer(details.get("cache_write_tokens")) if provider == 'openai' else _integer(details.get('cache_write_tokens',0))
        uncached = incoming - cached - writes if incoming is not None and cached is not None and writes is not None else None
    elif provider == "anthropic":
        uncached = _integer(raw.get("input_tokens"))
        cached = _integer(raw.get("cache_read_input_tokens", 0))
        writes = _integer(raw.get("cache_creation_input_tokens", 0))
        incoming = _sum_known(uncached, cached, writes)
        outgoing = _integer(raw.get("output_tokens"))
        reasoning = _integer((raw.get("output_tokens_details") or {}).get("thinking_tokens"))
    elif provider == "gemini":
        incoming = _integer(raw.get("promptTokenCount"))
        cached = _integer(raw.get("cachedContentTokenCount", 0))
        reasoning = _integer(raw.get("thoughtsTokenCount", 0))
        # Google reports candidates separately from thinking; both are generated output.
        outgoing = _sum_known(_integer(raw.get("candidatesTokenCount")), reasoning)
        writes = 0
        uncached = incoming - cached if incoming is not None and cached is not None else None
    elif provider == "glm":
        incoming, outgoing = _integer(raw.get("prompt_tokens")), _integer(raw.get("completion_tokens"))
        cached = _integer((raw.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        reasoning = _integer((raw.get("completion_tokens_details") or {}).get("reasoning_tokens"))
        writes = 0
        uncached = incoming - cached if incoming is not None and cached is not None else None
    else:
        raise ValueError("Unknown provider")
    if uncached is not None and uncached < 0:
        raise TransportError("inconsistent_usage", cost_unknown=True)
    total = _integer(raw.get("totalTokenCount" if provider == "gemini" else "total_tokens"))
    if total is None:
        total = _sum_known(incoming, outgoing)
    return {"input_tokens": incoming, "output_tokens": outgoing,
            "uncached_input_tokens": uncached, "cached_input_tokens": cached,
            "cache_write_input_tokens": writes, "reasoning_tokens": reasoning,
            "total_tokens": total, "raw": raw}


def estimate_cost(usage: dict, pricing: dict | None) -> tuple[float | None, str]:
    """Use caller-pinned USD/MTok rates. Missing rates/counters are not zero cost."""
    if pricing is None:
        return None, "No pinned pricing supplied"
    categories = [("uncached_input_tokens", "input_per_million"),
                  ("cached_input_tokens", "cached_input_per_million"),
                  ("cache_write_input_tokens", "cache_write_input_per_million"),
                  ("output_tokens", "output_per_million")]
    amount = 0.0
    for counter, rate_name in categories:
        count = usage.get(counter)
        if count is None:
            return None, f"Missing usage counter: {counter}"
        if count == 0:
            continue
        rate = pricing.get(rate_name)
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate < 0:
            return None, f"Missing/invalid pinned rate: {rate_name}"
        amount += count * float(rate) / 1_000_000
    return amount, "Estimate from reported usage and caller-pinned rates; not an invoice"


def _messages(messages: list[dict]) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty canonical text conversation")
    result = []
    for item in messages:
        if not isinstance(item, dict) or item.get("role") not in {"system", "developer", "user", "assistant"} or not isinstance(item.get("content"), str):
            raise ValueError("Canonical messages require role and string content")
        result.append({"role": item["role"], "content": item["content"]})
    return result


def _validate_entry(entry: dict, model: str):
    provider = entry.get("id")
    if provider not in _HOSTS or entry.get("protocol") != _PROTOCOLS[provider]:
        raise ValueError("Unknown or inconsistent provider transport")
    if provider == 'gemini' and entry.get('authentication_selection','gcloud_cli_oauth') != 'gcloud_cli_oauth':
        raise ValueError('This transport implements the explicitly selected trusted-host gcloud OAuth flow only')
    allowed = entry.get("configured_models", [entry.get("model")])
    if model not in allowed or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", model):
        raise ValueError("Model must be an exact configured model ID")
    url = urlparse(entry.get("base_url", ""))
    if url.scheme != "https" or url.hostname not in _HOSTS[provider] or url.username or url.password or url.query or url.fragment:
        raise ValueError("Native provider HTTPS endpoint required; credentials cannot be sent to a proxy")
    return provider


def _native_append(protocol: str, messages: list[dict]):
    if protocol == "google_generate_content":
        return [{"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [{"text": m["content"]}]} for m in messages]
    if protocol == "anthropic_messages":
        return [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
    return copy.deepcopy(messages)


def _history(entry: dict, model: str, messages: list[dict], native_state: dict | None):
    protocol = entry["protocol"]
    system = "\n\n".join(m["content"] for m in messages if m["role"] in {"system", "developer"})
    if native_state is not None:
        if not isinstance(native_state,dict):
            raise ValueError('Native history state must be a trusted controller object')
        count = native_state.get("canonical_count")
        if (native_state.get("provider") != entry["id"] or native_state.get("model") != model or
                isinstance(count,bool) or not isinstance(count, int) or count > len(messages) or count < 1 or
                not isinstance(native_state.get('history'),list) or
                _hash(native_state.get('history')) != native_state.get('native_history_sha256') or
                _hash(messages[:count]) != native_state.get("canonical_prefix_sha256")):
            raise ValueError("Native history does not match unchanged canonical conversation")
        added = messages[count:]
        if any(m["role"] in {"system", "developer"} for m in added):
            raise ValueError("Do not modify system instructions during a frozen trial")
        history = copy.deepcopy(native_state["history"])
        history.extend(_native_append(protocol, added))
        return history, system
    if protocol in {"anthropic_messages", "google_generate_content"}:
        first_non_system = next((i for i,m in enumerate(messages) if m["role"] not in {"system", "developer"}), len(messages))
        if any(m["role"] in {"system", "developer"} for m in messages[first_non_system:]):
            raise ValueError("System messages must be an unchanged initial prefix")
        return _native_append(protocol, messages[first_non_system:]), system
    native_messages = copy.deepcopy(messages)
    if entry['id'] in {'deepseek','glm'}:
        # DeepSeek documents developer as user; GLM documents system, not developer.
        # Preserve the canonical high-priority instruction intent on both routes.
        for message in native_messages:
            if message['role']=='developer':message['role']='system'
    return _native_append(protocol, native_messages), system


def build_request(entry: dict, model: str, messages: list[dict], *,
                  max_output_tokens: int = 32768, native_state: dict | None = None,
                  json_mode: bool = True, action_schema: dict | None = None,
                  environment: Mapping[str, str] | None = None) -> dict:
    """Build an unauthenticated request plan; does not invoke a provider."""
    provider = _validate_entry(entry, model)
    suffix=entry.get('inference_path_template' if provider=='gemini' else 'inference_path','')
    if not isinstance(suffix,str) or not suffix.startswith('/') or suffix.startswith('//'):
        raise ValueError('Native inference path must be an absolute path suffix')
    messages = _messages(messages)
    if json_mode and action_schema is None and provider in {'openai','deepseek'} and not any(re.search(r'\bjson\b',m['content'],re.I) for m in messages):
        raise ValueError('JSON mode requires an explicit JSON instruction in the shared prompt')
    if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or not 1 <= max_output_tokens <= 32768:
        raise ValueError("Common per-turn output-token limit must be 1..32768")
    env = os.environ if environment is None else environment
    history, system = _history(entry, model, messages, native_state)
    body = {"model": model, **copy.deepcopy(entry.get("reasoning_parameters", {}))}
    if entry["protocol"] == "openai_responses":
        body.update(input=history, max_output_tokens=max_output_tokens, stream=False,
                    store=False, tools=[], tool_choice="none")
        if provider == "openai":
            body["include"] = ["reasoning.encrypted_content"]
            body['service_tier'] = 'default'
        if json_mode:
            output_format = ({"type": "json_schema", "name": "benchmark_action",
                              "schema": action_schema} if action_schema else {"type": "json_object"})
            if action_schema and provider == "openai":
                output_format['strict'] = True
            body["text"] = {"format": output_format}
        url = entry["base_url"].rstrip("/") + entry["inference_path"]
    elif entry["protocol"] == "anthropic_messages":
        body.update(messages=history, max_tokens=max_output_tokens, stream=False,
                    thinking={"type": "adaptive"})
        if system:
            body["system"] = system
        if json_mode and action_schema:
            body.setdefault("output_config", {})["format"] = {"type": "json_schema", "schema": action_schema}
        url = entry["base_url"].rstrip("/") + entry["inference_path"]
    elif entry["protocol"] == "google_generate_content":
        project = env.get(entry.get("project_env", "GOOGLE_CLOUD_PROJECT"))
        location = env.get(entry.get("location_env", "GOOGLE_CLOUD_LOCATION"), entry.get("location_default", "global"))
        if not project or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._-]*", project) or not re.fullmatch(r"[A-Za-z0-9-]+", location):
            raise ValueError("Google project/location must be configured on the trusted controller")
        url = entry["base_url"].rstrip("/") + entry["inference_path_template"].format(project=quote(project,safe=""), location=quote(location,safe=""), model=quote(model,safe=""))
        body.pop("model")
        body.update(contents=history)
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        body.setdefault("generationConfig", {})["maxOutputTokens"] = max_output_tokens
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
            if action_schema:
                body["generationConfig"]["responseJsonSchema"] = action_schema
    else:
        body.update(messages=history, max_tokens=max_output_tokens, stream=False)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        url = entry["base_url"].rstrip("/") + entry["inference_path"]
    target=urlparse(url)
    if target.scheme!='https' or target.hostname not in _HOSTS[provider] or target.username or target.password or target.port not in (None,443) or target.query or target.fragment:
        raise ValueError('Final inference URL must remain on the configured native provider')
    return {"provider": provider, "protocol": entry["protocol"], "url": url,
            "body": body, "canonical_messages": messages, "history": history}


def _output(provider: str, response: dict):
    unexpected_tools = False
    if provider in {"openai", "deepseek"}:
        items = response.get("output", [])
        if not isinstance(items, list):
            raise TransportError("malformed_response", cost_unknown=True)
        texts = []
        for item in items:
            if item.get("type") == "message":
                texts.extend(p.get("text", "") for p in item.get("content", []) if p.get("type") == "output_text")
            elif item.get("type") != "reasoning":
                unexpected_tools = True
        status = response.get("status")
        finish = (response.get("incomplete_details") or {}).get("reason") or status
        return "".join(texts), copy.deepcopy(items), finish, unexpected_tools
    if provider == "anthropic":
        content = response.get("content", [])
        if not isinstance(content, list):
            raise TransportError("malformed_response", cost_unknown=True)
        text = "".join(p.get("text", "") for p in content if p.get("type") == "text")
        unexpected_tools = any(p.get("type") not in {"text", "thinking", "redacted_thinking"} for p in content)
        return text, [{"role": "assistant", "content": copy.deepcopy(content)}], response.get("stop_reason"), unexpected_tools
    if provider == "gemini":
        candidates = response.get("candidates", [])
        if not isinstance(candidates, list) or len(candidates) > 1:
            raise TransportError("malformed_response", cost_unknown=True)
        candidate = candidates[0] if candidates else {}
        content = candidate.get("content", {"role": "model", "parts": []})
        parts = content.get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought", False) and "text" in p)
        unexpected_tools = any("functionCall" in p or "executableCode" in p for p in parts)
        return text, [copy.deepcopy(content)] if candidates else [], candidate.get("finishReason") or "blocked", unexpected_tools
    choices = response.get("choices", [])
    if not isinstance(choices, list) or len(choices) != 1:
        raise TransportError("malformed_response", cost_unknown=True)
    choice = choices[0]; message = choice.get("message", {})
    text = message.get("content") or ""
    if not isinstance(text, str):
        raise TransportError("malformed_response", cost_unknown=True)
    return text, [copy.deepcopy(message)], choice.get("finish_reason"), bool(message.get("tool_calls"))


def normalize_response(entry: dict, model: str, plan: dict, response: dict, *,
                       response_headers=None, wall_seconds: float = 0.0,
                       pricing: dict | None = None) -> dict:
    """Native state is private memory, not a serializable public transcript."""
    provider = entry["id"]
    usage = normalize_usage(provider, response)
    cost, cost_note = estimate_cost(usage, pricing)
    headers = {str(k).lower():v for k,v in (response_headers or {}).items()}
    response_id = response.get("id") or response.get("responseId")
    request_id = headers.get("x-request-id") or headers.get("request-id") or response.get('request_id') or response_id
    reported = response.get("model") or response.get("modelVersion")
    try:
        text, returned_history, finish, unexpected_tools = _output(provider, response)
    except (TransportError, AttributeError, TypeError, ValueError):
        raise TransportError('malformed_response',request_id=request_id,cost_unknown=cost is None,
            partial_result={'provider':provider,'model_requested':model,'model_reported':reported,
                            'request_id':request_id,'response_id':response_id,'usage':usage,
                            'estimated_cost_usd':cost,'cost_note':cost_note,'wall_seconds':wall_seconds}) from None
    result = {"output_text": text, "usage": usage, "request_id": request_id, "response_id": response_id,
              "model_requested": model, "model_reported": reported,
              "finish_reason": finish, "wall_seconds": wall_seconds,
              "estimated_cost_usd": cost, "cost_note": cost_note,
              "provider": provider, "automatically_retried": False,
              "service_tier": response.get('service_tier') or usage['raw'].get('service_tier'),
              "speed": response.get('speed') or usage['raw'].get('speed'),
              "native_tool_calls": unexpected_tools,
              "continuation_available": bool(response.get("continuationToken")),
              "max_output_tokens_requested": plan["body"].get("max_output_tokens", plan["body"].get("max_tokens", plan["body"].get("generationConfig",{}).get("maxOutputTokens")))}
    if unexpected_tools or response.get("status") == "failed":
        safe = {k:v for k,v in result.items() if k != "output_text"}
        raise TransportError("unexpected_native_tool" if unexpected_tools else "provider_response_failed", partial_result=safe,
                             cost_unknown=cost is None, request_id=request_id)
    canonical = plan["canonical_messages"] + [{"role": "assistant", "content": text}]
    next_history=plan["history"] + returned_history
    result["native_state"] = {"provider": provider, "model": model,
        "canonical_count": len(canonical), "canonical_prefix_sha256": _hash(canonical),
        "history": next_history, 'native_history_sha256':_hash(next_history)}
    return result


def _google_token(remaining: float, runner) -> str:
    # Fixed trusted command, not a command taken from model output/configuration.
    try:
        completed = runner(["gcloud", "auth", "print-access-token"], capture_output=True,
                           text=True, check=True, timeout=min(30.0, remaining))
        value = completed.stdout.strip()
    except Exception:
        raise TransportError("google_authentication_failed") from None
    if not isinstance(value,str) or not value or any(c.isspace() for c in value):
        raise TransportError("google_authentication_failed")
    return value


def generate(provider_config: dict, model: str, messages: list[dict], remaining_seconds: float,
             credential_loader: Callable[[dict], str] | None = None, *,
             max_output_tokens: int = 32768, native_state: dict | None = None,
             json_mode: bool = True, action_schema: dict | None = None,
             environment: Mapping[str, str] | None = None, pricing: dict | None = None,
             timeout_cap: float = 600.0, opener=None,
             token_command_runner=None, clock=None) -> dict:
    """Perform exactly one native generation with bounded timeouts and safe errors.

    Keep ``native_state`` in private controller memory. On the next call append
    exactly ``output_text`` as an assistant message to the canonical history,
    then the new user observation, and pass the state back. History edits fail
    closed. Token-counting/context admission belongs to the common broker;
    characters are not silently treated as exact tokens here.
    """
    if isinstance(remaining_seconds, bool) or not isinstance(remaining_seconds,(int,float)) or not math.isfinite(remaining_seconds) or remaining_seconds <= 0:
        raise TransportError("trial_deadline_exhausted")
    if isinstance(timeout_cap, bool) or not isinstance(timeout_cap,(int,float)) or not math.isfinite(timeout_cap) or timeout_cap <= 0:
        raise ValueError("timeout_cap must be finite and positive")
    now = time.monotonic if clock is None else clock
    started = now(); deadline = started + remaining_seconds
    plan = build_request(provider_config, model, messages, max_output_tokens=max_output_tokens,
                         native_state=native_state, json_mode=json_mode, action_schema=action_schema,
                         environment=environment)
    auth_remaining=deadline-now()
    if auth_remaining <= 0:
        raise TransportError('trial_deadline_exhausted')
    if provider_config["id"] == "gemini":
        token = _google_token(auth_remaining, token_command_runner or subprocess.run)
    else:
        if credential_loader is None:
            raise TransportError("credential_loader_required")
        try:
            token = credential_loader(provider_config)
        except Exception:
            raise TransportError("credential_unavailable") from None
        if not isinstance(token, str) or not token.strip() or any(c.isspace() for c in token):
            raise TransportError("credential_unavailable")
    timeout = min(float(timeout_cap), deadline-now())
    if timeout <= 0:
        raise TransportError("trial_deadline_exhausted")
    headers = {"Content-Type": "application/json", "Accept": "application/json",
               "Authorization": "Bearer " + token}
    if provider_config["id"] == "anthropic":
        headers["anthropic-version"] = provider_config.get("api_version", "2026-10-21")
    body = json.dumps(plan["body"], ensure_ascii=False, allow_nan=False).encode()
    req = request.Request(plan["url"], data=body, headers=headers, method="POST")
    call = request.build_opener(_NoRedirect()).open if opener is None else opener
    try:
        with call(req, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            response_headers = dict(response.headers.items())
    except error.HTTPError as exc:
        # The body can echo prompt/auth data. Do not read or include it.
        identifier = exc.headers.get("x-request-id") or exc.headers.get("request-id") if exc.headers else None
        raise TransportError("http_error", status_code=exc.code, request_id=identifier,
                             retryable=exc.code in {408,409,429,500,502,503,504},
                             cost_unknown=True) from None
    except (TimeoutError, socket.timeout):
        raise TransportError("request_timeout", retryable=False, cost_unknown=True) from None
    except error.URLError:
        raise TransportError("network_error", retryable=False, cost_unknown=True) from None
    except Exception:
        raise TransportError("request_failed", retryable=False, cost_unknown=True) from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise TransportError("response_size_limit", cost_unknown=True)
    try:
        def reject_nonfinite(value):raise ValueError('Non-finite JSON number')
        parsed = json.loads(raw,parse_constant=reject_nonfinite)
        if not isinstance(parsed, dict):
            raise ValueError()
    except (ValueError, UnicodeError):
        raise TransportError("malformed_response", cost_unknown=True) from None
    result = normalize_response(provider_config, model, plan, parsed,
                                response_headers=response_headers,
                                wall_seconds=now()-started, pricing=pricing)
    result["request_timeout_seconds"] = timeout
    result["received_after_trial_deadline"] = now() > deadline
    return result
