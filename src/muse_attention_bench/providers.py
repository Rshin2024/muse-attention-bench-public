"""Provider configuration and redacted local preflight, with no network/inference.

This module does not read credential file contents or invoke model providers.
An external trusted broker must implement and validate the native transports.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/providers.example.json"
PROTOCOLS = {"openai_responses", "anthropic_messages", "google_generate_content", "openai_chat_completions"}
PROVIDERS = {"openai", "anthropic", "deepseek", "gemini", "glm"}
SECRET_FIELDS = {"api_key", "access_token", "token", "password", "client_secret", "authorization"}


def _reject_inline_secrets(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in SECRET_FIELDS:
                raise ValueError("Credential values must stay outside provider configuration")
            _reject_inline_secrets(child)
    elif isinstance(value, list):
        for child in value:
            _reject_inline_secrets(child)


def validate_configuration(config: dict) -> dict:
    """Validate a configuration-only plan, without claiming provider availability."""
    _reject_inline_secrets(config)
    if config.get("schema_version") != 1 or config.get("inference_enabled") is not False:
        raise ValueError("This preflight configuration must explicitly disable inference")
    entries = config.get("providers", [])
    ids = [entry.get("id") for entry in entries]
    if set(ids) != PROVIDERS or len(ids) != len(PROVIDERS):
        raise ValueError("Expected exactly one entry for each of the five providers")
    for entry in entries:
        for field in ('api_key_env', 'api_key_file_env', 'credential_file_env', 'project_env', 'location_env'):
            name=entry.get(field)
            if name is not None and (not isinstance(name,str) or not re.fullmatch(r'[A-Z_][A-Z0-9_]*',name)):
                raise ValueError('Credential/config references must be environment-variable names, never values')
        model = entry.get("model")
        if not isinstance(model, str) or not model or model.lower() == "latest" or any(c.isspace() for c in model):
            raise ValueError("Use an exact documented model ID")
        if entry.get("protocol") not in PROTOCOLS:
            raise ValueError("Unknown provider transport")
        url = urlparse(entry.get("base_url", ""))
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query:
            raise ValueError("Provider base URL must be HTTPS without embedded credentials or query")
        if entry.get("inference_validated") is not False:
            raise ValueError("A configuration-only plan cannot assert inference validation")
    return config


def load_configuration(path: Path | str = DEFAULT_CONFIG) -> dict:
    return validate_configuration(json.loads(Path(path).read_text()))


def local_context_environment(path: Path, config: dict, environment=None) -> dict:
    """Load credential-file references and project/location only, never API keys."""
    allowed=set()
    for entry in config['providers']:
        allowed.update(entry.get(field) for field in
                       ('api_key_file_env','credential_file_env','project_env','location_env') if entry.get(field))
    values=json.loads(path.read_text())
    if not isinstance(values,dict) or any(k not in allowed or not isinstance(v,str) for k,v in values.items()):
        raise ValueError('Local context accepts only configured credential-file/project/location references')
    return {**(os.environ if environment is None else environment),**values}


def local_access_status(config: dict, environment: Mapping[str, str] | None = None) -> dict:
    """Return booleans and declared env names only; never keys or file paths."""
    config = validate_configuration(config)
    environment = os.environ if environment is None else environment
    providers = []
    for entry in config["providers"]:
        row = {"provider": entry["id"], "model": entry["model"],
               "protocol": entry["protocol"], "inference_called": False,
               "account_entitlement_verified_by_this_command": False}
        key_env, file_env = entry.get("api_key_env"), entry.get("api_key_file_env")
        row["api_key_env_name"] = key_env
        row["api_key_env_present"] = bool(key_env and environment.get(key_env))
        row["api_key_file_env_name"] = file_env
        file_hint = environment.get(file_env, "") if file_env else ""
        row["credential_file_reference_present"] = bool(file_hint and Path(file_hint).is_file())
        if entry["id"] == "gemini":
            row["project_env_present"] = bool(environment.get(entry["project_env"]))
            row["location_env_present"] = bool(environment.get(entry["location_env"]))
            adc_path = environment.get(entry["credential_file_env"], "")
            row["explicit_adc_file_reference_present"] = bool(adc_path and Path(adc_path).is_file())
            row["default_adc_status"] = "not inspected; absence of an env reference does not prove ADC is unavailable"
        if entry.get("endpoint_selection_required"):
            row["endpoint_selection_required"] = True
        providers.append(row)
    return {"schema_version": 1, "scope": "redacted local configuration preflight only",
            "network_called": False, "inference_called": False, "providers": providers}


def catalog_request_plan(entry: dict) -> dict:
    """Describe a read-only request for a trusted host; never populate its auth."""
    path = entry.get("catalog_path")
    if not path:
        return {"provider": entry["id"], "available": False,
                "reason": "Use the provider-specific authenticated account/project discovery flow; no endpoint is guessed"}
    plan = {"provider": entry["id"], "available": True, "method": "GET",
            "url": entry["base_url"].rstrip("/") + path,
            "credential_env_name": entry.get("api_key_env"),
            "credential_file_env_name": entry.get("api_key_file_env"),
            "executes_request": False}
    if entry.get("api_version"):
        plan["version_header"] = {"anthropic-version": entry["api_version"]}
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Redacted provider configuration preflight; no network or model calls")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--catalog-plans", action="store_true")
    parser.add_argument('--local-context',type=Path,help='Private JSON with credential-file/project/location references only')
    args = parser.parse_args(argv)
    try:
        config = load_configuration(args.config)
        environment=local_context_environment(args.local_context,config) if args.local_context else None
        result = ([catalog_request_plan(p) for p in config["providers"]]
                  if args.catalog_plans else local_access_status(config,environment))
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
