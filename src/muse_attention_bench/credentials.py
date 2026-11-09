"""Hidden local API-key entry and reference-only credential metadata.

No provider/network call is made. Keys stay outside the benchmark checkout and
must only be resolved by the trusted controller, never an actor container.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Callable
import warnings

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_ROOT = Path.home() / ".config" / "muse-attention-bench"
DEFAULT_REPO_CONTEXT = ROOT / ".local" / "provider-context.json"
PROVIDER_KEYS = {
    "openai": ("OPENAI_API_KEY", "OPENAI_API_KEY_FILE", "openai.key", "OpenAI"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_FILE", "anthropic.key", "Anthropic"),
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_FILE", "deepseek.key", "DeepSeek"),
    "glm": ("ZAI_API_KEY", "ZAI_API_KEY_FILE", "glm.key", "Z.ai GLM"),
}
CONTEXT_FIELDS = {entry[1] for entry in PROVIDER_KEYS.values()} | {
    "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"
}


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("Credential paths must not contain symbolic links")


def _root(config_root: Path | None) -> Path:
    root = _absolute(config_root or DEFAULT_CONFIG_ROOT)
    _reject_symlink_components(root)
    if root.is_relative_to(ROOT):
        raise ValueError("Store API keys outside the benchmark checkout")
    return root


def _private_directory(path: Path) -> None:
    _reject_symlink_components(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError("Credential storage must be a regular directory")
    path.chmod(0o700)


def _regular_file(path: Path) -> bool:
    _reject_symlink_components(path)
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def _read_context(path: Path) -> dict[str, str]:
    if not _regular_file(path):
        if path.exists():
            raise ValueError("Credential context must be a regular JSON file")
        return {}
    if path.stat().st_size > 65536:
        raise ValueError("Credential context exceeds the size limit")
    try:
        value = json.loads(path.read_text())
    except (UnicodeError, ValueError):
        raise ValueError("Credential context is not valid JSON") from None
    if not isinstance(value, dict) or any(
        key not in CONTEXT_FIELDS or not isinstance(item, str) or "\n" in item or "\r" in item
        for key, item in value.items()
    ):
        raise ValueError("Credential context accepts only file/project/location references")
    return value


def _write_private(path: Path, content: bytes, *, overwrite: bool = False) -> None:
    _reject_symlink_components(path)
    if path.exists() and not _regular_file(path):
        raise ValueError("Credential destination must be a regular file")
    if not overwrite:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            # fdopen owns the descriptor after construction.
            if path.exists():
                path.unlink()
            raise
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".credential-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _reject_symlink_components(path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _save_context(path: Path, context: dict[str, str]) -> None:
    _write_private(path, (json.dumps(context, sort_keys=True, indent=2) + "\n").encode(), overwrite=path.exists())


def _hidden_prompt(label: str) -> str:
    if not sys.stdin.isatty():
        raise ValueError("Run key setup in an interactive local terminal; hidden input is required")
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        try:
            return getpass.getpass(f"{label} API key (hidden; Enter to skip): ")
        except getpass.GetPassWarning:
            raise ValueError("Hidden terminal input is unavailable; no key was collected") from None


def _key_bytes(value: str) -> bytes:
    if not isinstance(value, str) or not value or len(value) > 8192 or not value.isascii() or any(
        ord(character) < 33 or ord(character) > 126 for character in value
    ):
        raise ValueError("API key must be nonempty printable ASCII without whitespace")
    return value.encode() + b"\n"


def credential_status(config_root: Path | None = None, *, environment=None) -> dict:
    """Read references and file metadata only; never key-file contents."""
    root = _root(config_root)
    context_path = root / "credential-context.json"
    context = _read_context(context_path)
    environment = os.environ if environment is None else environment
    rows = []
    for provider, (key_env, file_env, _, _) in PROVIDER_KEYS.items():
        reference = environment.get(file_env) or context.get(file_env)
        exists = private = False
        if reference:
            path = _absolute(Path(reference))
            exists = _regular_file(path)
            private = exists and not (stat.S_IMODE(path.stat().st_mode) & 0o077)
        rows.append({"provider": provider, "api_key_env_name": key_env,
                     "api_key_env_present": bool(environment.get(key_env)),
                     "api_key_file_env_name": file_env, "file_reference_present": bool(reference),
                     "file_exists": exists, "file_permissions_private": private,
                     "inference_entitlement_verified": False})
    rows.append({"provider": "gemini", "authentication": "gcloud_cli_oauth",
                 "api_key_required": False, "inference_entitlement_verified": False})
    return {"schema_version": 1, "config_root": str(root), "context_file": str(context_path),
            "context_exists": context_path.is_file(),
            "context_permissions_private": context_path.is_file() and not (stat.S_IMODE(context_path.stat().st_mode) & 0o077),
            "providers": rows, "network_called": False, "inference_called": False,
            "key_file_contents_read": False}


def setup_credentials(config_root: Path | None = None, providers: list[str] | None = None, *,
                      overwrite: bool = False, repo_context: Path | None = DEFAULT_REPO_CONTEXT,
                      password_reader: Callable[[str], str] | None = None) -> dict:
    """Collect hidden keys into private files, preserving existing keys by default.

    ``password_reader`` is an injection point for offline tests or a trusted UI;
    the CLI always uses the no-echo local terminal reader.
    """
    root = _root(config_root)
    chosen = list(PROVIDER_KEYS) if providers is None else list(providers)
    if len(chosen) != len(set(chosen)) or any(provider not in {*PROVIDER_KEYS, "gemini"} for provider in chosen):
        raise ValueError("Choose each supported provider at most once")
    existing = _read_context(root / "credential-context.json")
    imported = _read_context(_absolute(repo_context)) if repo_context is not None else {}
    context = {**imported, **existing}
    _private_directory(root)
    _private_directory(root / "keys")
    context_path = root / "credential-context.json"
    _save_context(context_path, context)
    reader = password_reader or _hidden_prompt
    actions = []
    for provider in chosen:
        if provider == "gemini":
            actions.append({"provider": provider, "action": "uses_gcloud_oauth_no_api_key"})
            continue
        _, file_env, filename, label = PROVIDER_KEYS[provider]
        previous = context.get(file_env)
        if previous and _regular_file(_absolute(Path(previous))) and not overwrite:
            actions.append({"provider": provider, "action": "kept_existing_reference"})
            continue
        destination = root / "keys" / filename
        if destination.exists() and not overwrite:
            if not _regular_file(destination):
                raise ValueError("Credential destination must be a regular file")
            context[file_env] = str(destination)
            _save_context(context_path, context)
            actions.append({"provider": provider, "action": "kept_existing_key"})
            continue
        value = reader(label)
        if value == "":
            actions.append({"provider": provider, "action": "skipped_empty_input"})
            continue
        _write_private(destination, _key_bytes(value), overwrite=overwrite)
        context[file_env] = str(destination)
        _save_context(context_path, context)
        actions.append({"provider": provider, "action": "stored_private_key"})
    result = credential_status(root)
    result["actions"] = actions
    return result


def link_existing(provider: str, key_file: Path, config_root: Path | None = None, *, overwrite: bool = False) -> dict:
    """Link a regular existing key file without opening or copying its contents."""
    if provider not in PROVIDER_KEYS:
        raise ValueError("This provider does not use a configured API-key file")
    root, path = _root(config_root), _absolute(key_file)
    if not _regular_file(path):
        raise ValueError("Existing credential must be a regular file")
    context_path = root / "credential-context.json"
    context = _read_context(context_path)
    file_env = PROVIDER_KEYS[provider][1]
    if file_env in context and context[file_env] != str(path) and not overwrite:
        raise FileExistsError("An existing reference is configured; use --overwrite to replace it")
    _private_directory(root)
    _private_directory(root / "keys")
    context[file_env] = str(path)
    _save_context(context_path, context)
    return credential_status(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local hidden API-key setup; no network/model calls")
    parser.add_argument("--config-root", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="Enter API keys using no-echo local terminal input")
    setup.add_argument("--provider", choices=[*PROVIDER_KEYS, "gemini"], action="append")
    setup.add_argument("--overwrite", action="store_true")
    commands.add_parser("status", help="Inspect references and permissions without reading key contents")
    link = commands.add_parser("link", help="Reference an existing key file without copying its contents")
    link.add_argument("--provider", choices=list(PROVIDER_KEYS), required=True)
    link.add_argument("--file", type=Path, required=True)
    link.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "setup":
            result = setup_credentials(args.config_root, args.provider, overwrite=args.overwrite)
        elif args.command == "link":
            result = link_existing(args.provider, args.file, args.config_root, overwrite=args.overwrite)
        else:
            result = credential_status(args.config_root)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        # Do not include arbitrary exception text: an OS error may contain an
        # attacker-chosen path or input. Credential values never enter errors.
        print(json.dumps({"error": "Credential setup failed safely", "reason_type": type(exc).__name__}), file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print("Key setup cancelled; previously saved entries were retained.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
