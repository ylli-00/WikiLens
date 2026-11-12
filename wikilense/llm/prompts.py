"""Versioned prompt sets: one system template and one user template per condition.

A prompt set is a folder ``wikilense/llm/prompts/<prompt_id>/`` with ``prompt.json``::

    {"prompt_id": "generic_probe_v0",
     "system": "system.txt",
     "user": {"no_retrieval": "user_no_retrieval.txt", "retrieved": "...", "gold": "..."},
     "schema": "schema.json",
     "label_path": "result.label"}

The **prompt hash** is the SHA-256 of a canonical JSON of everything that is sent or that decides
how the answer is read: the prompt id, the system text, every user template, the schema and the
label path. It is written into every row, the manifest and the resume key, so editing any of
these files makes a resumed run refuse the folder (``runner.check_resume``) and a new run ask
again.

The system prompt is one text for all three conditions (kit REPORT §4 "Prompt layout"): the CLI
caches the system prompt and the schema tool as one shared prefix with a 1-hour TTL (kit D-59;
live 2026-11-11: 1,113 tokens read back from call 2 on), so one prefix serves every condition,
and the wording is not a confound between them. Only the user turn differs; it ends with the
claim, because text after the per-call payload is never cached (kit D-117 §2).

Templates use ``{{field}}`` placeholders. A missing field raises instead of sending ``{{claim}}``
to the model (kit ``cc_runner.render``).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
CONDITIONS: tuple[str, ...] = ("no_retrieval", "retrieved", "gold")
LABELS: tuple[str, ...] = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
_FIELD_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
#: Linux caps one argv string at 128 KiB (MAX_ARG_STRLEN); the system prompt goes on argv.
MAX_SYSTEM_BYTES = 120_000


class PromptError(ValueError):
    """A prompt set that cannot be loaded or a template that cannot be filled."""


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    """Return ``value`` as compact JSON with sorted keys (the form that is hashed)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class PromptSet:
    """A loaded prompt set. ``prompt_hash`` covers every field that is sent or read back."""

    prompt_id: str
    directory: Path
    system: str
    user_templates: dict[str, str]
    schema: dict[str, Any]
    label_path: tuple[str, ...]
    prompt_hash: str
    file_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def is_generic_probe(self) -> bool:
        """Return True for the probe-only prompts (id starting ``generic_probe``)."""
        return self.prompt_id.startswith("generic_probe")

    @property
    def schema_json(self) -> str:
        """Return the schema as the compact JSON string passed to ``--json-schema``."""
        return json.dumps(self.schema, separators=(",", ":"), ensure_ascii=False)

    def render_user(self, condition: str, fields: dict[str, Any]) -> str:
        """Return the user turn for ``condition``: its template with every ``{{field}}`` filled.

        Raises :class:`PromptError` for an unknown condition, or a field that is missing, None
        or not a string (lists are rendered by :mod:`wikilense.llm.claims` first, never here).
        """
        try:
            template = self.user_templates[condition]
        except KeyError:
            raise PromptError(
                f"prompt {self.prompt_id!r} has no user template for {condition!r}"
            ) from None

        def sub(match: re.Match[str]) -> str:
            name = match.group(1)
            value = fields.get(name)
            if not isinstance(value, str):
                raise PromptError(
                    f"template {condition!r} needs a string field {name!r}, got {type(value).__name__}"
                )
            return value

        return _FIELD_RE.sub(sub, template)


def load_prompt_set(prompt_id: str, prompts_dir: Path | None = None) -> PromptSet:
    """Load ``<prompts_dir>/<prompt_id>/prompt.json`` and its files; compute the prompt hash.

    Checks: the folder name equals the id in ``prompt.json``; every condition in
    :data:`CONDITIONS` has a user template; the schema has exactly one top-level property (kit
    D-50/D-53/D-63: flat schemas made the CLI loop far more often); the schema carries no
    ``maxLength``/``maxItems`` (kit D-85/D-86: size limits make the model rewrite its answer);
    the system prompt fits on one argv string.
    """
    base = (prompts_dir or PROMPTS_DIR) / prompt_id
    spec_path = base / "prompt.json"
    if not spec_path.is_file():
        raise PromptError(f"no prompt set {prompt_id!r} at {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("prompt_id") != prompt_id:
        raise PromptError(f"{spec_path} says prompt_id {spec.get('prompt_id')!r}, not {prompt_id!r}")

    def read(name: str) -> str:
        path = base / name
        if not path.is_file():
            raise PromptError(f"prompt {prompt_id!r}: missing file {path}")
        return path.read_text(encoding="utf-8")

    system = read(spec["system"])
    if len(system.encode("utf-8")) > MAX_SYSTEM_BYTES:
        raise PromptError(f"system prompt over {MAX_SYSTEM_BYTES} bytes (it goes on argv)")
    users = {cond: read(name) for cond, name in dict(spec.get("user") or {}).items()}
    missing = [c for c in CONDITIONS if c not in users]
    if missing:
        raise PromptError(f"prompt {prompt_id!r} has no user template for {missing}")
    schema = json.loads(read(spec["schema"]))
    if len(schema.get("properties") or {}) != 1:
        raise PromptError("the answer schema needs exactly one top-level property (kit D-53/D-63)")
    if any(k in canonical_json(schema) for k in ('"maxLength"', '"maxItems"')):
        raise PromptError("put length limits in the prompt, not the schema (kit D-85/D-86)")
    label_path = tuple(str(spec.get("label_path", "result.label")).split("."))
    material = {
        "prompt_id": prompt_id,
        "system": system,
        "user": users,
        "schema": schema,
        "label_path": list(label_path),
    }
    files = {"prompt.json": spec_path, spec["system"]: base / spec["system"],
             spec["schema"]: base / spec["schema"]}
    files.update({name: base / name for name in dict(spec.get("user") or {}).values()})
    return PromptSet(
        prompt_id=prompt_id,
        directory=base,
        system=system,
        user_templates=users,
        schema=schema,
        label_path=label_path,
        prompt_hash=sha256_text(canonical_json(material)),
        file_hashes={name: sha256_text(path.read_text(encoding="utf-8"))
                     for name, path in sorted(files.items())},
    )
