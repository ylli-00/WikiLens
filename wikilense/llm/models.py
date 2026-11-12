"""The runner's model option names and the full model ids they stand for.

Only full ids are ever passed to ``--model``. The kit saw ``--model sonnet`` serve an older model
(``claude-sonnet-4-6`` on 2026-08-14; kit ``code/reference/src/models.py``), and its runner
refuses aliases. The table is the single place to change when a newer model comes out.
"""

from __future__ import annotations

#: option name -> full model id. The latest Claude Sonnet, Opus and Haiku (2026-11).
MODEL_IDS: dict[str, str] = {
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
    "haiku": "claude-haiku-5-5",
}

#: Names the CLI resolves to a model of its own choosing; never valid as a --model value.
CLI_ALIASES: frozenset[str] = frozenset(
    {"sonnet", "opus", "haiku", "fable", "default", "opusplan", "best"}
)


class ModelError(ValueError):
    """An unknown model option name, or a model id that is not a full id."""


def resolve_models(names: str | list[str]) -> dict[str, str]:
    """Return ``{option name: full id}`` for a comma-separated string or a list of option names.

    Order is kept and duplicates are dropped. Raises :class:`ModelError` for an unknown name.
    """
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",")]
    result: dict[str, str] = {}
    for name in names:
        if not name:
            continue
        key = name.lower()
        if key not in MODEL_IDS:
            raise ModelError(
                f"unknown model option {name!r}; choose from {sorted(MODEL_IDS)}"
            )
        result[key] = MODEL_IDS[key]
    if not result:
        raise ModelError("no model given")
    for full_id in result.values():
        check_full_id(full_id)
    return result


def check_full_id(model_id: str) -> str:
    """Return ``model_id`` when it is a full id (``claude-...``, not an alias); else raise."""
    if model_id.lower() in CLI_ALIASES or not model_id.startswith("claude-"):
        raise ModelError(
            f"{model_id!r} is not a full model id; aliases resolve to whatever the CLI picks"
        )
    return model_id


def option_of(model_id: str) -> str:
    """Return the option name of a full id from :data:`MODEL_IDS` (``model_id`` if unknown)."""
    for name, full_id in MODEL_IDS.items():
        if full_id == model_id:
            return name
    return model_id
