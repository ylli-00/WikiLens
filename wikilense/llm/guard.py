"""The item-count guard, checked before any call.

Per model, the runner counts the (claim, condition) items of this run: the items it plans to send
plus the items already done in a resumed run folder (their union, so a resume with a smaller
``--limit-claims`` cannot hide earlier work). Two rules:

1. A prompt whose id starts with ``generic_probe`` never goes over :data:`ITEM_LIMIT` items per
   model, **even with --full**. The generic prompt only proves that the runner works; the owner
   replaces it before any full run.
2. Any run with more than :data:`ITEM_LIMIT` items per model needs ``--full``, whatever the
   prompt.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

ITEM_LIMIT = 50


class GuardError(RuntimeError):
    """The run would send more items than the guard allows."""


def items_per_model(
    planned: Iterable[tuple[str, int, str]],
    done: Iterable[tuple[str, int, str]] = (),
) -> dict[str, int]:
    """Return ``{model_id: number of distinct (claim_id, condition)}`` over planned and done.

    Both arguments are ``(model_id, claim_id, condition)`` triples.
    """
    seen: dict[str, set[tuple[int, str]]] = {}
    for model_id, claim_id, condition in (*planned, *done):
        seen.setdefault(model_id, set()).add((int(claim_id), condition))
    return {model_id: len(items) for model_id, items in seen.items()}


def check_item_budget(prompt_id: str, counts: Mapping[str, int], full: bool) -> None:
    """Raise :class:`GuardError` when ``counts`` (per model) break either rule; else return."""
    over = {m: n for m, n in counts.items() if n > ITEM_LIMIT}
    if not over:
        return
    detail = ", ".join(f"{m}: {n}" for m, n in sorted(over.items()))
    if prompt_id.startswith("generic_probe"):
        raise GuardError(
            f"prompt {prompt_id!r} is a generic probe prompt: at most {ITEM_LIMIT} items per model, "
            f"even with --full ({detail}). Write the real prompt first."
        )
    if not full:
        raise GuardError(
            f"more than {ITEM_LIMIT} items per model ({detail}) needs --full"
        )
