"""LLM fact-checking runner: FEVEROUS claims labelled by Claude models, in three conditions.

Each claim gets one label, SUPPORTS, REFUTES or NOT ENOUGH INFO, in three conditions:
``no_retrieval`` (the claim alone), ``retrieved`` (the claim plus WikiLense's own top-k chunks)
and ``gold`` (the claim plus its gold evidence text).

The runs use the Claude Code CLI in headless mode on a Claude subscription
(:mod:`wikilense.llm.cli_backend`). :mod:`wikilense.llm.backend` defines the small interface
both backends implement. ``wikilense/llm/NOTES.md`` maps each design choice to the finding it
follows.

Modules:
- :mod:`~wikilense.llm.models`: option names (sonnet, opus, haiku) mapped to full model ids.
- :mod:`~wikilense.llm.prompts`: versioned prompt sets, each with a prompt id and a content hash.
- :mod:`~wikilense.llm.claims`: the claims-file contract, evidence rendering and file hashing.
- :mod:`~wikilense.llm.retrieval`: CPU query embedding, ``search.search`` and the result cache.
- :mod:`~wikilense.llm.guard`: the item-count guard (generic probe prompts never exceed 50).
- :mod:`~wikilense.llm.backend` / :mod:`~wikilense.llm.cli_backend`: one call, with its retries.
- :mod:`~wikilense.llm.runner`: run folders, manifest, resume, workers, stops.
- :mod:`~wikilense.llm.probe`: builds the probe claims file from the 75-claim corpus.
- :mod:`~wikilense.llm.report`: measured summary of a run folder.
"""
