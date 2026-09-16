#!/usr/bin/env python3
"""FEVEROUS multi-article evidence analysis for Wikilense.

Streams the FEVEROUS train/dev annotation files (JSONL) and measures how many
claims need evidence from two or more Wikipedia pages, plus related statistics.
Every figure is computed from the files. The only constants are the paper's
reference numbers (comparison table), the tier thresholds from the project
brief, the timestamp threshold used to describe annotator_operations, and the
id/page shown in the paper's QA-interface screenshot (a lookup, not a result).

Standard library only (Python 3.8+).

Usage (from the project root):
    python3 analysis/feverous/feverous_analysis.py --out-md analysis/feverous/feverous_analysis_output.md
Defaults: data is read from <project root>/data/feverous; the CSV and JSON are
(re)written next to this script. Without --out-md the markdown report goes to stdout.

Definitions (see FEVEROUS_ANALYSIS.md):
  page of an element id    el.split("_")[0]   (official FEVEROUS rule; cross-checked
                           against a strict "<page>_<type>_<numbers>" regex)
  pages of an evidence set distinct pages over the set's "content" ids (exact string)
  min_pages / max_pages    min / max of pages over a claim's evidence sets
  strict multi-article     min_pages >= 2   (every alternative set needs 2+ pages)
  lenient multi-article    max_pages >= 2   (at least one set spans 2+ pages)
  set modality             sentence-only: all ids are sentences
                           table-only:    all ids are cell / header_cell / table_caption
                           list-only:     all ids are item
                           mixed:         anything else
"""

import argparse
import collections
import csv
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

SPLIT_FILES = {
    "train": "feverous_train_challenges.jsonl",
    "dev": "feverous_dev_challenges.jsonl",
}
SPLITS = ["train", "dev"]
ALL_SPLITS = ["train", "dev", "train+dev"]
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
LABEL_GROUPS = ["ALL"] + LABELS
REQUIRED_KEYS = {"id", "claim", "label", "evidence", "annotator_operations", "challenge"}
SET_KEYS = {"content", "context"}

ELEMENT_RE = re.compile(
    r"^(?P<page>.*?)_(?P<type>sentence|header_cell|cell|table_caption|item)_(?P<num>\d+(?:_\d+)*)$"
)
EXPECTED_ARITY = {"sentence": 1, "table_caption": 1, "item": 2, "cell": 3, "header_cell": 3}
SECTION_CTX_RE = re.compile(r"^(?P<page>.*)_section_(?P<num>\d+)$")
TABLE_TYPES = {"cell", "header_cell", "table_caption"}
MODALITIES = ["sentence-only", "table-only", "list-only", "mixed"]
PAGE_BINS = ["1", "2", "3", "4", "5+"]
MULTIHOP_TAG = "Multi-hop Reasoning"
ANOMALY_KINDS = [
    "extra_record_keys",
    "extra_evidence_set_keys",
    "empty_content_set",
    "duplicate_id_within_set",
    "context_keys_differ_from_content",
    "duplicate_evidence_set_in_claim",
    "split_rule_disagrees_with_regex",
    "unexpected_id_arity",
    "context_title_page_differs",
    "context_section_page_differs",
]
# annotator_operations "time" is documented as seconds from the start of the annotation;
# values above this threshold after a claim's first "finish" are counted separately.
TIME_JUMP_THRESHOLD = 1e6

# Reference numbers from the paper (arXiv:2106.05707 v3): Table 2 (claims, labels, evidence
# sets by type) and Table 5 "Verification Challenges". Used only for the comparison table.
PAPER_CHALLENGES = [MULTIHOP_TAG, "Numerical Reasoning", "Entity Disambiguation",
                    "Combining Tables and Text", "Search terms not in claim", "Other"]
SETS_SENT, SETS_CELLS, SETS_BOTH = ("sets: sentences only", "sets: cells only (tables/lists, no sentence)",
                                    "sets: sentences + cells")
PAPER_REFERENCE = {
    ("train", "claims"): 71291, ("dev", "claims"): 7890,
    ("train", "SUPPORTS"): 41835, ("train", "REFUTES"): 27215, ("train", "NOT ENOUGH INFO"): 2241,
    ("dev", "SUPPORTS"): 3908, ("dev", "REFUTES"): 3481, ("dev", "NOT ENOUGH INFO"): 501,
    ("train", MULTIHOP_TAG): 11624, ("dev", MULTIHOP_TAG): 1281,
    ("train", "Numerical Reasoning"): 7214, ("dev", "Numerical Reasoning"): 873,
    ("train", "Entity Disambiguation"): 1353, ("dev", "Entity Disambiguation"): 201,
    ("train", "Combining Tables and Text"): 10083, ("dev", "Combining Tables and Text"): 1035,
    ("train", "Search terms not in claim"): 824, ("dev", "Search terms not in claim"): 131,
    ("train", "Other"): 40193, ("dev", "Other"): 4369,
    ("train", SETS_SENT): 31607, ("dev", SETS_SENT): 3745,
    ("train", SETS_CELLS): 25020, ("dev", SETS_CELLS): 2738,
    ("train", SETS_BOTH): 20865, ("dev", SETS_BOTH): 2468,
}
# The paper's QA-interface screenshot (v3 Fig. 14) shows the claim "Days of War is a first-person
# shooter game with a multiplayer mode, it is developed by Driven Arts. (31027)". Used only to look
# up what the released files hold for that id and that page.
SCREENSHOT_ID = 31027
SCREENSHOT_PAGE = "Days of War"


class FormatError(Exception):
    pass


def page_of(element_id):
    """Official FEVEROUS rule: page title is everything before the first underscore."""
    return element_id.split("_")[0]


def page_bin(n):
    return "5+" if n >= 5 else str(n)


def modality_of(types):
    if not types:
        return "empty"
    if types == {"sentence"}:
        return "sentence-only"
    if types <= TABLE_TYPES:
        return "table-only"
    if types == {"item"}:
        return "list-only"
    return "mixed"


def tier_of(n):
    if n >= 10000:
        return 5
    if n >= 5000:
        return 4
    if n >= 2000:
        return 3
    if n >= 500:
        return 2
    if n >= 1:
        return 1
    return 0


def digest(obj):
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def normalise_claim(text):
    """Near-duplicate key: NFKC, casefold, drop all non-word characters (whitespace, punctuation)."""
    return re.sub(r"\W+", "", unicodedata.normalize("NFKC", text).casefold())


def iter_records(path):
    """Yield (line_number, record) for every annotation line after the header line."""
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise FormatError(f"{path} line {lineno}: invalid JSON ({e})")
            if lineno == 1:
                # The official reader skips line 1 unconditionally; we also check it is a header.
                if not (isinstance(rec, dict) and all(v == "" for v in rec.values())):
                    raise FormatError(f"{path}: line 1 is not the expected empty header: {rec!r}")
                continue
            if not isinstance(rec, dict):
                raise FormatError(f"{path} line {lineno}: record is {type(rec).__name__}, expected object")
            yield lineno, rec


class Analysis:
    def __init__(self):
        self.claims = collections.Counter()          # (split, label_group)
        self.flags = collections.Counter()           # (split, label_group, flag)
        self.set_pages = collections.Counter()       # (split, label_group, bin)
        self.set_modality = collections.Counter()    # (split, label_group, modality)
        self.set_mixed = collections.Counter()       # (split, label_group, composition of a mixed set)
        self.n_sets = collections.Counter()          # (split, n_sets)
        self.challenge = collections.Counter()       # (split, label_group, challenge, flag)
        self.elements = collections.Counter()        # (split, element_type)
        self.sentences = collections.Counter()       # (split, label_group, key)
        self.pages_cited = collections.defaultdict(set)
        self.max_set_pages = collections.Counter()   # split -> max pages in one set
        self.anomalies = collections.Counter()       # (split, kind)
        self.anomaly_examples = {k: [] for k in ANOMALY_KINDS}
        self.example_record = None
        self.record_keysets = collections.Counter()  # (split, "k1,k2,...")
        self.set_keysets = collections.Counter()     # (split, "k1,k2")
        self.op_claims = collections.Counter()       # (split, fact) -- counts of claims
        self.op_entries = collections.Counter()      # (split, fact) -- counts of operation entries
        self.op_names = collections.Counter()        # (split, operation name)
        self.claim_ids = collections.defaultdict(set)
        # Duplicate detection keeps only digests: text key -> [(split, id, label, evidence digest)]
        self.exact_texts = collections.defaultdict(list)
        self.normalised_texts = collections.defaultdict(list)
        self.screenshot_lookup = []                  # records matching SCREENSHOT_ID or SCREENSHOT_PAGE

    # ------------------------------------------------------------------ helpers
    def _anomaly(self, split, kind, example):
        self.anomalies[(split, kind)] += 1
        self.anomalies[("train+dev", kind)] += 1
        if len(self.anomaly_examples[kind]) < 5:
            self.anomaly_examples[kind].append(example)

    def _operations(self, split, ops):
        """Structural facts about annotator_operations (no interpretation)."""
        if not isinstance(ops, list):
            raise FormatError(f"annotator_operations is {type(ops).__name__}, expected list")
        oc, oe = self.op_claims, self.op_entries
        starts = finishes = 0
        prev, backwards = None, False
        after_finish = large_time_after_finish = False
        for op in ops:
            if not isinstance(op, dict):
                raise FormatError(f"annotator_operations entry is {type(op).__name__}, expected object")
            past_first_finish = finishes > 0  # a "finish" occurred before this operation
            after_finish |= past_first_finish
            oe[(split, "entries with keys {" + ", ".join(sorted(op)) + "}")] += 1
            oe[(split, f"entries with time of type {type(op.get('time')).__name__}")] += 1
            oe[(split, f"entries with value of type {type(op.get('value')).__name__}")] += 1
            name = op.get("operation")
            self.op_names[(split, name)] += 1
            starts += name == "start"
            finishes += name == "finish"
            try:
                t = float(op.get("time"))
            except (TypeError, ValueError):
                oe[(split, "entries with unparseable time")] += 1
                continue
            if prev is not None and t < prev:
                backwards = True
            if past_first_finish and t > TIME_JUMP_THRESHOLD:
                large_time_after_finish = True
            prev = t
        oc[(split, "claims")] += 1
        oc[(split, "claims without a start operation")] += starts == 0
        oc[(split, "claims with 2+ start operations")] += starts >= 2
        oc[(split, "claims without a finish operation")] += finishes == 0
        oc[(split, "claims with 2+ finish operations")] += finishes >= 2
        oc[(split, "claims not ending with finish")] += not ops or ops[-1].get("operation") != "finish"
        oc[(split, "claims with operations after the first finish")] += after_finish
        oc[(split, f"claims with a time > {TIME_JUMP_THRESHOLD:g} after the first finish")] += large_time_after_finish
        oc[(split, "claims whose time goes backwards")] += backwards

    @staticmethod
    def _groups(split, label):
        for s in (split, "train+dev"):
            for g in ("ALL", label):
                yield s, g

    # ------------------------------------------------------------------ per claim
    def analyse_claim(self, split, lineno, rec):
        where = f"{split} line {lineno}"
        missing = REQUIRED_KEYS - rec.keys()
        if missing:
            raise FormatError(f"{where}: missing keys {sorted(missing)}")
        label = rec["label"]
        if label not in LABELS:
            raise FormatError(f"{where}: unexpected label {label!r}")
        evidence = rec["evidence"]
        if not isinstance(evidence, list):
            raise FormatError(f"{where}: evidence is {type(evidence).__name__}, expected list")
        for s in evidence:
            if not isinstance(s, dict) or not SET_KEYS <= s.keys():
                raise FormatError(f"{where}: evidence set without content/context: {s!r}")
            if not isinstance(s["content"], list) or not isinstance(s["context"], dict):
                raise FormatError(f"{where}: evidence set content/context have unexpected types")
            self.set_keysets[(split, ",".join(sorted(s.keys())))] += 1
            if s.keys() - SET_KEYS:
                self._anomaly(split, "extra_evidence_set_keys", (rec["id"], sorted(s.keys() - SET_KEYS)))
        extra = rec.keys() - REQUIRED_KEYS
        if extra:
            self._anomaly(split, "extra_record_keys", (rec["id"], sorted(extra)))
        self.record_keysets[(split, ",".join(sorted(rec.keys())))] += 1
        try:
            self._operations(split, rec["annotator_operations"])
        except FormatError as e:
            raise FormatError(f"{where}: {e}")
        if self.example_record is None and evidence and evidence[0]["content"]:
            self.example_record = describe_structure(rec)

        set_pages, set_pages_cf, set_modalities, set_units = [], [], [], []
        gold_sentences, gold_sentences_outside_lead = set(), set()
        any_element_outside_lead = False
        seen_sets = set()

        for s in evidence:
            content, context = s["content"], s["context"]
            if not content:
                self._anomaly(split, "empty_content_set", rec["id"])
            if len(set(content)) != len(content):
                self._anomaly(split, "duplicate_id_within_set", rec["id"])
            if set(context.keys()) != set(content):
                self._anomaly(split, "context_keys_differ_from_content", rec["id"])
            key = frozenset(content)
            if key in seen_sets:
                self._anomaly(split, "duplicate_evidence_set_in_claim", rec["id"])
            seen_sets.add(key)

            pages, pages_cf, types, units = set(), set(), set(), set()
            for el in content:
                m = ELEMENT_RE.match(el) if isinstance(el, str) else None
                if m is None:
                    raise FormatError(f"{where}: unrecognised element id {el!r}")
                etype, page = m.group("type"), page_of(el)
                if m.group("page") != page:
                    self._anomaly(split, "split_rule_disagrees_with_regex", (rec["id"], el))
                if m.group("num").count("_") + 1 != EXPECTED_ARITY[etype]:
                    self._anomaly(split, "unexpected_id_arity", (rec["id"], el))
                pages.add(page)
                pages_cf.add(page.casefold())
                types.add(etype)
                self.elements[(split, etype)] += 1
                self.elements[("train+dev", etype)] += 1
                self.pages_cited[split].add(page)
                self.pages_cited["train+dev"].add(page)

                # Annotation-context proxy (no Wikipedia DB needed): the context lists the section
                # the element is in and its parent sections. No section id => the element comes
                # before the first section of its page, i.e. it is in the lead.
                sections = []
                for c in context.get(el, []):
                    if c.endswith("_title"):
                        if c[: -len("_title")] != page:
                            self._anomaly(split, "context_title_page_differs", (rec["id"], el, c))
                        continue
                    sm = SECTION_CTX_RE.match(c)
                    if sm:
                        sections.append(c)
                        if sm.group("page") != page:
                            self._anomaly(split, "context_section_page_differs", (rec["id"], el, c))
                units.add((page, frozenset(sections)))
                outside_lead = bool(sections)
                any_element_outside_lead |= outside_lead
                if etype == "sentence":
                    gold_sentences.add(el)
                    if outside_lead:
                        gold_sentences_outside_lead.add(el)

            n_pages = len(pages)
            set_pages.append(n_pages)
            set_pages_cf.append(len(pages_cf))
            set_units.append(len(units))
            modality = modality_of(types)
            set_modalities.append(modality)
            self.max_set_pages[split] = max(self.max_set_pages[split], n_pages)
            self.max_set_pages["train+dev"] = max(self.max_set_pages["train+dev"], n_pages)
            families = sorted({"sentence" if t == "sentence" else "list" if t == "item" else "table" for t in types})
            for sp, g in self._groups(split, label):
                self.set_pages[(sp, g, page_bin(n_pages))] += 1
                self.set_modality[(sp, g, modality)] += 1
                if modality == "mixed":
                    self.set_mixed[(sp, g, "+".join(families))] += 1

        n_sets = len(evidence)
        min_pages = min(set_pages) if set_pages else 0
        max_pages = max(set_pages) if set_pages else 0
        flags = {
            "strict": min_pages >= 2,
            "lenient": max_pages >= 2,
            "strict_casefold": bool(set_pages_cf) and min(set_pages_cf) >= 2,
            "lenient_casefold": bool(set_pages_cf) and max(set_pages_cf) >= 2,
            "strict_units_ctx": bool(set_units) and min(set_units) >= 2,
            "lenient_units_ctx": bool(set_units) and max(set_units) >= 2,
            "has_sentence_only_set": "sentence-only" in set_modalities,
            "has_sentence_only_multi_page_set": any(
                m == "sentence-only" and p >= 2 for m, p in zip(set_modalities, set_pages)
            ),
            "nonempty_evidence": any(p > 0 for p in set_pages),
            "has_gold_sentence": bool(gold_sentences),
            "any_element_outside_lead_ctx": any_element_outside_lead,
            "any_sentence_outside_lead_ctx": bool(gold_sentences_outside_lead),
            "multihop_tag": rec["challenge"] == MULTIHOP_TAG,
        }

        self.n_sets[(split, n_sets)] += 1
        self.n_sets[("train+dev", n_sets)] += 1
        for sp, g in self._groups(split, label):
            self.claims[(sp, g)] += 1
            for name, value in flags.items():
                if value:
                    self.flags[(sp, g, name)] += 1
            self.sentences[(sp, g, "gold_sentences")] += len(gold_sentences)
            self.sentences[(sp, g, "gold_sentences_outside_lead_ctx")] += len(gold_sentences_outside_lead)
            ch = rec["challenge"]
            self.challenge[(sp, g, ch, "claims")] += 1
            for name in ("strict", "lenient", "strict_units_ctx", "lenient_units_ctx"):
                if flags[name]:
                    self.challenge[(sp, g, ch, name)] += 1

        # Identity and duplicate bookkeeping (digests only).
        self.claim_ids[split].add(rec["id"])
        entry = (split, rec["id"], label, digest(sorted(sorted(s["content"]) for s in evidence)))
        self.exact_texts[digest(rec["claim"])].append(entry)
        self.normalised_texts[digest(normalise_claim(rec["claim"]))].append(entry)
        if rec["id"] == SCREENSHOT_ID or any(page_of(el) == SCREENSHOT_PAGE for s in evidence for el in s["content"]):
            self.screenshot_lookup.append({
                "split": split, "id": rec["id"], "label": label, "challenge": rec["challenge"],
                "claim": rec["claim"], "evidence": [s["content"] for s in evidence],
                "finish_ops": sum(op.get("operation") == "finish" for op in rec["annotator_operations"]),
            })

        return {
            "id": rec["id"],
            "split": split,
            "label": label,
            "challenge": rec["challenge"],
            "n_sets": n_sets,
            "min_pages": min_pages,
            "max_pages": max_pages,
            "has_sentence_only_set": int(flags["has_sentence_only_set"]),
            "modality_summary": "|".join(set_modalities),
            "set_pages": "|".join(str(p) for p in set_pages),
            "strict_multi_article": int(flags["strict"]),
            "lenient_multi_article": int(flags["lenient"]),
            "has_sentence_only_multi_page_set": int(flags["has_sentence_only_multi_page_set"]),
        }


# ---------------------------------------------------------------------- derived summaries
def derive(a):
    """Values computed from the counters, shared by the markdown report and the JSON file."""
    d = {}

    rows = []
    for split in SPLITS:
        observed = {"claims": a.claims[(split, "ALL")]}
        observed.update({lab: a.claims[(split, lab)] for lab in LABELS})
        observed.update({ch: a.challenge[(split, "ALL", ch, "claims")] for ch in PAPER_CHALLENGES})
        mod = lambda m: a.set_modality[(split, "ALL", m)]
        list_table = a.set_mixed[(split, "ALL", "list+table")]
        observed[SETS_SENT] = mod("sentence-only")
        observed[SETS_CELLS] = mod("table-only") + mod("list-only") + list_table
        observed[SETS_BOTH] = mod("mixed") - list_table
        for key, value in observed.items():
            ref = PAPER_REFERENCE[(split, key)]
            rows.append({"split": split, "count": key, "paper": ref, "files": value, "diff": value - ref})
    d["paper_comparison"] = rows
    d["challenge_counts_match_table5"] = all(
        r["diff"] == 0 for r in rows if r["count"] in PAPER_CHALLENGES)

    d["unicode_pages"] = {}
    for s in ALL_SPLITS:
        pages = a.pages_cited[s]
        non_ascii = [x for x in pages if not x.isascii()]
        not_nfc = sorted(x for x in non_ascii if unicodedata.normalize("NFC", x) != x)
        groups = collections.defaultdict(set)
        for x in pages:
            groups[unicodedata.normalize("NFC", x).casefold()].add(x)
        d["unicode_pages"][s] = {
            "distinct_pages": len(pages),
            "non_ascii": len(non_ascii),
            "non_ascii_not_nfc": len(not_nfc),
            "ids_removed_by_nfc": len(pages) - len({unicodedata.normalize("NFC", x) for x in pages}),
            "ids_removed_by_nfc_casefold": len(pages) - len(groups),
            "nfc_casefold_collision_groups": sorted(sorted(g) for g in groups.values() if len(g) > 1),
            "examples_not_nfc_shown_nfc": [unicodedata.normalize("NFC", x) for x in not_nfc[:2]],
        }

    d["claim_ids"] = {
        "train_distinct": len(a.claim_ids["train"]),
        "dev_distinct": len(a.claim_ids["dev"]),
        "shared_between_splits": len(a.claim_ids["train"] & a.claim_ids["dev"]),
    }

    def dup_summary(index):
        groups = [sorted(v, key=lambda e: (e[0], e[1])) for v in index.values() if len(v) > 1]
        groups.sort(key=lambda g: (g[0][0], g[0][1]))
        return {
            "groups": len(groups),
            "groups_within_train": sum(1 for g in groups if {e[0] for e in g} == {"train"}),
            "groups_within_dev": sum(1 for g in groups if {e[0] for e in g} == {"dev"}),
            "groups_across_splits": sum(1 for g in groups if {e[0] for e in g} == {"train", "dev"}),
            "detail": [{
                "claims": [f"{e[0]} {e[1]}" for e in g],
                "labels": [e[2] for e in g],
                "same_label": len({e[2] for e in g}) == 1,
                "same_evidence_sets": len({e[3] for e in g}) == 1,
            } for g in groups],
        }

    d["claim_text_duplicates"] = {"exact": dup_summary(a.exact_texts),
                                  "normalised": dup_summary(a.normalised_texts)}

    tp = a.challenge[("train+dev", "ALL", MULTIHOP_TAG, "strict")]
    d["tier"] = {}
    for name, flag in (("strict", "strict"), ("lenient", "lenient")):
        total = a.flags[("train+dev", "ALL", flag)]
        excl = total - a.flags[("train+dev", "NOT ENOUGH INFO", flag)]
        d["tier"][name] = {"train_dev": total, "tier": tier_of(total), "excl_nei": excl, "tier_excl_nei": tier_of(excl)}
    d["multihop_tag_vs_multi_article"] = {}
    for s in ALL_SPLITS:
        for measure in ("strict", "lenient"):
            tp = a.challenge[(s, "ALL", MULTIHOP_TAG, measure)]
            tagged = a.challenge[(s, "ALL", MULTIHOP_TAG, "claims")]
            positives = a.flags[(s, "ALL", measure)]
            fp, fn = tagged - tp, positives - tp
            d["multihop_tag_vs_multi_article"][f"{s}|{measure}"] = {
                "tp": tp, "fp": fp, "fn": fn, "tn": a.claims[(s, "ALL")] - tp - fp - fn,
                "precision": tp / tagged if tagged else None,
                "recall": tp / positives if positives else None,
            }
    return d


# ---------------------------------------------------------------------- output
def pct(n, d, digits=1):
    return f"{n:,} ({100.0 * n / d:.{digits}f}%)" if d else f"{n:,} (n/a)"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def describe_structure(rec):
    """Compact structural description of one record (keys, types, nesting). Needs non-empty evidence."""
    ev = rec["evidence"]
    first = ev[0]
    first_id = first["content"][0]
    lines = [
        "keys: " + ", ".join(f"{k} ({type(v).__name__})" for k, v in rec.items()),
        f"id = {rec['id']!r}; label = {rec['label']!r}; challenge = {rec['challenge']!r}",
        f"claim = {rec['claim']!r}",
        f"evidence: list of {len(ev)} set(s); set keys = {sorted(first.keys())}",
        f"evidence[0].content (list of {len(first['content'])} str) = {first['content']!r}",
        f"evidence[0].context (dict str -> list[str]); e.g. {first_id!r} -> {first['context'].get(first_id)!r}",
        f"annotator_operations: list of {len(rec['annotator_operations'])} dicts; first 3 = "
        + json.dumps(rec["annotator_operations"][:3], ensure_ascii=False),
    ]
    return "\n".join("    " + line for line in lines)


def report(a, d, out):
    p = lambda *s: print(*s, file=out)

    p("# FEVEROUS analysis output\n")
    p("## Example record (first train record with non-empty evidence)\n")
    p("```")
    p(a.example_record or "    (no record with non-empty evidence)")
    p("```\n")

    p("## Format checks\n")
    kinds = ANOMALY_KINDS + sorted({k for (_, k) in a.anomalies} - set(ANOMALY_KINDS))
    rows = [[k] + [f"{a.anomalies[(s, k)]:,}" for s in ALL_SPLITS] + [repr(a.anomaly_examples.get(k, [])[:3]) if a.anomaly_examples.get(k) else ""]
            for k in kinds]
    p(md_table(["check (count of violations)"] + ALL_SPLITS + ["examples (claim id, ...)"], rows))
    p("\nChecks that raise instead of counting: header line, JSON syntax, required record keys, label values, "
      "evidence is a list of objects with content (list) and context (object), annotator_operations is a list "
      "of objects, every element id matches `<page>_<type>_<numbers>`.\n")

    p("Record and evidence-set key sets:\n")
    rows = [[s, "record", ks, f"{n:,}"] for (s, ks), n in sorted(a.record_keysets.items())]
    rows += [[s, "evidence set", ks, f"{n:,}"] for (s, ks), n in sorted(a.set_keysets.items())]
    p(md_table(["split", "object", "keys", "count"], rows))
    p()

    p("Unicode form of page ids parsed from evidence (a join against an NFC-normalised Wikipedia table needs normalising):\n")
    rows = []
    for s in ALL_SPLITS:
        u = d["unicode_pages"][s]
        rows.append([s, f"{u['distinct_pages']:,}", f"{u['non_ascii']:,}", f"{u['non_ascii_not_nfc']:,}",
                     u["ids_removed_by_nfc"], u["ids_removed_by_nfc_casefold"],
                     repr(u["nfc_casefold_collision_groups"]), repr(u["examples_not_nfc_shown_nfc"])])
    p(md_table(["split", "distinct pages", "non-ASCII", "non-ASCII and not NFC", "ids removed by NFC",
                "ids removed by NFC+casefold", "NFC+casefold collision groups", "examples of non-NFC ids (shown NFC)"], rows))
    p()

    p("annotator_operations, per claim:\n")
    facts = sorted({f for (_, f) in a.op_claims})
    p(md_table(["claims ...", "train", "dev"], [[f] + [f"{a.op_claims[(s, f)]:,}" for s in SPLITS] for f in facts]))
    p()
    p("annotator_operations, per operation entry:\n")
    facts = sorted({f for (_, f) in a.op_entries})
    p(md_table(["operation entries ...", "train", "dev"], [[f] + [f"{a.op_entries[(s, f)]:,}" for s in SPLITS] for f in facts]))
    p()
    names = sorted({n for (_, n) in a.op_names},
                   key=lambda n: (-a.op_names[("train", n)], -a.op_names[("dev", n)], str(n)))
    p(md_table(["operation", "train", "dev"], [[n, f"{a.op_names[('train', n)]:,}", f"{a.op_names[('dev', n)]:,}"] for n in names]))
    p()

    ci = d["claim_ids"]
    p(f"Claim ids: train {ci['train_distinct']:,} distinct, dev {ci['dev_distinct']:,} distinct, "
      f"shared between splits {ci['shared_between_splits']:,}.\n")
    p("Repeated claim texts (exact string; and normalised = NFKC + casefold + all non-word characters removed):\n")
    rows = []
    for kind in ("exact", "normalised"):
        dup = d["claim_text_duplicates"][kind]
        rows.append([kind, dup["groups"], dup["groups_within_train"], dup["groups_within_dev"], dup["groups_across_splits"]])
    p(md_table(["match", "groups of 2+ claims", "within train", "within dev", "across splits"], rows))
    p()
    detail = d["claim_text_duplicates"]["normalised"]["detail"]
    if detail:
        p(md_table(["normalised-text group", "labels", "same label", "same evidence sets"],
                   [[", ".join(g["claims"]), ", ".join(g["labels"]), g["same_label"], g["same_evidence_sets"]] for g in detail]))
        p()

    p(f"Released records with id {SCREENSHOT_ID} or evidence on page '{SCREENSHOT_PAGE}' "
      "(lookup for the paper's QA-interface screenshot):\n")
    rows = [[r["split"], r["id"], r["label"], r["challenge"], repr(r["claim"]), repr(r["evidence"]), r["finish_ops"]]
            for r in a.screenshot_lookup]
    p(md_table(["split", "id", "label", "challenge", "claim", "evidence sets (content)", "finish ops"], rows))
    p()

    p("## Basic counts vs. paper (arXiv:2106.05707 v3: Table 2 claims, labels and evidence sets; Table 5 verification challenges)\n")
    p(md_table(["split", "count", "paper", "files", "diff", "match"],
               [[r["split"], r["count"], f"{r['paper']:,}", f"{r['files']:,}", f"{r['diff']:+,}", "yes" if r["diff"] == 0 else "NO"]
                for r in d["paper_comparison"]]))
    p("\nRows starting with 'sets:' count evidence sets, not claims. To follow the paper's grouping, sets made of "
      "table and list elements without a sentence count as 'cells only' (they are 'mixed' in the modality table).\n")

    p("## Evidence sets per claim\n")
    ns = sorted({n for (_, n) in a.n_sets})
    rows = [[s] + [pct(a.n_sets[(s, n)], a.claims[(s, "ALL")]) for n in ns] for s in ALL_SPLITS]
    p(md_table(["split"] + [f"{n} set(s)" for n in ns], rows))
    p()

    p("## Element types in evidence content (occurrences)\n")
    types = ["sentence", "cell", "header_cell", "table_caption", "item"]
    rows = []
    for s in ALL_SPLITS:
        tot = sum(a.elements[(s, t)] for t in types)
        rows.append([s] + [pct(a.elements[(s, t)], tot) for t in types] + [f"{tot:,}", f"{len(a.pages_cited[s]):,}", a.max_set_pages[s]])
    p(md_table(["split"] + types + ["total", "distinct pages cited", "max pages in one set"], rows))
    p()

    p("## Multi-article claims (strict: min_pages >= 2; lenient: max_pages >= 2)\n")
    rows = []
    for s in ALL_SPLITS:
        for g in LABEL_GROUPS:
            n = a.claims[(s, g)]
            rows.append([s, g, f"{n:,}", pct(a.flags[(s, g, "strict")], n, 2), pct(a.flags[(s, g, "lenient")], n, 2)])
        n = a.claims[(s, "ALL")] - a.claims[(s, "NOT ENOUGH INFO")]
        rows.append([s, "SUPPORTS+REFUTES (excl. NEI)", f"{n:,}",
                     pct(a.flags[(s, "ALL", "strict")] - a.flags[(s, "NOT ENOUGH INFO", "strict")], n, 2),
                     pct(a.flags[(s, "ALL", "lenient")] - a.flags[(s, "NOT ENOUGH INFO", "lenient")], n, 2)])
    p(md_table(["split", "label", "claims", "strict multi-article", "lenient multi-article"], rows))
    p()

    page_cols = (["0"] if any(k[2] == "0" for k in a.set_pages) else []) + PAGE_BINS
    modality_cols = MODALITIES + (["empty"] if any(k[2] == "empty" for k in a.set_modality) else [])
    for title, counter, cols in [
        ("Pages per evidence set (share of evidence sets)", a.set_pages, page_cols),
        ("Evidence modality per set (share of evidence sets)", a.set_modality, modality_cols),
        ("Composition of 'mixed' sets (share of mixed sets; table = cell/header_cell/table_caption, list = item)",
         a.set_mixed, sorted({k[2] for k in a.set_mixed})),
    ]:
        p(f"## {title}\n")
        rows = []
        for s in ALL_SPLITS:
            for g in LABEL_GROUPS:
                tot = sum(counter[(s, g, c)] for c in cols)
                rows.append([s, g] + [pct(counter[(s, g, c)], tot) for c in cols] + [f"{tot:,}"])
        p(md_table(["split", "label"] + cols + ["sets"], rows))
        p()

    p("## Challenge tag vs. multi-article\n")
    if d["challenge_counts_match_table5"]:
        p("The released `challenge` field is the verification challenge chosen by the verifying annotator: "
          "its train and dev counts equal the paper's Table 5 'Verification Challenges' rows (see above).\n")
    else:
        p("Note: the `challenge` counts do NOT all equal the paper's Table 5 'Verification Challenges' rows (see above).\n")
    challenges = sorted({k[2] for k in a.challenge})
    for s in ALL_SPLITS:
        p(f"### {s}\n")
        rows = []
        for ch in challenges:
            n = a.challenge[(s, "ALL", ch, "claims")]
            if n:
                rows.append([ch, f"{n:,}", pct(a.challenge[(s, 'ALL', ch, 'strict')], n), pct(a.challenge[(s, 'ALL', ch, 'lenient')], n)])
        n = a.claims[(s, "ALL")]
        rows.append(["**all**", f"{n:,}", pct(a.flags[(s, 'ALL', 'strict')], n), pct(a.flags[(s, 'ALL', 'lenient')], n)])
        p(md_table(["challenge tag", "claims", "strict multi-article", "lenient multi-article"], rows))
        p()
        p(f"Strict multi-article by label ({s}; cells are strict / claims):\n")
        rows = []
        for ch in challenges + [None]:
            row = [ch or "**all**"]
            for g in LABEL_GROUPS:
                if ch is None:
                    n, k = a.claims[(s, g)], a.flags[(s, g, "strict")]
                else:
                    n, k = a.challenge[(s, g, ch, "claims")], a.challenge[(s, g, ch, "strict")]
                row.append(f"{k:,} / {n:,} ({100.0 * k / n:.1f}%)" if n else "0 / 0")
            rows.append(row)
        p(md_table(["challenge tag"] + LABEL_GROUPS, rows))
        p()
        rows = []
        for measure in ("strict", "lenient"):
            m = d["multihop_tag_vs_multi_article"][f"{s}|{measure}"]
            prec = f"{100.0 * m['precision']:.1f}%" if m["precision"] is not None else "n/a"
            rec = f"{100.0 * m['recall']:.1f}%" if m["recall"] is not None else "n/a"
            rows.append([measure, f"{m['tp']:,}", f"{m['fp']:,}", f"{m['fn']:,}", f"{m['tn']:,}", prec, rec])
        p(f"'{MULTIHOP_TAG}' tag as a predictor of multi-article evidence ({s}):\n")
        p(md_table(["target", "tag & multi (TP)", "tag & not multi (FP)", "no tag & multi (FN)", "neither (TN)",
                    "precision (share of tagged)", "recall (share of multi covered by tag)"], rows))
        p()

    p("## Usable slices for a prose-only index\n")
    rows = []
    for s in ALL_SPLITS:
        for g in LABEL_GROUPS:
            n = a.claims[(s, g)]
            rows.append([s, g, f"{n:,}", pct(a.flags[(s, g, "has_sentence_only_set")], n),
                         pct(a.flags[(s, g, "has_sentence_only_multi_page_set")], n)])
    p(md_table(["split", "label", "claims", "(a) >=1 sentence-only set", "(b) >=1 sentence-only set with 2+ pages"], rows))
    p()
    rows = [[s, f"{a.claims[(s, 'NOT ENOUGH INFO')]:,}",
             pct(a.flags[(s, 'NOT ENOUGH INFO', 'nonempty_evidence')], a.claims[(s, 'NOT ENOUGH INFO')])] for s in ALL_SPLITS]
    p(md_table(["split", "NEI claims", "(c) NEI claims with non-empty evidence"], rows))
    p()

    p("## Tier (strict multi-article, train+dev)\n")
    t = d["tier"]
    p(f"strict multi-article claims (train+dev): {t['strict']['train_dev']:,} -> tier {t['strict']['tier']}  ")
    p(f"lenient multi-article claims (train+dev): {t['lenient']['train_dev']:,} -> tier {t['lenient']['tier']} (same scale, for reference)  ")
    p(f"excluding NOT ENOUGH INFO: strict {t['strict']['excl_nei']:,} -> tier {t['strict']['tier_excl_nei']}; "
      f"lenient {t['lenient']['excl_nei']:,} -> tier {t['lenient']['tier_excl_nei']}")
    p()

    p("## Sensitivity and annotation-context proxies\n")
    p("Case-insensitive page names (page.casefold()) instead of exact strings:\n")
    rows = [[s, pct(a.flags[(s, 'ALL', 'strict')], a.claims[(s, 'ALL')], 2), pct(a.flags[(s, 'ALL', 'strict_casefold')], a.claims[(s, 'ALL')], 2),
             pct(a.flags[(s, 'ALL', 'lenient')], a.claims[(s, 'ALL')], 2), pct(a.flags[(s, 'ALL', 'lenient_casefold')], a.claims[(s, 'ALL')], 2)]
            for s in ALL_SPLITS]
    p(md_table(["split", "strict (exact)", "strict (casefold)", "lenient (exact)", "lenient (casefold)"], rows))
    p()
    p("Multi-unit evidence from annotation context: a unit is (page, set of section ids in the element's context). "
      "Strict = every set has 2+ units; lenient = some set has 2+ units. This approximates the tag's "
      "'two or more sections or articles'. Not verified against the Wikipedia DB.\n")
    for s in ALL_SPLITS:
        rows = []
        for ch in challenges:
            n = a.challenge[(s, "ALL", ch, "claims")]
            if n:
                rows.append([ch, f"{n:,}", pct(a.challenge[(s, 'ALL', ch, 'strict_units_ctx')], n),
                             pct(a.challenge[(s, 'ALL', ch, 'lenient_units_ctx')], n)])
        n = a.claims[(s, "ALL")]
        rows.append(["**all**", f"{n:,}", pct(a.flags[(s, 'ALL', 'strict_units_ctx')], n), pct(a.flags[(s, 'ALL', 'lenient_units_ctx')], n)])
        p(f"### Multi-unit (ctx): {s}\n")
        p(md_table(["challenge tag", "claims", "strict multi-unit (ctx)", "lenient multi-unit (ctx)"], rows))
        p()
    p("### Lead vs. later sections (ctx)\n")
    p("Lead vs. later sections from annotation context (an element is outside the lead when its context lists "
      "at least one section id). Gold sentences are counted once per claim (distinct element ids). "
      "Not verified against the Wikipedia DB.\n")
    rows = []
    for s in ALL_SPLITS:
        for g in LABEL_GROUPS:
            gs = a.sentences[(s, g, "gold_sentences")]
            n = a.claims[(s, g)]
            rows.append([s, g, f"{gs:,}", pct(a.sentences[(s, g, 'gold_sentences_outside_lead_ctx')], gs),
                         pct(a.flags[(s, g, 'any_sentence_outside_lead_ctx')], n),
                         pct(a.flags[(s, g, 'any_element_outside_lead_ctx')], n)])
    p(md_table(["split", "label", "gold sentences", "gold sentences outside lead (ctx)",
                "claims with >=1 gold SENTENCE outside lead (ctx)",
                "claims with >=1 gold element of ANY type outside lead (ctx)"], rows))
    p()


def stats_to_json(a, d):
    def flat(counter):
        return {"|".join(map(str, k)) if isinstance(k, tuple) else str(k): v
                for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}

    out = {
        "claims": flat(a.claims),
        "flags": flat(a.flags),
        "set_pages": flat(a.set_pages),
        "set_modality": flat(a.set_modality),
        "set_mixed": flat(a.set_mixed),
        "n_sets": flat(a.n_sets),
        "challenge": flat(a.challenge),
        "elements": flat(a.elements),
        "sentences": flat(a.sentences),
        "distinct_pages_cited": {s: len(a.pages_cited[s]) for s in ALL_SPLITS},
        "max_set_pages": {s: a.max_set_pages[s] for s in ALL_SPLITS},
        "record_keysets": flat(a.record_keysets),
        "evidence_set_keysets": flat(a.set_keysets),
        "annotator_operations_claims": flat(a.op_claims),
        "annotator_operations_entries": flat(a.op_entries),
        "operation_names": flat(a.op_names),
        "anomalies": {f"{s}|{k}": a.anomalies[(s, k)] for s in ALL_SPLITS for k in ANOMALY_KINDS},
        "anomaly_examples": {k: [repr(x) for x in a.anomaly_examples[k]] for k in ANOMALY_KINDS},
        "screenshot_lookup": a.screenshot_lookup,
    }
    out.update(d)
    return out


CSV_FIELDS = [
    "id", "split", "label", "challenge", "n_sets", "min_pages", "max_pages", "has_sentence_only_set",
    "modality_summary", "set_pages", "strict_multi_article", "lenient_multi_article",
    "has_sentence_only_multi_page_set",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent            # <project root>/analysis/feverous
    project_root = here.parents[1]
    ap.add_argument("--data-dir", type=Path, default=project_root / "data" / "feverous")
    ap.add_argument("--csv", type=Path, default=here / "feverous_claims.csv")
    ap.add_argument("--stats-json", type=Path, default=here / "feverous_stats.json")
    ap.add_argument("--out-md", type=Path, default=None, help="write the markdown report here instead of stdout")
    args = ap.parse_args()

    a = Analysis()
    tmp_csv = args.csv.with_name(args.csv.name + ".tmp")
    try:
        with open(tmp_csv, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            for split in SPLITS:
                path = args.data_dir / SPLIT_FILES[split]
                if not path.exists():
                    raise FormatError(f"missing file: {path}")
                for lineno, rec in iter_records(path):
                    writer.writerow(a.analyse_claim(split, lineno, rec))
    except FormatError as e:
        tmp_csv.unlink(missing_ok=True)
        sys.exit(f"FORMAT ERROR: {e}")
    os.replace(tmp_csv, args.csv)

    d = derive(a)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8", newline="\n") as f:
            report(a, d, f)
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        report(a, d, sys.stdout)
    with open(args.stats_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats_to_json(a, d), f, indent=1, ensure_ascii=False)
    print(f"Wrote {args.csv}, {args.stats_json}" + (f", {args.out_md}" if args.out_md else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
