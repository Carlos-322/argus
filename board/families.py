"""Which problem each flagged issue is: the classification behind the board's problem filter.

families.json lists the families. Every flagged issue (a data issue or an operator mistake) belongs to exactly one
family: the first listed family whose tags, plain names (tag_names.json) or text it matches, else a family named
after its tag's plain name ("d:<name>" or "m:<name>"). A family can also be raised by one of the deterministic
checks or by the episode's outcome, and then counts at any severity. So does each problem an episode was kept and
flagged with (context.json reader_issues, copied into dataset_checks by board/build.py): the listed family whose
"reader_issues" names its kind, else a data family named after the kind ("d:<words of the kind>"). Only a family of a
list in COUNTED_LISTS (a fault in the recording, an operator mistake) counts; a reader issue of a model reply that
gave no labels (list "labelling") or of how we read or showed the recording (list "handling": a limit of ours, such as
a table read every so many rows or a signal kept as its lowest, mean and highest value, or a property of the recording
that is not a fault in it, such as a camera that is not colour) is shown on the episode and returned under
not_counted, never in the counts or the filter. A family limited to some datasets ("datasets") is only matched on
those.

A family's text is read only where the tag says too little: a tag the model files several distinct problems under
(a family's "among", such as camera_fault: a camera turned away, a frozen image, glare) or a tag it coined (no entry
in tag_names.json). A known tag keeps its own name whatever its prose mentions. A family with "among" matches its text
only on issues whose tag reads as one of those plain names. The text claims a family only where its clause asserts the
problem (board/polarity.py: not where it is denied, retracted, asked about or supposed), the issue sentence before
its evidence, and a word two swapped letters away from a family's word counts as that word.

What counts (Families.counts, the one statement of the rule):
  - a data issue at medium or high severity;
  - an operator mistake that changes the outcome (its tag's kind in tag_names.json is "outcome") at medium or high;
  - any operator mistake at high.
Rated issues below these thresholds are minor. Issues without a valid severity are unrated.
Both remain visible on the episode and are excluded from severity totals.

    fam = Families()
    c = fam.classify(episode)      # {"counted": {slug: [issues]}, "minor": {...}, "unrated": {...}, "not_counted": {...}}
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

from board.label_projection import severity as _sev
from board.polarity import asserted, denied_spans

HERE = Path(__file__).resolve().parent
LISTS = {"data_issues": "data", "operator_mistakes": "mistake"}
COUNTED_LISTS = set(LISTS.values())


def union_seconds(spans) -> float:
    """Seconds covered by a set of (start, end) spans, overlaps counted once."""
    total, end = 0.0, None
    for a, b in sorted((a, b) for a, b in spans if b > a):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def check_hit(d: dict, key: str) -> bool:
    """key is "check.field": the deterministic check's flag on this episode."""
    name, field = key.split(".")
    return bool(((d.get("dataset_checks") or {}).get(name) or {}).get(field))


class Families:
    def __init__(self, path: Path | None = None, tag_names_path: Path | None = None):
        spec = json.loads(Path(path or HERE / "families.json").read_text())
        tn_path = Path(tag_names_path or HERE / "tag_names.json")
        tn = json.loads(tn_path.read_text()) if tn_path.exists() else {}
        self.plain = {k: {t: v.get("name") for t, v in d.items()} for k, d in tn.items() if not k.startswith("_")}
        self.kinds = {t: v.get("kind") for t, v in tn.get("operator_mistakes", {}).items()}
        self.defs = spec["families"]
        self._re = {f["slug"]: re.compile(f["text"], re.I) for f in self.defs if f.get("text")}
        self._spell_cache = {}

    def tag_name(self, key: str, tag: str | None) -> str:
        """The tag's plain name (tag_names.json), else its words with the first letter capitalised."""
        n = self.plain.get(key, {}).get(tag or "")
        if n:
            return n
        w = (tag or "untagged").replace("_", " ")
        return w[:1].upper() + w[1:]

    def catalog(self) -> dict:
        """The listed families, in order, for a page: slug -> name, list, and whether only a check raises it."""
        return {f["slug"]: {"name": f["name"], "list": f["list"],
                            "check": bool(f.get("checks") or f.get("reader_issues"))
                            and not (f.get("names") or f.get("tags") or f.get("text"))}
                for f in self.defs}

    def reader_family(self, kind: str) -> str:
        """The family a reader issue of this kind raises: the listed family that names the kind, else "d:" and the
        kind's words."""
        for f in self.defs:
            if kind in (f.get("reader_issues") or []):
                return f["slug"]
        w = str(kind or "reader issue").replace("_", " ")
        return "d:" + w[:1].upper() + w[1:]

    def list_of(self, slug: str) -> str:
        """The list a family is on: a listed family's own, else data for "d:" and mistake for "m:"."""
        f = next((f for f in self.defs if f["slug"] == slug), None)
        return f["list"] if f else "mistake" if str(slug).startswith("m:") else "data"

    @staticmethod
    def rated(i: dict) -> bool:
        return _sev(i) is not None

    def counts(self, key: str, i: dict) -> bool:
        """Whether an issue of this list ("data_issues" or "operator_mistakes") counts (the module docstring)."""
        s = _sev(i)
        if key == "data_issues":
            return s in ("medium", "high")
        return s == "high" or (s == "medium" and self.kinds.get(i.get("category")) == "outcome")

    def family_of(self, key: str, i: dict, dataset: str | None = None) -> str:
        lst = LISTS[key]
        cat = i.get("category") or ""
        plain = self.tag_name(key, cat)
        eligible = [f for f in self.defs if f['list'] == lst
                    and (not f.get('datasets') or dataset in f['datasets'])]
        # Structured categories outrank incidental words anywhere in the prose.
        for f in eligible:
            if cat in (f.get("tags") or []) or plain in (f.get("names") or []):
                return f["slug"]
        # Words refine only a tag that names no one problem: one the model files several problems under (a family's
        # "among") or one it coined (no entry in tag_names.json). A known tag keeps its own name.
        if not (self._broad(plain) or cat not in self.plain.get(key, {})):
            return ("d:" if lst == "data" else "m:") + plain
        # The issue says what the problem is; its evidence is read only when the issue asserts none of the families.
        # A word counts only where its clause asserts it (board/polarity.py).
        for field in ('issue', 'evidence'):
            text = self._spelled(str(i.get(field) or ''))
            spans = denied_spans(text) if text else []
            for f in eligible:
                if (f["slug"] in self._re and (not f.get('among') or plain in f['among'])
                        and asserted(self._re[f["slug"]], text, spans)):
                    return f["slug"]
        return ("d:" if lst == "data" else "m:") + plain

    def _spelled(self, text: str) -> str:
        """text with each word that two swapped letters keep from being a family's word put right ("dorpout"). A
        swap keeps every character offset, so the clause reading is unchanged."""
        if text in self._spell_cache:
            return self._spell_cache[text]
        out = text
        for m in re.finditer(r"[^\W\d_]{4,}", text):
            a, b = m.span()
            if any(x.start() < b and x.end() > a for r in self._re.values() for x in r.finditer(out)):
                continue
            w = m.group()
            for k in range(len(w) - 1):
                if w[k] == w[k + 1]:
                    continue
                cand = out[:a] + w[:k] + w[k + 1] + w[k] + w[k + 2:] + out[b:]
                if any(x.start() < b and x.end() > a for r in self._re.values() for x in r.finditer(cand)):
                    out = cand
                    break
        if len(self._spell_cache) > 4096:
            self._spell_cache.clear()
        self._spell_cache[text] = out
        return out

    def _broad(self, plain: str) -> bool:
        return any(plain in (f.get('among') or []) for f in self.defs)

    def classify(self, d: dict) -> dict:
        """The episode's families: counted ones with the issues that count under each (empty when a check or the
        outcome raised it), minor and unrated ones with no counted issue, and not_counted, the families of reader issues that are
        no fault in the recording (a list outside COUNTED_LISTS), shown on the episode and never counted."""
        counted, minor, unrated = (collections.defaultdict(list) for _ in range(3))
        not_counted = {}
        ds = d.get("dataset")
        for key in LISTS:
            for i in d.get(key) or []:
                if i and i.get("issue"):
                    group = counted if self.counts(key, i) else minor if self.rated(i) else unrated
                    group[self.family_of(key, i, ds)].append(i)
        # the episode's outcome, and each part's of a long recording labelled in parts (label/pieces.py puts those
        # under tasks), so a part reached and then undone counts as a short episode's does
        outcomes = {str((d.get("completion") or {}).get("task_completed") or "").lower()}
        outcomes |= {str(t.get("outcome") or "").lower() for t in d.get("tasks") or [] if isinstance(t, dict)}
        outcomes.discard("")
        for f in self.defs:
            if f.get("datasets") and ds not in f["datasets"]:
                continue
            if (any(check_hit(d, k) for k in f.get("checks") or [])
                    or outcomes & set(f.get("completion") or [])):
                counted.setdefault(f["slug"], [])
        for x in (d.get("dataset_checks") or {}).get("reader_issues") or []:
            if isinstance(x, dict) and x.get("kind"):
                fam = self.reader_family(str(x["kind"]))
                (counted if self.list_of(fam) in COUNTED_LISTS else not_counted).setdefault(fam, [])
        return {"counted": dict(counted), "minor": {k: v for k, v in minor.items() if k not in counted},
                "unrated": {k: v for k, v in unrated.items() if k not in counted},
                "not_counted": {k: v for k, v in not_counted.items() if k not in counted}}

    def hands_hidden_seconds(self, d: dict) -> float | None:
        """For head-camera footage, the seconds in which the wearer's hands are out of view (the dense timeline marks
        each segment); None for any other rig, where the question does not apply."""
        if d.get("_rig") != "ego_head":
            return None
        return union_seconds([(float(e["t_s"]), float(e["end_s"])) for e in d.get("event_labels") or []
                              if e.get("hands_visible") is False and e.get("t_s") is not None
                              and e.get("end_s") is not None])
