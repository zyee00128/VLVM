"""T: class-level text alignment between the target text and GD phrases.

TSP3D has no token output (`n_classes=1`, the query text is fed back), so the only
meaningful alignment is **class level**. GD phrases come from a caption built out
of a class list; a single box may therefore name several classes ("bed couch"),
which is *not* the target. `match_phrase` implements the strict single-class rule:
one named class is required, otherwise the phrase can never be a target match.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from .types import MatchResult

# VLVM's HM3D ObjectNav classes (VLFM's caption uses the longer MP3D list; the list
# is injectable through `build_alias_map` / `build_caption`).
DEFAULT_VOCAB: List[str] = ["chair", "bed", "potted plant", "toilet", "tv", "couch"]

DEFAULT_ALIASES: Dict[str, str] = {
    "sofa": "couch",
    "settee": "couch",
    "television": "tv",
    "television set": "tv",
    "monitor": "tv",
    "armchair": "chair",
    "seat": "chair",
    "wc": "toilet",
    "plant": "potted plant",
    "pot plant": "potted plant",
}


def normalize_phrase(phrase: str) -> str:
    """Lower-case, strip separators and collapse whitespace."""
    s = str(phrase).lower().strip().replace(".", " ").replace("_", " ")
    return " ".join(s.split())


def singularize(word: str) -> str:
    """Minimal English singulariser (aliases/plurals only; no NLP pipeline)."""
    w = word.strip()
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es") and not w.endswith("ses"):
        return w[:-2]
    if len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def build_alias_map(class_names: Sequence[str]) -> Dict[str, str]:
    """Map every alias / singular form of the class list onto its canonical name.

    `"tv"` as target text keeps `"tv"` as canonical name because the class list is
    used as the authority (`"chair|seat"` style entries map onto their first part).
    """
    alias: Dict[str, str] = dict(DEFAULT_ALIASES)
    for name in class_names:
        parts = [normalize_phrase(p) for p in str(name).split("|") if p.strip()]
        if not parts:
            continue
        for p in parts:
            alias[p] = parts[0]
            alias[singularize(p)] = parts[0]
    return alias


def canonicalize(phrase: str, alias_map: Dict[str, str]) -> Optional[str]:
    """Canonical class of a phrase (None when the phrase names no known class)."""
    p = normalize_phrase(phrase)
    if not p:
        return None
    if p in alias_map:
        return alias_map[p]
    if singularize(p) in alias_map:
        return alias_map[singularize(p)]
    hits = [a for a in alias_map if a and a in p]
    return alias_map[max(hits, key=len)] if hits else None


def phrase_classes(phrase: str, alias_map: Dict[str, str]) -> List[str]:
    """Distinct canonical classes named by the phrase (word-level alias hits)."""
    p = normalize_phrase(phrase)
    if not p:
        return []
    words = set(p.split())
    out: List[str] = []
    for alias, canon in alias_map.items():
        if not alias:
            continue
        hit = (alias in words) or ((" " in alias) and (alias in p))
        if hit and canon not in out:
            out.append(canon)
    return out


def token_hits(phrase: str, target: str) -> float:
    """Fraction of target tokens present in the phrase (partial-match grade)."""
    p_tokens = set(normalize_phrase(phrase).split())
    t_tokens = [singularize(t) for t in normalize_phrase(target).split()]
    if not t_tokens:
        return 0.0
    return sum(1 for t in t_tokens if t in p_tokens or any(t in pt for pt in p_tokens)) / len(t_tokens)


def match_phrase(phrase: str, target_classes: Sequence[str], alias_map: Dict[str, str]) -> MatchResult:
    """Strict class-level match of one GD phrase against the target classes.

    * exactly one named class and it equals a target class -> `is_target=True`;
    * several named classes (a multi-class span) -> never a target match, and the
      partial-match strength is capped at 0.5 (weak "any entity" evidence only);
    * the canonical name is still reported for diagnostics / provenance.
    """
    p = normalize_phrase(phrase)
    named = phrase_classes(phrase, alias_map)
    canon: Optional[str] = named[0] if len(named) == 1 else None
    if not named and p:
        for cand in (p, singularize(p)):
            if cand in alias_map:
                canon = alias_map[cand]
                break

    for cls in target_classes:
        cls_canon = canonicalize(cls, alias_map) or normalize_phrase(cls)
        if canon is not None and canon == cls_canon:
            return MatchResult(canon=canon, strength=1.0, is_target=True, n_named=len(named))

    best = max((token_hits(phrase, cls) for cls in target_classes), default=0.0)
    if len(named) > 1:
        best = min(best, 0.5)
    return MatchResult(
        canon=canon if (canon is not None and best >= 0.5) else None,
        strength=best,
        is_target=False,
        n_named=len(named),
    )


def build_caption(target_classes: Sequence[str], style: str = "vocab",
                  vocab: Optional[Sequence[str]] = None) -> str:
    """caption string for the GD server.

    `"vocab"` = the whole class list (VLFM's caption style, needed so that other
    classes can also be recognised); `"target"` = only the target classes.
    """
    if style == "target":
        tokens = [normalize_phrase(c) for c in target_classes]
    else:
        tokens = [normalize_phrase(c) for c in (vocab if vocab is not None else DEFAULT_VOCAB)]
    return (" . ".join(tokens) + " .") if tokens else "object ."


def canonical_pairs(class_names: Sequence[str]) -> List[Tuple[str, str]]:
    """Debug helper: (alias, canonical) pairs of a class list."""
    alias = build_alias_map(class_names)
    return sorted(alias.items())
