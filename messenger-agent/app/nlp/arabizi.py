"""Arabizi/darija → Arabic script, rule-based. No LLM, no network, ~0 ms.

WHAT THIS IS FOR
    Improving RETRIEVAL only. The customer's original text is what the agent
    generates from — this output is used to build the search query.

    Rationale: BM25 scored 0.00 on darija queries against the English policy
    docs. Normalising the QUERY helps that. Normalising the message the agent
    replies to would mean a bad transliteration becomes a confidently wrong
    answer, with nothing in the logs to show why.

DESIGN NOTE — a rejected first attempt
    v1 substituted digits for letters (3→ع, 7→ح, 9→ق), so `ta3` → `taع`.
    That was wrong: the output is a hybrid token matching neither Latin nor
    Arabic text, so it did nothing for retrieval — the entire purpose.

    v2 (this file) maps WHOLE high-frequency darija words to their Arabic
    spelling. `chhal` → بشحال is a real Arabic token that can match Arabic
    policy text.

THE BUG THIS DESIGN AVOIDS BY CONSTRUCTION
    Algerian customers write prices and sizes as digits:
        "3500 DA bezzaf"   "taille 42"   "3 semaines"   "size 38"
    A digit-substituting normaliser turns "3500 DA" into "ع500 DA", and a bot
    that then quotes a wrong price is worse than one that never normalised.

    With a word lexicon this cannot happen: "3500" is not a dictionary key, so
    it is never rewritten. Number safety is structural, not a special case.
"""

from __future__ import annotations

import re

# ── word-level lexicon ───────────────────────────────────────────────────────
# Digit-only substitution was the first design and it was WRONG: `ta3` became
# `taع`, a hybrid token matching neither Latin nor Arabic text — useless for
# retrieval, which was the entire point.
#
# What actually helps is mapping whole high-frequency darija words to their
# Arabic spelling, so the query contains real Arabic tokens that can match
# Arabic policy text. Deterministic, ~0 ms, no API call, and easy to extend
# from real logs.
#
# Keep this list SHORT and HIGH-FREQUENCY. A long tail of rare words adds
# false matches without adding recall.
LEXICON = {
    # question words — the highest-signal tokens in a customer message
    "chhal": "بشحال", "chhall": "بشحال", "bechhal": "بشحال",
    "taman": "ثمن", "tamen": "ثمن", "thaman": "ثمن",
    "9adach": "قداش", "qadach": "قداش", "9adech": "قداش",
    "wach": "واش", "wech": "واش", "wachta": "وقتاش",
    "wqtach": "وقتاش", "waqtach": "وقتاش", "weqtach": "وقتاش",
    "kifach": "كيفاش", "kifech": "كيفاش",
    "win": "وين", "winta": "وينتا",
    "3lach": "علاش", "3lech": "علاش",
    # availability / possession
    "kayen": "كاين", "kayn": "كاين", "kain": "كاين",
    "makanch": "ماكانش", "makach": "ماكاش",
    "3andek": "عندك", "3andkom": "عندكم", "3andi": "عندي",
    "mazal": "مازال",
    # wanting / buying
    "bghit": "بغيت", "habit": "حابيت", "7abit": "حابيت",
    "nchri": "نشري", "nechri": "نشري",
    "najem": "نجم", "nqder": "نقدر", "n9der": "نقدر",
    "nbadel": "نبدل", "nrod": "نرد", "nrodha": "نردها",
    # common connectives / modifiers
    "ta3": "تاع", "t3": "تاع", "ta3i": "تاعي", "ta3ha": "تاعها",
    "m3a": "معا", "bezzaf": "بزاف", "chwiya": "شوية",
    "wela": "ولا", "ghir": "غير", "ghi": "غي",
    "dork": "دروك", "drok": "دروك",
    "sahit": "صحيت", "3aychek": "عيشك", "3aslama": "عسلامة",
    "salam": "سلام", "slm": "سلام",
    "machi": "ماشي", "hadik": "هاديك", "hada": "هذا",
    "ana": "انا", "rani": "راني", "rah": "راح",
    "tosel": "توصل", "tousel": "توصل",
}


def _lexicon_lookup(token: str) -> str | None:
    """Whole-token lookup, case-insensitive. Returns None when unknown."""
    return LEXICON.get(token.lower())


# Any Latin/digit token. Lexicon lookup decides whether it is darija; a token
# that is not in the lexicon is left exactly as written, which is what keeps
# French product names ("veste", "livraison") and every number intact.
_TOKEN = re.compile(r"[a-zA-Z0-9]+")


def looks_like_arabizi(text: str) -> bool:
    """True when at least one token is known darija.

    Cheap pre-check so callers skip the work on pure French, pure Arabic or
    emoji-only messages.
    """
    if not text:
        return False
    return any(_lexicon_lookup(t) for t in _TOKEN.findall(text))


def normalize_arabizi(text: str) -> str:
    """Map known darija tokens to Arabic. Everything else passes through.

    Numbers are safe by construction: "3500" is not in the lexicon, so it is
    never rewritten. Same for French words and product names.

        >>> normalize_arabizi("chhal taman ta3 la veste")
        'بشحال taman تاع la veste'
        >>> normalize_arabizi("3500 DA bezzaf")
        '3500 DA بزاف'
        >>> normalize_arabizi("taille 42")
        'taille 42'
    """
    if not text:
        return text
    return _TOKEN.sub(lambda m: _lexicon_lookup(m.group(0)) or m.group(0), text)


def build_search_query(text: str) -> str:
    """The retrieval query: original PLUS normalised form.

    Both, not either. Keeping the original preserves French product names
    ('la veste rouge') that a customer may have typed exactly as they appear
    in the catalog — those are often the highest-signal tokens in the whole
    message. Appending the normalised form adds the Arabic-script variants
    that match Arabic policy text.

    Concatenating is deliberately cruder than picking one: it cannot lose a
    term, and for BM25/embedding search a slightly redundant query costs far
    less than a missing keyword.
    """
    original = (text or "").strip()
    if not original:
        return ""
    normalised = normalize_arabizi(original)
    if normalised == original:
        return original
    return f"{original} {normalised}"


# ── stripping, for the SQL product lookup ────────────────────────────────────
# `normalize_arabizi` / `build_search_query` ADD Arabic words. That helps an
# embedding or BM25 search over Arabic policy prose.
#
# The product lookup is the opposite problem. `repo.find_products` runs:
#     Product.name ILIKE '%<query>%'
# against English/French catalog names. Any darija word in the query makes the
# LIKE pattern longer than the product name, so it matches NOTHING:
#     "kayen la veste rouge"  ILIKE  → 0 rows
#     "la veste rouge"        ILIKE  → 1 row
#
# So here we REMOVE darija instead of translating it, leaving just the product
# words. Same lexicon, inverted use.

# Words to drop from a product query. Every key of LEXICON is darija, so those
# come for free; these are the extra French/English filler words that are also
# never part of a product name.
_STOPWORDS_EXTRA = {
    # French filler
    "bonjour", "salut", "svp", "merci", "cv", "ca", "va", "je", "veux",
    "est", "ce", "que", "il", "y", "a", "combien", "prix", "le", "la", "les",
    "un", "une", "des", "du", "de", "pour", "avec", "et", "ou", "sur",
    "taille", "pointure", "couleur", "disponible", "dispo", "stock",
    "livraison", "commande", "acheter",
    # English filler
    "hi", "hello", "please", "thanks", "how", "much", "is", "the", "do",
    "you", "have", "want", "size", "price", "available", "in", "any", "got",
    # darija written in Arabic script (the LEXICON values, which appear when
    # a customer types Arabic directly rather than arabizi)
    "بشحال", "قداش", "واش", "وقتاش", "كيفاش", "وين", "علاش", "كاين",
    "ماكانش", "عندك", "عندكم", "عندي", "مازال", "بغيت", "حابيت",
    "نشري", "نجم", "نقدر", "نبدل", "تاع", "معا", "بزاف", "ولا", "غير",
    "سلام", "انا", "راني", "راح", "توصل", "هذا", "هاديك", "ماشي",
}

# Any word token, Latin or Arabic, WITH digits kept attached.
#
# BUG THIS FIXES: `[^\W\d_]+|\d+` splits `ta3` into `ta` + `3` and `3andek`
# into `3` + `andek`, so no arabizi word ever matched the lexicon and the
# stray digits leaked into the ILIKE pattern. Arabizi words are letters AND
# digits in one token, so the character class must include both.
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def is_darija_stopword(token: str) -> bool:
    """True when the token is darija or generic filler, not a product word."""
    lowered = token.lower()
    return lowered in LEXICON or lowered in _STOPWORDS_EXTRA


def strip_darija_stopwords(text: str) -> str:
    """Leave only the words that could plausibly be in a product name.

    For `repo.find_products`, whose ILIKE pattern must be a SUBSTRING of the
    catalog name. Darija and filler words are removed; product words, numbers
    and unknown words are kept.

        >>> strip_darija_stopwords("kayen la veste rouge?")
        'veste rouge'
        >>> strip_darija_stopwords("chhal taman ta3 sac beige")
        'taman sac beige'
        >>> strip_darija_stopwords("3andkom botte cuir noir taille 43")
        'botte cuir noir 43'

    SAFETY: if stripping would remove everything, the ORIGINAL is returned.
    An empty ILIKE pattern ('%%') matches every product, which would make the
    bot list the whole catalog — worse than finding nothing.
    """
    if not text:
        return text
    kept = [w for w in _WORD.findall(text) if not is_darija_stopword(w)]
    if not kept:
        return text.strip()
    return " ".join(kept)
