"""Rule-based arabizi normalisation.

The tests that matter most are the NEGATIVE ones: prices and sizes must
survive untouched. A corrupted price is the most expensive bug this bot can
ship, so those cases are asserted explicitly rather than assumed.
"""

import pytest

from app.nlp.arabizi import (
    build_search_query,
    looks_like_arabizi,
    normalize_arabizi,
)

# ── known darija words ARE mapped to Arabic ──────────────────────────────────

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("chhal", "بشحال"),
        ("ta3", "تاع"),
        ("3andek", "عندك"),
        ("7abit", "حابيت"),
        ("9adach", "قداش"),
        ("kayen", "كاين"),
        ("wach", "واش"),
        ("bezzaf", "بزاف"),
    ],
)
def test_known_darija_words_mapped_to_arabic(raw, expected):
    """Whole-word mapping, so the output is real Arabic that can match
    Arabic policy text — not a Latin/Arabic hybrid."""
    assert normalize_arabizi(raw) == expected


def test_case_insensitive():
    assert normalize_arabizi("Chhal") == "بشحال"
    assert normalize_arabizi("KAYEN") == "كاين"


def test_unknown_words_pass_through_unchanged():
    """Graceful degradation: a darija word not in the lexicon is left alone
    rather than guessed at."""
    assert normalize_arabizi("ma3jebnich") == "ma3jebnich"
    assert normalize_arabizi("labsstha") == "labsstha"


# ── THE CRITICAL CASES: numbers must survive ─────────────────────────────────

@pytest.mark.parametrize(
    "raw",
    [
        "3500 DA",                       # price
        "taille 42",                     # shoe size
        "size 38",                       # size
        "3 semaines",                    # quantity
        "2 pieces",                      # quantity
        "43",                            # bare number
        "7200",                          # price, no unit
        "numero 0555123456",             # phone number
        "9",                             # single digit alone
    ],
)
def test_standalone_numbers_never_touched(raw):
    """Numbers are safe by construction: a number is never a lexicon key, so
    prices, sizes and phone numbers cannot be rewritten.

    NOTE: a sentence mixing a price with darija ("3500 DA bezzaf") is not
    tested here by exact string equality, because `bezzaf` correctly DOES
    convert. That mixed case has its own test below."""
    assert normalize_arabizi(raw) == raw


def test_price_and_darija_in_same_message():
    """The hardest real case: '3500' is a price, 'bezzaf' is darija."""
    out = normalize_arabizi("3500 DA bezzaf, ma3andi ghir 3000")
    assert "3500" in out          # price intact
    assert "3000" in out          # price intact
    assert "بزاف" in out          # darija mapped
    assert "غير" in out           # darija mapped
    assert "ع500" not in out      # the corruption we are guarding against


def test_size_number_and_darija_in_same_message():
    out = normalize_arabizi("kayen chi taille 38 fel veste? ana 3andi 38")
    assert out.count("38") == 2   # both sizes intact
    assert "كاين" in out
    assert "عندي" in out
    assert "taille" in out        # French preserved


# ── French and product names must pass through ───────────────────────────────

@pytest.mark.parametrize(
    "phrase",
    [
        "la veste rouge",
        "bonjour",
        "livraison gratuite",
        "Botte Cuir Noir",
        "SVP je veux la veste",
        "Tizi Ouzou",
        "Yalidine",
    ],
)
def test_french_and_product_names_untouched(phrase):
    """Product names are the highest-signal tokens in the message. If
    normalisation mangles them, retrieval gets worse, not better."""
    assert normalize_arabizi(phrase) == phrase


def test_tokens_outside_lexicon_untouched():
    """Nothing needs an explicit protect-list: not being a lexicon key is
    already protection."""
    assert normalize_arabizi("mp3") == "mp3"
    assert normalize_arabizi("h24") == "h24"
    assert normalize_arabizi("Yalidine") == "Yalidine"


# ── Arabic script input is already normalised ────────────────────────────────

def test_arabic_script_unchanged():
    arabic = "بشحال الفيست الحمرا"
    assert normalize_arabizi(arabic) == arabic


def test_emoji_and_punctuation_unchanged():
    assert normalize_arabizi("👍") == "👍"
    assert normalize_arabizi("?") == "?"


# ── properties ───────────────────────────────────────────────────────────────

def test_idempotent():
    """Running twice must equal running once — no Latin+digit tokens remain."""
    once = normalize_arabizi("3andek sac beige? chhal? 2100 DA")
    assert normalize_arabizi(once) == once


def test_empty_and_none_safe():
    assert normalize_arabizi("") == ""
    assert normalize_arabizi(None) is None


def test_detection_flags_darija_only():
    assert looks_like_arabizi("ta3 la veste") is True
    assert looks_like_arabizi("chhal taman") is True
    assert looks_like_arabizi("la veste rouge") is False
    assert looks_like_arabizi("بشحال") is False
    assert looks_like_arabizi("👍") is False
    # A bare price is NOT darija — this keeps the cheap path cheap.
    assert looks_like_arabizi("3500 DA") is False


# ── the search query keeps both forms ────────────────────────────────────────

def test_search_query_keeps_original_and_normalised():
    q = build_search_query("chhal taman ta3 la veste rouge")
    assert "la veste rouge" in q    # French product name preserved for BM25
    assert "chhal" in q             # original kept
    assert "بشحال" in q             # Arabic form added for Arabic policy text
    assert "تاع" in q


def test_search_query_no_duplication_when_nothing_to_convert():
    q = build_search_query("la veste rouge")
    assert q == "la veste rouge"    # not doubled


def test_search_query_empty_safe():
    assert build_search_query("") == ""
    assert build_search_query(None) == ""


# ═══════════════════════════════════════════════════════════════════════════
# strip_darija_stopwords — for the SQL product lookup
# ═══════════════════════════════════════════════════════════════════════════

from app.nlp.arabizi import is_darija_stopword, strip_darija_stopwords


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("kayen la veste rouge?", "veste rouge"),
        ("3andek sac beige? chhal?", "sac beige"),
        ("3andkom botte cuir noir taille 43 wela la", "botte cuir noir 43"),
        ("wach 3andkom des chemises blanches", "chemises blanches"),
        ("bghit nchri la veste", "veste"),
    ],
)
def test_darija_stripped_leaving_product_words(raw, expected):
    """The ILIKE pattern must be a SUBSTRING of the catalog name, so every
    darija and filler word has to go."""
    assert strip_darija_stopwords(raw) == expected


def test_arabizi_words_with_digits_are_recognised():
    """REGRESSION: the first token regex split `ta3` into `ta` + `3`, so no
    arabizi word matched the lexicon and stray digits leaked into the ILIKE
    pattern. Arabizi words are letters AND digits in one token."""
    out = strip_darija_stopwords("chhal taman ta3 la veste rouge?")
    # `taman` (price) was added to the lexicon after a live test showed it
    # leaking into the ILIKE pattern and blocking the match.
    assert out == "veste rouge"
    assert " 3 " not in out          # no orphan digit
    assert "ta " not in out          # no orphan letter fragment


def test_numbers_kept_because_sizes_matter():
    """A size or model number can be part of what the customer means."""
    assert "43" in strip_darija_stopwords("3andkom botte taille 43")
    assert "3500" in strip_darija_stopwords("3500 DA bezzaf")


def test_arabic_script_product_words_survive():
    """الفيست is French 'veste' written in Arabic — a PRODUCT word, not filler."""
    assert strip_darija_stopwords("بشحال الفيست الحمرا") == "الفيست الحمرا"


def test_never_returns_empty():
    """CRITICAL: an empty ILIKE pattern ('%%') matches EVERY product, so the
    bot would list the whole catalog. Returning the original is far safer."""
    for only_darija in ("chhal", "wach kayen", "salam 3aychek", "👍", "?"):
        assert strip_darija_stopwords(only_darija).strip() != ""


def test_stopword_detection():
    assert is_darija_stopword("chhal") is True
    assert is_darija_stopword("KAYEN") is True      # case-insensitive
    assert is_darija_stopword("bonjour") is True    # French filler
    assert is_darija_stopword("taille") is True     # French filler
    assert is_darija_stopword("veste") is False     # PRODUCT word
    assert is_darija_stopword("rouge") is False     # PRODUCT word
    assert is_darija_stopword("Yalidine") is False  # brand


def test_empty_safe():
    assert strip_darija_stopwords("") == ""
    assert strip_darija_stopwords(None) is None
