"""Download and preparation of the text corpora.

Sources (public domain):
  * Project Gutenberg -- https://www.gutenberg.org  (English, ``lang="en"``)
  * Wolne Lektury     -- https://wolnelektury.pl    (Polish,  ``lang="pl"``)

Each language has a disjoint training set (used to estimate the n-gram model)
and test set (from which the plaintexts to encrypt are drawn), so the
evaluation is not inflated by the model having "seen" the test texts.
"""
from __future__ import annotations

import random
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
N_LETTERS = 26
SPACE = 26           # index of the space symbol (mode with spaces kept)

BOOKS = {
    "en": {
        "train": {
            1342: "Austen, Pride and Prejudice",
            2701: "Melville, Moby Dick",
            1400: "Dickens, Great Expectations",
            2600: "Tolstoy, War and Peace",
            98: "Dickens, A Tale of Two Cities",
            768: "Bronte, Wuthering Heights",
        },
        "test": {
            11: "Carroll, Alice's Adventures in Wonderland",
            1661: "Doyle, The Adventures of Sherlock Holmes",
            84: "Shelley, Frankenstein",
            345: "Stoker, Dracula",
            120: "Stevenson, Treasure Island",
            174: "Wilde, The Picture of Dorian Gray",
        },
    },
    "pl": {
        "train": {
            "pan-tadeusz": "Mickiewicz, Pan Tadeusz",
            "lalka-tom-pierwszy": "Prus, Lalka, t. 1",
            "lalka-tom-drugi": "Prus, Lalka, t. 2",
            "quo-vadis": "Sienkiewicz, Quo vadis",
            "potop-tom-pierwszy": "Sienkiewicz, Potop, t. 1",
            "ogniem-i-mieczem-tom-pierwszy": "Sienkiewicz, Ogniem i mieczem, t. 1",
            "chlopi-czesc-pierwsza-jesien": "Reymont, Chłopi, cz. 1",
        },
        "test": {
            "w-pustyni-i-w-puszczy": "Sienkiewicz, W pustyni i w puszczy",
            "krzyzacy-tom-pierwszy": "Sienkiewicz, Krzyżacy, t. 1",
            "faraon-tom-pierwszy": "Prus, Faraon, t. 1",
            "przedwiosnie": "Żeromski, Przedwiośnie",
            "syzyfowe-prace": "Żeromski, Syzyfowe prace",
            "ziemia-obiecana-tom-pierwszy": "Reymont, Ziemia obiecana, t. 1",
        },
    },
}


def book_urls(lang: str, ident) -> list[str]:
    if lang == "en":
        return [f"https://www.gutenberg.org/cache/epub/{ident}/pg{ident}.txt",
                f"https://www.gutenberg.org/files/{ident}/{ident}-0.txt",
                f"https://www.gutenberg.org/ebooks/{ident}.txt.utf-8"]
    if lang == "pl":
        return [f"https://wolnelektury.pl/media/book/txt/{ident}.txt"]
    raise ValueError(f"unknown language: {lang}")


def download(lang: str, ident, force: bool = False, attempts: int = 4) -> Path:
    """Downloads a book (once) into ``data/raw/<lang>/`` and returns the path.
    The servers are sometimes overloaded (HTTP 504), so several attempts and
    several mirror URLs are tried."""
    path = RAW_DIR / lang / f"{ident}.txt"
    if path.exists() and path.stat().st_size > 1000 and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    error: Exception | None = None
    for attempt in range(attempts):
        for url in book_urls(lang, ident):
            request = urllib.request.Request(url, headers={"User-Agent": "gasub"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read()
                if len(payload) > 1000:
                    path.write_bytes(payload)
                    return path
            except (urllib.error.URLError, OSError) as e:
                error = e
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"could not download {lang}/{ident}: {error}")


def strip_boilerplate(text: str, lang: str) -> str:
    """Removes the publisher's header and footer (licence, metadata)."""
    if lang == "en":
        start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
        end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text)
        if start and end:
            text = text[start.end():end.start()]
    else:
        lines = text.splitlines()
        for i, line in enumerate(lines[:40]):
            if line.startswith("ISBN"):
                lines = lines[i + 1:]
                break
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("-----"):
                lines = lines[:i]
                break
        text = "\n".join(lines)
    return text


_TRANSLITERATION = str.maketrans({"ł": "l", "Ł": "L", "ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe",
                                  "Œ": "OE", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D"})


def normalize(text: str, keep_spaces: bool = False) -> str:
    """Reduces the text to the 26 capital letters A-Z (Polish diacritics are
    replaced by their base letters: ą->A, ś->S, ł->L etc.). With
    ``keep_spaces`` single spaces between words are kept as well;
    punctuation and digits are removed."""
    text = text.translate(_TRANSLITERATION)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).upper()
    if keep_spaces:
        text = re.sub(r"[^A-Z]+", " ", text)
        return text.strip()
    return re.sub(r"[^A-Z]", "", text)


def book_text(lang: str, ident, keep_spaces: bool = False) -> str:
    raw = download(lang, ident).read_text(encoding="utf-8", errors="ignore")
    return normalize(strip_boilerplate(raw, lang), keep_spaces)


def corpus(lang: str, subset: str = "train", keep_spaces: bool = False) -> str:
    """Concatenated, normalised texts of a subset ("train" or "test")."""
    sep = " " if keep_spaces else ""
    return sep.join(book_text(lang, ident, keep_spaces) for ident in BOOKS[lang][subset])


def fragment(text: str, length: int, rng: random.Random, keep_spaces: bool = False) -> str:
    """Random fragment of the text with the given length (in characters).
    With spaces kept the fragment starts at a word boundary."""
    if length > len(text):
        raise ValueError("fragment longer than the text")
    start = rng.randrange(len(text) - length + 1)
    if keep_spaces:
        sp = text.find(" ", start)
        if 0 <= sp < len(text) - length:
            start = sp + 1
    return text[start:start + length]


def random_test_fragment(lang: str, length: int, rng: random.Random,
                         keep_spaces: bool = False) -> tuple[str, str]:
    """Draws a book from the test set and a fragment from it; returns (fragment, book description)."""
    ident = rng.choice(list(BOOKS[lang]["test"]))
    return fragment(book_text(lang, ident, keep_spaces), length, rng, keep_spaces), BOOKS[lang]["test"][ident]


if __name__ == "__main__":
    for lang in BOOKS:
        for subset in ("train", "test"):
            n = sum(len(book_text(lang, i)) for i in BOOKS[lang][subset])
            print(f"{lang} {subset:6s}: {len(BOOKS[lang][subset])} books, {n:,} letters")
