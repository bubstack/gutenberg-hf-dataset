# Gutenberg HuggingFace Dataset Pipeline - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a pipeline that converts the full Project Gutenberg corpus into a multi-config HuggingFace dataset with rich metadata, chapter/paragraph chunking, and automated weekly updates via GitHub Actions.

**Architecture:** Two-stage pipeline. Stage 1 runs locally to do the initial full build of 75k+ books. Stage 2 is a GitHub Actions weekly cron that diffs the PG catalog, downloads new books, processes them, and appends to the HF dataset. Three dataset configs: `books`, `chapters`, `paragraphs`.

**Tech Stack:** Python 3.11+, gutenbergpy, lxml, pandas, pyarrow, datasets, huggingface_hub. GitHub Actions for CI/CD.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`
- Create: `README.md`

**Step 1: Initialize git repo and create pyproject.toml**

```bash
cd /Users/zakkeown/Documents/Datasets/gutenberg
git init
```

Create `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "gutenberg-hf-dataset"
version = "0.1.0"
description = "Pipeline to build and maintain a HuggingFace dataset from Project Gutenberg"
requires-python = ">=3.11"
license = "Apache-2.0"
dependencies = [
    "gutenbergpy>=0.3.5",
    "lxml>=5.0",
    "pandas>=2.0",
    "pyarrow>=15.0",
    "datasets>=2.18",
    "huggingface-hub>=0.21",
    "requests>=2.31",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create .gitignore**

```gitignore
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.venv/
venv/
*.parquet
data/raw/
data/output/
.DS_Store
```

**Step 3: Create empty init files and README**

```bash
mkdir -p src tests data
touch src/__init__.py tests/__init__.py
```

Create `README.md`:

```markdown
# Gutenberg HuggingFace Dataset Pipeline

Pipeline to build and maintain a comprehensive HuggingFace dataset from Project Gutenberg.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Full build (local, one-time)
python -m src.build --full

# Incremental update (used by GitHub Actions)
python -m src.build --incremental
```
```

**Step 4: Install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Step 5: Commit**

```bash
git add pyproject.toml .gitignore src/__init__.py tests/__init__.py README.md
git commit -m "feat: scaffold project with dependencies and structure"
```

---

### Task 2: RDF Metadata Parser

**Files:**
- Create: `src/metadata.py`
- Create: `tests/test_metadata.py`
- Create: `tests/fixtures/pg2701.rdf` (test fixture)

**Step 1: Create a test RDF fixture**

Download a real RDF file to use as a test fixture:

```bash
mkdir -p tests/fixtures
curl -o tests/fixtures/pg2701.rdf https://www.gutenberg.org/cache/epub/2701/pg2701.rdf
```

**Step 2: Write the failing test**

Create `tests/test_metadata.py`:

```python
from pathlib import Path
from src.metadata import parse_rdf

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_rdf_extracts_title():
    meta = parse_rdf(FIXTURES / "pg2701.rdf")
    assert meta["title"] == "Moby Dick; Or, The Whale"


def test_parse_rdf_extracts_author():
    meta = parse_rdf(FIXTURES / "pg2701.rdf")
    assert meta["author"] == "Melville, Herman"


def test_parse_rdf_extracts_author_years():
    meta = parse_rdf(FIXTURES / "pg2701.rdf")
    assert meta["author_birth_year"] == 1819
    assert meta["author_death_year"] == 1891


def test_parse_rdf_extracts_subjects():
    meta = parse_rdf(FIXTURES / "pg2701.rdf")
    assert "Whaling -- Fiction" in meta["subjects"]
    assert isinstance(meta["subjects"], list)


def test_parse_rdf_extracts_bookshelves():
    meta = parse_rdf(FIXTURES / "pg2701.rdf")
    assert "Best Books Ever Listings" in meta["bookshelves"]
    assert isinstance(meta["bookshelves"], list)


def test_parse_rdf_extracts_locc():
    meta = parse_rdf(FIXTURES / "pg2701.rdf")
    assert meta["locc"] == "PS"


def test_parse_rdf_extracts_language():
    meta = parse_rdf(FIXTURES / "pg2701.rdf")
    assert meta["language"] == "en"


def test_parse_rdf_extracts_release_date():
    meta = parse_rdf(FIXTURES / "pg2701.rdf")
    assert meta["release_date"] == "2001-07-01"


def test_parse_rdf_extracts_rights():
    meta = parse_rdf(FIXTURES / "pg2701.rdf")
    assert "Public domain" in meta["rights"]


def test_parse_rdf_extracts_id():
    meta = parse_rdf(FIXTURES / "pg2701.rdf")
    assert meta["id"] == "2701"


def test_parse_rdf_extracts_contributors():
    meta = parse_rdf(FIXTURES / "pg2701.rdf")
    assert isinstance(meta["contributors"], list)


def test_parse_rdf_extracts_summary():
    meta = parse_rdf(FIXTURES / "pg2701.rdf")
    # marc520 field -- may or may not exist for all books
    assert isinstance(meta["summary"], (str, type(None)))
```

**Step 3: Run tests to verify they fail**

```bash
pytest tests/test_metadata.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.metadata'`

**Step 4: Write minimal implementation**

Create `src/metadata.py`:

```python
from pathlib import Path
from lxml import etree

NAMESPACES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dcterms": "http://purl.org/dc/terms/",
    "pgterms": "http://www.gutenberg.org/2009/pgterms/",
    "dcam": "http://purl.org/dc/dcam/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
}


def _xpath_text(element, xpath: str, default=None) -> str | None:
    """Extract first text match or return default."""
    results = element.xpath(xpath, namespaces=NAMESPACES)
    if results:
        return str(results[0]).strip()
    return default


def _xpath_texts(element, xpath: str) -> list[str]:
    """Extract all text matches as a list."""
    results = element.xpath(xpath, namespaces=NAMESPACES)
    return [str(r).strip() for r in results]


def _xpath_int(element, xpath: str, default=None) -> int | None:
    """Extract first text match as int or return default."""
    text = _xpath_text(element, xpath)
    if text is not None:
        try:
            return int(text)
        except ValueError:
            return default
    return default


def parse_rdf(rdf_path: Path) -> dict:
    """Parse a Project Gutenberg RDF file and return structured metadata."""
    tree = etree.parse(str(rdf_path))
    ebook_elements = tree.xpath("//pgterms:ebook", namespaces=NAMESPACES)
    if not ebook_elements:
        raise ValueError(f"No pgterms:ebook found in {rdf_path}")
    ebook = ebook_elements[0]

    # Extract eBook ID from rdf:about attribute (e.g., "ebooks/2701")
    about = ebook.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about", "")
    ebook_id = about.split("/")[-1] if "/" in about else about

    # Subjects: only LCSH (not LCC)
    subjects = _xpath_texts(
        ebook,
        'dcterms:subject/rdf:Description'
        '[dcam:memberOf/@rdf:resource="http://purl.org/dc/terms/LCSH"]'
        '/rdf:value/text()',
    )

    # LoC Classification: LCC
    locc_list = _xpath_texts(
        ebook,
        'dcterms:subject/rdf:Description'
        '[dcam:memberOf/@rdf:resource="http://purl.org/dc/terms/LCC"]'
        '/rdf:value/text()',
    )

    # Bookshelves
    bookshelves = _xpath_texts(
        ebook,
        "pgterms:bookshelf/rdf:Description/rdf:value/text()",
    )

    # Contributors from marc508
    contributors_text = _xpath_text(ebook, "pgterms:marc508/text()")
    contributors = (
        [c.strip() for c in contributors_text.split(",")]
        if contributors_text
        else []
    )

    return {
        "id": ebook_id,
        "title": _xpath_text(ebook, "dcterms:title/text()", ""),
        "author": _xpath_text(
            ebook,
            "dcterms:creator/pgterms:agent/pgterms:name/text()",
            "",
        ),
        "author_birth_year": _xpath_int(
            ebook,
            "dcterms:creator/pgterms:agent/pgterms:birthdate/text()",
        ),
        "author_death_year": _xpath_int(
            ebook,
            "dcterms:creator/pgterms:agent/pgterms:deathdate/text()",
        ),
        "contributors": contributors,
        "subjects": subjects,
        "bookshelves": bookshelves,
        "locc": locc_list[0] if locc_list else "",
        "language": _xpath_text(
            ebook,
            "dcterms:language/rdf:Description/rdf:value/text()",
            "",
        ),
        "release_date": _xpath_text(ebook, "dcterms:issued/text()", ""),
        "rights": _xpath_text(ebook, "dcterms:rights/text()", ""),
        "summary": _xpath_text(ebook, "pgterms:marc520/text()"),
    }
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_metadata.py -v
```

Expected: All PASS

**Step 6: Commit**

```bash
git add src/metadata.py tests/test_metadata.py tests/fixtures/pg2701.rdf
git commit -m "feat: add RDF metadata parser with tests"
```

---

### Task 3: Text Cleaning (Header/Footer Stripping)

**Files:**
- Create: `src/clean.py`
- Create: `tests/test_clean.py`
- Create: `tests/fixtures/raw_sample.txt` (test fixture)

**Step 1: Create a test fixture**

Download a real Gutenberg text to use as a fixture:

```bash
curl -o tests/fixtures/raw_sample.txt https://www.gutenberg.org/cache/epub/2701/pg2701.txt
```

**Step 2: Write the failing test**

Create `tests/test_clean.py`:

```python
from pathlib import Path
from src.clean import strip_gutenberg_headers

FIXTURES = Path(__file__).parent / "fixtures"


def test_strip_headers_removes_preamble():
    raw = FIXTURES / "raw_sample.txt"
    text = strip_gutenberg_headers(raw.read_bytes())
    assert not text.startswith("The Project Gutenberg")
    assert "Project Gutenberg" not in text[:200]


def test_strip_headers_removes_footer():
    raw = FIXTURES / "raw_sample.txt"
    text = strip_gutenberg_headers(raw.read_bytes())
    assert "END OF THE PROJECT GUTENBERG" not in text
    assert "*** END OF" not in text


def test_strip_headers_preserves_content():
    raw = FIXTURES / "raw_sample.txt"
    text = strip_gutenberg_headers(raw.read_bytes())
    # Moby Dick should contain "Call me Ishmael"
    assert "Call me Ishmael" in text


def test_strip_headers_returns_string():
    raw = FIXTURES / "raw_sample.txt"
    text = strip_gutenberg_headers(raw.read_bytes())
    assert isinstance(text, str)


def test_strip_headers_handles_encoding_fallback():
    # Latin-1 encoded bytes that aren't valid UTF-8
    raw = "Héllo wörld".encode("latin-1")
    # Should not raise, should return a string
    text = strip_gutenberg_headers(raw)
    assert isinstance(text, str)
```

**Step 3: Run tests to verify they fail**

```bash
pytest tests/test_clean.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 4: Write minimal implementation**

Create `src/clean.py`:

```python
import gutenbergpy.textget


def strip_gutenberg_headers(raw_bytes: bytes) -> str:
    """Strip Project Gutenberg headers/footers and return clean text as str.

    Args:
        raw_bytes: Raw book content as bytes (UTF-8 or other encoding).

    Returns:
        Cleaned text as a Python string.
    """
    # gutenbergpy.textget.strip_headers expects and returns bytes
    try:
        clean_bytes = gutenbergpy.textget.strip_headers(raw_bytes)
    except Exception:
        # If strip_headers fails, fall back to raw text
        clean_bytes = raw_bytes

    # Decode to string with fallback
    try:
        return clean_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return clean_bytes.decode("latin-1")
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_clean.py -v
```

Expected: All PASS

**Step 6: Commit**

```bash
git add src/clean.py tests/test_clean.py tests/fixtures/raw_sample.txt
git commit -m "feat: add text cleaning with header/footer stripping"
```

---

### Task 4: Chapter Detection

**Files:**
- Create: `src/chunk.py`
- Create: `tests/test_chunk.py`

**Step 1: Write the failing tests**

Create `tests/test_chunk.py`:

```python
from src.chunk import detect_chapters, split_paragraphs


class TestDetectChapters:
    def test_detects_chapter_with_roman_numerals(self):
        text = (
            "Some intro text.\n\n"
            "CHAPTER I\n\n"
            "First chapter content here.\n\n"
            "CHAPTER II\n\n"
            "Second chapter content here."
        )
        chapters = detect_chapters(text)
        assert len(chapters) == 3  # intro + 2 chapters

    def test_detects_chapter_with_arabic_numerals(self):
        text = (
            "CHAPTER 1\n\n"
            "First chapter content.\n\n"
            "CHAPTER 2\n\n"
            "Second chapter content."
        )
        chapters = detect_chapters(text)
        assert len(chapters) == 2

    def test_detects_mixed_case_chapter(self):
        text = (
            "Chapter 1. The Beginning\n\n"
            "Some content.\n\n"
            "Chapter 2. The Middle\n\n"
            "More content."
        )
        chapters = detect_chapters(text)
        assert len(chapters) == 2

    def test_fallback_single_chapter_when_no_markers(self):
        text = "Just a plain text with no chapter markers.\n\nAnother paragraph."
        chapters = detect_chapters(text)
        assert len(chapters) == 1
        assert chapters[0]["chapter_index"] == 0
        assert chapters[0]["chapter_title"] is None

    def test_chapter_has_expected_fields(self):
        text = "CHAPTER I\n\nContent here."
        chapters = detect_chapters(text)
        assert "chapter_index" in chapters[0]
        assert "chapter_title" in chapters[0]
        assert "text" in chapters[0]

    def test_chapter_title_extracted(self):
        text = "CHAPTER I. Loomings\n\nCall me Ishmael."
        chapters = detect_chapters(text)
        assert "Loomings" in (chapters[0]["chapter_title"] or "")

    def test_detects_book_markers(self):
        text = (
            "BOOK I\n\nFirst book.\n\n"
            "BOOK II\n\nSecond book."
        )
        chapters = detect_chapters(text)
        assert len(chapters) == 2

    def test_detects_part_markers(self):
        text = (
            "PART ONE\n\nFirst part.\n\n"
            "PART TWO\n\nSecond part."
        )
        chapters = detect_chapters(text)
        assert len(chapters) == 2


class TestSplitParagraphs:
    def test_splits_on_double_newline(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        paragraphs = split_paragraphs(text)
        assert len(paragraphs) == 3

    def test_merges_short_fragments(self):
        text = "A\n\nThis is a real paragraph with enough content."
        paragraphs = split_paragraphs(text)
        # "A" is too short (<20 chars), should be merged
        assert len(paragraphs) == 1

    def test_strips_whitespace(self):
        text = "  First paragraph.  \n\n  Second paragraph.  "
        paragraphs = split_paragraphs(text)
        assert paragraphs[0] == "First paragraph."
        assert paragraphs[1] == "Second paragraph."

    def test_empty_text_returns_empty_list(self):
        assert split_paragraphs("") == []
        assert split_paragraphs("   ") == []

    def test_returns_strings(self):
        text = "Hello world.\n\nGoodbye world."
        paragraphs = split_paragraphs(text)
        assert all(isinstance(p, str) for p in paragraphs)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_chunk.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/chunk.py`:

```python
import re

# Chapter marker patterns, ordered by priority
CHAPTER_PATTERNS = [
    # "CHAPTER I", "CHAPTER 1", "CHAPTER I. Title", "CHAPTER 1 - Title"
    re.compile(
        r"^(CHAPTER\s+[IVXLCDM\d]+[.\s\-:]*.*?)$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "BOOK I", "BOOK ONE"
    re.compile(
        r"^(BOOK\s+[IVXLCDM\d]+[.\s\-:]*.*?)$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "PART ONE", "PART I", "PART 1"
    re.compile(
        r"^(PART\s+(?:[IVXLCDM\d]+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)[.\s\-:]*.*?)$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # "ACT I", "SCENE 1"
    re.compile(
        r"^((?:ACT|SCENE)\s+[IVXLCDM\d]+[.\s\-:]*.*?)$",
        re.IGNORECASE | re.MULTILINE,
    ),
]

MIN_PARAGRAPH_LENGTH = 20


def detect_chapters(text: str) -> list[dict]:
    """Detect chapter boundaries in text.

    Returns a list of dicts with keys:
        chapter_index: int (zero-based)
        chapter_title: str | None
        text: str
    """
    # Try each pattern until one produces 2+ matches
    for pattern in CHAPTER_PATTERNS:
        matches = list(pattern.finditer(text))
        if len(matches) >= 2:
            return _split_at_matches(text, matches)

    # Fallback: entire text is one chapter
    return [{"chapter_index": 0, "chapter_title": None, "text": text.strip()}]


def _split_at_matches(text: str, matches: list[re.Match]) -> list[dict]:
    """Split text at regex match positions into chapters."""
    chapters = []

    # Content before the first match (if any, non-trivial)
    pre = text[: matches[0].start()].strip()
    if len(pre) > MIN_PARAGRAPH_LENGTH:
        chapters.append(
            {"chapter_index": 0, "chapter_title": None, "text": pre}
        )

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        chapters.append(
            {
                "chapter_index": len(chapters),
                "chapter_title": title,
                "text": body,
            }
        )

    return chapters


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on double newlines.

    Merges fragments shorter than MIN_PARAGRAPH_LENGTH with the next paragraph.
    """
    if not text or not text.strip():
        return []

    raw_parts = re.split(r"\n\s*\n", text)
    raw_parts = [p.strip() for p in raw_parts if p.strip()]

    if not raw_parts:
        return []

    # Merge short fragments with the following paragraph
    merged = []
    buffer = ""
    for part in raw_parts:
        if buffer:
            part = buffer + " " + part
            buffer = ""
        if len(part) < MIN_PARAGRAPH_LENGTH:
            buffer = part
        else:
            merged.append(part)

    # Flush remaining buffer
    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)

    return merged
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_chunk.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add src/chunk.py tests/test_chunk.py
git commit -m "feat: add chapter detection and paragraph splitting"
```

---

### Task 5: Download Module

**Files:**
- Create: `src/download.py`
- Create: `tests/test_download.py`

**Step 1: Write the failing tests**

Create `tests/test_download.py`:

```python
import csv
import gzip
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.download import (
    parse_catalog_csv,
    diff_catalogs,
    download_book_text,
    download_book_rdf,
)


def test_parse_catalog_csv(tmp_path):
    csv_path = tmp_path / "pg_catalog.csv"
    csv_path.write_text(
        "Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves\n"
        '1,Text,2001-01-01,"Title One","en","Author One","Subject","PS","Shelf"\n'
        '2,Text,2002-02-02,"Title Two","fr","Author Two","Subject","PQ","Shelf"\n'
    )
    catalog = parse_catalog_csv(csv_path)
    assert len(catalog) == 2
    assert catalog[0]["id"] == "1"
    assert catalog[1]["id"] == "2"


def test_diff_catalogs_finds_new_ids():
    old = [{"id": "1"}, {"id": "2"}]
    new = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
    new_ids = diff_catalogs(old, new)
    assert new_ids == {"3"}


def test_diff_catalogs_empty_old():
    old = []
    new = [{"id": "1"}, {"id": "2"}]
    new_ids = diff_catalogs(old, new)
    assert new_ids == {"1", "2"}


@patch("src.download.requests.get")
def test_download_book_text(mock_get, tmp_path):
    mock_response = MagicMock()
    mock_response.content = b"The Project Gutenberg eBook\nContent here."
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    path = download_book_text("12345", tmp_path)
    assert path.exists()
    assert path.read_bytes() == mock_response.content
    mock_get.assert_called_once()


@patch("src.download.requests.get")
def test_download_book_rdf(mock_get, tmp_path):
    mock_response = MagicMock()
    mock_response.content = b"<rdf>test</rdf>"
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    path = download_book_rdf("12345", tmp_path)
    assert path.exists()
    assert path.read_bytes() == mock_response.content
    mock_get.assert_called_once()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_download.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/download.py`:

```python
import csv
import gzip
import time
from pathlib import Path

import requests

PG_BASE = "https://www.gutenberg.org"
PG_FEEDS = f"{PG_BASE}/cache/epub/feeds"
RATE_LIMIT_SECONDS = 2


def download_catalog(dest_dir: Path) -> Path:
    """Download pg_catalog.csv.gz from PG feeds."""
    url = f"{PG_FEEDS}/pg_catalog.csv.gz"
    dest = dest_dir / "pg_catalog.csv.gz"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def parse_catalog_csv(csv_path: Path) -> list[dict]:
    """Parse a pg_catalog.csv file into a list of dicts.

    Handles both .csv and .csv.gz files.
    """
    if csv_path.suffix == ".gz":
        f = gzip.open(csv_path, "rt", encoding="utf-8")
    else:
        f = open(csv_path, "r", encoding="utf-8")

    with f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # The first column is "Text#" in the PG catalog
            book_id = row.get("Text#", row.get("id", ""))
            rows.append({"id": str(book_id).strip(), **row})
        return rows


def diff_catalogs(
    old_catalog: list[dict], new_catalog: list[dict]
) -> set[str]:
    """Return set of book IDs present in new_catalog but not old_catalog."""
    old_ids = {row["id"] for row in old_catalog}
    new_ids = {row["id"] for row in new_catalog}
    return new_ids - old_ids


def download_book_text(book_id: str, dest_dir: Path) -> Path:
    """Download a single book's plain text file."""
    url = f"{PG_BASE}/cache/epub/{book_id}/pg{book_id}.txt"
    dest = dest_dir / f"{book_id}.txt"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    time.sleep(RATE_LIMIT_SECONDS)
    return dest


def download_book_rdf(book_id: str, dest_dir: Path) -> Path:
    """Download a single book's RDF metadata file."""
    url = f"{PG_BASE}/cache/epub/{book_id}/pg{book_id}.rdf"
    dest = dest_dir / f"pg{book_id}.rdf"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    time.sleep(RATE_LIMIT_SECONDS)
    return dest


def download_bulk_texts(dest_dir: Path) -> Path:
    """Download the bulk txt-files.tar.zip archive (~10GB)."""
    url = f"{PG_FEEDS}/txt-files.tar.zip"
    dest = dest_dir / "txt-files.tar.zip"
    print(f"Downloading {url} (this will take a while)...")
    resp = requests.get(url, timeout=3600, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192 * 1024):
            f.write(chunk)
    return dest


def download_bulk_rdf(dest_dir: Path) -> Path:
    """Download the bulk rdf-files.tar.bz2 archive (~119MB)."""
    url = f"{PG_FEEDS}/rdf-files.tar.bz2"
    dest = dest_dir / "rdf-files.tar.bz2"
    print(f"Downloading {url}...")
    resp = requests.get(url, timeout=600, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192 * 1024):
            f.write(chunk)
    return dest
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_download.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add src/download.py tests/test_download.py
git commit -m "feat: add download module for catalog and book files"
```

---

### Task 6: Build Module (Parquet Generation)

**Files:**
- Create: `src/build.py`
- Create: `tests/test_build.py`

**Step 1: Write the failing tests**

Create `tests/test_build.py`:

```python
from pathlib import Path
from src.build import process_book, build_books_rows, build_chapters_rows, build_paragraphs_rows


def test_process_book_returns_expected_keys():
    meta = {
        "id": "1",
        "title": "Test",
        "author": "Author",
        "author_birth_year": None,
        "author_death_year": None,
        "contributors": [],
        "subjects": [],
        "bookshelves": [],
        "locc": "",
        "language": "en",
        "release_date": "2000-01-01",
        "rights": "Public domain in the USA.",
        "summary": None,
    }
    text = "CHAPTER 1\n\nFirst paragraph.\n\nSecond paragraph.\n\nCHAPTER 2\n\nThird paragraph."

    result = process_book(meta, text)
    assert "book_row" in result
    assert "chapter_rows" in result
    assert "paragraph_rows" in result


def test_process_book_sets_has_chapters():
    meta = {
        "id": "1", "title": "Test", "author": "Author",
        "author_birth_year": None, "author_death_year": None,
        "contributors": [], "subjects": [], "bookshelves": [],
        "locc": "", "language": "en", "release_date": "",
        "rights": "", "summary": None,
    }
    text_with_chapters = "CHAPTER 1\n\nContent.\n\nCHAPTER 2\n\nMore content."
    result = process_book(meta, text_with_chapters)
    assert result["book_row"]["has_chapters"] is True
    assert result["book_row"]["chapter_count"] == 2

    text_without = "Just plain text here.\n\nAnother paragraph."
    result2 = process_book(meta, text_without)
    assert result2["book_row"]["has_chapters"] is False
    assert result2["book_row"]["chapter_count"] == 1


def test_process_book_paragraph_rows_have_correct_fields():
    meta = {
        "id": "42", "title": "Test", "author": "Author",
        "author_birth_year": None, "author_death_year": None,
        "contributors": [], "subjects": [], "bookshelves": [],
        "locc": "", "language": "en", "release_date": "",
        "rights": "", "summary": None,
    }
    text = "A substantial first paragraph with enough text.\n\nA substantial second paragraph with enough text."
    result = process_book(meta, text)
    for row in result["paragraph_rows"]:
        assert "id" in row
        assert "chapter_index" in row
        assert "paragraph_index" in row
        assert "text" in row
        assert row["id"] == "42"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_build.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/build.py`:

```python
from pathlib import Path

from src.chunk import detect_chapters, split_paragraphs


def process_book(meta: dict, text: str) -> dict:
    """Process a single book into rows for all three dataset configs.

    Args:
        meta: Metadata dict from parse_rdf().
        text: Cleaned text (headers stripped).

    Returns:
        Dict with keys 'book_row', 'chapter_rows', 'paragraph_rows'.
    """
    chapters = detect_chapters(text)
    has_chapters = len(chapters) > 1 or (
        len(chapters) == 1 and chapters[0]["chapter_title"] is not None
    )

    book_row = {
        **meta,
        "has_chapters": has_chapters,
        "chapter_count": len(chapters),
        "text": text,
    }

    chapter_rows = []
    paragraph_rows = []

    for chapter in chapters:
        chapter_rows.append(
            {
                "id": meta["id"],
                "chapter_index": chapter["chapter_index"],
                "chapter_title": chapter["chapter_title"],
                "text": chapter["text"],
            }
        )

        paragraphs = split_paragraphs(chapter["text"])
        for para_idx, para_text in enumerate(paragraphs):
            paragraph_rows.append(
                {
                    "id": meta["id"],
                    "chapter_index": chapter["chapter_index"],
                    "paragraph_index": para_idx,
                    "text": para_text,
                }
            )

    return {
        "book_row": book_row,
        "chapter_rows": chapter_rows,
        "paragraph_rows": paragraph_rows,
    }
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_build.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add src/build.py tests/test_build.py
git commit -m "feat: add build module for processing books into dataset rows"
```

---

### Task 7: Upload Module (HuggingFace Hub)

**Files:**
- Create: `src/upload.py`
- Create: `tests/test_upload.py`

**Step 1: Write the failing tests**

Create `tests/test_upload.py`:

```python
from unittest.mock import patch, MagicMock
from src.upload import upload_dataset


@patch("src.upload.Dataset")
def test_upload_dataset_calls_push_to_hub_for_each_config(mock_dataset_cls):
    mock_ds = MagicMock()
    mock_dataset_cls.from_dict.return_value = mock_ds

    books = [{"id": "1", "title": "Test", "text": "Content"}]
    chapters = [{"id": "1", "chapter_index": 0, "text": "Content"}]
    paragraphs = [{"id": "1", "chapter_index": 0, "paragraph_index": 0, "text": "Content"}]

    upload_dataset("test-user/test-repo", books, chapters, paragraphs)

    assert mock_dataset_cls.from_dict.call_count == 3
    assert mock_ds.push_to_hub.call_count == 3

    # Verify config names
    config_names = [
        call.kwargs["config_name"]
        for call in mock_ds.push_to_hub.call_args_list
    ]
    assert "books" in config_names
    assert "chapters" in config_names
    assert "paragraphs" in config_names
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_upload.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

Create `src/upload.py`:

```python
from datasets import Dataset


def _rows_to_columnar(rows: list[dict]) -> dict[str, list]:
    """Convert list of row dicts to columnar dict for Dataset.from_dict()."""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: [row[k] for row in rows] for k in keys}


def upload_dataset(
    repo_id: str,
    book_rows: list[dict],
    chapter_rows: list[dict],
    paragraph_rows: list[dict],
) -> None:
    """Upload three configs to HuggingFace Hub as a multi-config dataset.

    Args:
        repo_id: HuggingFace repo ID (e.g., "user/gutenberg-corpus").
        book_rows: List of book row dicts.
        chapter_rows: List of chapter row dicts.
        paragraph_rows: List of paragraph row dicts.
    """
    configs = {
        "books": book_rows,
        "chapters": chapter_rows,
        "paragraphs": paragraph_rows,
    }

    for config_name, rows in configs.items():
        ds = Dataset.from_dict(_rows_to_columnar(rows))
        ds.push_to_hub(
            repo_id,
            config_name=config_name,
            split="train",
            private=False,
        )
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_upload.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add src/upload.py tests/test_upload.py
git commit -m "feat: add upload module for multi-config HF dataset push"
```

---

### Task 8: Full Build Script (CLI Entry Point)

**Files:**
- Create: `src/__main__.py`
- Modify: `src/build.py` (add full_build and incremental_build functions)

**Step 1: Write the full build orchestrator**

Add to `src/build.py` (append after the existing `process_book` function):

```python
import tarfile
import bz2
import zipfile
import json
import logging

from src.download import (
    download_catalog,
    download_bulk_rdf,
    download_bulk_texts,
    parse_catalog_csv,
    download_book_text,
    download_book_rdf,
    diff_catalogs,
)
from src.metadata import parse_rdf
from src.clean import strip_gutenberg_headers
from src.upload import upload_dataset

logger = logging.getLogger(__name__)


def full_build(repo_id: str, data_dir: Path) -> None:
    """Run the full build: download everything, process, upload."""
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download bulk files
    logger.info("Downloading catalog...")
    catalog_path = download_catalog(raw_dir)
    catalog = parse_catalog_csv(catalog_path)
    logger.info(f"Catalog has {len(catalog)} entries")

    logger.info("Downloading RDF metadata archive...")
    rdf_archive = download_bulk_rdf(raw_dir)

    logger.info("Downloading text archive...")
    txt_archive = download_bulk_texts(raw_dir)

    # 2. Extract RDF files
    logger.info("Extracting RDF files...")
    rdf_dir = raw_dir / "rdf"
    rdf_dir.mkdir(exist_ok=True)
    with tarfile.open(rdf_archive, "r:bz2") as tar:
        tar.extractall(path=rdf_dir, filter="data")

    # 3. Extract text files
    logger.info("Extracting text files...")
    txt_dir = raw_dir / "txt"
    txt_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(txt_archive, "r") as zf:
        zf.extractall(path=txt_dir)

    # 4. Process all books
    all_book_rows = []
    all_chapter_rows = []
    all_paragraph_rows = []
    errors = []

    for entry in catalog:
        book_id = entry["id"]
        if not book_id or not book_id.isdigit():
            continue

        try:
            # Find RDF file (structure: cache/epub/{id}/pg{id}.rdf)
            rdf_path = rdf_dir / "cache" / "epub" / book_id / f"pg{book_id}.rdf"
            if not rdf_path.exists():
                logger.warning(f"No RDF for book {book_id}, skipping")
                continue

            meta = parse_rdf(rdf_path)

            # Find text file
            txt_path = txt_dir / f"{book_id}" / f"pg{book_id}.txt"
            if not txt_path.exists():
                # Try alternative path patterns
                alt_paths = list(txt_dir.rglob(f"pg{book_id}.txt"))
                if alt_paths:
                    txt_path = alt_paths[0]
                else:
                    logger.warning(f"No text for book {book_id}, skipping")
                    continue

            raw_bytes = txt_path.read_bytes()
            clean_text = strip_gutenberg_headers(raw_bytes)

            if not clean_text.strip():
                logger.warning(f"Empty text for book {book_id}, skipping")
                continue

            result = process_book(meta, clean_text)
            all_book_rows.append(result["book_row"])
            all_chapter_rows.extend(result["chapter_rows"])
            all_paragraph_rows.extend(result["paragraph_rows"])

        except Exception as e:
            errors.append((book_id, str(e)))
            logger.error(f"Error processing book {book_id}: {e}")

    logger.info(
        f"Processed {len(all_book_rows)} books, "
        f"{len(all_chapter_rows)} chapters, "
        f"{len(all_paragraph_rows)} paragraphs. "
        f"{len(errors)} errors."
    )

    # 5. Upload to HuggingFace
    logger.info(f"Uploading to {repo_id}...")
    upload_dataset(repo_id, all_book_rows, all_chapter_rows, all_paragraph_rows)
    logger.info("Upload complete!")

    # 6. Save catalog snapshot
    snapshot_dir = data_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(catalog_path, snapshot_dir / "pg_catalog.csv.gz")

    if errors:
        errors_path = data_dir / "build_errors.json"
        errors_path.write_text(json.dumps(errors, indent=2))
        logger.warning(f"Errors saved to {errors_path}")


def incremental_build(repo_id: str, data_dir: Path) -> None:
    """Run incremental update: diff catalog, process new books, append."""
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = data_dir / "snapshots"

    # 1. Download fresh catalog
    logger.info("Downloading fresh catalog...")
    new_catalog_path = download_catalog(raw_dir)
    new_catalog = parse_catalog_csv(new_catalog_path)

    # 2. Load previous snapshot
    old_snapshot = snapshot_dir / "pg_catalog.csv.gz"
    if old_snapshot.exists():
        old_catalog = parse_catalog_csv(old_snapshot)
    else:
        logger.warning("No previous snapshot found, treating all as new")
        old_catalog = []

    # 3. Diff
    new_ids = diff_catalogs(old_catalog, new_catalog)
    logger.info(f"Found {len(new_ids)} new books")

    if not new_ids:
        logger.info("No new books, nothing to do")
        return

    # 4. Download and process new books
    all_book_rows = []
    all_chapter_rows = []
    all_paragraph_rows = []
    errors = []

    for book_id in sorted(new_ids):
        try:
            rdf_path = download_book_rdf(book_id, raw_dir)
            meta = parse_rdf(rdf_path)

            txt_path = download_book_text(book_id, raw_dir)
            raw_bytes = txt_path.read_bytes()
            clean_text = strip_gutenberg_headers(raw_bytes)

            if not clean_text.strip():
                logger.warning(f"Empty text for book {book_id}, skipping")
                continue

            result = process_book(meta, clean_text)
            all_book_rows.append(result["book_row"])
            all_chapter_rows.extend(result["chapter_rows"])
            all_paragraph_rows.extend(result["paragraph_rows"])

        except Exception as e:
            errors.append((book_id, str(e)))
            logger.error(f"Error processing book {book_id}: {e}")

    logger.info(f"Processed {len(all_book_rows)} new books")

    # 5. Upload (append)
    if all_book_rows:
        logger.info(f"Uploading {len(all_book_rows)} new books to {repo_id}...")
        upload_dataset(repo_id, all_book_rows, all_chapter_rows, all_paragraph_rows)

    # 6. Update snapshot
    snapshot_dir.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(new_catalog_path, old_snapshot)
    logger.info("Snapshot updated")
```

**Step 2: Create CLI entry point**

Create `src/__main__.py`:

```python
import argparse
import logging
from pathlib import Path

from src.build import full_build, incremental_build

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="Build Gutenberg HuggingFace dataset"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full build (downloads everything, ~10GB)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Run incremental update (only new books)",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo ID (e.g., user/gutenberg-corpus)",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Local data directory (default: ./data)",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if args.full:
        full_build(args.repo_id, data_dir)
    elif args.incremental:
        incremental_build(args.repo_id, data_dir)
    else:
        parser.error("Specify --full or --incremental")


if __name__ == "__main__":
    main()
```

**Step 3: Run all tests**

```bash
pytest tests/ -v
```

Expected: All PASS

**Step 4: Commit**

```bash
git add src/__main__.py src/build.py
git commit -m "feat: add CLI entry point and full/incremental build orchestration"
```

---

### Task 9: GitHub Actions Workflow

**Files:**
- Create: `.github/workflows/update.yml`

**Step 1: Create the workflow**

Create `.github/workflows/update.yml`:

```yaml
name: Weekly Gutenberg Update

on:
  schedule:
    # Every Sunday at 4:00 AM UTC
    - cron: '0 4 * * 0'
  workflow_dispatch:  # Allow manual trigger

permissions:
  contents: write
  issues: write

jobs:
  update:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e .

      - name: Run incremental update
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: python -m src --incremental --repo-id ${{ vars.HF_REPO_ID || 'your-username/gutenberg-corpus' }} --data-dir ./data

      - name: Commit updated catalog snapshot
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          if [ -f data/snapshots/pg_catalog.csv.gz ]; then
            git add data/snapshots/pg_catalog.csv.gz
            git diff --staged --quiet || git commit -m "chore: update catalog snapshot [skip ci]"
            git push
          fi

      - name: Create issue on failure
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `Weekly Gutenberg update failed (${new Date().toISOString().split('T')[0]})`,
              body: `The weekly incremental update workflow failed.\n\nRun: ${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`,
              labels: ['bug', 'automated']
            });
```

**Step 2: Commit**

```bash
mkdir -p .github/workflows
git add .github/workflows/update.yml
git commit -m "feat: add GitHub Actions weekly update workflow"
```

---

### Task 10: Create GitHub Remote and Set Secrets

**Step 1: Create the GitHub repo**

```bash
gh repo create gutenberg-hf-dataset --public --source=. --push --description "Pipeline to build and maintain a HuggingFace dataset from Project Gutenberg"
```

**Step 2: Set HF_TOKEN as a repo secret**

```bash
echo "$HF_TOKEN" | gh secret set HF_TOKEN
```

**Step 3: Set repo variable for HF repo ID**

```bash
gh variable set HF_REPO_ID --body "your-username/gutenberg-corpus"
```

(Replace `your-username/gutenberg-corpus` with the actual desired HF dataset name.)

**Step 4: Verify**

```bash
gh secret list
gh variable list
```

**Step 5: Commit and push any remaining changes**

```bash
git push
```

---

### Task 11: Integration Test with Small Batch

Before running the full 75k build, test with a small batch.

**Step 1: Write an integration test**

Create `tests/test_integration.py`:

```python
"""Integration test: process a single real book end-to-end."""

import pytest
from pathlib import Path
from src.download import download_book_text, download_book_rdf
from src.metadata import parse_rdf
from src.clean import strip_gutenberg_headers
from src.build import process_book


@pytest.mark.integration
def test_end_to_end_single_book(tmp_path):
    """Download, parse, clean, and process Moby Dick."""
    book_id = "2701"

    # Download
    rdf_path = download_book_rdf(book_id, tmp_path)
    txt_path = download_book_text(book_id, tmp_path)

    # Parse metadata
    meta = parse_rdf(rdf_path)
    assert meta["title"] == "Moby Dick; Or, The Whale"
    assert meta["author"] == "Melville, Herman"
    assert meta["language"] == "en"

    # Clean text
    raw = txt_path.read_bytes()
    text = strip_gutenberg_headers(raw)
    assert "Call me Ishmael" in text
    assert "Project Gutenberg" not in text[:200]

    # Process into rows
    result = process_book(meta, text)
    assert result["book_row"]["id"] == "2701"
    assert result["book_row"]["has_chapters"] is True
    assert result["book_row"]["chapter_count"] > 10  # Moby Dick has 135 chapters
    assert len(result["chapter_rows"]) > 10
    assert len(result["paragraph_rows"]) > 100
```

**Step 2: Run the integration test**

```bash
pytest tests/test_integration.py -v -m integration
```

Expected: PASS (requires network access, downloads ~1.3MB)

**Step 3: Add pytest marker config**

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = [
    "integration: tests that require network access",
]
```

**Step 4: Commit**

```bash
git add tests/test_integration.py pyproject.toml
git commit -m "test: add integration test for end-to-end single book processing"
git push
```

---

### Task 12: Run Full Local Build

This is the one-time heavy lift. Run locally.

**Step 1: Create the full build script**

Create `scripts/full_build.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:?Usage: full_build.sh <hf-repo-id>}"
DATA_DIR="${2:-./data}"

echo "Starting full build..."
echo "  HF Repo: $REPO_ID"
echo "  Data Dir: $DATA_DIR"

python -m src --full --repo-id "$REPO_ID" --data-dir "$DATA_DIR"

echo "Full build complete!"
```

```bash
chmod +x scripts/full_build.sh
```

**Step 2: Run it**

```bash
./scripts/full_build.sh your-username/gutenberg-corpus ./data
```

This will:
- Download ~10GB of text files + 119MB of RDF + 5MB catalog
- Process all 75k+ books
- Upload 3 Parquet configs to HuggingFace
- Save the catalog snapshot for future diffing

**Step 3: Commit the snapshot**

```bash
git add scripts/full_build.sh data/snapshots/pg_catalog.csv.gz
git commit -m "feat: add full build script; save initial catalog snapshot"
git push
```

---

## Task Summary

| Task | What | Depends On |
|------|------|-----------|
| 1 | Project scaffolding | - |
| 2 | RDF metadata parser | 1 |
| 3 | Text cleaning | 1 |
| 4 | Chapter detection + paragraph splitting | 1 |
| 5 | Download module | 1 |
| 6 | Build module (orchestrates 2-5) | 2, 3, 4 |
| 7 | Upload module | 1 |
| 8 | CLI entry point | 6, 7 |
| 9 | GitHub Actions workflow | 8 |
| 10 | Create GitHub remote + secrets | 9 |
| 11 | Integration test | 8 |
| 12 | Run full local build | 10, 11 |

Tasks 2, 3, 4, 5 are independent and can be parallelized. Task 7 is also independent of 2-5.
