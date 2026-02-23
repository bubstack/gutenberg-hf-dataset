# Project Gutenberg HuggingFace Dataset Pipeline - Design Document

**Date**: 2026-02-23
**Status**: Draft

## Problem Statement

Project Gutenberg contains 75,000+ public domain books across 60+ languages. Despite 10+ existing HuggingFace datasets, none offers the combination of:
- Full corpus coverage
- Rich structured metadata (not JSON blobs)
- RAG-friendly chunking (chapters + paragraphs)
- Regular automated updates

Every existing dataset was uploaded once and abandoned. The best metadata dataset (Despina) covers only 9% of the corpus. The most complete dataset (manu) has only book ID + raw text with no metadata.

## Goals

1. Comprehensive, regularly-updated HuggingFace dataset of all Project Gutenberg texts
2. Rich structured metadata extracted from RDF catalog
3. Three granularity levels: full books, chapters, paragraphs
4. Automated weekly pipeline to incorporate new books (~30/week)
5. All 60+ languages included

## Non-Goals

- Real-time updates (weekly is sufficient)
- Hosting images or non-text content from Gutenberg
- Copyright analysis per jurisdiction (we note US public domain status)
- Cleaning/normalizing OCR errors in the source texts

## Dataset Schema

### Config: `books` (one row per book)

| Column | Type | Description |
|---|---|---|
| `id` | string | Project Gutenberg eBook ID |
| `title` | string | Book title |
| `author` | string | Primary author (Last, First) |
| `author_birth_year` | int | Author birth year (null if unknown) |
| `author_death_year` | int | Author death year (null if unknown) |
| `contributors` | list[string] | Translators, editors, illustrators with role |
| `subjects` | list[string] | Subject headings |
| `bookshelves` | list[string] | Gutenberg bookshelves/categories |
| `locc` | string | Library of Congress Classification |
| `language` | string | ISO language code |
| `release_date` | string | Original PG release date (YYYY-MM-DD) |
| `rights` | string | Copyright status |
| `summary` | string | AI-generated summary (if available from PG) |
| `has_chapters` | bool | Whether chapter boundaries were detected |
| `chapter_count` | int | Number of detected chapters |
| `text` | string | Full book text, headers/footers stripped |

### Config: `chapters` (one row per chapter)

| Column | Type | Description |
|---|---|---|
| `id` | string | Project Gutenberg eBook ID |
| `chapter_index` | int | Zero-based chapter index |
| `chapter_title` | string | Detected chapter title (null if none) |
| `text` | string | Chapter text |

If no chapters are detected in a book, the entire text appears as a single chapter (index=0, title=null).

### Config: `paragraphs` (one row per paragraph)

| Column | Type | Description |
|---|---|---|
| `id` | string | Project Gutenberg eBook ID |
| `chapter_index` | int | Chapter this paragraph belongs to |
| `paragraph_index` | int | Zero-based index within the chapter |
| `text` | string | Paragraph text |

Paragraphs are split on double newlines. Short fragments (<20 chars) are merged with adjacent paragraphs.

## Architecture

### Two-Stage Pipeline

**Stage 1 - Initial Full Build (runs locally, one-time)**

```
pg_catalog.csv.gz (5.2 MB) ───┐
                               ├──> Python pipeline ──> 3 Parquet configs ──> HF Hub
rdf-files.tar.bz2 (119 MB) ───┤
                               │
txt-files.tar.zip (10 GB) ─────┘
```

1. Download bulk archives from `gutenberg.org/cache/epub/feeds/`
2. Parse CSV catalog for book IDs
3. Parse RDF files for rich metadata (lxml)
4. Extract .txt files, strip headers/footers (gutenbergpy)
5. Detect chapter boundaries (heuristic regex cascade)
6. Split paragraphs (double-newline)
7. Generate Parquet, upload to HuggingFace Hub

**Stage 2 - Incremental Updates (GitHub Actions, weekly)**

```
Sunday 4AM UTC cron ──> Download pg_catalog.csv.gz
                        ──> Diff against committed snapshot
                        ──> Download new books (.txt + .rdf)
                        ──> Process (strip, chunk, metadata)
                        ──> Append rows to HF dataset
                        ──> Commit updated catalog snapshot
```

1. Cron triggers weekly on Sundays
2. Download fresh catalog, diff against last snapshot
3. For new book IDs: fetch individual .txt and .rdf files
4. Process each book through the same pipeline
5. Append to existing HF dataset via `huggingface_hub`
6. Commit updated snapshot to GitHub repo
7. Open GitHub issue on failure

### Chapter Detection Strategy

Priority cascade of heuristics:

1. Regex: `^CHAPTER\s+[IVXLCDM\d]+`, `^Chapter\s+`, `^BOOK\s+`, `^PART\s+`, `^ACT\s+`, `^SCENE\s+`
2. All-caps short lines (<100 chars) surrounded by blank lines
3. Structural dividers: `* * *`, `---`
4. Fallback: entire book = single chapter (chapter_index=0)

## Technology Stack

- **Python 3.11+**
- `huggingface_hub` - HF dataset upload
- `datasets` - Parquet generation
- `gutenbergpy` - Gutenberg header/footer stripping
- `lxml` - RDF/XML metadata parsing
- `pandas` / `pyarrow` - data wrangling and Parquet I/O

## Repository Structure

```
gutenberg-hf-dataset/
├── .github/
│   └── workflows/
│       └── update.yml              # Weekly cron GitHub Action
├── src/
│   ├── download.py                 # Bulk download + individual book fetch
│   ├── metadata.py                 # RDF + CSV parsing
│   ├── clean.py                    # Header stripping, text cleaning
│   ├── chunk.py                    # Chapter detection + paragraph splitting
│   ├── build.py                    # Parquet generation for all 3 configs
│   └── upload.py                   # HF Hub push
├── scripts/
│   ├── full_build.sh               # One-time local full build
│   └── incremental_update.py       # Called by GitHub Actions
├── data/
│   └── pg_catalog_snapshot.csv.gz  # Last catalog state for diffing
├── tests/
├── pyproject.toml
└── README.md
```

## Licensing

- **Book texts**: Public domain in the United States
- **Dataset compilation + pipeline code**: Apache 2.0
- **Gutenberg headers/footers**: Stripped from all texts to avoid trademark requirements
- **Dataset name**: Avoids "Project Gutenberg" as a formal name; references it as the data source in the dataset card
- **Dataset card**: Includes jurisdiction disclaimer about copyright varying outside the US

## Key Design Decisions

1. **Strip all PG boilerplate** rather than keeping it. This avoids trademark entanglement and produces cleaner text for downstream use.

2. **Structured metadata columns** rather than JSON blobs. This is the gap in every existing HF dataset -- `sedthh` stores metadata as a serialized JSON string which is awkward to query.

3. **Three configs rather than one** to serve different use cases without forcing users to chunk themselves.

4. **Weekly incremental rather than daily**. PG adds ~30 books/week. Weekly batching is simpler and well within GH Actions free tier limits.

5. **Catalog diffing** rather than RSS polling. The CSV catalog is authoritative and catches all changes, not just the last 24 hours.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Chapter heuristics fail for non-English texts | Incorrect chapter boundaries | `has_chapters` flag lets users filter; paragraphs config unaffected |
| GH Actions timeout on large batches | Incremental update fails | Cap at 100 books per run; if more, split across runs |
| PG changes bulk file URLs/format | Pipeline breaks | Monitor with GH issue on failure; URLs have been stable for years |
| HF Hub API changes | Upload fails | Pin `huggingface_hub` version; test in CI |
| Text encoding issues | Garbled text for some books | Use UTF-8 with fallback to latin-1; log warnings |
