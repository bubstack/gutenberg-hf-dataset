import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Patterns for split-volume titles
SPLIT_PATTERNS = [
    re.compile(r",?\s*Part\s+\d+", re.IGNORECASE),
    re.compile(r",?\s*Vol(?:ume)?\.?\s+\d+", re.IGNORECASE),
    re.compile(r",?\s*Book\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r",?\s*Chapters?\s+\d+", re.IGNORECASE),
    re.compile(r",?\s*Tome\s+\d+", re.IGNORECASE),
]

COMPLETE_PATTERN = re.compile(r",?\s*Complete$", re.IGNORECASE)

LEADING_ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
SUBTITLE_SEP = re.compile(r"\s*[;:]\s*.*$")


def normalize_title(title: str) -> str:
    """Normalize a title for fuzzy matching."""
    t = title.strip()
    # Remove ", Complete" suffix
    t = COMPLETE_PATTERN.sub("", t)
    # Remove split-volume suffixes
    for pat in SPLIT_PATTERNS:
        t = pat.sub("", t)
    # Remove subtitle after ; or :
    t = SUBTITLE_SEP.sub("", t)
    # Remove leading articles
    t = LEADING_ARTICLES.sub("", t)
    return t.strip().lower()


def normalize_author(author: str) -> str:
    """Normalize an author name for matching."""
    return author.strip().lower()


def is_split_volume(title: str) -> bool:
    """Check if a title indicates a split volume (Part N, Volume N, etc.)."""
    for pat in SPLIT_PATTERNS:
        if pat.search(title):
            return True
    return False


def _get_title(entry: dict) -> str:
    """Get title from entry, handling both RDF and CSV field names."""
    return entry.get("title", "") or entry.get("Title", "")


def _get_author(entry: dict) -> str:
    """Get author from entry, handling both RDF and CSV field names."""
    return entry.get("author", "") or entry.get("Authors", "")


def _get_date(entry: dict) -> str:
    """Get release date from entry, handling both RDF and CSV field names."""
    return entry.get("release_date", "") or entry.get("Issued", "")


def deduplicate_catalog(catalog: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deduplicate a catalog of book metadata.

    Handles both RDF-parsed metadata (lowercase keys: title, author, release_date)
    and CSV catalog entries (capitalized keys: Title, Authors, Issued).

    Returns:
        (kept, removed) - two lists of catalog entries.
    """
    if not catalog:
        return [], []

    # Group by normalized (author, title)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in catalog:
        key = (
            normalize_author(_get_author(entry)),
            normalize_title(_get_title(entry)),
        )
        groups[key].append(entry)

    kept = []
    removed = []

    for key, entries in groups.items():
        if len(entries) == 1:
            kept.append(entries[0])
            continue

        # Separate into split volumes vs. complete/normal editions
        split_entries = [
            e for e in entries
            if is_split_volume(_get_title(e))
        ]
        non_split_entries = [
            e for e in entries
            if not is_split_volume(_get_title(e))
        ]

        # If all entries are split volumes with no complete/normal edition,
        # keep them all -- they represent different parts of the same work
        if split_entries and not non_split_entries:
            kept.extend(split_entries)
            continue

        # If there are both split and non-split entries, remove the splits
        if non_split_entries and split_entries:
            removed.extend(split_entries)
            entries = non_split_entries

        if len(entries) <= 1:
            kept.extend(entries)
            continue

        # Among remaining duplicates, prefer most recent release date
        entries.sort(key=lambda e: _get_date(e), reverse=True)
        kept.append(entries[0])
        removed.extend(entries[1:])

    logger.info(
        f"Deduplication: kept {len(kept)}, removed {len(removed)} "
        f"({len(removed)}/{len(catalog)} = {100*len(removed)/max(len(catalog),1):.1f}%)"
    )

    return kept, removed
