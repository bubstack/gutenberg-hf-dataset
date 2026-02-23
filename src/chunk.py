import re

CHAPTER_PATTERNS = [
    re.compile(r"^(CHAPTER\s+[IVXLCDM\d]+[.\- \t:]*[^\n]*)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(BOOK\s+[IVXLCDM\d]+[.\- \t:]*[^\n]*)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(PART\s+(?:[IVXLCDM\d]+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)[.\- \t:]*[^\n]*)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^((?:ACT|SCENE)\s+[IVXLCDM\d]+[.\- \t:]*[^\n]*)$", re.IGNORECASE | re.MULTILINE),
]

MIN_PARAGRAPH_LENGTH = 5


def detect_chapters(text: str) -> list[dict]:
    for pattern in CHAPTER_PATTERNS:
        matches = list(pattern.finditer(text))
        if len(matches) >= 2:
            return _split_at_matches(text, matches)
        if len(matches) == 1:
            return _split_at_matches(text, matches)
    return [{"chapter_index": 0, "chapter_title": None, "text": text.strip()}]


def _split_at_matches(text: str, matches: list[re.Match]) -> list[dict]:
    chapters = []
    pre = text[: matches[0].start()].strip()
    if len(pre) > MIN_PARAGRAPH_LENGTH:
        chapters.append({"chapter_index": 0, "chapter_title": None, "text": pre})

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        chapters.append({"chapter_index": len(chapters), "chapter_title": title, "text": body})

    return chapters


def split_paragraphs(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    raw_parts = re.split(r"\n\s*\n", text)
    raw_parts = [p.strip() for p in raw_parts if p.strip()]
    if not raw_parts:
        return []

    merged = []
    buffer = ""
    for part in raw_parts:
        if buffer:
            part = buffer + " " + part
            buffer = ""
        if len(part) < MIN_PARAGRAPH_LENGTH and len(part.split()) < 4:
            buffer = part
        else:
            merged.append(part)

    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)
    return merged
