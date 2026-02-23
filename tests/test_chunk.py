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
