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
    assert "Call me Ishmael" in text

def test_strip_headers_returns_string():
    raw = FIXTURES / "raw_sample.txt"
    text = strip_gutenberg_headers(raw.read_bytes())
    assert isinstance(text, str)

def test_strip_headers_handles_encoding_fallback():
    raw = "Héllo wörld".encode("latin-1")
    text = strip_gutenberg_headers(raw)
    assert isinstance(text, str)
