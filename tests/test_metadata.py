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
    assert isinstance(meta["summary"], (str, type(None)))
