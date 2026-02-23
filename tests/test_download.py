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
