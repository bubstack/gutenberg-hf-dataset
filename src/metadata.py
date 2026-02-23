"""Parse RDF/XML metadata files from Project Gutenberg."""

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
    """Return the first text result of an XPath query, or default."""
    results = element.xpath(xpath, namespaces=NAMESPACES)
    if results:
        return str(results[0]).strip()
    return default


def _xpath_texts(element, xpath: str) -> list[str]:
    """Return all text results of an XPath query."""
    results = element.xpath(xpath, namespaces=NAMESPACES)
    return [str(r).strip() for r in results]


def _xpath_int(element, xpath: str, default=None) -> int | None:
    """Return the first integer result of an XPath query, or default."""
    text = _xpath_text(element, xpath)
    if text is not None:
        try:
            return int(text)
        except ValueError:
            return default
    return default


def parse_rdf(rdf_path: Path) -> dict:
    """Parse a Project Gutenberg RDF file and return a metadata dictionary.

    Args:
        rdf_path: Path to the RDF/XML file.

    Returns:
        Dictionary with keys: id, title, author, author_birth_year,
        author_death_year, contributors, subjects, bookshelves, locc,
        language, release_date, rights, summary.

    Raises:
        ValueError: If no pgterms:ebook element is found in the file.
    """
    tree = etree.parse(str(rdf_path))
    ebook_elements = tree.xpath("//pgterms:ebook", namespaces=NAMESPACES)
    if not ebook_elements:
        raise ValueError(f"No pgterms:ebook found in {rdf_path}")
    ebook = ebook_elements[0]

    # Extract ebook ID from rdf:about attribute (e.g. "ebooks/2701" -> "2701")
    about = ebook.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about", "")
    ebook_id = about.split("/")[-1] if "/" in about else about

    # LCSH subjects
    subjects = _xpath_texts(
        ebook,
        'dcterms:subject/rdf:Description'
        '[dcam:memberOf/@rdf:resource="http://purl.org/dc/terms/LCSH"]'
        '/rdf:value/text()',
    )

    # Library of Congress Classification
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

    # Contributors from MARC 508 field
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
            ebook, "dcterms:creator/pgterms:agent/pgterms:name/text()", ""
        ),
        "author_birth_year": _xpath_int(
            ebook, "dcterms:creator/pgterms:agent/pgterms:birthdate/text()"
        ),
        "author_death_year": _xpath_int(
            ebook, "dcterms:creator/pgterms:agent/pgterms:deathdate/text()"
        ),
        "contributors": contributors,
        "subjects": subjects,
        "bookshelves": bookshelves,
        "locc": locc_list[0] if locc_list else "",
        "language": _xpath_text(
            ebook, "dcterms:language/rdf:Description/rdf:value/text()", ""
        ),
        "release_date": _xpath_text(ebook, "dcterms:issued/text()", ""),
        "rights": _xpath_text(ebook, "dcterms:rights/text()", ""),
        "summary": _xpath_text(ebook, "pgterms:marc520/text()"),
    }
