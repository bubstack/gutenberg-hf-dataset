import gutenbergpy.textget


def strip_gutenberg_headers(raw_bytes: bytes) -> str:
    """Strip Project Gutenberg headers/footers and return clean text as str."""
    try:
        clean_bytes = gutenbergpy.textget.strip_headers(raw_bytes)
    except Exception:
        clean_bytes = raw_bytes

    try:
        return clean_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return clean_bytes.decode("latin-1")
