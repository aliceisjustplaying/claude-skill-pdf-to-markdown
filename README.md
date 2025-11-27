# PDF to Markdown Converter

Convert PDF documents to clean, structured Markdown with table and image extraction.

## Features

- **Text extraction** with formatting preservation (headers, bold, italic, lists)
- **Table extraction** with two modes:
  - Fast mode: PyMuPDF (good for simple tables)
  - Accurate mode: IBM Docling AI (better for complex/borderless tables)
- **Image extraction** to cache directory with paths in output
- **Aggressive caching** - extract once, reuse forever
- **Page slicing** - request specific pages from cached full extraction

## Installation

```bash
cd ~/.claude/skills/pdf-to-markdown
uv venv .venv
uv pip install --python .venv/bin/python pymupdf pymupdf4llm docling docling-core
```

## Usage

```bash
# Basic conversion
.venv/bin/python scripts/pdf_to_md.py document.pdf --stdout

# High-accuracy tables (slower)
.venv/bin/python scripts/pdf_to_md.py document.pdf --docling --stdout

# Specific pages
.venv/bin/python scripts/pdf_to_md.py document.pdf --pages 1-10 --stdout

# Skip images (faster)
.venv/bin/python scripts/pdf_to_md.py document.pdf --no-images --stdout

# Save to file
.venv/bin/python scripts/pdf_to_md.py document.pdf output.md
```

## Options

| Option | Description |
|--------|-------------|
| `--stdout` | Print to stdout instead of file |
| `--pages RANGE` | Page range (e.g., "1-5" or "1,3,5-7") |
| `--docling` | Use Docling AI for high-accuracy tables |
| `--no-images` | Skip image extraction |
| `--no-metadata` | Skip metadata header |
| `--no-cache` | Bypass cache (still updates it) |
| `--clear-cache` | Clear cache for this PDF |
| `--clear-all-cache` | Clear entire cache |
| `--cache-stats` | Show cache statistics |

## Project Structure

```
scripts/
  pdf_to_md.py    # Main CLI tool
  extractor.py    # PDF extraction library (fast + accurate modes)
```

## Cache

PDFs are cached in `~/.cache/pdf-to-markdown/`. Cache is invalidated when:
- Source PDF is modified
- Extractor version changes
- Explicitly cleared with `--clear-cache`

## License

MIT
