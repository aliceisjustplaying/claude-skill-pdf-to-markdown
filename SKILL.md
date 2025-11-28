---
name: pdf-to-markdown
description: Convert entire PDF documents to clean, structured Markdown for full context loading. Use this skill when the user wants to extract ALL text from a PDF into context (not grep/search), when discussing or analyzing PDF content in full, when the user mentions "load the whole PDF", "bring the PDF into context", "read the entire PDF", or when partial extraction/grepping would miss important context. This is the preferred method for PDF text extraction over page-by-page or grep approaches.
---

# PDF to Markdown Converter

Extract complete PDF content as structured Markdown, preserving:
- Headers (detected by font size, converted to # tags)
- Bold, italic, monospace formatting
- Tables (converted to Markdown tables)
- Lists (ordered and unordered)
- Multi-column layouts (correct reading order)
- Code blocks
- **Images** (extracted to files with paths in output)

## When to Use This Skill

**USE THIS** when:
- User wants the "whole PDF" or "entire document" in context
- Analyzing, summarizing, or discussing PDF content
- User says "load", "read", "bring in", "extract" a PDF
- Grepping/searching would miss context or structure
- PDF has tables, formatting, or structure to preserve

**USE `--pages`** when user only needs specific pages (faster, less output).

## Environment Setup

This skill uses a dedicated virtual environment at `~/.claude/skills/pdf-to-markdown/.venv/` to avoid polluting the user's working directory.

### First-Time Setup (if .venv doesn't exist)
```bash
# For fast mode only (PyMuPDF):
cd ~/.claude/skills/pdf-to-markdown && uv venv .venv && uv pip install --python .venv/bin/python pymupdf pymupdf4llm

# For --docling mode (high-accuracy tables):
cd ~/.claude/skills/pdf-to-markdown && uv venv .venv && uv pip install --python .venv/bin/python pymupdf docling docling-core

# Or install everything:
cd ~/.claude/skills/pdf-to-markdown && uv venv .venv && uv pip install --python .venv/bin/python pymupdf pymupdf4llm docling docling-core
```

### Verify Installation
```bash
# Verify fast mode:
~/.claude/skills/pdf-to-markdown/.venv/bin/python -c "import pymupdf; import pymupdf4llm; print('OK')"

# Verify docling mode:
~/.claude/skills/pdf-to-markdown/.venv/bin/python -c "import pymupdf; import docling; import docling_core; print('OK')"
```

## Quick Start

### Using the Script (Recommended)

The script automatically uses the skill's dedicated venv:

```bash
# Convert PDF to markdown (images extracted by default)
~/.claude/skills/pdf-to-markdown/.venv/bin/python ~/.claude/skills/pdf-to-markdown/scripts/pdf_to_md.py document.pdf --stdout

# Skip image extraction (faster, smaller output)
~/.claude/skills/pdf-to-markdown/.venv/bin/python ~/.claude/skills/pdf-to-markdown/scripts/pdf_to_md.py document.pdf --no-images --stdout

# Specific pages only
~/.claude/skills/pdf-to-markdown/.venv/bin/python ~/.claude/skills/pdf-to-markdown/scripts/pdf_to_md.py document.pdf --pages 1-10 --stdout
```

## Standard Workflow

When user provides a PDF and wants full content in context:

### Step 1: Ensure the skill venv exists
```bash
# For fast mode (default):
test -d ~/.claude/skills/pdf-to-markdown/.venv || (cd ~/.claude/skills/pdf-to-markdown && uv venv .venv && uv pip install --python .venv/bin/python pymupdf pymupdf4llm)
```

### Step 2: Convert PDF to Markdown
```bash
~/.claude/skills/pdf-to-markdown/.venv/bin/python ~/.claude/skills/pdf-to-markdown/scripts/pdf_to_md.py "/path/to/document.pdf" --stdout
```

### Step 3: Load into context
The markdown output is now available. Display it or use it directly.

## Caching

PDFs are **aggressively cached** to avoid re-processing. First extraction is slow, every subsequent request is instant.

### How It Works
- **Cache location**: `~/.cache/pdf-to-markdown/<cache_key>/`
- **Cache key**: Based on file path + size + modification time
- **Full PDF cached**: Even if you request `--pages 1-10`, the full PDF is extracted and cached. Page slicing happens from the cached result.
- **Invalidation**: Cache is invalidated when:
  - Source PDF is modified (size or mtime changes)
  - Extractor version changes (automatic re-extraction)
  - Explicitly cleared with `--clear-cache` or `--clear-all-cache`

### Cache Commands
```bash
# Clear cache for a specific PDF
~/.claude/skills/pdf-to-markdown/.venv/bin/python ~/.claude/skills/pdf-to-markdown/scripts/pdf_to_md.py document.pdf --clear-cache

# Clear entire cache
~/.claude/skills/pdf-to-markdown/.venv/bin/python ~/.claude/skills/pdf-to-markdown/scripts/pdf_to_md.py --clear-all-cache

# Show cache statistics
~/.claude/skills/pdf-to-markdown/.venv/bin/python ~/.claude/skills/pdf-to-markdown/scripts/pdf_to_md.py --cache-stats

# Bypass cache entirely (no read or write)
~/.claude/skills/pdf-to-markdown/.venv/bin/python ~/.claude/skills/pdf-to-markdown/scripts/pdf_to_md.py document.pdf --no-cache --stdout
```

### Cache Contents
```
~/.cache/pdf-to-markdown/<cache_key>/
├── metadata.json    # source path, mtime, size, total_pages
├── full_output.md   # cached full markdown
└── images/          # extracted images
```

## Image Handling

By default, images are:
1. **Extracted** to cache directory `~/.cache/pdf-to-markdown/<cache_key>/images/`
2. **Referenced** in the markdown with full paths like:
   ```
   ![alt text](image.png)

   **[Image: image.png (800x600, 45.2KB) → ~/.cache/pdf-to-markdown/<key>/images/image.png]**
   ```
3. **Summarized** in a table at the end of the document

### Auto-View Behavior for Images

**IMPORTANT:** When the extracted markdown contains image references like:
```
**[Image: figure_1.png (1200x800, 125.3KB) → /Users/.../.cache/pdf-to-markdown/abc123/images/figure_1.png]**
```

And the user asks about something that might be visual (charts, graphs, diagrams, figures, screenshots, layouts, designs, plots, illustrations), **automatically use the Read tool** to view the relevant image file(s) before answering. Don't ask the user - just look at it.

**Examples of when to auto-view images:**
- User: "What does the chart on page 3 show?" → Read the image file
- User: "Summarize the figures in this paper" → Read all image files
- User: "What's in the diagram?" → Read the image file
- User: "Describe the architecture shown" → Read the image file
- User: "What are the results?" (and there's a results figure) → Read it

**When NOT to auto-view:**
- User only asks about text content
- User explicitly says they don't need images
- No images were extracted (--no-images was used)

### Image Options

```bash
# Default: extract images to cache directory
~/.claude/skills/pdf-to-markdown/.venv/bin/python ~/.claude/skills/pdf-to-markdown/scripts/pdf_to_md.py doc.pdf --stdout

# Skip images entirely (faster)
~/.claude/skills/pdf-to-markdown/.venv/bin/python ~/.claude/skills/pdf-to-markdown/scripts/pdf_to_md.py doc.pdf --no-images --stdout
```

## Output Format

The markdown output includes:

### Header (metadata)
```yaml
---
source: document.pdf
total_pages: 42
pages_extracted: 42
extracted_at: 2025-01-15T10:30:00
from_cache: true
images_dir: /Users/.../.cache/pdf-to-markdown/abc123/images
---
```

### Content with image references
```markdown
# Main Title

## Section Header

Regular paragraph text with **bold**, *italic*, and `code` formatting.

![Figure 1](figure_1.png)

**[Image: figure_1.png (800x600, 45.2KB) → ~/.cache/pdf-to-markdown/abc123/images/figure_1.png]**

| Column A | Column B |
|----------|----------|
| Data 1   | Data 2   |
```

### Image summary table (at end)
```markdown
---

## Extracted Images

| # | File | Dimensions | Size | Path |
|---|------|------------|------|------|
| 1 | figure_1.png | 800x600 | 45.2KB | `~/.cache/pdf-to-markdown/abc123/images/figure_1.png` |
| 2 | chart_2.png | 1200x800 | 89.1KB | `~/.cache/pdf-to-markdown/abc123/images/chart_2.png` |
```

## Script Reference

Location: `~/.claude/skills/pdf-to-markdown/scripts/pdf_to_md.py`

```
Usage: pdf_to_md.py <input.pdf> [output.md] [options]

Options:
  --stdout          Print to stdout instead of file
  --pages RANGE     Page range (e.g., "1-5" or "1,3,5-7")
  --docling         Use Docling AI for high-accuracy tables (~1 sec/page)
  --images-scale N  Image resolution for Docling mode (default: 4.0)
  --no-images       Skip image extraction (faster)
  --no-metadata     Skip metadata header in output
  --no-progress     Disable progress indicator

Cache Options:
  --no-cache           Bypass cache entirely (no read or write)
  --clear-cache        Clear cache for this PDF (works even if PDF was deleted)
  --clear-all-cache    Clear entire cache directory and exit
  --cache-stats        Show cache statistics and exit
  --force-stale-cache  Use cached extraction even if version differs (when PDF missing)
```

**Performance:** First extraction is cached, so subsequent requests for the same PDF are instant.

## Advanced Usage

### Extract Specific Pages
```bash
~/.claude/skills/pdf-to-markdown/.venv/bin/python ~/.claude/skills/pdf-to-markdown/scripts/pdf_to_md.py document.pdf --pages 1-10 --stdout
```

### Handle Scanned PDFs (OCR)
For scanned PDFs without extractable text, pymupdf4llm will attempt OCR automatically if Tesseract is available:
```bash
# Install Tesseract first (macOS)
brew install tesseract

# Then convert - OCR happens automatically for image-based pages
~/.claude/skills/pdf-to-markdown/.venv/bin/python ~/.claude/skills/pdf-to-markdown/scripts/pdf_to_md.py scanned.pdf --stdout
```

## Troubleshooting

### "No module named pymupdf4llm" or venv doesn't exist
Recreate the skill's virtual environment:
```bash
# For fast mode:
cd ~/.claude/skills/pdf-to-markdown && rm -rf .venv && uv venv .venv && uv pip install --python .venv/bin/python pymupdf pymupdf4llm

# For docling mode:
cd ~/.claude/skills/pdf-to-markdown && rm -rf .venv && uv venv .venv && uv pip install --python .venv/bin/python pymupdf docling docling-core
```

### Poor extraction quality
- Try `marker-pdf` for complex layouts (install into the skill venv):
  ```bash
  uv pip install --python ~/.claude/skills/pdf-to-markdown/.venv/bin/python marker-pdf
  ```
- For scanned PDFs, ensure Tesseract OCR is installed: `brew install tesseract`

### Very large PDFs
- Use `--pages` to extract only needed sections
- Use `--no-images` to skip image extraction (faster)

### Tables not formatting correctly
pymupdf4llm handles most tables well. For complex tables:
```bash
~/.claude/skills/pdf-to-markdown/.venv/bin/python -c "
import pymupdf4llm
md_text = pymupdf4llm.to_markdown('doc.pdf', table_strategy='lines_strict')
print(md_text)
"
```

## High-Accuracy Mode (Docling)

For PDFs with complex tables that need high accuracy, use the `--docling` flag:

```bash
~/.claude/skills/pdf-to-markdown/.venv/bin/python \
    ~/.claude/skills/pdf-to-markdown/scripts/pdf_to_md.py \
    document.pdf --docling --stdout
```

**When to use `--docling`:**
- PDF has complex tables (borderless, merged cells, multi-column)
- Table accuracy is critical (medical data, financial reports)
- You're seeing garbled table output in default mode

**Trade-offs:**
- ~1 second per page (vs instant for fast mode)
- First run downloads AI models (~500MB one-time)
- Higher-resolution images (4x default)

**Image resolution:**
```bash
# Default: 4x resolution (crisp images)
... --docling --stdout

# Custom resolution (2x for smaller files)
... --docling --images-scale 2.0 --stdout
```

**Note:** `--accurate` is an alias for `--docling` for backwards compatibility.

## Comparison with Other Approaches

| Approach | Use Case | Limitations |
|----------|----------|-------------|
| **This skill (pymupdf4llm)** | Full document context with images | Large PDFs may exceed context |
| **--docling mode** | Complex tables, medical/financial PDFs | Slower (~1 sec/page), larger models |
| Grepping PDF | Find specific text | Loses structure, no images |
| Page-by-page extraction | Targeted pages | Manual, loses cross-page context |
| Read tool on PDF | Quick preview | Limited formatting preservation |
