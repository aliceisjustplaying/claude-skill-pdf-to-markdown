#!/usr/bin/env python3
"""
PDF to Markdown Converter for LLM Context
Extracts entire PDF content as clean, structured markdown.
Images are extracted to cache directory by default.

Features:
- Aggressive persistent caching (extracts once, reuses forever)
- Full PDF cached, pages sliced on demand
- Cache only cleared on explicit request or source file change

Usage:
    python pdf_to_md.py <input.pdf> [output.md]
    python pdf_to_md.py <input.pdf> --stdout
    python pdf_to_md.py <input.pdf> --pages 1-5
    python pdf_to_md.py <input.pdf> --clear-cache
    python pdf_to_md.py --clear-all-cache

Dependencies:
    uv pip install pymupdf4llm
"""

import argparse
import sys
import os
import re
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count

# Minimum pages to use parallel processing (overhead not worth it for small PDFs)
PARALLEL_THRESHOLD = 100

# Cache directory
CACHE_DIR = Path.home() / ".cache" / "pdf-to-markdown"


# =============================================================================
# CACHING FUNCTIONS
# =============================================================================

def get_cache_key(pdf_path: str) -> str:
    """Generate cache key from absolute path + size + mtime."""
    p = Path(pdf_path).resolve()
    stat = p.stat()
    raw = f"{p}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cache_dir(cache_key: str) -> Path:
    """Get cache directory for a given cache key."""
    return CACHE_DIR / cache_key


def is_cache_valid(pdf_path: str) -> tuple:
    """
    Check if valid cache exists for this PDF.

    Returns:
        (is_valid: bool, cache_key: str)
    """
    try:
        cache_key = get_cache_key(pdf_path)
    except (FileNotFoundError, OSError):
        return False, ""

    cache_dir = get_cache_dir(cache_key)
    metadata_file = cache_dir / "metadata.json"
    output_file = cache_dir / "full_output.md"

    if not metadata_file.exists() or not output_file.exists():
        return False, cache_key

    # Verify metadata matches current file
    try:
        with open(metadata_file) as f:
            metadata = json.load(f)

        p = Path(pdf_path).resolve()
        stat = p.stat()

        if (metadata.get("source_size") != stat.st_size or
            metadata.get("source_mtime") != stat.st_mtime):
            return False, cache_key

        return True, cache_key
    except (json.JSONDecodeError, KeyError, OSError):
        return False, cache_key


def load_from_cache(cache_key: str, pages: list = None) -> tuple:
    """
    Load markdown from cache, optionally slice specific pages.

    Returns:
        (markdown: str, image_dir: Path or None, total_pages: int)
    """
    cache_dir = get_cache_dir(cache_key)

    # Load full markdown
    full_md = (cache_dir / "full_output.md").read_text(encoding="utf-8")

    # Load metadata for total pages
    with open(cache_dir / "metadata.json") as f:
        metadata = json.load(f)
    total_pages = metadata.get("total_pages", 0)

    # Check for cached images
    image_dir = cache_dir / "images"
    if not image_dir.exists() or not any(image_dir.iterdir()):
        image_dir = None

    # Slice pages if requested
    if pages:
        full_md = slice_pages_from_markdown(full_md, pages, total_pages)

    return full_md, image_dir, total_pages


def save_to_cache(cache_key: str, markdown: str, image_dir: Path, pdf_path: str, total_pages: int):
    """Save full extraction to cache."""
    cache_dir = get_cache_dir(cache_key)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save markdown
    (cache_dir / "full_output.md").write_text(markdown, encoding="utf-8")

    # Save metadata
    p = Path(pdf_path).resolve()
    stat = p.stat()
    metadata = {
        "source_path": str(p),
        "source_mtime": stat.st_mtime,
        "source_size": stat.st_size,
        "cache_key": cache_key,
        "cached_at": datetime.now().isoformat(),
        "total_pages": total_pages,
    }
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Copy images to cache
    if image_dir and Path(image_dir).exists():
        cache_images = cache_dir / "images"
        if cache_images.exists():
            shutil.rmtree(cache_images)
        shutil.copytree(image_dir, cache_images)


def slice_pages_from_markdown(full_md: str, pages: list, total_pages: int) -> str:
    """
    Extract specific pages from full markdown.

    pymupdf4llm typically outputs page separators like:
    -----
    or includes page numbers. We'll split on common patterns.
    """
    # Try to split on page separator pattern (horizontal rules)
    # pymupdf4llm uses "-----" as page separator
    page_pattern = r'\n-----\n'
    parts = re.split(page_pattern, full_md)

    if len(parts) <= 1:
        # No clear page separators, return full content
        return full_md

    # Convert 0-indexed pages to parts indices
    selected_parts = []
    for page_num in pages:
        if 0 <= page_num < len(parts):
            selected_parts.append(parts[page_num])

    if not selected_parts:
        return full_md

    return "\n-----\n".join(selected_parts)


def clear_cache(pdf_path: str = None):
    """Clear cache for specific PDF or entire cache."""
    if pdf_path:
        try:
            cache_key = get_cache_key(pdf_path)
            cache_dir = get_cache_dir(cache_key)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                return True
        except (FileNotFoundError, OSError):
            pass
        return False
    else:
        # Clear all cache
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            return True
        return False


def get_cache_stats() -> dict:
    """Get statistics about the cache."""
    if not CACHE_DIR.exists():
        return {"entries": 0, "total_size_mb": 0, "cache_dir": str(CACHE_DIR)}

    entries = 0
    total_size = 0

    for entry in CACHE_DIR.iterdir():
        if entry.is_dir():
            entries += 1
            for f in entry.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size

    return {
        "entries": entries,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "cache_dir": str(CACHE_DIR),
    }


# =============================================================================
# PDF PROCESSING FUNCTIONS
# =============================================================================

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import pymupdf4llm
        return True
    except ImportError:
        print("ERROR: pymupdf4llm not installed.", file=sys.stderr)
        print("Install with: uv pip install pymupdf4llm", file=sys.stderr)
        return False


def parse_page_range(page_str, total_pages):
    """Parse page range string like '1-5' or '1,3,5-7'."""
    if not page_str:
        return None

    pages = []
    for part in page_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            start = int(start) - 1  # Convert to 0-indexed
            end = int(end)  # End is inclusive, so no -1
            pages.extend(range(start, min(end, total_pages)))
        else:
            pages.append(int(part) - 1)

    return sorted(set(p for p in pages if 0 <= p < total_pages))


def get_image_info(image_dir):
    """
    Get information about extracted images.

    Returns:
        List of dicts with image metadata
    """
    if not image_dir:
        return []

    image_dir = Path(image_dir)
    if not image_dir.exists():
        return []

    images = []
    for img_path in sorted(image_dir.glob("*")):
        if img_path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'):
            try:
                # Get file size
                size_bytes = img_path.stat().st_size
                size_kb = size_bytes / 1024

                # Try to get dimensions using pymupdf
                try:
                    import pymupdf
                    pix = pymupdf.Pixmap(str(img_path))
                    dimensions = f"{pix.width}x{pix.height}"
                    pix = None
                except:
                    dimensions = "unknown"

                images.append({
                    'filename': img_path.name,
                    'path': str(img_path),
                    'size_kb': round(size_kb, 1),
                    'dimensions': dimensions,
                })
            except Exception:
                pass

    return images


def enhance_markdown_with_image_paths(markdown, image_dir):
    """
    Enhance image references in markdown with full absolute paths.
    """
    if not image_dir:
        return markdown

    image_dir = Path(image_dir)

    def replace_image_ref(match):
        alt_text = match.group(1)
        filename = match.group(2)
        full_path = image_dir / filename

        if full_path.exists():
            try:
                size_kb = round(full_path.stat().st_size / 1024, 1)
                try:
                    import pymupdf
                    pix = pymupdf.Pixmap(str(full_path))
                    dims = f"{pix.width}x{pix.height}"
                    pix = None
                except:
                    dims = "?"

                return f"![{alt_text}]({filename})\n\n**[Image: {filename} ({dims}, {size_kb}KB) → {full_path}]**"
            except:
                return f"![{alt_text}]({filename})\n\n**[Image: {filename} → {full_path}]**"

        return match.group(0)

    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    return re.sub(pattern, replace_image_ref, markdown)


def create_image_summary(images):
    """Create a summary section listing all extracted images."""
    if not images:
        return ""

    lines = [
        "",
        "---",
        "",
        "## Extracted Images",
        "",
        "| # | File | Dimensions | Size | Path |",
        "|---|------|------------|------|------|",
    ]

    for i, img in enumerate(images, 1):
        lines.append(
            f"| {i} | {img['filename']} | {img['dimensions']} | {img['size_kb']}KB | `{img['path']}` |"
        )

    lines.append("")
    return "\n".join(lines)


def process_page_batch(args):
    """Worker function: process a batch of pages and return markdown."""
    pdf_path, page_numbers, kwargs = args
    import pymupdf4llm

    result = pymupdf4llm.to_markdown(pdf_path, pages=page_numbers, **kwargs)
    return (page_numbers[0], result)


def convert_parallel(pdf_path, pages_to_process, workers, batch_size,
                     image_dir=None, no_images=False, show_progress=False):
    """Convert PDF to markdown using parallel processing."""
    kwargs = {
        'show_progress': False,
    }

    if not no_images and image_dir:
        kwargs['write_images'] = True
        kwargs['image_path'] = image_dir

    batches = []
    for i in range(0, len(pages_to_process), batch_size):
        batch_pages = pages_to_process[i:i + batch_size]
        batches.append((pdf_path, batch_pages, kwargs))

    total_batches = len(batches)
    results = []

    if show_progress:
        print(f"Processing {len(pages_to_process)} pages in {total_batches} batches using {workers} workers...", file=sys.stderr)

    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_page_batch, batches)):
            results.append(result)
            if show_progress:
                completed = i + 1
                print(f"\rBatch {completed}/{total_batches} complete", end="", file=sys.stderr)

    if show_progress:
        print(file=sys.stderr)

    results.sort(key=lambda x: x[0])
    return "\n\n".join(r[1] for r in results)


def convert_sequential(pdf_path, pages=None, chunked=False,
                       image_dir=None, no_images=False, show_progress=False):
    """Convert PDF to markdown using single-threaded processing."""
    import pymupdf4llm

    kwargs = {
        'show_progress': show_progress,
    }

    if pages is not None:
        kwargs['pages'] = pages

    if chunked:
        kwargs['page_chunks'] = True

    if not no_images and image_dir:
        kwargs['write_images'] = True
        kwargs['image_path'] = image_dir

    return pymupdf4llm.to_markdown(pdf_path, **kwargs)


def add_metadata_header(markdown, pdf_path, total_pages, pages_extracted, image_dir=None, cached=False):
    """Add metadata header to markdown output."""
    filename = os.path.basename(pdf_path)

    header_lines = [
        "---",
        f"source: {filename}",
        f"total_pages: {total_pages}",
        f"pages_extracted: {pages_extracted}",
        f"extracted_at: {datetime.now().isoformat()}",
    ]

    if cached:
        header_lines.append("from_cache: true")

    if image_dir:
        header_lines.append(f"images_dir: {image_dir}")

    header_lines.extend(["---", "", ""])

    return "\n".join(header_lines) + markdown


def setup_temp_image_dir(pdf_path):
    """Create temporary image directory for extraction."""
    pdf_name = Path(pdf_path).stem
    safe_name = re.sub(r'[^\w\-_]', '_', pdf_name)
    image_dir = Path("/tmp/pdf_images") / safe_name

    if image_dir.exists():
        shutil.rmtree(image_dir)

    image_dir.mkdir(parents=True, exist_ok=True)
    return str(image_dir)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert PDF to Markdown for LLM context (with persistent caching)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_to_md.py document.pdf                    # Output to document.md (cached)
  python pdf_to_md.py document.pdf --stdout           # Print to stdout
  python pdf_to_md.py document.pdf --pages 1-10      # Only pages 1-10 (from cache)
  python pdf_to_md.py document.pdf --no-cache        # Bypass cache
  python pdf_to_md.py document.pdf --clear-cache     # Clear cache for this PDF
  python pdf_to_md.py --clear-all-cache              # Clear entire cache

Caching:
  PDFs are cached in ~/.cache/pdf-to-markdown/
  Cache is keyed by file path + size + modification time.
  Full PDF is always extracted and cached; --pages slices from cache.
  Cache persists until explicitly cleared or source PDF changes.
        """
    )

    parser.add_argument('input', nargs='?', help='Input PDF file path')
    parser.add_argument('output', nargs='?', help='Output markdown file path (default: <input>.md)')
    parser.add_argument('--stdout', action='store_true', help='Print to stdout instead of file')
    parser.add_argument('--pages', help='Page range to extract (e.g., "1-5" or "1,3,5-7")')
    parser.add_argument('--chunked', action='store_true', help='Output as JSON with page chunks and metadata')
    parser.add_argument('--no-images', action='store_true', help='Skip image extraction (faster)')
    parser.add_argument('--no-metadata', action='store_true', help='Skip metadata header')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress indicator')
    parser.add_argument('--workers', type=int, default=None,
                        help=f'Number of parallel workers (default: CPU count, currently {cpu_count()})')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Pages per worker batch (default: 50)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')

    # Cache options
    parser.add_argument('--no-cache', action='store_true',
                        help='Bypass cache, process fresh (still updates cache)')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear cache for this PDF before processing')
    parser.add_argument('--clear-all-cache', action='store_true',
                        help='Clear entire cache directory and exit')
    parser.add_argument('--cache-stats', action='store_true',
                        help='Show cache statistics and exit')

    args = parser.parse_args()

    # Handle cache management commands first
    if args.clear_all_cache:
        if clear_cache():
            print(f"Cache cleared: {CACHE_DIR}", file=sys.stderr)
        else:
            print("Cache was already empty.", file=sys.stderr)
        sys.exit(0)

    if args.cache_stats:
        stats = get_cache_stats()
        print(f"Cache directory: {stats['cache_dir']}", file=sys.stderr)
        print(f"Cached PDFs: {stats['entries']}", file=sys.stderr)
        print(f"Total size: {stats['total_size_mb']} MB", file=sys.stderr)
        sys.exit(0)

    # Require input for all other operations
    if not args.input:
        parser.error("the following arguments are required: input")

    # Validate input
    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not args.input.lower().endswith('.pdf'):
        print(f"WARNING: File may not be a PDF: {args.input}", file=sys.stderr)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Clear cache for this specific PDF if requested
    if args.clear_cache:
        if clear_cache(args.input):
            print(f"Cache cleared for: {args.input}", file=sys.stderr)
        else:
            print(f"No cache found for: {args.input}", file=sys.stderr)

    # Get total pages
    import pymupdf
    doc = pymupdf.open(args.input)
    total_pages = len(doc)
    doc.close()

    # Parse page range (for output slicing, not extraction)
    requested_pages = parse_page_range(args.pages, total_pages) if args.pages else None
    pages_to_output = len(requested_pages) if requested_pages else total_pages

    # Determine if progress should be shown
    show_progress = sys.stderr.isatty() and not args.no_progress and not args.stdout

    # Check cache
    cache_hit = False
    cache_key = ""
    result = None
    image_dir = None

    if not args.no_cache:
        valid, cache_key = is_cache_valid(args.input)
        if valid:
            if show_progress:
                print(f"Loading from cache...", file=sys.stderr)
            result, image_dir, cached_total = load_from_cache(cache_key, requested_pages)
            cache_hit = True

    # If no cache hit, extract full PDF
    if not cache_hit:
        # Get cache key if we don't have it
        if not cache_key:
            cache_key = get_cache_key(args.input)

        # Setup image directory for extraction (temporary)
        temp_image_dir = None
        if not args.no_images:
            temp_image_dir = setup_temp_image_dir(args.input)

        # Determine if we should use parallel processing
        workers = args.workers or cpu_count()
        use_parallel = (
            not args.no_parallel
            and not args.chunked
            and total_pages >= PARALLEL_THRESHOLD
            and workers > 1
        )

        # Extract FULL PDF (always, for caching)
        try:
            if use_parallel:
                result = convert_parallel(
                    args.input,
                    pages_to_process=list(range(total_pages)),
                    workers=workers,
                    batch_size=args.batch_size,
                    image_dir=temp_image_dir,
                    no_images=args.no_images,
                    show_progress=show_progress
                )
            else:
                if show_progress and total_pages > 50:
                    print(f"Processing {total_pages} pages...", file=sys.stderr)

                result = convert_sequential(
                    args.input,
                    pages=None,  # Always extract full PDF
                    chunked=args.chunked,
                    image_dir=temp_image_dir,
                    no_images=args.no_images,
                    show_progress=show_progress
                )
        except Exception as e:
            print(f"ERROR: Conversion failed: {e}", file=sys.stderr)
            sys.exit(1)

        # Save to cache (full result)
        if not args.no_cache and not args.chunked:
            save_to_cache(cache_key, result, temp_image_dir, args.input, total_pages)
            if show_progress:
                print(f"Cached: {get_cache_dir(cache_key)}", file=sys.stderr)

        # Set image_dir to cached location
        cached_image_dir = get_cache_dir(cache_key) / "images"
        if cached_image_dir.exists() and any(cached_image_dir.iterdir()):
            image_dir = cached_image_dir
        else:
            image_dir = temp_image_dir

        # Slice pages if requested (after caching full result)
        if requested_pages and not args.chunked:
            result = slice_pages_from_markdown(result, requested_pages, total_pages)

    # Format output
    if args.chunked:
        output = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        output = result

        # Enhance image references with full paths
        if image_dir:
            output = enhance_markdown_with_image_paths(output, image_dir)

            # Add image summary table at the end
            images = get_image_info(image_dir)
            if images:
                output += create_image_summary(images)

        if not args.no_metadata:
            output = add_metadata_header(
                output, args.input, total_pages, pages_to_output,
                image_dir, cached=cache_hit
            )

    # Write output
    if args.stdout:
        print(output)
    else:
        output_path = args.output or os.path.splitext(args.input)[0] + '.md'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output)

        msg = f"Converted {pages_to_output} pages to: {output_path}"
        if cache_hit:
            msg += " (from cache)"
        if image_dir:
            images = get_image_info(image_dir)
            if images:
                msg += f" ({len(images)} images)"
        print(msg, file=sys.stderr)


if __name__ == '__main__':
    main()
