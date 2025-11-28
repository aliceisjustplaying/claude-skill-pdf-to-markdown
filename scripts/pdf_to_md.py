#!/usr/bin/env python3
"""
PDF to Markdown Converter for LLM Context
Extracts entire PDF content as clean, structured markdown.
Images are extracted to cache directory by default.

Features:
- High-accuracy table extraction using IBM Docling (TableFormer AI model)
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
    uv pip install docling pymupdf
"""

import argparse
import sys
import os
import re
import json
import hashlib
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

# Cache directory
CACHE_DIR = Path.home() / ".cache" / "pdf-to-markdown"


# =============================================================================
# CACHING FUNCTIONS
# =============================================================================


def get_cache_key(
    pdf_path: str, docling: bool = False, images_scale: float = 4.0
) -> str:
    """Generate cache key from file content + size + mode (path-independent).

    Args:
        pdf_path: Path to the PDF file
        docling: Whether Docling mode is used
        images_scale: Image resolution multiplier (only affects cache key in Docling mode)
    """
    p = Path(pdf_path).resolve()
    stat = p.stat()
    file_size = stat.st_size

    # Hash first 64KB + last 64KB for fast content-based identity
    chunk_size = 65536  # 64KB
    hasher = hashlib.sha256()

    with open(p, "rb") as f:
        # Read first chunk
        hasher.update(f.read(chunk_size))

        # Read last chunk (if file is large enough)
        if file_size > chunk_size * 2:
            f.seek(-chunk_size, 2)  # Seek from end
            hasher.update(f.read(chunk_size))

    # Include images_scale in mode for Docling (affects extracted image resolution)
    if docling:
        mode = f"docling_{images_scale}"
    else:
        mode = "fast"
    raw = f"{file_size}|{hasher.hexdigest()}|{mode}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cache_dir(cache_key: str) -> Path:
    """Get cache directory for a given cache key."""
    return CACHE_DIR / cache_key


def is_cache_valid(
    pdf_path: str, docling: bool = False, images_scale: float = 4.0
) -> tuple:
    """
    Check if valid cache exists for this PDF.

    Returns:
        (is_valid: bool, cache_key: str)
    """
    from extractor import EXTRACTOR_VERSION

    try:
        cache_key = get_cache_key(pdf_path, docling=docling, images_scale=images_scale)
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

        if (
            metadata.get("source_size") != stat.st_size
            or metadata.get("source_mtime") != stat.st_mtime
        ):
            return False, cache_key

        # Check extractor version - invalidate if extraction logic changed
        if metadata.get("extractor_version") != EXTRACTOR_VERSION:
            return False, cache_key

        return True, cache_key
    except (json.JSONDecodeError, KeyError, OSError):
        return False, cache_key


def load_from_cache(
    cache_key: str, pages: list = None, no_images: bool = False
) -> tuple:
    """
    Load markdown from cache, optionally slice specific pages.

    Args:
        cache_key: The cache key to load from
        pages: Optional list of page numbers to slice
        no_images: If True, skip loading image directory even if cached

    Returns:
        (markdown: str, image_dir: Path or None, total_pages: int)
        Returns (None, None, 0) if cache is corrupted (caller should treat as cache miss)
    """
    cache_dir = get_cache_dir(cache_key)

    try:
        # Load full markdown
        full_md = (cache_dir / "full_output.md").read_text(encoding="utf-8")

        # Load metadata for total pages
        with open(cache_dir / "metadata.json") as f:
            metadata = json.load(f)
        total_pages = metadata.get("total_pages", 0)
    except (FileNotFoundError, IOError, json.JSONDecodeError, OSError) as e:
        # Cache is corrupted or incomplete - delete it and return cache miss
        print(
            f"WARNING: Cache corrupted ({e.__class__.__name__}), regenerating...",
            file=sys.stderr,
        )
        try:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
        except OSError:
            pass  # Best effort cleanup
        return None, None, 0

    # Check for cached images (skip if no_images flag is set)
    image_dir = None
    if not no_images:
        cached_image_dir = cache_dir / "images"
        if cached_image_dir.exists() and any(cached_image_dir.iterdir()):
            image_dir = cached_image_dir

    # Slice pages if requested
    if pages:
        full_md = slice_pages_from_markdown(full_md, pages, total_pages)

    return full_md, image_dir, total_pages


def save_to_cache(
    cache_key: str,
    markdown: str,
    image_dir: Path,
    pdf_path: str,
    total_pages: int,
    docling: bool = False,
    images_scale: float = 4.0,
):
    """Save full extraction to cache using atomic writes.

    Uses temp files + os.replace() to ensure cache integrity even if
    the process is interrupted mid-write (power loss, Ctrl+C, etc.).
    """
    from extractor import EXTRACTOR_VERSION

    cache_dir = get_cache_dir(cache_key)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build metadata
    p = Path(pdf_path).resolve()
    stat = p.stat()
    mode = f"docling_{images_scale}" if docling else "fast"

    metadata = {
        "source_path": str(p),
        "source_mtime": stat.st_mtime,
        "source_size": stat.st_size,
        "cache_key": cache_key,
        "cached_at": datetime.now().isoformat(),
        "total_pages": total_pages,
        "extractor_version": EXTRACTOR_VERSION,
        "mode": mode,
        "images_scale": images_scale if docling else None,
    }

    # Write to temp files first, then atomic rename
    # (same filesystem guarantees atomicity via os.replace)
    temp_md = None
    temp_json = None
    try:
        # Write markdown to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", dir=cache_dir, suffix=".md.tmp", delete=False, encoding="utf-8"
        ) as f:
            f.write(markdown)
            temp_md = f.name

        # Write metadata to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", dir=cache_dir, suffix=".json.tmp", delete=False
        ) as f:
            json.dump(metadata, f, indent=2)
            temp_json = f.name

        # Atomic moves (os.replace is atomic on POSIX when same filesystem)
        os.replace(temp_md, cache_dir / "full_output.md")
        temp_md = None  # Successfully moved, don't cleanup
        os.replace(temp_json, cache_dir / "metadata.json")
        temp_json = None  # Successfully moved, don't cleanup

        # Copy images to temp dir, then rename atomically
        if image_dir and Path(image_dir).exists():
            temp_images = cache_dir / "images.tmp"
            final_images = cache_dir / "images"

            # Clean up any stale temp directory
            if temp_images.exists():
                shutil.rmtree(temp_images)

            shutil.copytree(image_dir, temp_images)

            # Remove old images dir and rename temp to final
            if final_images.exists():
                shutil.rmtree(final_images)
            os.rename(temp_images, final_images)

    finally:
        # Cleanup temp files on failure
        if temp_md and os.path.exists(temp_md):
            os.unlink(temp_md)
        if temp_json and os.path.exists(temp_json):
            os.unlink(temp_json)


def slice_pages_from_markdown(full_md: str, pages: list, total_pages: int) -> str:
    """
    Extract specific pages from full markdown.

    Uses explicit <!-- PAGE_BREAK --> sentinels inserted during extraction.
    This is more reliable than matching "-----" which could appear in content.
    """
    # Split on explicit page break sentinel
    page_pattern = r"\n<!-- PAGE_BREAK -->\n"
    parts = re.split(page_pattern, full_md)

    if len(parts) <= 1:
        # No page separators found (single page or Docling mode)
        return full_md

    # Convert 0-indexed pages to parts indices
    selected_parts = []
    for page_num in pages:
        if 0 <= page_num < len(parts):
            selected_parts.append(parts[page_num])

    if not selected_parts:
        return full_md

    return "\n<!-- PAGE_BREAK -->\n".join(selected_parts)


def find_cache_by_source_path(
    pdf_path: str, docling: bool = None, images_scale: float = None
) -> list:
    """
    Find cache entries by source path in metadata.

    Used as fallback when the source PDF no longer exists (can't compute hash).

    Args:
        pdf_path: Path to the PDF file (used to match source_path in metadata)
        docling: If specified, filter to only caches matching this mode
        images_scale: If specified (and docling=True), filter to matching scale

    Returns:
        List of (cache_dir, metadata) tuples sorted by cached_at (freshest first)
    """
    if not CACHE_DIR.exists():
        return []

    pdf_path_resolved = str(Path(pdf_path).resolve())
    matching = []

    for entry in CACHE_DIR.iterdir():
        if not entry.is_dir():
            continue
        metadata_file = entry / "metadata.json"
        if not metadata_file.exists():
            continue
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)

            if metadata.get("source_path") != pdf_path_resolved:
                continue

            # Filter by mode if specified
            if docling is not None:
                # Determine mode from cache_key pattern in metadata or by checking the key
                cache_key = metadata.get("cache_key", entry.name)
                # Docling cache keys contain "docling" in the mode component
                # We can infer mode from whether the cache was created with docling
                # Check if this looks like a docling cache by examining the entry
                # Actually, we need to store mode in metadata - let's check if it exists
                cached_mode = metadata.get("mode")
                if cached_mode is None:
                    # Legacy cache without mode - try to infer from cache key
                    # This is imperfect but better than nothing
                    continue  # Skip caches without mode info for filtering
                if docling and not cached_mode.startswith("docling"):
                    continue
                if not docling and cached_mode != "fast":
                    continue

                # Filter by images_scale if docling mode and scale specified
                if docling and images_scale is not None:
                    cached_scale = metadata.get("images_scale")
                    if cached_scale is not None and cached_scale != images_scale:
                        continue

            matching.append((entry, metadata))
        except (json.JSONDecodeError, OSError):
            continue

    # Sort by cached_at (freshest first)
    matching.sort(
        key=lambda x: x[1].get("cached_at", ""),
        reverse=True,
    )

    return matching


def clear_cache(pdf_path: str = None):
    """Clear cache for specific PDF (all modes and scale variants) or entire cache."""
    if pdf_path:
        cleared = False

        # Try to clear fast mode cache by computing key (if file exists)
        try:
            cache_key = get_cache_key(pdf_path, docling=False)
            cache_dir = get_cache_dir(cache_key)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                cleared = True
        except (FileNotFoundError, OSError):
            pass

        # Use source_path lookup to find all variants (docling with any scale, etc.)
        # This handles: docling mode with any images_scale, and fast mode if file was moved/deleted
        matching_caches = find_cache_by_source_path(pdf_path)
        for cache_dir, _metadata in matching_caches:
            if cache_dir.exists():  # May have been cleared above
                shutil.rmtree(cache_dir)
                cleared = True

        return cleared
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


def check_dependencies(docling_mode: bool = False):
    """
    Check if required packages are installed for the requested mode.

    Args:
        docling_mode: If True, check for Docling dependencies.
                      If False, check for fast mode (PyMuPDF) dependencies.
    """
    missing = []

    # pymupdf is always needed (for page count, image extraction in fast mode)
    try:
        import pymupdf
    except ImportError:
        missing.append("pymupdf")

    if docling_mode:
        # Docling mode requires docling + docling_core
        try:
            import docling
        except ImportError:
            missing.append("docling")

        try:
            import docling_core
        except ImportError:
            missing.append("docling-core")

        install_cmd = "uv pip install pymupdf docling docling-core"
    else:
        # Fast mode requires pymupdf4llm
        try:
            import pymupdf4llm
        except ImportError:
            missing.append("pymupdf4llm")

        install_cmd = "uv pip install pymupdf pymupdf4llm"

    if missing:
        print(f"ERROR: Missing dependencies: {', '.join(missing)}", file=sys.stderr)
        print(f"Install with: {install_cmd}", file=sys.stderr)
        return False

    return True


class PageRangeError(ValueError):
    """Error raised when page range string is invalid."""

    pass


def parse_page_range(page_str, total_pages):
    """Parse page range string like '1-5' or '1,3,5-7'.

    Raises:
        PageRangeError: If the page range string is invalid (non-numeric, invalid range, etc.)
    """
    if not page_str:
        return None

    pages = []
    requested_any = False  # Track if user specified at least one page token

    for part in page_str.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            parts = part.split("-", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise PageRangeError(
                    f"Invalid range '{part}'. Expected format: start-end (e.g., '1-5')"
                )
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError:
                raise PageRangeError(
                    f"Invalid range '{part}'. Page numbers must be integers."
                )

            if start > end:
                raise PageRangeError(
                    f"Invalid range '{part}'. Start page ({start}) cannot be greater than end page ({end})."
                )
            if start < 1:
                raise PageRangeError(
                    f"Invalid range '{part}'. Page numbers must be >= 1."
                )

            requested_any = True  # User specified a valid range
            # Convert to 0-indexed
            start_idx = start - 1
            # end is inclusive, so no -1 for end
            pages.extend(range(start_idx, min(end, total_pages)))
        else:
            try:
                page = int(part)
            except ValueError:
                raise PageRangeError(
                    f"Invalid page number '{part}'. Page numbers must be integers."
                )
            if page < 1:
                raise PageRangeError(
                    f"Invalid page number '{part}'. Page numbers must be >= 1."
                )
            requested_any = True  # User specified a valid page number
            pages.append(page - 1)

    result = sorted(set(p for p in pages if 0 <= p < total_pages))

    if not result and requested_any:
        # User specified pages but all are out of range
        raise PageRangeError(
            f"All requested pages are out of range. Document has {total_pages} pages."
        )

    return result


def extract_referenced_images(markdown: str) -> set:
    """
    Extract the set of image filenames referenced in markdown.

    Returns:
        Set of image filenames (without directory path)
    """
    # Match markdown image references: ![alt](path)
    pattern = r"!\[[^\]]*\]\(([^)]+)\)"
    matches = re.findall(pattern, markdown)
    # Extract just the filename from each path
    return {Path(m).name for m in matches}


def get_image_info(image_dir, referenced_only: set = None):
    """
    Get information about extracted images.

    Args:
        image_dir: Directory containing images
        referenced_only: If provided, only include images with filenames in this set

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
        if img_path.suffix.lower() in (
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".webp",
        ):
            # Filter to only referenced images if specified
            if referenced_only is not None and img_path.name not in referenced_only:
                continue

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

                images.append(
                    {
                        "filename": img_path.name,
                        "path": str(img_path),
                        "size_kb": round(size_kb, 1),
                        "dimensions": dimensions,
                    }
                )
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
        filename_raw = match.group(2)
        # Strip any directory components (e.g., "images/figure_0001.png" -> "figure_0001.png")
        # This handles Docling's output which includes "images/" prefix
        filename = Path(filename_raw).name
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

                return f"![{alt_text}]({filename_raw})\n\n**[Image: {filename} ({dims}, {size_kb}KB) → {full_path}]**"
            except:
                return f"![{alt_text}]({filename_raw})\n\n**[Image: {filename} → {full_path}]**"

        return match.group(0)

    pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
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


def convert_pdf(
    pdf_path,
    image_dir=None,
    no_images=False,
    show_progress=False,
    docling=False,
    images_scale=4.0,
):
    """
    Convert PDF to markdown.

    Args:
        pdf_path: Path to PDF file
        image_dir: Directory to extract images to
        no_images: Skip image extraction
        show_progress: Show progress output
        docling: Use Docling AI for high-accuracy tables (slower)
        images_scale: Image resolution multiplier for Docling mode (default: 4.0)
    """
    if docling:
        from extractor import extract_pdf_docling

        # Docling extracts both text and images together
        markdown, _image_paths = extract_pdf_docling(
            pdf_path,
            output_dir=image_dir if not no_images else None,
            images_scale=images_scale,
            show_progress=show_progress,
        )
        return markdown
    else:
        from extractor import extract_pdf_to_markdown, extract_images

        # Fast mode: separate text and image extraction
        markdown = extract_pdf_to_markdown(
            pdf_path, accurate=False, show_progress=show_progress
        )

        if not no_images and image_dir:
            extract_images(pdf_path, image_dir, show_progress=show_progress)

        return markdown


def add_metadata_header(
    markdown, pdf_path, total_pages, pages_extracted, image_dir=None, cached=False
):
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
    """
    Create temporary image directory for extraction.

    Uses tempfile.mkdtemp() for:
    - Unique directory names (safe for concurrent runs)
    - Cross-platform compatibility (works on Windows)
    """
    pdf_name = Path(pdf_path).stem
    safe_name = re.sub(r"[^\w\-_]", "_", pdf_name)
    # Create unique temp directory with prefix based on PDF name
    image_dir = tempfile.mkdtemp(prefix=f"pdf_images_{safe_name}_")
    return image_dir


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown for LLM context (with persistent caching)",
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
        """,
    )

    parser.add_argument("input", nargs="?", help="Input PDF file path")
    parser.add_argument(
        "output", nargs="?", help="Output markdown file path (default: <input>.md)"
    )
    parser.add_argument(
        "--stdout", action="store_true", help="Print to stdout instead of file"
    )
    parser.add_argument(
        "--pages",
        help='Page range to extract (e.g., "1-5" or "1,3,5-7"). Note: only effective with fast mode (pymupdf4llm); Docling mode always extracts full document.',
    )
    parser.add_argument(
        "--docling",
        "--accurate",
        action="store_true",
        dest="docling",
        help="Use Docling AI for complex/borderless tables (slower, ~1 sec/page)",
    )
    parser.add_argument(
        "--images-scale",
        type=float,
        default=4.0,
        help="Image resolution multiplier for Docling mode (default: 4.0)",
    )
    parser.add_argument(
        "--no-images", action="store_true", help="Skip image extraction (faster)"
    )
    parser.add_argument(
        "--no-metadata", action="store_true", help="Skip metadata header"
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress indicator"
    )

    # Cache options
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass cache entirely (no read or write)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache for this PDF before processing",
    )
    parser.add_argument(
        "--clear-all-cache",
        action="store_true",
        help="Clear entire cache directory and exit",
    )
    parser.add_argument(
        "--cache-stats", action="store_true", help="Show cache statistics and exit"
    )
    parser.add_argument(
        "--force-stale-cache",
        action="store_true",
        help="Use cached extraction even if extractor version differs (when PDF missing)",
    )

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

    # Handle --clear-cache before existence check (allows clearing cache for deleted PDFs)
    if args.clear_cache:
        if clear_cache(args.input):
            print(f"Cache cleared for: {args.input}", file=sys.stderr)
        else:
            print(f"No cache found for: {args.input}", file=sys.stderr)
        # If only clearing cache and file doesn't exist, exit successfully
        if not os.path.exists(args.input):
            sys.exit(0)

    # Validate input and check for cache fallback
    pdf_exists = os.path.exists(args.input)
    cache_fallback = False
    fallback_cache_dir = None

    if not pdf_exists:
        # Try to find cached extraction by source path, filtered by requested mode
        if not args.no_cache:
            # First try to find cache matching the requested mode/scale
            matching_caches = find_cache_by_source_path(
                args.input,
                docling=args.docling,
                images_scale=args.images_scale if args.docling else None,
            )
            # If no exact match, try any cache for this file
            if not matching_caches:
                matching_caches = find_cache_by_source_path(args.input)

            if matching_caches:
                # Use the freshest matching cache (already sorted by cached_at desc)
                fallback_cache_dir, fallback_metadata = matching_caches[0]

                # Check extractor version compatibility
                from extractor import EXTRACTOR_VERSION

                cached_version = fallback_metadata.get("extractor_version")
                if cached_version != EXTRACTOR_VERSION and not args.force_stale_cache:
                    print(
                        f"ERROR: Cached extraction version mismatch",
                        file=sys.stderr,
                    )
                    print(
                        f"  Cached: {cached_version}, Current: {EXTRACTOR_VERSION}",
                        file=sys.stderr,
                    )
                    print(
                        f"  Re-extract with original PDF or use --force-stale-cache",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                cache_fallback = True
                cached_mode = fallback_metadata.get("mode", "unknown")
                version_warning = ""
                if cached_version != EXTRACTOR_VERSION:
                    version_warning = f" [version {cached_version}, current is {EXTRACTOR_VERSION}]"
                print(
                    f"WARNING: Source PDF not found, using cached extraction ({cached_mode} mode){version_warning}",
                    file=sys.stderr,
                )
            else:
                print(f"ERROR: File not found: {args.input}", file=sys.stderr)
                print(
                    "  (No cached extraction available either)",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            print(f"ERROR: File not found: {args.input}", file=sys.stderr)
            sys.exit(1)

    if pdf_exists and not args.input.lower().endswith(".pdf"):
        print(f"WARNING: File may not be a PDF: {args.input}", file=sys.stderr)

    # Check dependencies for the requested mode (skip if using cache fallback)
    if not cache_fallback and not check_dependencies(docling_mode=args.docling):
        sys.exit(1)

    # Get total pages (from PDF or from cached metadata)
    if cache_fallback:
        # Use metadata we already loaded during cache lookup
        total_pages = fallback_metadata.get("total_pages", 0)
    else:
        from extractor import get_page_count

        total_pages = get_page_count(args.input)

    # Parse page range (for output slicing, not extraction)
    try:
        requested_pages = (
            parse_page_range(args.pages, total_pages) if args.pages else None
        )
    except PageRangeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "  Expected format: 1-5 or 1,3,5-7 (page numbers start at 1)",
            file=sys.stderr,
        )
        sys.exit(1)
    # Warn if --pages used with --docling (page slicing not supported)
    # Clear requested_pages and use total_pages so metadata doesn't lie
    if args.pages and args.docling:
        print(
            "WARNING: --pages is not supported in Docling mode (no page delimiters in output). "
            "Full document will be returned.",
            file=sys.stderr,
        )
        requested_pages = None  # Don't attempt slicing

    pages_to_output = len(requested_pages) if requested_pages else total_pages

    # Determine if progress should be shown
    show_progress = sys.stderr.isatty() and not args.no_progress and not args.stdout

    # Check cache
    cache_hit = False
    cache_key = ""
    result = None
    image_dir = None

    # If using cache fallback (PDF missing), load directly from the found cache
    if cache_fallback:
        cache_key = fallback_cache_dir.name
        result, image_dir, cached_total = load_from_cache(
            cache_key, requested_pages, no_images=args.no_images
        )
        if result is None:
            # Cache was corrupted and PDF doesn't exist - can't recover
            print(
                "ERROR: Cache was corrupted and source PDF is not available.",
                file=sys.stderr,
            )
            sys.exit(1)
        cache_hit = True
    elif not args.no_cache:
        valid, cache_key = is_cache_valid(
            args.input, docling=args.docling, images_scale=args.images_scale
        )
        if valid:
            if show_progress:
                mode = "docling" if args.docling else "fast"
                print(f"Loading from cache ({mode} mode)...", file=sys.stderr)
            result, image_dir, cached_total = load_from_cache(
                cache_key, requested_pages, no_images=args.no_images
            )
            # If cache was corrupted, treat as cache miss (will re-extract below)
            if result is not None:
                cache_hit = True

    # If no cache hit, extract full PDF
    if not cache_hit:
        # Get cache key if we don't have it
        if not cache_key:
            cache_key = get_cache_key(
                args.input, docling=args.docling, images_scale=args.images_scale
            )

        # Setup image directory for extraction (temporary)
        temp_image_dir = None
        if not args.no_images:
            temp_image_dir = setup_temp_image_dir(args.input)

        # Extract FULL PDF
        try:
            if show_progress:
                if args.docling:
                    print(
                        f"Extracting {total_pages} pages with Docling AI (~1 sec/page)...",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"Extracting {total_pages} pages with PyMuPDF (fast mode)...",
                        file=sys.stderr,
                    )

            result = convert_pdf(
                args.input,
                image_dir=temp_image_dir,
                no_images=args.no_images,
                show_progress=show_progress,
                docling=args.docling,
                images_scale=args.images_scale,
            )
        except Exception as e:
            print(f"ERROR: Conversion failed: {e}", file=sys.stderr)
            sys.exit(1)

        # Save to cache (full result)
        if not args.no_cache:
            save_to_cache(
                cache_key,
                result,
                temp_image_dir,
                args.input,
                total_pages,
                docling=args.docling,
                images_scale=args.images_scale,
            )
            if show_progress:
                print(f"Cached: {get_cache_dir(cache_key)}", file=sys.stderr)

        # Set image_dir for output
        # When --no-cache is set, always use temp_image_dir (don't reference stale cache)
        # Otherwise, prefer cached location if available
        if not args.no_images:
            if args.no_cache:
                image_dir = temp_image_dir
            else:
                cached_image_dir = get_cache_dir(cache_key) / "images"
                if cached_image_dir.exists() and any(cached_image_dir.iterdir()):
                    image_dir = cached_image_dir
                    # Clean up temp directory since images are now in cache
                    if temp_image_dir and os.path.exists(temp_image_dir):
                        shutil.rmtree(temp_image_dir)
                else:
                    image_dir = temp_image_dir

        # Slice pages if requested (after caching full result)
        if requested_pages:
            result = slice_pages_from_markdown(result, requested_pages, total_pages)

    # Determine output path early (needed for --no-cache image handling)
    output_path = None
    if not args.stdout:
        output_path = args.output or os.path.splitext(args.input)[0] + ".md"

    # Handle --no-cache image directory: copy to output location or warn
    if args.no_cache and image_dir and not args.no_images:
        if output_path:
            # Copy images to a directory next to the output file (e.g., doc_images/)
            output_images_dir = Path(str(output_path).rsplit(".", 1)[0] + "_images")
            if Path(image_dir).exists() and any(Path(image_dir).iterdir()):
                if output_images_dir.exists():
                    shutil.rmtree(output_images_dir)
                shutil.copytree(image_dir, output_images_dir)
                # Clean up temp directory
                if os.path.exists(image_dir):
                    shutil.rmtree(image_dir)
                image_dir = output_images_dir
                if show_progress:
                    print(f"Images copied to: {output_images_dir}", file=sys.stderr)
        else:
            # Outputting to stdout with --no-cache: warn about ephemeral paths
            print(
                f"WARNING: --no-cache with stdout: image paths reference temporary directory {image_dir} which may be cleaned up by the system.",
                file=sys.stderr,
            )

    # Format output
    output = result

    # Extract referenced images before enhancement (for filtering summary)
    # This ensures we only show images that are actually in the sliced output
    referenced_images = extract_referenced_images(result) if result else set()

    # Enhance image references with full paths (skip if --no-images)
    if image_dir and not args.no_images:
        output = enhance_markdown_with_image_paths(output, image_dir)

        # Add image summary table at the end (filtered to referenced images only)
        images = get_image_info(image_dir, referenced_only=referenced_images)
        if images:
            output += create_image_summary(images)

    if not args.no_metadata:
        output = add_metadata_header(
            output,
            args.input,
            total_pages,
            pages_to_output,
            image_dir,
            cached=cache_hit,
        )

    # Write output
    if args.stdout:
        print(output)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)

        msg = f"Converted {pages_to_output} pages to: {output_path}"
        if cache_hit:
            msg += " (from cache)"
        if image_dir and not args.no_images:
            # Use the same filtered image set for consistency
            images = get_image_info(image_dir, referenced_only=referenced_images)
            if images:
                msg += f" ({len(images)} images)"
        print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
