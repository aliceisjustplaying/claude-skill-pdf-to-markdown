#!/usr/bin/env python3
"""
PDF to Markdown Converter for LLM Context
Extracts entire PDF content as clean, structured markdown.
Images are extracted to /tmp/pdf_images/<pdf_name>/ by default.

Usage:
    python pdf_to_md.py <input.pdf> [output.md]
    python pdf_to_md.py <input.pdf> --stdout
    python pdf_to_md.py <input.pdf> --pages 1-5
    python pdf_to_md.py <input.pdf> --no-images

Dependencies:
    uv pip install pymupdf4llm
"""

import argparse
import sys
import os
import re
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count

# Minimum pages to use parallel processing (overhead not worth it for small PDFs)
PARALLEL_THRESHOLD = 100

# Default image output directory
DEFAULT_IMAGE_DIR = "/tmp/pdf_images"


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


def setup_image_dir(pdf_path, custom_dir=None):
    """
    Create and return image directory for this PDF.

    Returns:
        Path to image directory (e.g., /tmp/pdf_images/document_name/)
    """
    pdf_name = Path(pdf_path).stem
    # Sanitize name for filesystem
    safe_name = re.sub(r'[^\w\-_]', '_', pdf_name)

    if custom_dir:
        image_dir = Path(custom_dir) / safe_name
    else:
        image_dir = Path(DEFAULT_IMAGE_DIR) / safe_name

    # Clean up existing directory to avoid stale images
    if image_dir.exists():
        shutil.rmtree(image_dir)

    image_dir.mkdir(parents=True, exist_ok=True)
    return str(image_dir)


def get_image_info(image_dir):
    """
    Get information about extracted images.

    Returns:
        List of dicts with image metadata
    """
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
    pymupdf4llm outputs: ![description](filename.png)
    We convert to: ![description](filename.png)\n[Image: filename.png → /full/path/filename.png]
    """
    image_dir = Path(image_dir)

    def replace_image_ref(match):
        alt_text = match.group(1)
        filename = match.group(2)
        full_path = image_dir / filename

        if full_path.exists():
            # Get image info
            try:
                size_kb = round(full_path.stat().st_size / 1024, 1)
                try:
                    import pymupdf
                    pix = pymupdf.Pixmap(str(full_path))
                    dims = f"{pix.width}x{pix.height}"
                    pix = None
                except:
                    dims = "?"

                # Return enhanced reference
                return f"![{alt_text}]({filename})\n\n**[Image: {filename} ({dims}, {size_kb}KB) → {full_path}]**"
            except:
                return f"![{alt_text}]({filename})\n\n**[Image: {filename} → {full_path}]**"

        return match.group(0)

    # Match markdown image syntax: ![alt](filename)
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
    """
    Worker function: process a batch of pages and return markdown.

    Args:
        args: Tuple of (pdf_path, page_numbers, kwargs)

    Returns:
        Tuple of (first_page_number, markdown_result)
    """
    pdf_path, page_numbers, kwargs = args
    import pymupdf4llm

    # Process this batch of pages
    result = pymupdf4llm.to_markdown(pdf_path, pages=page_numbers, **kwargs)

    # Return with first page number for ordering
    return (page_numbers[0], result)


def convert_parallel(pdf_path, pages_to_process, workers, batch_size,
                     image_dir=None, no_images=False, show_progress=False):
    """
    Convert PDF to markdown using parallel processing.

    Args:
        pdf_path: Path to PDF file
        pages_to_process: List of page numbers (0-indexed) to process
        workers: Number of worker processes
        batch_size: Pages per batch
        image_dir: Directory to save images (None = no images)
        no_images: If True, skip image extraction entirely
        show_progress: If True, show progress indicator

    Returns:
        Markdown string
    """
    # Build kwargs for to_markdown (excluding pages, which we handle per-batch)
    kwargs = {
        'show_progress': False,  # Disable per-batch progress, we show our own
    }

    if not no_images and image_dir:
        kwargs['write_images'] = True
        kwargs['image_path'] = image_dir

    # Create page batches
    batches = []
    for i in range(0, len(pages_to_process), batch_size):
        batch_pages = pages_to_process[i:i + batch_size]
        batches.append((pdf_path, batch_pages, kwargs))

    total_batches = len(batches)
    results = []

    if show_progress:
        print(f"Processing {len(pages_to_process)} pages in {total_batches} batches using {workers} workers...", file=sys.stderr)

    # Process in parallel with progress reporting
    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_page_batch, batches)):
            results.append(result)
            if show_progress:
                completed = i + 1
                print(f"\rBatch {completed}/{total_batches} complete", end="", file=sys.stderr)

    if show_progress:
        print(file=sys.stderr)  # Newline after progress

    # Sort results by first page number and merge
    results.sort(key=lambda x: x[0])
    return "\n\n".join(r[1] for r in results)


def convert_sequential(pdf_path, pages=None, chunked=False,
                       image_dir=None, no_images=False, show_progress=False):
    """
    Convert PDF to markdown using single-threaded processing.

    Args:
        pdf_path: Path to PDF file
        pages: List of page numbers (0-indexed) or None for all
        chunked: If True, return list of page chunks with metadata
        image_dir: Directory to save images (None = no images)
        no_images: If True, skip image extraction entirely
        show_progress: If True, show progress indicator

    Returns:
        Markdown string or list of chunk dicts
    """
    import pymupdf4llm

    # Build kwargs for to_markdown
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

    # Convert
    return pymupdf4llm.to_markdown(pdf_path, **kwargs)


def add_metadata_header(markdown, pdf_path, total_pages, pages_extracted, image_dir=None):
    """Add metadata header to markdown output."""
    from datetime import datetime

    filename = os.path.basename(pdf_path)

    header_lines = [
        "---",
        f"source: {filename}",
        f"total_pages: {total_pages}",
        f"pages_extracted: {pages_extracted}",
        f"extracted_at: {datetime.now().isoformat()}",
    ]

    if image_dir:
        header_lines.append(f"images_dir: {image_dir}")

    header_lines.extend(["---", "", ""])

    return "\n".join(header_lines) + markdown


def main():
    parser = argparse.ArgumentParser(
        description='Convert PDF to Markdown for LLM context',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_to_md.py document.pdf                    # Output to document.md (images extracted)
  python pdf_to_md.py document.pdf output.md         # Specify output file
  python pdf_to_md.py document.pdf --stdout          # Print to stdout
  python pdf_to_md.py document.pdf --pages 1-10      # Only pages 1-10
  python pdf_to_md.py document.pdf --pages 1,3,5-7   # Specific pages
  python pdf_to_md.py document.pdf --no-images       # Skip image extraction
  python pdf_to_md.py document.pdf --image-dir /tmp  # Custom image directory
  python pdf_to_md.py document.pdf --chunked         # Output as JSON chunks
  python pdf_to_md.py document.pdf --no-metadata     # Skip metadata header
  python pdf_to_md.py document.pdf --workers 4       # Use 4 parallel workers
  python pdf_to_md.py document.pdf --no-parallel     # Force single-threaded

Images:
  By default, images are extracted to /tmp/pdf_images/<pdf_name>/
  Image references in markdown include full paths for easy viewing.
  Use --no-images to skip extraction (faster, smaller output).
        """
    )

    parser.add_argument('input', help='Input PDF file path')
    parser.add_argument('output', nargs='?', help='Output markdown file path (default: <input>.md)')
    parser.add_argument('--stdout', action='store_true', help='Print to stdout instead of file')
    parser.add_argument('--pages', help='Page range to extract (e.g., "1-5" or "1,3,5-7")')
    parser.add_argument('--chunked', action='store_true', help='Output as JSON with page chunks and metadata')
    parser.add_argument('--no-images', action='store_true', help='Skip image extraction (faster)')
    parser.add_argument('--image-dir', help=f'Directory for extracted images (default: {DEFAULT_IMAGE_DIR}/<pdf_name>/)')
    parser.add_argument('--no-metadata', action='store_true', help='Skip metadata header')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress indicator')
    parser.add_argument('--workers', type=int, default=None,
                        help=f'Number of parallel workers (default: CPU count, currently {cpu_count()})')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Pages per worker batch (default: 50)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not args.input.lower().endswith('.pdf'):
        print(f"WARNING: File may not be a PDF: {args.input}", file=sys.stderr)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Get total pages first for page range parsing
    import pymupdf
    doc = pymupdf.open(args.input)
    total_pages = len(doc)
    doc.close()

    # Parse page range
    pages = parse_page_range(args.pages, total_pages) if args.pages else None
    pages_to_process = pages if pages else list(range(total_pages))
    pages_extracted = len(pages_to_process)

    # Setup image directory (unless --no-images)
    image_dir = None
    if not args.no_images:
        image_dir = setup_image_dir(args.input, args.image_dir)

    # Determine if progress should be shown (default: yes for TTY, no for stdout/piped)
    show_progress = sys.stderr.isatty() and not args.no_progress and not args.stdout

    # Determine if we should use parallel processing
    workers = args.workers or cpu_count()
    use_parallel = (
        not args.no_parallel
        and not args.chunked  # Chunked mode not supported in parallel
        and pages_extracted >= PARALLEL_THRESHOLD
        and workers > 1
    )

    # Convert
    try:
        if use_parallel:
            result = convert_parallel(
                args.input,
                pages_to_process=pages_to_process,
                workers=workers,
                batch_size=args.batch_size,
                image_dir=image_dir,
                no_images=args.no_images,
                show_progress=show_progress
            )
        else:
            if show_progress and pages_extracted > 50:
                mode = "sequential (--chunked mode)" if args.chunked else "sequential"
                print(f"Processing {pages_extracted} pages ({mode})...", file=sys.stderr)

            result = convert_sequential(
                args.input,
                pages=pages,
                chunked=args.chunked,
                image_dir=image_dir,
                no_images=args.no_images,
                show_progress=show_progress
            )
    except Exception as e:
        print(f"ERROR: Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Format output
    if args.chunked:
        import json
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
            output = add_metadata_header(output, args.input, total_pages, pages_extracted, image_dir)

    # Write output
    if args.stdout:
        print(output)
    else:
        output_path = args.output or os.path.splitext(args.input)[0] + '.md'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output)

        # Summary message
        msg = f"Converted {pages_extracted} pages to: {output_path}"
        if image_dir:
            images = get_image_info(image_dir)
            if images:
                msg += f" ({len(images)} images → {image_dir})"
        print(msg, file=sys.stderr)


if __name__ == '__main__':
    main()
