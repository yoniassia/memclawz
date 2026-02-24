#!/usr/bin/env python3.10
"""
Markdown chunker for QMDZvec.
Splits .md files into searchable chunks by heading sections or fixed-size windows.
"""
import os
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    path: str
    start_line: int
    end_line: int
    heading: str = ""


def chunk_by_heading(text: str, path: str = "", min_size: int = 50) -> list[Chunk]:
    """Split markdown by headings (##, ###, etc). Merge tiny sections."""
    lines = text.split("\n")
    chunks = []
    current_lines = []
    current_start = 1
    current_heading = ""

    for i, line in enumerate(lines, 1):
        if re.match(r'^#{1,4}\s+', line) and current_lines:
            body = "\n".join(current_lines).strip()
            if len(body) >= min_size:
                chunks.append(Chunk(
                    text=body, path=path,
                    start_line=current_start, end_line=i - 1,
                    heading=current_heading
                ))
            elif chunks:
                # Merge tiny section into previous
                chunks[-1].text += "\n\n" + body
                chunks[-1].end_line = i - 1
            current_lines = [line]
            current_start = i
            current_heading = line.strip("# ").strip()
        else:
            current_lines.append(line)
            if not current_heading and re.match(r'^#{1,4}\s+', line):
                current_heading = line.strip("# ").strip()

    # Last chunk
    if current_lines:
        body = "\n".join(current_lines).strip()
        if len(body) >= min_size:
            chunks.append(Chunk(
                text=body, path=path,
                start_line=current_start, end_line=len(lines),
                heading=current_heading
            ))
        elif chunks:
            chunks[-1].text += "\n\n" + body
            chunks[-1].end_line = len(lines)

    return chunks


def chunk_by_window(text: str, path: str = "", window_size: int = 500, overlap: int = 100) -> list[Chunk]:
    """Fixed-size character window chunking with overlap."""
    chunks = []
    lines = text.split("\n")
    flat = text
    pos = 0
    while pos < len(flat):
        end = min(pos + window_size, len(flat))
        chunk_text = flat[pos:end].strip()
        if chunk_text:
            # Approximate line numbers
            start_line = flat[:pos].count("\n") + 1
            end_line = flat[:end].count("\n") + 1
            chunks.append(Chunk(
                text=chunk_text, path=path,
                start_line=start_line, end_line=end_line
            ))
        pos += window_size - overlap
    return chunks


def chunk_file(filepath: str, method: str = "heading", **kwargs) -> list[Chunk]:
    """Chunk a markdown file."""
    with open(filepath) as f:
        text = f.read()
    if method == "heading":
        return chunk_by_heading(text, path=filepath, **kwargs)
    else:
        return chunk_by_window(text, path=filepath, **kwargs)


def chunk_directory(dirpath: str, method: str = "heading", extensions: tuple = (".md",), **kwargs) -> list[Chunk]:
    """Chunk all matching files in a directory."""
    all_chunks = []
    for root, dirs, files in os.walk(dirpath):
        for f in sorted(files):
            if any(f.endswith(ext) for ext in extensions):
                fp = os.path.join(root, f)
                all_chunks.extend(chunk_file(fp, method=method, **kwargs))
    return all_chunks
