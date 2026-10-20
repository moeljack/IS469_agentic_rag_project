"""
Clean scraped web pages and split them into section-aware chunks.

The scraped text has no blank lines and almost no Markdown headings, so structure is recovered from the lines
themselves: Markdown headings (`## Title`) if the scraper provides them, otherwise short Title-Case lines. A wrong
guess only costs a slightly odd heading prefix; the text itself is never dropped.
"""
import re

TABLE_SEPARATOR = re.compile(r"^\|[\s:|-]+\|$")
FILE_TAG = re.compile(r"\s*\((?:PDF|XLSX?|DOCX?|PPTX?|ZIP|MP4)[^)]*\)", re.I)   # "(PDF, 61KB)"
BUTTON = re.compile(r"^(?:DOWNLOAD|VIEW|APPLY|SUBMIT|START)\b[A-Z0-9 ]*$")        # "DOWNLOAD RECORD KEEPING ASSISTANT"
EXAMPLE = re.compile(r"^(?:Example|Scenario|Case|Step)\s*\d*\b", re.I)            # sub-section under a heading
SMALL_WORDS = {"a", "an", "and", "of", "or", "the", "to", "for", "in", "on", "with", "by", "at", "from", "as",
               "is", "are", "be", "your", "you", "that", "this", "if", "vs"}
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def clean_title(title, publisher=""):
    title = re.sub(r"^\s*" + re.escape(publisher) + r"\s*\|\s*", "", title) if publisher else title
    return title.strip()


def clean_text(text):
    lines = []
    for line in text.replace(" ", " ").replace("​", "").split("\n"):
        line = FILE_TAG.sub("", line)
        line = re.sub(r"[ \t]{2,}", " ", line).strip()
        if not line or BUTTON.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def _is_heading(line, next_line):
    """Title-Case heuristic: short, no sentence punctuation at the end, mostly capitalised words."""
    if next_line is None or not 3 <= len(line) <= 90 or line[0] in "-*|#[" or line[-1] in ".:;,?!":
        return False
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'/&-]*", line)]
    significant = [w for w in words if w.lower() not in SMALL_WORDS]
    if not significant or len(words) > 14:
        return False
    return sum(w[0].isupper() for w in significant) / len(significant) >= 0.75


def parse_blocks(text):
    """Turn cleaned text into ('heading', level, text) and ('block', kind, lines) items."""
    lines = text.split("\n")
    use_markdown = any(re.match(r"#{1,6} ", l) for l in lines)
    items, i = [], 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if use_markdown and (m := re.match(r"(#{1,6}) (.+)", line)):
            items.append(("heading", len(m.group(1)), m.group(2).strip()))
        elif not use_markdown and EXAMPLE.match(line) and len(line) <= 90 and nxt is not None:
            items.append(("heading", 2, line.rstrip(":")))
        elif not use_markdown and _is_heading(line, nxt):
            items.append(("heading", 1, line))
        elif line.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            items.append(("block", "table", lines[i:j]))
            i = j
            continue
        elif line[:2] in ("- ", "* "):
            j = i
            while j < len(lines) and lines[j][:2] in ("- ", "* "):
                j += 1
            items.append(("block", "list", lines[i:j]))
            i = j
            continue
        else:
            items.append(("block", "para", [line]))
        i += 1

    merged = []   # keep an intro line ("...are able to:") together with the list or table it introduces
    for item in items:
        if (item[0] == "block" and item[1] in ("list", "table") and merged and merged[-1][0] == "block"
                and merged[-1][1] == "para" and merged[-1][2][-1].endswith(":")):
            intro = merged.pop()
            item = ("block", item[1], intro[2] + item[2], intro[2])
        merged.append(item)
    return merged


def _split_block(kind, lines, fits, intro_len=0):
    """Split one over-long block into pieces that each satisfy fits(text)."""
    if kind == "table":
        body_start = intro_len
        header = []
        if len(lines) - body_start >= 2 and TABLE_SEPARATOR.match(lines[body_start + 1]):
            header = lines[body_start:body_start + 2]
            body_start += 2
        pieces, current = [], list(lines[:body_start])   # first piece keeps intro + header
        for row in lines[body_start:]:
            if len(current) > len(header) and not fits("\n".join(current + [row])):
                pieces.append("\n".join(current))
                current = list(header)
            current.append(row)
        pieces.append("\n".join(current))
        return pieces
    if kind == "list":
        units = lines
    else:   # paragraph: split on sentences, then on words as a last resort
        units = [s for line in lines for s in SENTENCE_END.split(line)]
    pieces, current = [], ""
    for unit in units:
        if not fits(unit):
            words, unit_parts, buf = unit.split(), [], ""
            for w in words:
                if buf and not fits(buf + " " + w):
                    unit_parts.append(buf)
                    buf = w
                else:
                    buf = f"{buf} {w}".strip()
            unit_parts.append(buf)
        else:
            unit_parts = [unit]
        for part in unit_parts:
            joined = f"{current}\n{part}" if kind == "list" and current else f"{current} {part}".strip()
            if current and not fits(joined):
                pieces.append(current)
                current = part
            else:
                current = joined
    if current:
        pieces.append(current)
    return pieces


def chunk_page(title, text, count_tokens, max_tokens):
    """Section-aware chunks for one page: [{'headings': 'Title > Section', 'text': ..., 'n_tokens': ...}].

    count_tokens measures the text that will be embedded (heading path + chunk), so chunks fit max_tokens.
    """
    chunks, stack = [], []   # stack of (level, text)
    current = {"path": None, "parts": []}

    def path_str():
        return " > ".join([title] + [t for _, t in stack])

    def flush():
        if current["parts"]:
            body = "\n".join(current["parts"])
            heading = current["path"]
            chunks.append({"headings": heading, "text": body, "n_tokens": count_tokens(f"{heading}\n{body}")})
        current["parts"] = []

    for item in parse_blocks(text):
        if item[0] == "heading":
            flush()
            _, level, heading = item
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            continue

        _, kind, lines, *rest = item
        intro_len = len(rest[0]) if rest else 0
        path = path_str()
        if current["path"] != path:
            flush()
            current["path"] = path
        budget_fits = lambda t, p=path: count_tokens(f"{p}\n{t}") <= max_tokens

        block_text = "\n".join(lines)
        if budget_fits("\n".join(current["parts"] + [block_text])):
            current["parts"].append(block_text)
            continue
        flush()
        current["path"] = path
        if budget_fits(block_text):
            current["parts"].append(block_text)
            continue
        for piece in _split_block(kind, lines, budget_fits, intro_len):
            current["parts"] = [piece]
            flush()
            current["path"] = path
    flush()
    return chunks
