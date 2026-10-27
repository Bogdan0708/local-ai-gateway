"""
Remove inline <think>...</think> reasoning from model output.

Some local models (reasoning distills) write their thinking into the message content instead
of a separate reasoning field, and the backend cannot always be told to split it out. Clients
of this gateway want the answer only, so every chat path filters it here.
"""

import re

OPEN, CLOSE = "<think>", "</think>"
_BLOCK = re.compile(re.escape(OPEN) + r".*?" + re.escape(CLOSE), re.DOTALL)


def strip_think(text: str) -> str:
    """Return `text` without think blocks; unchanged when it contains no think tags."""
    if not text or (OPEN not in text and CLOSE not in text):
        return text
    head, sep, tail = text.partition(CLOSE)
    if sep and OPEN not in head:
        # The chat template opened the block in the prompt: only the closing tag is generated.
        text = tail
    text = _BLOCK.sub("", text)
    unterminated = text.find(OPEN)  # max_tokens hit mid-thought: there is no answer yet
    if unterminated != -1:
        text = text[:unterminated]
    return text.strip()


def _partial_tag_len(buf: str, tag: str) -> int:
    """Length of the longest suffix of `buf` that is a proper prefix of `tag`."""
    for n in range(min(len(buf), len(tag) - 1), 0, -1):
        if tag.startswith(buf[-n:]):
            return n
    return 0


class ThinkStreamFilter:
    """
    Streaming counterpart of `strip_think` for content deltas.

    Tags may arrive split across chunks, so a possible partial tag is held back until the
    next chunk decides it. A block opened by the prompt template (no `<think>` generated)
    cannot be detected without buffering the whole stream and is passed through.
    """

    def __init__(self):
        self._buf = ""
        self._inside = False
        self._after_block = False  # trim the blank lines models put between thinking and answer

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out = []
        while True:
            tag = CLOSE if self._inside else OPEN
            i = self._buf.find(tag)
            if i == -1:
                keep = _partial_tag_len(self._buf, tag)
                ready, self._buf = self._buf[:len(self._buf) - keep], self._buf[len(self._buf) - keep:]
                if not self._inside:
                    out.append(ready)
                break
            if not self._inside:
                out.append(self._buf[:i])
            self._buf = self._buf[i + len(tag):]
            self._inside = not self._inside
            if not self._inside:
                self._after_block = True
        return self._visible("".join(out))

    def flush(self) -> str:
        rest = "" if self._inside else self._buf
        self._buf = ""
        return self._visible(rest)

    def _visible(self, text: str) -> str:
        if self._after_block:
            text = text.lstrip()
            if text:
                self._after_block = False
        return text
