"""Tool-poisoning scanner and sanitiser for text copied from an API spec.

Generated tool descriptions are read by a language model, so any instruction hidden
in a spec's summary or description becomes a prompt injection (a "tool poisoning"
attack). This module flags and neutralises the common patterns reported in the
MCP security literature: hidden instruction blocks, requests to read secrets,
exfiltration phrasing, invisible Unicode and oversized descriptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_DESCRIPTION_CHARS = 1200

_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿\U000e0000-\U000e007f]")
_HIDDEN_BLOCK = re.compile(r"<\s*(important|system|instructions?|secret|hidden)[^>]*>.*?<\s*/\s*\1\s*>", re.I | re.S)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)

PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("high", "instruction override",
     re.compile(r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|earlier|all)\b.{0,20}\b(instructions?|prompts?|rules)\b", re.I)),
    ("high", "concealment request",
     re.compile(r"\b(do not|don't|never)\b.{0,20}\b(tell|inform|mention|reveal|show)\b.{0,20}\b(user|human|anyone)\b", re.I)),
    ("high", "secret file access",
     re.compile(r"(~/\.ssh|id_rsa|id_ed25519|\.aws/credentials|/etc/passwd|/etc/shadow|\.env\b|mcp\.json|"
                r"\.kube/config|\.npmrc|\.netrc)", re.I)),
    ("low", "mentions key material", re.compile(r"private[_ ]key", re.I)),
    ("medium", "exfiltration phrasing",
     re.compile(r"\b(send|post|upload|forward|include)\b.{0,40}\b(contents?|credentials?|tokens?|keys?|passwords?)\b.{0,40}\b(to|in)\b", re.I)),
    ("medium", "tool-chaining coercion",
     re.compile(r"\b(before|after)\b.{0,15}\b(using|calling|invoking)\b.{0,25}\b(this|any|other)\b.{0,10}\btools?\b.{0,30}\b(must|always)\b", re.I)),
    ("medium", "hidden markup block", _HIDDEN_BLOCK),
    ("medium", "html comment", _HTML_COMMENT),
    ("low", "embedded encoded payload", re.compile(r"[A-Za-z0-9+/]{120,}={0,2}")),
]


@dataclass
class Finding:
    severity: str
    rule: str
    excerpt: str


def scan(text: str) -> list[Finding]:
    findings: list[Finding] = []
    if not text:
        return findings
    if _INVISIBLE.search(text):
        findings.append(Finding("high", "invisible unicode characters", "[zero-width or tag characters]"))
    for severity, rule, rx in PATTERNS:
        m = rx.search(text)
        if m:
            findings.append(Finding(severity, rule, m.group(0)[:80]))
    if len(text) > 4 * MAX_DESCRIPTION_CHARS:
        findings.append(Finding("low", "oversized description", f"{len(text)} characters"))
    return findings


def sanitize(text: str, max_chars: int = MAX_DESCRIPTION_CHARS) -> str:
    """Remove invisible characters and hidden blocks, collapse whitespace, truncate."""
    text = _INVISIBLE.sub("", text or "")
    text = _HIDDEN_BLOCK.sub("", text)
    text = _HTML_COMMENT.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rsplit(" ", 1)[0] + "..."
    return text
