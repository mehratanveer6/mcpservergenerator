"""Programmatic (weak) labelling for operation risk.

Hand-labelling tens of thousands of API operations is not practical for a single
developer, so the training corpus is labelled by transparent rules (labelling
functions) defined in mcpforge/rules.py. A separate, manually reviewed gold set is
used for the final evaluation, so reported scores are not measured against the
rules themselves.

Classes
  read         no state change on the server (safe to auto-approve)
  write        creates or updates state; recoverable
  destructive  deletes data, revokes access, cancels or reverses a transaction,
               or is otherwise hard to undo
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcpforge.rules import label_operation  # noqa: E402,F401

LABELS = ("read", "write", "destructive")
