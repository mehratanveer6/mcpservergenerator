"""Domain lexicons shared by the weak labelling functions and the feature extractor."""

DESTRUCTIVE_VERBS = frozenset({
    "delete", "remove", "destroy", "purge", "revoke", "cancel", "terminate", "void",
    "refund", "wipe", "erase", "reset", "deactivate", "uninstall", "abort", "kill",
    "expire", "invalidate", "ban", "unpublish", "unsubscribe", "unlink", "detach",
    "disable", "suspend", "drop", "truncate", "rollback", "reject", "decline",
})
READ_VERBS = frozenset({
    "get", "list", "search", "query", "find", "lookup", "retrieve", "fetch", "read",
    "preview", "validate", "check", "verify", "estimate", "calculate", "count",
    "describe", "show", "view", "export", "download", "render", "simulate", "compare",
    "test", "inspect", "analyze", "analyse", "evaluate", "suggest", "autocomplete",
})
