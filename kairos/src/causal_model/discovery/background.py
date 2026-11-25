from dataclasses import dataclass, field

KINDS = ("continuous", "binary", "ordinal", "nominal")


@dataclass
class Background:
    """What is known before looking at the data.

    The outcome is a yes/no variable that never has children; a root never has
    parents (a protected attribute such as gender, say). Data without an outcome
    (a physical system, say) leaves it as None.

    `kinds` says how each variable is measured: "continuous", "binary",
    "ordinal" (levels in a known order, like a credit band) or "nominal"
    (levels with no order, like an event type). A variable not listed is
    continuous. Binary, ordinal and nominal values must be codes 0, 1, 2, ...
    """

    outcome: str | None = None
    roots: set[str] = field(default_factory=set)
    kinds: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        unknown = {name: kind for name, kind in self.kinds.items() if kind not in KINDS}
        if unknown:
            raise ValueError(f"unknown kinds {unknown}; choose from {KINDS}")

    def kind(self, name: str) -> str:
        return "binary" if name == self.outcome else self.kinds.get(name, "continuous")

    def allows(self, parent: str, child: str) -> bool:
        return parent != child and parent != self.outcome and child not in self.roots
