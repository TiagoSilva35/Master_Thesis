Arrow = tuple[str, str]


class Graph:
    """The arrows found so far: the parents of each variable, and the
    ancestors of each variable for the loop check."""

    def __init__(self, variables: list[str]):
        self.variables = list(variables)
        self.parents: dict[str, list[str]] = {name: [] for name in variables}
        self.ancestors: dict[str, set[str]] = {name: set() for name in variables}

    def copy(self) -> "Graph":
        copy = Graph(self.variables)
        copy.parents = {name: list(parents) for name, parents in self.parents.items()}
        copy.ancestors = {name: set(ancestors) for name, ancestors in self.ancestors.items()}
        return copy

    def creates_loop(self, parent: str, child: str) -> bool:
        """Adding parent → child closes a loop exactly when child is already an
        ancestor of parent (there is a path child → ... → parent)."""
        return child == parent or child in self.ancestors[parent]

    def add(self, parent: str, child: str) -> None:
        """Add the arrow. The parent and its ancestors become ancestors of the
        child and of everything below the child."""
        self.parents[child].append(parent)
        new = {parent} | self.ancestors[parent]
        for name in [child] + self.descendants(child):
            self.ancestors[name] |= new

    def remove(self, parent: str, child: str) -> None:
        """Remove the arrow. Removing can shrink many ancestor sets, so they are
        all worked out again."""
        self.parents[child].remove(parent)
        self.ancestors = {}
        for name in self.variables:
            self._find_ancestors(name)

    def descendants(self, name: str) -> list[str]:
        return [other for other in self.variables if name in self.ancestors[other]]

    def reversal_creates_loop(self, parent: str, child: str) -> bool:
        """Reversing parent → child closes a loop when there is another path from
        parent to child, through some other parent of the child."""
        return any(parent in self.ancestors[other] for other in self.parents[child] if other != parent)

    def arrows(self) -> list[Arrow]:
        return [(parent, child) for child in self.variables for parent in self.parents[child]]

    def order(self) -> list[str]:
        """The variables with every parent before its children. An ancestor
        always has fewer ancestors than its descendants (its own ancestors
        plus itself), so sorting by that count puts parents first."""
        return sorted(self.variables, key=lambda name: (len(self.ancestors[name]), name))

    def _find_ancestors(self, name: str) -> set[str]:
        if name not in self.ancestors:
            found: set[str] = set()
            for parent in self.parents[name]:
                found |= {parent} | self._find_ancestors(parent)
            self.ancestors[name] = found
        return self.ancestors[name]
