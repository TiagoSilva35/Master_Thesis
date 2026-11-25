import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

REPOSITORY = "https://huggingface.co/datasets/LAMDA-Tabular/CausalArena/resolve/main"

Fetch = Callable[[str, Path], None]  # (path in the repository, where to save it)


class ArenaFiles:
    """CausalArena's files (Hugging Face, CC-BY-4.0). Each one is downloaded the
    first time it is needed and kept under `root`, so a run downloads only what it
    uses and nothing is downloaded before a run."""

    def __init__(self, root: Path, fetch: Fetch | None = None):
        self.root = root
        self.fetch = fetch or self.download

    def path(self, relative: str) -> Path:
        """The local copy of a repository file, downloaded if missing."""
        local = self.root / relative
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            self.fetch(relative, local)
        return local

    @staticmethod
    def download(relative: str, local: Path) -> None:
        request = urllib.request.Request(f"{REPOSITORY}/{relative}", headers={"User-Agent": "benchmark-client"})
        partial = local.with_name(local.name + ".part")  # a cut-off download is never mistaken for a file
        with urllib.request.urlopen(request, timeout=60) as response, open(partial, "wb") as file:
            shutil.copyfileobj(response, file)
        partial.rename(local)
