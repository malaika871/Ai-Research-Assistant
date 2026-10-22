from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "hf_deploy"

# Edit this whitelist to control what enters the Hugging Face Space bundle.
DEPLOY_ITEMS = [
    "Dockerfile",
    "requirements.txt",
    "config.py",
    "backend/__init__.py",
    "backend/main.py",
    "src/chunker.py",
    "src/doc_loader.py",
    "src/document_processor.py",
    "src/embedder.py",
    "src/generator.py",
    "src/loader_factory.py",
    "src/models.py",
    "src/pdf_loader.py",
    "src/rag_engine.py",
    "src/retriever.py",
    "src/services.py",
    "src/txt_loader.py",
    "src/vector_store.py",
    "src/web_search.py",
]

OPTIONAL_ENV_EXAMPLE = "src/.env.example"
SKIP_DIRS = {"__pycache__", "chroma_db"}
SKIP_PATTERNS = ("*.pyc", "*.sqlite3", ".env*")
PLACEHOLDER_MARKERS = ("your_", "placeholder", "replace_me", "change_me", "<your")
SECRET_NAME_MARKERS = ("TOKEN", "KEY", "SECRET", "PASSWORD")
MAX_WARN_BYTES = 10 * 1024 * 1024


def should_skip(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in SKIP_PATTERNS)


def is_placeholder(value: str) -> bool:
    normalized = value.strip().strip("\"'").lower()
    return not normalized or any(marker in normalized for marker in PLACEHOLDER_MARKERS)


def safe_env_example(source: Path) -> bool:
    """Allow the example only when credential-like entries are blank/placeholders."""
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return False

    found_placeholder = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        name = name.strip().upper()
        value = value.split("#", 1)[0].strip()
        if any(marker in name for marker in SECRET_NAME_MARKERS):
            if not is_placeholder(value):
                return False
            if value and is_placeholder(value):
                found_placeholder = True

    return found_placeholder


def copy_item(relative_path: str) -> list[Path]:
    source = ROOT / relative_path
    if not source.exists():
        print(f"Skipping missing whitelist item: {relative_path}")
        return []

    copied: list[Path] = []
    if source.is_file():
        if should_skip(Path(relative_path)):
            return []
        destination = OUTPUT_DIR / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return [destination]

    for child in source.rglob("*"):
        if not child.is_file():
            continue
        relative_child = child.relative_to(ROOT)
        if should_skip(relative_child):
            continue
        destination = OUTPUT_DIR / relative_child
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(child, destination)
        copied.append(destination)
    return copied


def write_space_readme() -> Path:
    readme = OUTPUT_DIR / "README.md"
    readme.write_text(
        "---\n"
        "title: NexusAI\n"
        "emoji: 🤖\n"
        "colorFrom: blue\n"
        "colorTo: purple\n"
        "sdk: docker\n"
        "app_port: 7860\n"
        "pinned: false\n"
        "---\n"
        "# AI Research Assistant\n\n"
        "Upload research documents and ask questions answered using retrieved "
        "document passages, with optional web-search fallback.\n",
        encoding="utf-8",
    )
    return readme


def print_bundle(files: list[Path]) -> None:
    relative_files = sorted(path.relative_to(OUTPUT_DIR) for path in files)
    total_size = sum((OUTPUT_DIR / path).stat().st_size for path in relative_files)

    tree: dict[str, dict] = {}
    for relative in relative_files:
        current = tree
        for part in relative.parts[:-1]:
            current = current.setdefault(part, {})
        current[relative.name] = None

    def print_tree(branch: dict[str, dict | None], prefix: str = "") -> None:
        entries = sorted(branch.items())
        for index, (name, child) in enumerate(entries):
            is_last = index == len(entries) - 1
            print(f"{prefix}{'└── ' if is_last else '├── '}{name}{'/' if child is not None else ''}")
            if child is not None:
                print_tree(child, prefix + ("    " if is_last else "│   "))

    print("hf_deploy/")
    print_tree(tree)
    print(f"Total size: {total_size:,} bytes ({total_size / (1024 * 1024):.2f} MiB)")

    oversized = [
        (path, (OUTPUT_DIR / path).stat().st_size)
        for path in relative_files
        if (OUTPUT_DIR / path).stat().st_size > MAX_WARN_BYTES
    ]
    for path, size in oversized:
        print(f"WARNING: {path} is over 10 MB ({size:,} bytes)")


def main() -> None:
    if OUTPUT_DIR.is_symlink():
        raise RuntimeError("Refusing to remove hf_deploy because it is a symlink.")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    copied: list[Path] = []
    for item in DEPLOY_ITEMS:
        copied.extend(copy_item(item))

    env_example_path = ROOT / OPTIONAL_ENV_EXAMPLE
    if env_example_path.is_file() and safe_env_example(env_example_path):
        destination = OUTPUT_DIR / OPTIONAL_ENV_EXAMPLE
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(env_example_path, destination)
        copied.append(destination)
    else:
        print("Skipping src/.env.example (missing or contains non-placeholder credentials).")

    copied.append(write_space_readme())
    print_bundle(copied)


if __name__ == "__main__":
    main()
