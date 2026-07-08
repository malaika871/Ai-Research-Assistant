"""
Interactive test harness for the AI Research Assistant RAG pipeline.

Indexes whatever documents are already sitting in your uploads/ folder,
then drops you into a simple loop: you type a question, it prints the
answer and which source document(s) it pulled from. No pre-written gold
answers, no automated scoring -- you judge each answer yourself, against
documents you actually care about.

Usage:
    python -m tests.evaluate

Requires HF_TOKEN in src/.env (loaded automatically).
Type 'exit' or 'quit' to stop.
"""
import os
import sys
from pathlib import Path


def find_project_root(start: Path) -> Path:
    """
    Walk upward from this file's location until config.py is found. Works
    no matter where this script is run from or placed within the project.
    """
    current = start
    for _ in range(5):
        if (current / "config.py").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    raise FileNotFoundError(
        f"Could not find config.py by walking up from {start}. "
        f"Make sure this script is somewhere inside your AI-Research-Assistant project."
    )


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
env_path = PROJECT_ROOT / "src" / ".env"
load_dotenv(env_path, override=True)

import config

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def find_upload_files():
    upload_dir = PROJECT_ROOT / config.UPLOAD_DIR
    if not upload_dir.exists():
        return []
    return [
        str(p) for p in sorted(upload_dir.iterdir())
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    ]


def run_interactive():
    from src.rag_engine import RAGEngine

    files = find_upload_files()
    if not files:
        print(f"No .pdf/.docx/.txt files found in {PROJECT_ROOT / config.UPLOAD_DIR}")
        print("Drop 1-2 documents there first, then run this again.")
        return

    print(f"Found {len(files)} document(s) in uploads/:")
    for f in files:
        print(f"  - {Path(f).name}")

    print("\nIndexing...")
    engine = RAGEngine()
    result = engine.index_documents(files)
    print(f"Indexed {result['indexed_files']} file(s), {result['total_chunks']} chunk(s).\n")

    print("Ask questions about these documents. Type 'exit' or 'quit' to stop.")
    print("After each answer, rate it: 'y' (correct), 'n' (wrong), 'p' (partially correct).\n")

    scores = []  # 1.0 = correct, 0.5 = partial, 0.0 = wrong

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        try:
            response = engine.ask(question)
        except Exception as e:
            print(f"  ERROR: {e}\n")
            continue

        answer = response.get("answer", "")
        sources = response.get("sources", [])
        source_names = sorted({s["source"] for s in sources})

        print(f"A: {answer}")
        print(f"   (from: {', '.join(source_names) if source_names else 'no sources retrieved'})")

        rating = input("   Correct? [y/n/p, or Enter to skip scoring]: ").strip().lower()
        if rating == "y":
            scores.append(1.0)
        elif rating == "n":
            scores.append(0.0)
        elif rating == "p":
            scores.append(0.5)
        # anything else (blank, etc.) -> not scored, doesn't count toward accuracy
        print()

    if scores:
        accuracy = round(sum(scores) / len(scores) * 100, 1)
        print("=" * 50)
        print(f"Accuracy score: {accuracy}/100  ({len(scores)} question(s) rated)")
        print("=" * 50)
        print("This reflects YOUR judgment of correctness on the questions you")
        print("asked, not an automated benchmark -- it's only as representative")
        print("as the questions you chose to ask.")
    else:
        print("No questions were rated, so no accuracy score to report.")


if __name__ == "__main__":
    if not os.getenv("HF_TOKEN"):
        print("HF_TOKEN not set.")
        print(f"  Looked for it in: {env_path}")
        print(f"  That file exists: {env_path.exists()}")
        sys.exit(1)
    run_interactive()