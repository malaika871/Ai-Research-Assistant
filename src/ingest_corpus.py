import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import chromadb

import config
from src.embedder import EmbeddingError


MANIFEST_VERSION = 1


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description="Index documents listed in SOURCES_CSV.")
    parser.add_argument("--rebuild", action="store_true", help="Recreate only the configured Chroma collection.")
    parser.add_argument("--only", metavar="DOC_ID", help="Process only this document ID.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without writing anything.")
    parser.add_argument(
        "--allow-default-collection",
        action="store_true",
        help="Allow use of the protected research_documents_v2 collection.",
    )
    return parser.parse_args(argv)


def _read_sources(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as source_file:
        rows = csv.DictReader(source_file)
        if not rows.fieldnames:
            raise ValueError(f"Sources CSV has no header: {csv_path}")
        normalized_rows = []
        for row in rows:
            normalized_rows.append(
                {
                    (key or "").strip(): (value or "").strip()
                    for key, value in row.items()
                    if key is not None
                }
            )
        return normalized_rows


def _resolve_source(row):
    local_path = row.get("local_path", "").replace("\\", "/")
    basename = local_path.rsplit("/", 1)[-1]
    if not basename:
        return None
    return os.path.join(config.CORPUS_DIR, basename)


def _sha256(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as document:
        for block in iter(lambda: document.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(manifest_path):
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid manifest format in {manifest_path}")
    return manifest


def _write_manifest(manifest_path, manifest):
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    os.replace(temporary_path, path)


def _build_plan(rows, manifest, selected_doc_id, rebuild):
    plan = []
    seen_doc_ids = set()
    for row in rows:
        doc_id = row.get("doc_id", "")
        if selected_doc_id and doc_id != selected_doc_id:
            continue
        if not doc_id:
            plan.append({"row": row, "doc_id": "<missing doc_id>", "path": None, "action": "invalid"})
            continue
        if doc_id in seen_doc_ids:
            plan.append({"row": row, "doc_id": doc_id, "path": None, "action": "duplicate"})
            continue
        seen_doc_ids.add(doc_id)

        file_path = _resolve_source(row)
        if not file_path or not os.path.isfile(file_path):
            plan.append({"row": row, "doc_id": doc_id, "path": file_path, "action": "missing"})
            continue

        digest = _sha256(file_path)
        previous = manifest.get(doc_id, {})
        unchanged = (
            previous.get("sha256") == digest
            and previous.get("embedding_model") == config.EMBEDDING_MODEL
        )
        action = "rebuild" if rebuild else "skip" if unchanged else "index"
        plan.append(
            {
                "row": row,
                "doc_id": doc_id,
                "path": file_path,
                "sha256": digest,
                "action": action,
            }
        )
    return plan


def _print_plan(plan):
    print("Plan:")
    for item in plan:
        if item["action"] == "missing":
            print(f"  WARN {item['doc_id']}: missing file {item['path']}; skip")
        else:
            print(f"  {item['action'].upper():7} {item['doc_id']}: {item['path'] or 'no usable path'}")


def _confirm_rebuild(collection_name):
    print(f"Rebuild will delete and recreate ONLY Chroma collection: {collection_name}")
    try:
        confirmation = input(f"Type {collection_name} to continue: ")
    except EOFError:
        return False
    return confirmation == collection_name


def _recreate_collection(collection_name):
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    existing = client.list_collections()
    existing_names = {item if isinstance(item, str) else item.name for item in existing}
    if collection_name in existing_names:
        client.delete_collection(name=collection_name)


def _existing_counts(vector_store, doc_id, fallback_pages):
    data = vector_store.collection.get(
        where={"doc_id": doc_id},
        include=["metadatas"],
    )
    metadatas = data.get("metadatas") or []
    pages = {metadata.get("page") for metadata in metadatas if metadata.get("page") is not None}
    return len(pages) or fallback_pages, len(metadatas)


def _print_results(results, failures, total_chunks):
    headings = ("doc_id", "pages", "chunks", "seconds")
    rows = [
        (result["doc_id"], str(result["pages"]), str(result["chunks"]), f"{result['seconds']:.2f}")
        for result in results
    ]
    widths = [max(len(headings[index]), *(len(row[index]) for row in rows)) for index in range(len(headings))]
    print("\n" + " | ".join(headings[index].ljust(widths[index]) for index in range(len(headings))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[index].ljust(widths[index]) for index in range(len(headings))))
    print(f"Total chunks: {total_chunks}")
    if failures:
        print("Failures:")
        for doc_id, message in failures:
            print(f"  {doc_id}: {message}")


def main(argv=None):
    args = _arguments(argv)
    if config.COLLECTION_NAME == "research_documents_v2" and not args.allow_default_collection:
        print(
            "Refusing to run against protected collection 'research_documents_v2'. "
            "Pass --allow-default-collection to continue.",
            file=sys.stderr,
        )
        return 2

    try:
        rows = _read_sources(config.SOURCES_CSV)
        manifest_path = os.path.join(config.CORPUS_DIR, "index_manifest.json")
        manifest = _load_manifest(manifest_path)
    except (OSError, csv.Error, json.JSONDecodeError, ValueError) as exc:
        print(f"Cannot load corpus inputs: {exc}", file=sys.stderr)
        return 2

    if args.only and not any(row.get("doc_id") == args.only for row in rows):
        print(f"DOC_ID not found in {config.SOURCES_CSV}: {args.only}", file=sys.stderr)
        return 2

    plan = _build_plan(rows, manifest, args.only, args.rebuild)
    if not plan:
        print("No matching CSV rows to process.")
        return 0
    _print_plan(plan)

    if args.dry_run:
        print("Dry run: no files, manifest, or collections were changed.")
        return 0

    if args.rebuild:
        if not _confirm_rebuild(config.COLLECTION_NAME):
            print("Rebuild cancelled; no collection was deleted.")
            return 1
        try:
            _recreate_collection(config.COLLECTION_NAME)
            manifest = {}
            _write_manifest(manifest_path, manifest)
        except Exception as exc:
            print(f"Could not recreate collection '{config.COLLECTION_NAME}': {exc}", file=sys.stderr)
            return 2

    try:
        from src.document_processor import DocumentProcessor
        from src.vector_store import VectorStore

        vector_store = VectorStore()
        processor = DocumentProcessor()
    except Exception as exc:
        print(f"Could not initialize corpus indexing: {exc}", file=sys.stderr)
        return 2

    results = []
    failures = []
    total_chunks = 0
    stop_for_credits = False

    for item in plan:
        doc_id = item["doc_id"]
        row = item["row"]
        action = item["action"]
        fallback_pages = int(row.get("pages") or 0)
        started = time.monotonic()

        if action == "missing":
            print(f"WARN {doc_id}: missing file {item['path']}; skipped")
            results.append({"doc_id": doc_id, "pages": 0, "chunks": 0, "seconds": 0.0})
            continue
        if action in {"invalid", "duplicate"}:
            message = "missing doc_id" if action == "invalid" else "duplicate doc_id in CSV"
            failures.append((doc_id, message))
            results.append({"doc_id": doc_id, "pages": fallback_pages, "chunks": 0, "seconds": 0.0})
            continue
        if action == "skip":
            try:
                pages, chunks = _existing_counts(vector_store, doc_id, fallback_pages)
            except Exception:
                pages, chunks = fallback_pages, 0
            results.append({"doc_id": doc_id, "pages": pages, "chunks": chunks, "seconds": 0.0})
            total_chunks += chunks
            continue

        try:
            vector_store.delete_by_doc_id(doc_id)
            manifest.pop(doc_id, None)
            _write_manifest(manifest_path, manifest)
            stats = processor.index(
                [item["path"]],
                display_names={item["path"]: os.path.basename(item["path"])},
                metadata=row,
            )
            pages = int(stats.get("total_pages", fallback_pages))
            chunks = int(stats.get("total_chunks", 0))
            elapsed = time.monotonic() - started
            results.append({"doc_id": doc_id, "pages": pages, "chunks": chunks, "seconds": elapsed})
            total_chunks += chunks
            if chunks == 0:
                print(f"WARN {doc_id}: indexed zero chunks; this may be a scanned PDF that needs OCR.")
            manifest[doc_id] = {
                "sha256": item["sha256"],
                "embedding_model": config.EMBEDDING_MODEL,
            }
            _write_manifest(manifest_path, manifest)
        except Exception as exc:
            message = str(exc)
            elapsed = time.monotonic() - started
            results.append({"doc_id": doc_id, "pages": fallback_pages, "chunks": 0, "seconds": elapsed})
            failures.append((doc_id, message))
            if isinstance(exc, EmbeddingError) and "credits are exhausted" in message.lower():
                print(f"Stopping immediately: {message}", file=sys.stderr)
                stop_for_credits = True
                break

    _print_results(results, failures, total_chunks)
    return 1 if failures or stop_for_credits else 0


if __name__ == "__main__":
    raise SystemExit(main())