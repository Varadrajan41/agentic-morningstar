"""
Knowledge base backup and export utilities.

Exports all ChromaDB collections to a portable JSON file so the
knowledge base can be recovered if the DB directory is corrupted or deleted.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from src.tools.chroma_tools import get_chroma_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_BACKUP_DIR = Path("./morningstar_backups")


def export_knowledge_base(output_path: str = "") -> Dict[str, Any]:
    """
    Export all ChromaDB collections to a JSON file.

    Args:
        output_path: File path to write to. If empty, auto-generates a
                     timestamped filename in ./morningstar_backups/.

    Returns:
        Dict with export stats: path, collections, total_docs, timestamp
    """
    chroma = get_chroma_manager()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not output_path:
        DEFAULT_BACKUP_DIR.mkdir(exist_ok=True)
        output_path = str(DEFAULT_BACKUP_DIR / f"morningstar_backup_{timestamp}.json")

    export_data = {
        "exported_at": datetime.now().isoformat(),
        "version": "1.0",
        "collections": {}
    }

    total_docs = 0
    for coll_name, collection in chroma.collections.items():
        data = collection.get(include=["documents", "metadatas"])
        docs = []
        for doc_id, document, metadata in zip(data["ids"], data["documents"], data["metadatas"]):
            docs.append({
                "id": doc_id,
                "document": document,
                "metadata": metadata
            })
        export_data["collections"][coll_name] = docs
        total_docs += len(docs)
        logger.info(f"Exported {len(docs)} docs from '{coll_name}'")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(export_data, indent=2, ensure_ascii=False))

    stats = {
        "path": str(output_file),
        "collections": list(export_data["collections"].keys()),
        "total_docs": total_docs,
        "timestamp": timestamp,
        "size_kb": round(output_file.stat().st_size / 1024, 1)
    }
    logger.info(f"✅ Backup complete → {stats['path']} ({stats['size_kb']} KB, {total_docs} docs)")
    return stats


def list_backups() -> list:
    """Return available backup files sorted newest first."""
    if not DEFAULT_BACKUP_DIR.exists():
        return []
    files = sorted(DEFAULT_BACKUP_DIR.glob("morningstar_backup_*.json"), reverse=True)
    return [
        {
            "path": str(f),
            "name": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        }
        for f in files
    ]
