"""Acquire and inventory the official Planck inputs with bounded download workers."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from cmbcst import data


def download(product: data.Product) -> dict:
    for attempt in range(4):
        try:
            path = data.fetch(product)
            record = json.loads(path.with_name(path.name + ".provenance.json").read_text())
            if record["bytes"] != path.stat().st_size or record["key"] != product.key:
                raise ValueError("download provenance does not describe the completed file")
            return record
        except (OSError, ValueError, data.requests.RequestException) as error:
            print(f"[{product.key}] attempt {attempt + 1}: {type(error).__name__}: {error}", flush=True)
            if attempt == 3:
                raise
            time.sleep(min(3 * 2 ** attempt, 12))
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--inventory", type=Path, default=REPO / "data/evidence/planck_dataset.json")
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    records, failures = {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, product): product for product in data.MANIFEST}
        for future in as_completed(futures):
            product = futures[future]
            try:
                records[product.key] = future.result()
                print(f"[{len(records)}/{len(data.MANIFEST)} verified] {product.key}", flush=True)
            except Exception as error:
                failures[product.key] = f"{type(error).__name__}: {error}"
    inventory = {
        "schema": "cmbcst.planck-dataset.v1", "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if not failures and len(records) == len(data.MANIFEST) else "incomplete",
        "raw_directory": str(data.raw_directory()), "workers": args.workers,
        "products": [records[p.key] for p in data.MANIFEST if p.key in records], "failures": failures,
        "total_bytes": sum(r["bytes"] for r in records.values()),
        "downloader_source_sha256": hashlib.sha256(Path(data.__file__).read_bytes()).hexdigest(),
        "hash_scope": "Local SHA256 provenance; these are not publisher-supplied cryptographic digests.",
    }
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    args.inventory.write_text(json.dumps(inventory, indent=2) + "\n")
    print(f"Dataset {inventory['status']}: {inventory['total_bytes'] / 1e9:.3f} GB; {args.inventory}", flush=True)
    return 0 if inventory["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
