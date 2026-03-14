#!/usr/bin/env python3
"""Remove all but the top-N entries from a built dataset, based on a ranking JSON."""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Prune dataset to top-N entries from ranking JSON")
    parser.add_argument("ranking_json", help="Ranking JSON file (entries ordered by quality)")
    parser.add_argument("dataset_dir", help="Dataset directory containing .npy and _meta.json files")
    parser.add_argument("-n", "--top-n", type=int, default=2000, help="Number of top entries to keep (default: 2000)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    args = parser.parse_args()

    with open(args.ranking_json) as f:
        data = json.load(f)

    keep = set()
    for entry in data[:args.top_n]:
        if "filename" in entry:
            keep.add(entry["filename"])
        else:
            keep.add(os.path.splitext(entry["file"])[0])

    print(f"Keeping top {len(keep)} stems from ranking")

    to_delete = []
    for fname in sorted(os.listdir(args.dataset_dir)):
        if fname.endswith("_meta.json"):
            stem = fname.replace("_meta.json", "")
        elif fname.endswith(".npy"):
            stem = fname.replace(".npy", "")
        else:
            continue
        if stem not in keep:
            to_delete.append(fname)

    print(f"Files to delete: {len(to_delete)} ({len(to_delete) // 2} samples)")
    print(f"Files to keep:   {len(os.listdir(args.dataset_dir)) - len(to_delete)}")

    if args.dry_run:
        print("\nDry run — no files deleted. First 10 deletions:")
        for f in to_delete[:10]:
            print(f"  {f}")
        return

    for fname in to_delete:
        os.remove(os.path.join(args.dataset_dir, fname))

    print("Done.")


if __name__ == "__main__":
    main()
