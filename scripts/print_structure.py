"""
Quick structure summary helper.

Run:
    python scripts/print_structure.py

to see the high-level folders and what's inside ml/ and lab/.
"""

from pathlib import Path


def list_dir(path: Path, indent: int = 0, max_entries: int = 10):
    prefix = " " * indent
    if not path.exists():
        print(f"{prefix}- {path.name}/ (missing)")
        return
    print(f"{prefix}- {path.name}/")
    entries = sorted(path.iterdir(), key=lambda p: p.name)
    for i, p in enumerate(entries):
        if i >= max_entries:
            print(f"{prefix}  ... ({len(entries) - max_entries} more)")
            break
        if p.is_dir():
            print(f"{prefix}  - {p.name}/")
        else:
            print(f"{prefix}  - {p.name}")


def main():
    root = Path(".").resolve()
    print(f"Project root: {root}")

    for name in ["data", "out", "ml", "lab", "runs", "scripts", "ui", "docs"]:
        path = root / name
        list_dir(path, indent=0 if name in ("data", "out") else 2)


if __name__ == "__main__":
    main()
