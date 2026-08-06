"""Dependency-free helpers for bounded per-category disk caches."""

from __future__ import annotations

from pathlib import Path


def category_cache_snapshot(cache_root: str, category: str) -> set[Path]:
    root = Path(cache_root).resolve()
    category_root = (root / category).resolve()
    try:
        category_root.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"cache category escapes cache root: {category_root}") from exc
    if not category_root.exists():
        return set()
    return {path.resolve() for path in category_root.glob("*.pth") if path.is_file()}


def delete_new_category_cache_files(
    cache_root: str,
    category: str,
    before: set[Path],
) -> int:
    """Delete only .pth files created during one completed category."""

    root = Path(cache_root).resolve()
    after = category_cache_snapshot(cache_root, category)
    created = sorted(after.difference(before))
    for path in created:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"refusing to delete cache outside {root}: {path}") from exc
        path.unlink()
    category_root = (root / category).resolve()
    if category_root.exists() and not any(category_root.iterdir()):
        category_root.rmdir()
    return len(created)
