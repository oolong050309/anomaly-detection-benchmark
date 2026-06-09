"""拉取 ADBench 演示 .npz 到 `data/{modality}/`，供 Streamlit PCA 与本地 smoke test 使用。

示例：
    python -m scripts.fetch_demo_data
    python -m scripts.fetch_demo_data --datasets cardio credit_card
    python -m scripts.fetch_demo_data --all-selected
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import fetch_adbench_demo_file, get_project_root  # noqa: E402
from adapters.adbench_adapter import ADBENCH_DATASETS, normalize_adbench_name  # noqa: E402

_FILE_TO_NAME = {spec["file"]: key for key, spec in ADBENCH_DATASETS.items()}


def _names_from_selected(data_root: Path) -> list[str]:
    path = data_root / "selected_files.json"
    if not path.exists():
        return ["cardio"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for modality in ("tabular", "cv", "nlp"):
        for fname in payload.get(modality, []):
            key = _FILE_TO_NAME.get(str(fname))
            if key:
                names.append(key)
    return sorted(set(names))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ADBench demo npz files into data/")
    parser.add_argument("--data-root", type=Path, default=get_project_root() / "data")
    parser.add_argument("--datasets", nargs="*", default=["cardio"], help="Dataset names, e.g. cardio credit_card")
    parser.add_argument("--all-selected", action="store_true", help="Download all npz listed in selected_files.json")
    args = parser.parse_args()

    names = _names_from_selected(args.data_root) if args.all_selected else args.datasets
    ok_n = 0
    for name in names:
        key = normalize_adbench_name(name)
        ok, msg = fetch_adbench_demo_file(key, args.data_root)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: {msg}")
        if ok:
            ok_n += 1
    print(f"Done: {ok_n}/{len(names)} ready under {args.data_root}")


if __name__ == "__main__":
    main()
