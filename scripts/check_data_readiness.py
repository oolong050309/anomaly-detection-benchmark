"""对照计划书 §2 与 selected_files.json，检查数据清单 / EDA / 本地文件就位情况。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN_TS_SHORT = [
    "006_NAB_id_6_Traffic",
    "149_Stock_id_1_Finance",
    "171_MITDB_id_2_Medical",
    "225_MGAB_id_1_Synthetic",
    "276_IOPS_id_17_WebService",
    "331_UCR_id_29_Facility",
    "337_UCR_id_35_HumanActivity",
    "550_SWaT_id_1_Sensor",
]


def _tabular_key(fname: str) -> str:
    from adapters.adbench_adapter import ADBENCH_DATASETS

    key = {spec["file"]: k for k, spec in ADBENCH_DATASETS.items()}.get(fname)
    if key:
        return key
    if fname.startswith("13_"):
        return "credit_card"
    return fname.split("_", 1)[1].replace(".npz", "").lower()


def _ts_short(fname: str) -> str:
    for token in PLAN_TS_SHORT:
        if token in fname:
            return token
    return fname.replace(".csv", "")


def main() -> None:
    data_root = ROOT / "data"
    selected = json.loads((data_root / "selected_files.json").read_text(encoding="utf-8"))
    eda = json.loads((data_root / "eda_summary.json").read_text(encoding="utf-8"))

    def eda_row(name: str, modality: str | None = None) -> dict | None:
        for row in eda:
            if str(row.get("name")) != name:
                continue
            if modality and row.get("modality") != modality:
                continue
            return row
        return None

    def disk_npz(mod: str, fname: str) -> bool:
        p = data_root / mod / fname
        return p.is_file() and p.stat().st_size > 100

    def disk_csv(fname: str) -> bool:
        p = data_root / "timeseries" / fname
        return p.is_file() and p.stat().st_size > 100

    def disk_graph(name: str) -> bool:
        gdir = data_root / "graph"
        if not gdir.exists():
            return False
        if (gdir / name).exists():
            return True
        return any(p.is_file() and name.lower() in p.name.lower() for p in gdir.rglob("*"))

    rows: list[tuple[str, str, bool, str | None, bool]] = []

    for f in selected["tabular"]:
        key = _tabular_key(f)
        row = eda_row(key, "tabular")
        rows.append(("tabular", key, disk_npz("tabular", f), row.get("status") if row else None, row is not None))

    for f in selected["cv"]:
        key = f.replace(".npz", "").lower()
        row = eda_row(key, "cv")
        rows.append(("cv", key, disk_npz("cv", f), row.get("status") if row else None, row is not None))

    for f in selected["nlp"]:
        key = f.replace(".npz", "").lower()
        row = eda_row(key, "nlp")
        rows.append(("nlp", key, disk_npz("nlp", f), row.get("status") if row else None, row is not None))

    for f in selected["timeseries"]:
        key = _ts_short(f)
        row = eda_row(key, "timeseries")
        rows.append(("timeseries", key, disk_csv(f), row.get("status") if row else None, row is not None))

    for g in selected["graph"]:
        row = eda_row(g, "graph")
        rows.append(("graph", g, disk_graph(g), row.get("status") if row else None, row is not None))

    print(f"{'mod':12} {'name':32} {'local':6} {'eda':10} {'in_eda':6}")
    for mod, name, local, status, has_eda in rows:
        print(f"{mod:12} {name:32} {str(local):6} {str(status):10} {str(has_eda):6}")

    n_local = sum(r[2] for r in rows)
    n_eda_ok = sum(1 for r in rows if r[3] == "ok")
    n_eda_any = sum(1 for r in rows if r[4])
    print()
    print(f"计划子集（selected_files）: {len(rows)} 项")
    print(f"本地 data/ 就位: {n_local}/{len(rows)}")
    print(f"EDA status=ok（含完整统计）: {n_eda_ok}/{len(rows)}")
    print(f"EDA 有条目（含 selected）: {n_eda_any}/{len(rows)}")

    missing_local = [f"{m}:{n}" for m, n, loc, _, _ in rows if not loc]
    missing_eda_ok = [f"{m}:{n}" for m, n, _, st, has in rows if has and st != "ok"]
    no_eda = [f"{m}:{n}" for m, n, _, _, has in rows if not has]

    if missing_local:
        print("\n本地缺失:", ", ".join(missing_local))
    if missing_eda_ok:
        print("EDA 未生成完整统计（非 ok）:", ", ".join(missing_eda_ok))
    if no_eda:
        print("EDA 无条目:", ", ".join(no_eda))


if __name__ == "__main__":
    main()
