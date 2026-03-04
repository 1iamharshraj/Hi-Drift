from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _style_from_locomo_category(category: int) -> str:
    if category == 1:
        return "concise"
    if category == 2:
        return "detailed"
    return "bullet"


def _style_from_longmem_type(question_type: str) -> str:
    qt = question_type.lower()
    if "temporal" in qt or "reasoning" in qt:
        return "detailed"
    if "compare" in qt or "list" in qt or "multi" in qt:
        return "bullet"
    return "concise"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def convert_locomo(raw_path: Path, output_path: Path, max_rows: int = 3000) -> int:
    payload = _load_json(raw_path)
    rows: list[dict[str, Any]] = []
    prev_style: str | None = None
    prev_task: str | None = None
    for entry in payload:
        qa = entry.get("qa", [])
        for item in qa:
            question = _to_text(item.get("question")).strip()
            answer = _to_text(item.get("answer")).strip()
            if not question or not answer:
                continue
            category = int(item.get("category", 1))
            style = _style_from_locomo_category(category)
            task = f"locomo_cat_{category}"
            drift = (prev_style is not None and style != prev_style) or (prev_task is not None and task != prev_task)
            rows.append(
                {
                    "user_input": question,
                    "expected_style": style,
                    "task_label": task,
                    "oracle_fact": answer,
                    "drift": bool(drift),
                }
            )
            prev_style = style
            prev_task = task
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n", encoding="utf-8")
    return len(rows)


def _convert_longmem_payload(payload: list[dict[str, Any]], rows: list[dict[str, Any]], max_rows: int) -> None:
    prev_task: str | None = None
    prev_style: str | None = None
    for item in payload:
        question = _to_text(item.get("question")).strip()
        answer = _to_text(item.get("answer")).strip()
        if not question or not answer:
            continue
        qtype = _to_text(item.get("question_type")).strip() or "generic"
        task = qtype.replace(" ", "_").replace("/", "_").lower()
        style = _style_from_longmem_type(qtype)
        drift = (prev_task is not None and task != prev_task) or (prev_style is not None and style != prev_style)
        rows.append(
            {
                "user_input": question,
                "expected_style": style,
                "task_label": task,
                "oracle_fact": answer,
                "drift": bool(drift),
            }
        )
        prev_task = task
        prev_style = style
        if len(rows) >= max_rows:
            return


def convert_longmem(raw_paths: list[Path], output_path: Path, max_rows: int = 5000) -> int:
    rows: list[dict[str, Any]] = []
    for path in raw_paths:
        if not path.exists():
            continue
        try:
            payload = _load_json(path)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            _convert_longmem_payload(payload, rows=rows, max_rows=max_rows)
        if len(rows) >= max_rows:
            break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n", encoding="utf-8")
    return len(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare official benchmark JSONL files for ICCV pipeline.")
    parser.add_argument(
        "--locomo-raw",
        default="data/benchmarks/official/locomo/locomo10.json",
        help="Path to raw LoCoMo JSON",
    )
    parser.add_argument(
        "--locomo-out",
        default="data/benchmarks/official/locomo/locomo_official.jsonl",
        help="Output path for LoCoMo converted JSONL",
    )
    parser.add_argument(
        "--longmem-raw",
        nargs="*",
        default=[
            "data/benchmarks/official/longmem/longmemeval_oracle.json",
            "data/benchmarks/official/longmem/longmemeval_s_cleaned.json",
            "data/benchmarks/official/longmem/longmemeval_m_cleaned.json",
        ],
        help="Input LongMemEval JSON files in priority order",
    )
    parser.add_argument(
        "--longmem-out",
        default="data/benchmarks/official/longmem/longmem_official.jsonl",
        help="Output path for LongMemEval converted JSONL",
    )
    parser.add_argument("--locomo-max-rows", type=int, default=3000)
    parser.add_argument("--longmem-max-rows", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    locomo_rows = convert_locomo(
        raw_path=Path(args.locomo_raw),
        output_path=Path(args.locomo_out),
        max_rows=args.locomo_max_rows,
    )
    longmem_rows = convert_longmem(
        raw_paths=[Path(p) for p in args.longmem_raw],
        output_path=Path(args.longmem_out),
        max_rows=args.longmem_max_rows,
    )
    summary = {
        "locomo_rows": locomo_rows,
        "longmem_rows": longmem_rows,
        "locomo_out": args.locomo_out,
        "longmem_out": args.longmem_out,
    }
    print(json.dumps(summary, indent=2))
    return 0 if (locomo_rows > 0 and longmem_rows > 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
