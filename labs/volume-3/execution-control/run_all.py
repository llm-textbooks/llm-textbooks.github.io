#!/usr/bin/env python3
"""execution-control fixture 일괄 실행기.

harness/check_*.py 전부를 기본 검증 모드로 실행한다. 각 harness는
in-process 결정적 시뮬레이션을 재실행해 oracle과 커밋된 event ledger의
SHA-256을 대조한다. 외부 프로세스·네트워크는 사용하지 않는다.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_FIXTURES = 10


def main() -> None:
    harnesses = sorted((ROOT / "harness").glob("check_*.py"))
    if len(harnesses) != EXPECTED_FIXTURES:
        raise SystemExit(
            f"expected {EXPECTED_FIXTURES} harnesses, found {len(harnesses)}: "
            + ", ".join(p.name for p in harnesses)
        )
    results = []
    failures = 0
    for harness in harnesses:
        proc = subprocess.run(
            [sys.executable, str(harness)], capture_output=True, text=True
        )
        if proc.returncode != 0:
            failures += 1
            sys.stderr.write(f"FAIL {harness.name}\n{proc.stdout}{proc.stderr}\n")
            results.append({"harness": harness.name, "result": "fail"})
            continue
        line = proc.stdout.strip().splitlines()[-1]
        results.append(json.loads(line))
    print(
        json.dumps(
            {
                "lab": "volume3-execution-control",
                "harnesses": len(harnesses),
                "failures": failures,
                "results": results,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
