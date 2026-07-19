"""Render the three CAffNet proof certificates to portable JSON."""
from __future__ import annotations

import json
from pathlib import Path

from theorem_certificates import all_certificates


def main() -> None:
    result = all_certificates()
    output = Path(__file__).resolve().parents[1] / "outputs" / "theorem_certificates.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "all_valid": result["all_valid"]}, indent=2))
    if not result["all_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
