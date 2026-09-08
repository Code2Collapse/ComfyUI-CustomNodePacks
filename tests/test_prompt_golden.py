"""Prompt-library golden regression test.

Asserts that every ``.j2`` template renders byte-identically to its
frozen ``golden_sha256`` in ``c2c_ai/prompts/MANIFEST.json``.

If a developer deliberately edits a prompt, they must:
  1. Bump that template's ``version`` in MANIFEST.json.
  2. Run ``python c2c_ai/prompts/_freeze.py`` to recompute the hash.
  3. Re-run this test — it should now pass.

Any silent prompt drift (e.g. an editor stripping a trailing newline,
or a search-and-replace touching an AI system message) fails CI here
BEFORE the bad prompt reaches users' models.

Run directly:
    python tests/test_prompt_golden.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the pack root is importable when run from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c2c_ai import prompts as P


def main() -> int:
    results = P.verify_goldens()
    if not results:
        print("FAIL: no templates discovered under c2c_ai/prompts/templates/")
        return 1
    failures: list[dict] = []
    for r in results:
        tag = "OK  " if r["ok"] else "FAIL"
        print(f"  [{tag}] {r['name']}")
        if not r["ok"]:
            print(f"           want: {r['want']}")
            print(f"           got:  {r['got']}")
            failures.append(r)

    # Extra invariants
    inv_fail = 0
    for tpl in P.list_templates():
        if not tpl["golden_sha256"].startswith("sha256:"):
            print(f"  [FAIL] {tpl['name']}: missing/malformed golden_sha256")
            inv_fail += 1
        if not tpl["version"] or tpl["version"] == "0.0.0":
            print(f"  [FAIL] {tpl['name']}: version missing or 0.0.0")
            inv_fail += 1

    total = len(results)
    bad = len(failures) + inv_fail
    print(f"\n{total - bad}/{total} templates match goldens; {inv_fail} invariant failures")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
