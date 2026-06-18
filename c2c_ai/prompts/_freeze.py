"""Bootstrap golden hashes for the c2c_ai prompt library."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c2c_ai import prompts as P

m = P.freeze_goldens()
print("FROZEN templates:")
for name, entry in m["templates"].items():
    print(f"  {name}: v{entry['version']} {entry['golden_sha256']}")

print("\nVERIFY (re-render and match):")
fail = 0
for r in P.verify_goldens():
    status = "OK  " if r["ok"] else "FAIL"
    if not r["ok"]:
        fail += 1
    print(f"  [{status}] {r['name']}  want={r['want']}  got={r['got']}")

print(f"\nResult: {len(m['templates'])} templates, {fail} failures")
sys.exit(0 if fail == 0 else 1)
