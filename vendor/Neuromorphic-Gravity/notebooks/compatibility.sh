python - <<'PY'
import re, sys, shutil, pathlib, site

def find_pkg():
    for p in site.getsitepackages() + [site.getusersitepackages()]:
        d = pathlib.Path(p) / "neurobench"
        if d.exists():
            return d
    return None

nb = find_pkg()
if not nb:
    print("Could not find 'neurobench' in this environment.")
    sys.exit(1)

backup = nb.with_name(nb.name + "_backup_py39_fix2")
if backup.exists():
    shutil.rmtree(backup)
shutil.copytree(nb, backup)
print(f"Backup saved to: {backup}")

# ---- helpers ----
# Add "from __future__ import annotations" near the top if missing
def add_future_annotations(text: str) -> str:
    if "from __future__ import annotations" in text:
        return text
    lines = text.splitlines()
    insert_at = 0
    # keep shebang/encoding at very top
    while insert_at < len(lines) and (
        lines[insert_at].startswith("#!") or
        re.match(r'#\s*-\*-\s*coding\s*:\s*[-\w]+\s*-\*-', lines[insert_at])
    ):
        insert_at += 1
    # place after any __future__ imports if present
    for i,l in enumerate(lines[:10]):
        if l.startswith("from __future__ import"):
            insert_at = i+1
    lines.insert(insert_at, "from __future__ import annotations")
    return "\n".join(lines)

# Rewrite unions inside parameter/return annotations across multiple lines
param_ann = re.compile(r'(:\s*)(?P<ann>(?:[^#\n\)=]|\\\n|\n(?!\s*def\b|\s*class\b))*?)\s*(?=[,)\n])', re.DOTALL)
ret_ann   = re.compile(r'(->\s*)(?P<ann>(?:[^#\n:]+|\n(?!\s*def\b|\s*class\b))+?)\s*(?=[:\n])', re.DOTALL)

def to_union(s: str) -> str:
    # quick skip
    if '|' not in s or 'Union[' in s:
        return s
    # split top-level on '|'
    parts = [p.strip() for p in s.split('|')]
    # keep empty-safe
    parts = [p for p in parts if p]
    return f"Union[{', '.join(parts)}]"

def rewrite_ann_blocks(text: str) -> str:
    changed = False
    def repl(m):
        nonlocal changed
        ann = m.group('ann')
        if '|' in ann and 'Union[' not in ann:
            new_ann = to_union(ann)
            changed = True
            return m.group(1) + new_ann
        return m.group(0)
    t2 = param_ann.sub(repl, text)
    t3 = ret_ann.sub(repl, t2)
    if changed and 'Union[' in t3:
        if not re.search(r'from\s+typing\s+import\b.*\bUnion\b', t3):
            # add Union to existing typing import or create a new one
            lines = t3.splitlines()
            for i,l in enumerate(lines[:80]):
                m = re.match(r'from\s+typing\s+import\s+(.*)', l)
                if m:
                    if 'Union' not in m.group(1):
                        lines[i] = l.rstrip() + ', Union'
                    t3 = "\n".join(lines)
                    break
            else:
                # no typing import found; place near top after futures
                lines = t3.splitlines()
                insert_at = 0
                for i,l in enumerate(lines[:10]):
                    if l.startswith('from __future__ import'):
                        insert_at = i+1
                lines.insert(insert_at, 'from typing import Union')
                t3 = "\n".join(lines)
    return t3

changed_files = 0
for py in nb.rglob('*.py'):
    s = py.read_text(encoding='utf-8', errors='ignore')
    orig = s
    s = add_future_annotations(s)
    s = rewrite_ann_blocks(s)
    if s != orig:
        py.write_text(s, encoding='utf-8')
        changed_files += 1
        rel = py.relative_to(nb)
        print("Patched:", rel)

print(f"Done. Files changed: {changed_files}")
PY
