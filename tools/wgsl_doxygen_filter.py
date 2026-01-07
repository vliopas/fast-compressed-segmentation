#!/usr/bin/env python3
import sys
import re

def transform_wgsl(src: str) -> str:
    out_lines = []
    skipping_body = False
    brace_depth = 0

    for line in src.splitlines():
        # Preserve documentation comments
        if line.strip().startswith("///") or line.strip().startswith("/**") or line.strip().startswith("/*") or line.strip().startswith("* "):
            out_lines.append(line)
            continue

        # Strip WGSL attributes like @compute, @group(0), @binding(1), etc.
        cleaned = re.sub(r"@\w+(\([^)]*\))?", " ", line)

        # Detect function start: fn name(params)
        m = re.match(r"\s*fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", cleaned)
        if m:
            name = m.group(1)
            params = m.group(2)
            # Simplify WGSL types for readability
            params_simple = re.sub(r"vec\d<[^>]+>", "vec", params)
            params_simple = re.sub(r"mat\d[x\d]*<[^>]+>", "mat", params_simple)
            params_simple = re.sub(r"ptr<[^>]+>", "ptr", params_simple)
            prototype = f"void {name}({params_simple});"
            out_lines.append(prototype)
            # Start skipping body until matching closing brace
            skipping_body = True
            brace_depth = 0
            continue

        if skipping_body:
            brace_depth += line.count("{")
            brace_depth -= line.count("}")
            if brace_depth <= 0:
                skipping_body = False
            # Do not emit body lines
            continue

        # Emit non-empty cleaned lines that look declarative
        if cleaned.strip():
            out_lines.append(cleaned)

    return "\n".join(out_lines) + "\n"

if __name__ == "__main__":
    src = sys.stdin.read()
    sys.stdout.write(transform_wgsl(src))
