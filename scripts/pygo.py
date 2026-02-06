#!/usr/bin/env python3
"""
Simple code navigation using jedi.

Usage:
    pygo.py goto <file> <line> <column>     # Go to definition
    pygo.py refs <file> <line> <column>     # Find references
    pygo.py sig <file> <line> <column>      # Get signature
    pygo.py completions <file> <line> <column>  # Get completions
    pygo.py find <symbol> <path>            # Find symbol in file/dir

Examples:
    pygo.py goto train_v4.py 45 10
    pygo.py find "XTransUNet" .
"""

import sys
import os

try:
    import jedi
except ImportError:
    print("Error: jedi not installed. Run: pip install jedi", file=sys.stderr)
    sys.exit(1)


def goto(filepath, line, column):
    """Go to definition."""
    with open(filepath) as f:
        source = f.read()
    
    script = jedi.Script(source, path=filepath)
    defs = script.goto(line, column)
    
    if not defs:
        print("No definition found")
        return
    
    for d in defs:
        print(f"{d.module_path}:{d.line}:{d.column}  {d.type} {d.name}")


def refs(filepath, line, column):
    """Find all references."""
    with open(filepath) as f:
        source = f.read()
    
    script = jedi.Script(source, path=filepath)
    references = script.get_references(line, column)
    
    if not references:
        print("No references found")
        return
    
    for r in references:
        print(f"{r.module_path}:{r.line}:{r.column}  {r.name}")


def sig(filepath, line, column):
    """Get function signature."""
    with open(filepath) as f:
        source = f.read()
    
    script = jedi.Script(source, path=filepath)
    sigs = script.get_signatures(line, column)
    
    if not sigs:
        print("No signature found")
        return
    
    for s in sigs:
        print(f"{s.name}({', '.join(p.name for p in s.params)})")


def completions(filepath, line, column):
    """Get completions."""
    with open(filepath) as f:
        source = f.read()
    
    script = jedi.Script(source, path=filepath)
    comps = script.complete(line, column)
    
    for c in comps[:20]:  # Limit to 20
        print(f"{c.name:30} {c.type}")


def find_symbol(symbol, path):
    """Find symbol definition in file or directory."""
    import glob
    
    if os.path.isfile(path):
        files = [path]
    else:
        files = glob.glob(os.path.join(path, "**/*.py"), recursive=True)
    
    for filepath in files:
        try:
            with open(filepath) as f:
                source = f.read()
            
            script = jedi.Script(source, path=filepath)
            names = script.get_names(all_scopes=True, definitions=True)
            
            for n in names:
                if n.name == symbol:
                    print(f"{filepath}:{n.line}:{n.column}  {n.type} {n.name}")
        except Exception as e:
            pass  # Skip files that can't be parsed


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "goto" and len(sys.argv) == 5:
        goto(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "refs" and len(sys.argv) == 5:
        refs(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "sig" and len(sys.argv) == 5:
        sig(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "completions" and len(sys.argv) == 5:
        completions(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "find" and len(sys.argv) == 4:
        find_symbol(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
