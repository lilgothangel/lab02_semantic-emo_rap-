# python pipeline.py [extract|verify|search|graph|all]
import sys

import extract, graph_rag, search, verify

STEPS = {
    "extract": extract.main,
    "verify": verify.sample,
    "search": search.compare,
    "graph": graph_rag.main,
}

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in STEPS.items():
        if step in (name, "all"):
            print(f"\n===== {name} =====")
            fn()
