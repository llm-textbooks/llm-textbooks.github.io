#!/usr/bin/env python3
"""Public artifact-verification harness for Qdrant/OpenFGA generation skew."""
from verify_recorded_wave75 import main
if __name__ == "__main__":
    import sys
    sys.argv.extend(["--case", "generation-skew"])
    main()
