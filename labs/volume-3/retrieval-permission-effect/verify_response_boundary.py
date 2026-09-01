#!/usr/bin/env python3
"""Public artifact-verification harness for a TCP response-loss boundary."""
from verify_recorded_wave75 import main
if __name__ == "__main__":
    import sys
    sys.argv.extend(["--case", "response-boundary"])
    main()
