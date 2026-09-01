#!/usr/bin/env python3
"""Public artifact-verification harness for lease/fence takeover and recovery."""
from verify_recorded_wave75 import main
if __name__ == "__main__":
    import sys
    sys.argv.extend(["--case", "lease-takeover"])
    main()
