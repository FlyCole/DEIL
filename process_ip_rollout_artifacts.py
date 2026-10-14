#!/usr/bin/env python3
"""Backward-compatible launcher for the relocated IP rollout processor."""

from data_analysis.ip.process_ip_rollout_artifacts import *  # noqa: F401,F403
from data_analysis.ip.process_ip_rollout_artifacts import main


if __name__ == "__main__":
    main()
