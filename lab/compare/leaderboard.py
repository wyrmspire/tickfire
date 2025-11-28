"""
Sweep-level comparison utilities.

Responsibilities:
- scan runs/sweeps/<sweep_id>/*/eval/metrics.json
- compute scores per run
- write a leaderboard.json at the sweep root
- optionally mark "keep"/"prune" flags for each run
"""
