# Superseded five-minute Dilbert evaluation

See [STATUS.md](STATUS.md). The 68 objective reruns finished, but the comparison stopped at an overly strict round-count audit before optimizer replay. The user then requested excluding Dilbert and selecting another dataset. No completed optimizer comparison is claimed for this version.

The active successor is [expanded-prior-replacement-v1](../expanded-prior-replacement-v1/README.md). Its mfeat-factors update reuses the successful record from this run; Dilbert is excluded. Original run records and logs remain locally in `runs/expanded-prior-300s-v1`.

The [frozen protocol](PROTOCOL.md), pipeline, input checksums and launch history preserve the attempted experiment. The reproduction helper can independently train selected configurations with a five-minute guard, but does not repair or claim results for this superseded comparison.
