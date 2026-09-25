# Superseded after collection

The 68 requested reruns completed: 10 now complete, 58 still CPU-censored, zero training failures. Collection used 32 physical cores for 730.6 seconds and 19,747.1 job CPU-seconds.

The controller then stopped before replay at a check requiring the new learning curve to be at least as long as the old curve. Dilbert/eval_009 reached 63 rounds at the old 120-second limit but only 58 rounds at the new 300-second limit. Its overlapping loss prefix passed numerical agreement. Increasing a CPU-time limit does not guarantee increasing round count when concurrency and resource contention change; the extra round-count assertion was too strong. This was not a training failure or detected loss-prefix mismatch.

Before further analysis, the user requested removing Dilbert and selecting a replacement. The new active study is `../expanded-prior-replacement-v1`. No optimizer comparison from this superseded run was completed or published as a result. The successful mfeat-factors/eval_011 record is reused by the successor study. Original records and the full 68 reruns remain archived locally.
