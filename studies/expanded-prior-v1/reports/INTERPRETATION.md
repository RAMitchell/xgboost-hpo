# Interpretation

The new historical prior is useful on many target families, but **the full comparison does not establish a robust improvement over the GP without history**.

The primary early-budget mean normalized regret (evaluations1–16) is0.08453 for the historical GP versus0.08035 without history: a5.2% higher point estimate, with a family-bootstrap95%relative interval of[-23.6%,+47.7%]. The prior wins on22of30families, but its large regression on Dilbert reverses the aggregate mean. It does not statistically separate from the external baselines under the prespecified Holm-adjusted primary tests. The prior-guided two starts do improve over giving that same prior two uniform starts (11.9%lower earlyarea, Holm p=.033).

## Time limits matter

Dilbert has67of96configurations CPU-censored; mfeat-factors has1. There are no failed fits. The primary analysis treats a censored fit as an unsuccessful query and supplies a finite penalty. This is a conservative evaluation of completing the specified training procedure; it is not an analysis that accepts the capped model's best observed prefix as a successful bounded-time model. All prefixes and curves remain archived. Do not generalize the primary conclusion to a different censoring policy without a separately labeled analysis.

The historical GP's Dilbert earlyarea is0.6871 versus0.2217 without history. It selects capped configurations more often; see censoring_diagnostic.json. The prior does not model runtime or the chance of completing under the CPU cap. This is a concrete robustness problem under the frozen evaluation policy, not evidence that its predicted ranking is universally worse.

The predeclared sensitivity restricted to28fully-completed pools gives0.06593 for the historical GP versus0.07799 without history: **15.5%lower earlyregret**, bootstrap95%improvement interval[4.5%,26.1%]. Relative improvements are19.9%versus the online-kernel GP and23.1%versus SMAC/TPE. These are secondary, conditional comparisons; excluding difficult datasets cannot establish full-panel superiority. Median-scaled regret on all30also favors the no-history GP over the historical GP, so the primary issue is not resolved merely by changing normalization.

## Later budget and objectives

At32evaluations, mean normalized regrets are0.01667(historicalGP),0.01699(no-historyGP),0.01511(online-kernelGP),0.01128(SMAC),0.01667(TPE),0.02209(random). These are point estimates; this study's formal multiplicity-adjusted tests target the earlyarea, not an endpoint superiority claim. SMAC has the lowest endpoint point estimate.

The five regression families show a favorable prior earlyarea point estimate,0.06983vs0.08303without history. The25classification families give0.08747vs0.07982, including the difficult capped workloads. Five regression families are too few for a strong separate conclusion. All losses are validation objectives on families excluded from prior fitting; test losses were never evaluated. These families were inspected in earlier different experiments, so this is not wholly untouched research-level confirmation.

## Cost and next decision

2,880newobjective fits took23.46minutes on up to16physical cores,5.16jobCPU-hours. There were2,812completed fits,68censored fits and289successful round-ceiling hits. The two historical XGBoost surrogates and three kernel starts took0.28CPU seconds. All2,100optimizer trajectories /67,200queries were then replayed in19seconds with no additional objective training.

Mean optimizerCPU time per32-step run was approximately0.011seconds for the historical GP,0.012for the no-history GP,0.627for the online-kernelGP,0.172forSMAC and0.098forTPE. This excludes objective training. Historical priors were pretrained; their fitting cost is reported separately.

Retain the prior as a promising optional component, not a universally better default. Before deployment, test explicit timeout/failure handling or a bounded-time objective that accepts capped prefixes. Keep the current30-family result fixed and clearly label any follow-up on it as development. Also collect broader configuration coverage:50families do not turn40shared configurations into1,999distinct parameter locations.
