# Pre-registration

## Task 4 (anti-windup) — REVISED, recorded 2026-09-16 before any anti-windup run

The earlier prediction ("within 1pp and within 0.15 exponent") was designed such
that success equalled no measurable change, and is withdrawn. The load-bearing
prediction is behavioural:

> Anti-windup **flattens the post-knee accuracy decay** — accuracy at 4x the knee
> lands within 1pp of peak, against the 3-4pp drop measured in unmodified PC-ALM —
> while **bounding the dual norm** and leaving **peak accuracy within 1pp**.

The flattening is the claim. The bounded dual norm is the proposed mechanism.
Each clause is reported separately; any clause that fails is reported as a failure.

Superseded prediction (kept for the record): "anti-windup flattens the post-knee
decay, bounds the dual norm, and leaves peak accuracy within 1pp and the depth
exponent within 0.15 of unmodified PC-ALM."

## Task 2 (step-size control) — recorded before the run completed

Pre-specified decision rule, no preferred outcome:
- decay persists at all four step-size scales -> dual windup
- decay weakens/vanishes as the step shrinks -> step-size interaction

Either result is a mechanism finding and Sections 6-7 are written to whichever
holds. Neither is cut. As of this writing the 4x-knee points at reduced scales
are not yet reached, so the comparison is not yet available.

## Decision gate (go/no-go for ICLR vs TMLR)

Submit only if the depth exponent is stable across dataset, width, and epoch
count. If the exponent moves with width or with epochs, do not submit to ICLR;
the study goes to TMLR. Reported as a single mechanical table.

---

## Task 3 outcome (recorded 2026-09-16, after running): WINDUP HYPOTHESIS FALSIFIED

The windup mechanism proposed after Task 2 is dead. Integral windup requires
unbounded dual growth driven by a persistent residual. Measured at L=16/64/128,
3 seeds, T to 4096, trained params:

- dual norm is BOUNDED: peaks at T ~ 6L then falls back and plateaus
  (L=16: 5.82@103 -> 3.99; L=64: 11.55@384 -> 8.99; L=128: 17.98@829 -> 12.60)
- residual FALLS 3-6x from T=2L to T=4096, it does not persist
- linear / power-law / exponential fits all have R^2 = 0.03-0.12; the power-law
  exponent is ~0. No growth model describes the curve because it is not growth.

Neither necessary condition for windup holds. Growth class: NEITHER windup
(linear/polynomial) NOR instability (exponential). The iteration enters a
BOUNDED NON-SETTLING regime.

Task 2's FINDING is untouched (decay is real, worsens 27x as eta_h shrinks,
argmax marches inward 3.33 -> 1.00 x knee). Only its explanation is withdrawn.

## Task 4 REPLACED: oscillation, not windup
### Pre-registered 2026-09-16, BEFORE any damping run

Motivation: the only measured quantity that tracks the failure monotonically is
the tail fluctuation amplitude of the dual norm, which grows with depth
(5.2% -> 6.7% -> 9.9% at L=16/64/128).

> **Prediction.** Damping the dual step (reducing alpha, or averaging duals over
> the last k iterations) reduces tail fluctuation amplitude, AND the reduction in
> fluctuation is accompanied by a reduction in post-knee accuracy decay. Peak
> accuracy stays within 1pp and the depth exponent within 0.15.

**The load-bearing clause is the COUPLING**: fluctuation down and decay down
*together*. If damping reduces fluctuation while the accuracy decay persists
unchanged, oscillation is NOT the cause either, and that is reported as a
falsification -- not softened into a partial success. Each clause is reported
separately.

Merged with the alpha sweep: alpha is the dual step size, hence the natural
damping knob. One experiment yields fluctuation-vs-alpha, decay-vs-alpha, and
depth-exponent-vs-alpha.

Superseded predictions, kept for the record:
  (i)  "anti-windup flattens the post-knee decay ... while bounding the dual
       norm" -- withdrawn: the duals are already bounded, so the mechanism
       clause was false before the experiment began and a positive flattening
       result would have been uninterpretable.
  (ii) the earlier null-shaped prediction, already withdrawn above.

---

## Item 1: misalignment localization
### Pre-registered 2026-09-17, BEFORE running

Known: duals bounded, residual falls 3-6x, yet cosine to the true BP gradient
peaks ~0.96 at T~3.5-4.4L then decays to 0.91-0.95. The direction wanders. Where?

> **Prediction.** Misalignment concentrates in EARLY layers -- the dual
> correction over-rotates the credit signal where it had to travel furthest --
> and the affected fraction of the network grows with depth.

Decomposition used (exact, multiplicative): with per-layer weight gradients
g_l and BP gradients b_l,
    cos(g, b) = MAG * DIR,
    MAG = sum_l ||g_l|| ||b_l|| / (||g|| ||b||)        <= 1, =1 iff norm profiles proportional
    DIR = sum_l w_l cos(g_l, b_l),  w_l = ||g_l||||b_l|| / sum_k ||g_k||||b_k||
so a MAGNITUDE effect (norm profile drift) and a DIRECTION effect (per-layer
angle) are reported separately rather than inferred from the total.

Outcomes, both of which go in the study:
- localizes (early layers, fraction growing with depth) -> mechanism found,
  Section 7 becomes a mechanism section
- uniform across layers -> FIFTH falsified candidate, reported as such

Prior falsified candidates: ballistic=L^1 (paper's), integral windup (user's,
fed by my step-size argument), oscillation (user's), dual-trajectory phase
(user's).

### Item 1: how the decomposition will be READ
### Recorded 2026-09-17. CORRECTION: the header first written said "with 5/9
### cells complete". That count was taken one turn earlier; by the time the text
### was actually written all 9 cells had finished. The substance of the
### pre-commitment is unaffected -- no decomposition output had been inspected,
### loaded, or printed at the time of writing, and this correction was made
### before the first inspection. Recording the discrepancy rather than leaving
### the inaccurate count, because a pre-registration's only value is its
### accuracy.

cos(g,b) = MAG * DIR is exact. Committing now to which outcome counts as a
mechanism, so the reading cannot be retrofitted. Outcomes in order of how much
they would license:

1. **DIR degrades, MAG holds** -> per-layer ROTATION. The credit signal is
   mis-pointed layer by layer; implicates the dual correction directly.
   COUNTS AS A MECHANISM.
2. **MAG degrades, DIR holds** -> per-layer WEIGHTING. Directions stay correct
   but effort is distributed wrongly across layers. COUNTS AS A MECHANISM, and
   is the more actionable of the two: it says the failure is about how credit is
   apportioned across depth, not about where it points.
3. **Both degrade** -> weaker. Reported as "misalignment is not attributable to
   either factor alone"; NOT claimed as a mechanism.
4. **Neither degrades materially / loss is uniform across layers** -> FIFTH
   falsified candidate, reported as such.

Localization is judged separately from (1)-(4): affected layer range reported
as a fraction of depth AND in absolute layers, at L=16/64/128.

Aggregation guard: DIR's weights w_l = ||g_l||||b_l|| / sum_k(...) can drift
with T on their own, which would mix a weighting change into what looks like an
angle change. UNWEIGHTED per-layer cos(g_l,b_l) is therefore reported alongside
DIR, so layer-level rotation is visible independent of aggregation. Both are
already saved by analysis/run_item1.py (per_layer_cos, grad_norms, bp_norms).

---

## Item 2: causal intervention on the localized rotation
### Pre-registered 2026-09-17, BEFORE implementing or running

Item 1 established (a) the cosine loss is carried entirely by DIR, per-layer
rotation, with MAG improving; and (b) rotation concentrates in the input-side
portion, affected fraction growing 56% -> 64% -> 67% at L=16/64/128. That is a
localization, not a cause.

Intervention: freeze the dual variables (lambda held at 0, no dual ascent) in a
contiguous half of the layers, leaving the other half updating normally.

  INPUT-HALF FROZEN : layers [0, L/2)      -- inside the affected region
  OUTPUT-HALF FROZEN: layers [L/2, L-1)    -- outside it; SAME NUMBER of layers

The output-half arm is the control and is what makes this causal rather than
suggestive. Without it, "freezing any duals helps" is an unexcluded explanation.

Measured at L = 64 and 128, 3 seeds, budgets at 0.5/1/2/4x each condition's own
knee (96 and 256), reporting peak accuracy, argmax T, accuracy at 4x knee, and
drop = peak - acc@4x.

> **Prediction.** Freezing the INPUT half reduces the post-knee decay (drop at
> least halved relative to unmodified PC-ALM), while freezing the OUTPUT half
> leaves it essentially unchanged.

Outcomes, all of which are reported:
- input-half flattens AND output-half does not -> rotation in input-side layers
  is CAUSAL and localized. Mechanism established.
- both flatten similarly -> freezing duals helps generically; NOT localized.
  Reported as a weaker, non-localized finding.
- neither flattens -> localized rotation is NOT the cause. FIFTH falsified
  candidate, reported as a falsification.

Guard: peak accuracy is reported for every arm. If an intervention destroys
peak accuracy, its effect on the decay is uninterpretable and will be said so
rather than counted as a success.

---

## OUT-OF-SAMPLE PREDICTION: conv T_target exponent, from spectra alone
### Recorded 2026-09-17. Conv TRAINING IS PAUSED (0 processes); only 4 of ~45
### (depth,seed,method) cells are complete, so no conv T_target exponent is
### computable at the time of writing. Verified before recording.

Claim under test: conditioning CAUSES the iteration count, i.e. kappa determines
T. So far this rests on two exponents agreeing on one architecture (dense:
kappa ~ L^2.024, T_PC ~ L^2.000). That is consistency, not prediction.

kappa is measurable WITHOUT training. Measured for the conv architecture
(analysis/kappa_conv.py, on the validated generic_pc energy), 3 seeds:

    L=4: 53.7    L=8: 304.7    L=16: 1327.6    L=32: 5680.6
    conv kappa ~ L^2.108, CI [2.009, 2.207], R2=1.0000  (depths 8-32,
    matched to the depths the conv training runs use)

Also: lambda_max at L=32 is 4.394 (conv) vs 4.364 (dense) -- the Laplacian
ceiling is architecture-independent, as the first-difference argument predicts.

Conversion used, fixed in advance:
  - PC is unaccelerated, so T ~ kappa. Dense ratio T_PC/kappa exponent = 0.988.
  - PC-ALM is partially accelerated. Dense ratio T_PCALM/kappa exponent = 0.627.
    This ratio is ASSUMED to transfer across architecture; that assumption is
    itself what the PC-ALM arm tests.

> **PREDICTION (conv, depths 8-32, T_target at 90% of depth-matched BP):**
>   T_PC    exponent = 2.08
>   T_PCALM exponent = 1.32
> Counted as confirmed if the measured exponent's 95% CI contains the predicted
> value. PC is the primary test (its conversion rests on theory); PC-ALM is
> secondary (its conversion rests on an empirical ratio transferring).

Failure modes, all reported as failures:
  - PC misses -> the conditioning->iterations chain does not transfer across
    architecture, and the causal reading of kappa is withdrawn.
  - PC hits, PC-ALM misses -> the chain holds for the unaccelerated method, but
    PC-ALM's acceleration is architecture-dependent.

### Item 2, first pass: INCONCLUSIVE (design flaw, recorded 2026-09-18)

All four intervention arms returned drop = 0.00 +- 0.00 with argmax = 4.00, i.e.
accuracy was still rising at the largest budget measured. Freezing duals slows
the method and moves its knee outward, but all arms were measured at multiples
of the UNFROZEN knee, so the frozen arms were evaluated entirely pre-peak.

This is the same confound already anticipated for alpha ("alpha is
simultaneously the damping knob and a determinant of where the knee sits") and
not applied here. Neither pre-registered clause can be evaluated from this pass.

Remedy: extend to 8x and 16x knee so argmax is interior. The prediction is
UNCHANGED and is not being revised in light of the failed pass; only the
measurement window is corrected. Peak accuracy and argmax remain reported for
every arm, and any arm whose argmax sits at the window edge will again be
declared uninterpretable rather than read.

---

## 2026-09-19 — T_target ladder refinement (registered BEFORE the data exists)

Batch 2 (`results/bisect/`, 28 jobs) refines T_target for PC-ALM inside the single
ladder bracket that contains each crossing, at L = 8, 16, 32, 64, 128, 256. Written and
committed before any refined run had produced a row.

**Why.** The ±0.15 grid systematic is a resolution limit, not a noise limit. Adding
seeds cannot reduce it; only finer budget spacing can. The bracket is the interval
between the last rung below threshold and the first rung at or above it.

**Estimator, fixed in advance.** `analysis/refine_ttarget.py`:

    T_target = min{ T : acc(T') >= thr for every measured T' in [T, W*T] }

with hold factor W. W = 1 is the naive first crossing. The hold window exists because
PC-ALM's accuracy is non-monotone in T (it peaks then decays), so a single point that
clears the bar and is not sustained would otherwise be read as the crossing. Finer
spacing makes that failure mode more likely, not less, which is why the definition is
fixed now rather than after seeing the refined curves.

**Checked on the coarse ladder before registering.** W = 1, 1.5 and 2 give *identical*
T_target at every depth and seed, for both PC and PC-ALM. So the hold window is
currently inert; it is a precaution for the refined data, and if it ever changes a value
that change is a finding about non-monotonicity, not a tuning knob.

**Two further rules, fixed now.**
- Thresholds are depth- AND seed-matched to that seed's own BP accuracy. (An earlier
  draft of the script collapsed BP to one value per depth and produced a spurious
  T_target of 192 at L=32; caught and fixed before any refined data arrived.)
- Any depth with a censored seed is excluded from the fit. Using only the seeds that
  converged inside the budget is survivorship selection and biases the slope downward.
  This is why PC at L = 256 (2 of 3 seeds reaching, at T = 12288) stays out.

**Predictions.**
1. The refined PC-ALM exponent stays inside [1.10, 1.35]. A value outside that range
   means the coarse reading was resolution-limited in a way we did not anticipate.
2. The CI narrows relative to the current [1.173, 1.315].
3. The grid systematic, re-estimated from three offset grids (batch 1), comes out
   smaller than ±0.15, because ±0.15 was a single-replicate estimate.
4. Exclusion of 1 is *not* expected to become comfortable: the current lower edge net of
   the systematic is 1.023, and refinement moves the resolution term, not the 0.24 gap.

Registered before: any `results/bisect/` row, and any offset09/offset30 fit.
