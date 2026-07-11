# The ssl-speedrun journey — seven rounds of testing latent prediction for planning

*Written 2026-07-11, after round 7. Full tables live in `results/`; design docs
in `ABLATIONS.md`. This file is the narrative: for each round — why we ran it,
what we expected, why the testbed, what we ran, what we measured, and what we
actually got.*

**The core question.** Does adding a JEPA-style latent-prediction auxiliary
loss to a decoder LM (NTP) give it planning ability that token-level losses
don't? The intuition from vision SSL: predicting in latent space lets the model
ignore unpredictable surface detail and spend capacity on semantics. The
discrete control is MTP (multi-token prediction) — same "predict the future"
pressure, but in token space.

**The verdict up front.** After 7 rounds (~90 training runs, 3 seeds each),
no latent-loss configuration ever beat the discrete control on any planning
metric. The latent family's one reliable effect is representational (effective
rank up to 85 vs MTP's ~20), never behavioral. The post-mortem diagnosis
(§Post-mortem) is that teacher forcing makes the latent *targets themselves*
short-sighted — the failure is scheme-level, not loss-level.

---

## Round 1 — Tier-0 first pass: star graph + maze, 6 methods (2026-07-07)

**Why this setup?** We needed the cheapest possible test of the hypothesis
before touching real data. Star graph (Bachmann & Nagarajan, "The Pitfalls of
Next-Token Prediction") is the *minimal task NTP is known to fail*: pick the
right arm of a star at the first token, where the answer is trivially
verifiable by planning but the first-token decision carries all the difficulty
— teacher-forced NTP learns the easy continuation ("copy the arm") and never
learns the hard first step. If latent prediction fixes anything about
planning, the cheapest place it could show up is here. Maze (path-finding in a
grid-maze tree) was added as the *graded* companion: star graph is a cliff
(first token right or wrong, chance = 1/d), so if everything fails on it we
learn nothing about *relative* method quality; maze decisions are local
junction choices, so partial planning ability shows up as partial accuracy —
it's the testbed where ablations have signal. TinyStories was queued as Tier 1
because it is about the smallest real NLP pretraining corpus on which models
produce coherent language — the smallest "real testbed" for the idea.

**Expected results.** The kill-gate from the proposal: a latent loss should
beat NTP+MTP on a planning metric. Concretely we hoped: NTP fails star graph
(replicating the paper), and some latent arm cracks it or at least beats MTP
on maze.

**Setup & why it's a good testbed.** 6L/256d GPT (~5M params), 4000 steps,
batch 128, 3 seeds — one run ≈ 3 min on an RTX 5090, so the full method matrix
is ~1 GPU-hour. Fixed budget on purpose: at convergence all proper losses
agree; separation lives in sample efficiency. Six methods: ntp, mtp, nitp
(per-token latent M=1), pertoken (M=4), jepa (pooled future window), pyramid
(both).

**Experiments run.** 6 methods × 2 tasks × 3 seeds = 36 runs; plus TinyStories
6 methods × 1 seed (27M params, ~650M tokens).

**Main metric & why.** `decision_acc` — accuracy on exactly the tokens where a
planning decision is made (star-graph first token; maze junction choices).
Overall token accuracy is dominated by forced moves and would hide the
question; `path_acc` (whole answer correct) and `erank` (effective rank,
representation health) are secondary.

**Results.** Star graph: universal failure — every method at/below chance 0.2
(ntp .076±.057, mtp .185±.040, jepa .094±.072). Maze: mtp .821±.014 ≫ ntp
.696±.043, best latent arm (pyramid) .731±.025. TinyStories: all latent arms
≤ NTP val loss (best −0.007 nats — tiny), MTP slightly worse; all erank ~190+.
**Expectation not met**: the NTP star-graph failure replicated, but no latent
loss touched it, and on maze the *discrete* future-prediction loss was the one
that helped. Consistent with the source paper: their fix was teacherless
training (a scheme change), not an auxiliary loss. The one hopeful signal:
latent arms consistently raise erank while MTP collapses it (3.4 on star
graph) — geometry and behavior dissociate.

## Round 2 — the 22-cell JEPA ablation grid + combinations (2026-07-08)

**Why this setup?** Round 1's latent arms used guessed hyperparameters
(shallow same-pass targets, untuned weights/horizons). Before concluding
anything, give the latent family its best shot: sweep every design axis the
JEPA literature says matters, on maze, where there is signal to differentiate.

**Expected results.** Some region of the design space (deep targets + EMA
teacher, longer horizons, better objectives) closes the ~.11 gap to MTP; and
the best ingredients stack.

**Setup & why.** Same maze 6×6 protocol (differences < .03 = seed noise). Axes:
teacher (same-pass stop-grad vs EMA) × target depth (1/2/4/6), source depth,
horizon k (4–32), per-token count M, chunk pooling (c=4/8, mean vs attention),
objective (cosine / L2 / +SIGReg / +VICReg / InfoNCE). 22 cells × 3 seeds,
then a stage-2: best-combo stacking, MTP+JEPA stacking, star-graph spot checks.

**Experiments run.** 22 + 4 stage-2 cells × 3 seeds = 78 runs.

**Main metric & why.** `decision_acc` again — the sweep must be judged on the
planning behavior the hypothesis is about, not on aux-loss values or erank.

**Results.** One real finding: **EMA matters because it makes deep targets
usable** (+.05 at tgt-depth ≥ 4 vs +.01 shallow; deep same-pass targets land
*below* NTP). Best cells: InfoNCE .750±.037 (erank 85 — healthiest geometry
of the project) and chunk-4 .746±.015. But the horizon curve is flat for
latents while it *rises* for MTP (k=8: .845±.025), combos don't stack (combo
.746 = best single ingredient), and MTP+JEPA on one trunk *interferes*
(.845→.795). Star graph: still exactly chance for everything, including MTP
with heads spanning the whole answer. **Expectation not met**: the user's
read proved right — most components don't matter much (everything lands
.71–.75), and the tuned best is still −.10 behind MTP. Verdict: on Tier 0 the
latent benefit is representational, not behavioral.

## Round 3 — the hardness program (2026-07-08/09)

**Why this setup?** Post-round-2 theory: maybe the latent task is too *easy* —
the aux loss collapses to ~0.02 nats early, so it stops shaping the trunk.
Make the auxiliary task genuinely hard and see if difficulty was the missing
ingredient. Two families: loss-side hardness (same inputs, harder prediction
problem) and input-side hardness (corrupt the student's context, data2vec
style, so the latent task requires reconstruction from partial information).

**Expected results.** Cells whose aux loss stays high (task provably still
hard at step 4000) should beat the .744 plateau, if hardness is what's
missing.

**Setup & why.** Maze 6×6 again — the plateau to beat is well-calibrated
there (.744±.020 JEPA, .845 MTP). The aux-loss value at step 2k/4k acts as a
built-in *health metric*: it verifies the manipulation actually made the task
harder, separating "hardness didn't help" from "we failed to make it hard."

**Experiments run.** 5 loss-side cells (target gap 8/16, discrete 64-bit LSH
codes, hard-negative InfoNCE, entropy-weighted loss mass) + 5 corruption cells
(p=.25/.5 with/without latent loss, noisy-target control) + 2 repair cells
(mask ill-posed NTP targets; 3-pass clean-NTP decontamination). 36 runs.

**Main metric & why.** decision_acc, judged jointly with the aux-loss health
metric — a hardness cell only counts as evidence if its aux loss stayed hard.

**Results.** Everything lands on the same .735–.745 plateau. The decisive
cells: LSH (0.20 nats at 4k) and hard-neg InfoNCE (0.89 nats) **stayed
provably hard to the end and still bought nothing**. Input corruption is
actively harmful when NTP rides the corrupted pass (−.09 at p=.25, −.16 at
p=.5 — label noise + distribution shift on the primary loss); the 3-pass
repair gets back to .733 but never above. **Expectation not met, cleanly**:
hardness was not the missing ingredient. As the user later put it — these
auxiliary tasks are proxies; MTP works on the real tokens, which carry the
most accurate information, and making a proxy harder doesn't make it more
informative.

## Round 4 — D2′ depth separation (2026-07-09)

**Why this setup?** Round 2's one crisp interference result (MTP+JEPA on the
same trunk: .845→.795) suggested a mechanistic fix: give the two objectives
different depths — latent loss shapes the mid-trunk "planner," MTP owns the
top. Last loss-level idea standing.

**Expected results.** Depth-separated stacks recover MTP's .845 and ideally
exceed it (planning-shaped trunk + discrete heads).

**Setup & why.** Same maze protocol; latent src→tgt at 2→2, 4→4, and weight
.5 vs .25, always with NTP + MTP k=8 on top.

**Experiments run.** 3 cells × 3 seeds = 9 runs (+ refs).

**Main metric & why.** decision_acc vs the .845 MTP-alone reference — the
question is purely "does the latent loss stop subtracting."

**Results.** Mid-trunk taps (.787/.790) sit exactly where same-depth stacking
sits (.795); halving the weight recovers half the gap (.812). **The damage
scales with weight, not placement** — extrapolate to weight 0 and you get MTP
alone. The latent loss is monotonically parasitic on MTP's planning gain in
every configuration, even while doubling erank. **Expectation not met.** This
closed the loss-level hypothesis space: rounds 1–4 tested the latent family
alone, tuned, hardened, and stacked — its reliable effect is never behavioral.

## Round 5 — synonym-rendered maze: many-to-one emission (2026-07-10)

**Why this setup?** A testbed pivot, from the user's diagnosis: maze and star
graph have *one-to-one* emission — token space **is** the semantic space — so
token CE is already the maximally informative signal and latent prediction has
nothing to offer by construction. In language/vision, many outputs are
semantically equivalent; token-space learning is noisy there, and that's
where latent learning should stand out. So instead of making the *loss*
harder (round 3), make the *dataset* hostile to token-level methods: if MTP
pays attention to superficial surface information, it should struggle.

**Expected results.** MTP degrades as surface entropy grows (its k-step
discrete targets acquire log(s) nats/token of irreducible noise); JEPA is
untouched (its pooled latent target is invariant to rendering); a crossover
appears at some s.

**Setup & why.** Each maze cell gets s surface synonyms sampled i.i.d. per
occurrence (s = 1/4/16, plus answer-only variant); evaluation scores
equivalence classes, so the semantic task is unchanged and s=1 is
bit-identical to the old task — a perfectly controlled nuisance dial. An
embedding-cosine probe (within-class vs between-class) directly measures
whether the nuisance gets absorbed.

**Experiments run.** 3 methods (ntp / mtp8 / best-jepa) × 3 nuisance settings
× 3 seeds + anchors = 27 runs.

**Main metric & why.** Class-aware decision_acc — surface-invariant, so any
method movement is attributable to the nuisance, not to a changed task.

**Results.** Null, with a sharp mechanism. MTP is *exactly flat* (.845→.846 at
s=16, despite 2.77 nats/token of target noise); everyone flat. The probe
explains it: every method learns near-perfect synonym clustering in the
embedding table (within-cos ~.9, between ~.0) from NTP pressure alone. Once
that invariance is learned, the log(s) floor is a **constant loss offset, not
gradient noise** — the within-group residual backprops through (e_t − ē) ≈ 0.
**Expectation not met, and the user called the reason in advance**: token-level
invariance is defeatable by a per-token lookup; the nuisance must be
*structural* to have any chance.

## Round 6 — top-3 longest paths: order-invariant answers (2026-07-10)

**Why this setup?** The escalation round 5 demanded: an invariance that no
embedding lookup can absorb. The user's idea — ask for the 3 longest paths in
the maze tree, output **in any order**, either direction. Now the nuisance
lives in the sequence structure (log(3!·2³) ≈ 3.9 nats at path boundaries),
composable with synonyms to mimic both invariance types of language at once.

**Expected results.** MTP's k-step heads straddle path boundaries where the
target marginal is a mixture over order/direction choices — discrete targets
get genuinely noisy in a way no lookup fixes; the pooled latent target is
order-robust (mean over the window). If the latent story is right anywhere on
Tier 0, it is here — this cell was *designed* to be JEPA-favorable.

**Setup & why.** Ground truth is exact: the k longest paths in a tree =
max-distance endpoint pairs from all-pairs BFS; eval accepts any distinct
valid simple paths whose length multiset matches gold, so no gold-sequence
matching anywhere. Pilots calibrated size (tp1 5×5 saturated at .941 → grid
on tp3 5×5).

**Experiments run.** 3 methods × {order-only, order×syn16} × 3 seeds + pilots
= ~24 runs. Plus an output-space audit at the user's request.

**Main metric & why.** `set_acc` — all 3 emitted paths valid, simple,
distinct, lengths matching the gold multiset. Verification-based, so all
equivalent orderings count; measures exactly the semantic task.

**Results.** Order nuisance alone: null — all methods .79–.80, and notably it
**neutralizes MTP** (its +.15 maze edge vanishes into the mixture, exactly the
predicted mechanism — but NTP/JEPA don't *gain*, everyone just converges).
Compositional order×syn16 is the first nuisance that hurts at all, and the
ranking is **MTP .746 > JEPA .697 > NTP .673** — the discrete auxiliary is
the *most* robust precisely in the cell built to favor pooled latents.
**Expectation falsified in the strongest available form.** The audit also
exposed a design flaw we fixed in round 7: path lengths cluster (spread ≤ 1
by a trimming argument), so ~82% of instances fall to a copy-paste shortcut —
transcribe edges from the input, decide a few forks.

## Round 7 — gridembed: draw the maze (2026-07-11)

**Why this setup?** Kill the copy shortcut. The user's inversion: don't find
paths in the maze — *generate the maze figure itself* from a shuffled edge
list over randomly relabeled vertices. Every placement is constrained by the
whole edge list; nothing in the input can be transcribed into the output.

**Expected results.** Two live hypotheses: (a) NTP gets stuck in myopic
placements (the Bachmann–Nagarajan regime, but graded rather than a cliff);
(b) whichever auxiliary provides real lookahead pressure escapes. This round
was also the first with a *quantified* difficulty floor: greedy baselines
solve 0.000 (naive) and 3%/0.1% (locally-pruned, 4×4/5×5) — so any score
above ~3% is evidence of genuine lookahead.

**Setup & why.** Output = the full (2H−1)×(2W−1) drawing: vertex labels at
even/even slots, explicit E(dge)/W(all) tokens between them (the user's
correction — without explicit edge slots we can't verify the model knows the
connectivity). Per-sample nuisance: fresh label permutation + shuffled edge
list + random D4 perspective, so the target is never the same twice;
evaluation is pure verification (permutation valid ∧ stated edge set ==
input edge set exactly) on *held-out* mazes (hash-disjoint from training).
Median 324 valid embeddings per maze at 4×4 — a real many-to-one output
space, ~1e−11 acceptance fraction.

**Experiments run.** NTP difficulty pilots at 4×4/5×5, then 4×4 × {ntp, mtp8,
jepa} × 3 seeds + 5×5 mtp8 floor-lift check = 14 runs.

**Main metric & why.** `embed_acc` (fully valid drawing) with `edge_f1` as
the graded signal — F1 between stated and true edges keeps measuring progress
when full solutions are at 0, which is what star graph's binary metric could
never do.

**Results.** The clearest separation of the project — in MTP's favor. Best
embed_acc: **mtp8 .203±.082 > jepa .065±.019 > ntp .040±.035**; edge_f1 .878
vs .793 vs .564. NTP's trajectory is the finding: it first emits valid
permutations with wrong edges, then trades permutation validity for local
edge satisfaction, plateauing at 3–5% — **numerically the greedy baseline**.
NTP learns the myopic policy, exactly what the star-graph authors found, now
visible in a graded metric. At 5×5, MTP lifts edge_f1 to .78 where NTP sits
at .08 — lookahead training extracts structure where NTP can't start.
**Expectation half-met**: the task worked exactly as designed (planning-
limited, no shortcut, NTP = greedy), but the beneficiary of that design was
the token-space auxiliary, again.

---

## Post-mortem: why the latent losses never won

**The empirical law (rounds 1–7).** On every Tier-0 task — one-to-one,
many-to-one, order-invariant, compositional, planning-limited — the ranking
among auxiliaries was MTP ≥ JEPA ≥ NTP whenever anything separated at all,
and JEPA stacked with MTP strictly subtracts. Every nuisance we injected was
either absorbed as a learned invariance (synonyms → embedding lookup; order →
boundary bookkeeping), after which it contributes a *constant loss offset and
no gradient*, or it degraded token CE less than it degraded latent regression.

**The diagnosis (user, 2026-07-11): teacher forcing makes the latent targets
short-sighted.** The JEPA target is a hidden state of the (EMA) teacher — but
that teacher encodes the *ground-truth prefix* under teacher forcing, and its
hidden states are shaped almost entirely by NTP pressure. Features that
predict the next one-or-two tokens are what NTP rewards, so that is what the
hidden states contain; information about the far future has little incentive
to be there. Asking the student to predict those states is therefore asking
it to predict *short-range features by proxy* — the auxiliary inherits the
myopia of the primary loss it was meant to correct. MTP escapes this because
its k-step-ahead targets are raw future tokens predicted **without
conditioning on the intermediate ground truth**: the model must bridge the
gap internally, which is exactly a (bounded-horizon) lookahead demand. This
one mechanism explains the project's three stubborn regularities:

1. *Why no latent variant mattered* (rounds 2–4): teacher/depth/horizon/
   objective all reshape *how* the short-sighted target is predicted, never
   *what information it contains*. Hardness (round 3) kept the proxy task
   hard — but a harder prediction of short-range features is still
   short-range. Deep-target + EMA being the best variant fits too: deeper
   teacher layers are the *least* token-local features available — the
   direction was right, the ceiling was low.
2. *Why MTP wins exactly on planning-limited tasks* (maze junctions, round
   7) and loses its edge when its targets become mixtures (round 6): its
   advantage tracks the cleanliness of unconditioned future-token targets,
   not anything representational.
3. *Why star graph resists everything*: the required lookahead (whole arm)
   exceeds any k we trained, and the fix in the literature is teacherless
   training — a scheme change that removes the ground-truth crutch entirely,
   for tokens and latents alike.

**What this predicts (untested).** If the diagnosis is right, the fix is not
a better latent loss but a better latent *target*: (a) targets from a teacher
that is itself trained without teacher forcing (or on backward/suffix
context, so its states must summarize the future); (b) latent prediction at
rollout positions rather than forced positions; (c) teacherless/multi-forward
training with latent consistency. Also still open: Tier-1 transfer — the one
consistent latent benefit is representation geometry (erank 85 vs 20), and
TinyStories probes could show that geometry buying something token metrics
don't see. Those are the experiments this document hands to the next phase.
