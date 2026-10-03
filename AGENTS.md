# AGENTS

## Purpose and scope

This is a small research portfolio on data-efficient binary roof segmentation.
Prioritize an understandable experiment over a larger framework. The researcher
must be able to explain every modelling choice, result and integrity check.

Write repository documents, notebook prose, comments and docstrings in concise
professional English. Communicate with the user in German.

Implement the requested stage and its necessary support code. Consult
WORK_PLAN.md for stage scope, decisions and experiment priorities when relevant;
do not reread every document for every edit. Complete the stage, summarize the
evidence and stop at its agreed discussion point. Do not launch later experiment
blocks automatically. Routine implementation choices need no extra confirmation.

## Scientific notebooks and readable code

Notebooks are scientific reports: question, method, evidence, interpretation
and limitations. Explain consequential parameters once near their use. Avoid
repeated implementation details, installation logs and chronological work diaries.

Keep short analytical steps and experiment settings visible in notebook cells.
Move substantial loading, transformations, training, metrics and plotting into
focused modules under src/. Scripts are execution entry points. Prefer one
canonical implementation and established library operations. Avoid trivial
wrappers, speculative abstractions and empty future-stage modules.

Use descriptive names and type hints for public functions. Public modules need
brief docstrings; public functions need useful NumPy-style documentation,
including relevant tensor shapes, units, ranges and validity conventions.

Figures should identify samples, align image/label/prediction panels, explain
colours and show both successes and failures. Save useful figures under reports/.
Notebook outputs must agree with the delivered code.

## Data and evaluation

Preserve downloaded raw data. Record dataset version, source, label mapping and
any exclusions. Establish the actual data contract before adapting earlier code:
do not assume an alpha channel, a particular class encoding or complete labels.

Split by geographic region before augmentation. Check image footprints and
repeated buildings across splits; separate file names do not establish spatial
independence. Keep a fixed test region outside tuning and experiment selection.
Make geometric transforms consistent across images, targets and validity masks.
Define interpolation, padding and invalid-region handling explicitly.

Use paired data subsets for comparisons and nested subsets for learning curves.
Record exact sample IDs and seeds. Count augmented views as views, not additional
labelled examples. Disclose the labels used for validation and the limits of
image count as a proxy for annotation effort.

Change one experimental factor at a time initially. Give pretrained and randomly
initialized models adequate training and comparable tuning budgets. Report
updates and runtime as well as epochs. Freeze each comparison protocol before
its confirmatory runs; distinguish exploration from confirmation and final test
evaluation. New ideas must be motivated by training/validation evidence.

Compare common evaluation metrics across objectives, not raw totals of different
loss functions. State metric aggregation and empty-mask conventions. Never
invent results or imply that repeated seeds prove geographic generalization.

## Simple artifacts and integrity

Use one readable directory per training run under runs/. Keep exactly three
standard JSON files unless a concrete need justifies changing this contract:

- config.json: resolved configuration, including dataset/split/subset references,
  initialization, seeds, optimization, augmentation and training budget.
- metadata.json: actual input identities and hashes, weight source/version,
  Git commit, package versions and device.
- summary.json: completion status, selected epoch, aggregate metrics and runtime.

Use history.csv for epoch records, metrics.csv for per-image results and a
checkpoint for model state. Save larger predictions only when an analysis needs
them. Do not introduce overlapping manifests or archives of source code.

Keep one shared data/metadata/data_manifest.json for file pairing and raw-file
hashes, and one shared data/metadata/splits.json for regions and exact subsets.
Centralize hashing and compatibility checks in a small, plainly named module.
Document each reference's origin, what is compared and when. A locally computed
hash identifies bytes; it does not establish annotation quality or authenticity.

Record pinned pretrained-weight identities; verify trusted reference hashes
where available and load model tensors strictly. Never regenerate a reference
just to silence a mismatch. Git identifies source versions; a dependency lock
identifies the environment. Use clean committed code for reported training runs.

Write outputs safely. Reuse a completed run only when its required artifacts,
configuration and input identities match; report conflicting fields clearly.
Incomplete runs must not appear complete. New configurations get new runs.
Do not require historical scores, epoch numbers or old source snapshots to
accept a new run. Reading saved results must not require training checkpoints
or the original source checkout.

## Verification and execution

Use a project-local Python environment and a dependency lock, preferably uv.
Keep paths project-relative and installation out of notebooks. Support CPU for
data work and small tests; use MPS where available for local training. Record
the actual device and do not silently change it during an experiment. Do not
claim CUDA or another platform works without checking it.

Tests should target consequential behavior: pairing and spatial separation,
joint transforms, masked losses/metrics, pretrained initialization, BatchNorm
modes, checkpoint restoration and run compatibility. Prefer small synthetic
fixtures for logic; external data and full training do not belong in routine CI.
Use meaningful numerical tolerances; do not demand bitwise retraining equality
across devices.

Run a small real forward/backward or learning check when training behavior
changes. Execute modified notebooks with a fresh kernel and inspect changed
figures. Do not rerun expensive training for presentation-only edits. Training
starts through explicit commands, never notebook analysis or module imports.

## Git workflow and documentation

Develop in the planned pull-request-sized stages, with a few coherent commits
per stage. Commit messages describe actual changes. Keep tests and explanatory
documentation with the behavior they support. Review changed code at PR boundaries;
reserve broader reviews for the training foundation and final publication.

Keep README.md as the public scientific entry point and WORK_PLAN.md as the
concise roadmap/status record. Add setup instructions when executable code exists.
Keep prompts and personal coordination notes in ignored .local/. Exclude raw
data, weight files, environments and bulk run artifacts from ordinary Git history;
commit selected small results and figures with the necessary attribution.

Reuse appropriate scientific code from earlier work without importing private
application material or its historical artifact system. Respect source and data
licenses, keep their scopes distinct, and describe coding assistance accurately.
Do not copy an earlier repository wholesale.

Deliver a compact account of what changed, what was checked, observed findings
and remaining decisions. Batch independent work and experiment execution.
Avoid repeated repository-wide reviews or a separate discussion per training run.
