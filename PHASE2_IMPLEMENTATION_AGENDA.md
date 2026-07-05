# Phase 2 Implementation Agenda — hybrid-rl-index-trading

Elaborated by Fable (planning/teaching pass) from `RESEARCH_ROADMAP.md` Phase 2 plus item 1.5, for direct handoff to a coding agent. Companion to `PHASE1_IMPLEMENTATION_AGENDA.md` — read that first; Phase 2 assumes all of Phase 1 (per-window ratio normalisation, `(ratio−1)·100` scaling inside `WeightedMSELoss`, temporal split with gap `rolling_window_size + forecast_horizon`, best-model checkpoint, batch-averaged loss, gradient clipping, a-presets dropped, fresh-model bootstrap per Phase 1 §7) has landed.

**Scope:** 1.5 (walk-forward validation harness), 2.1 (two-stage pretrain + fine-tune), 2.2 (temporal/recency loss weighting), 2.3 (monthly drift monitoring), 2.4 (transaction-cost-aware evaluation).

All paths relative to repo root. Line numbers refer to the pre-Phase-1 files read for this plan — **Phase 1 shifts lines, so navigate by symbol name, not line number.** Where Phase 1 changes a mechanism (split logic, normalisation), this agenda describes the post-Phase-1 state.

Key files:
- `src/pipeline/predictors.py` — `NNPredictor` (`run_training`, `split_data`, `prepare_data`, `dataset_train`/`dataset_val` properties, `loss_criterion` property, `initial_lr`), `LSTMModel.run_epoch` / `TransformerModel.run_epoch`, `PredictorManager` (`fine_tune_predictors`, `instantiate_predictor`, `get_predictors_by_type_sorted`, `add_predictors_from_dir`, `preset_type_dict`), `TimeSeriesDataset`
- `src/pipeline/pt_metrics.py` — `WeightedMSELoss` (post-Phase-1 version with `target_scale=100`), `HitRateMetric`
- `src/pipeline/preprocessing.py` — `create_rolling_window_view` (L528), `create_train_validation_split` (post-Phase-1 version with `gap_size`, `extra_arrays`), `StockPriceDataManager` (`b/c/d_interp_prices`)
- `src/workflow.py` — path constants (L36–69), `fine_tune_predictors` (L357), `predict_and_trade` (L533), `function_schedule` (L691), `request_map` (L727), `chatter` (L125, `MailChatbot`), decorators pattern (`@timed_callback_decorator(callback=chatter)` / `@retry_decorator(on_error_callback=chatter)`)
- `src/pipeline/rl_environments.py` — `RLTradingEnv.compute_predicted_prices` (L936), `compute_predicted_potentials` (L915), `current_potential_estimates` (L989), `commission_rate` (L158, buy at `price·(1+c)` L501, sell at `price·(1−c)` L561)
- `src/pipeline/financial_products.py` — `KOCertificate.leverage_series` (L660: `underlying · subscription_ratio / price`), `intrinsic_value_series` (L704: `|underlying − base|·ratio`, → `1e-10` after KO), `price_series` (L735: `intrinsic + risk_premium`), `is_ko_series` (L626), `base_price_change_per_annum` (L480), `KOCertificateSet.load_from_csv` (L892), `leverage_frame` (L1176)
- `data/portfolios/*.csv` — scraped certificate sets; **actual columns:** `isin, direction, issue_date, subscription_ratio, risk_premium, date_base_price_tuple_0, date_base_price_tuple_1, date_base_price_tuple2_0, date_base_price_tuple2_1`. **There is no bid/ask or spread column** — see §5 flag.

---

## 0. Data-density ground truth (drives several decisions below)

Interpolated price history runs **2014-10-23 → present** (~11.7 years; latest file `2026-06-21 … from 2014-10-23 to 2026-06-18.csv`). Because every preset uses `daily_prediction_hour=16`, `create_rolling_window_view` yields **one sample per trading day** for b- and c-presets and **one sample per week** for d-presets, regardless of sampling rate:

| Preset | RW + FH (rows) | Samples/year | Total samples (~11.7y) | Samples in 12 months | Samples in a 6-month window |
|---|---|---|---|---|---|
| b1 | 42+14=56 | ~253 | ~2,900 | ~253 | ~126 |
| b2 | 70+14=84 | ~253 | ~2,900 | ~253 | ~126 |
| c1 | 15+3=18 | ~253 | ~2,900 | ~253 | ~126 |
| c2 | 40+5=45 | ~253 | ~2,900 | ~253 | ~126 |
| d1 | 16+2=18 | ~52 | ~590 | ~52 | ~26 |
| d2 | 24+3=27 | ~52 | ~580 | ~52 | ~26 |
| d3 | 48+6=54 | ~52 | ~555 | ~52 | ~26 |

Remember the Phase-1 gap costs an additional `RW + FH` **sample rows** between train and val. Two hard consequences, both flagged in detail below:

1. **A 12-month fine-tuning window is impossible for d-presets** (§2 flag). d3 needs 53 weekly rows just to form one (X, Y) pair; 52 weeks of data yields zero samples. Even d1 yields ~35 samples, of which the gap (18) eats half. **The roadmap's D2 decision ("12 months") can only apply to b/c-presets.**
2. **6-month walk-forward validation folds are too thin for d-presets** (~26 samples/fold → a hit-rate estimate with ±10pp standard error). Use 12-month folds for d-presets (§1).

---

## 0b. Recommended implementation order

```
Step 1   1.5  Walk-forward harness (new src/pipeline/evaluation.py) — with a pluggable
              training-procedure hook so it can later mimic two-stage training
Step 2   2.3a Live-prediction LOGGING only (hook in predict_and_trade) — 5 lines in the
              workflow + one helper; start accruing live data immediately, months before
              the comparison logic can be enabled
Step 3   2.1  Two-stage training (pretrain base store + fine_tune_predictors rework)
Step 4   2.2  Recency-weighted training loss (depends on 2.1's window for its default)
Step 5   ---  Run the first walk-forward baseline using the FINAL production procedure
              (two-stage + recency weighting) — this produces the bounds file
Step 6   2.4  Transaction-cost evaluation (consumes walk-forward fold predictions)
Step 7   2.3b Drift comparison + monthly schedule + chatbot alert (needs Step 5's bounds)
```

Reasoning:
- **1.5 first** (confirming the prompt's hypothesis) — but with a twist: 2.3's bounds must describe the *deployed* models. If the baseline is generated with plain fresh training and production then switches to two-stage + recency weighting (2.1/2.2), the bounds describe a different model family and drift comparisons are meaningless. Hence the harness is *built* first (Step 1) but the *baseline run* happens after 2.1/2.2 land (Step 5). Build the harness with a `training_procedure` parameter so this doesn't require rework.
- **2.3 logging split from 2.3 comparison.** The rolling 30-day live accuracy needs weeks of logged predictions before it says anything. Deploying the logging hook early (Step 2) is nearly free and means real data exists by the time Step 7 lands. The comparison half has a designed no-baseline fallback anyway (§4), so there is no correctness risk in the interim.
- **2.2 after 2.1**: the half-life default is defined relative to the fine-tuning window (§3); implementing it first would mean choosing a constant that 2.1 immediately invalidates.
- **2.4 after the baseline run**: it can technically run on any predictor's val split, but its honest input is walk-forward fold predictions (many out-of-sample samples across regimes), so sequence it after Step 5. Its premium-curve half (`derive_premium_curve`) has no dependencies and can be built any time.

Commit after each step; each leaves the system runnable.

---

## 1. Item 1.5 — Walk-forward validation harness

### 1a. Location: new module `src/pipeline/evaluation.py`

Recommendation between the roadmap's two options (`predictors.py` vs new `evaluation.py`): **new `evaluation.py`**, and make it the home for all of Phase 2's offline-evaluation code (this harness, 2.4's cost evaluation, 2.3's baseline load/compare helpers). Reasoning:
- `predictors.py` is already ~2,200 lines and Phase 1 grows it further; the harness is a *consumer* of predictors, not part of them.
- 2.3 and 2.4 both need to import baseline/aggregation logic; putting that in `predictors.py` would drag model classes into `workflow.py` import paths that only need a CSV comparison.
- It keeps the harness invocable standalone (`python -m src.pipeline.evaluation` or from a notebook) without touching the workflow module, matching the roadmap's "not wired into the weekly cron" requirement.

`evaluation.py` imports from `predictors.py` (`LSTMPredictor`, `TransformerPredictor`, `PredictorManager` for the preset dict), `preprocessing.py`, and `pt_metrics.py`. It must NOT import `workflow.py` (which executes initialisation code, reads `private/`, and would make the module unimportable in a clean environment).

### 1b. Fold scheme (decided, with per-preset adjustment)

- **Expanding (anchored) window** — confirmed as correct for this system: the production two-stage design (§2) pretrains on *full* history that only ever grows; a sliding-window harness would evaluate a regime the live system never uses. (The 12-month fine-tune stage is *also* reproduced per fold when the two-stage procedure is plugged in — see 1e.)
- **Validation window per fold:** ~6 months (126 samples) for b/c-presets; **12 months (52 samples) for d-presets** — the per-preset-category adjustment required by §0. 26 samples per fold would make fold-level hit rates almost pure noise (binomial SE at p=0.5, n=26 is ±9.8pp; at n=52 it is ±6.9pp, still wide but usable when aggregated over folds).
- **Evaluation span:** last 4 years of data → **8 folds** for b/c (6mo each), **4 folds** for d (12mo each). The first fold then trains on ~7.7 years (d3: ~400 weekly rows → ~347 samples — comfortably enough). Do not push the evaluation span further back: earlier folds would train on <6 years, and the 2014–2019 regime is already well-represented in every fold's training set.
- **Gap:** reuse Phase 1's rule — `rolling_window_size + forecast_horizon` sample rows between each fold's train end and val start. Same conservative rows-vs-samples caveat as Phase 1 §5 (over-conservative for b-presets by ~14x in wall-clock terms, but only ~56–84 samples, which is affordable; exact for c/d).

### 1c. Mechanism — reuse via series truncation, not new split machinery

The cleanest reuse of existing machinery: **per fold, instantiate a fresh predictor on a truncated price series and let the (post-Phase-1) chronological split produce the fold's val block as its tail.** No changes to `create_train_validation_split` or `split_data` are needed at all:

```python
# evaluation.py (sketch — signatures are normative, bodies illustrative)
from dataclasses import dataclass, field

@dataclass
class FoldResult:
    fold: int
    train_end: pd.Timestamp        # last Y-target date used for training
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    n_train: int
    n_val: int
    loss_val: float                # scaled WeightedMSELoss, batch-averaged — comparable to run_training's
    hit_rate_val: float
    naive_loss_val: float          # Phase 1's always-ratio-1.0 baseline
    final_step_mse: float          # ((pred[:, -1] - Y[:, -1]) * 100)**2 mean — see 1f, needed by 2.3
    n_epochs_ran: int
    best_epoch: int
    predictions_val: np.ndarray = field(repr=False)   # ratio space, kept for 2.4
    targets_val: np.ndarray = field(repr=False)
    reference_prices_val: np.ndarray = field(repr=False)

def walk_forward_validate(architecture: str,               # 'LSTM' | 'Transformer'
                          preset_type: str,                # 'b1' ... 'd3'
                          price_series: pd.Series,         # full interp series for the preset category
                          n_folds: int = None,             # default: 8 for b/c, 4 for d
                          val_window_days: int = None,     # default: 182 for b/c, 365 for d
                          train_epochs: int = 200,
                          early_stopping_patience: int = 10,
                          training_procedure: callable = None,   # None => fresh training; see 1e
                          **predictor_kwargs) -> list[FoldResult]:
    ...
```

Per fold `k` (0-based, oldest first):
1. Compute `val_end_k = series_end − (n_folds − 1 − k) · val_window` and truncate: `fold_series = price_series[:val_end_k]`.
2. Compute the fractional split that makes the chronological tail exactly the fold's val window. Because sample counts, not calendar days, drive the split, do this empirically: instantiate the predictor on `fold_series`, read `len(predictor.X)` (call it `n`), count samples whose `Y_dates[:, -1] > val_end_k − val_window` (call it `n_val`), and set `predictor.validation_split = n_val / n` **before** training (the setter re-splits). The Phase-1 split then automatically inserts the gap between train and val.
3. Train from **fresh random init** (default) or via `training_procedure` (1e). Do **not** warm-start fold k+1 from fold k — folds must be independent draws of the training procedure, otherwise fold-to-fold variance (which we explicitly want to surface) is suppressed.
4. Record `FoldResult` from the predictor's post-training properties (`loss_val`, `hit_rate_val`, `naive_loss_val` from Phase 1 §4f, `predictions_val`, `Y_val`, `X_reference_prices_val`).
5. Set `model_save_directory=None` on fold predictors — fold models are throwaway; they must NOT be saved into `data/saved_models/` where `PredictorManager` would rank them by their (older-data) ValHR.

**Why truncation beats explicit index plumbing:** every fold sees exactly what a model trained at that historical date would have seen (windowing, gap, split all recomputed on the truncated series), and zero new parameters thread through `NNPredictor`/`create_train_validation_split`. The cost — re-running `prepare_data` per fold — is milliseconds.

**Early-stopping caveat (state it in the docstring):** within each fold, the fold's val block is used both for early stopping/checkpoint selection and as the fold's reported OOS metric. This mildly optimistic coupling *matches production behaviour* (the weekly fine-tune also early-stops on its chronological tail), so the bounds describe the deployed procedure — which is what 2.3 needs. A nested split (train / early-stop-val / test) would be purer but cuts d-preset samples a third way; explicitly rejected for now. Consequence: live accuracy (truly forward, §4) will sit slightly below fold accuracy on average — one more reason the drift trigger uses `mean − 1·std` sustained 2 months, not the mean itself.

### 1d. Aggregation and the bounds file

```python
def aggregate_folds(fold_results: list[FoldResult]) -> dict:
    hr = np.array([f.hit_rate_val for f in fold_results])
    return {
        'n_folds': len(fold_results),
        'fold_hit_rates': hr.tolist(),            # keep raw per-fold values — variance is signal
        'hit_rate_mean': hr.mean(), 'hit_rate_median': np.median(hr), 'hit_rate_std': hr.std(ddof=1),
        'hit_rate_lower_bound': hr.mean() - hr.std(ddof=1),      # consumed by 2.3
        'loss_mean': ..., 'loss_std': ...,
        'final_step_mse_mean': ..., 'final_step_mse_std': ...,   # consumed by 2.3 (MSE track)
        'naive_loss_mean': ...,
        'val_window_days': ..., 'fold_boundaries': [...],
    }

def save_walk_forward_baseline(results_per_preset: dict[str, dict], directory: Path,
                               procedure_label: str) -> Path:
    # writes data/walk_forward_baselines/<file_title('Walk-Forward Baseline', '.json')>
    # top-level: {'created': ..., 'procedure': procedure_label, 'presets': {…aggregates…}}

def load_latest_baseline(directory: Path) -> dict:
    # filemgmt.most_recent_file(directory, '.json') → json.load; returns None if dir empty/missing
```

- **Bounds are `mean − 1·std` across folds** (hit rate) and `mean + 1·std` (loss/MSE, higher-is-worse), per the roadmap's stated trigger. With 8 folds the std estimate is rough; that is acceptable — do not substitute a parametric CI, but DO surface the raw `fold_hit_rates` list in the file and in the run's printed summary. **High fold-to-fold dispersion is itself a finding** (regime sensitivity); the summary should print a one-line warning when `hit_rate_std > 0.05`.
- New path constant in `workflow.py` (and mirrored as a default in `evaluation.py` without importing workflow): `WALK_FORWARD_BASELINES = DATA / "walk_forward_baselines"`. Use `filemgmt.file_title()` naming so `most_recent_file` works on it.
- The baseline file must record `procedure` (`'fresh'` vs `'two_stage_recency'`) — 2.3 refuses to compare against a baseline whose procedure differs from what production currently runs (cheap string check, prevents the exact staleness bug described in §0b).

### 1e. `training_procedure` hook (forward-compatibility with 2.1/2.2)

`training_procedure(predictor, fold_series) -> None` is called instead of the default `predictor.run_training(...)` when provided. After 2.1/2.2 land, add to `evaluation.py`:

```python
def two_stage_procedure_factory(pretrain_epochs, finetune_epochs, finetune_lr,
                                finetune_window_days_by_category, recency_half_life_fraction):
    def procedure(predictor, fold_series):
        # stage 1: pretrain on the fold's full (truncated) history, default LR, no recency weighting
        predictor.run_training(custom_n_epochs=pretrain_epochs, ...)
        # stage 2: re-point the SAME instance at the trailing window, drop LR, enable recency weighting
        #          (mirrors PredictorManager.fine_tune_predictors — see §2; keep the two code paths
        #           behaviourally identical, ideally by calling a shared helper)
        ...
    return procedure
```

The Step-5 baseline run uses this procedure so bounds describe production. Keep the fold-internal fine-tune window per-category identical to production (§2c): 365d for b/c, 1825d for d.

### 1f. `final_step_mse` — why it exists

Fold `loss_val` is the multi-step `WeightedMSELoss` (with forecast-step weights). Live monitoring (§4) can only score the *final* forecast step (that's what a resolved prediction is). To make the MSE track comparable, each fold also records plain final-step scaled MSE: `np.mean(((predictions_val[:, -1] - targets_val[:, -1]) * 100) ** 2)`. Hit rate has no such mismatch (`HitRateMetric` already uses only the final step).

### 1g. Invocation, cost, parallelism

- **Standalone only.** Add an `if __name__ == '__main__':` block in `evaluation.py` that runs all 7 presets × LSTM with defaults and saves the baseline. Do NOT add to `function_schedule`. Operational cadence (quarterly / event-driven) is a human decision per the roadmap's schedule; optionally add a `describe`-style read-only entry to `request_map` later ("describe walk-forward baseline") but no `do`-entry.
- **Cost estimate:** 7 presets × 8 folds × (≤200 epochs, patience 10) ≈ 56 training runs. At minutes per run on CPU this is hours — fine for a quarterly manual job. Structure fold execution as a pure function `_run_fold(fold_spec) -> FoldResult` so it can later be dispatched via `multiprocessing.Pool`; do not implement the pool now (torch CPU threading already parallelises within a run, and nested parallelism needs care).

### Verification (1.5)

1. **Fold-boundary tripwires** (assert inside the harness, always on): for every fold, `Y_dates_train[-1][-1] < X_dates_val[0][0]` (Phase 1's no-leakage property) and fold k's `val_end` == fold k+1's `val_start` (contiguous, non-overlapping val coverage).
2. Run the harness on `c1` with `n_folds=2, train_epochs=5` as a smoke test: completes, returns 2 `FoldResult`s, JSON round-trips through `save_walk_forward_baseline`/`load_latest_baseline`.
3. **Monotonic-sample check:** `n_train` strictly increases across folds (expanding window); `n_val` ≈ 126 (b/c) / ≈ 52 (d) per fold.
4. Sanity: fold hit rates should scatter around ~0.5 post-Phase-1 (the honest baseline). If a fold reports >0.65, suspect a leakage regression before celebrating.

---

## 2. Item 2.1 — Two-stage training (pretrain + fine-tune)

### 2a. Confirmed bug in the current weekly flow (fix, don't just extend)

`workflow.fine_tune_predictors` (workflow.py L357) builds a temp `PredictorManager` over `SAVED_MODELS` with `not_older_than_n_days=60`, and `PredictorManager.fine_tune_predictors` (predictors.py L2038) then warm-starts from the **best-ValHR model of the last 60 days — which is almost always last week's fine-tuned output** (fresh files land in `WORKING_DIR_PREDICTORS`, inside the recursive scan tree). So the deployed model is a chain: week N's weights = week N−1's weights + 200 more epochs (patience 10) on (nearly) the same full-history data, with `initial_lr=0.001` every time, and with each week's link *selected by its validation hit rate*. Three compounding problems:

1. **Cumulative overfitting:** every week adds another early-stopped optimisation run on ~the same dataset from an already-fitted start point. Effective training exposure grows without bound while the data grows by ~5 samples/week.
2. **Selection-bias ratchet:** picking the max-ValHR ancestor each week and saving its ValHR into the filename ratchets the lineage toward validation-set luck (winner's-curse on a fixed val block).
3. **No stable reference:** there is no fixed "base" model whose behaviour drift could even be measured against.

Two-stage training replaces the chain with: **every week = (stored pretrained base) + (one fine-tune run on the trailing window at low LR)**. Week N and week N−1 differ only by one week of data and one fine-tune run — drift cannot compound across weeks.

### 2b. Storage layout

- New directory: `data/pretrained_base_models/` — new constant in workflow.py: `PRETRAINED_BASE_MODELS = DATA / "pretrained_base_models"`.
- **It must live outside `SAVED_MODELS`** (`data/saved_models/`). Same trap as Phase 1 §7: the deployment `pred_manager` scans `SAVED_MODELS` with `recursive=True`, and a base model with a fresh date in its filename would be picked up, ranked by its ValHR, and potentially deployed *and* warm-started from — silently recreating the chain. A sibling directory is the only safe placement.
- Base files use the existing naming convention (`YYYY-MM-DD hh_mm_ss LSTM Model SR… RW… FH… TrainL… ValL… TrainHR… ValHR….pt` via `save_model_file`) so a `PredictorManager(initialisation_dir=PRETRAINED_BASE_MODELS)` can parse/select them with zero new parsing code. Selection of "the" base per (architecture, preset): most recent by filename date — NOT best ValHR. Rationale: the base's job is maximum-history coverage, and each pretrain refresh supersedes the previous one; ranking bases by ValHR would reintroduce the selection ratchet at the base level. Implement via a small helper in `PredictorManager` (`get_latest_predictor(architecture, preset_type)` — filter as in `get_predictors_by_type_sorted`, then sort by the `YYYY-MM-DD` filename prefix, which `add_predictors_from_dir` already parses).

### 2c. Fine-tuning window — per-category, because 12 months is impossible for d-presets

**FLAG (contradicts roadmap D2 as written):** with Phase-1 windowing + gap, a 365-day slice yields for d-presets: d1 ≈ 35 samples minus 18-row gap → ~10 train / 7 val; d2 ≈ 26 samples with a 27-row gap → **negative**; d3: 52 weeks < RW+FH = 54 rows → **zero samples before any split**. The D2 decision can only mean b/c-presets. Decision to encode:

| Category | `finetune_window_days` | Resulting samples (train+gap+val) | Rationale |
|---|---|---|---|
| b, c | **365** | b1: ~143/56/50 · c2: ~121/45/42 | D2 as decided |
| d | **1825 (5 years)** | d3: ~112/54/41 | Smallest window giving d3 a ~100+ train block; "recent regime" is inherently coarse at weekly resolution — accept and document |

Implement as a dict constant next to `preset_type_dict` in `predictors.py` (`FINETUNE_WINDOW_DAYS_PER_CATEGORY = {'b': 365, 'c': 365, 'd': 1825}`) so the walk-forward two-stage procedure (§1e) and the weekly job share one source of truth. Add a guard inside the fine-tune path: after truncation and `prepare_data`, raise `ValueError` if `len(X_train) < 50` (mirrors Phase 1 §5's gap-ate-the-training-set check) rather than training on a sliver.

### 2d. Learning rates — concrete values and where they live

- **Pretrain LR = `initial_lr = 0.001`** — the existing default in `NNPredictor.__init__`/`LSTMPredictor.__init__` (predictors.py L663/L1355). Unchanged; pretraining is just `train_fresh_predictors` (Phase 1 §7) with defaults.
- **Fine-tune LR = `1e-4`** (10x lower, per roadmap). Plumbing already exists end-to-end: `initial_lr` is a constructor kwarg → `instantiate_predictor(**predictor_kwargs)` forwards kwargs → `get_predictors_by_type_sorted(..., **instantiation_kwargs)` forwards them → `run_training` reads `self.initial_lr` when constructing the Adam optimiser. So the fine-tune path only needs to pass `initial_lr=finetune_lr` at instantiation. Expose `finetune_lr: float = 1e-4` as a parameter of `PredictorManager.fine_tune_predictors` (not buried as a magic number) and echo it in the function's status print. Note the `ReduceLROnPlateau` (factor 0.5, patience 5) still operates below the fine-tune LR — that's fine and intended.

### 2e. `PredictorManager.fine_tune_predictors` — required changes

Current body (predictors.py L2038–2086) selects best-ValHR recent model per (arch, preset), sets `model_save_directory`, and calls `run_training`. New behaviour:

```python
def fine_tune_predictors(self,
                         architectures_to_finetune, types_to_finetune,
                         finetune_working_directory,
                         pretrained_base_manager: 'PredictorManager' = None,   # NEW
                         finetune_lr: float = 1e-4,                            # NEW
                         finetune_window_days_per_category: dict = None,       # NEW, defaults to module constant
                         train_epochs=200, early_stopping_patience=10,
                         verbose_training=True, custom_step_loss_weight_range=None,
                         recency_half_life_days: float = 'auto',               # NEW in 2.2, see §3
                         ):
```

Per (architecture, preset):
1. **Load the base, not last week's output:** `instance = pretrained_base_manager.instantiate_predictor(<latest base for arch+preset>, initial_lr=finetune_lr)`. `pretrained_base_manager` is a `PredictorManager(data_manager=<same>, initialisation_dir=PRETRAINED_BASE_MODELS, recursive=False)` constructed by the caller in `workflow.py`. If `pretrained_base_manager is None`, fall back to the old behaviour (self-scan) with a loud printed deprecation warning — keeps notebooks working, but `workflow.py` always passes it.
2. **Truncate the series before training:** `instantiate_predictor` resolves the full category series; after instantiation do
   ```python
   window_days = finetune_window_days_per_category[preset[0]]
   full = instance.price_series
   instance.price_series = full[full.index >= full.index.max() - pd.Timedelta(days=window_days)]
   instance.prepare_data(); instance.split_data()
   ```
   (The `price_series` setter already resets predictions; `prepare_data`/`split_data` rebuild X/Y/refs on the truncated series. Verify the setter ordering — if Phase 1 made these fully lazy, explicit calls are still harmless.)
   Then apply the `len(X_train) < 50` guard from 2c.
3. Set `model_save_directory = finetune_working_directory`, run `run_training(custom_n_epochs=train_epochs, custom_early_stopping_patience=early_stopping_patience)` as today. Fine-tuned outputs keep landing in `WORKING_DIR_PREDICTORS` under `SAVED_MODELS`, keeping deployment selection (`pred_manager`, `not_older_than_n_days=10/30`) unchanged.

`workflow.fine_tune_predictors` (workflow.py L357) changes:
- Replace the `temp_pred_manager` (60-day self-scan) with the base manager: its whole reason to exist was finding a warm-start ancestor, which two-stage removes. Keep reinitialising the global `pred_manager` at the end (unchanged).
- The `patience_tuple` loop can stay (it just runs the base→fine-tune procedure with different patience values; each run starts from the base, so multiple runs no longer chain).
- Chatter messages: mention which base date each preset fine-tuned from (one line via the base manager's parsed names) — this is the operator's visibility into base staleness.

### 2f. Pretrain refresh cadence — recommendation

**Refresh the pretrained bases on the same quarterly, manually-triggered cycle as the walk-forward baseline (Step 5's runner), not on their own schedule.** Concretely: `evaluation.py`'s `__main__` runner gains a `--refresh-pretrain` mode (or a separate function `refresh_pretrained_bases()`) that calls `PredictorManager.train_fresh_predictors` (Phase 1 §7) with `save_directory=PRETRAINED_BASE_MODELS` for all (LSTM × 7 presets). Justification:
- The base's marginal staleness cost is low: it misses at most one quarter of data, and the *fine-tune stage sees that data every week anyway* — the base only needs to encode general dynamics, which change slowly.
- Coupling to the walk-forward run keeps base and bounds consistent by construction: bounds are regenerated with (and describe) the same-vintage pretrain procedure. Refreshing bases *without* re-running walk-forward would silently invalidate 2.3's bounds; coupling makes that mistake structurally impossible.
- Cost: 7 full-history trainings/quarter — trivial next to the 56 fold runs it accompanies.

Between refreshes, bases are read-only. Old base files can stay in the directory (latest-by-date selection ignores them); prune manually if it bothers anyone.

### 2g. First-run edge case — the Phase 1 bootstrap IS pretrain #1

Phase 1 §7's `train_fresh_predictors` bootstrap (run once after the normalisation change invalidated all old models) produces exactly what 2.1 calls a pretrained base: fresh-init, full-history, default-LR models. **Do not retrain.** Migration step, one sitting:
1. Create `data/pretrained_base_models/`; **copy** (not move) the bootstrap `.pt` files there — one per (arch, preset), the most recent if the bootstrap ran multiple times. They stay in `SAVED_MODELS` too, where they serve as the deployable models until the first two-stage fine-tune supersedes them (`not_older_than` selection handles this automatically).
2. First Saturday after 2.1 lands, the weekly job runs the new path: loads bases from the new directory, fine-tunes on trailing windows. Nothing special-cased in code — but DO add an explicit error message when `pretrained_base_manager` finds no model for a preset: `"No pretrained base for {arch}/{preset} in {dir}. Run refresh_pretrained_bases() or copy the Phase-1 bootstrap models."` (replacing the silent `print('No predictor found!'); continue`, which would otherwise let a misconfigured base dir quietly skip all fine-tuning forever — send it through the chatter callback too if one is threaded in, see §4d).

### Verification (2.1)

1. **Window truncation:** after step 2 in 2e, assert `instance.price_series.index.min() >= max − window_days` and, for c1, `len(instance.X)` ≈ 235 (not ~2,900).
2. **LR actually applied:** assert `instance.initial_lr == 1e-4` post-instantiation, and eyeball the progress bar's `LRate:` field on the first fine-tune epoch.
3. **Chain broken (the point of it all):** run the weekly job twice in a row; instrument that both runs warm-start from the same base file (log the base file path). Week-2's model must NOT descend from week-1's output. A direct check: `torch.equal` of a base parameter tensor against the pre-training weights of run 2's instance.
4. **d-preset guard:** temporarily set `finetune_window_days_per_category['d'] = 365` and confirm the `ValueError` guard fires for d2/d3 instead of training on garbage.
5. **Deployment isolation:** after a fine-tune run, `PredictorManager(initialisation_dir=SAVED_MODELS, recursive=True).predictors` contains the new fine-tuned files but nothing from `pretrained_base_models/`.

---

## 3. Item 2.2 — Temporal (recency) loss weighting

### 3a. Decisions being encoded (from roadmap R5, verified)

- `weight = exp(-lambda * days_ago)`, `lambda = ln(2) / half_life_days`. Plain exponential; no power variant.
- **Training loss only.** Validation epochs, early stopping, `ReduceLROnPlateau`, and Phase 1.3's best-checkpoint comparisons keep using the unweighted (but scaled) `WeightedMSELoss`. Verified against the code: `run_training` currently passes the same `self.loss_criterion` to both the training and validation `run_epoch` calls — this is precisely the line that must fork.
- Recency weighting applies **only in the fine-tune stage** (2.1), never in pretraining — pretraining's job is general dynamics including old crises; discounting them there defeats its purpose.

### 3b. Half-life default — computed, not hand-waved

The roadmap's "~6 months" was proposed before D2 fixed the window at 12 months. Within a 12-month window, a 6-month half-life gives the oldest sample weight `exp(-ln2·365/182) = 0.25` and, after mean-normalisation (3d), a newest-to-oldest ratio of 4:1 with the effective sample count reduced to roughly 70% of nominal. That is a reasonable, moderately aggressive tilt — **but only because 6mo = window/2.** Applying a fixed 6-month half-life to the d-presets' 1825-day window (§2c) would down-weight the oldest half of an already-thin dataset to ≤6% — effectively discarding data the window was widened to keep.

**Decision: half-life scales with the window — `half_life_days = finetune_window_days / 2`.** So b/c: 182.5 days (matching the roadmap's instinct exactly); d: 912.5 days. Encode as the `'auto'` default: `recency_half_life_days='auto'` in `fine_tune_predictors` resolves to `finetune_window_days_per_category[category] / 2`; a float overrides; `None` disables weighting entirely (pretrain path uses `None` implicitly by never passing the parameter).

### 3c. Where `days_ago` comes from — plumbing audit (this is the real work)

Audit result: per-sample dates do **not** currently reach the loss. `run_epoch` (both models) iterates `for idx, (x, y) in enumerate(dataloader)`; `TimeSeriesDataset.__getitem__` returns `(x, y)`; dates stop at `NNPredictor._Y_dates_train`. Three plumbing changes, all small:

1. **`TimeSeriesDataset`** (predictors.py L26): add optional constructor arg `sample_weights: np.ndarray = None` (shape `(n_samples,)`). Store as `float32`. `__getitem__` returns `(x, y, w)` when weights exist, `(x, y)` otherwise. Length-check against `x`.
2. **`NNPredictor`:**
   - New constructor param `recency_half_life_days: float = None` (threaded through `LSTMPredictor`/`TransformerPredictor` super-calls like every other base param). `None` ⇒ everything behaves exactly as today.
   - `dataset_train` property computes weights when enabled:
     ```python
     @property
     def dataset_train(self):
         w = None
         if self.recency_half_life_days is not None:
             target_dates = self.Y_dates_train[:, -1]                      # last forecast-step date per sample
             ref = self.Y_dates_train[:, -1].max()                        # newest training target — NOT datetime.now()
             days_ago = (ref - target_dates) / np.timedelta64(1, 'D')
             w = np.exp(-np.log(2) / self.recency_half_life_days * days_ago)
             w = w / w.mean()                                              # normalise to mean 1.0 — see 3d
         return TimeSeriesDataset(self.X_train, self.Y_train, sample_weights=w)
     ```
     Use `Y_dates_train[:, -1]` (the date the prediction resolves) rather than `X_dates`; and use the max *training* date as reference, not wall-clock now — deterministic, reproducible, and unaffected by weekend-vs-weekday run times. `dataset_val` never gets weights.
   - `train_loss_criterion` property: returns `metrics.RecencyWeightedMSELoss(step_weights=<same linspace as loss_criterion>)` when `recency_half_life_days` is set, else `self.loss_criterion`. `run_training` passes `loss_criterion=self.train_loss_criterion` to the **training** `run_epoch` and keeps `self.loss_criterion` for the **validation** `run_epoch`. The reported/`describe()`d `loss_train` property (full-dataset recompute) can stay on the unweighted criterion — document that "final training loss" is unweighted for comparability.
3. **`run_epoch`** (both `LSTMModel` L551 and `TransformerModel` L279 — mirror the edit): unpack flexibly and forward weights:
   ```python
   for batch in dataloader:
       x, y = batch[0].to(device), batch[1].to(device)
       w = batch[2].to(device) if len(batch) == 3 else None
       out = self(x) / self(x, y, tf_ratio)              # per current class
       loss = loss_criterion(out.contiguous(), y.contiguous()) if w is None \
              else loss_criterion(out.contiguous(), y.contiguous(), sample_weights=w)
   ```
   Validation loaders produce 2-tuples, so validation calls are untouched by construction.

### 3d. `RecencyWeightedMSELoss` — class design (`pt_metrics.py`)

Subclass, don't wrap — the weight logic composes multiplicatively with the existing step weights and target scaling:

```python
class RecencyWeightedMSELoss(WeightedMSELoss):
    """WeightedMSELoss with an additional per-SAMPLE weight (recency). Training use only:
    validation / early stopping must stay on the unweighted parent (see RESEARCH_ROADMAP 2.2)."""
    def forward(self, predictions, targets, sample_weights=None):
        if sample_weights is None:
            return super().forward(predictions, targets)          # graceful degradation
        if not isinstance(predictions, torch.Tensor): predictions = torch.tensor(predictions)
        if not isinstance(targets, torch.Tensor): targets = torch.tensor(targets)
        scaled_diff = (predictions - targets) * self.target_scale  # Phase-1 scaling preserved
        losses = scaled_diff ** 2                                  # (B, S)
        if self.step_weights is not None:
            losses = losses * self.step_weights.to(losses.device)  # per-step, broadcasts over batch
        return torch.mean(losses * sample_weights.unsqueeze(1))    # (B,1) broadcasts over steps
```

Notes:
- **Weights are computed in the dataset, not the loss.** The loss stays stateless w.r.t. dates; the half-life hyperparameter lives on `NNPredictor` (constructor arg, per-instance), which is the same altitude as `forecast_step_loss_weight_range`. No config file entry — it is set by the fine-tune call chain (§2e) and defaults to off.
- **Mean-1.0 normalisation (in `dataset_train`, above) is load-bearing**, not cosmetic: without it, the *absolute* training-loss magnitude shrinks with the weighting strength, which would (a) change the gradient scale relative to Phase 1.6's `clip_grad_norm_(max_norm=…)` — silently converting the clip into a stronger/weaker LR cap depending on half-life — and (b) make training-loss curves incomparable across half-life settings. With mean-normalised weights, expected loss magnitude and gradient scale match the unweighted case. (One subtlety: normalisation is exact over the dataset, approximate per batch — with `shuffle=True` batches are unbiased samples of the weight distribution, so per-batch means hover around 1.0. Fine.)
- Per the roadmap's R5 check: `ReduceLROnPlateau` consumes only the *validation* scalar, which remains unweighted — no scheduler interaction. Early stopping and best-checkpoint likewise.

### 3e. Interaction risks

- **With 2.1:** weighting strength is defined relative to the window (3b). If someone later changes `finetune_window_days_per_category` without touching half-life, the `'auto'` coupling keeps intent intact — this is why `'auto'` is the default rather than a frozen number.
- **With 1.5/2.3:** the Step-5 baseline run must use the same `'auto'` half-life via the two-stage procedure (§1e), or bounds won't describe production. The baseline JSON's `procedure` label should therefore include the half-life mode, e.g. `'two_stage_recency_auto'`.
- **With walk-forward folds:** inside a fold, `ref = Y_dates_train.max()` is the fold's train end — correct historical simulation, no code change needed (this is a consequence of choosing the dataset-max reference over wall-clock).

### Verification (2.2)

1. Unit: `RecencyWeightedMSELoss()(p, t, sample_weights=torch.ones(B))` equals `WeightedMSELoss()(p, t)` exactly (weights=1 ⇒ parent behaviour). And with `sample_weights=torch.tensor([2.,0.])`, only sample 0 contributes.
2. Dataset: instantiate a c1 predictor with `recency_half_life_days=182.5` on a 365-day slice; assert `dataset_train[0]` is a 3-tuple, weights are monotonically ordered by `Y_dates_train[:, -1]` (newest ≈ largest), `weights.mean() ≈ 1.0`, and `max/min ≈ 4` (2 half-lives across the window after the gap trims the newest… note: the *train* block is the chronologically older ~60% of the window, so the realised max/min within train is nearer `exp(ln2·(train_span/182.5))` — compute from actual dates, don't hardcode 4).
3. Training loop: one epoch with weighting on — no shape errors on either architecture (the Transformer's 3-tuple unpack is the likely typo site); validation epoch still runs 2-tuples.
4. Regression: `recency_half_life_days=None` ⇒ `dataset_train[0]` is a 2-tuple and a short training run is bit-comparable in structure (same code path as pre-2.2).

---

## 4. Item 2.3 — Monthly drift monitoring

### 4a. What gets logged, where, and by whom

**Logging hook — inside `predict_and_trade()`** (workflow.py L533), right after the environment/data update and before/independent of trading logic. Do not reuse `env.current_potential_estimates` for logging: `compute_predicted_potentials` (rl_environments.py L929–933) silently *skips* predictors whose rolling window can't be filled and returns an unlabeled positional array — fine for the agent, unacceptable for a log that must attribute rows to presets. Instead call `env.compute_predicted_prices()` (L936), which returns `{predictor.name: (predictor_input, predicted_price_series)}` — one extra forward pass per predictor per day, negligible.

New helper in `evaluation.py`:

```python
def log_live_predictions(pred_price_dict: dict, predictor_instances: list, tracking_file: Path) -> int:
    """Append one row per predictor to the live-prediction log. Returns number of rows written."""
```

Row schema (CSV, append-only, header auto-created; new constant `PREDICTION_TRACKING = DATA / "prediction_tracking"`, file `live_predictions.csv`):

| column | content |
|---|---|
| `prediction_timestamp` | `predictor_input.index.max()` (the reference time) |
| `predictor_name`, `preset_type`, `architecture` | from the instance (`.name`, `.preset_type`, class name) |
| `reference_price` | `predictor_input.iloc[-1]` — the ratio-space anchor |
| `predicted_price` | `predicted_series.iloc[-1]` (final forecast step, price space) |
| `predicted_ratio` | `predicted_price / reference_price` |
| `target_timestamp` | `predicted_series.index.max()` |
| `resolved`, `actual_price`, `actual_ratio`, `hit`, `sq_error_scaled` | empty at write time; filled by resolution (4b) |

**Idempotency:** `predict_and_trade` is also invocable via chatbot (`"do step environment"`); dedupe on `(prediction_timestamp, predictor_name)` — `log_live_predictions` loads the existing keys (file is small: 7 rows/day ≈ 1,800 rows/year) and skips duplicates.

In `predict_and_trade`, wrap the call in `try/except Exception` with a `chatter("Prediction logging failed: …")` — monitoring must never break trading. Deploy this hook in **Step 2** (§0b), long before the comparison logic exists.

### 4b. Resolution + comparison — new workflow function `monitor_drift()`

New function in `workflow.py` (thin) delegating to `evaluation.py` (logic), decorated like its siblings (`@timed_callback_decorator(callback=chatter)`, `@retry_decorator(on_error_callback=chatter)`):

1. **Resolve matured predictions:** for each unresolved row with `target_timestamp <= now`, look up the actual price from the *same series family the prediction was made on*: `data_manager.{b,c,d}_interp_prices` chosen by `preset_type[0]`, via `series.asof(target_timestamp)` with a staleness guard (reject if the matched index is > 1 sampling interval away — leave unresolved, retry next month). Then `actual_ratio = actual_price / reference_price`, `hit = sign(predicted_ratio − 1) == sign(actual_ratio − 1)` (matching `HitRateMetric`'s final-step semantics), `sq_error_scaled = ((predicted_ratio − actual_ratio) · 100)²` (matching §1f's `final_step_mse`). Rewrite the CSV. Consistency note: predictions and resolutions both use the interpolated series, so the ETF-vs-index scale and any interpolation bias cancel inside the ratio.
2. **Rolling 30-day live metrics per preset:** over resolved rows with `target_timestamp` in the last 30 days: `live_hit_rate = mean(hit)`, `live_mse = mean(sq_error_scaled)`, `n`. (b/c presets: ~21 resolutions/month; d presets: horizons are 2–6 *weeks*, so a 30-day resolution window still yields ~21 rows/month since predictions are made daily — but they overlap heavily and are not independent; note this in the docstring and lean on the hit-rate track for d-presets rather than over-reading `n`.)
3. **Compare against walk-forward bounds:** `baseline = load_latest_baseline(WALK_FORWARD_BASELINES)`. Per preset: `below = live_hit_rate < (hit_rate_mean − hit_rate_std)`; secondary signal `live_mse > final_step_mse_mean + final_step_mse_std` (report it, but the alert trigger is the hit-rate track — MSE across a live month is noisy and the roadmap's trigger is stated in accuracy terms).
4. **Persistence for the 2-consecutive-months rule:** append one row per preset per check to `data/prediction_tracking/drift_checks.csv`: `check_date, preset_type, n_resolved_30d, live_hit_rate, live_mse, baseline_mean, baseline_std, below_bound, consecutive_below`. `consecutive_below = previous row's consecutive_below + 1 if below else 0` (read the last check row per preset from the same file — the file *is* the state; no separate state object).
5. **Alert:** if `consecutive_below >= 2` → `chatter(f"*DRIFT ALERT* {preset}: live 30d hit-rate {…:.1%} has been below the walk-forward lower bound {…:.1%} for {n} consecutive monthly checks (n={…}). Per the operational schedule this triggers an architecture reassessment / walk-forward re-run.")`. `chatter` is the module-global `MailChatbot` (workflow.py L125); plain `chatter(msg)` is exactly how every other workflow function sends text — no new mechanism. Also send a non-alert monthly summary line per preset (hit rate vs bound) so the monitor's liveness is itself observable.

### 4c. Scheduling

`function_schedule` entry (format `[day, weekday, hour, minute]`, `check_schedule` in `processing_tools.py` L71 supports lists per slot — "first Saturday of the month" = day ∈ 1..7 AND weekday == 5):

```python
monitor_drift: [[1, 2, 3, 4, 5, 6, 7], 5, 12, 0],   # 1st Saturday, 12:00 — hours before fine_tune_predictors (Mon 17:54) and clear of Saturday jobs
```

(Verify against `verify_schedule`'s constraints when adding.) Also add `"describe drift status"` to `request_map` returning a formatted tail of `drift_checks.csv` — read-only, cheap, and the natural way the operator will actually consume this. Do NOT add `monitor_drift` itself to the request map's `do`-section (it mutates the tracking file; the schedule owns it — same reasoning as the shadowed-out `do finetuning`).

### 4d. Edge cases (all must be handled, none may raise)

- **No baseline file yet** (the explicitly-required fallback): `load_latest_baseline` returns `None` → resolve predictions, compute and log live metrics with `baseline_mean/std` empty and `below_bound=False`, and chatter one info line: `"Drift check: no walk-forward baseline found yet — logging live metrics only ({n} resolved predictions). Run the walk-forward baseline (evaluation.py) to enable drift comparison."` The 30-day metrics still accumulate history, so the first real comparison has context.
- **Baseline procedure mismatch** (§1d): if `baseline['procedure']` ≠ the current production procedure label, treat as no-baseline and say so in the message — a stale-procedure comparison is worse than none.
- **Thin data:** if `n_resolved_30d < 15` for a preset, log but skip the below-bound evaluation for it (`below_bound=None`) — early weeks after deployment, or gaps from `predict_and_trade` failures.
- **Preset absent from baseline** (e.g. baseline predates a preset-composition change): skip comparison for it, note in summary.
- **First-ever check:** no prior row in `drift_checks.csv` → `consecutive_below = 1 if below else 0`.

### Verification (2.3)

1. **Logging:** run `predict_and_trade` once (or call `log_live_predictions` with a mocked `pred_price_dict`); assert 1 row per env predictor, correct `target_timestamp = prediction_timestamp + FH·sampling_rate`, and that a second identical call writes 0 rows (dedupe).
2. **Resolution:** hand-craft a log row with `target_timestamp` in the past; run resolution; assert `hit`/`sq_error_scaled` match a manual computation from the interp series.
3. **Trigger logic table-test** (pure function — factor `evaluate_drift(live, baseline, prev_consecutive) -> (below, consecutive, alert)` so it is testable without files): below→below ⇒ alert on 2nd; below→ok ⇒ reset; no-baseline ⇒ never alerts.
4. **End-to-end dry run:** with a fabricated baseline JSON whose `hit_rate_mean − std` is above any plausible live value, run `monitor_drift` twice with `check_date` a month apart (parameterise the date for testability) and confirm the chatter alert fires exactly on the second run.

---

## 5. Item 2.4 — Transaction-cost-aware evaluation

### 5a. Scope and location

Standalone analysis code in `evaluation.py` (per the roadmap's explicit decision: independent of `RLTradingEnv` — it answers "is this signal worth anything after costs?", it does not simulate the agent/portfolio). Two public functions plus one data-derivation helper:

```python
def derive_premium_curve(portfolio_dir: Path, underlying_price_series: pd.Series,
                         leverage_bins=(1, 2, 3, 4, 5, 7, 10)) -> pd.DataFrame: ...
def evaluate_signal_after_costs(predictions: np.ndarray, targets: np.ndarray,   # ratio space, (N, FH)
                                preset_type: str,
                                leverage_categories=(1., 2., 3., 4., 5.),
                                premium_curve: pd.DataFrame = None,
                                daily_sigma: float = None,                       # trailing realised vol
                                commission_rate: float = 0.001,                  # matches RLTradingEnv
                                financing_rate_pa: float = 0.03,                 # matches update_portfolio's enforce_base_price_increase_per_annum(.03)
                                ) -> pd.DataFrame: ...
def transaction_cost_report(fold_results_per_preset: dict, **kwargs) -> pd.DataFrame:
    # convenience: runs evaluate_signal_after_costs on walk-forward fold predictions per preset
```

Input predictions/targets come from walk-forward `FoldResult.predictions_val` / `targets_val` (preferred — many OOS samples across regimes) or any predictor's `predictions_val`/`Y_val`.

### 5b. **FLAG — roadmap assumption partially wrong: there is no spread data in the CSVs**

The roadmap says "use empirical, leverage-dependent spreads from the scraped certificate CSVs". The actual files (§ Key files) contain **no bid/ask columns**. What exists, and what it means:
- `risk_premium` — the issuer premium in absolute certificate-price units: `price_series = intrinsic_value + risk_premium` (financial_products.py L745). Scraped per certificate per snapshot; 24 snapshot files from 2025-10 to 2026-03 exist. Spot check: `DE000TT2CH77` (long, base 13,536, ratio 0.01, DAX ~24,000 ⇒ price ≈ 104.6, leverage ≈ 2.3): premium 0.1111 ⇒ **0.11% of price**; `DE000TT0ZP72` (base 9,283 ⇒ leverage ≈ 1.6): premium 0.9792 ⇒ **0.66% of price**. So a leverage-dependent relative premium is derivable — and it is a real holding cost (paid on entry, decays/lost at exit or KO) — but it is **not the bid-ask spread**.
- True bid-ask would require extending the Boerse-Frankfurt scrape (`web_interaction.fetch_future_info_from_boerse_fra` — grep confirms no bid/ask handling today). **Out of scope for 2.4** — note it as a follow-up in the report's caveats, do not build it now.

**Decided cost model for 2.4** (state these as explicit, revisable assumptions in the module docstring):
- **Trading friction:** `commission_rate = 0.001` per side (round trip 0.2%), exactly as `RLTradingEnv` applies it (buy at `·(1+c)` L501, sell at `·(1−c)` L561) — the flat placeholder remains the *friction floor*.
- **Empirical leverage-dependent premium** from `derive_premium_curve` as the leverage-scaling cost component, treated as paid once per round trip (entry premium not recovered at exit; conservative).

### 5c. `derive_premium_curve` — mechanism

For each `"Scraped Certificate Set"` CSV in `data/portfolios/` (glob on the keyword, skip the `"Artificial"` set):
1. Parse the scrape timestamp from the filename (`filemgmt` convention `YYYY-MM-DD hh_mm_ss`).
2. `KOCertificateSet.load_from_csv(file, underlying_price_series=<c-series or non_etf series — must be index-scaled, i.e. the same scale as the stored base prices; use data_manager.non_etf_env_interp_prices as workflow.py does>)`.
3. Per certificate, at the scrape date `t` (use `.asof`-style nearest index): `price = product.price_series[t]`, `leverage = product.leverage_series[t]`, `rel_premium = product.risk_premium / price`. Drop knocked-out entries (`leverage == 0` per the `is_ko_series` multiplication at L673) and any `rel_premium > 0.10` (bad scrape guard).
4. Pool all (leverage, rel_premium) points across files (~24 files × ~10 certs ≈ 200+ points), bucket by `pd.cut(leverage, leverage_bins)`, and return per bucket: `n`, `median_rel_premium`, `mean`, `p90`. Median is the curve used downstream (scrapes contain outliers); expect it to *increase* with leverage since premium is roughly constant in absolute terms while price shrinks as `~U·ratio/L`.
5. Lookup helper: `premium_for_leverage(curve, L)` → the bucket median, falling back to the nearest populated bucket (the live portfolio is concentrated at low leverage; high-leverage buckets may be thin — report `n` so the operator sees this).

### 5d. Per-trade P&L simulation (move magnitude, not just hit rate)

All in ratio space (post Phase 1: `targets[:, -1]` is the realised end-of-horizon ratio, `predictions[:, -1]` the predicted one). Per sample `i` and leverage `L`:

```
u_i      = targets[i, -1] − 1                    # realised underlying return over the horizon
s_i      = sign(predictions[i, -1] − 1)          # signal direction (skip samples with s_i == 0)
gross_i  = clip(L · s_i · u_i, min=−1.0)         # leveraged return per unit stake; floor at −100%:
                                                 #   the KO structure caps the loss at the stake —
                                                 #   losses beyond −1/L·L are absorbed by knockout,
                                                 #   which the KO-risk term below prices separately
net_i    = gross_i − round_trip_cost(L) − ko_cost_i(L)
```

with `round_trip_cost(L) = 2·commission_rate + premium_for_leverage(curve, L) + financing_rate_pa · (L−1) · h/365`.

The financing term derivation (ground it in the code, cite in a comment): the issuer raises the base price by `financing_rate_pa` p.a. (`enforce_base_price_increase_per_annum(abs_increase_pa=.03)`, workflow.py L479); the certificate loses `ΔB·ratio` of value per unit time; relative to the certificate price `(U−B)·ratio`, with `B = U·(1−1/L)`, that is `0.03·(L−1)/365` per day; `h` = holding horizon in **calendar days** = the preset's forecast horizon converted via the preset table (c1: 3 business days ⇒ h≈4.2 calendar; simpler and acceptable: use business-day count directly and say so — the term is small at these horizons).

From the per-sample series, conditional on direction (evaluate long-signals and short-signals separately AND pooled — asymmetry is common and a pooled-only number can hide an untradeable short side):
- `p = mean(gross_i > 0)` — leveraged hit rate (≈ directional hit rate; leverage doesn't change sign)
- `W = mean(gross_i | gross_i > 0)`, `Lo = −mean(gross_i | gross_i ≤ 0)` — win/loss magnitudes
- `expected_net = mean(net_i)` — equivalently `p·W − (1−p)·Lo − costs`, but compute from the sample directly (exact, handles the clip)

### 5e. Knock-out risk term — volatility-scaled proxy (grounded, not a pricing model)

A KO certificate at leverage `L` is knocked out when the underlying touches the base price — a drawdown of `≈ 1/L` from entry (exactly `1/L` when premium≈0: `distance = (U−B)/U = U·ratio/(L·U·ratio) · L/L… = 1/L` since `L = U·ratio/price` and `price ≈ (U−B)·ratio`). The final-step comparison in 5d only sees end-of-horizon prices; an intraday touch during the holding period is a separate, path-dependent loss the roadmap requires as its own term. Proxy (driftless random walk + reflection principle):

```
P_KO(L, h) ≈ 2 · Φ( −(1/L) / (daily_sigma · sqrt(h)) )        # prob. of touching −1/L within h days
ko_cost_i(L) = P_KO(L, h) · (1 + s_i·u_i·L·0.5)  →  simplify to  P_KO(L, h) · 1.0
```

Use the simple form `ko_cost = P_KO · 1.0` (KO ⇒ total loss of stake; residual value is `1e-10` per `intrinsic_value_series` L720): it double-counts slightly with the end-of-horizon loss when the path both touches and ends down — acceptable conservatism, state it. `daily_sigma`: std of daily log returns over the trailing 60 trading days of the c-series (compute in the function if `daily_sigma=None`; expose so walk-forward callers can pass fold-appropriate historical vol). `Φ` via `scipy.stats.norm.cdf` if scipy is in the env, else the `math.erf` identity `Φ(x) = 0.5·(1+erf(x/√2))` — check `environment.yml` before importing scipy. Sanity anchors to assert in tests: `P_KO(1, any)` ≈ 0 (100% buffer); `P_KO(5, 42)` with σ=1%/day ⇒ `2·Φ(−0.2/0.0648)` ≈ 0.002 — small for c-presets, material only for d3's 6-week horizon at higher leverage.

### 5f. Break-even hit rate and report shape

Solve `p*·W − (1−p*)·Lo − C = 0` (with `C = round_trip_cost + ko_cost`, magnitudes from 5d):

```
p_breakeven = (Lo + C) / (W + Lo)
```

Output `DataFrame` — one row per (preset, leverage, direction ∈ {long, short, pooled}):

`preset, L, direction, n_signals, p (hit rate), W, Lo, spread_cost (=2c+premium), financing_cost, ko_prob, expected_net_per_trade, p_breakeven, edge (= p − p_breakeven), tradeable (edge > 0)`

Report destination: returned frame + a formatted `print`; this is a developer/analysis tool run alongside the quarterly walk-forward (add its invocation to `evaluation.py`'s `__main__` after the baseline run, feeding fold predictions directly). No cron, no chatbot wiring — though the quarterly operator may paste the summary into the chat manually.

**Interpretation guard (put in the docstring):** `expected_net` here is per-signal-per-horizon and assumes one round trip per signal with full stake; the live agent trades thresholds and partial positions, so this is a *signal quality* metric, not a portfolio P&L forecast. Its job is the roadmap's: "this preset needs > p_breakeven hit rate to be worth trading after costs" — compare that against the walk-forward `hit_rate_mean` and lower bound per preset, which is the single most decision-relevant table Phase 2 produces.

### Verification (2.4)

1. `derive_premium_curve` on the real `data/portfolios/` dir: returns populated buckets; hand-verify one certificate (`DE000TT2CH77` numbers above) end-to-end.
2. Synthetic P&L test: predictions all-up, targets +1% (u=0.01), L=2, zero costs ⇒ `p=1, W=0.02, expected_net=0.02, p_breakeven = C/(W)=0`. Add `C=0.005` ⇒ `expected_net=0.015`, `p_breakeven=0.005/0.02… = (0+0.005)/(0.02+0) = 0.25`. Pin these exactly.
3. Degenerate cases: all signals flat (`s_i==0` everywhere) ⇒ empty row with `n_signals=0`, no division errors; `Lo=0` (no losses) ⇒ `p_breakeven = C/(W)` finite.
4. Cross-check against the env's cost convention: for L=1, `round_trip_cost ≈ 0.002 + premium` — consistent with two `commission_rate` applications in `RLTradingEnv`.

---

## 6. Cross-cutting: the 2.1 ↔ 2.2 ↔ 1.5 ↔ 2.3 coupling, restated as invariants

These four items form one system; the following invariants keep it coherent. Violating any of them produces silently-wrong drift monitoring, which is worse than none:

1. **One training procedure, two call sites.** The weekly production fine-tune (`PredictorManager.fine_tune_predictors`) and the walk-forward two-stage procedure (§1e) must execute the same stages, windows (`FINETUNE_WINDOW_DAYS_PER_CATEGORY`), LRs, and half-life mode. Factor shared logic into one helper if the implementations start to diverge.
2. **Bounds must match the deployed procedure.** The baseline JSON carries a `procedure` label; `monitor_drift` compares it against the current production label and degrades to logging-only on mismatch (§4d). Any change to window lengths, half-life, LR, or pretrain vintage ⇒ re-run the walk-forward baseline (which, per §2f, also refreshes the pretrain bases — same trigger, same sitting).
3. **Recency weighting never touches validation.** Enforced structurally: weights exist only in `dataset_train`, and only `train_loss_criterion` accepts them. Any future refactor that passes `train_loss_criterion` to the validation `run_epoch` breaks 1.3's checkpoint semantics — the §3 verification's 2-tuple/3-tuple asymmetry is the tripwire.
4. **Base models live outside the deployment scan tree**, fold/throwaway models are never saved into it, and fine-tuned weekly models are the only new `.pt` files `pred_manager` ever sees (§2b, §1c step 5).
5. **Everything the monitor compares is final-step, scaled-ratio-space:** live `sq_error_scaled`, walk-forward `final_step_mse`, and hit rates all use the `(ratio−1)·100` / last-step conventions established in Phase 1 (§1f, §4b). No quantity in the monitoring path may be multi-step-weighted or price-space.

## 7. Out-of-scope / follow-ups deliberately not done here

- Bid/ask scraping from Boerse Frankfurt (would upgrade §5b's premium proxy to true spreads) — extend `web_interaction` later.
- Nested (train/early-stop/test) walk-forward splits (§1c caveat) — revisit if drift alerts over-trigger.
- Parallelised fold execution (§1g) — structure is ready, pool deliberately not implemented.
- Half-life tuning via walk-forward (roadmap: "window length is a hyperparameter to tune") — the harness + `training_procedure` hook make this a config sweep when someone wants it; not part of this batch.
- Transformer presets remain excluded from fine-tuning/monitoring defaults (workflow uses `architecture='LSTM'` throughout) — unchanged; the R3 Transformer audit still gates Phase 4.
