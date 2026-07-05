# Research & Implementation Roadmap

Compiled from pipeline audit (June/July 2026). Items are grouped by phase; each phase builds on the previous.

---

## Phase 1 — Fix Evaluation & Training Foundations

These must land before any architectural experiments, since current metrics are unreliable.

### 1.1 Per-window ratio normalisation
- **What:** Replace global `Normaliser` (fit on full series) with per-window normalisation: divide each input window by its last value so the model works in ratio space (~1.0 centered).
- **Why:** Eliminates data leakage through normalisation, resolves the artificial ceiling under temporal split, and makes cross-source data (ETF vs index) compatible without a conversion factor.
- **Where:** `NNPredictor.prepare_data()`, `NNPredictor.predict()`, `Normaliser` class.
- **Considerations:** Targets (Y) must also be expressed as ratios relative to the same reference point (last X value). Inverse transform at prediction time multiplies by the known current price.
- **R2 findings (resolved — code read of `pt_metrics.py`):**
  - `HitRateMetric` works unchanged in ratio space. It only compares `sign(prediction - feature[-1])` vs `sign(target - feature[-1])`; dividing by a positive per-window constant preserves sign, and `feature[:,-1]` becomes exactly `1.0` (no float precision issue — `x/x == 1.0` exactly in IEEE-754). No fix needed.
  - `WeightedMSELoss` is mathematically still valid but numerically fragile: ratio targets cluster tightly around 1.000 (e.g. 0.995–1.010), so MSE collapses to ~1e-5–1e-6. Early-stopping thresholds and the LR plateau scheduler would then operate near float noise, and a model that trivially predicts "no change" (ratio=1.0) looks deceptively good.
  - **Fix (part of 1.1, not deferred):** scale targets as `(ratio - 1) * 100` (percent deviation from no-change) to keep loss magnitude in a healthy numeric range.
  - **Critical implementation detail (Fable-flagged):** this scaling must happen **inside `WeightedMSELoss`**, not in `prepare_data()`/dataset preparation. If the dataset targets themselves get scaled, model outputs live in scaled space and `HitRateMetric`'s sign comparison (which relies on `feature[:,-1] == 1.0` in unscaled ratio space) silently breaks. Keep ratio-space (unscaled) as the canonical representation for data/predictions; scale only at the loss computation boundary.
  - **Add as diagnostic:** compute a naive "always predict ratio=1.0" baseline loss/hit-rate alongside the trained model's metrics, so a model that hasn't learned anything beyond triviality is visible immediately.

### 1.2 Temporal train/val split with gap
- **What:** Replace random shuffle split with chronological split. Insert a gap of `rolling_window_size + forecast_horizon` steps between train and val to prevent any window overlap.
- **Why:** Random splits leak ~97% of each validation window into training via overlapping rolling windows. Current validation hit rates are unreliable.
- **Where:** `create_train_validation_split()` in `preprocessing.py`. Remove `randomise_validation_data_every` from `NNPredictor`.
- **Considerations:** Walk-forward validation (see 1.4) supersedes this for evaluation, but a simple temporal split is still needed for early stopping during individual training runs.

### 1.3 Best-model checkpoint at early stopping
- **What:** Save model state_dict when `best_loss` improves. Restore best weights when early stopping triggers (or training ends).
- **Why:** Currently keeps weights from `patience` epochs past the optimum.
- **Where:** `NNPredictor.run_training()`.

### 1.4 Batch-averaged epoch loss
- **What:** Divide `epoch_loss` by the number of batches so the scalar is independent of batch size.
- **Why:** Early stopping patience and LR scheduler plateau detection currently change behavior when batch size changes.
- **Where:** `LSTMModel.run_epoch()`, `TransformerModel.run_epoch()`.

### 1.5 Walk-forward validation harness
- **What:** Implement an expanding-window walk-forward evaluation loop. Train on `[0...T]`, gap, validate on `[T+gap...T+window]`, expand T, repeat. Aggregate val metrics across folds.
- **Why:** Gold standard for financial time-series evaluation. Tells us the true expected out-of-sample performance.
- **Where:** New function, likely in `predictors.py` or a new `evaluation.py`.
- **Usage:** Run infrequently (when architecture/features change). Not part of weekly fine-tuning.
- **Considerations:** Computationally expensive (multiple full training runs). Design to be parallelisable.

### 1.6 Gradient clipping
- **What:** Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` after `loss.backward()`.
- **Why:** LSTMs and Transformers on financial data can hit gradient spikes. Cheap safety net.
- **Where:** `LSTMModel.run_epoch()`, `TransformerModel.run_epoch()`.

---

## Phase 2 — Training Strategy Improvements

### 2.1 Two-stage training (pretrain + fine-tune)
- **What:** Pretrain on full history (done once, or infrequently). Fine-tune on a recent window (e.g., last 12–18 months) with a reduced learning rate.
- **Why:** Full-history training lets the model learn general market dynamics (including crashes/crises). Recent-window fine-tuning adapts to the current regime without being dominated by old-regime gradients.
- **Where:** `PredictorManager.fine_tune_predictors()` and the Saturday workflow.
- **Considerations:**
  - Pretrained base models could be stored separately and reused across fine-tuning runs.
  - Fine-tuning LR should be ~10x lower than pretraining LR.
  - The recent window length is a hyperparameter to tune via walk-forward.
- **D2 corrected (was: flat 12 months — superseded by Phase 2 Fable elaboration):** a flat 12-month fine-tuning window is mathematically impossible for d-presets. All presets produce 1 sample/day (b/c) or 1/week (d) due to `daily_prediction_hour=16`; d3 needs 54 weekly rows to form a single window, so 52 weeks (12 months) yields **zero** training samples. **Corrected to per-category windows:** 365 days for b/c-presets, 1825 days (5 years) for d-presets, sharing one constant with a `<50 train samples` guard (error, not silent failure) if a preset still ends up too thin. See `PHASE2_IMPLEMENTATION_AGENDA.md` §2.1 for full detail.

### 2.2 Temporal loss weighting
- **What:** Weight loss by recency: `weight = exp(-lambda * days_ago)` so recent observations matter more.
- **Why:** Complements two-stage training. Even within the fine-tuning window, more recent data should dominate.
- **Where:** Custom loss function wrapping `WeightedMSELoss`.
- **Considerations:** Lambda controls the half-life. A half-life of ~6 months seems reasonable as starting point.
- **R5 findings (resolved — web research):**
  - Standard formula: `weight = exp(-lambda * days_ago)`, with `lambda = ln(2) / half_life` so the half-life parameter is directly interpretable (weight at exactly one half-life = 0.5). Confirms the planned approach.
  - A power-exponential variant `exp(-lambda * days_ago^p)` exists for accelerating/decelerating decay, and a comparative study found linear decay performs strongly, close behind exponential. Not worth the extra complexity here — stick with plain exponential decay as planned.
  - **Critical scoping decision:** recency weighting must apply to the **training loss only**, not the validation loss. Validation/early-stopping loss should stay on the unweighted `WeightedMSELoss` fixed in Phase 1. Reasoning: `ReduceLROnPlateau` and the Phase 1.3 best-checkpoint logic both compare validation loss across epochs — if validation itself were recency-weighted, "best_loss" would no longer be an honest, stable comparison (checked confirms `ReduceLROnPlateau` only consumes the final validation scalar and has no visibility into how it was computed internally, so there's no scheduler-side conflict — the risk is purely on the interpretability/consistency of "best" model selection).

### 2.4 Transaction-cost-aware evaluation
- **What:** Translate predictor hit-rate / directional accuracy into expected P&L after realistic knock-out certificate costs (spread, financing/overnight cost, commission).
- **Why:** A sub-0.5% hourly directional edge can be fully consumed by certificate spread and financing costs. Hit rate alone doesn't tell us whether a signal is tradeable — this must be evaluated early, not bundled into the deferred RL/reward-shaping bucket.
- **Where:** `financial_products.py` already models `KOCertificate` leverage/intrinsic value; `RLTradingEnv` already has a `commission_rate` parameter (currently 0.001, described as "reflects a typical spread"). Build a standalone cost-aware evaluation function that takes predictor output + a representative certificate cost structure and computes expected net return per signal, independent of the RL environment.
- **Considerations:** This is an evaluation tool, not a trading system change — it answers "is this signal worth anything after costs?" before we invest further in improving the signal itself.
- **Fable-flagged gaps to close:**
  - **Move magnitude, not just hit rate:** expected P&L needs `avg_win × p − avg_loss × (1−p) − costs`, not hit rate alone. A 55% hit rate on small wins and large losses can still be net-negative. The evaluation must track win/loss magnitude conditional on hit/miss, not just direction.
  - **Knock-out risk is a separate cost from spread/financing:** a KO certificate can be knocked out intraday even if the predicted direction is eventually correct. This needs its own term, not folded into spread.
  - **Use empirical, leverage-dependent spreads** from the scraped certificate CSVs (`data/portfolios/`) instead of the flat `commission_rate=0.001` placeholder — real spreads widen with leverage.
  - **Output a break-even hit rate per horizon/preset** — directly actionable: "this preset needs >X% hit rate to be worth trading after costs."

### 2.3 Monthly drift monitoring
- **What:** Track rolling 30-day live prediction accuracy (hit rate, MSE) and compare against walk-forward expected performance. Alert (via chatbot) if live accuracy drops below walk-forward lower bound for 2+ consecutive months.
- **Why:** Detects regime shifts or model degradation without re-running full walk-forward.
- **Where:** Integrate into `predict_and_trade()` workflow; store results in a tracking file.
- **Dependency decision:** chose to resolve R4/D5 (walk-forward fold size/gap parameters) now rather than scoping 2.3 down to a simpler baseline, so 2.3 works as originally envisioned. **On hold** — pausing this research until the Phase 1 Opus implementation completes (sequencing: finish and review Phase 1 before spending further Fable budget on Phase 2 planning).

---

## Phase 3 — Data Pipeline Enhancements

### 3.1 Real intraday data via yfinance (+ Twelve Data fallback)
- **What:** Download 60-day history at 60min intervals from yfinance (`^GDAXI`) for the b-presets (hourly models). Keep daily Alpha Vantage data for c/d-presets.
- **Why:** Current 15min/60min data is synthetically interpolated from daily points. Real intraday data captures actual volatility, gaps, and microstructure.
- **Where:** New download method in `StockPriceDataManager` or standalone function. Wire into `update()`.
- **Research findings (R1, completed):**

  | Provider | Free tier | Intraday intervals | History depth | DAX coverage | Verdict |
  |---|---|---|---|---|---|
  | yfinance | Unlimited but unofficial, rate-limited (observed live) | 1min (7d), 5–60min (60d) | Rolling window only | `^GDAXI` direct | **Primary** — real data, free, already integrated |
  | Twelve Data | 800 req/day, 8/min | 1min+ | Months to ~2yr (plan-dependent) | `GDAXI` via `time_series` endpoint, confirmed | **Fallback/secondary** — generous quota, good redundancy |
  | EODHD | 20 calls/day, intraday = 5 calls/req → ~4 intraday req/day | 1min, 5min | 1yr (paid) | `GDAXI.INDX` confirmed | Rejected — too thin for daily automation |
  | Finnhub | 60 req/min (generous) | Mostly US equities on free tier | Unclear for DAX | Uncertain | Rejected — weak on European indices |
  | FDAX futures (Eurex/Databento/dxFeed) | No free real-time feed; 10–15min delayed via Barchart | — | — | Could cover extended hours | Needs paid feed — deferred, see 3.3 |

- **Decision:** yfinance as primary source, Twelve Data as fallback/secondary for redundancy. EODHD and Finnhub ruled out for this use case.
- **Alpha Vantage is NOT being replaced.** It remains the source for daily data (c/d-presets) — this works well and isn't in question. The R1 research only addresses the intraday gap (b-presets), since Alpha Vantage's `get_intraday()` is premium-only (established earlier). Final picture: **Alpha Vantage (daily, c/d) + yfinance (intraday, b, primary) + Twelve Data (intraday, fallback)**.
- **Considerations:**
  - yfinance rate limits (already observed). Implement retry with backoff.
  - Per-window ratio normalisation makes ETF-vs-index price levels irrelevant for training, but intraday data covers XETRA hours only (9:00–17:30 CET).
  - **Critical:** yfinance only exposes a rolling window (60-day for hourly, 7-day for 1min) — build a persistent archive that appends each day's new data, rather than relying on the live window (data is lost otherwise, same accumulation pattern as the existing Alpha Vantage daily downloads).
  - Twelve Data integration can be deferred until yfinance reliability becomes a real bottleneck (not urgent for initial implementation).

### 3.2 Multi-feature input
- **What:** Expand model input from 1 feature (price) to multiple: price, high-low range (volatility proxy), volume, and optionally a few moving average ratios (MA5/MA20/MA50 relative to current price).
- **Why:** Volume and volatility carry information about trend strength and exhaustion that price alone can't provide.
- **Where:** `TimeSeriesDataset`, model `input_size` parameter, data preparation in `NNPredictor`.
- **Considerations:**
  - Each feature needs its own normalisation (per-window ratios for prices, separate scaling for volume).
  - Start with price + high-low range + volume (3 features) before adding MAs.
  - Only the Transformer presets should use multi-feature initially; keep LSTM presets single-feature as a baseline.

### 3.3 Trading hours alignment
- **What:** Explicitly define and handle the different trading hour windows across data sources and deployment.
- **Sources & hours (CET):**
  - Wikifolio (deployment): 8:00–22:00
  - DAX XETRA (yfinance `^GDAXI`): 9:00–17:30
  - Alpha Vantage ETF: US hours, ~15:30–22:00
  - Current interpolation setting: `manual_operating_h_tuple=(8, 22)`
- **Decision needed:** Either restrict model predictions and trading to XETRA hours (safest, avoids thin-market artifacts), or source extended-hours data (e.g., futures) to cover 8:00–22:00.
- **Research findings (from R1):** FDAX futures (Eurex) would cover extended hours, but no free real-time feed exists — Databento/dxFeed require payment, Barchart is 10–15min delayed. Free extended-hours coverage is not currently achievable.
- **Leaning:** Given no free extended-hours source exists, D1 defaults to **XETRA-hours-only (9:00–17:30 CET)** for now, unless there's appetite to pay for a futures feed later.
- **Impact:** Affects interpolation, the `step_timestamp_list` in `RLTradingEnv`, and the `predict_and_trade()` schedule.

---

## Phase 4 — Architecture Evolution

### 4.1 Multi-feature Transformer presets
- **What:** Add Transformer-based presets (e.g., `b1_tf`, `c1_tf`) that accept multi-feature input, alongside existing LSTM single-feature presets.
- **Why:** Attention handles heterogeneous features (price + volume + volatility) better than LSTMs. Cross-feature temporal patterns ("volume spike 5 steps ago + price stall now → reversal") are learnable.
- **Where:** `TransformerPredictor` already exists. Needs multi-feature `input_size` wiring.
- **Considerations:**
  - Run both LSTM (single-feature) and Transformer (multi-feature) in the ensemble. Walk-forward will show which contributes.
  - Transformer presets need their own preset definitions in the preset_type_dict.

### 4.2 Hierarchical prediction: regime + direction
- **What:** Two-layer model architecture:
  - **Layer 1 (Context):** Regime classifier → {trending-up, trending-down, range-bound, high-volatility}
  - **Layer 2 (Signal):** Directional models (LSTM/Transformer) → price prediction conditioned on regime
- **Why:** Separates "what kind of market is this?" from "where is price going?" Each sub-model has a cleaner target, is independently validatable, and is more interpretable.
- **Open questions:**
  - **Labeling:** How to define regimes? Options: HMM-based (unsupervised), volatility-threshold-based (rules), or cluster-based (returns distribution).
  - **Conditioning:** Does the directional model receive regime as an input feature, or do we train separate directional models per regime?
  - **Avoiding ambiguity:** Keep to 2 layers (regime + direction) initially. Finer splits (trend duration, trend strength, reversal timing) risk compounding label noise and should only be added once the 2-layer system is validated.

### 4.3 CNN feature extractor (research item)
- **What:** 1D CNN processing raw OHLC windows into a feature vector, feeding into the ensemble alongside LSTM/Transformer outputs.
- **Why:** Translation-invariant pattern recognition (chart patterns: head-and-shoulders, flags, double bottoms). Complementary to sequential models.
- **Approach:** Train end-to-end as a feature extractor, not as a standalone predictor. Avoids the need for pattern labeling.
- **Priority:** Lower than 4.1 and 4.2. Worth prototyping once the multi-feature Transformer is working.

### 4.4 Fundamental vs cyclical price distinction (long-term research)
- **What:** Overlay that distinguishes sustainable price moves (productivity gains, policy shifts) from cyclical mean-reversion.
- **Why:** Determines whether a price level is a new equilibrium or a temporary extreme.
- **Data requirements:** Macro features beyond price data — ECB rates, Bund yields, PMI/IFO indices, earnings growth, EUR/USD. This is a fundamentally different data pipeline.
- **Priority:** Later-stage enhancement. The price-based system should be validated first. Frame as a separate sub-project.

---

## Deferred Buckets (out of scope for now)

### RL Agent & Reward Shaping
Deferred until predictor pipeline is validated (Phases 1–4). Notes for future reference:
- Reward normalisation (percentage return or Sharpe-ratio-based)
- Adaptive volatility-scaled thresholds (rolling ATR)
- Only train RL on validated predictor signals
- Consider continuous action space vs discrete leverage categories

### Fundamental vs Cyclical Price Distinction
Requires macro data pipeline (ECB rates, PMI/IFO, earnings, EUR/USD). Separate sub-project.

---

## Decisions Made (post Fable review)

- **Declined: frozen before/after baseline.** Fable suggested recording current val/hit-rate metrics on a fixed temporal holdout before touching anything, so we could later prove the Phase 1 fixes helped. Declined — the fixes (normalisation leakage, overlapping-window shuffle, missing checkpoint) are considered necessary regardless of measured before/after impact; no need to justify them with a comparison.
- **Confirmed sequencing:** for any deferred/paused research item (R2, R3, R4/1.5), the rule is "clarify before coding it in" — not "code around it and revisit later." This resolves Fable's 2.3↔1.5 dependency concern: 1.5/R4 will be resolved before 2.3 is implemented, not skipped past.
  - **Scope (Fable-flagged caveat):** this rule applies only when the deferred item is a genuine dependency of the work about to be coded — not every open question in the roadmap. Clarification steps should be timeboxed; otherwise every tangential question becomes a gate and a solo project stalls.
- **Transaction costs added** as 2.4, not bundled with the deferred RL/reward-shaping bucket — evaluating whether a signal survives real trading costs belongs early.
- **D3 decided: drop the a-presets (15min) now.** No real intraday source existed for them (synthetically interpolated from daily data), so there's no defensible reason to keep training on invented data. Remove `a1` from `preset_type_dict` and any a-preset handling in `PredictorManager`/`workflow.py`. Revisit only if/when a genuine 15min real data source appears.
- **1.5 (Walk-forward harness) excluded from the immediate Phase 1 implementation batch.** Its prerequisites (R4 literature review, D5 fold/gap parameters) are paused. Phase 1 implementation scope is now: **1.1, 1.2, 1.3, 1.4, 1.6** (normalisation, temporal split, checkpointing, batch-averaged loss, gradient clipping). Walk-forward (1.5) will be picked up in a later pass once R4/D5 are resolved, per the "clarify before coding it in" rule.

## Open Items & Known Issues

- **Transformer training instability:** Transformer scripts have not always worked reliably. Audit `TransformerModel` and `TransformerPredictor` for training issues (NaN losses, convergence failures, teacher forcing behavior) before relying on them for multi-feature presets in Phase 4.
- **Trading hours mismatch:** See 3.3. Decision pending.
- **yfinance rate limiting:** Observed during data update. Needs retry/backoff implementation before relying on it for automated intraday downloads.

---

## Proposed Operational Schedule

### Weekly (Saturdays — existing slot)
- Fine-tune best predictors on recent data window (two-stage: use pretrained base, fine-tune on last 12–18 months)
- Runs automatically via `fine_tune_predictors()` in the cron schedule

### Monthly (1st Saturday of month)
- **Drift check:** Compare rolling 30-day live hit rate against walk-forward expected bounds
- If within bounds: no action, continue weekly fine-tuning
- If below bounds for 2+ consecutive months: trigger architecture reassessment
- Log results to a tracking file; alert via chatbot

### Quarterly (or event-driven)
- **Backtest current ensemble** via walk-forward on latest data to refresh expected performance bounds
- Review whether predictor preset composition (which b/c/d types are included) should change
- Low cost: reuses existing `back_test_predictors()` infrastructure

### On architecture change (event-driven only)
- **Full walk-forward validation** of the new architecture/feature set
- Compare against current walk-forward baseline
- Only adopt if statistically significant improvement across folds
- Triggers: drift alert sustained 3+ months, new data source available, new model type implemented

---

## Decision Issues vs Research Issues

### Decisions (require your input before implementation)

| # | Question | Options | Impact |
|---|----------|---------|--------|
| D1 | Trading hours: restrict to XETRA (9–17:30) or try to cover full Wikifolio window (8–22)? | (a) XETRA only — safest, real data only (b) Full window — needs futures/extended-hours data source | Affects data pipeline, interpolation, prediction schedule, and deployment |
| D2 | Fine-tuning window length for two-stage training? | 6 / 12 / 18 months — tune via walk-forward, but need a starting point | Shorter = more adaptive but less robust to rare events |
| D3 | When to drop the a-presets (15min)? | Now (synthetic data, no real source) vs after real intraday data is available | a-presets currently train on interpolated data; questionable value |
| D4 | Transformer audit: fix existing implementation or rewrite? | Fix (lower effort, keep existing trained models) vs rewrite (cleaner, but loses compatibility) | Blocks Phase 4 |
| D5 | Walk-forward fold size and gap for initial baseline run? | e.g., 6-month val windows, 2-month gap — or longer? | Determines computational cost and statistical power of evaluation |

### Research (I can look up / prototype independently)

| # | Question | What I'd do |
|---|----------|-------------|
| R1 | What intraday data sources exist for DAX beyond yfinance? (Twelve Data, Finnhub, EODHD, etc.) | Web search for free/cheap DAX intraday APIs, compare coverage & hours |
| R2 | How does per-window ratio normalisation interact with the existing `WeightedMSELoss` and `HitRateMetric`? | Read `pt_metrics.py`, verify both metrics still make sense in ratio space |
| R3 | What's the current state of the Transformer training code — what fails and why? | Audit `TransformerModel.forward()`, run a small training loop, identify issues |
| R4 | Optimal walk-forward parameters for financial time-series (fold size, gap, expanding vs sliding window) — literature review | Web search for best practices, especially for daily/hourly equity data |
| R5 | Exponential decay loss weighting — existing PyTorch implementations and interaction with LR schedulers | Search for reference implementations, check for scheduler conflicts |
| R6 | DAX futures data sources for extended-hours coverage (8:00–22:00 CET) | Web search — relevant only if D1 resolves to "full window" |

---

## Research Agenda (next session)

**Status (as of this discussion):** R2 (metrics check), R3 (Transformer audit), and R4/1.5 (walk-forward) are **deferred/paused** — not needed to get started, and each opens its own can of worms. Revisit R3 before Phase 4 (Transformer presets) actually begins; revisit R4/1.5 before running the first walk-forward evaluation.

**Active now:**
1. **R1** — Survey DAX intraday data sources beyond yfinance — cost, coverage, rate limits, trading-hours alignment (feeds D1). This is the immediate foundation: without a reliable intraday source, Phase 3.1 can't proceed. (web search)

**Paused for later:**
2. ~~R2~~ — Verify `WeightedMSELoss`/`HitRateMetric` work in ratio space. Needed before 1.1 ships, not before.
3. ~~R3~~ — Transformer training audit. Needed before Phase 4.1, not before.
4. ~~R4~~ — Walk-forward parameter literature scan. Needed before 1.5 is implemented, not before.

After R1, the plan is to have a second model (Fable) review the resulting strategy/roadmap before implementation starts.

---

## Implementation Order

```
Phase 1 (Foundations)     <- current priority, immediate implementation batch
  1.1  Per-window ratio normalisation (incl. R2 loss-scaling fix)
  1.2  Temporal split with gap
  1.3  Best-model checkpoint
  1.4  Batch-averaged loss
  1.6  Gradient clipping
  --   Drop a-presets (D3, decided) as part of this pass
  --   1.5 Walk-forward harness EXCLUDED from this batch — paused pending R4/D5, revisit later

Phase 2 (Training strategy)
  2.1  Two-stage training
  2.2  Temporal loss weighting
  2.3  Drift monitoring

Phase 3 (Data pipeline)
  3.1  Real intraday data
  3.2  Multi-feature input
  3.3  Trading hours alignment

Phase 4 (Architecture)
  4.1  Multi-feature Transformer presets
  4.2  Hierarchical regime + direction
  4.3  CNN feature extractor

Deferred: RL & reward shaping, fundamental/cyclical overlay
```
