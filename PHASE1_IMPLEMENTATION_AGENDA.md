# Phase 1 Implementation Agenda — hybrid-rl-index-trading

Elaborated by Fable (planning/teaching pass) from `RESEARCH_ROADMAP.md` Phase 1, for direct handoff to a coding agent.

**Scope:** 1.1 (per-window ratio normalisation + loss scaling), 1.2 (temporal split with gap), 1.3 (best-model checkpoint), 1.4 (batch-averaged loss), 1.6 (gradient clipping), and dropping the a-presets. **1.5 (walk-forward) is explicitly excluded — do not implement it.**

All paths are relative to repo root.

Key files:
- `src/pipeline/predictors.py` — `TimeSeriesDataset` (L26), `TransformerModel` (L59, `run_epoch` L279), `LSTMModel` (L383, `run_epoch` L551), `NNPredictor` (L636: `preset_type_dict` L678, `normalised_price_series` L807, `prepare_data` L1061, `split_data` L1083, plotting L1101–1158, `run_training` L1170, `predict` L1266), `LSTMPredictor` (L1320), `TransformerPredictor` (L1538), `PredictorManager` (L1760: `preset_type_dict` L1781, `instantiate_predictor` L1983, `fine_tune_predictors` L2038)
- `src/pipeline/preprocessing.py` — `Normaliser` (L17), `create_rolling_window_view` (L528), `create_train_validation_split` (L600)
- `src/pipeline/pt_metrics.py` — `HitRateMetric` (L8), `WeightedMSELoss` (L27)
- `src/workflow.py` — `fine_tune_predictors` (L357), env wiring (~L590–615), toggle functions (L610–646)

---

## 0. Recommended implementation order and why

The roadmap lists 1.1 first, but the correct **code** order differs because of two dependencies:

1. **1.3 (best checkpoint) depends on 1.2 (temporal split).** Today `run_training()` re-randomises the train/val split every `randomise_validation_data_every` epochs (predictors.py L1216–1218). While that behaviour exists, "best validation loss so far" compares losses computed on *different validation sets* — a checkpoint keyed on it is meaningless. The split must become fixed (1.2) before 1.3 lands.
2. **1.3 also depends on 1.4 and 1.1's loss scaling**, because `best_loss` semantics should be settled (batch-averaged, scaled) before you start persisting "best" states and embedding loss values in saved filenames.

Order:

| Step | Item | Reason |
|---|---|---|
| 1 | Drop a-presets | Fully isolated; shrinks the surface for everything after |
| 2 | 1.4 batch-averaged loss + 1.6 gradient clipping | Both live in the same two `run_epoch` methods; 5 lines total; settle loss-scalar semantics early |
| 3 | 1.1 per-window normalisation + `WeightedMSELoss` scaling | The big change; touches `prepare_data`, `split_data` consumers, `predict`, plotting, loss |
| 4 | 1.2 temporal split with gap + remove `randomise_validation_data_every` | Builds on 1.1's new data flow; fixes the val set |
| 5 | 1.3 best-model checkpoint | Only meaningful once 2–4 are in |
| 6 | Model bootstrap (fresh retrain) + old-model archival | See §7 — old `.pt` files are invalid under the new scheme |

Commit after each step; each leaves the codebase in a runnable state.

---

## 1. Drop the a-presets (D3, decided)

### Locations and changes

1. **`NNPredictor.__init__`** (predictors.py L678–694): delete the `'a1': (15, 13, False, 20, 12)` entry from `preset_type_dict`. Also update the `Literal['a1', 'b1', ...]` type hint on the `preset_type` parameter (L638) and on the `preset_type` property (L817).
2. **`LSTMPredictor.__init__`** (L1324) and **`TransformerPredictor.__init__`** (L1542): same `Literal` hint update.
3. **`PredictorManager.__init__`** (L1781–1790): delete the `'a1'` entry from `preset_type_dict`. `describe_preset_types()` and `_infer_preset_type()` iterate over the dict, so they need no code change — removal of the entry is enough.
4. **`PredictorManager.instantiate_predictor`** (L2006–2017): remove the `if preset_category == 'a': price_series = self.data_manager.a_interp_prices` branch and update the error message at L2017 to `'b', 'c' or 'd'`.
5. **`workflow.py`**: no `a1` references found (grepped). `fine_tune_predictors` (L357) and `back_test_predictors` (L394) already default to `("b1", "b2", "c1", "c2", "d1", "d2", "d3")`, and the toggle functions only cover b/c/d. Nothing to change in code.
6. **Runtime config (not in git):** `private/env_configuration.txt` holds `predictors_to_include`. If it contains `a1`, `update_env_predictors()` would crash after this change. **Flag to the user as a manual check** — cannot be verified/edited by a coding agent without access to `private/`.

### Do NOT touch

- **`StockPriceDataManager`** (preprocessing.py): `a_interp_prices` (L239), `a_sampling_rate_str` (L203) and the 15-min entry in `update_interpolated_data()` (L343) **must stay**. The trap: `minutes_to_str_dict = {15: self.a_sampling_rate_str, ...}` (L198) means the *RL environment* itself can run at 15-min sampling (`env_sampling_rate_minutes: Literal[15, 60, 1440]`) and `env_interp_prices` resolves through the same 15-min interpolated file. Removing the 15-min interpolation to "clean up" the a-preset would silently break the environment. Only the predictor-level references go.
- Old a1 model files in `data/saved_models/`: after the dict entry is removed, `_infer_preset_type` returns `None` for them and `get_predictors_by_type_sorted(preset_type='a1')` returns nothing — they become inert. No file deletion needed (they get archived anyway in §7).

### Verification

`python -c "from src.pipeline.predictors import PredictorManager; m = PredictorManager(); assert 'a1' not in m.preset_type_dict; print(m.describe_preset_types())"` — confirm no a1 in output. Then instantiate `LSTMPredictor(preset_type='b1', price_series=<any series>)` to confirm nothing else referenced the removed entry.

---

## 2. Item 1.4 — Batch-averaged epoch loss

### Locations

- `LSTMModel.run_epoch` (predictors.py L551–600)
- `TransformerModel.run_epoch` (predictors.py L279–326)

### Mechanism

Both methods accumulate `epoch_loss += loss.detach().item()` per batch and return the raw sum. Change the return to divide by the number of batches:

```python
n_batches = len(dataloader)
...
return epoch_loss / max(n_batches, 1), lr
```

Also fix the misleading comment at L595 (`# without / batchsize because loss is already averaged`) — the per-batch loss *is* element-averaged, but summing across batches still scales linearly with dataset-size/batch-size, which is exactly the bug: early-stopping patience and `ReduceLROnPlateau` behaved differently when `batch_size` changed.

Note: mean-of-batch-means weights a smaller final batch slightly higher than exact per-sample mean. That's acceptable here; do not bother with sample-weighted accumulation.

### Interactions

- Docstrings of both `run_epoch` methods say "Sum of batch losses across the epoch" — update to "Batch-averaged loss".
- `run_training` uses `loss_val` for the plateau scheduler, early stopping, the progress bar, `self._loss_train/_loss_val`, and the **saved filename** (`TrainL{loss_train} ValL{loss_val}`, L1251). All of these just get better-behaved numbers; no code change needed there. But it means loss values in new filenames are not comparable to old filenames — fine, since old models are being retired anyway (§7).
- `PredictorManager` only parses `TrainHR`/`ValHR`/`SR/RW/FH` from filenames (L2106–2134), not `TrainL`/`ValL` — so the filename change is safe for model discovery.

### Verification

On a small run, train the same predictor twice with `batch_size=32` and `batch_size=64` for 1 epoch on identical data: reported epoch losses should now be the same order of magnitude (previously the 32-batch run reported ~2x the loss). Simplest: instantiate one predictor, call `nn_model.run_epoch(dataloader, ..., is_training=False)` with two dataloaders of different batch size over the same dataset and compare.

---

## 3. Item 1.6 — Gradient clipping

### Locations

Same two `run_epoch` methods as 1.4 — do them in the same commit.

### Mechanism

In the `is_training` branch, order is: `zero_grad()` (already at loop top) → forward → loss → `loss.backward()` → **clip** → `optimiser.step()`:

```python
if is_training:
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
    optimiser.step()
```

Clipping must sit strictly between `backward()` (gradients exist) and `step()` (gradients consumed). `self.parameters()` is correct in both classes since `run_epoch` is a method of the `nn.Module`.

### Interaction risk with 1.1's loss scaling (important)

The `(ratio − 1) * 100` scaling inside `WeightedMSELoss` (§4) multiplies the loss — and therefore all gradients — by 10⁴ relative to unscaled ratio space. Adam is largely scale-invariant, but **a fixed `max_norm=1.0` is not**. If clipping fires on virtually every step after the loss scaling lands, it stops being a spike safety-net and becomes a de-facto learning-rate cap. Mitigation: after both changes are in, log `total_norm` (the return value of `clip_grad_norm_`) for a few epochs on a real preset (c1 is cheapest: ~15-window, 3-step horizon). If the pre-clip norm exceeds 1.0 on >~20% of batches, raise `max_norm` (5.0 or 10.0) so clipping only catches genuine spikes. Make `max_norm` a module-level constant or a `run_epoch` keyword arg (`clip_grad_max_norm: float = 1.0`) so it's tunable without editing the loop.

### Verification

Temporarily print/collect `total_norm` from `clip_grad_norm_` for one training run and confirm (a) it is finite, (b) it is not saturated at `max_norm` every step. Then remove the print.

---

## 4. Item 1.1 — Per-window ratio normalisation (+ loss scaling)

This is the core change. Canonical representation after this change: **X and Y live in unscaled ratio space** (window divided by its own last X value, so `X[:, -1] == 1.0` exactly), model outputs are ratios, and the `(ratio − 1) * 100` scaling exists **only inside `WeightedMSELoss`** — never in the data.

### 4a. What to do with the `Normaliser` class — recommendation

**Leave `preprocessing.Normaliser` in place, untouched, and stop using it in `NNPredictor`. Do not implement per-window normalisation as a class.**

Reasoning:
- `Normaliser` is used **only** inside `predictors.py` (constructor at L722, `normalised_price_series` at L807–810, plotting L1104/1106/1143–1146, `predict` L1279/L1290). Nothing else imports it, so removing its usage from `NNPredictor` fully retires it without breaking anything.
- Per-window ratio normalisation is stateless — there is no fitted `mu`/`sd` to carry between fit and inference; the reference value is *per sample* and known at the call site (last window value / current price). Wrapping that in a stateful transformer class would recreate the exact "implicit global state" shape that caused the leakage bug. Two module-level functions in `preprocessing.py` are the right altitude:

```python
def normalise_windows(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divide each (X, Y) row pair by the last value of its X window.
    Returns (X_ratio, Y_ratio, reference_prices) with reference_prices shape (n_samples, 1)."""
    ref = X[:, -1:].copy()          # copy is essential: X is a stride-tricks *view* into the price array
    return X / ref, Y / ref, ref
```

  (Inverse is just `values * ref`; no function needed, but you may add `denormalise_windows` for symmetry.)
- Do **not** delete the `Normaliser` class itself in this pass (zero-risk to leave it; it's 25 lines and could serve Phase 3.2's volume scaling later). Optionally add a docstring note: "No longer used by NNPredictor since per-window ratio normalisation (Phase 1.1)."

**Critical detail — `.copy()` on the reference:** `create_rolling_window_view` builds `X` with `np.lib.stride_tricks.sliding_window_view`, so `X` rows are overlapping *views* into one underlying buffer. `X / ref` creates a new array (safe), but if you instead wrote in-place ops (`X /= ref`) you would corrupt overlapping windows and the source series. Never mutate `X`/`Y` in place; and take `ref` as a `.copy()` before dividing so `ref` can't alias anything.

### 4b. `NNPredictor` changes

**`prepare_data()` (L1061–1081):**
- Feed the **raw** price series to `create_rolling_window_view`: change `input_series=self.normalised_price_series` → `input_series=self.price_series`.
- After the call, apply `normalise_windows` and store the reference:

```python
self._X, self._Y, self._X_reference_prices = preprocessing.normalise_windows(self._X, self._Y)
```

- Add `_X_reference_prices = None` to the placeholder block in `__init__` (L720–735) and expose a `X_reference_prices` property mirroring the `X`/`Y` lazy-init pattern.

**`__init__` (L721–723):** delete `self._normalised_price_series = None`, `self._normaliser = preprocessing.Normaliser()`, and the eager-fit line `_ = self.normalised_price_series`. Delete the `normalised_price_series` property (L806–810) and the `normaliser` property (L812–814). Grep the file afterwards for `_normaliser` / `normalised_price_series` to catch stragglers — the known consumers are the two plotting methods and `predict()` (handled below). Note the eager-fit line at L723 currently forces a data import at construction time; removing it makes construction lazier, which is fine (all consumers go through lazy properties).

**Reference prices must survive the split.** `split_data()` (L1083–1098) must also split `_X_reference_prices` into `_X_reference_prices_train` / `_X_reference_prices_val` with the *same* boundary as X/Y. Under 1.2 the split becomes a deterministic chronological cut, so the cleanest approach is to extend `create_train_validation_split` to accept and split one extra optional array (see §5) — do 1.1 and 1.2 as one combined change to `split_data`/`create_train_validation_split` if that's easier; they touch the same seam.

**Plotting methods** (these are the easy-to-miss consumers):
- `plot_train_validation_overview` (L1104/1106): replace `self._normaliser.inverse_transform(self.Y_train[:, 0])` with `self.Y_train[:, 0] * self.X_reference_prices_train[:, 0]` (and same for val).
- `plot_prediction_overview` (L1143–1146): the loop unpacks per-row `features`, `pred`, `target`. Also zip in the per-row reference (select train/val reference array alongside the other arrays at L1123–1127) and replace each `self._normaliser.inverse_transform(v)` with `v * ref_row`.

**`describe()` / `__str__` note:** no normalisation references there, but they do reference `randomise_validation_data_every` — handled in §5.

### 4c. `NNPredictor.predict()` (L1266–1317)

Walkthrough of the new logic:

```python
input_values = np.array(input_values, dtype=np.float32)
reference_price = float(input_values[-1])          # the known current price
if reference_price <= 0: raise ValueError(...)      # cheap sanity guard
normalised_input = input_values / reference_price   # ratio space; last element == 1.0 exactly

input_tensor = torch.unsqueeze(torch.Tensor(normalised_input), dim=0)
input_tensor = torch.unsqueeze(input_tensor, dim=2)   # (1, window, 1) — unchanged
...
predictions = self.nn_model(input_tensor)              # ratio-space outputs
predictions = np.squeeze(predictions.cpu().detach().numpy()) * reference_price   # back to price space
```

Concretely: L1279 (`self.normaliser.transform(...)`) becomes the division; L1290 (`self._normaliser.inverse_transform(predictions)`) becomes multiplication by `reference_price`. Everything downstream (`tendency = predictions[-1] > input_values[-1]`, date construction, plotting, pandas output) already operates on price-space values and needs no change. This is also where the scheme's cross-source benefit materialises: at inference the reference is whatever series the environment feeds in (ETF-scaled or index-scaled), so no `non_etf_price_factor`-style conversion is needed for the *model*, only for display.

**Interaction with `RLTradingEnv`:** the environment consumes predictor outputs as "price potentials." Since `predict()` still returns price-space values, the env contract is unchanged — but this only holds if `predict()` is the env's entry point. Grep `rl_environments.py` for direct uses of `normaliser` or `nn_model` to confirm nothing bypasses `predict()`; if something does, it must be updated identically.

### 4d. Model internals — any baked-in normalisation assumptions?

- **`LSTMModel.forward`** (L473–549): no assumption about input scale; recursive multi-step feedback (`x[:, -1] = outputs[:, step-1]`, L503–505) writes raw model output back as next input — self-consistent in ratio space.
- **`TransformerModel.forward`** (L188–277): teacher forcing (L267–270) feeds ground-truth `y` through `linear_1` back into the decoder — with `y` now in ratio space this stays consistent because training inputs are also ratio space. One subtlety: inputs clustered at ~1.0 (instead of z-scored ~N(0,1)) pass through `linear_1` + ReLU — with Xavier-initialised weights and zero bias this is fine; nothing assumes zero-centred input. No changes needed to either model class.
- **Pre-existing bug, out of scope, flag only:** `TransformerModel.forward` L231, `decoder_input = encoder_output[: -1:, :]` slices the **batch** dimension (`[:-1]`), not the sequence dimension — it should be `encoder_output[:, -1:, :]`. It only triggers when `use_start_token=False` (default is True). This belongs to the deferred R3 Transformer audit; leave it but note it in a `# BUG(R3):` comment if touching nearby code.

### 4e. `WeightedMSELoss` scaling fix (`src/pipeline/pt_metrics.py` L27–45)

Per the roadmap's Fable-flagged constraint: the `(ratio − 1) * 100` scaling goes **inside the loss only**. Data, model outputs, `HitRateMetric` inputs all remain unscaled ratios.

```python
class WeightedMSELoss(nn.Module):
    def __init__(self, step_weights=None, target_scale: float = 100.0):
        super().__init__()
        self.target_scale = target_scale
        self.step_weights = None if step_weights is None else torch.tensor(step_weights, dtype=torch.float32)

    def forward(self, predictions, targets):
        if not isinstance(predictions, torch.Tensor): predictions = torch.tensor(predictions)
        if not isinstance(targets, torch.Tensor): targets = torch.tensor(targets)
        scaled_diff = (predictions - targets) * self.target_scale     # == (p-1)*100 - (t-1)*100
        losses = scaled_diff ** 2
        if self.step_weights is None:
            return torch.mean(losses)
        return torch.mean(losses * self.step_weights.to(losses.device))
```

Notes:
- Mathematically `((p−1)·100 − (t−1)·100) = (p−t)·100`, so the fix is exactly a constant 10⁴ factor on the MSE. Be honest about what it buys: it does **not** change relative comparisons (early stopping uses strict `<`; `ReduceLROnPlateau` default `threshold_mode='rel'` is scale-invariant), but it (a) lifts logged/filename losses out of the 1e-6 float-noise range into a readable ~0.1–10 range, (b) scales gradients up out of the denormal-adjacent regime relative to Adam's `eps=1e-9` (optimiser initialised at predictors.py L1177), and (c) makes the "trivially predicts no-change" failure mode visible. Implement it as specified; just don't expect it to change training dynamics much beyond the clipping interaction in §3.
- The rewrite also fixes an incidental latent bug: the current code rebuilds `torch.Tensor(self.step_weights, device=predictions.device)` every forward call, and the legacy `torch.Tensor(data, device=...)` constructor rejects non-CPU devices — it would crash the moment `use_mps_if_available=True` or CUDA is used. Converting once in `__init__` and `.to(device)` in forward fixes both.
- `HitRateMetric` (pt_metrics.py L8–24): **no change**, per the roadmap's resolved R2 finding. It compares `sign(pred[:,-1] - X[:,-1])` vs `sign(target[:,-1] - X[:,-1])`; in ratio space `X[:,-1] == 1.0` exactly (IEEE-754 `x/x == 1.0`, and the float32 cast in `TimeSeriesDataset` preserves 1.0 exactly; note `hit_rate_train/val` at predictors.py L1043–1050 pass the numpy `X_train`/`X_val` directly, also exact). Do not "helpfully" scale anything here — that's the silent-breakage trap the roadmap warns about.

### 4f. Naive-baseline diagnostic (part of 1.1 per roadmap)

Add to `NNPredictor` two properties (near `loss_val`/`hit_rate_val`, ~L1037–1050):

```python
@property
def naive_loss_val(self):
    """Loss of trivially predicting 'no change' (ratio 1.0) on the validation split."""
    return self.loss_criterion(np.ones_like(self.Y_val), self.Y_val)

@property
def naive_hit_rate_val(self):
    return metrics.HitRateMetric()(np.ones_like(self.Y_val), self.Y_val, self.X_val)
```

(Caveat: `sign(1.0 - 1.0) == 0` for the naive prediction, so `naive_hit_rate_val` counts a hit only when the target is *exactly* flat — it will be ~0. That is itself informative; the meaningful comparison is `naive_loss_val` vs `loss_val`: a trained model whose val loss is not clearly below the naive loss has learned nothing.) Print both in `run_training`'s verbose summary (L1254–1256) so every training run shows model-vs-naive at a glance.

### Verification (1.1)

1. Instantiate `LSTMPredictor(preset_type='c1', price_series=<real c-series>)` without training. Assert:
   - `np.all(predictor.X[:, -1] == 1.0)` — exactly, not `allclose`.
   - `np.allclose(predictor.X * predictor.X_reference_prices, <raw windows>)` — reconstruct raw windows by running `create_rolling_window_view` on the raw series directly and compare.
   - `predictor.Y` values are ~1.0 (e.g. all within 0.8–1.2 for daily data).
2. Round-trip `predict()`: feed the last `rolling_window_size` raw prices; assert returned predictions are in the raw price range (thousands for DAX-index-scaled data, not ~1.0 and not z-scores).
3. `WeightedMSELoss()(torch.ones(4, 3) * 1.01, torch.ones(4, 3))` should return exactly `1.0` (0.01 · 100 = 1, squared = 1) — pins the scaling.
4. Train a few epochs on a tiny slice and confirm `hit_rate_val` computes without error and losses are in a readable range (~0.1–100, not 1e-6).

---

## 5. Item 1.2 — Temporal train/val split with gap

### Locations

- `create_train_validation_split` (preprocessing.py L600–632)
- `NNPredictor.split_data` (predictors.py L1083–1098) and `run_training` (L1215–1218)
- `NNPredictor.__init__` / `LSTMPredictor.__init__` / `TransformerPredictor.__init__` signatures and the three `describe()` methods
- `dataloader_val` property (L947–949)

### Mechanism

Rewrite `create_train_validation_split`:

```python
def create_train_validation_split(X, Y, X_dates, Y_dates,
                                  validation_split: float = 0.2,
                                  gap_size: int = 0,
                                  extra_arrays: tuple = (),   # e.g. (X_reference_prices,)
                                  verbose: bool = False):
    n = X.shape[0]
    n_val = int(n * validation_split)
    val_start = n - n_val
    train_end = max(val_start - gap_size, 0)
    # train = [:train_end], gap = [train_end:val_start] (discarded), val = [val_start:]
```

Slice all arrays (X, Y, both date arrays, and each array in `extra_arrays`) with the same two boundaries. **Remove the `randomise` parameter entirely** (grep confirms `split_data` is its only caller). Validation is the chronologically *last* block — that is already the non-randomised behaviour of the current code; the gap rows between `train_end` and `val_start` are discarded from both sets.

In `NNPredictor.split_data`, pass:

```python
gap_size=self.rolling_window_size + self.forecast_horizon,
extra_arrays=(self.X_reference_prices,)
```

and unpack the split reference arrays into `_X_reference_prices_train` / `_X_reference_prices_val`.

**Gap-size subtlety (rows vs samples):** the leakage condition is overlap in *samples*. When `daily_prediction_hour` filtering is active (all current presets), consecutive **rows** are ≥1 sample apart — b-presets: 14 samples/day apart; c/d-presets: 1 sample apart. A gap of `rolling_window_size + forecast_horizon` **rows** is therefore always *sufficient* (conservative: for b-presets it discards ~14x more history than strictly needed, but b-preset row counts are large so the cost is negligible; for c/d it is exact). Keep it simple and conservative — do not compute per-preset sample spacing. **Data-cost flag for the user:** for d3 (48-week windows + 6-week horizon), the gap discards 54 weekly rows ≈ one year of data on top of the 20% val block. With DAX history this leaves a workably sized train set, but d-preset training sets were already small; if d3 ends up with too few rows, the remedy is a longer price history, not a smaller gap.

Add a safety check: raise `ValueError` if `train_end <= 0` or `n_val == 0` (gap ate the whole training set — will happen if someone runs a d-preset on a short series).

### Remove `randomise_validation_data_every`

- Delete the parameter from `NNPredictor.__init__` (L654), `LSTMPredictor.__init__` (L1338), `TransformerPredictor.__init__` (L1556), and the three `super().__init__` calls.
- Delete the attribute assignment (L704) and the re-split block in `run_training` (L1215–1218). This block is precisely what made 1.3 impossible; its removal is a hard prerequisite there.
- Update the three `describe()` methods (L753, L1409, L1628) — each embeds `randomise validation data every: {self.randomise_validation_data_every}th epoch` in its f-string and will raise `AttributeError` if the attribute goes but the string stays.
- Grep the repo for `randomise_validation_data_every` afterwards; notebooks may pass it as a kwarg (out of scope to fix notebooks, but tell the user which ones would now raise `TypeError`).

### Related one-liner

`dataloader_val` (L949) currently constructs its DataLoader with `shuffle=True`. Change to `shuffle=False`. (Shuffling val batches never leaked data, but it's pointless and makes val loss traversal order nondeterministic. `dataloader_train`'s `shuffle=True` is correct — within-epoch batch shuffling of a fixed train set is standard and leaks nothing. Note `nn_model.predict` already rebuilds a `shuffle=False, batch_size=1` loader, which is why `predictions_val` aligns row-wise with `Y_val` for the hit-rate properties — that alignment is preserved.)

### Interaction with 1.1

The split must slice the reference-price array with identical boundaries (done via `extra_arrays` above). A careless implementation that recomputes references *after* splitting, or splits refs with a different index, silently mismatches windows and reference prices — plots and any future de-normalised evaluation would be wrong while losses/hit rates (pure ratio space) still look fine. That's why refs ride through the same function.

### Verification

1. `assert X_dates_train[-1][-1] < X_dates_val[0][0]` — last train window ends strictly before the first val window starts.
2. Stronger (the actual no-overlap property): `Y_dates_train[-1][-1] < X_dates_val[0][0]` — the last train **target** date precedes the first val **input** date. This is the leakage that the gap exists to kill; assert it in a quick script for one b-, one c-, one d-preset.
3. Confirm determinism: call `split_data()` twice and assert `np.array_equal` of the two `X_val`s.
4. Expected side effect — **validation hit rates will drop** relative to historical values (they were inflated by ~97% window overlap). This is correct behaviour, not a regression. Warn the user so they don't "fix" it.

---

## 6. Item 1.3 — Best-model checkpoint at early stopping

### Location

`NNPredictor.run_training` (predictors.py L1170–1264). One implementation serves both architectures since both subclasses inherit it.

### Mechanism

Restructure the tracking so it is (a) independent of early stopping being enabled, and (b) initialised before the loop (the current `if epoch == 0: ... continue` at L1226–1229 skips the progress-bar update on epoch 0 and entangles initialisation with the patience logic — replace it):

```python
import copy  # top of file

# before the loop:
best_loss_val = float('inf')
best_loss_train = None
best_epoch = -1
best_state_dict = None
counter = 0

# inside the loop, after loss_val is computed (after the scheduler step is fine;
# ReduceLROnPlateau.step(loss_val) does not modify the model):
if loss_val < best_loss_val:
    best_loss_val = loss_val
    best_loss_train = loss_train
    best_epoch = epoch
    best_state_dict = copy.deepcopy(self.nn_model.state_dict())
    counter = 0
else:
    counter += 1
    if self.early_stopping_patience != 0 and counter >= self.early_stopping_patience:
        print(f"Early stopping at epoch {epoch}; restoring best epoch {best_epoch} (val loss {best_loss_val})")
        self._n_train_epochs = epoch + 1
        break

# after the loop (runs on both break and normal completion):
if best_state_dict is not None:
    self.nn_model.load_state_dict(best_state_dict)
    self._predictions_train = self._predictions_val = None   # force recompute with restored weights
loss_train, loss_val = best_loss_train, best_loss_val        # report best, not last
self._loss_train = loss_train
self._loss_val = loss_val
```

Key details:
- **`copy.deepcopy` is mandatory** — `state_dict()` returns references to live parameter tensors; without deepcopy the "checkpoint" mutates with every subsequent `optimiser.step()` and you silently keep the last-epoch weights (the exact bug being fixed, reintroduced invisibly). These models are small (LSTM ~64 hidden / Transformer 256), so in-memory deepcopy per improvement is cheap; no need to write to disk.
- **`torch.compile` compatibility:** both `lstm_model` and `transformer_model` properties wrap the module in `torch.compile` (L1495, L1717), which prefixes state-dict keys with `_orig_mod.`. Because we save from and load into the *same* compiled instance, keys match and this is a non-issue. Do not save the checkpoint to disk and reload into a fresh uncompiled model inside this function — that would hit the prefix mismatch.
- **Checkpoint even when `early_stopping_patience == 0`**: training to a fixed epoch count should still end on the best epoch. The restructure above does this naturally.
- **Filename correctness** (the subtle part): `save_model_file` is called at L1249–1251 with `TrainL{loss_train} ValL{loss_val} ... ValHR{self.hit_rate_val}`. After restoration, `loss_train`/`loss_val` locals must be overwritten with the best-epoch values (done above) **before** the save call, and `_predictions_*` must be reset **before** `self.hit_rate_val` is evaluated so the hit rate is computed from the restored weights. Order in the tail of the function: restore weights → reset predictions → overwrite loss locals → `save_model_file(...)`. If you skip the prediction reset, the filename's `ValHR` — which is what `PredictorManager` uses to *select* models — would describe the discarded last-epoch weights.
- The `progress_bar.desc` f-string at L1242 references `counter` — with the restructure `counter` is always defined, which also fixes the epoch-0 gap.
- The verbose loss-history plot (L1257 ff.) uses `self.n_train_epochs` as x-range; with early stop `self._n_train_epochs` was updated — keep using `len(loss_train_history)` instead to be safe.

### Ordering dependency (restated)

Do not implement 1.3 before 1.2 has removed the periodic re-split: `best_loss_val` across epochs is only comparable when the validation set is fixed.

### Verification

Synthetic run: build a tiny predictor (small window, ~200-point sine-plus-noise series), train with `early_stopping_patience=3` and enough epochs to trigger stopping. Then:
1. After `run_training` returns, run one manual validation pass: `loss, _ = predictor.nn_model.run_epoch(predictor.dataloader_val, optimiser=<dummy>, loss_criterion=predictor.loss_criterion, is_training=False)` and assert `loss ≈ predictor.loss_val` — i.e. the *restored* model reproduces the recorded best loss, not the (worse) loss at the stopping epoch. (Dropout is inactive in eval mode and the val loader is now unshuffled, so this is deterministic.)
2. Assert the saved filename's `ValL` value equals `best_loss_val`, not the last printed epoch loss.

---

## 7. Old `.pt` models, `fine_tune_predictors`, and the retraining question

### Are existing saved models compatible? **No — treat them as invalid.**

The `.pt` files in `data/saved_models/` are full-module pickles (`torch.save(self.lstm_model, ...)`, loaded with `weights_only=False`). They will still **load without error** — architecture classes are unchanged — which is exactly why a coding agent might miss this. But they are semantically broken under the new pipeline in three independent ways:

1. **Input-space mismatch:** their weights map z-scored inputs (roughly −2…+2, varying) to z-scored outputs. The new pipeline feeds ratios (~1.0) and multiplies outputs by the current price. A z-space output of, say, 0.3 becomes "predicted DAX = 0.3 × 18,000 = 5,400". Not degraded — garbage.
2. **Lost normaliser state:** the old scheme's `mu`/`sd` lived in the `NNPredictor` instance (refit from the price series at construction), not in the `.pt` file — so there is no clean way to even run old models correctly once `Normaliser` is removed from the predictor.
3. **Selection-metric mismatch:** `PredictorManager` ranks by `ValHR` parsed from filenames (L1970). Old ValHRs are inflated by the random-split leakage; new ones (temporal split + gap) will be systematically lower. Mixing old and new files in one directory means `get_predictors_by_type_sorted` would *always* prefer old (invalid) models.

### Does "continuing training" on old models make sense? **No — Phase 1 requires a fresh-training bootstrap cycle.**

`PredictorManager.fine_tune_predictors` (L2038–2086) instantiates the best existing model and warm-starts from its weights. Warm-starting from z-space weights into ratio-space training is worse than random init (the model must first unlearn its input scaling) and, per point 3, the "best existing" selection itself is corrupted. Required actions:

1. **Archive old models outside the scan tree.** Trap: `add_predictors_from_dir` globs `**/*.pt` **recursively** (L1912), and `workflow.py` initialises with `recursive=True` — moving old files into a subfolder like `data/saved_models/archive/` would NOT hide them. Move them to a sibling outside `SAVED_MODELS`, e.g. `data/saved_models_pre_phase1_archive/`. (Also archive `data/saved_models/working_dir_pred_manager/` contents — that's the fine-tune working dir visible in git status.)
2. **Provide a bootstrap path for fresh models.** `fine_tune_predictors` prints "No predictor found!" and skips when a preset has no models (L2072–2074) — so after archiving, the weekly Saturday job would train nothing, forever. Add a small method to `PredictorManager`, mirroring `fine_tune_predictors`' structure:

```python
def train_fresh_predictors(self, architectures, presets, save_directory,
                           train_epochs=200, early_stopping_patience=10, **predictor_kwargs):
    """Train new randomly-initialised predictors (no warm start). Bootstrap after scheme changes."""
    for architecture, preset in product(architectures, presets):
        price_series = <resolve series from preset first letter, same b/c/d branch as instantiate_predictor —
                        factor that branch out into a helper `_resolve_price_series(preset_category)` and reuse it>
        cls = LSTMPredictor if architecture == 'LSTM' else TransformerPredictor
        instance = cls(preset_type=preset, price_series=price_series,
                       model_save_directory=save_directory, verbose=True, **predictor_kwargs)
        instance.run_training(custom_n_epochs=train_epochs,
                              custom_early_stopping_patience=early_stopping_patience)
```

Run it once (manually or via a temporary workflow call) for `['LSTM'] × ('b1','b2','c1','c2','d1','d2','d3')` to repopulate `SAVED_MODELS`. After that, the existing weekly `fine_tune_predictors` warm-start flow is coherent again (new-scheme models warm-starting new-scheme training).
3. **Runtime continuity warning for the user:** between archiving and the bootstrap run completing, `update_env_predictors()` / env initialisation in `workflow.py` (L604–609) will fail (`IndexError` on the empty `get_predictors_by_type_sorted(...)[0]`). Do the archive + bootstrap in one sitting, not across a live trading day.

---

## 8. Cross-cutting verification (after all steps)

1. **End-to-end smoke test:** fresh-train one cheap preset (`c1`, small window/horizon) for ~30 epochs. Confirm: training runs, early stopping and restoration message fire, a `.pt` file appears with plausible `TrainL/ValL/TrainHR/ValHR` in the name, `PredictorManager.add_predictors_from_dir` picks it up and infers `preset_type='c1'`, `instantiate_predictor` reloads it, and `predict()` on the latest window returns DAX-scale prices.
2. **Leakage tripwires stay green:** `X[:, -1] == 1.0` exactly; `Y_dates_train[-1][-1] < X_dates_val[0][0]`; `split_data()` deterministic.
3. **Sanity expectations to tell the user:** validation hit rates near ~0.5 and val loss near the naive baseline are now *plausible honest outcomes*, not bugs. The artificial ceiling and the inflated hit rates were artifacts of the two bugs being fixed; the new numbers are the real baseline Phase 2+ has to improve on.
4. **Out-of-scope items deliberately not touched:** walk-forward harness (1.5), the Transformer `use_start_token=False` slicing bug (R3, §4d), notebooks passing removed kwargs, and `private/env_configuration.txt` possibly containing `a1` (user must check manually).
