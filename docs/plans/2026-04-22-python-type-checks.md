# Python Type Checks Rollout Plan

> **For Hermes:** Keep this rollout incremental. Prioritize a clean, enforced baseline over attempting to annotate the entire training stack in one pass.

**Goal:** Add practical Python type checking to the x-veon repo by introducing a checker configuration, wiring it into the Conda environment, and bringing a focused subset of actively maintained infrastructure modules under type checking.

**Architecture:** Use `mypy` as the initial type checker because the repo is plain Python, already uses standard typing syntax, and has no packaging metadata. Start with the new detached-training/state-management modules and a small utility module, while explicitly scoping or suppressing legacy modules that still need broader annotation work.

**Tech Stack:** Conda environment (`environment.yml`), `mypy`, stdlib typing, existing Python modules.

---

## Scope for this pass

Target modules:
- `observer.py`
- `state_server.py`
- `state_client.py`
- `checkpoint_registry.py`
- `cfa.py`
- `model.py`
- `build_dataset.py`
- `export_onnx.py`
- `tests/test_training_state_ipc.py`

Support work:
- add `mypy` to `environment.yml`
- add `mypy.ini`
- fix local type issues in targeted files
- document current scope in config comments

Non-goals for this pass:
- full repo typing for heavy Torch/data/model modules
- fixing all legacy typing issues in `dashboard.py`, `dataset.py`, `model.py`, `losses.py`, `train.py`

---

## Verification

Run in the `x-veon` Conda env:

```bash
mypy
python -m unittest tests.test_training_state_ipc -v
```

Expected:
- `mypy` passes for the configured baseline
- tests still pass
