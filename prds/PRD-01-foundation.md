# PRD-01: Foundation

## Objective
Set up project structure, virtual environment, dependencies, and configuration files.

## Deliverables
1. `pyproject.toml` with hatchling backend, torch cu128, all dependencies
2. `.venv/` created with `uv venv --python 3.11`
3. `configs/paper.toml` -- paper-faithful hyperparameters
4. `configs/debug.toml` -- smoke test config (2 epochs, batch 2)
5. `src/dark_driving/__init__.py` -- package init with version
6. `anima_module.yaml` -- module manifest
7. Verify `uv sync` succeeds and `import dark_driving` works

## Acceptance Criteria
- `uv run python -c "import dark_driving; print(dark_driving.__version__)"` prints version
- All config files parse without error
- ruff check passes with zero errors
