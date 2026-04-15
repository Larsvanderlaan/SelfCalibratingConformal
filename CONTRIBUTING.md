# Contributing

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

## Common commands

```bash
pytest
python -m build
jupyter nbconvert --to notebook --execute quickstart_regression.ipynb --output /tmp/quickstart_regression.ipynb
mkdocs build
```

## Release checklist

1. Update user-facing docs and notebooks if APIs changed.
2. Run `pytest` and notebook smoke tests.
3. Build the package with `python -m build`.
4. Update `CHANGELOG.md`.
5. Create or update a GitHub Release from the tested commit.

## Compatibility policy

- Keep the established `SelfCalibratingConformalPredictor` imports and method names working.
- Prefer additive changes plus deprecation guidance over silent breaking changes.
- Add tests for any new user-facing option, especially custom scorers, calibrators, and quantile interval logic.
