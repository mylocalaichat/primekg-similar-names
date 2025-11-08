# PrimeKG Similar Names

## Launch

### VS Code
Press F5 or use the "Dagster Dev" launch configuration

### Command Line
```bash
DAGSTER_HOME=${PWD}/.dagster uv run dagster dev -m dagster_assets
```

## Development

### Run Tests
```bash
uv run pytest tests/ -v
```

### CI/CD
GitHub Actions runs tests automatically on push and pull requests. All tests must pass before merging.
