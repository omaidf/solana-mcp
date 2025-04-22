# Python Version Management

## Quick Update

```bash
# Run to update Python version across all files
python scripts/update_python_version.py 3.13
```

## Files Affected

- Dockerfile: `FROM python:3.13-slim`
- runtime.txt: `python-3.13`
- .tool-versions: `python 3.13`
- .python-version: `3.13`
- pyproject.toml: `python_version = "3.10"` 