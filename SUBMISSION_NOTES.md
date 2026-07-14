# Submission Notes

Use the clean artifact archive for submission instead of uploading the working
directory directly.

Excluded local-only files:

- `.venv/`
- `.idea/`
- `.pytest_cache/`
- `__pycache__/`
- `*.egg-info/`
- `.Rhistory`
- `.DS_Store`

The main, noise, rework-sensitivity, mixed, and three focused training-size
runs include full per-condition artifacts. The broader
`configs/sensitivity.json` configuration remains a prepared extension and is
not part of the reported final results.
