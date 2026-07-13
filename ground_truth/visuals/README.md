# Ground-Truth Visual Companions

The PNG and PDF files in this folder are generated, reviewer-friendly diagrams
of the documented synthetic scenarios. They clarify the intended baseline,
optional work, alternative order, rework path, and mixed setting.

They complement the formal BPMN source files in `../bpmn/`. The exact accepted
finite trace language remains `../valid_variants.csv`.

Regenerate these figures together with the paper figures from the repository
root:

```bash
.venv/bin/python generate_paper_figures.py
```
