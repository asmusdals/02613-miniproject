# 02613 Mini-Project: Wall Heating

Source of truth: `docs/project_exercises.pdf`.

This repo is structured for iterative development and DTU HPC batch runs.

## Quick start (local)

```bash
python simulate.py 10
```

## Suggested structure

- `docs/`: assignment PDF and reference scripts/templates.
- `src/`: reusable Python code (library-style).
- `scripts/`: runnable entrypoints (optimized variants, plotting, analysis).
- `jobs/`: DTU HPC LSF batch scripts (`bsub < ...`).
- `notebooks/`: exploration/visualization notebooks.
- `reports/`: report material and figures (commit the final figures you use).
- `data/`: local data (gitignored).
- `results/`: generated outputs (gitignored).

