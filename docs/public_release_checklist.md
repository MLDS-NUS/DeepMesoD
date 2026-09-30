# Public release checklist

- Run `python -m pytest` and the synthetic smoke configuration.
- Run `python tools/self_test_agent_workflow.py` from the built wheel environment.
- Run `deepmesod reproduce` once with a fresh private output directory and inspect
  its JSON, Markdown, PDF, and PNG reports.
- Confirm every bundled example points to an explicit manifest and declares its
  model seeds and checkpoint policy.
- Check that forward increments are formed before trajectories are merged.
- Check the complete `K` response and conditional covariance calibration.
- Check unit-sum and configured symmetry residuals for `J`.
- Scan for absolute paths, credentials, datasets, checkpoints, generated figures,
  caches, and user-specific environment files.
- Confirm `.gitignore` uses root-anchored private directories and does not hide
  `src/deepmesod/data/`.
- Confirm source, documentation, configuration, comments, and command output use
  professional English and contain no Chinese text.
- Keep collaborator-owned ASD or CrBr3 code out until its provenance is supplied.
- Choose the repository license and author metadata before publishing.
- Run `python tools/audit_release.py` on the release tree.
