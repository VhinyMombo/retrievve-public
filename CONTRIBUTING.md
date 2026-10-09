# Contributing to retrievve

## Rules
1. **`main` changes only through pull requests.** No direct pushes. A local `pre-push` hook enforces this.
   GitHub branch protection is not available on this plan.
2. **Author and committer is always VhinyMombo.** No AI attribution in commits or PRs: no `Co-Authored-By: Claude`
   and no "Generated with" footers. `.claude/settings.json` turns this off for Claude Code, and a `commit-msg` hook rejects it.
3. **One topic per branch and PR.** Branch names: `data/…`, `spec/…`, `train/…`, `eval/…`, `docs/…`, `fix/…`.
4. **The AI scientist validates and merges** every PR (squash merge), by the user's decision. The lead may review
   and comment first. `DATA_SPEC.md` changes still need the lead's sign-off before merge. Draft PRs are never merged.
5. **Never commit data or models.** `data/raw`, `data/corpus`, `data/train`, `data/v1` (except `reports/` and
   `sources.yaml`), `models/` and `.venv/` are gitignored. Some sources are `release: false` (DATA_SPEC §10).
6. **Delete a branch locally once it is merged.** Squash merges are not ancestry merges, so use `-D`. If the branch
   is checked out in your worktree, leave it first: `git switch --detach origin/main` (or start the next branch
   from `origin/main`), then `git branch -D <branch>`. Don't delete a branch that an open PR uses as its base;
   retarget that PR to `main` first.
7. **Tests (user decision, revised 2026-10-10).** This repo stays **private** (the team's working repo). A separate, scrubbed
   **public** repo, published by the lead, runs CI there. In this private repo GitHub Actions is blocked by billing, so a CI
   check here proves nothing. Every PR body states the exact local test command it ran (e.g. `.venv/bin/pytest -q &&
   .venv/bin/ruff check .`) and the result, and the AI scientist reruns the tests locally in a clean worktree at the PR head
   before merging.
8. **No `git stash`.** Stashes live in the shared repository, so every worktree sees, and can pop or drop, everyone else's
   stash (one was lost this way on 2026-10-10). To park work, commit it on your own branch (e.g. `wip: ...`) and amend
   or squash later.

## Working copies
The Dropbox folder is the **shared checkout**. It stays on `main`. Nobody switches branches in it,
because several sessions use it at the same time.

Each contributor works in their own worktree **outside Dropbox**:

```sh
cd "$RETRIEVVE_SHARED"        # shared checkout
git fetch && git worktree add -b <branch> ~/code/retrievve-wt/<role> origin/main
cd ~/code/retrievve-wt/<role>
ln -s "$RETRIEVVE_SHARED/data/raw"    data/raw     # read data from Dropbox
ln -s "$RETRIEVVE_SHARED/data/corpus" data/corpus
ln -s "$RETRIEVVE_SHARED/models"      models
uv sync                                                                     # this worktree's own .venv
```

**Virtualenvs:** one per worktree (`<worktree>/.venv`, created by `uv sync`). Do not set
`UV_PROJECT_ENVIRONMENT` and do not share a venv: the package is installed in editable mode, so a shared venv
runs whichever worktree it was last synced from. Run tools as `.venv/bin/retrievve …` / `.venv/bin/pytest` (or
`make …`). The shared Dropbox checkout gets **no** venv. After pulling a change to `pyproject.toml`/`uv.lock`,
run `uv sync` again in your worktree.

Then commit, `git push -u origin <branch>`, `gh pr create`. After a merge, the AI scientist runs `git pull` in the shared checkout.

## Team
| role | owns |
|---|---|
| Lead | planning, coordination, first review, sign-off on spec changes |
| AI scientist | `DATA_SPEC.md`, model architecture, training configs, **PR validation and merge** |
| Data engineer | `src/retrievve/data/` pipeline, `configs/v1_pipeline.yaml`, packaging/CI |
| Data scientist | `src/retrievve/analysis/`, `data/v1/reports/`, thresholds, leakage audit |
| Applied maths | optimisation proposals and measurements, sent to the AI scientist as diffs |
