# Repository Guidelines

## Project Structure & Module Organization

LF3R treats the repository checkout itself as `PROJECT_ROOT`; its absolute path may differ across servers. Place upstream code in `repos/`, with one subdirectory per project (for example, `repos/SAFE` and `repos/LIBERO`). Store datasets and model weights in `datasets/` and `checkpoints/`; never commit these large artifacts. Generated results belong in `outputs/`, command logs in `logs/`, and machine or dependency reports in `environment_reports/`. Conda environments and package caches are project-local under `conda_envs/` and `cache/`. Treat `project_env.sh`, `SYSTEM_INFO.txt`, and `SETUP_STATUS.md` as the top-level operational record.

## Environment, Build, and Development Commands

From the repository root, begin a shell session with:

```bash
source ./project_env.sh
test "$PROJECT_ROOT" = "$(pwd -P)"
```

`project_env.sh` derives the root from its own location, so do not hard-code a server-specific LF3R absolute path. If an exceptional deployment needs a different root, set `LF3R_PROJECT_ROOT` before sourcing the environment. Scripts under `tools/` should locate `project_env.sh` relative to `${BASH_SOURCE[0]}` rather than assuming a fixed mount point.

There is no repository-wide build command yet. Run setup, lint, and test commands from the relevant directory under `repos/`, following that upstream project's documentation. Use the project-local environment explicitly, such as `conda run -p "$LF3R_ENV_SAFE" <command>`. Capture reproducible diagnostics in `logs/` and environment summaries in `environment_reports/`.

## Coding Style & Naming Conventions

Preserve each upstream repository's formatter, linter, and language conventions. For LF3R-owned shell scripts, use Bash, four-space indentation, quoted variable expansions, and descriptive `UPPER_SNAKE_CASE` environment variables. Name generated reports and logs with timestamps (`YYYYMMDD_HHMMSS`) to avoid overwrites. Keep configuration portable by deriving paths from `PROJECT_ROOT`; do not add absolute paths outside it or symbolic links that redirect project data elsewhere.

## Testing Guidelines

Use the official test or smoke-test entry point for each component. Keep validation minimal: dependency imports, CLI startup, and at most one documented inference or rollout smoke test. GPU jobs and downloads larger than 1 GB must run sequentially. Record the exact command, environment, timestamp, outcome, and relevant commit hash. After three non-destructive failures for one component, stop and mark it `BLOCKED` in `SETUP_STATUS.md`.

## Commit & Pull Request Guidelines

No local Git history currently defines a convention. Use short, imperative commit subjects with an optional component prefix, for example `SAFE: document smoke-test setup`. Keep commits focused and exclude caches, environments, checkpoints, datasets, outputs, and logs. Pull requests should explain scope, list verification commands and results, link relevant issues, note hardware or storage assumptions, and include screenshots only for visual changes.

## Safety & Configuration

Do not use `sudo`, alter system Python/CUDA or shell startup files, authenticate to Hugging Face, start large-scale training, or move data outside `PROJECT_ROOT`. Check available GPU and storage capacity before installations, downloads, or inference.
