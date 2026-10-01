# Robo-Dopamine

Upstream:
https://github.com/FlagOpen/Robo-Dopamine.git

Upstream branch:
`main`

Imported upstream commit:
`2c714abca66329f8e2113587226bc9556726f6c2`

Management:
`git subtree --squash` under `repos/Robo-Dopamine`, matching ProcVLM.
LF3R-specific changes are committed directly in the LF3R repository.
There is no independent Git repository or submodule in this directory.

## Sync upstream

From the LF3R repository root, with a clean working tree:

```bash
git subtree pull --prefix=repos/Robo-Dopamine https://github.com/FlagOpen/Robo-Dopamine.git main --squash
```

No additional Git remote is required. Keep using `--squash` for future pulls.
Resolve any conflicts with LF3R changes in the LF3R repository.

The import uses the unchanged official checkout already present locally.
An online upstream check timed out during migration; this commit is the
imported version, not a claim that upstream `main` has no newer commits.
