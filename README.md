# joint-inference-python

`jointfm-inference` is the inference runner for JointFM: it will load released JointFM checkpoints and serve joint probabilistic forecasts over HTTP. Applications talk to a running instance through the [`jointfm-client`](https://github.com/datarobot/joint-client-python) SDK ([PyPI](https://pypi.org/project/jointfm-client/)).

The repository currently holds the project tooling only; the serving code is not here yet.

## Development

Prerequisites: [uv](https://docs.astral.sh/uv/) and [Task](https://taskfile.dev/). Ubuntu, Amazon Linux, and macOS are supported.

```sh
task setup        # create .venv, install dependencies, pre-commit hooks, ripgrep, and typos
task pre-commit   # run every check that gates a commit
```

| Task | What it does |
|---|---|
| `task setup` | Create or reuse `.venv`, sync all dependency groups, install the git hooks, ripgrep, and typos |
| `task run -- <command>` | Run a command inside the project environment |
| `task test` | Run the unit tests |
| `task coverage` | Run the unit tests and enforce the coverage floor |
| `task check` | Read-only static checks: typos, Ruff lint and format check, ty |
| `task leak-guard` | Fail when a file in the working tree carries private material in its path or content |
| `task leak-guard:history` | Fail when any commit reachable from any ref carries private material in its message, paths, or files |
| `task format` | Apply the Ruff formatter |
| `task pre-commit` | Run every pre-commit hook on all files and check the pending commit message |
| `task build` | Build the source distribution and wheel into `dist/` |

Every task prints an `OK:` or `FAIL:` line. Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/), checked by Commitizen in the `commit-msg` hook. CI runs `task pre-commit` and `task build` on Ubuntu and macOS.

### Leak guard

This repository is published, so `scripts/leak_guard.py` fails on material that must stay private: proprietary file headers, DataRobot object IDs, internal hosts and private addresses, links to non-public repositories, planning labels, names of unreleased model generations, and credential shapes. It runs as a pre-commit hook over the working tree, as a `commit-msg` hook over each message, and in CI over the full history, because a published history cannot be cleaned.

Some rules cannot be published because they name what they guard. Maintainers keep those in a git-ignored `.leak-patterns.private` at the repository root: one Python regular expression per line, blank lines and `#` comment lines skipped. The guard reports whether it loaded the file and runs the public rules alone when it is absent, as in CI. A dangling link fails. A passing guard does not prove a change safe to publish.

Rules for contributors and coding agents are in [AGENTS.md](AGENTS.md).

## License

This repository does not carry a license yet.
