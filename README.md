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
| `task format` | Apply the Ruff formatter |
| `task pre-commit` | Run every pre-commit hook on all files and check the pending commit message |
| `task build` | Build the source distribution and wheel into `dist/` |

Every task prints an `OK:` or `FAIL:` line. Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/), checked by Commitizen in the `commit-msg` hook. CI runs `task pre-commit` and `task build` on Ubuntu and macOS.

Rules for contributors and coding agents are in [AGENTS.md](AGENTS.md).

## License

This repository does not carry a license yet.
