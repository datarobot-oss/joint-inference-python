# Contributing to `jointfm-inference`

Thanks for your interest in improving the JointFM inference runner. This repository is public, so
every commit, pull request, and comment is published.

## Ground rules

- The project is licensed under **Apache-2.0**. By contributing, you agree that your contribution
  is licensed under the same terms (inbound = outbound).
- Follow the project rules in [AGENTS.md](AGENTS.md), in particular what must never be committed.
- Be respectful in issues, reviews, and discussions.

## Reporting issues

- **Bugs and feature requests:** open a GitHub issue at
  [datarobot/joint-inference-python](https://github.com/datarobot/joint-inference-python/issues)
  with a minimal reproduction, the package version (`version` in `pyproject.toml`), and
  the Python version.
- **Security issues:** do not open a public issue. Follow [SECURITY.md](SECURITY.md).

## Local setup and checks

The [Development](README.md#development) section of the README lists the prerequisites and every
`task` target. In short:

```sh
task setup        # create .venv, install dependencies, git hooks, ripgrep, and typos
task pre-commit   # run every check that gates a commit
```

Run `task pre-commit` and fix every reported issue before opening a pull request.

## Pull requests

1. Fork the repository, or branch directly if you are a maintainer.
2. Create a feature branch off `main`.
3. Make focused commits that pass `task pre-commit`.
4. Open a pull request against `main` describing what changed, why, and how you verified it.

## Maintainers

Stefan Hackmann (@shackmann) and Matt Nitzken (@mjnitz02), as listed in
[CODEOWNERS](.github/CODEOWNERS).
