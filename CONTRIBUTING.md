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
  [datarobot-oss/joint-inference-python](https://github.com/datarobot-oss/joint-inference-python/issues)
  with a minimal reproduction, the package version (`version` in `pyproject.toml`), and
  the Python version.
- **Security issues:** do not open a public issue. Follow [SECURITY.md](SECURITY.md).

## Local setup and checks

Prerequisites: [uv](https://docs.astral.sh/uv/), [Task](https://taskfile.dev/), and docker for the
license header hook. Ubuntu, Amazon Linux, and macOS are supported.

```sh
task setup        # create .venv, install dependencies, git hooks, ripgrep, and typos
task pre-commit   # run every check that gates a commit
```

Run `task pre-commit` and fix every reported issue before opening a pull request.

| Task | What it does |
|---|---|
| `task setup` | Create or reuse `.venv`, sync all dependency groups, install the git hooks, ripgrep, and typos |
| `task start` | Write `.env` with the DataRobot CLI, which also logs in and fills in the DataRobot endpoint and token |
| `task run -- <command>` | Run a command inside the project environment |
| `task test` | Run the unit tests |
| `task coverage` | Run the unit tests and enforce the coverage floor |
| `task check` | Read-only static checks: typos, Ruff lint and format check, ty |
| `task license-check` | Fail when a Python file lacks the license header (needs docker) |
| `task license-fix` | Add the license header to every Python file that lacks it (needs docker) |
| `task leak-guard` | Fail when a file in the working tree carries private material in its path or content |
| `task leak-guard:history` | Fail when any commit reachable from any ref carries private material in its message, paths, or files; pass revisions to narrow the scan (`task leak-guard:history -- origin/main..HEAD`) |
| `task secret-scan:history` | Fail when any commit reachable from any ref carries a credential, scanned by gitleaks |
| `task vuln-scan` | Fail when a package pinned in `uv.lock` has a known vulnerability, scanned by osv-scanner (needs docker) |
| `task format` | Apply the Ruff formatter |
| `task pre-commit` | Run every pre-commit hook on all files |
| `task build` | Build the source distribution and wheel into `dist/` |
| `task serve` | Serve the checkpoint `.env` configures, downloading it first when `.env` names a URL |
| `task deploy:preview`, `task deploy`, `task deploy:destroy` | Preview, apply, or delete the DataRobot Workload serving the checkpoint `.env` configures |
| `task deploy:outputs`, `task deploy:probe` | Print the deployed Workload's endpoint and IDs, or probe it through DataRobot |
| `task deploy:bundles` | List the resource bundles the DataRobot cluster offers Workloads |
| `task probe` | Probe the local service, or the services named after `--` |
| `task docker:build` | Build the container image, which holds no weights |
| `task docker:run` | Run the container image on the checkpoint `.env` configures |

Every task prints an `OK:` or `FAIL:` line. CI runs `task pre-commit` and `task build` on Ubuntu
and macOS; the macOS job skips the license header hook because its runners have no docker.

## Package layout

| Package | Contents |
|---|---|
| `jointfm_inference.model.service` | HTTP service: environment configuration, checkpoint download, request validation, conditions, scoring, JSON serialization |
| `jointfm_inference.model.inference` | Inference session, backend registry, backend-neutral contract, conditioning, the JointFM 1 serving backend |
| `jointfm_inference.model.jointfm_1` | The JointFM 1 network, its Student-t mixture head, and its checkpoint loader |
| `jointfm_inference.model` | Checkpoint reading, linear algebra, precision, and the command-line probe |
| `jointfm_inference.cli` | The `jointfm-inference` command and its `serve` and `probe` subcommands |
| `jointfm_inference.data` | Canonical samples, request times, DataFrame conversion |

## Checkpoint format

A checkpoint is a plain-data mapping saved with `torch.save` and loaded with `weights_only=True`.
Its layout is versioned, and the runner reads exactly format version 1
(`jointfm_inference.model.checkpoint`); a file of any other version, or with any entry outside the
layout, is refused before a backend is resolved:

| Entry | Holds |
| --- | --- |
| `cfg` | The network sizes; the generation's backend defines the fields |
| `model_state` | The network's state dict, every floating tensor in the dtype `weights_dtype` names |
| `runtime_metadata` | Exactly `generation`, `format_version` (`1`), and `weights_dtype` (`fp32`, `bf16`, or `fp16`; loading upcasts to float32) |
| `inference_metadata` | Exactly `time_index_support` and `data_generation`, the capacity envelope that `/healthz` reports |

`runtime_metadata.generation` names the generation that wrote the file; the session resolves that
marker to a serving backend through the `jointfm_inference.backends` entry-point group and fails
at load when no installed distribution registers it. This distribution registers `jointfm-1`.

## Keeping the repository publishable

This repository is published, so `scripts/leak_guard.py` fails on material that must stay private:
proprietary file headers, DataRobot object IDs, internal hosts and private addresses, links to
non-public repositories, planning labels, names of unreleased model generations, and credential
shapes. It runs as a pre-commit hook over the working tree, as a `commit-msg` hook over each
message, and in CI over the full history, because a published history cannot be cleaned.

Some rules cannot be published because they name what they guard. Maintainers keep those in a
git-ignored `.leak-patterns.private` at the repository root: one Python regular expression per
line, blank lines and `#` comment lines skipped. The guard reports whether it loaded the file and
runs the public rules alone when it is absent, as in CI. A dangling link fails. A passing guard
does not prove a change safe to publish.

Credentials are a separate check: the `gitleaks` pre-commit hook scans each staged change, and
`task secret-scan:history` scans every reachable commit in CI.

Known vulnerabilities are a third: `task vuln-scan` checks every package pinned in `uv.lock`
against the OSV database on each pull request, each push to `main`, and weekly, since advisories
land without a code change. [osv-scanner.toml](osv-scanner.toml) holds the waivers, each with an
expiry date, for fixes that are still too new for the `exclude-newer` cooldown in
`pyproject.toml`.

## License headers

Every Python file carries the license header that [.licenserc.yaml](.licenserc.yaml) defines; the
`insert-license` pre-commit hook adds it with
[skywalking-eyes](https://github.com/apache/skywalking-eyes), which runs as a docker image.

## Pull requests

1. Fork the repository, or branch directly if you are a maintainer.
2. Create a feature branch off `main`.
3. Make focused commits that pass `task pre-commit`.
4. Open a pull request against `main` describing what changed, why, and how you verified it.

## Maintainers

Stefan Hackmann (@shackmann) and Matt Nitzken (@mjnitz02), as listed in
[CODEOWNERS](.github/CODEOWNERS).
