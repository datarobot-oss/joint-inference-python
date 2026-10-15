# Agent Contract

Rules for coding agents and human contributors working in this repository. `CLAUDE.md` is a symbolic link to this file.

## Public by design

This repository is published. Anyone can read every file and every commit in its history, and history cannot be cleaned after publication, so each commit must be safe to publish on its own. Never add:

- code, names, or descriptions of model generations whose weights are not openly released;
- training or data-generation code — this project loads and serves released checkpoints, nothing more;
- run configurations, or the names of runs, checkpoints, or experiments;
- internal file paths, hostnames, service or deployment identifiers, credentials, or tokens;
- planning documents, roadmap or milestone labels, or ticket codes;
- links to, or references into, private repositories.

`task leak-guard` (a pre-commit hook), a `commit-msg` hook, and `task leak-guard:history` (in CI) enforce what can be checked mechanically, adding a maintainer's git-ignored `.leak-patterns.private` rules when present; see the README. A passing guard does not make a change safe to publish. Never weaken a rule to get a change through; remove the material instead.

Dependencies point one way: this project depends only on public packages and never on private code. Private code may build on it through its public interfaces. When unsure whether something is safe to publish, leave it out and ask.

## Engineering

- **Fail fast.** No silent fallbacks, no compatibility paths, no deprecation flags, no `try/except` that swallows or rewrites errors. Exceptions are for IO and races; otherwise re-raise.
- **Ground in facts.** Before planning, claiming, or editing, read the relevant source, configuration, docs, and test output, and act only on beliefs backed by something just read or run.
- **Commands.** Run ad-hoc Python and CLI commands through `task run -- <command>`, never bare `python` or hand-written `uv run`, so no invocation re-resolves the `.venv` or `uv.lock` by accident. Explain what each command does and why before running it.
- **Taskfile targets** print an explicit `OK:` or `FAIL:` line; success is never inferred from silence.
- **Platforms.** Everything works on Ubuntu, Amazon Linux, and macOS; Windows is not supported. Shell code in the `Taskfile` and scripts uses POSIX syntax.
- **File size.** A Python source file stays at or below 1000 lines; a module that would grow past that becomes a package of submodules grouped by concern.
- **Comments** explain why, not what. Configuration comments state only what stays true whatever the value is — never the current value or arithmetic on it.
- **No transient labels** — no phase, milestone, sprint, or ticket identifiers in docstrings, comments, help strings, error messages, or tests. Describe the lasting behavior instead.
- **Tests** are fast, run on CPU, and create any checkpoint they need instead of assuming one exists locally. When fixing a bug, add the failing test first.

## Git

Never run a state-changing git command — staging, committing, branching, stashing, resetting, cleaning, pushing, or rewriting history — unless the user explicitly asks for that specific command. Read-only inspection (`git status`, `git diff`, `git log`, `git show`) is always allowed. If a task seems to need a state-changing operation, stop and ask the user to run it.

## Finishing a change

Run `task pre-commit` and fix every reported issue. Then show the diff, explain why each change is necessary, and end with a suggested commit message in Conventional Commits style (`feat: Message`, `fix: Message`, `docs: Message`).

## Communication

Lead with the outcome: what now works, what the answer is, and anything the user must decide or act on. Evidence follows underneath — file and line references, command output, measured numbers. Leave out what the user can already see: no restatement of the request, no narration of steps taken, no closing summary. Present a set of items compared along the same few dimensions as a Markdown table. Before asking the user a question, explain the context and terminology it depends on. Do not mention or recommend credential rotation.
