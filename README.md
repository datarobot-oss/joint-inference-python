<p align="center">
  <a href="https://github.com/datarobot-oss/joint-inference-python">
    <img src="https://raw.githubusercontent.com/datarobot-oss/joint-inference-python/main/.github/datarobot_logo.avif" width="600px" alt="DataRobot Logo"/>
  </a>
</p>
<h3 align="center">JointFM Inference Runner</h3>

<p align="center">
  <a href="https://www.datarobot.com/">Homepage</a>
  ·
  <a href="https://pypi.org/project/jointfm-inference/">PyPI</a>
  ·
  <a href="https://github.com/datarobot-oss/joint-client-python">Python client</a>
  ·
  <a href="https://docs.datarobot.com/en/docs/get-started/troubleshooting/general-help.html">Support</a>
</p>

<p align="center">
  <a href="https://github.com/datarobot-oss/joint-inference-python/actions/workflows/ci.yml">
    <img src="https://github.com/datarobot-oss/joint-inference-python/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://pypi.org/project/jointfm-inference/">
    <img src="https://img.shields.io/pypi/v/jointfm-inference" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/jointfm-inference/">
    <img src="https://img.shields.io/pypi/pyversions/jointfm-inference" alt="Python versions">
  </a>
  <a href="https://github.com/datarobot-oss/joint-inference-python/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/datarobot-oss/joint-inference-python" alt="License">
  </a>
</p>

`jointfm-inference` is the inference runner for JointFM: it loads JointFM checkpoints and serves joint probabilistic forecasts over HTTP. Applications talk to a running instance through the [`jointfm-client`](https://github.com/datarobot-oss/joint-client-python) SDK ([PyPI](https://pypi.org/project/jointfm-client/)). The service runs on its own, started by `jointfm-inference serve` or the container image, or your application starts and stops it from Python, with this package installed beside the client ([Start from Python](#start-from-python)); then there is no separate service to run.

The repository holds the serving code for JointFM 1 checkpoints, the `jointfm-inference` command, and a container image. Neither the package nor the image contains weights: every service names the checkpoint it serves in its own configuration.

> **Status: beta.** Every released JointFM 1 checkpoint keeps loading and predicting as it did, and every documented request keeps working; a checkpoint layout is dropped only after a notice of at least 90 days ([Compatibility of released exports](https://github.com/datarobot-oss/joint-inference-python/blob/main/CONTRIBUTING.md#compatibility-of-released-exports)). Anything else can still change in a minor release while the version is `0.x`.

## Quickstart

1. **Check the requirements:** Python 3.13 or newer on Ubuntu, Amazon Linux, or macOS. [Hardware](#hardware) lists the GPU, memory, and disk space the service needs.
2. **Get a weights file.** Serving needs a JointFM serving weights file: a JointFM 1 checkpoint in the format the runner reads. How to obtain one will be documented here.
3. **Install the package** into a fresh environment:

   ```sh
   python3.13 -m venv .venv
   . .venv/bin/activate
   pip install jointfm-inference
   ```

4. **Serve the weights file** from a shell, or let your application start the service instead, as [Start from Python](#start-from-python) shows. [Configuration](#configuration) lists every other setting, such as the GPU to run on:

   ```sh
   JOINTFM_INFERENCE_CHECKPOINT_PATH=/path/to/weights.pt jointfm-inference serve   # stop with Ctrl-C
   ```

5. **Check it** from a second shell in the same environment; this requests every return mode the service advertises:

   ```sh
   jointfm-inference probe --service-url http://127.0.0.1:8080
   ```

6. **Forecast** from Python with [`jointfm-client`](https://github.com/datarobot-oss/joint-client-python), pointing its local selector `JOINTFM_LOCAL_BASE_URL` at `http://127.0.0.1:8080`.

### From a clone

The `task` targets serve from a clone of this repository instead and read the settings from a `.env` file; they also build and run the [container](#container). They need [uv](https://docs.astral.sh/uv/) and [Task](https://taskfile.dev/), and the container needs Docker:

```sh
git clone https://github.com/datarobot-oss/joint-inference-python.git
cd joint-inference-python
cp .env.template .env   # then set the weights file in .env
task serve              # creates the environment on first use; stop with Ctrl-C
task probe              # from a second shell
```

In `.env`, set `JOINTFM_INFERENCE_CHECKPOINT_PATH` to the weights file, or to where it should be downloaded together with `JOINTFM_INFERENCE_CHECKPOINT_URL` and `JOINTFM_INFERENCE_CHECKPOINT_SHA256`.

## Start from Python

An application can serve the weights itself, with this package and the client installed in its own environment, instead of running a separate service. `LocalService` starts the service as a child process of the application's interpreter when its `with` block begins, returns once the service answers its health check, and stops it when the block ends, also when the block raises:

```sh
pip install jointfm-inference jointfm-client
```

```python
from jointfm_client import JointFMClient
from jointfm_inference import LocalService, ServiceSettings

settings = ServiceSettings(checkpoint_path="/path/to/weights.pt", device="cuda:0")
with LocalService(settings) as service:
    client = JointFMClient.from_env(
        env={
            "JOINTFM_LOCAL_BASE_URL": service.base_url,
            "JOINTFM_SCHEMA_VERSION": "v6",
        },
        dotenv_path=None,
        config_path=None,
    )
    print(client.health())  # forecast as the client's README shows
```

- **Settings.** `ServiceSettings` has one field per variable in [Configuration](#configuration), named after it in lower case (`device` for `JOINTFM_INFERENCE_DEVICE`); a field left unset keeps that variable's default. Only these settings configure the service: the application's own `JOINTFM_INFERENCE_*` variables do not reach it. The service checks every value when it starts.
- **Address.** `jointfm-inference serve` listens on every network interface by default, but `LocalService` listens on the loopback address unless `bind_host` says otherwise, and on a free port unless `port` names one; `service.base_url` is where it answers. The service has no authentication, so binding it to another interface opens it to that network.
- **Failures.** Starting raises `LocalServiceError` when the service exits before it answers or does not answer within `ready_timeout_seconds`, by default long enough to download the weights first; the message ends with the service's last lines of output, which say why. That output goes to a temporary file removed when the block ends, or is appended to `log_path`.
- **Stopping.** Leaving the block sends the service SIGTERM and kills it if it is still running after `stop_timeout_seconds`. An application killed without unwinding, as by SIGKILL, leaves the service running.
- **Environment.** This package needs Python 3.13 or newer, the client 3.11 or newer, so the application needs 3.13. Importing `jointfm_inference` loads neither torch nor the model into the application; only the service process does.

## Hardware

These figures are for the JointFM 1 open weights at the default `bf16` precision, rounded up from our estimate for the largest request the model accepts: 100 series, 732 steps of history, and all 92 forecast steps. Smaller requests need less, and next-day forecasts need far less.

**GPU.** Any NVIDIA GPU of the Ampere generation or newer with 16 GB of memory. `bf16` needs native bfloat16 support, and the service refuses to start with `bf16` on a GPU without it; there, set `JOINTFM_INFERENCE_PRECISION=fp32`, which needs more memory. GPU memory grows with the samples, forecast steps, and series a request asks for, because the model runs once per request and draws all of its samples from that run:

| Samples per request (`JOINTFM_INFERENCE_MAX_SAMPLE_COUNT`) | GPU memory for the largest request |
| --- | --- |
| 1,000 (the default) | about 10 GB |
| 5,000 | about 12 GB |
| 20,000 | about 20 GB |

With `JOINTFM_INFERENCE_MAX_CONCURRENT_REQUESTS` above 1, each concurrent request needs that much again, apart from the roughly 2 GB of weights and CUDA context they share.

**RAM.** About 4 GB when the model runs on a GPU, plus about 20 bytes for every value a response returns (samples × forecast steps × series), so about 8 GB for 20,000 samples of the largest request. On the CPU, the model itself runs in RAM: plan for about 16 GB at 1,000 samples and about 32 GB at 20,000.

**Disk.** About 0.7 GB for the weights; a download is written next to the file before it replaces it, so replacing a file briefly needs twice that. About 6 GB for the Python environment on Linux, most of it PyTorch with its CUDA libraries, and about as much again for the [container](#container) image.

## Configuration

`jointfm-inference serve` takes no options: it reads every setting from `JOINTFM_INFERENCE_*` environment variables, and the `Taskfile` targets load them from `.env`, a git-ignored file that [.env.template](https://github.com/datarobot-oss/joint-inference-python/blob/main/.env.template) documents. The checkpoint is either already at `JOINTFM_INFERENCE_CHECKPOINT_PATH` or downloaded there at startup from `JOINTFM_INFERENCE_CHECKPOINT_URL`. A download needs the file's SHA-256 digest: it streams into a temporary file, is verified, and only then moves into place, and a file already at the path is reused only when its digest matches, so changing the URL never serves the old weights. The service stops cleanly on SIGINT or SIGTERM.

| Variable | Default | Meaning |
| --- | --- | --- |
| `JOINTFM_INFERENCE_CHECKPOINT_PATH` | required | Checkpoint file to serve, or to download to |
| `JOINTFM_INFERENCE_CHECKPOINT_VERSION` | first 12 hexadecimal digits of the checkpoint's SHA-256 | Version label `/healthz` reports as `checkpoint_version`; the default changes exactly when the weights do |
| `JOINTFM_INFERENCE_CHECKPOINT_URL` | unset | `https` or `http` URL to download the checkpoint from when the file is absent |
| `JOINTFM_INFERENCE_CHECKPOINT_SHA256` | unset | Lowercase hexadecimal SHA-256 of the checkpoint; required with a URL, refused without one |
| `JOINTFM_INFERENCE_CHECKPOINT_TOKEN` | unset | Bearer token sent with the download; only to an `https` URL |
| `JOINTFM_INFERENCE_DEVICE` | `cpu` | Torch device the model runs on, such as `cuda:0` |
| `JOINTFM_INFERENCE_MAP_LOCATION` | the device | Device the checkpoint tensors are first read onto |
| `JOINTFM_INFERENCE_PRECISION` | `bf16` | Model precision of every forward: `fp32`, `bf16`, or `fp16` |
| `JOINTFM_INFERENCE_BIND_HOST` | `0.0.0.0` | Address the service listens on |
| `JOINTFM_INFERENCE_PORT` | `8080` | Port the service listens on |
| `JOINTFM_INFERENCE_MAX_SAMPLE_COUNT` | `1000` | Largest sample count one request may ask for |
| `JOINTFM_INFERENCE_MAX_REQUEST_BYTES` | admits a request at every limit of the checkpoint's capacity envelope | Largest request body read, which `/healthz` reports as `max_request_bytes`; a larger one is refused unread with HTTP 413 and `INPUT_SIZE_EXCEEDED`. The default budgets each value at full float64 precision keyed by a column name of up to 64 bytes, so raise it for longer names |
| `JOINTFM_INFERENCE_MAX_CONCURRENT_REQUESTS` | `1` | Prediction requests run at the same time; raise it only as far as the largest requests fit the device together |
| `JOINTFM_INFERENCE_MODEL_VERSION` | `jointfm-inference:<version>+ckpt.<checkpoint version>` | Model version `/healthz` reports; a request that names a model version must match it |

A connection that sends or accepts no byte for 60 seconds is closed, a stalled request body with HTTP 408, so a stalled client cannot hold a server thread.

`jointfm-inference probe --service-url URL` (repeatable) checks each service's health and requests every return mode it advertises; `task probe` probes the local service, and `task probe -- --service-url URL` any others.

`jointfm-inference profile` shows where a request's time goes. It loads the checkpoint that the same settings configure into its own process, answers synthetic forecast requests one at a time without HTTP, over a grid of column counts (`--columns`), sample counts (`--samples`), and return modes (`--return-modes`), and prints a Markdown table of median milliseconds per stage: preparation, the network forward (encoder, then pooling and head), the read-out (for samples: Cholesky factorization, Gaussian draws, Gamma draws, and assembly), serialization, and JSON encoding, next to the end-to-end time with the stage timers off. `--output FILE` also writes every timing as JSON, and `task profile -- <options>` runs it from `.env`. Profile on a device no other process uses: a timer waits for all work on its device, so other work there lands in the stages. Library code turns the same timers on with `jointfm_inference.core.timing.set_enabled(True)`.

## HTTP API

The service answers `GET /healthz` and `POST /predict`. [openapi.json](https://github.com/datarobot-oss/joint-inference-python/blob/main/src/jointfm_inference/model/service/wire_schema/openapi.json) is their [OpenAPI 3.1](https://spec.openapis.org/oas/v3.1.0) document: every request, response, health, and error field with its type and meaning, and one example request per query mode and return mode. The service's own code generates it, so it cannot drift from what the service accepts and sends: `jointfm-inference schema` writes it, a commit hook rewrites a stale copy and stops the commit until the rewritten one is staged, and the package ships it as data, so every release tag carries the document of the contract that release serves. Its `info.version` is the `schema_version` every prediction request sends.

## Container

`task docker:build` builds `jointfm-inference:<project version>` from [container/Dockerfile](https://github.com/datarobot-oss/joint-inference-python/blob/main/container/Dockerfile). The image holds the package and its dependencies, never weights: its command is `jointfm-inference serve`, it listens on port 8080, and it expects the checkpoint at `/models/jointfm.pt` unless `JOINTFM_INFERENCE_CHECKPOINT_PATH` says otherwise. `/models` is a volume, so mount a directory there that holds the checkpoint, or one the container may download it into:

```sh
docker run --rm -p 8080:8080 -v "$PWD/models:/models" jointfm-inference:<version>
```

`task docker:run` does this from `.env`: it passes every `JOINTFM_INFERENCE_*` setting on to the container, mounts the directory of `JOINTFM_INFERENCE_CHECKPOINT_PATH` at `/models`, runs as the calling user so a downloaded file belongs to them, and publishes the container's port 8080 at `.env`'s bind host and port. Options after `--` reach `docker run`, such as `task docker:run -- --gpus all` together with `JOINTFM_INFERENCE_DEVICE=cuda`.

## Releases

A release is a tagged version of this repository, uploaded to [PyPI](https://pypi.org/project/jointfm-inference/). Cutting one adds a commit `bump: version <old> → <new>` to `main` that sets `version` in [pyproject.toml](https://github.com/datarobot-oss/joint-inference-python/blob/main/pyproject.toml) and `uv.lock` and adds the new version's entries to `CHANGELOG.md`, and tags that commit `v<new>`. Pushing the tag runs the [Publish to PyPI](https://github.com/datarobot-oss/joint-inference-python/blob/main/.github/workflows/publish.yml) workflow. It uploads the release only after `task build` has built the distribution, installed its wheel into a fresh environment without `uv.lock`, as `pip` would for a user, served a test checkpoint from it, and probed every return mode, and only when the tag names the version it built. The tag, the wheel, and the PyPI version therefore always agree, and a wheel that lacks a module or a dependency never reaches PyPI. No container image is pushed. Install a release with `pip install jointfm-inference==<version>`.

The workflow authenticates through PyPI's [trusted publishing](https://docs.pypi.org/trusted-publishers/), so the repository stores no PyPI token. It needs two settings outside the repository, made once by an owner of the PyPI project and an administrator of the repository:

- a trusted publisher of the PyPI project `jointfm-inference` for owner `datarobot-oss`, repository `joint-inference-python`, workflow `publish.yml`, and environment `pypi`; before the first upload, a [pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) creates the project;
- a GitHub Actions environment named `pypi` on the repository, which may require a reviewer to approve each upload.

### Version numbers

[Commitizen](https://commitizen-tools.github.io/commitizen/) picks the next [Semantic Versioning](https://semver.org/spec/v2.0.0.html) number from the [Conventional Commit](https://www.conventionalcommits.org/) subjects on `main` since the last `v*` tag. Pull requests are squash-merged with their title as the commit subject, so the title decides the bump:

| Commit | Bump |
| --- | --- |
| `feat:` | minor |
| `fix:`, `perf:`, `refactor:` | patch |
| breaking change: `feat!:`, `fix!:`, or a `BREAKING CHANGE:` footer | major, but minor while the version is `0.x` |
| any other type (`docs:`, `build:`, `chore:`, `ci:`, `test:`), or not a Conventional Commit | none, and left out of `CHANGELOG.md` |

A release whose commits drive no bump, such as dependency updates alone, goes out as a patch. `task release -- --increment minor` (or `major`, `patch`) overrides the inferred bump. `cz bump` owns the `version` in `pyproject.toml`; never edit it by hand.

### Cutting a release

Maintainers release from an up-to-date `main` checkout set up with `task setup`, with docker running:

```sh
task release:dry      # preview the next version and its CHANGELOG entries; changes nothing
task release          # commit the bump and tag it, locally
task release:publish  # push main and the new tag to origin
```

`task release` first checks that the tag of the current version exists, the working tree is clean, the branch is `main`, `main` matches `origin/main` after a fetch, and `main` has commits since that tag; any failure stops it before a file changes. It then runs `cz bump`, which writes the new `CHANGELOG.md` section, sets the version in `pyproject.toml`, runs `uv lock` so `uv.lock` records it, commits the three files, and creates the tag. That commit runs the commit hooks like any other, including the license header hook (hence docker) and the full test suite with its coverage floor. When a hook fails there is no commit and no tag, but the three files stay changed and staged; discard those changes before trying again.

Nothing has left the machine at this point. Inspect the commit with `git show`; to cut a different version, drop the tag and the commit (`git tag -d v<new>`, then `git reset --hard origin/main`) and run `task release` again.

`task release:publish` checks again that the tree is clean on `main`, that the tag of the version in `pyproject.toml` points at `HEAD`, and that `origin` does not have that tag yet, since a published tag never moves. It lists the commits it will push, asks for confirmation when run from a terminal, and pushes `main` first, then that one tag. The push to `main` starts the workflows every push to `main` runs; the tag starts Publish to PyPI.

- The release commit goes to `main` directly, while the repository's rules require pull-request review and passing status checks for `main`. GitHub accepts the push only from an account that may bypass those rules.
- When a pull request merged into `main` after `task release`, the push of `main` is rejected as not a fast-forward and nothing is pushed. Drop the tag and the commit as above and cut the release again; rebasing would move the release commit away from its tag, which `task release:publish` refuses.
- When `main` was pushed but the tag push failed, run `task release:publish` again: the checks still pass and only the tag is left to push.
- When Publish to PyPI fails, the tag is published but the version is not on PyPI. Rerun the workflow once a failing setting outside the repository is fixed; when the distribution itself must change, cut a new release, because a published tag never moves.

Commitizen bumps from the tag that matches the version in `pyproject.toml`. Before the first release that tag does not exist yet, so every release task stops and prints the `git tag` command that seeds it on the commit that set the version.

## Contributing

[CONTRIBUTING.md](https://github.com/datarobot-oss/joint-inference-python/blob/main/CONTRIBUTING.md) covers the development setup, every `task` target, the package layout, the checkpoint format, and the checks that keep this public repository free of private material. Rules for contributors and coding agents are in [AGENTS.md](https://github.com/datarobot-oss/joint-inference-python/blob/main/AGENTS.md).

## License

Licensed under the [Apache License, Version 2.0](https://github.com/datarobot-oss/joint-inference-python/blob/main/LICENSE).
