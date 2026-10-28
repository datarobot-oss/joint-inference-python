# joint-inference-python

`jointfm-inference` is the inference runner for JointFM: it loads JointFM checkpoints and serves joint probabilistic forecasts over HTTP. Applications talk to a running instance through the [`jointfm-client`](https://github.com/datarobot/joint-client-python) SDK ([PyPI](https://pypi.org/project/jointfm-client/)).

The repository holds the serving code for JointFM 1 checkpoints, the `jointfm-inference` command, and a container image. Neither the package nor the image contains weights: every service names the checkpoint it serves in its own configuration.

**Status: pre-alpha.** The HTTP contract and the checkpoint format may change between releases.

Serving needs a JointFM serving weights file: a JointFM 1 checkpoint in the [format below](#checkpoints). How to obtain one will be documented here.

## Layout

| Package | Contents |
|---|---|
| `jointfm_inference.model.service` | HTTP service: environment configuration, checkpoint download, request validation, conditions, scoring, JSON serialization |
| `jointfm_inference.model.inference` | Inference session, backend registry, backend-neutral contract, conditioning, the JointFM 1 serving backend |
| `jointfm_inference.model.jointfm_1` | The JointFM 1 network, its Student-t mixture head, and its checkpoint loader |
| `jointfm_inference.model` | Checkpoint reading, linear algebra, precision, and the command-line probe |
| `jointfm_inference.cli` | The `jointfm-inference` command and its `serve` and `probe` subcommands |
| `jointfm_inference.data` | Canonical samples, request times, DataFrame conversion |

## Checkpoints

A checkpoint is a plain-data mapping saved with `torch.save` and loaded with `weights_only=True`. Its layout is versioned, and the runner reads exactly format version 1 (`jointfm_inference.model.checkpoint`); a file of any other version, or with any entry outside the layout, is refused before a backend is resolved:

| Entry | Holds |
| --- | --- |
| `cfg` | The network sizes; the generation's backend defines the fields |
| `model_state` | The network's state dict, every floating tensor in the dtype `weights_dtype` names |
| `runtime_metadata` | Exactly `generation`, `format_version` (`1`), and `weights_dtype` (`fp32`, `bf16`, or `fp16`; loading upcasts to float32) |
| `inference_metadata` | Exactly `time_index_support` and `data_generation`, the capacity envelope that `/healthz` reports |

`runtime_metadata.generation` names the generation that wrote the file; the session resolves that marker to a serving backend through the `jointfm_inference.backends` entry-point group and fails at load when no installed distribution registers it. This distribution registers `jointfm-1`.

## Serving

```sh
cp .env.template .env   # name the checkpoint to serve, then:
task serve              # serve it on this host
task probe              # check /healthz and every advertised return mode
```

`jointfm-inference serve` takes no options: it reads every setting from `JOINTFM_INFERENCE_*` environment variables, and the `Taskfile` targets load them from `.env`, a git-ignored file that [.env.template](.env.template) documents. The checkpoint is either already at `JOINTFM_INFERENCE_CHECKPOINT_PATH` or downloaded there at startup from `JOINTFM_INFERENCE_CHECKPOINT_URL`. A download needs the file's SHA-256 digest: it streams into a temporary file, is verified, and only then moves into place, and a file already at the path is reused only when its digest matches, so changing the URL never serves the old weights. The service reports the first 12 hexadecimal digits of the checkpoint's SHA-256 digest as `checkpoint_version`, so the version changes exactly when the weights do. The service stops cleanly on SIGINT or SIGTERM.

With the [DataRobot CLI](https://github.com/datarobot-oss/cli), `task start` (or `dr start`) writes `.env` instead of the copy: `dr` logs in to DataRobot, fills in `DATAROBOT_ENDPOINT` and `DATAROBOT_API_TOKEN`, and asks for the checkpoint and the deploy settings that [.datarobot/cli/jointfm.yaml](.datarobot/cli/jointfm.yaml) defines. The optional settings stay documented in `.env.template`; add any of them to `.env` by hand, and `dr` keeps them on later runs.

| Variable | Default | Meaning |
| --- | --- | --- |
| `JOINTFM_INFERENCE_CHECKPOINT_PATH` | required | Checkpoint file to serve, or to download to |
| `JOINTFM_INFERENCE_CHECKPOINT_URL` | unset | `https` or `http` URL to download the checkpoint from when the file is absent |
| `JOINTFM_INFERENCE_CHECKPOINT_SHA256` | unset | Lowercase hexadecimal SHA-256 of the checkpoint; required with a URL, refused without one |
| `JOINTFM_INFERENCE_CHECKPOINT_TOKEN` | unset | Bearer token sent with the download; only to an `https` URL |
| `JOINTFM_INFERENCE_DEVICE` | `cpu` | Torch device the model runs on, such as `cuda:0` |
| `JOINTFM_INFERENCE_MAP_LOCATION` | the device | Device the checkpoint tensors are first read onto |
| `JOINTFM_INFERENCE_PRECISION` | `bf16` | Model precision of every forward: `fp32`, `bf16`, or `fp16` |
| `JOINTFM_INFERENCE_BIND_HOST` | `0.0.0.0` | Address the service listens on |
| `JOINTFM_INFERENCE_PORT` | `8080` | Port the service listens on |
| `JOINTFM_INFERENCE_MAX_SAMPLE_COUNT` | `1000` | Largest sample count one request may ask for |
| `JOINTFM_INFERENCE_MAX_CONCURRENT_REQUESTS` | `1` | Prediction requests run at the same time; raise it only as far as the largest requests fit the device together |
| `JOINTFM_INFERENCE_MODEL_VERSION` | `jointfm-inference:<version>+ckpt.<checkpoint version>` | Model version `/healthz` reports; a request that names a model version must match it |

`jointfm-inference probe --service-url URL` (repeatable) checks each service's health and requests every return mode it advertises; `task probe` probes the local service, and `task probe -- --service-url URL` any others.

### Container

`task docker:build` builds `jointfm-inference:<project version>` from [container/Dockerfile](container/Dockerfile). The image holds the package and its dependencies, never weights: its command is `jointfm-inference serve`, it listens on port 8080, and it expects the checkpoint at `/models/jointfm.pt` unless `JOINTFM_INFERENCE_CHECKPOINT_PATH` says otherwise. `/models` is a volume, so mount a directory there that holds the checkpoint, or one the container may download it into:

```sh
docker run --rm -p 8080:8080 -v "$PWD/models:/models" jointfm-inference:<version>
```

`task docker:run` does this from `.env`: it passes every `JOINTFM_INFERENCE_*` setting on to the container, mounts the directory of `JOINTFM_INFERENCE_CHECKPOINT_PATH` at `/models`, runs as the calling user so a downloaded file belongs to them, and publishes the container's port 8080 at `.env`'s bind host and port. Options after `--` reach `docker run`, such as `task docker:run -- --gpus all` together with `JOINTFM_INFERENCE_DEVICE=cuda`.

### Deploying to DataRobot

The same `.env` that `task serve` reads also deploys the service to DataRobot as a Workload, a container DataRobot runs and scales. Fill in the deploy section of [.env.template](.env.template) (a DataRobot API endpoint and token, a Pulumi passphrase, and optionally a resource bundle), install the [Pulumi CLI](https://www.pulumi.com/docs/install/), and run:

```sh
task deploy:bundles   # list the resource bundles the cluster offers
task deploy:preview   # show what would be created or changed
task deploy           # build the image in DataRobot and start the Workload
task deploy:probe     # check /healthz and every return mode through DataRobot
task deploy:destroy   # delete the Workload and its image
```

The [Pulumi program](deploy/stack.py) needs neither docker nor an image registry: it stages a build context under `deploy/build/` and DataRobot builds the image from it. That context is the [container](container/Dockerfile) build plus one [extra stage](deploy/bundled.Dockerfile) that copies the checkpoint at `JOINTFM_INFERENCE_CHECKPOINT_PATH` into the image, so the checkpoint must be a file on the deploying machine; the download settings are not used. The optional service settings of `.env` pass through to the container.

`JOINTFM_DEPLOY_BUNDLE` picks the node each replica runs on, from the cluster's catalog of resource bundles (default `cpu.5xlarge`, 2 CPU and 8 GiB); `task deploy:bundles` lists them. A bundle with a GPU runs the model on CUDA: the deploy sets `JOINTFM_INFERENCE_DEVICE=cuda`, and refuses a device in `.env` that disagrees with the bundle. `JOINTFM_DEPLOY_CPU`, `JOINTFM_DEPLOY_MEMORY`, and `JOINTFM_DEPLOY_GPU` optionally narrow the container's share of the bundle, and the deploy refuses an override the bundle cannot hold, before it stages or uploads anything. `JOINTFM_DEPLOY_REPLICAS` sets the replica count. Both routes require a DataRobot API token as a bearer token: `{"Authorization": "Bearer <token>"}` on requests to `<endpoint>/healthz` and `<endpoint>/predict`, where `task deploy:outputs` prints the endpoint.

Pulumi records what it created in a state backend, by default the local `~/.pulumi`; set `PULUMI_BACKEND_URL` to share state between machines. `JOINTFM_DEPLOY_STACK` names the stack, and each stack is one Workload.

## Development

Prerequisites: [uv](https://docs.astral.sh/uv/), [Task](https://taskfile.dev/), and docker for the license header hook. Ubuntu, Amazon Linux, and macOS are supported.

```sh
task setup        # create .venv, install dependencies, pre-commit hooks, ripgrep, and typos
task pre-commit   # run every check that gates a commit
```

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

Every task prints an `OK:` or `FAIL:` line. CI runs `task pre-commit` and `task build` on Ubuntu and macOS; the macOS job skips the license header hook because its runners have no docker.

### Leak guard

This repository is published, so `scripts/leak_guard.py` fails on material that must stay private: proprietary file headers, DataRobot object IDs, internal hosts and private addresses, links to non-public repositories, planning labels, names of unreleased model generations, and credential shapes. It runs as a pre-commit hook over the working tree, as a `commit-msg` hook over each message, and in CI over the full history, because a published history cannot be cleaned.

Some rules cannot be published because they name what they guard. Maintainers keep those in a git-ignored `.leak-patterns.private` at the repository root: one Python regular expression per line, blank lines and `#` comment lines skipped. The guard reports whether it loaded the file and runs the public rules alone when it is absent, as in CI. A dangling link fails. A passing guard does not prove a change safe to publish.

Credentials are a separate check: the `gitleaks` pre-commit hook scans each staged change, and `task secret-scan:history` scans every reachable commit in CI.

Known vulnerabilities are a third: `task vuln-scan` checks every package pinned in `uv.lock` against the OSV database on each pull request, each push to `main`, and weekly, since advisories land without a code change. [osv-scanner.toml](osv-scanner.toml) holds the waivers, each with an expiry date, for fixes that are still too new for the `exclude-newer` cooldown in `pyproject.toml`.

Rules for contributors and coding agents are in [AGENTS.md](AGENTS.md).

## License

Licensed under the [Apache License, Version 2.0](LICENSE). Every Python file carries the license header that [.licenserc.yaml](.licenserc.yaml) defines; the `insert-license` pre-commit hook adds it with [skywalking-eyes](https://github.com/apache/skywalking-eyes), which runs as a docker image.
