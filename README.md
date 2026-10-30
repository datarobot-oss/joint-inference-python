<p align="center">
  <a href="https://github.com/datarobot-oss/joint-inference-python">
    <img src="https://raw.githubusercontent.com/datarobot-oss/joint-inference-python/main/.github/datarobot_logo.avif" width="600px" alt="DataRobot Logo"/>
  </a>
</p>
<h3 align="center">JointFM Inference Runner</h3>

<p align="center">
  <a href="https://www.datarobot.com/">Homepage</a>
  ·
  <a href="https://github.com/datarobot-oss/joint-client-python">Python client</a>
  ·
  <a href="https://docs.datarobot.com/en/docs/get-started/troubleshooting/general-help.html">Support</a>
</p>

<p align="center">
  <a href="https://github.com/datarobot-oss/joint-inference-python/actions/workflows/ci.yml">
    <img src="https://github.com/datarobot-oss/joint-inference-python/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://github.com/datarobot-oss/joint-inference-python/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/datarobot-oss/joint-inference-python" alt="License">
  </a>
</p>

`jointfm-inference` is the inference runner for JointFM: it loads JointFM checkpoints and serves joint probabilistic forecasts over HTTP. Applications talk to a running instance through the [`jointfm-client`](https://github.com/datarobot-oss/joint-client-python) SDK ([PyPI](https://pypi.org/project/jointfm-client/)).

The repository holds the serving code for JointFM 1 checkpoints, the `jointfm-inference` command, and a container image. Neither the package nor the image contains weights: every service names the checkpoint it serves in its own configuration.

> **Status: pre-alpha.** The HTTP contract and the checkpoint format may change between releases.

## Quickstart

1. **Install the prerequisites:** [uv](https://docs.astral.sh/uv/) and [Task](https://taskfile.dev/). Ubuntu, Amazon Linux, and macOS are supported. Docker is needed only for the [container](#container).
2. **Get a weights file.** Serving needs a JointFM serving weights file: a JointFM 1 checkpoint in the format the runner reads. How to obtain one will be documented here.
3. **Clone the repository:**

   ```sh
   git clone https://github.com/datarobot-oss/joint-inference-python.git
   cd joint-inference-python
   ```

4. **Configure the service.** Copy [.env.template](.env.template) to `.env` and set `JOINTFM_INFERENCE_CHECKPOINT_PATH` to the weights file, or to where it should be downloaded together with `JOINTFM_INFERENCE_CHECKPOINT_URL` and `JOINTFM_INFERENCE_CHECKPOINT_SHA256`. With the [DataRobot CLI](https://github.com/datarobot-oss/cli), `task start` writes `.env` instead (see [Configuration](#configuration)).
5. **Serve it:**

   ```sh
   task serve   # creates the environment on first use; stop with Ctrl-C
   ```

6. **Check it** from a second shell; this requests every return mode the service advertises:

   ```sh
   task probe
   ```

7. **Forecast** from Python with [`jointfm-client`](https://github.com/datarobot-oss/joint-client-python), pointing its local selector `JOINTFM_LOCAL_BASE_URL` at `http://127.0.0.1:8080`.

## Configuration

`jointfm-inference serve` takes no options: it reads every setting from `JOINTFM_INFERENCE_*` environment variables, and the `Taskfile` targets load them from `.env`, a git-ignored file that [.env.template](.env.template) documents. The checkpoint is either already at `JOINTFM_INFERENCE_CHECKPOINT_PATH` or downloaded there at startup from `JOINTFM_INFERENCE_CHECKPOINT_URL`. A download needs the file's SHA-256 digest: it streams into a temporary file, is verified, and only then moves into place, and a file already at the path is reused only when its digest matches, so changing the URL never serves the old weights. The service stops cleanly on SIGINT or SIGTERM.

With the [DataRobot CLI](https://github.com/datarobot-oss/cli), `task start` (or `dr start`) writes `.env` instead of the copy: `dr` logs in to DataRobot, fills in `DATAROBOT_ENDPOINT` and `DATAROBOT_API_TOKEN`, and asks for the checkpoint and the deploy settings that [.datarobot/cli/jointfm.yaml](.datarobot/cli/jointfm.yaml) defines. The optional settings stay documented in `.env.template`; add any of them to `.env` by hand, and `dr` keeps them on later runs.

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
| `JOINTFM_INFERENCE_MAX_CONCURRENT_REQUESTS` | `1` | Prediction requests run at the same time; raise it only as far as the largest requests fit the device together |
| `JOINTFM_INFERENCE_MODEL_VERSION` | `jointfm-inference:<version>+ckpt.<checkpoint version>` | Model version `/healthz` reports; a request that names a model version must match it |

`jointfm-inference probe --service-url URL` (repeatable) checks each service's health and requests every return mode it advertises; `task probe` probes the local service, and `task probe -- --service-url URL` any others.

## Container

`task docker:build` builds `jointfm-inference:<project version>` from [container/Dockerfile](container/Dockerfile). The image holds the package and its dependencies, never weights: its command is `jointfm-inference serve`, it listens on port 8080, and it expects the checkpoint at `/models/jointfm.pt` unless `JOINTFM_INFERENCE_CHECKPOINT_PATH` says otherwise. `/models` is a volume, so mount a directory there that holds the checkpoint, or one the container may download it into:

```sh
docker run --rm -p 8080:8080 -v "$PWD/models:/models" jointfm-inference:<version>
```

`task docker:run` does this from `.env`: it passes every `JOINTFM_INFERENCE_*` setting on to the container, mounts the directory of `JOINTFM_INFERENCE_CHECKPOINT_PATH` at `/models`, runs as the calling user so a downloaded file belongs to them, and publishes the container's port 8080 at `.env`'s bind host and port. Options after `--` reach `docker run`, such as `task docker:run -- --gpus all` together with `JOINTFM_INFERENCE_DEVICE=cuda`.

## Deploying to DataRobot

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

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) covers the development setup, every `task` target, the package layout, the checkpoint format, and the checks that keep this public repository free of private material. Rules for contributors and coding agents are in [AGENTS.md](AGENTS.md).

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
