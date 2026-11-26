## v0.11.0 (2026-11-26)

### Feat

- Answer a server failure with a generic 500 INTERNAL_ERROR whose message names only a reference, and log its traceback under that reference instead of returning the exception's text, which can name files, devices, or memory state, catch failures of GET /healthz at the same boundary, send the service's log records from INFO up to stderr and, with the new JOINTFM_INFERENCE_LOG_DIR, to a jointfm-inference.log file that rolls over at 32 MiB and keeps 3 files, configure the log before the checkpoint loads so download and load records reach it, and mount that directory at /logs in task docker:run

## v0.10.2 (2026-11-25)

### Fix

- Refuse a request with more columns than the checkpoint's max_series with a 400 INPUT_SIZE_EXCEEDED on columns that carries the requested and allowed counts, before any descriptor is parsed, instead of an INTERNAL_ERROR from the network's column check that clients retried, describe min_series as the fewest columns a training sample carried, which limits no request, and state the column refusal in the wire schema's error, error value, and columns descriptions

## v0.10.1 (2026-11-21)

### Fix

- Refuse a sample bound that redraws cannot meet with a 400 VALIDATION_ERROR on its column's lower_bound or upper_bound instead of an INTERNAL_ERROR, raise UnmetSampleBoundError from PredictorAdapter.sample naming that bound, and document in the wire schema how bounds redraw, what a seed reproduces, and what parallel_dense decoding means

## v0.10.0 (2026-11-20)

### Feat

- Start an inference service from Python with LocalService and ServiceSettings, and require torch>=2.15.0 instead of exactly 2.15.0

## v0.9.0 (2026-11-20)

### Feat

- Serve locally and from the container image

## v0.8.0 (2026-11-18)

### Feat

- Bound request bodies by a max_request_bytes cap that /healthz reports, refusing a larger or negative Content-Length before reading and a stalled body with 408, generate an OpenAPI 3.1 document of the HTTP contract with jointfm-inference schema, and carry requested and allowed on INPUT_SIZE_EXCEEDED errors

## v0.7.1 (2026-11-14)

### Fix

- Refuse a pinned value too far out to condition on as a validation error that names the cause, instead of non-finite parameters or an internal error

## v0.7.0 (2026-11-14)

### Feat

- Serve JointFM 1 heads with a two-sided per-column sinh-arcsinh tail map, covering sampling, density, exact means and quantiles, and equality and interval conditioning, and report the map as tail_map in /healthz

## v0.6.0 (2026-11-12)

### Feat

- Add off-by-default per-stage timers to served requests and a profile command that reports median stage latency over column counts, sample counts, and return modes

### Fix

- Bound the total Cholesky jitter so nearly determined conditional covariances factorize, and refuse non-finite matrices without retrying

## v0.5.0 (2026-11-09)

### Feat

- Answer JointFM 1 means and quantiles exactly and add sign-symmetric prediction

## v0.4.0 (2026-11-09)

### Feat

- Reject flat columns (#29)
- add guidelines for on/off settings in configuration
- add stats task to show lines added and removed by commits over various timeframes

## v0.3.0 (2026-11-06)

### BREAKING CHANGE

- JointFM 1 checkpoints exported for `v0.2.0` and earlier no longer load. They still carry format version 1, so the load fails on their parameter names and shapes. Re-export the checkpoint, or serve it with `v0.2.0`.

### Feat

- add release management tasks and update pyproject.toml for changelog handling (#23)
- Make the JointFM 1 network column-equivariant (#22)
- Updating CUDA (#10)
- Updating CUDA
- Add CODEOWNERS file to define repository code owners

## v0.2.0 (2026-11-06)

### Feat

- Switch JointFM 1 Student-t df readout to log scale and release 0.2.0
- Add Apache 2.0 License headers to all test files and source files
- Add JOINTFM_SCHEMA_VERSION to .env.sample for client compatibility
- Initial inference logic
- Enhance leak guard to support revision scanning for commit history
- Implement leak guard to prevent publishing private material in commits and history
- Scaffold the jointfm-inference project tooling

### Refactor

- Refactor inference service and checkpoint handling
