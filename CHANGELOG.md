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
