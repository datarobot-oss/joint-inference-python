# Backward compatibility tests

People hold JointFM 1 checkpoints exported for earlier releases of this runner. These tests prove
that every one of those checkpoint layouts still loads and still predicts what it did. They run as
their own check, **Backward compatibility**, so a failure is named for what it is.

Run them with `task test:compatibility`. `task test` and `task coverage` leave them out, and the
`backward-compatibility` pre-commit hook runs them locally.

## The frozen checkpoints

`jointfm_1_checkpoints/` holds tiny JointFM 1 checkpoints with random weights. Each was written by
the release that introduced its layout, and none is ever regenerated: a regenerated file is written
by the current code, so it proves nothing about older files.

| File | Holds |
|---|---|
| `<layout>.pt` | A complete checkpoint in that layout: `cfg`, `model_state`, `runtime_metadata`, and `inference_metadata` |
| `<layout>.outputs.pt` | For a supported layout: a fixed input (`past`, `horizons`) and the network outputs it produced when frozen |

`test_jointfm_1_checkpoints.py` lists each layout as `SUPPORTED` (must load and match its frozen
outputs) or `RETIRED` (must be refused for its format version).

| Layout | Written by | Status |
|---|---|---|
| `column-equivariant` | `v0.3.0` and later | Supported |
| `column-slot` | `v0.2.0` and earlier | Retired before this policy, so it still carries format version 1; its refusal test is an expected failure |

## When a test fails

The change breaks a checkpoint someone may hold. Do not edit the frozen files, the expected outputs,
or the `SUPPORTED` list to make the test pass. Either keep the old layout loading, or treat the
change as a deprecation and follow "Compatibility of released exports" in `CONTRIBUTING.md`, which
needs the maintainers' approval, a higher `CHECKPOINT_FORMAT_VERSION`, a clear refusal, and notice.

## Freezing a new layout

A change that adds a layout, while keeping the old ones loading, freezes the new one in the same
pull request. Write it with the code of that change:

```sh
task run -- python -c "
import torch
from tests.support.models import checkpoint_mapping, tiny_model
model = tiny_model(seed=0)
torch.save(checkpoint_mapping(model), 'tests/compatibility/jointfm_1_checkpoints/<layout>.pt')
generator = torch.Generator().manual_seed(0)
past, horizons = torch.randn(2, 8, 3, generator=generator), torch.arange(4)
with torch.no_grad():
    outputs = model(past, horizons)
torch.save({'past': past, 'horizons': horizons, 'outputs': dict(outputs)},
           'tests/compatibility/jointfm_1_checkpoints/<layout>.outputs.pt')
"
```

Then add the layout to `SUPPORTED` with the release that first writes it.
