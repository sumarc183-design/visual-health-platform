# Attributions and licensing

## RETFound (src/models/retfound_backbone.py, src/models/pos_embed.py)

- Repository: https://github.com/rmaphoh/RETFound_MAE
- **License: CC BY-NC 4.0 (Attribution-NonCommercial) — NOT MIT/Apache.**

This means the RETFound code and pretrained weights are free to use for
**research and non-commercial purposes only** (exactly our phase 1:
reproduction and internal benchmarking). They **cannot** be used in a
commercial product or service without a separate license from the
authors.

**Action needed before any commercial launch of Visual Health**: either
(a) negotiate a commercial license with the RETFound authors, or (b) plan
a backbone that is not NC-encumbered — e.g. self-supervised pretraining on
our own fundus data, or another retina foundation model released under a
permissive license (MIT/Apache-2.0). Do not ship RETFound weights inside a
paid product until this is resolved.

Code adapted here is limited to model construction and checkpoint loading
(architecture + interpolate_pos_embed), kept faithful to the original so
official checkpoints load without remapping.
