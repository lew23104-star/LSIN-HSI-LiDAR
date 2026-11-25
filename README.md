# Latent Scene Inference Network (LSIN)

LSIN is a framework for joint hyperspectral image (HSI) and LiDAR land-cover
classification. This repository provides a feature-level PyTorch reference
for conditional multiview latent fusion and the observation-decoder routing.
The input is a pair of spatially aligned native feature sequences produced
by sensor-specific encoders. The example uses synthetic inputs and a
lightweight classification head. Reproducing the paper's benchmark results
requires the full encoders, spatial refiner, prediction heads, training
configuration, and checkpoints, which are outside the scope of this release.

`lsin.py` constructs two conditional views, applies a shared Gaussian mapping,
and combines the four factors with receiving-modality grouping, normalised
precision pooling, and cross-group Gaussian fusion. It exposes joint and
single-sensor latent fields, and uses the joint field for centre-pixel
classification. `ObservationDecoders` connects joint reconstruction and
bidirectional cross-modal prediction using the same supplied decoder for each
observation target.

## Quick start

Use Python 3.10 or later. Install the dependency with
`python -m pip install -r requirements.txt`.

```python
import torch
from torch import nn
from lsin import LSIN, ObservationDecoders

model = LSIN(feature_dim=64, num_classes=15, patch_size=5)
hsi = torch.randn(2, 25, 64)
lidar = torch.randn(2, 25, 64)
logits = model(hsi, lidar)  # [2, 15]

# Example observation heads, using synthetic target dimensions.
decoders = ObservationDecoders(
    hsi_centre=nn.Linear(64, 10),
    hsi_relative=nn.Linear(64, 10),
    lidar=nn.Linear(64, 1),
)
observations = decoders(model.latent_fields(hsi, lidar))
```

The inputs follow `[batch, patch positions, feature channels]`. Flatten each
odd square patch in row-major order so the middle token is its centre. A
custom spatial refiner may be supplied as a shape-preserving module on latent
tokens. The observation decoders receive field means before refinement and
return their output in token order, apart from the centre-spectrum output.

## Dataset access

See [DATASETS.md](DATASETS.md) for source links, split sizes, and the fixed
MUUFL training-centre coordinates. Obtain the scene data from the linked
providers under their respective terms.
