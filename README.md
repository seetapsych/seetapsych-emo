# SeetaPsych Emotion

> Facial affect/emotion recognition modules for SeetaPsych

## Usage

This project is already included in the seetapsych-lib default configuration. Download and use it via `seetapsych-manager download`.

For usage, refer to [SeetaPsych](https://github.com/seetapsych/seetapsych-lib).

You can additionally add this algorithm module using the following methods.

### WebUI

Run `seetapsych-webui` with the `--dirs` argument to use it.

```
seetapsych-webui --files seetapsych_emo/modules/ufanet.yml
```

### Programmatic Usage

Add the following code in your program to use this algorithm module.

```python
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.pipeline import Pipeline

factory = Factory()
factory.load_file_modules("seetapsych_emo/modules/ufanet.yml")

pipeline = Pipeline(factory, ...)

pipeline.add_attributes("face/expression", "face/action_units", "face/dimensional_affect")
```

## Introduction

### UFANet (Unified Facial Affect Network)

PyTorch-based unified facial affect recognition model built on MAE-ViT-Backbone (mae_vit_base_patch16). Performs 5-point facial alignment and cropping, then simultaneously predicts categorical expressions, Action Units (AUs), and dimensional affect (valence/arousal).

Module config: [ufanet.yml](seetapsych_emo/modules/ufanet.yml).
Provide Attributes: `face/expression`, `face/action_units`, `face/dimensional_affect`.

Requires: `face/landmarks` (5-point facial landmarks for alignment and cropping).

Available model: `seeta-emo-ufanet-2604.safetensors` (recommended).

Output details:
- `face/expression`: 7 basic expression classes with confidence scores — neutral, anger, disgust, fear, happy, sad, surprise.
- `face/action_units`: 16 Facial Action Units with intensities — AU1, AU2, AU4, AU5, AU6, AU7, AU9, AU10, AU12, AU15, AU17, AU20, AU23, AU24, AU25, AU26.
- `face/dimensional_affect`: Continuous valence and arousal values in a dict `{valence: float, arousal: float}`.
