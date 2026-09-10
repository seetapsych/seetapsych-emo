# SeetaPsych Emo

> Facial affect/emotion recognition modules for SeetaPsych

## Usage

This project is already included in the seetapsych-lib default configuration. Download and use it via `seetapsych-manager download`.

For usage, refer to [SeetaPsych](https://github.com/seetapsych/seetapsych-lib).

You can additionally add this algorithm module using the following methods.

### WebUI

Run `seetapsych-webui` with the `--dirs` argument to use it.

```
seetapsych-webui --files seetapsych_emo/modules/emonet.yml
```

### Programmatic Usage

Add the following code in your program to use this algorithm module.

```python
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.pipeline import Pipeline

factory = Factory()
factory.load_file_modules("seetapsych_emo/modules/emonet.yml")

pipeline = Pipeline(factory, ...)

pipeline.add_attributes("face/expression", "face/action_units", "face/dimensional_affect")
```

For a complete end-to-end example with visualization, see [examples/image_emonet.py](https://github.com/seetapsych/seetapsych-emo/blob/main/examples/image_emonet.py).

### Module Catalog

| YAML Path | Packages |
|---|---|
| [emonet.yml](https://github.com/seetapsych/seetapsych-emo/blob/main/seetapsych_emo/modules/emonet.yml) | Emotions-SeetaEmoNet |

### SeetaEmoNet

Multi-task facial affect estimation: action units, categorical expressions, and continuous valence-arousal dimensions.

<div align="center" id="figure-emonet-result">
  <img src="https://raw.githubusercontent.com/seetapsych/seetapsych-emo/main/assets/example-emonet.jpg" alt="SeetaEmoNet visualization of AUs, expressions and valence-arousal on a sample face" height="480"/>
  <p><em><strong>Figure 1.</strong> SeetaEmoNet output visualization — predicted AUs, expressions, and valence-arousal.</em></p>
</div>

Module config: [emonet.yml](https://github.com/seetapsych/seetapsych-emo/blob/main/seetapsych_emo/modules/emonet.yml)

| Package Name | Provides Attributes | Requires Attributes |
|---|---|---|
| Emotions-SeetaEmoNet | face/action_units, face/expression, face/dimensional_affect | face/landmarks |

**Description**: Unified multi-task MAE-ViT model predicting 16 AUs, 7 expressions, and valence-arousal simultaneously from 5-point aligned face crops

**Parameters**: *(none)*

**Models**

| Name | Recommended |
|---|---|
| seeta-emo-ufanet-2604.safetensors | ✓ |

**Output Attributes**
- `face/action_units` — [spec](https://github.com/seetapsych/seetapsych-attributes#faceaction_units).
- `face/expression` — [spec](https://github.com/seetapsych/seetapsych-attributes#faceexpression).
- `face/dimensional_affect` — [spec](https://github.com/seetapsych/seetapsych-attributes#facedimensional_affect).

**Output details**:
- `face/expression`: 7 basic expression classes with confidence scores — neutral, anger, disgust, fear, happy, sad, surprise.
- `face/action_units`: 16 Facial Action Units with intensities — AU1, AU2, AU4, AU5, AU6, AU7, AU9, AU10, AU12, AU15, AU17, AU20, AU23, AU24, AU25, AU26.
- `face/dimensional_affect`: Continuous valence and arousal values in a dict `{valence: float, arousal: float}`.
