# SeetaPsych Emo

> SeetaPsych 面部情感识别算法模块，覆盖动作单元、离散表情与效价-唤醒维度情感估计。

简体中文 | [English](README.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](pyproject.toml)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

## 使用方式

本项目已纳入 seetapsych-lib 默认配置，直接通过 `seetapsych-manager download` 即可下载使用。

具体用法请参考 [SeetaPsych](https://github.com/seetapsych/seetapsych-lib) 主库文档。

如需额外加载本项目的算法模块，可通过以下方式：

### WebUI

运行 `seetapsych-webui` 时通过 `--files` 参数加载：

```
seetapsych-webui --files seetapsych_emo/modules/emonet.yml
```

### 编程调用

在代码中添加以下内容以加载并使用本算法模块：

```python
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.pipeline import Pipeline

factory = Factory()
factory.load_file_modules("seetapsych_emo/modules/emonet.yml")

pipeline = Pipeline(factory, ...)

pipeline.add_attributes("face/expression", "face/action_units", "face/dimensional_affect")
```

完整的端到端示例（含可视化效果）参见 [examples/image_emonet.py](examples/image_emonet.py)。

### 模块列表

| YAML 路径 | 算法包 |
|---|---|
| [emonet.yml](seetapsych_emo/modules/emonet.yml) | Emotions-SeetaEmoNet |

### SeetaEmoNet

多任务面部情感估计：动作单元（Action Units, AU）、离散表情分类、连续效价-唤醒（Valence-Arousal, VA）维度情感回归。

<div align="center" id="figure-emonet-result">
  <img src="assets/example-emonet.jpg" alt="SeetaEmoNet 可视化：样本人脸上的动作单元、表情与效价-唤醒维度" height="480"/>
  <p><em><strong>图 1</strong> SeetaEmoNet 输出可视化 — 预测得到的动作单元、表情及效价-唤醒维度。</em></p>
</div>

模块配置：[emonet.yml](seetapsych_emo/modules/emonet.yml)

| 算法包名称 | 提供属性 | 依赖属性 |
|---|---|---|
| Emotions-SeetaEmoNet | `face/action_units`, `face/expression`, `face/dimensional_affect` | `face/landmarks` |

**说明**：统一多任务 MAE-ViT 架构，从 5 点对齐后的人脸裁剪图中同时预测 16 个动作单元强度、7 类离散表情以及效价-唤醒连续维度。

**参数**：*(无)*

**模型**

| 名称 | 推荐 |
|---|---|
| seeta-emo-ufanet-2604.safetensors | ✓ |

**输出属性**
- `face/action_units` — [规格定义](https://github.com/seetapsych/seetapsych-attributes#faceaction_units)。
- `face/expression` — [规格定义](https://github.com/seetapsych/seetapsych-attributes#faceexpression)。
- `face/dimensional_affect` — [规格定义](https://github.com/seetapsych/seetapsych-attributes#facedimensional_affect)。

**输出说明**：
- `face/expression`：7 类基本表情及对应置信度 — neutral（中性）、anger（愤怒）、disgust（厌恶）、fear（恐惧）、happy（高兴）、sad（悲伤）、surprise（惊讶）。
- `face/action_units`：16 个面部动作单元及其强度 — AU1、AU2、AU4、AU5、AU6、AU7、AU9、AU10、AU12、AU15、AU17、AU20、AU23、AU24、AU25、AU26。
- `face/dimensional_affect`：效价与唤醒的连续值，以字典形式返回 `{valence: float, arousal: float}`。
