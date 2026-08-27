# FaboPsy_UFAnet
Unified Facial Affect Network (UFA-Net)
面部表情多任务识别项目，支持单图推理。

**任务包括**：表情分类（7类） · AU 检测（16个） · 效价-唤醒度回归

## 环境依赖安装
**pip安装**
pip install git+https://github.com/lcqin111/fabopsy_ufanet.git

**uv安装**
uv add "fabopsy_ufanet @ git+https://github.com/lcqin111/fabopsy_ufanet.git"

## 快速开始
参考demo.py

### 人脸对齐方法
参考fabopsy_ufanet/five_pts_alignment.py

### Detector.detect说明
- **输入**：`img` (numpy.array)
- **输出**：依次为七个基本表情概率、16个AU预测概率、valence 与 arousal
#### 输出字段定义
表情类别返回值为标准 Python 列表，具体结构如下：
["neutral", "anger", "disgust", "fear", "happy", "sad", "surprise"]
AU返回值为标准 Python 列表，具体结构如下：
["AU1", "AU2", "AU4", "AU5", "AU6", "AU7","AU9", "AU10", "AU12", "AU15", "AU17", "AU20","AU23", "AU24", "AU25", "AU26"]
