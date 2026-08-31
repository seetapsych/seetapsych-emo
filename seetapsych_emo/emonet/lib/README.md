# FaboPsy_UFAnet
Unified Facial Affect Network (UFA-Net)
Multi-task facial expression recognition project, supporting single-image inference.

**Tasks include**: Expression Classification (7 classes) · AU Detection (16 AUs) · Valence-Arousal Regression

## Environment Installation
**pip install**
pip install git+https://github.com/lcqin111/fabopsy_ufanet.git

**uv add**
uv add "fabopsy_ufanet @ git+https://github.com/lcqin111/fabopsy_ufanet.git"

## Quick Start
Refer to demo.py

### Face Alignment Method
Refer to fabopsy_ufanet/five_pts_alignment.py

### Detector.detect Documentation
- **Input**: `img` (numpy.array)
- **Output**: Seven basic expression probabilities, 16 AU prediction probabilities, valence and arousal values, in that order
#### Output Field Definitions
Expression class return value is a standard Python list, structure as follows:
["neutral", "anger", "disgust", "fear", "happy", "sad", "surprise"]
AU return value is a standard Python list, structure as follows:
["AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10", "AU12", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26"]
