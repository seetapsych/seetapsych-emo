import numpy as np
from .matlab_cp2tform import get_similarity_transform_for_cv2
import cv2


N_POINT = 5
# 96x112 --> 112 x 112 --> x+=8
# REFERENCE_FACIAL_POINTS = np.array([
#     [(30.29459953+8)*2,  (51.69630051-8)*2],
#     [(65.53179932+8)*2,  (51.50139999-8)*2],
#     [(48.02519989+8)*2,  (71.73660278-8)*2],
#     [(33.54930115+8)*2,  (92.3655014-8)*2],
#     [(62.72990036+8)*2,  (92.20410156-8)*2]
# ])


REFERENCE_FACIAL_POINTS = np.array([
    [89.3095,  72.9025],
    [169.3095,  72.9025 ],
    [127.8949,  127.0441],
    [96.8796,  184.8907],
    [159.1065,  184.7601]
])

CROP_SIZE = 256

def crop_align(img, location):
    tfm = get_similarity_transform_for_cv2(location,REFERENCE_FACIAL_POINTS)
    src_img = img
    # Use border pixel replication to fill pixels beyond source image range
    # dst_img = cv2.warpAffine(src_img, tfm, (CROP_SIZE,CROP_SIZE),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    # Use black fill
    dst_img = cv2.warpAffine(src_img, tfm, (CROP_SIZE,CROP_SIZE),flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return dst_img