# -*- coding: utf-8 -*-

from typing import Any

import numpy

from fabopsy_lib import api
from fabopsy_ufanet.core import Detector
from fabopsy_ufanet.five_pts_alignment import face_align_crop

CLASS_MAP = {
    'mae_vit_base_patch16': {
        'expression': [
            "neutral", "anger", "disgust", "fear", "happy", "sad", "surprise"
        ],
        'action_units': [
            "AU1", "AU2", "AU4", "AU5", "AU6", "AU7","AU9", "AU10", "AU12",
            "AU15", "AU17", "AU20","AU23", "AU24", "AU25", "AU26"
        ],
    }
}

class Instance(api.Instance):
    def __init__(self, model_path: str, model_name, device: api.Device):
        # torch_device = torch.device(str(device))
        model = Detector(model_path, model_name=model_name, device=str(device))

        assert model_name in CLASS_MAP
        class_map = CLASS_MAP[model_name]
        assert 'expression' in class_map
        assert 'action_units' in class_map

        self.__model = model
        self.__class_map = class_map

    def inference(self, *,
                  data: dict[str, Any],
                  report: dict[str, Any],
                  **kwargs) -> dict[str, Any]:
        input_data = data['default']
        input_data = numpy.ascontiguousarray(input_data)  # [H, W, C] format, BGR layout

        face_landmarks = report.get('face_landmarks', [])

        face_action_units = []
        face_expression = []
        face_dimensional_affect = []

        for the_landmarks in face_landmarks:
            landmarks = the_landmarks.get('landmarks', [])
            landmarks = numpy.asarray(landmarks).reshape((-1, 2))   # [5, 2]

            face = face_align_crop(input_data, landmarks)
            face_rgb = face[:, :, ::-1]

            cls_pred, au_pred, valence_pred, arousal_pred = self.__model.detect(face_rgb)

            face_action_units.append({ k: v for k, v in zip(self.__class_map['action_units'], au_pred)})
            face_expression.append({ k: v for k, v in zip(self.__class_map['expression'], cls_pred)})
            face_dimensional_affect.append({
                'valence': valence_pred,
                'arousal': arousal_pred,
            })
        return {
            'face_action_units': face_action_units,
            'face_expression': face_expression,
            'face_dimensional_affect': face_dimensional_affect,
        }


class Package(api.Package):
    def create(self, *,
               models: list[api.UsageModel],
               parameters: dict[str, Any],
               device: api.Device | None,
               **kwargs) -> Instance:
        assert len(models) >= 1, api.MissingModelError('At least one model required')

        model_name = models[0].metadata.get('model_name', 'mae_vit_base_patch16')

        model_path = models[0].cache()
        return Instance(
            model_path,
            model_name,
            api.Device('cpu') if device is None else device,
        )


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == '__main__':
    main()
