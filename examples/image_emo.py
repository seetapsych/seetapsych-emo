# -*- coding: utf-8 -*-

import json
import os

import cv2
from seetapsych_lib.runtime.factory import Factory

# from seetapsych_lib.runtime.runner import Runner
from seetapsych_lib.runtime.parallel_runner import ParallelRunner as Runner
from seetapsych_lib.runtime.pipeline import Pipeline

override_modules = [
    os.path.join(os.path.dirname(__file__), "../../seetapsych-emo/seetapsych_emo/modules"),
    os.path.join(os.path.dirname(__file__), "../../seetapsych-face-hub/seetapsych_face_hub/modules"),
]

image_path = os.path.join(os.path.dirname(__file__), "michael-dam.jpg")


def main():
    factory = Factory()
    # Attempt to override default modules with local module files, can still run even if no local files exist
    for root in override_modules:
        factory.load_dir_modules(root)

    pipeline = Pipeline(
        factory,
        packages=[
            "2951cff6-46c0-4506-8f6b-ba016a00a35b",  # Emotions-SeetaEmoNet
        ],
        attributes=[
            # 'face/expression',
            # 'face/action_units',
            # 'face/dimensional_affect',
        ],
    )

    # print(pipeline.problem())
    pipeline.solve()

    # print(pipeline.satisfied())
    pipeline.install_requirements()
    pipeline.cache_models()

    runner = Runner(pipeline)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    report = runner.run(data={"default": image})
    print(json.dumps(report, indent=2, ensure_ascii=False))

    runner.dispose()


if __name__ == "__main__":
    main()
