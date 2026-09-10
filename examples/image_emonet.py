# -*- coding: utf-8 -*-

import json
import os
from typing import Any

import cv2
import numpy
from seetapsych_lib.runtime.factory import Factory

# from seetapsych_lib.runtime.runner import Runner
from seetapsych_lib.runtime.parallel_runner import ParallelRunner as Runner
from seetapsych_lib.runtime.pipeline import Pipeline

override_modules = [
    os.path.join(os.path.dirname(__file__), "../../seetapsych-emo/seetapsych_emo/modules"),
    os.path.join(os.path.dirname(__file__), "../../seetapsych-face-hub/seetapsych_face_hub/modules"),
]

image_path = os.path.join(os.path.dirname(__file__), "michael-dam.jpg")


def _display_label(key: str) -> str:
    """Return display-ready key: AU tokens pass through, others title-cased.

    Args:
        key: Raw attribute key (e.g. "AU1", "happy").

    Returns:
        Formatted label string.
    """
    if key.startswith("AU"):
        return key
    return key.capitalize()


# Per-category text colors — OpenCV BGR order.
COLOR_ACTION_UNITS: tuple[int, int, int] = (56, 158, 249)
COLOR_EXPRESSION: tuple[int, int, int] = (218, 148, 75)
COLOR_DIMENSIONAL_AFFECT: tuple[int, int, int] = (122, 163, 59)

# Face bbox visual styles — index 0 is the primary face whose attributes are rendered.
COLOR_FACE_PRIMARY: tuple[int, int, int] = (0, 255, 0)
COLOR_FACE_SECONDARY: tuple[int, int, int] = (255, 0, 255)
LINE_STYLE_SOLID: int = cv2.LINE_8
LINE_STYLE_DASH_GAP: int = 8


def fit_image(image: numpy.ndarray, max_width: int = 1280, max_height: int = 960) -> tuple[numpy.ndarray, float]:
    """Proportionally downscale image to fit within max dimensions.

    Args:
        image: Input BGR image.
        max_width: Width cap in pixels.
        max_height: Height cap in pixels.

    Returns:
        Resized image and applied scale (<= 1.0; 1.0 if no resize occurred).
    """
    h, w = image.shape[:2]
    if w <= max_width and h <= max_height:
        return image, 1.0
    scale = min(max_width / w, max_height / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def put_text_with_shadow(
    image: numpy.ndarray,
    text: str,
    org: tuple[int, int],
    font: int,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    *,
    shadow_color: tuple[int, int, int] = (96, 96, 96),
    shadow_base_offset_px: int = 1,
    line_type: int = cv2.LINE_AA,
) -> None:
    """Draw text with a down-right shadow that scales with font size.

    Args:
        image: Target image (in-place).
        text: String to render.
        org: Top-left anchor (x, y) of primary text.
        font: OpenCV FONT_HERSHEY_* constant.
        font_scale: Font scale, same semantics as ``cv2.putText``.
        color: Primary text BGR color.
        thickness: Stroke thickness for shadow and text.
        shadow_color: Shadow BGR color. Defaults to mid-gray.
        shadow_base_offset_px: Shadow offset at ``font_scale == 1.0``.
            Actual offset = ``round(base_offset * max(font_scale, 1.0))``.
        line_type: ``cv2.putText`` line type. Defaults to anti-aliased.
    """
    offset = int(round(shadow_base_offset_px * max(font_scale, 1.0)))
    x, y = org
    cv2.putText(image, text, (x + offset, y + offset), font, font_scale, shadow_color, thickness, line_type)
    cv2.putText(image, text, org, font, font_scale, color, thickness, line_type)


def draw_rectangle_dashed(
    image: numpy.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    gap: int = LINE_STYLE_DASH_GAP,
) -> None:
    """Draw a dashed rectangle by alternating line segments on each edge.

    OpenCV has no native dashed rectangle primitive; this helper walks each
    edge in ``gap``-length strides and alternates between draw and skip.

    Args:
        image: Target image (in-place).
        pt1: Top-left corner (x, y).
        pt2: Bottom-right corner (x, y).
        color: BGR stroke color.
        thickness: Line thickness.
        gap: On/off segment length in pixels.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    points: list[tuple[tuple[int, int], tuple[int, int]]] = []
    # Top edge: left -> right
    points.extend(((x, y1), (min(x + gap, x2), y1)) for x in range(x1, x2, gap * 2))
    # Bottom edge: left -> right
    points.extend(((x, y2), (min(x + gap, x2), y2)) for x in range(x1, x2, gap * 2))
    # Left edge: top -> bottom
    points.extend(((x1, y), (x1, min(y + gap, y2))) for y in range(y1, y2, gap * 2))
    # Right edge: top -> bottom
    points.extend(((x2, y), (x2, min(y + gap, y2))) for y in range(y1, y2, gap * 2))
    for a, b in points:
        cv2.line(image, a, b, color, thickness, lineType=LINE_STYLE_SOLID)


def draw_results(image: numpy.ndarray, report: dict[str, Any]) -> numpy.ndarray:
    """Render face bboxes and affect results onto a resized canvas.

    Multi-face bbox styling convention:
    * Face 0 (primary) — green solid rectangle; its values appear top-left.
    * Face N (N >= 1) — orange dashed rectangle.

    Args:
        image: Input BGR image.
        report: Algorithm result dict with per-face lists (index 0 = primary):
            ``face_detection`` (``xyxy``: list[4 float]),
            ``face_action_units``, ``face_expression``, ``face_dimensional_affect``.

    Returns:
        Annotated BGR image.
    """
    vis, scale = fit_image(image)

    face_detection = report.get("face_detection", [])
    face_action_units = report.get("face_action_units", [])
    face_expression = report.get("face_expression", [])
    face_dimensional_affect = report.get("face_dimensional_affect", [])

    for face_idx, bbox in enumerate(face_detection):
        xyxy = [int(round(v * scale)) for v in bbox["xyxy"]]
        pt1, pt2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
        if face_idx == 0:
            cv2.rectangle(vis, pt1, pt2, COLOR_FACE_PRIMARY, 2, lineType=LINE_STYLE_SOLID)
        else:
            draw_rectangle_dashed(vis, pt1, pt2, COLOR_FACE_SECONDARY, thickness=2)

    # Attribute panel intentionally shows only the first face; canvas layout
    # cannot legibly accommodate additional subjects without occlusion.
    flat_items: list[tuple[str, float, tuple[int, int, int]]] = []
    if face_action_units:
        flat_items.extend((k, v, COLOR_ACTION_UNITS) for k, v in face_action_units[0].items() if v is not None)
    if face_expression:
        flat_items.extend((k, v, COLOR_EXPRESSION) for k, v in face_expression[0].items() if v is not None)
    if face_dimensional_affect:
        flat_items.extend(
            (k, v, COLOR_DIMENSIONAL_AFFECT) for k, v in face_dimensional_affect[0].items() if v is not None
        )

    if flat_items:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (_, text_h), _ = cv2.getTextSize("AU00: 0.00", font, font_scale, thickness)
        line_spacing = int(text_h * 1.5)

        x = 16
        y = text_h + 16

        for key, val, color in flat_items:
            text = f"{_display_label(key)}: {val:.2f}"
            put_text_with_shadow(
                vis,
                text,
                (x, y),
                font,
                font_scale,
                color,
                thickness,
                shadow_base_offset_px=2,
            )
            y += line_spacing

    return vis


def main():
    factory = Factory()
    # load_dir_modules tolerates missing dirs — example runs fine without local overrides.
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

    vis = draw_results(image, report)

    stem, ext = os.path.splitext(image_path)
    out_path = f"{stem}_result{ext}"
    cv2.imwrite(out_path, vis)
    print(f"Saved result to: {out_path}")

    cv2.imshow("emo", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    runner.dispose()


if __name__ == "__main__":
    main()
