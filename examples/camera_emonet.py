# -*- coding: utf-8 -*-
"""Live emotion preview with asynchronous inference and an OpenCV dashboard."""

import os
import platform
import time
from functools import lru_cache
from typing import Any

import cv2
import numpy
from camera_window import CameraWindow, fit_geometry
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.parallel.future import Future
from seetapsych_lib.runtime.parallel_runner import ParallelRunner as Runner
from seetapsych_lib.runtime.pipeline import Pipeline

override_modules = [
    os.path.join(os.path.dirname(__file__), "../seetapsych_emo/modules"),
    os.path.join(os.path.dirname(__file__), "../../seetapsych-face-hub/seetapsych_face_hub/modules"),
]
EXPRESSION_NAMES = ("neutral", "anger", "disgust", "fear", "happy", "sad", "surprise")
ACCENT = (100, 220, 130)
VALENCE_NEGATIVE = (214, 154, 76)
VALENCE_NEUTRAL = (166, 166, 166)
VALENCE_POSITIVE = (75, 172, 242)
AROUSAL_LOW = (115, 105, 110)
AROUSAL_HIGH = (240, 138, 190)


def valence_color(value: float) -> tuple[int, int, int]:
    """Map signed valence to a blue-gray-orange display scale.

    Args:
        value: Valence; values outside [-1, 1] are clamped for display.

    Returns:
        BGR color, with gray at zero and increasing chroma toward either end.
    """
    value = float(numpy.clip(value, -1, 1))
    endpoint = VALENCE_NEGATIVE if value < 0 else VALENCE_POSITIVE
    blue, green, red = (
        int(round(start + abs(value) * (end - start))) for start, end in zip(VALENCE_NEUTRAL, endpoint, strict=True)
    )
    return blue, green, red


def arousal_color(value: float) -> tuple[int, int, int]:
    """Map arousal to an independent muted-to-bright purple scale.

    Args:
        value: Arousal; values outside [-1, 1] are clamped for display.

    Returns:
        BGR color that becomes brighter as activation increases.
    """
    level = (float(numpy.clip(value, -1, 1)) + 1) / 2
    blue, green, red = (
        int(round(start + level * (end - start))) for start, end in zip(AROUSAL_LOW, AROUSAL_HIGH, strict=True)
    )
    return blue, green, red


@lru_cache(maxsize=1)
def va_color_field() -> numpy.ndarray:
    """Return a cached VA legend with valence hue and arousal-dependent vividness.

    Returns:
        Read-only 104 by 128 BGR image; positive valence is right and high arousal is up.
    """
    colors = numpy.asarray([valence_color(float(value)) for value in numpy.linspace(-1, 1, 128)])
    strength = numpy.linspace(1.0, 0.35, 104).reshape(-1, 1, 1)
    field: numpy.ndarray = numpy.rint(112 + strength * (colors[numpy.newaxis, :, :] - 112)).astype(numpy.uint8)
    field.setflags(write=False)
    return field


def open_camera(camera_id: int = 0) -> cv2.VideoCapture:
    """Open a platform camera backend, falling back to automatic selection.

    Args:
        camera_id: Camera device index.

    Returns:
        Capture whose open state must be checked by the caller.
    """
    backend = {
        "Windows": cv2.CAP_DSHOW,
        "Linux": cv2.CAP_V4L2,
        "Darwin": cv2.CAP_AVFOUNDATION,
    }.get(platform.system(), cv2.CAP_ANY)
    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_id)
    return cap


def put_text(
    canvas: numpy.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = (235, 235, 235),
    *,
    font_scale: float = 1.0,
) -> None:
    """Draw anti-aliased text with an identical, pixel-offset shadow.

    Args:
        canvas: BGR destination.
        text: Display text.
        x: Left anchor.
        y: Text baseline.
        color: BGR text color.
        font_scale: UI scale relative to the reference dashboard.
    """
    scale = 0.58 * font_scale
    thickness = max(1, round(font_scale))
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    scale *= min(1.0, max(1, canvas.shape[1] - x - 12) / max(1, text_width))
    offset = max(1, round(font_scale))
    cv2.putText(
        canvas,
        text,
        (x + offset, y + offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_va_panel(canvas: numpy.ndarray, x: int, affect: dict[str, Any] | None, *, ui_scale: float = 1.0) -> None:
    """Draw a VA color legend, numeric values, and an independent arousal meter.

    Colors encode display dimensions, not discrete emotions or clinical categories.
    Numeric values are preserved; only marker coordinates and colors are clamped.

    Args:
        canvas: Dashboard destination, at least ``640 * ui_scale`` pixels tall.
        x: Left edge of the affect panel in viewport pixels.
        affect: Primary-face affect values; missing or non-finite pairs show no marker.
        ui_scale: Scale for geometry and native-resolution text.
    """

    def px(value: float) -> int:
        return x + round(value * ui_scale)

    def py(value: float) -> int:
        return round(value * ui_scale)

    put_text(canvas, "Valence / Arousal", x, py(412), font_scale=ui_scale)
    left, top = px(8), py(438)
    field = va_color_field()
    if ui_scale != 1.0:
        field = cv2.resize(field, (max(1, round(128 * ui_scale)), max(1, round(104 * ui_scale))))
    height, width = field.shape[:2]
    canvas[top : top + height, left : left + width] = field
    center = (left + width // 2, top + height // 2)
    cv2.rectangle(canvas, (left, top), (left + width - 1, top + height - 1), (200, 200, 200), 1)
    cv2.line(canvas, (center[0], top), (center[0], top + height - 1), (200, 200, 200), 1)
    cv2.line(canvas, (left, center[1]), (left + width - 1, center[1]), (200, 200, 200), 1)
    for text, point, color in (
        ("High A", (left, py(426)), (220, 220, 220)),
        ("Low A", (px(54), py(559)), (235, 235, 235)),
        ("-V", (left, py(559)), VALENCE_NEGATIVE),
        ("+V", (px(114), py(559)), VALENCE_POSITIVE),
        ("Low", (px(156), py(548)), AROUSAL_LOW),
        ("High", (px(264), py(548)), AROUSAL_HIGH),
    ):
        cv2.putText(
            canvas,
            text,
            point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4 * ui_scale,
            color,
            max(1, round(ui_scale)),
            cv2.LINE_AA,
        )

    cv2.rectangle(canvas, (px(156), py(522)), (px(296), py(530)), (65, 60, 55), -1)
    valence_raw = affect.get("valence") if affect else None
    arousal_raw = affect.get("arousal") if affect else None
    if valence_raw is None or arousal_raw is None or not numpy.isfinite([valence_raw, arousal_raw]).all():
        put_text(canvas, "V: --", px(156), py(456), VALENCE_NEUTRAL, font_scale=ui_scale)
        put_text(canvas, "Unavailable", px(156), py(482), VALENCE_NEUTRAL, font_scale=ui_scale)
        put_text(canvas, "A: --", px(156), py(510), VALENCE_NEUTRAL, font_scale=ui_scale)
        return

    valence, arousal = float(valence_raw), float(arousal_raw)
    v_color = valence_color(valence)
    a_color = arousal_color(arousal)
    point = (
        left + int(round((float(numpy.clip(valence, -1, 1)) + 1) * (width - 1) / 2)),
        top + int(round((1 - float(numpy.clip(arousal, -1, 1))) * (height - 1) / 2)),
    )
    cv2.circle(canvas, point, max(1, round(8 * ui_scale)), (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(canvas, point, max(1, round(6 * ui_scale)), v_color, -1, cv2.LINE_AA)
    cv2.circle(
        canvas,
        point,
        max(1, round(7 * ui_scale)),
        (255, 255, 255),
        max(1, round(ui_scale)),
        cv2.LINE_AA,
    )
    put_text(canvas, f"V: {valence:+.2f}", px(156), py(456), v_color, font_scale=ui_scale)
    label = "Negative" if valence < 0 else "Positive" if valence > 0 else "Neutral"
    put_text(canvas, label, px(156), py(482), v_color, font_scale=ui_scale)
    put_text(canvas, f"A: {arousal:+.2f}", px(156), py(510), a_color, font_scale=ui_scale)
    filled_width = int(round((float(numpy.clip(arousal, -1, 1)) + 1) * 140 * ui_scale / 2))
    if filled_width > 0:
        cv2.rectangle(canvas, (px(156), py(522)), (px(156) + filled_width, py(530)), a_color, -1)


def draw_results(
    frame: numpy.ndarray, report: dict[str, Any], status: str, display_size: tuple[int, int] | None = None
) -> numpy.ndarray:
    """Render live video alongside first-face expression and affect scores.

    Args:
        frame: Current unmirrored BGR camera frame.
        report: Last completed inference report, or empty before the first result.
        status: Preview and inference timing information.
        display_size: Optional viewport size for native-resolution dashboard rendering.

    Returns:
        Aspect-preserving mirrored preview and primary-face panel, sized to the viewport.
    """
    height, width = frame.shape[:2]
    if display_size is None:
        scale = min(960 / width, 720 / height)
        display_size = (int(width * scale) + 340, max(640, int(height * scale)))
        ui_scale = 1.0
    else:
        ui_scale = min(display_size[1] / 720, display_size[0] / 1000)
    preview_width = max(1, display_size[0] - round(340 * ui_scale))
    left, top, pw, ph = fit_geometry((width, height), (preview_width, display_size[1]))
    interpolation = cv2.INTER_AREA if pw < width else cv2.INTER_CUBIC
    preview = cv2.resize(cv2.flip(frame, 1), (pw, ph), interpolation=interpolation)
    canvas = numpy.full((display_size[1], display_size[0], 3), (28, 25, 23), dtype=numpy.uint8)
    canvas[top : top + ph, left : left + pw] = preview
    scale_x, scale_y = pw / width, ph / height
    expressions = report.get("face_expression", [])
    for index, bbox in enumerate(report.get("face_detection", [])):
        x1, y1, x2, y2 = bbox["xyxy"]
        pt1 = (left + int((width - 1 - x2) * scale_x), top + int(y1 * scale_y))
        pt2 = (left + int((width - 1 - x1) * scale_x), top + int(y2 * scale_y))
        cv2.rectangle(
            canvas, pt1, pt2, ACCENT if index == 0 else (150, 150, 150), max(1, round(2 * ui_scale)), cv2.LINE_AA
        )
        if index < len(expressions) and expressions[index]:
            scores = {
                key: float(value)
                for key, value in expressions[index].items()
                if value is not None and numpy.isfinite(value)
            }
            if scores:
                name = max(scores, key=lambda key: scores[key])
                put_text(
                    canvas,
                    name.capitalize(),
                    max(4, pt1[0]),
                    max(round(22 * ui_scale), pt1[1] - 8),
                    ACCENT,
                    font_scale=ui_scale,
                )

    x = preview_width + round(18 * ui_scale)
    put_text(canvas, "LIVE EMOTION", x, round(30 * ui_scale), ACCENT, font_scale=ui_scale)
    put_text(canvas, "Primary face (green box)", x, round(58 * ui_scale), font_scale=ui_scale)
    put_text(canvas, status, x, round(86 * ui_scale), font_scale=ui_scale)
    values = expressions[0] if expressions else {}
    finite_values = {key: float(value) for key, value in values.items() if value is not None and numpy.isfinite(value)}
    best = max(finite_values, key=lambda key: finite_values[key]) if finite_values else None
    put_text(
        canvas,
        best.capitalize() if best else "Waiting for face / fresh result",
        x,
        round(122 * ui_scale),
        ACCENT,
        font_scale=ui_scale,
    )
    for index, name in enumerate(EXPRESSION_NAMES):
        y = round((158 + index * 34) * ui_scale)
        value = float(numpy.clip(finite_values.get(name, 0), 0, 1))
        color = ACCENT if name == best else (175, 150, 100)
        put_text(canvas, name.capitalize(), x, y, font_scale=ui_scale)
        put_text(canvas, f"{value:.0%}" if values else "--", x + round(248 * ui_scale), y, font_scale=ui_scale)
        cv2.rectangle(
            canvas,
            (x, y + round(6 * ui_scale)),
            (x + round(300 * ui_scale), y + round(12 * ui_scale)),
            (65, 60, 55),
            -1,
        )
        if value > 0:
            cv2.rectangle(
                canvas,
                (x, y + round(6 * ui_scale)),
                (x + int(300 * value * ui_scale), y + round(12 * ui_scale)),
                color,
                -1,
            )

    affect = report.get("face_dimensional_affect", [])
    draw_va_panel(canvas, x, affect[0] if affect else None, ui_scale=ui_scale)
    units = report.get("face_action_units", [])
    if units:
        ranked = sorted(
            ((key, float(value)) for key, value in units[0].items() if value is not None and numpy.isfinite(value)),
            key=lambda pair: pair[1],
            reverse=True,
        )[:4]
        put_text(
            canvas,
            "Top AUs: " + " ".join(key for key, _ in ranked),
            x,
            round(578 * ui_scale),
            font_scale=ui_scale,
        )
        put_text(
            canvas,
            "  ".join(f"{value:.2f}" for _, value in ranked),
            x,
            round(605 * ui_scale),
            font_scale=ui_scale,
        )
    put_text(
        canvas,
        "F = Fullscreen | Esc / Q / X = Exit",
        round(16 * ui_scale),
        canvas.shape[0] - round(16 * ui_scale),
        font_scale=ui_scale,
    )
    return canvas


def run_camera(cap: cv2.VideoCapture) -> None:
    """Keep the preview responsive while a single emotion task runs.

    Completed results remain visible until replaced, including by a no-face result.
    Inference errors propagate and close the preview rather than retaining old scores.

    Args:
        cap: Open camera owned by the caller.
    """
    factory = Factory()
    for root in override_modules:
        factory.load_dir_modules(root)
    pipeline = Pipeline(factory, packages=["2951cff6-46c0-4506-8f6b-ba016a00a35b"], attributes=[])
    pipeline.solve()
    pipeline.install_requirements()
    pipeline.cache_models()
    runner = Runner(pipeline)
    pending: Future[dict[str, Any]] | None = None
    report: dict[str, Any] = {}
    submitted_at = 0.0
    inference_fps = 0.0
    preview_frames = 0
    preview_fps = 0.0
    meter_start = time.monotonic()
    window = "Camera Emotion"
    try:
        display = CameraWindow(window, (1300, 720))
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped delivering frames. Check the connection and camera permissions.")
                break
            now = time.monotonic()
            preview_frames += 1
            if now - meter_start >= 1.0:
                preview_fps = preview_frames / (now - meter_start)
                preview_frames = 0
                meter_start = now
            if pending is not None and pending.done():
                report = pending.get()
                inference_fps = 1.0 / max(now - submitted_at, 1e-6)
                pending = None
            if pending is None:
                submitted_at = now
                pending = runner.run_async(data={"default": frame.copy()}, timestamp=time.time())
            canvas = draw_results(
                frame, report, f"Video {preview_fps:.0f} | AI {inference_fps:.1f} FPS", display.size()
            )
            display.show(canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q"), ord("x"), ord("X")):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            display.handle_key(key)
    finally:
        if pending is not None:
            pending.cancel()
        runner.dispose()


def main() -> None:
    """Open the camera before preparing models and release resources on exit."""
    cap = open_camera(0)
    try:
        if not cap.isOpened():
            print("Could not open camera 0. Check the device index, camera permissions, and other camera apps.")
            return
        run_camera(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
