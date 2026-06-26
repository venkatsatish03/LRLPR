from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> int:
    args = parse_args()
    model_path = Path(args.model or settings.YOLO_MODEL_PATH)
    image_paths = find_images(Path(args.images))

    if not model_path.exists():
        print(json.dumps({"error": f"YOLO model not found: {model_path}"}, indent=2))
        return 2

    if not image_paths:
        print(json.dumps({"error": f"No images found under: {args.images}"}, indent=2))
        return 2

    start_load = time.perf_counter()
    model = YOLO(str(model_path))
    model_load_seconds = time.perf_counter() - start_load
    classes = resolve_classes(model, args.classes, args.class_names)

    total_predictions = 0
    total_inference_seconds = 0.0
    prediction_records = {}

    for image_path in image_paths:
        start = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            classes=classes,
            agnostic_nms=args.agnostic_nms,
            verbose=False,
        )[0]
        elapsed = time.perf_counter() - start
        total_inference_seconds += elapsed

        predictions = extract_predictions(result)
        total_predictions += len(predictions)
        prediction_records[image_path.stem] = predictions

    metrics = {
        "model_path": str(model_path),
        "images": len(image_paths),
        "predictions": total_predictions,
        "model_load_seconds": round(model_load_seconds, 4),
        "total_inference_seconds": round(total_inference_seconds, 4),
        "avg_inference_ms": round((total_inference_seconds / len(image_paths)) * 1000, 3),
        "images_per_second": round(len(image_paths) / max(total_inference_seconds, 0.0001), 3),
        "settings": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "max_det": args.max_det,
            "classes": classes,
            "agnostic_nms": args.agnostic_nms,
            "eval_iou": args.eval_iou,
        },
    }

    if args.labels:
        metrics["accuracy"] = evaluate_accuracy(
            prediction_records=prediction_records,
            image_paths=image_paths,
            labels_dir=Path(args.labels),
            iou_threshold=args.eval_iou,
        )
    else:
        metrics["accuracy"] = {
            "status": "skipped",
            "reason": "Provide --labels with YOLO-format .txt labels to compute precision/recall.",
        }

    output = json.dumps(metrics, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO license plate detection speed and accuracy.")
    parser.add_argument("--model", default=str(settings.YOLO_MODEL_PATH), help="Path to YOLO .pt weights.")
    parser.add_argument("--images", required=True, help="Directory containing benchmark images.")
    parser.add_argument("--labels", help="Optional directory with YOLO-format label .txt files.")
    parser.add_argument("--conf", type=float, default=settings.YOLO_CONFIDENCE_THRESHOLD)
    parser.add_argument("--iou", type=float, default=settings.YOLO_NMS_IOU_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=settings.YOLO_IMAGE_SIZE)
    parser.add_argument("--max-det", type=int, default=settings.YOLO_MAX_DETECTIONS)
    parser.add_argument("--classes", default=settings.YOLO_CLASSES)
    parser.add_argument("--class-names", default=settings.YOLO_CLASS_NAMES)
    parser.add_argument("--agnostic-nms", action="store_true", default=settings.YOLO_AGNOSTIC_NMS)
    parser.add_argument("--eval-iou", type=float, default=0.5, help="IoU threshold for accuracy matching.")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args()


def find_images(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_classes(model: YOLO, raw_classes: str, raw_class_names: str) -> list[int] | None:
    class_ids = []
    for raw_value in raw_classes.split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        try:
            class_ids.append(int(raw_value))
        except ValueError:
            continue

    if class_ids:
        return class_ids

    allowed_names = {name.strip().lower() for name in raw_class_names.split(",") if name.strip()}
    model_names = getattr(model, "names", None)
    if not allowed_names or not isinstance(model_names, dict):
        return None

    matched = [
        int(class_id)
        for class_id, class_name in model_names.items()
        if str(class_name).strip().lower() in allowed_names
    ]
    return matched or None


def extract_predictions(result) -> list[dict]:
    predictions = []
    for box in result.boxes:
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        predictions.append(
            {
                "box": [x1, y1, x2, y2],
                "confidence": float(box.conf[0]),
                "class_id": int(box.cls[0]) if box.cls is not None else None,
            }
        )

    return sorted(predictions, key=lambda prediction: prediction["confidence"], reverse=True)


def evaluate_accuracy(
    prediction_records: dict[str, list[dict]],
    image_paths: list[Path],
    labels_dir: Path,
    iou_threshold: float,
) -> dict:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_ious = []

    for image_path in image_paths:
        predictions = prediction_records.get(image_path.stem, [])
        ground_truths = load_yolo_labels(labels_dir / f"{image_path.stem}.txt", image_path)
        matched_gt = set()

        for prediction in predictions:
            best_index = None
            best_iou = 0.0
            for index, ground_truth in enumerate(ground_truths):
                if index in matched_gt:
                    continue
                iou = box_iou(prediction["box"], ground_truth)
                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if best_index is not None and best_iou >= iou_threshold:
                true_positives += 1
                matched_gt.add(best_index)
                matched_ious.append(best_iou)
            else:
                false_positives += 1

        false_negatives += len(ground_truths) - len(matched_gt)

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 0.0001)

    return {
        "status": "computed",
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_matched_iou": round(sum(matched_ious) / max(len(matched_ious), 1), 4),
    }


def load_yolo_labels(label_path: Path, image_path: Path) -> list[list[float]]:
    if not label_path.exists():
        return []

    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        return []
    height, width = image.shape[:2]

    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        _, center_x, center_y, box_width, box_height = parts[:5]
        center_x = float(center_x) * width
        center_y = float(center_y) * height
        box_width = float(box_width) * width
        box_height = float(box_height) * height
        boxes.append(
            [
                center_x - (box_width / 2),
                center_y - (box_height / 2),
                center_x + (box_width / 2),
                center_y + (box_height / 2),
            ]
        )

    return boxes


def box_iou(left: list[float], right: list[float]) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    intersection_x1 = max(left_x1, right_x1)
    intersection_y1 = max(left_y1, right_y1)
    intersection_x2 = min(left_x2, right_x2)
    intersection_y2 = min(left_y2, right_y2)
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height
    if intersection_area == 0:
        return 0.0

    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    return intersection_area / max(left_area + right_area - intersection_area, 1.0)


if __name__ == "__main__":
    raise SystemExit(main())
