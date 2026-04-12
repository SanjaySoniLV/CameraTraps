#!/usr/bin/env python3
import argparse
import json
import numpy as np
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import onnx
import onnxruntime as ort
import torch
from PIL import Image

from PytorchWildlife.models.detection.rtdetr_apache.megadetectorv6_apache import MegaDetectorV6Apache

TEST_IMAGE_URL = "https://tse1.mm.bing.net/th/id/OIP.UMlWMVIRUuY4mFqMgD01TAHaIL?rs=1&pid=ImgDetMain&o=7&rm=3"
DEFAULT_LOCAL_TEST_IMAGE = Path(__file__).resolve().parent.parent / "IMG-20260410-WA0011.jpg"


def download_image(url: str, destination: Path, timeout: int = 60):
    try:
        with urlopen(url, timeout=timeout) as response:
            destination.write_bytes(response.read())
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Failed to download test image from '{url}': {exc}") from exc


def prepare_inputs(detector: MegaDetectorV6Apache, test_image_path: Path = None, test_image_url: str = None):
    if test_image_path is not None:
        if test_image_path.exists():
            image = Image.open(test_image_path).convert("RGB")
            image_source = str(test_image_path.resolve())
        elif test_image_url is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                image_path = Path(tmp_dir) / "test_image.jpg"
                download_image(test_image_url, image_path)
                image = Image.open(image_path).convert("RGB")
                image_source = test_image_url
        else:
            raise FileNotFoundError(f"Test image file not found: '{test_image_path}'")
    elif test_image_url is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "test_image.jpg"
            download_image(test_image_url, image_path)
            image = Image.open(image_path).convert("RGB")
            image_source = test_image_url
    else:
        raise ValueError("Either test_image_path or test_image_url must be provided")

    width, height = image.size
    image_tensor = detector.transform(image).unsqueeze(0)
    original_image_size = torch.tensor([[width, height]], dtype=torch.float32)
    return image_tensor, original_image_size, image_source


def compare_outputs(torch_outputs, ort_outputs, rtol: float, atol: float):
    labels_torch, boxes_torch, scores_torch = torch_outputs
    labels_ort, boxes_ort, scores_ort = ort_outputs

    labels_torch_np = labels_torch.detach().cpu().numpy()
    boxes_torch_np = boxes_torch.detach().cpu().numpy()
    scores_torch_np = scores_torch.detach().cpu().numpy()

    labels_match = np.array_equal(labels_torch_np, labels_ort)
    boxes_close = np.allclose(boxes_torch_np, boxes_ort, rtol=rtol, atol=atol)
    scores_close = np.allclose(scores_torch_np, scores_ort, rtol=rtol, atol=atol)

    boxes_abs_diff = np.abs(boxes_torch_np - boxes_ort)
    scores_abs_diff = np.abs(scores_torch_np - scores_ort)

    return {
        "rtol": rtol,
        "atol": atol,
        "labels_exact_match": bool(labels_match),
        "boxes_allclose": bool(boxes_close),
        "scores_allclose": bool(scores_close),
        "boxes_max_abs_diff": float(boxes_abs_diff.max()) if boxes_abs_diff.size > 0 else 0.0,
        "scores_max_abs_diff": float(scores_abs_diff.max()) if scores_abs_diff.size > 0 else 0.0,
        "boxes_mean_abs_diff": float(boxes_abs_diff.mean()) if boxes_abs_diff.size > 0 else 0.0,
        "scores_mean_abs_diff": float(scores_abs_diff.mean()) if scores_abs_diff.size > 0 else 0.0,
    }


def export_onnx(output_path: Path, opset: int, test_image_path: Path = None, test_image_url: str = None, rtol: float = 1e-3, atol: float = 1e-4):
    detector = MegaDetectorV6Apache(device="cpu", pretrained=True, version="MDV6-apa-rtdetr-c")
    model = detector.model.eval().cpu()

    image_tensor, original_image_size, image_source = prepare_inputs(
        detector,
        test_image_path=test_image_path,
        test_image_url=test_image_url,
    )

    with torch.no_grad():
        torch_outputs = model(image_tensor, original_image_size)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (image_tensor, original_image_size),
            str(output_path),
            input_names=["images", "orig_target_sizes"],
            output_names=["labels", "boxes", "scores"],
            dynamic_axes={
                "images": {0: "batch_size"},
                "orig_target_sizes": {0: "batch_size"},
                "labels": {0: "batch_size", 1: "num_detections"},
                "boxes": {0: "batch_size", 1: "num_detections"},
                "scores": {0: "batch_size", 1: "num_detections"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    providers = ["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(str(output_path), providers=providers)
    ort_outputs = ort_session.run(
        None,
        {
            "images": image_tensor.numpy(),
            "orig_target_sizes": original_image_size.numpy(),
        },
    )

    if len(ort_outputs) != 3:
        raise RuntimeError(f"Unexpected ONNX Runtime output count: {len(ort_outputs)}")

    comparison = compare_outputs(torch_outputs, ort_outputs, rtol=rtol, atol=atol)
    if not (comparison["labels_exact_match"] and comparison["boxes_allclose"] and comparison["scores_allclose"]):
        raise RuntimeError(f"PyTorch vs ONNX Runtime output mismatch: {comparison}")

    return {
        "model": "MDV6-apa-rtdetr-c",
        "test_image_source": image_source,
        "onnx_path": str(output_path.resolve()),
        "onnx_size_bytes": output_path.stat().st_size,
        "outputs": {
            "labels_shape": list(ort_outputs[0].shape),
            "boxes_shape": list(ort_outputs[1].shape),
            "scores_shape": list(ort_outputs[2].shape),
        },
        "comparison": comparison,
    }


def main():
    parser = argparse.ArgumentParser(description="Export MDV6-apa-rtdetr-c to ONNX and validate with ONNX Runtime")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mdv6-apa-rtdetr-c.onnx"),
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--test-image-path",
        type=Path,
        default=DEFAULT_LOCAL_TEST_IMAGE,
        help="Local test image path used to validate ONNX Runtime inference",
    )
    parser.add_argument(
        "--test-image-url",
        type=str,
        default=None,
        help="Optional fallback URL of test image used only when no local test-image-path is provided",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for PyTorch vs ONNX Runtime output comparisons",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for PyTorch vs ONNX Runtime output comparisons",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write export summary as JSON",
    )

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    effective_test_image_url = args.test_image_url or TEST_IMAGE_URL

    summary = export_onnx(
        args.output,
        args.opset,
        test_image_path=args.test_image_path,
        test_image_url=effective_test_image_url,
        rtol=args.rtol,
        atol=args.atol,
    )

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
