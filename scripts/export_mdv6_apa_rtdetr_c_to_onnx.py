#!/usr/bin/env python3
import argparse
import json
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


def download_image(url: str, destination: Path, timeout: int = 60):
    try:
        with urlopen(url, timeout=timeout) as response:
            destination.write_bytes(response.read())
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Failed to download test image from '{url}': {exc}") from exc


def prepare_inputs(detector: MegaDetectorV6Apache, test_image_url: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = Path(tmp_dir) / "test_image.jpg"
        download_image(test_image_url, image_path)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image_tensor = detector.transform(image).unsqueeze(0)
        original_image_sizes = torch.tensor([[width, height]], dtype=torch.float32)
    return image_tensor, original_image_sizes


def export_onnx(output_path: Path, opset: int, test_image_url: str):
    detector = MegaDetectorV6Apache(device="cpu", pretrained=True, version="MDV6-apa-rtdetr-c")
    model = detector.model.eval().cpu()

    image_tensor, original_image_sizes = prepare_inputs(detector, test_image_url)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (image_tensor, original_image_sizes),
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
            "orig_target_sizes": original_image_sizes.numpy(),
        },
    )

    if len(ort_outputs) != 3:
        raise RuntimeError(f"Unexpected ONNX Runtime output count: {len(ort_outputs)}")

    return {
        "model": "MDV6-apa-rtdetr-c",
        "test_image_url": test_image_url,
        "onnx_path": str(output_path.resolve()),
        "onnx_size_bytes": output_path.stat().st_size,
        "outputs": {
            "labels_shape": list(ort_outputs[0].shape),
            "boxes_shape": list(ort_outputs[1].shape),
            "scores_shape": list(ort_outputs[2].shape),
        },
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
        "--test-image-url",
        type=str,
        default=TEST_IMAGE_URL,
        help="URL of test image used to validate ONNX Runtime inference",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write export summary as JSON",
    )

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    summary = export_onnx(args.output, args.opset, args.test_image_url)

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
