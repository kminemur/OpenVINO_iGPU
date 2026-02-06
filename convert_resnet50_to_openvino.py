from pathlib import Path
import argparse

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import openvino as ov


def _resolve_input_size(image_processor):
    size = getattr(image_processor, "size", None)
    if not size:
        return 224, 224
    # Transformers image processors may store size as dict or int
    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        if "shortest_edge" in size:
            edge = int(size["shortest_edge"])
            return edge, edge
    if isinstance(size, int):
        return int(size), int(size)
    return 224, 224


def main():
    parser = argparse.ArgumentParser(description="Download microsoft/resnet-50 and convert to OpenVINO IR.")
    parser.add_argument("--model", default="microsoft/resnet-50", help="Hugging Face model id")
    parser.add_argument("--output-dir", default="openvino_ir", help="Directory for IR output")
    args = parser.parse_args()

    model_id = args.model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()

    height, width = _resolve_input_size(image_processor)
    example = torch.zeros(1, 3, height, width)

    with torch.no_grad():
        ov_model = ov.convert_model(model, example_input=example)

    ir_path = output_dir / "resnet50.xml"
    ov.save_model(ov_model, str(ir_path))

    print(f"Saved OpenVINO IR to: {ir_path}")


if __name__ == "__main__":
    main()
