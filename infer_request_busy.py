import argparse
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov


def build_dummy_input(model):
    input_tensor = model.input(0)
    pshape = input_tensor.partial_shape
    if pshape.is_dynamic:
        rank = pshape.rank.get_length()
        if rank == 4:
            shape = [1, 3, 224, 224]
        else:
            shape = [1] * (rank if rank is not None else 1)
    else:
        shape = list(pshape.get_shape())
    return np.random.random_sample(shape).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Trigger 'Infer Request is Busy' in OpenVINO")
    parser.add_argument("--model", default="openvino_ir/resnet50.xml", help="Path to .xml model")
    parser.add_argument("--device", default="GPU", help="Device name, e.g., GPU or CPU")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    core = ov.Core()
    model = core.read_model(model_path)
    compiled = core.compile_model(model, args.device)

    dummy = build_dummy_input(model)
    input_key = compiled.input(0)

    # Create a single InferRequest and start it twice without waiting.
    # The second start should raise: "Infer Request is Busy".
    request = compiled.create_infer_request()
    request.start_async({input_key: dummy})

    # Immediately start again while the request is still running.
    # This is expected to raise a RuntimeError with "Infer Request is Busy".
    request.start_async({input_key: dummy})

    # If it didn't error (unlikely), wait to clean up.
    request.wait()


if __name__ == "__main__":
    main()