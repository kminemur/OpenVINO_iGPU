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
        # Dynamic or unknown shape: use a safe default for 4D image models.
        rank = pshape.rank.get_length()
        if rank == 4:
            shape = [1, 3, 224, 224]
        else:
            shape = [1] * (rank if rank is not None else 1)
    else:
        shape = list(pshape.get_shape())
    dtype = np.float32
    return np.random.random_sample(shape).astype(dtype)


def main():
    parser = argparse.ArgumentParser(description="Async inference with dummy data (OpenVINO)")
    parser.add_argument("--model", default="openvino_ir/resnet50.xml", help="Path to .xml model")
    parser.add_argument("--device", default="GPU", help="Device name, e.g., GPU or CPU")
    parser.add_argument("--num-requests", type=int, default=4, help="Number of async requests")
    parser.add_argument("--num-iterations", type=int, default=8, help="Number of async inference jobs")
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

    results = []

    def callback(request, userdata):
        output = request.get_output_tensor(0).data
        results.append((userdata, output))

    queue = ov.AsyncInferQueue(compiled, args.num_requests)
    queue.set_callback(callback)

    start = time.time()
    for i in range(args.num_iterations):
        queue.start_async({input_key: dummy}, userdata=i)
    queue.wait_all()
    end = time.time()

    results.sort(key=lambda x: x[0])
    print(f"Completed {len(results)} async inferences on {args.device} in {end - start:.3f}s")
    # Print top-5 indices for the first result as a simple sanity check
    first = results[0][1].squeeze()
    top5 = np.argsort(first)[-5:][::-1]
    print("Top-5 indices:", top5.tolist())


if __name__ == "__main__":
    main()
