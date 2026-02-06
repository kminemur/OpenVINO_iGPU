import argparse
import sys
import threading
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


def get_optimal_num_requests(compiled, fallback):
    try:
        return int(compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))
    except Exception:
        return fallback


def main():
    parser = argparse.ArgumentParser(
        description="Run async inference at throughput limit for 30,000,000 frames (OpenVINO)"
    )
    parser.add_argument("--model", default="openvino_ir/resnet50.xml", help="Path to .xml model")
    parser.add_argument("--device", default="GPU", help="Device name, e.g., GPU or CPU")
    parser.add_argument("--total-frames", type=int, default=30_000_000, help="Total frames to run")
    parser.add_argument("--num-requests", type=int, default=0, help="Async requests (0 = auto)")
    parser.add_argument("--perf-hint", default="THROUGHPUT", help="PERFORMANCE_HINT value")
    parser.add_argument("--report-every", type=int, default=1_000_000, help="Progress interval")
    args = parser.parse_args()

    if args.total_frames < 0:
        print("--total-frames must be >= 0", file=sys.stderr)
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    core = ov.Core()
    model = core.read_model(model_path)
    config = {"PERFORMANCE_HINT": args.perf_hint} if args.perf_hint else {}
    compiled = core.compile_model(model, args.device, config)

    dummy = build_dummy_input(model)
    input_key = compiled.input(0)

    num_requests = args.num_requests
    if num_requests <= 0:
        num_requests = get_optimal_num_requests(compiled, fallback=4)

    if args.total_frames == 0:
        print("total_frames = 0, nothing to do.")
        return

    queue = ov.AsyncInferQueue(compiled, num_requests)

    submitted = 0
    completed = 0
    lock = threading.Lock()
    done_event = threading.Event()
    start_time = time.time()

    def callback(request, userdata):
        nonlocal submitted, completed
        with lock:
            completed += 1
            if args.report_every > 0 and completed % args.report_every == 0:
                elapsed = time.time() - start_time
                fps = completed / elapsed if elapsed > 0 else 0.0
                print(f"Progress: {completed}/{args.total_frames} frames, {fps:.2f} FPS")
            if submitted < args.total_frames:
                queue.start_async({input_key: dummy}, userdata=submitted)
                submitted += 1
            if completed >= args.total_frames:
                done_event.set()

    queue.set_callback(callback)

    warmup = min(num_requests, args.total_frames)
    for i in range(warmup):
        queue.start_async({input_key: dummy}, userdata=i)
        submitted += 1

    done_event.wait()
    queue.wait_all()

    end_time = time.time()
    elapsed = end_time - start_time
    fps = completed / elapsed if elapsed > 0 else 0.0
    print(f"Completed {completed} frames on {args.device} in {elapsed:.3f}s ({fps:.2f} FPS)")


if __name__ == "__main__":
    main()
