from pathlib import Path
import argparse

import numpy as np
import openvino as ov
from transformers import AutoTokenizer


def _to_numpy_int64(values):
    if hasattr(values, "numpy"):
        return values.numpy().astype(np.int64)
    return np.asarray(values, dtype=np.int64)


def _build_feed(compiled_model, encoded):
    feed = {}
    input_ids = _to_numpy_int64(encoded["input_ids"])
    attention_mask = _to_numpy_int64(encoded.get("attention_mask")) if "attention_mask" in encoded else None

    for inp in compiled_model.inputs:
        name = inp.get_any_name()
        if name == "input_ids":
            feed[name] = input_ids
        elif name == "attention_mask" and attention_mask is not None:
            feed[name] = attention_mask
        elif "position" in name:
            seq_len = input_ids.shape[1]
            feed[name] = np.arange(seq_len, dtype=np.int64)[None, :]
        else:
            # Fallback for optional or model-specific int inputs.
            pshape = inp.partial_shape
            if pshape.is_dynamic:
                shape = [1]
            else:
                shape = [int(d) if int(d) > 0 else 1 for d in pshape.get_shape()]
            feed[name] = np.zeros(shape, dtype=np.int64)
    return feed


def _pick_logits(outputs):
    # Prefer output named "logits" when available.
    for key, value in outputs.items():
        if hasattr(key, "get_any_name") and key.get_any_name() == "logits":
            return value

    first_key = next(iter(outputs.keys()))
    return outputs[first_key]


def _fixed_seq_len(compiled_model):
    # Some traced models keep a fixed sequence length in graph internals.
    out_shape = compiled_model.output(0).partial_shape
    if len(out_shape) >= 2 and out_shape[1].is_static:
        value = int(out_shape[1].get_length())
        if value > 0:
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenVINO IR converted from Mamba 130M")
    parser.add_argument("--model", default="openvino_ir/mamba130m.xml", help="Path to OpenVINO IR .xml")
    parser.add_argument(
        "--tokenizer-dir",
        default="openvino_ir/mamba130m_tokenizer",
        help="Path to tokenizer directory saved during conversion",
    )
    parser.add_argument("--device", default="GPU", help="OpenVINO device: GPU or CPU")
    parser.add_argument("--prompt", default="OpenVINO is", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Greedy decode steps")
    args = parser.parse_args()

    model_path = Path(args.model)
    tokenizer_dir = Path(args.tokenizer_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

    core = ov.Core()
    model = core.read_model(model_path)
    compiled = core.compile_model(model, args.device, {"PERFORMANCE_HINT": "LATENCY"})
    seq_len = _fixed_seq_len(compiled)

    text = args.prompt
    for _ in range(args.max_new_tokens):
        if seq_len is None:
            encoded = tokenizer(text, return_tensors="np")
        else:
            encoded = tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                padding="max_length",
                max_length=seq_len,
            )
        feed = _build_feed(compiled, encoded)

        outputs = compiled(feed)
        logits = _pick_logits(outputs)
        if "attention_mask" in encoded:
            last_pos = int(encoded["attention_mask"][0].sum()) - 1
            last_pos = max(last_pos, 0)
        else:
            last_pos = logits.shape[1] - 1
        next_token_id = int(np.argmax(logits[0, last_pos, :]))

        next_piece = tokenizer.decode([next_token_id], skip_special_tokens=True)
        if not next_piece:
            break
        text += next_piece

    print("Prompt:", args.prompt)
    print("Generated:", text)


if __name__ == "__main__":
    main()
