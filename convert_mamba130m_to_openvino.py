from pathlib import Path
import argparse

import openvino as ov
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TraceableMambaForOV(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):
        kwargs = {
            "input_ids": input_ids,
            "use_cache": False,
            "return_dict": False,
        }
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        outputs = self.model(**kwargs)
        return outputs[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download state-spaces/mamba-130m-hf and convert to OpenVINO IR."
    )
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf", help="Hugging Face model id")
    parser.add_argument("--output-dir", default="openvino_ir", help="Directory to save OpenVINO IR")
    parser.add_argument("--ir-name", default="mamba130m", help="OpenVINO IR file name prefix")
    parser.add_argument(
        "--text",
        default="OpenVINO makes model deployment faster.",
        help="Sample text used to build tracing input",
    )
    parser.add_argument(
        "--fixed-batch",
        type=int,
        default=1,
        help="Fixed batch size for exported IR. NPU requires bounded/static shapes.",
    )
    parser.add_argument(
        "--fixed-seq-len",
        type=int,
        default=8,
        help="Fixed sequence length for exported IR. NPU requires bounded/static shapes.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()
    model.config.use_cache = False
    traceable_model = TraceableMambaForOV(model)
    traceable_model.eval()

    encoded = tokenizer(args.text, return_tensors="pt")
    example_input = {k: v for k, v in encoded.items()}

    with torch.no_grad():
        ov_model = ov.convert_model(traceable_model, example_input=example_input)

    # NPU compiler does not accept unbounded dynamic dimensions for this model.
    # Force static shapes for tokenizer inputs so benchmark_app can compile on NPU.
    ov_model.reshape(
        {
            "input_ids": [args.fixed_batch, args.fixed_seq_len],
            "attention_mask": [args.fixed_batch, args.fixed_seq_len],
        }
    )

    ir_path = output_dir / f"{args.ir_name}.xml"
    ov.save_model(ov_model, str(ir_path))

    tokenizer.save_pretrained(output_dir / f"{args.ir_name}_tokenizer")

    print(f"Saved OpenVINO IR: {ir_path}")
    print(f"Saved tokenizer: {output_dir / f'{args.ir_name}_tokenizer'}")


if __name__ == "__main__":
    main()
