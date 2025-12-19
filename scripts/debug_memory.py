"""Debug script to check memory layer behavior."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from src.modeling_senri import SenriForCausalLM


def main():
    model_path = "outputs/senri-model"  # Use converted model (before training)

    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Run: python scripts/convert_to_senri.py --output_dir outputs/senri-model")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    model = SenriForCausalLM.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get memory layer indices
    memory_indices = model.config.get_memory_layer_indices()
    print(f"Memory layers: {memory_indices}")

    # Simple test input
    test_text = "Hello, how are you?"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    print(f"\nInput: {test_text}")
    print(f"Input tokens: {inputs['input_ids'].shape}")

    # Check memory state before forward
    print("\n=== Before forward ===")
    for idx in memory_indices:
        layer = model.model.layers[idx]
        memory = layer.self_attn.memory
        print(f"Layer {idx} memory initialized: {memory.is_initialized}")
        if memory.is_initialized:
            print(f"  M shape: {memory.M.shape}, M sum: {memory.M.sum():.4f}")
            print(f"  z shape: {memory.z.shape}, z sum: {memory.z.sum():.4f}")

    # Forward pass
    print("\n=== Running forward ===")
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Output logits mean: {outputs.logits.mean():.4f}")
    print(f"Output logits std: {outputs.logits.std():.4f}")

    # Check memory state after forward
    print("\n=== After forward ===")
    for idx in memory_indices:
        layer = model.model.layers[idx]
        memory = layer.self_attn.memory
        print(f"Layer {idx}:")
        print(f"  M sum: {memory.M.sum():.4f}, M abs max: {memory.M.abs().max():.4f}")
        print(f"  z sum: {memory.z.sum():.4f}, z abs max: {memory.z.abs().max():.4f}")
        print(f"  Memory is_empty: {memory.is_empty}")

    # Check layer outputs
    print("\n=== Layer output comparison ===")
    # Run forward with hooks to capture intermediate outputs
    layer_outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                layer_outputs[name] = output[0].detach()
            else:
                layer_outputs[name] = output.detach()
        return hook

    # Register hooks
    hooks = []
    for idx in [14, 15, 16]:  # Layer before, memory layer, layer after
        layer = model.model.layers[idx]
        hook = layer.register_forward_hook(make_hook(f"layer_{idx}"))
        hooks.append(hook)

    # Forward again
    model.reset_memory()
    with torch.no_grad():
        model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Print layer outputs
    for name, output in layer_outputs.items():
        print(f"{name}: shape={output.shape}, mean={output.mean():.4f}, std={output.std():.4f}")

    # Generate test
    print("\n=== Generation test ===")
    model.reset_memory()
    with torch.no_grad():
        generated = model.generate(
            inputs.input_ids,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
