"""
Convert base model weights to Senri model.

This script loads pre-trained model weights (SmolLM, Llama, etc.) and transfers them
to the Senri model architecture. The Senri model adds memory layers
on top of the base architecture.

Supported base models:
- SmolLM-135M (HuggingFaceTB/SmolLM-135M) - Recommended for experiments
- SmolLM-360M (HuggingFaceTB/SmolLM-360M)
- Any LlamaConfig-based model

Usage:
    python scripts/convert_to_senri.py --output_dir ./senri-135m
    python scripts/convert_to_senri.py --model_name HuggingFaceTB/SmolLM-360M --output_dir ./senri-360m
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.configuration_senri import SenriConfig
from src.modeling_senri import SenriForCausalLM


def convert_layer_weights(
    base_state_dict: dict, layer_idx: int, has_memory: bool
) -> dict:
    """
    Convert weights for a single decoder layer.

    Args:
        base_state_dict: Base model state dict
        layer_idx: Layer index
        has_memory: Whether this layer has Senri Memory

    Returns:
        dict: Converted weights for this layer
    """
    converted = {}
    prefix = f"model.layers.{layer_idx}"

    # Layer norm weights
    converted[f"{prefix}.input_layernorm.weight"] = base_state_dict[
        f"{prefix}.input_layernorm.weight"
    ]
    converted[f"{prefix}.post_attention_layernorm.weight"] = base_state_dict[
        f"{prefix}.post_attention_layernorm.weight"
    ]

    # MLP weights
    converted[f"{prefix}.mlp.gate_proj.weight"] = base_state_dict[
        f"{prefix}.mlp.gate_proj.weight"
    ]
    converted[f"{prefix}.mlp.up_proj.weight"] = base_state_dict[
        f"{prefix}.mlp.up_proj.weight"
    ]
    converted[f"{prefix}.mlp.down_proj.weight"] = base_state_dict[
        f"{prefix}.mlp.down_proj.weight"
    ]

    # Attention weights - Q, K, V, O projections
    converted[f"{prefix}.self_attn.q_proj.weight"] = base_state_dict[
        f"{prefix}.self_attn.q_proj.weight"
    ]
    converted[f"{prefix}.self_attn.k_proj.weight"] = base_state_dict[
        f"{prefix}.self_attn.k_proj.weight"
    ]
    converted[f"{prefix}.self_attn.v_proj.weight"] = base_state_dict[
        f"{prefix}.self_attn.v_proj.weight"
    ]
    converted[f"{prefix}.self_attn.o_proj.weight"] = base_state_dict[
        f"{prefix}.self_attn.o_proj.weight"
    ]

    # Handle biases if present (SmolLM doesn't have attention biases, but some models do)
    for proj in ["q_proj", "k_proj", "v_proj"]:
        bias_key = f"{prefix}.self_attn.{proj}.bias"
        if bias_key in base_state_dict:
            converted[bias_key] = base_state_dict[bias_key]

    return converted


def convert_to_senri(
    model_name: str = "HuggingFaceTB/SmolLM-135M",
    output_dir: str = "./senri-135m",
    device: str = "cpu",
) -> SenriForCausalLM:
    """
    Convert base model to Senri model.

    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save the converted model
        device: Device to use for conversion

    Returns:
        SenriForCausalLM: Converted Senri model
    """
    print(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    )
    base_config = base_model.config
    base_state_dict = base_model.state_dict()

    print(f"Base config: hidden_size={base_config.hidden_size}, "
          f"num_layers={base_config.num_hidden_layers}, "
          f"num_heads={base_config.num_attention_heads}")

    # Calculate head_dim
    head_dim = base_config.hidden_size // base_config.num_attention_heads

    # Determine memory layer configuration based on model size
    num_layers = base_config.num_hidden_layers
    if num_layers <= 12:
        # Small model: 2 memory layers at 1/3 and 2/3
        num_memory_layers = 2
        first_memory_layer = num_layers // 3
        memory_layer_interval = num_layers // 3
    else:
        # Larger model: 3 memory layers
        num_memory_layers = 3
        first_memory_layer = num_layers // 2
        memory_layer_interval = num_layers // 6

    # Create Senri config based on base config
    senri_config = SenriConfig(
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
        intermediate_size=base_config.intermediate_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=getattr(base_config, 'num_key_value_heads', base_config.num_attention_heads),
        max_position_embeddings=base_config.max_position_embeddings,
        rms_norm_eps=getattr(base_config, 'rms_norm_eps', 1e-6),
        rope_theta=getattr(base_config, 'rope_theta', 10000.0),
        # Senri specific
        sliding_window_size=min(2048, base_config.max_position_embeddings),
        chunk_size=64,
        top_k_memories=min(64, base_config.hidden_size),
        num_memory_layers=num_memory_layers,
        first_memory_layer=first_memory_layer,
        memory_layer_interval=memory_layer_interval,
    )

    print("Senri config created")
    print(f"  Memory layers: {senri_config.get_memory_layer_indices()}")
    print(f"  Head dim: {head_dim}")

    # Create Senri model
    print("Creating Senri model...")
    senri_model = SenriForCausalLM(senri_config)

    # Convert and load weights
    print("Converting weights...")
    senri_state_dict = {}

    # Embeddings and final norm
    senri_state_dict["model.embed_tokens.weight"] = base_state_dict[
        "model.embed_tokens.weight"
    ]
    senri_state_dict["model.norm.weight"] = base_state_dict["model.norm.weight"]

    # LM head (may be tied to embeddings)
    if "lm_head.weight" in base_state_dict:
        senri_state_dict["lm_head.weight"] = base_state_dict["lm_head.weight"]
    else:
        # Tied embeddings
        senri_state_dict["lm_head.weight"] = base_state_dict["model.embed_tokens.weight"]

    # Convert each layer
    for layer_idx in range(base_config.num_hidden_layers):
        has_memory = senri_config.is_memory_layer(layer_idx)
        layer_weights = convert_layer_weights(base_state_dict, layer_idx, has_memory)
        senri_state_dict.update(layer_weights)

        if has_memory:
            print(f"  Layer {layer_idx}: Converted (with Senri Memory)")
        else:
            print(f"  Layer {layer_idx}: Converted")

    # Load weights into Senri model
    print("Loading weights into Senri model...")
    missing_keys, unexpected_keys = senri_model.load_state_dict(
        senri_state_dict, strict=False
    )

    if missing_keys:
        print("  Missing keys (expected for new Senri components):")
        for key in missing_keys[:10]:  # Show first 10
            print(f"    - {key}")
        if len(missing_keys) > 10:
            print(f"    ... and {len(missing_keys) - 10} more")

    if unexpected_keys:
        print("  Unexpected keys:")
        for key in unexpected_keys:
            print(f"    - {key}")

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving Senri model to {output_path}...")
    senri_model.save_pretrained(output_path)

    # Also save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_path)

    print("Conversion complete!")
    return senri_model


def verify_conversion(
    senri_model: SenriForCausalLM,
    model_name: str = "HuggingFaceTB/SmolLM-135M",
    device: str = "cpu",
):
    """
    Verify the conversion by testing model outputs.

    Args:
        senri_model: Converted Senri model
        model_name: Original model name for tokenizer
        device: Device to use
    """
    print("\nVerifying conversion...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Test input
    test_text = "Hello, how are you?"
    inputs = tokenizer(test_text, return_tensors="pt").to(device)

    # Get Senri output (in training mode to use standard Infini Attention)
    senri_model = senri_model.to(device)  # type: ignore[arg-type]
    senri_model.train()

    with torch.no_grad():
        senri_output = senri_model(**inputs)

    print(f"  Input: {test_text}")
    print(f"  Output logits shape: {senri_output.logits.shape}")
    print(f"  Output logits mean: {senri_output.logits.mean().item():.4f}")
    print(f"  Output logits std: {senri_output.logits.std().item():.4f}")

    # Generate some text
    senri_model.eval()
    senri_model.reset_memory(1, device=torch.device(device), dtype=torch.float32)

    generated = senri_model.generate(
        inputs.input_ids,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"  Generated: {generated_text}")


def main():
    parser = argparse.ArgumentParser(description="Convert base model to Senri model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceTB/SmolLM-135M",
        help="HuggingFace model name (default: SmolLM-135M)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./senri-135m",
        help="Directory to save the converted model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for conversion",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify conversion by testing outputs",
    )

    args = parser.parse_args()

    senri_model = convert_to_senri(
        model_name=args.model_name,
        output_dir=args.output_dir,
        device=args.device,
    )

    if args.verify:
        verify_conversion(senri_model, args.model_name, args.device)


if __name__ == "__main__":
    main()
