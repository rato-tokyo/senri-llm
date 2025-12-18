"""
Convert Qwen2.5-0.5B weights to Senri model.

This script loads pre-trained Qwen2.5-0.5B weights and transfers them
to the Senri model architecture. The Senri model adds memory layers
on top of the base Qwen architecture.

Usage:
    python scripts/convert_qwen_to_senri.py --output_dir ./senri-0.5b
    python scripts/convert_qwen_to_senri.py --model_name Qwen/Qwen2.5-1.5B --output_dir ./senri-1.5b
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


def get_weight_mapping():
    """
    Get mapping from Qwen2 weight names to Senri weight names.

    Returns:
        dict: Mapping of Qwen2 -> Senri weight names
    """
    # Most weights have the same names since Senri is based on Qwen2 architecture
    # The main differences are in attention layers with Senri Memory
    return {
        # Embeddings
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
        # Layer weights are mapped dynamically
    }


def convert_layer_weights(qwen_state_dict: dict, layer_idx: int, has_memory: bool) -> dict:
    """
    Convert weights for a single decoder layer.

    Args:
        qwen_state_dict: Qwen2 state dict
        layer_idx: Layer index
        has_memory: Whether this layer has Senri Memory

    Returns:
        dict: Converted weights for this layer
    """
    converted = {}
    prefix = f"model.layers.{layer_idx}"

    # Layer norm weights
    converted[f"{prefix}.input_layernorm.weight"] = qwen_state_dict[
        f"{prefix}.input_layernorm.weight"
    ]
    converted[f"{prefix}.post_attention_layernorm.weight"] = qwen_state_dict[
        f"{prefix}.post_attention_layernorm.weight"
    ]

    # MLP weights
    converted[f"{prefix}.mlp.gate_proj.weight"] = qwen_state_dict[
        f"{prefix}.mlp.gate_proj.weight"
    ]
    converted[f"{prefix}.mlp.up_proj.weight"] = qwen_state_dict[
        f"{prefix}.mlp.up_proj.weight"
    ]
    converted[f"{prefix}.mlp.down_proj.weight"] = qwen_state_dict[
        f"{prefix}.mlp.down_proj.weight"
    ]

    # Attention weights
    # For layers with Senri Memory, the attention structure is different
    # but the core Q, K, V, O projections are the same
    converted[f"{prefix}.self_attn.q_proj.weight"] = qwen_state_dict[
        f"{prefix}.self_attn.q_proj.weight"
    ]
    converted[f"{prefix}.self_attn.q_proj.bias"] = qwen_state_dict[
        f"{prefix}.self_attn.q_proj.bias"
    ]
    converted[f"{prefix}.self_attn.k_proj.weight"] = qwen_state_dict[
        f"{prefix}.self_attn.k_proj.weight"
    ]
    converted[f"{prefix}.self_attn.k_proj.bias"] = qwen_state_dict[
        f"{prefix}.self_attn.k_proj.bias"
    ]
    converted[f"{prefix}.self_attn.v_proj.weight"] = qwen_state_dict[
        f"{prefix}.self_attn.v_proj.weight"
    ]
    converted[f"{prefix}.self_attn.v_proj.bias"] = qwen_state_dict[
        f"{prefix}.self_attn.v_proj.bias"
    ]
    converted[f"{prefix}.self_attn.o_proj.weight"] = qwen_state_dict[
        f"{prefix}.self_attn.o_proj.weight"
    ]

    return converted


def convert_qwen_to_senri(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    output_dir: str = "./senri-0.5b",
    device: str = "cpu",
) -> SenriForCausalLM:
    """
    Convert Qwen2.5 model to Senri model.

    Args:
        model_name: HuggingFace model name for Qwen2.5
        output_dir: Directory to save the converted model
        device: Device to use for conversion

    Returns:
        SenriForCausalLM: Converted Senri model
    """
    print(f"Loading Qwen model: {model_name}")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    )
    qwen_config = qwen_model.config
    qwen_state_dict = qwen_model.state_dict()

    print(f"Qwen config: {qwen_config}")

    # Create Senri config based on Qwen config
    senri_config = SenriConfig(
        vocab_size=qwen_config.vocab_size,
        hidden_size=qwen_config.hidden_size,
        intermediate_size=qwen_config.intermediate_size,
        num_hidden_layers=qwen_config.num_hidden_layers,
        num_attention_heads=qwen_config.num_attention_heads,
        num_key_value_heads=qwen_config.num_key_value_heads,
        max_position_embeddings=qwen_config.max_position_embeddings,
        rms_norm_eps=qwen_config.rms_norm_eps,
        rope_theta=qwen_config.rope_theta,
        # Senri specific - adjust based on model size
        sliding_window_size=4096,
        chunk_size=64,
        top_k_memories=64,
        num_memory_layers=3,
        first_memory_layer=qwen_config.num_hidden_layers // 2,  # Start at middle
        memory_layer_interval=qwen_config.num_hidden_layers // 6,  # Spread evenly
    )

    print("Senri config created")
    print(f"  Memory layers: {senri_config.get_memory_layer_indices()}")

    # Create Senri model
    print("Creating Senri model...")
    senri_model = SenriForCausalLM(senri_config)

    # Convert and load weights
    print("Converting weights...")
    senri_state_dict = {}

    # Embeddings and final norm
    senri_state_dict["model.embed_tokens.weight"] = qwen_state_dict[
        "model.embed_tokens.weight"
    ]
    senri_state_dict["model.norm.weight"] = qwen_state_dict["model.norm.weight"]
    senri_state_dict["lm_head.weight"] = qwen_state_dict["lm_head.weight"]

    # Convert each layer
    for layer_idx in range(qwen_config.num_hidden_layers):
        has_memory = senri_config.is_memory_layer(layer_idx)
        layer_weights = convert_layer_weights(qwen_state_dict, layer_idx, has_memory)
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
        for key in missing_keys:
            print(f"    - {key}")

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
    model_name: str = "Qwen/Qwen2.5-0.5B",
    device: str = "cpu",
):
    """
    Verify the conversion by comparing outputs.

    Args:
        senri_model: Converted Senri model
        model_name: Original Qwen model name
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
    parser = argparse.ArgumentParser(description="Convert Qwen2.5 to Senri model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name for Qwen2.5",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./senri-0.5b",
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
        help="Verify conversion by comparing outputs",
    )

    args = parser.parse_args()

    senri_model = convert_qwen_to_senri(
        model_name=args.model_name,
        output_dir=args.output_dir,
        device=args.device,
    )

    if args.verify:
        verify_conversion(senri_model, args.model_name, args.device)


if __name__ == "__main__":
    main()
