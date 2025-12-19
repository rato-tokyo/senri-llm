"""Debug script to see what the model is actually generating."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from src.modeling_senri import SenriForCausalLM
from src.utils import get_device


def main():
    model_path = "outputs/senri-trained"

    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return

    device = get_device()
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

    # Create a simple NIAH-style prompt
    passkey = "7392"
    haystack = "The weather is sunny. " * 50
    needle = f"The secret passkey is {passkey}. Remember this number."
    question = "What is the secret passkey mentioned in the text above?"

    prompt = f"{haystack}\n\n{needle}\n\n{haystack}\n\n{question}"

    print("\n" + "=" * 60)
    print("PROMPT (truncated):")
    print("=" * 60)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print(f"\n... (total length: {len(prompt)} chars)")
    print(f"Expected answer: {passkey}")

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    print(f"\nInput tokens: {inputs['input_ids'].shape[1]}")

    # Generate
    print("\n" + "=" * 60)
    print("GENERATING...")
    print("=" * 60)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode full response
    tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_tokens = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    print("\nGenerated response (new tokens only):")
    print("-" * 40)
    print(repr(new_tokens))
    print("-" * 40)

    # Check if passkey in response
    if passkey in new_tokens:
        print(f"\n✅ SUCCESS: Passkey '{passkey}' found in response!")
    else:
        print(f"\n❌ FAILED: Passkey '{passkey}' NOT found in response.")

    # Test with a simpler prompt (no haystack)
    print("\n" + "=" * 60)
    print("TESTING SIMPLE PROMPT (no haystack)")
    print("=" * 60)

    simple_prompt = f"The secret passkey is {passkey}. What is the secret passkey?"

    inputs2 = tokenizer(
        simple_prompt,
        return_tensors="pt",
    ).to(device)

    print(f"Prompt: {simple_prompt}")
    print(f"Input tokens: {inputs2['input_ids'].shape[1]}")

    with torch.no_grad():
        outputs2 = model.generate(
            **inputs2,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response2 = tokenizer.decode(
        outputs2[0][inputs2["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    print(f"Response: {repr(response2)}")

    if passkey in response2:
        print("✅ SUCCESS: Passkey found!")
    else:
        print("❌ FAILED: Passkey NOT found.")

    # Test perplexity on a simple sentence
    print("\n" + "=" * 60)
    print("TESTING BASIC LANGUAGE ABILITY")
    print("=" * 60)

    test_prompts = [
        "The capital of France is",
        "1 + 1 =",
        "Hello, my name is",
    ]

    for test_prompt in test_prompts:
        inputs3 = tokenizer(test_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs3 = model.generate(
                **inputs3,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response3 = tokenizer.decode(
            outputs3[0][inputs3["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        print(f"\nPrompt: {test_prompt}")
        print(f"Response: {repr(response3)}")


if __name__ == "__main__":
    main()
