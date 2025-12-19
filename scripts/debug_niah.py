#!/usr/bin/env python3
"""Debug script to understand why NIAH tests fail."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.insert(0, ".")

from src.modeling_senri import SenriForCausalLM


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Senri model
    print("\nLoading Senri model...")
    model = SenriForCausalLM.from_pretrained("outputs/senri-trained")
    tokenizer = AutoTokenizer.from_pretrained("outputs/senri-trained")
    model = model.to(device)
    model.eval()
    print(f"Model dtype: {model.lm_head.weight.dtype}")

    # Test passkey
    passkey = "4729"
    needle = f"The secret passkey is {passkey}. Remember this number."
    question = "What is the secret passkey mentioned in the text above?"

    # Test 1: Simple prompt (no haystack)
    print("\n" + "="*60)
    print("Test 1: Simple prompt (no haystack)")
    print("="*60)

    prompt = f"{needle}\n\n{question}"
    print(f"Prompt:\n{prompt}\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"Input length: {inputs['input_ids'].shape[1]} tokens")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Response: {response}")
    print(f"Passkey in response: {passkey in response}")

    # Test 2: With haystack
    print("\n" + "="*60)
    print("Test 2: With haystack (needle at beginning)")
    print("="*60)

    haystack = "The quick brown fox jumps over the lazy dog. " * 50
    prompt2 = f"{needle} {haystack}\n\n{question}"

    inputs2 = tokenizer(prompt2, return_tensors="pt", truncation=True, max_length=1024).to(device)
    print(f"Input length: {inputs2['input_ids'].shape[1]} tokens")

    with torch.no_grad():
        outputs2 = model.generate(
            **inputs2,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response2 = tokenizer.decode(outputs2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Response: {response2}")
    print(f"Passkey in response: {passkey in response2}")

    # Test 3: Text continuation (basic LM ability)
    print("\n" + "="*60)
    print("Test 3: Text continuation (basic LM ability)")
    print("="*60)

    continuation_prompt = "Once upon a time, there was a"
    inputs3 = tokenizer(continuation_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs3 = model.generate(
            **inputs3,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response3 = tokenizer.decode(outputs3[0], skip_special_tokens=True)
    print(f"Senri model: {response3}")

    # Test 4: Compare with base SmolLM
    print("\n" + "="*60)
    print("Test 4: Base SmolLM comparison")
    print("="*60)

    print("Loading base SmolLM...")
    base_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M").to(device)
    base_model.eval()

    # Simple NIAH test on base model
    inputs_base = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs_base = base_model.generate(
            **inputs_base,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response_base = tokenizer.decode(outputs_base[0][inputs_base["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Base SmolLM response: {response_base}")
    print(f"Passkey in response: {passkey in response_base}")

    # Text continuation on base model
    with torch.no_grad():
        outputs_base_cont = base_model.generate(
            **inputs3,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response_base_cont = tokenizer.decode(outputs_base_cont[0], skip_special_tokens=True)
    print(f"Base SmolLM continuation: {response_base_cont}")

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Senri NIAH (no haystack): {'PASS' if passkey in response else 'FAIL'}")
    print(f"Senri NIAH (with haystack): {'PASS' if passkey in response2 else 'FAIL'}")
    print(f"Base SmolLM NIAH: {'PASS' if passkey in response_base else 'FAIL'}")


if __name__ == "__main__":
    main()
