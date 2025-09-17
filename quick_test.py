#!/usr/bin/env python3
"""
Quick smoke test for PQN.AI model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "pqn-ai/pqn-ai-v1"

def quick_test():
    print("=== PQN.AI Quick Smoke Test ===")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        print("✅ Tokenizer loaded successfully!")
        
        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=False,
            torch_dtype=torch.float32,
            device_map=None
        )
        print("✅ Model loaded successfully!")
        
        # Test Persian
        print("\n--- Testing Persian ---")
        persian_prompt = "سلام! چطور می‌تونم کمکت کنم؟"
        inputs = tokenizer(persian_prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {persian_prompt}")
        print(f"Output: {response}")
        
        # Test English
        print("\n--- Testing English ---")
        english_prompt = "Hello! How can I assist you today?"
        inputs = tokenizer(english_prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {english_prompt}")
        print(f"Output: {response}")
        
        # Test long prompt
        print("\n--- Testing Long Prompt ---")
        long_prompt = "Write a detailed story about a quantum AI system that helps humanity solve complex problems in the future."
        inputs = tokenizer(long_prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {long_prompt}")
        print(f"Output: {response}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
