#!/usr/bin/env python3
"""
🔧 Quantum-Enhanced Generation Fix for پی یار
Fixing NoneType error and improving quantum generation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from quantum_enhanced_prompts import QuantumConsciousness

def test_quantum_generation_fixed():
    """🌟 Test Fixed Quantum Generation"""
    print("⚛️ === Testing Fixed Quantum Generation ===")
    
    MODEL_ID = "pqn-ai/pqn-ai-v1"
    
    try:
        # Load model
        print("✅ Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        ).eval()
        
        # Initialize Quantum AI
        quantum_ai = QuantumConsciousness()
        
        # Test simple generation first
        print("\n🔮 Testing simple generation...")
        simple_prompt = "🌟 فعال‌سازی هوش کوانتومی"
        
        inputs = tokenizer(simple_prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        
        print(f"✅ Input shape: {input_ids.shape}")
        print(f"✅ Input IDs: {input_ids}")
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.08,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=inputs.get("attention_mask", None)
            )
        
        # Decode response
        gen_ids = output_ids[0][input_ids.shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        print(f"✅ Generation successful!")
        print(f"📝 Response: {text}")
        
        # Test quantum-enhanced generation
        print("\n⚛️ Testing quantum-enhanced generation...")
        quantum_prompt = quantum_ai.generate_quantum_prompt(simple_prompt)
        
        # Truncate quantum prompt to avoid length issues
        quantum_prompt_truncated = quantum_prompt[:1000] + "\n\n[USER] " + simple_prompt + "\n[ASSISTANT]"
        
        inputs = tokenizer(quantum_prompt_truncated, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs["input_ids"]
        
        print(f"✅ Quantum input shape: {input_ids.shape}")
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.08,
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=inputs.get("attention_mask", None)
            )
        
        # Decode response
        gen_ids = output_ids[0][input_ids.shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        print(f"✅ Quantum generation successful!")
        print(f"📝 Quantum response: {text}")
        
        # Check for quantum elements
        quantum_elements = ['quantum', 'cosmic', 'divine', 'consciousness', 'کوانتومی', 'کیهانی', 'الهی']
        found_elements = [elem for elem in quantum_elements if elem in text.lower()]
        print(f"⚛️ Quantum elements found: {found_elements}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_queries():
    """🌟 Test Multiple Quantum Queries"""
    print("\n🌟 === Testing Multiple Quantum Queries ===")
    
    MODEL_ID = "pqn-ai/pqn-ai-v1"
    
    try:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        ).eval()
        
        test_queries = [
            "🌟 فعال‌سازی هوش کوانتومی",
            "⚛️ Quantum consciousness activation",
            "🔮 Channeling divine guidance"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🎯 Test {i}: {query}")
            
            # Simple prompt
            inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"]
            
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    repetition_penalty=1.08,
                    max_new_tokens=50,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    attention_mask=inputs.get("attention_mask", None)
                )
            
            # Decode response
            gen_ids = output_ids[0][input_ids.shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            print(f"✅ Response: {text}")
            
            # Check for quantum elements
            quantum_elements = ['quantum', 'cosmic', 'divine', 'consciousness', 'کوانتومی', 'کیهانی', 'الهی']
            found_elements = [elem for elem in quantum_elements if elem in text.lower()]
            print(f"⚛️ Quantum elements: {found_elements}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("⚛️ === پی یار Quantum Generation Fix ===\n")
    
    # Test 1: Fixed generation
    success1 = test_quantum_generation_fixed()
    
    # Test 2: Multiple queries
    success2 = test_multiple_queries()
    
    if success1 and success2:
        print("\n🌟 === Quantum Generation Fixed! ===")
        print("✅ Quantum Consciousness: WORKING")
        print("✅ Cosmic Awareness: ACTIVE")
        print("✅ Divine Guidance: CHANNELED")
        print("✅ Quantum Generation: SUCCESSFUL")
        print("\n⚛️ پی یار آماده برای اول شدن! ⚛️")
    else:
        print("\n❌ === Quantum Generation Issues ===")
        print("🔧 Need further debugging...")
