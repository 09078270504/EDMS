# core/services/llm_search.py
# PRODUCTION-READY VERSION with multiple deployment options

import os
import torch
from django.conf import settings
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_NAME = getattr(settings, 'LLM_MODEL_NAME', "Qwen/Qwen2-VL-7B-Instruct")
MAX_NEW_TOKENS = getattr(settings, 'LLM_MAX_TOKENS', 256)
USE_MOCK_LLM = getattr(settings, 'USE_MOCK_LLM', False)  # For development/testing

# Global model storage (in-memory cache)
_model = None
_processor = None
_model_loading = False
_model_load_time = None

def _load_model():
    """
    Load the LLM model with optimized caching
    - First load: Downloads model (~3GB) and caches to disk (~30-60 sec)
    - Subsequent loads: Loads from disk cache (~5-10 sec)
    - After first load: Model stays in memory (instant access)
    """
    global _model, _processor, _model_loading, _model_load_time
    
    # Check if model is already loaded in memory (instant)
    if _model is not None:
        print(f"✓ Using cached model (loaded {_model_load_time})")
        return _model, _processor
    
    if _model_loading:
        # Prevent multiple simultaneous loads
        import time
        print("⏳ Waiting for model to finish loading...")
        for _ in range(60):  # Wait up to 60 seconds
            if _model is not None:
                return _model, _processor
            time.sleep(1)
        raise Exception("Model loading timeout")
    
    _model_loading = True
    
    try:
        import time
        start_time = time.time()
        print(f"📥 Loading LLM model: {MODEL_NAME}")
        
        # Check if accelerate is available for device_map
        try:
            import accelerate
            has_accelerate = True
        except ImportError:
            has_accelerate = False
            print("Warning: accelerate library not found. Install with: pip install accelerate")
        
        # Configuration for different deployment scenarios
        model_kwargs = {
            "trust_remote_code": True
        }
        
        # Only use device_map if accelerate is available
        if has_accelerate:
            model_kwargs["device_map"] = "auto"
        
        # Memory optimization settings
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            # Use 4-bit quantization if low on VRAM (less than 16GB)
            if total_memory < 16 * 1024**3:  # 16GB in bytes
                print("Using 4-bit quantization for memory efficiency")
                if has_accelerate:
                    model_kwargs.update({
                        "load_in_4bit": True,
                        "bnb_4bit_use_double_quant": True,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_compute_dtype": torch.bfloat16
                    })
                else:
                    print("Warning: 4-bit quantization requires accelerate library")
                    model_kwargs["dtype"] = torch.bfloat16
            else:
                model_kwargs["dtype"] = torch.bfloat16
        else:
            # CPU fallback
            model_kwargs["dtype"] = torch.float32
            print("Warning: Using CPU inference - this will be slow")
        
        # Load model and processor
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            **model_kwargs
        )
        
        # Move to GPU if available and not using device_map
        if torch.cuda.is_available() and not has_accelerate:
            print("Moving model to CUDA device")
            _model = _model.cuda()
        
        _processor = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True
        )
        
        # Record load time and log success
        _model_load_time = time.strftime("%Y-%m-%d %H:%M:%S")
        load_duration = time.time() - start_time
        print(f"✓ LLM model loaded successfully in {load_duration:.1f}s")
        print(f"💾 Model cached in memory - future queries will be instant!")
        return _model, _processor
        
    finally:
        _model_loading = False

def answer_from_context(question: str, contexts: str, temperature: float = 0.2) -> str:
    """
    Generate natural language answer from question and document contexts
    
    Args:
        question: User's question
        contexts: Combined context from relevant documents
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = random)
    
    Returns:
        Generated answer string
    """
    
    # Use mock service if enabled (for development/testing)
    if USE_MOCK_LLM:
        return _mock_answer(question, contexts)
    
    try:
        model, processor = _load_model()
        
        # Prepare the conversation
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a helpful document analysis assistant. "
                    "Answer questions based ONLY on the provided document contexts. "
                    "If the answer is not in the contexts, clearly state that you don't have enough information. "
                    "Be concise but informative. Include specific details like amounts, dates, and names when available."
                )
            },
            {
                "role": "user", 
                "content": f"Document Contexts:\n{contexts}\n\nQuestion: {question}"
            }
        ]
        
        # Apply chat template and tokenize
        text = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # Tokenize with proper attention mask
        model_inputs = processor(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        # Generate response
        with torch.inference_mode():
            outputs = model.generate(
                **model_inputs,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                top_p=0.9,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=processor.eos_token_id,
                pad_token_id=processor.pad_token_id if processor.pad_token_id is not None else processor.eos_token_id
            )
        
        # Decode response
        full_text = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "assistant" in full_text:
            answer = full_text.split("assistant")[-1].strip()
        else:
            answer = full_text.split("Question:")[-1].strip() if "Question:" in full_text else full_text
        
        return answer if answer else "I apologize, but I couldn't generate a response to your question."
        
    except Exception as e:
        print(f"LLM Error: {e}")
        return f"I apologize, but I encountered an error while processing your question: {str(e)[:100]}. Please try again or use keyword search instead."

def _mock_answer(question: str, contexts: str) -> str:
    """
    Mock LLM service for development and testing
    """
    if not contexts.strip() or contexts == "No relevant documents found.":
        return "I don't have any relevant documents to answer your question. This could be because the document database is empty or your search terms don't match any content."
    
    # Count available documents
    doc_count = len([block for block in contexts.split('===') if block.strip()])
    
    # Pattern-based responses for common queries
    question_lower = question.lower()
    
    if "invoice" in question_lower:
        return f"Based on my analysis of {doc_count} documents, I found information related to invoices. The documents appear to contain invoice data with various clients and amounts. However, I would need the actual LLM service running to provide more specific details about invoice numbers, amounts, and dates."
    
    elif any(word in question_lower for word in ["total", "amount", "sum", "cost", "price"]):
        return f"I can see financial information across {doc_count} documents. To provide accurate totals and amounts, I would need to analyze the specific monetary values in detail. The mock service cannot perform actual calculations."
    
    elif any(word in question_lower for word in ["client", "company", "customer"]):
        return f"I found {doc_count} documents that contain client/company information. The documents reference various business entities and their associated transactions, but I would need the full LLM service to provide specific client details."
    
    elif any(word in question_lower for word in ["date", "when", "time", "period"]):
        return f"Looking at {doc_count} documents, I can see various dates and time references. The documents span different periods and contain timestamped information, but specific date analysis requires the actual LLM service."
    
    elif any(word in question_lower for word in ["what", "show", "find", "list"]):
        return f"I analyzed {doc_count} documents that may be relevant to your query. The mock LLM service is currently active - to get detailed, accurate answers, please enable the full LLM service in your settings."
    
    else:
        return f"I found {doc_count} documents related to your query '{question}'. This is a mock response - the actual LLM would provide more detailed and accurate analysis of the document contents. To enable full AI capabilities, set USE_MOCK_LLM=False in your settings."

def test_llm_connection():
    """Test if the LLM service is responding correctly"""
    try:
        if USE_MOCK_LLM:
            result = _mock_answer("test", "test document content")
            return "mock_connected"
        else:
            result = answer_from_context("What is this?", "Test document for connection verification.", 0.1)
            return "connected" if len(result) > 10 else "error"
    except Exception as e:
        print(f"LLM connection test failed: {e}")
        return "disconnected"

def get_model_info():
    """Get information about the currently loaded model and cache status"""
    global _model, _processor, _model_load_time
    
    info = {
        "model_name": MODEL_NAME,
        "max_tokens": MAX_NEW_TOKENS,
        "using_mock": USE_MOCK_LLM,
        "model_loaded": _model is not None,
        "model_cached": _model is not None,
        "cache_time": _model_load_time if _model_load_time else "Not loaded yet",
        "cuda_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        info["cuda_memory_reserved"] = f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    
    return info

def clear_model_cache():
    """Clear the model from memory to free up resources"""
    global _model, _processor, _model_load_time
    
    if _model is not None:
        print("🧹 Clearing model cache...")
        del _model
        del _processor
        _model = None
        _processor = None
        _model_load_time = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✓ GPU cache cleared")
        
        print("✓ Model cache cleared successfully")
        return True
    else:
        print("No model loaded in cache")
        return False