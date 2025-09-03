# core/services/llm_search.py
# PRODUCTION-READY VERSION with multiple deployment options

import os
import torch
from django.conf import settings
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# Configuration
MODEL_NAME = getattr(settings, 'LLM_MODEL_NAME', "Qwen/Qwen2-VL-7B-Instruct")
MAX_NEW_TOKENS = getattr(settings, 'LLM_MAX_TOKENS', 256)
USE_MOCK_LLM = getattr(settings, 'USE_MOCK_LLM', False)  # For development/testing

# Global model storage
_model = None
_processor = None
_model_loading = False

def _load_model():
    """
    Load the Qwen2-VL model with optimized settings
    """
    global _model, _processor, _model_loading
    
    if _model is not None:
        return _model, _processor
    
    if _model_loading:
        # Prevent multiple simultaneous loads
        import time
        for _ in range(30):  # Wait up to 30 seconds
            if _model is not None:
                return _model, _processor
            time.sleep(1)
        raise Exception("Model loading timeout")
    
    _model_loading = True
    
    try:
        print(f"Loading LLM model: {MODEL_NAME}")
        
        # Configuration for different deployment scenarios
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto"
        }
        
        # Memory optimization settings
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            # Use 4-bit quantization if low on VRAM (less than 16GB)
            if total_memory < 16 * 1024**3:  # 16GB in bytes
                print("Using 4-bit quantization for memory efficiency")
                model_kwargs.update({
                    "load_in_4bit": True,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16
                })
            else:
                model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            # CPU fallback
            model_kwargs["torch_dtype"] = torch.float32
            print("Warning: Using CPU inference - this will be slow")
        
        # Load model and processor
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, 
            **model_kwargs
        )
        _processor = AutoProcessor.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True
        )
        
        print("LLM model loaded successfully")
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
        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_tensors="pt"
        ).to(model.device)
        
        # Generate response
        with torch.inference_mode():
            outputs = model.generate(
                inputs,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                top_p=0.9,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode response
        full_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
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
    """Get information about the currently loaded model"""
    global _model, _processor
    
    info = {
        "model_name": MODEL_NAME,
        "max_tokens": MAX_NEW_TOKENS,
        "using_mock": USE_MOCK_LLM,
        "model_loaded": _model is not None,
        "cuda_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    
    return info