HYBRID_OCR_CONFIG = {
    'confidence_threshold': 0.7,  # Adjust based on your document types
    'qwen_model_size': '7B',      # Options: '2B', '7B', '72B' 
    'device': 'cuda',             # 'cuda' or 'cpu'
    'max_generation_tokens': 2048,
    'easyocr_languages': ['en'],
    'easyocr_gpu': True
}