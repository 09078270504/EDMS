import os
import io
import gc
import time
import re
import logging
import tempfile
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import numpy as np
from PIL import Image, ImageEnhance
import easyocr
import fitz  # PyMuPDF
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """For Qwen2-VL starting engine"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._initialized = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.temp_dir = tempfile.mkdtemp()
        
        if self.device == "cuda":
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Using GPU with {memory_gb:.1f}GB memory")
        else:
            logger.info("Using CPU")
    
    def get_qwen_model(self, model_size="7B"):
        """Get Qwen2-VL model"""
        if not QWEN_AVAILABLE:
            return None, None
            
        model_key = f"qwen2vl_{model_size}"
        
        if model_key not in self.models:
            with self._lock:
                if model_key not in self.models:
                    try:
                        logger.info(f"Loading Qwen2-VL {model_size}")
                        model_name = f"Qwen/Qwen2-VL-{model_size}-Instruct"
                        
                        if self.device == "cuda" and model_size in ["7B", "72B"]:
                            # Use 4-bit quantization for larger models
                            from transformers import BitsAndBytesConfig
                            config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4"
                            )
                            model = Qwen2VLForConditionalGeneration.from_pretrained(
                                model_name, quantization_config=config,
                                device_map="auto", trust_remote_code=True
                            )
                        else:
                            dtype = torch.float16 if self.device == "cuda" else torch.float32
                            model = Qwen2VLForConditionalGeneration.from_pretrained(
                                model_name, torch_dtype=dtype,
                                device_map=self.device, trust_remote_code=True
                            )
                        
                        processor = AutoProcessor.from_pretrained(model_name)
                        model.eval()
                        
                        self.models[model_key] = (model, processor)
                        logger.info("Qwen2-VL loaded successfully")
                        
                    except Exception as e:
                        logger.error(f"Failed to load Qwen2-VL: {e}")
                        self.models[model_key] = (None, None)
        
        return self.models.get(model_key, (None, None))
    
    def get_easyocr_reader(self, languages=['en']):
        """Get EasyOCR reader"""
        model_key = f"easyocr_{'_'.join(languages)}"
        
        if model_key not in self.models:
            with self._lock:
                if model_key not in self.models:
                    try:
                        logger.info("Loading EasyOCR")
                        reader = easyocr.Reader(
                            languages, gpu=True, download_enabled=True,
                            model_storage_directory=os.path.join(self.temp_dir, 'easyocr')
                        )
                        self.models[model_key] = reader
                        logger.info("EasyOCR loaded")
                    except Exception as e:
                        logger.error(f"Failed to load EasyOCR: {e}")
                        self.models[model_key] = None
        
        return self.models.get(model_key)
    
    def cleanup(self):
        """Cleanup resources"""
        with self._lock:
            for value in self.models.values():
                if value is not None:
                    del value
            self.models.clear()
            
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()



def cleanup_models():
    """Clean up GPU memory and unload models to prevent memory leaks"""
    try:
        ModelManager().cleanup()
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")


class OCRProcessor:
    def __init__(self, languages=['en'], model_size="7B", device="auto", gpu=None, verbose=False, confidence_threshold=0.5, **kwargs):
        self.languages = languages
        self.model_size = model_size
        self.device = device  # Accept but delegate to ModelManager
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        self.model_manager = ModelManager()
        
        self.layout_prompt = (
            "Extract ALL visible text preserving layout and structure.\n\n"
            "Keep original:\n"
            "- Line breaks, spacing, section order\n"  
            "- Table alignment via spacing\n"
            "- Headers, bullets, numbers, dates, currencies (₱,$,PHP)\n"
            "- Names, addresses, IDs, fine print\n\n"
            "Output clean text only, no commentary."
            
        )
        
        self.simple_prompt = "Extract all visible text from this image as plain text."
        
        logger.info("OCR processor initialized")
    
    def _strip_llm_preamble(self, s: str) -> str:
        s = s.strip()
        s = re.sub(r'^\s*(here\s+is\s+the\s+extracted\s+text.*?:)\s*', '', s, flags=re.I)
        s = re.sub(r'^\s*(here\s+is\s+the\s+text.*?:)\s*', '', s, flags=re.I)
        return s

    def extract_text_from_file(self, file_path: str) -> Dict[str, Any]:
        """Main extraction method with lean pipeline"""
        start_time = time.time()
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if file_path.suffix.lower() == '.pdf':
                result = self._process_pdf(str(file_path))
            elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                result = self._process_image(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            cleaned_text = self._remove_bold_syntax(result['text'])

            result.update({
                'processing_time_seconds': time.time() - start_time,
                'file_path': str(file_path),
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'text': cleaned_text
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return self._error_result(str(e), file_path)
        
    def _remove_bold_syntax(self, text: str) -> str:
        """Remove markdown-like bold text syntax (e.g., '**bold**')"""
        return text.replace('**', '')
    
    def _process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Streamlined PDF processing pipeline"""
        # Step 1: Convert to images
        images = self._pdf_to_images(pdf_path)
        if not images:
            return self._error_result("No images extracted from PDF", pdf_path)
        
        # Step 2: Try Qwen2-VL with layout preservation
        if QWEN_AVAILABLE:
            result = self._try_qwen_extraction(images, use_layout_prompt=True)
            if result['success'] and len(result['text'].strip()) >= 50:
                result['method'] = 'qwen2vl_layout'
                return result
            
        # Step 3: Retry Qwen with simple prompt if layout failed
        logger.debug("Layout extraction insufficient, trying simple prompt")
        result = self._try_qwen_extraction(images, use_layout_prompt=False) 
        if result['success'] and len(result['text'].strip()) >= 20:
            result['method'] = 'qwen2vl_simple'
            return result
        
        # Step 4: Fallback to EasyOCR
        result = self._try_easyocr_extraction(images)
        if result['success']:
            result['method'] = 'easyocr'
            return result
        
        # Step 5: Last resort - basic text extraction
        result = self._try_basic_extraction(pdf_path)
        result['method'] = 'basic'
        return result
    
    def _process_image(self, image_path: str) -> Dict[str, Any]:
        """Streamlined image processing"""
        try:
            image = Image.open(image_path)
            images = [self._enhance_image(image)]
            
            # Try Qwen first
            if QWEN_AVAILABLE:
                result = self._try_qwen_extraction(images, use_layout_prompt=True)
                if result['success']:
                    result['method'] = 'qwen2vl_layout'
                    return result
            
            # Fallback to EasyOCR
            result = self._try_easyocr_extraction(images)
            result['method'] = 'easyocr'
            return result
            
        except Exception as e:
            return self._error_result(str(e), image_path)
    
    def _try_qwen_extraction(self, images: List[Image.Image], use_layout_prompt: bool) -> Dict[str, Any]:
        """Try Qwen2-VL extraction"""
        try:
            model, processor = self.model_manager.get_qwen_model(self.model_size)
            if not model or not processor:
                return self._error_result("Qwen2-VL not available")
            
            all_text = []
            all_confidences = []
            
            prompt = self.layout_prompt if use_layout_prompt else self.simple_prompt
            
            for page_num, image in enumerate(images):
                try:
                    result = self._qwen_inference(image, model, processor, prompt)
                    page_text = (result.get('text') or '').strip()
                    if result.get('success') and len(page_text) >= 5:
                        # Add page header
                        page_block = f"--- Page {page_num + 1} ---\n{page_text}"
                        all_text.append(page_block)
                        all_confidences.append(result.get('confidence', 0.75))
                except Exception as e:
                    logger.warning(f"Qwen page {page_num + 1} failed: {e}")
                    continue
            
            if all_text:
                combined_text = "\n\n".join(all_text)
                avg_confidence = sum(all_confidences) / len(all_confidences)
                
                return {
                    'text': combined_text,
                    'confidence': avg_confidence,
                    'success': True,
                    'error': None,
                    'pages_processed': len(all_text)
                }
            
            return self._error_result("No text extracted by Qwen")
            
        except Exception as e:
            return self._error_result(f"Qwen extraction failed: {e}")

    
    def _qwen_inference(self, image: Image.Image, model, processor, prompt: str) -> Dict[str, Any]:
        """Fast Qwen inference without temp files"""
        try:
            messages = [{
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            # Process inputs
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            )
            
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            
            # Fast generation settings
            start_time = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,  # Reduced for speed
                    do_sample=False,      # Deterministic for consistency
                    temperature=0.1,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    repetition_penalty=1.05
                )
            
            generation_time = time.time() - start_time
            
            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in
                zip(inputs.input_ids, generated_ids)
            ]
            
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            response = self._strip_llm_preamble(response)
            
            confidence = self._calculate_confidence(response, generation_time)
            
            return {
                'text': response.strip(),
                'confidence': confidence,
                'success': bool(response.strip()) and len(response.strip()) >= 3,
                'error': None
            }
            
        except Exception as e:
            return self._error_result(f"Qwen inference failed: {e}")
    
    def _try_easyocr_extraction(self, images: List[Image.Image]) -> Dict[str, Any]:
        """Try EasyOCR extraction with layout preservation"""
        try:
            reader = self.model_manager.get_easyocr_reader(self.languages)
            if not reader:
                return self._error_result("EasyOCR not available")
            
            all_text = []
            all_confidences = []
            
            for page_num, image in enumerate(images):
                try:
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    img_array = np.array(image)
                    results = reader.readtext(img_array, detail=1, paragraph=True)
                    
                    # Handle both (bbox, text, conf) and (bbox, text) formats
                    text_parts = []
                    confidences = []
                    
                    for result in results:
                        if len(result) == 3:
                            bbox, text, conf = result
                        else:
                            bbox, text = result
                            conf = 0.6  # Default confidence
                        
                        if conf > 0.3 and text.strip():
                            text_parts.append(text)
                            confidences.append(conf)
                    
                    if text_parts:
                        page_text = self._reconstruct_layout(results)
                        page_confidence = sum(confidences) / len(confidences) if confidences else 0.6
                        
                        all_text.append(f"--- Page {page_num+1} ---\n{page_text.lstrip()}")
                        combined_text = "\n\n".join(all_text)
                
                except Exception as e:
                    logger.warning(f"EasyOCR page {page_num + 1} failed: {e}")
                    continue
            
            if all_text:
                combined_text = "\n\n--- Page Break ---\n\n".join(all_text)
                avg_confidence = sum(all_confidences) / len(all_confidences)
                
                return {
                    'text': combined_text,
                    'confidence': avg_confidence,
                    'success': True,
                    'error': None,
                    'pages_processed': len(all_text)
                }
            
            return self._error_result("No text extracted by EasyOCR")
            
        except Exception as e:
            return self._error_result(f"EasyOCR failed: {e}")
    
    def _try_basic_extraction(self, pdf_path: str) -> Dict[str, Any]:
        """Basic PDF text extraction"""
        try:
            doc = fitz.open(pdf_path)
            all_text = []
            
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text.strip():
                        all_text.append(text)
                except Exception as e:
                    logger.warning(f"Basic extraction page {page_num} failed: {e}")
                    continue
            
            doc.close()
            
            if all_text:
                combined_text = "\n\n--- Page Break ---\n\n".join(all_text)
                return {
                    'text': combined_text,
                    'confidence': 0.6,
                    'success': True,
                    'error': None,
                    'pages_processed': len(all_text)
                }
            
            return self._error_result("No text found in PDF")
            
        except Exception as e:
            return self._error_result(f"Basic extraction failed: {e}")
    
    def _pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF to images efficiently"""
        images = []
        try:
            doc = fitz.open(pdf_path)
            zoom = 220 / 72  # 220 DPI
            mat = fitz.Matrix(zoom, zoom)
            
            for page_num in range(min(len(doc), 50)):
                try:
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    img_data = pix.pil_tobytes(format="PNG")
                    img = Image.open(io.BytesIO(img_data))
                    
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    if img.size[0] >= 100 and img.size[1] >= 100:
                        images.append(img)
                        
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num}: {e}")
                    continue
            
            doc.close()
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            
            # Fallback to pdf2image if available
            if PDF2IMAGE_AVAILABLE:
                try:
                    logger.info("Trying pdf2image fallback")
                    images = convert_from_path(pdf_path, dpi=220, fmt='RGB')
                    # Convert to PIL format expected by rest of pipeline
                    images = [img.convert('RGB') for img in images if img.size[0] >= 100 and img.size[1] >= 100]
                except Exception as e2:
                    logger.error(f"pdf2image fallback failed: {e2}")
        
        return images
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Light image enhancement"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too small
            width, height = image.size
            if width < 1200 or height < 900:
                scale = max(1200/width, 900/height, 1.0)
                new_size = (int(width * scale), int(height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Resize if too large
            if width > 4000 or height > 4000:
                image.thumbnail((4000, 4000), Image.Resampling.LANCZOS)
            
            # Light enhancement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            
            enhancer = ImageEnhance.Sharpness(image)  
            image = enhancer.enhance(1.05)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def _reconstruct_layout(self, ocr_results: List) -> str:
        """Simple layout reconstruction from OCR results"""
        if not ocr_results:
            return ""
        
        lines = {}   
        for result in ocr_results:
            # Handle both formats safely
            if len(result) == 3:
                bbox, text, conf = result
            else:
                bbox, text = result
                conf = 0.6
            
            if conf > 0.3 and text.strip():
                y_center = (bbox[0][1] + bbox[2][1]) / 2
                line_key = round(y_center / 10) * 10
                
                if line_key not in lines:
                    lines[line_key] = []
                
                x_left = bbox[0][0]
                lines[line_key].append((x_left, text))
        
        sorted_lines = []         # Reconstruct text line by line
        for y in sorted(lines.keys()):
            line_texts = sorted(lines[y], key=lambda x: x[0])
            line_text = " ".join([text for _, text in line_texts])
            sorted_lines.append(line_text)
        
        return "\n".join(sorted_lines)
    
    def _calculate_confidence(self, response: str, generation_time: float) -> float:
        """Simple confidence calculation"""
        if not response or len(response.strip()) < 3:
            return 0.1
        
        base_confidence = 0.75
        length_bonus = min(len(response) / 1000, 0.15)
        indicators = ['date', 'total', 'amount', '$', '₱', 'php', 'receipt', 'invoice']
        structure_count = sum(1 for indicator in indicators 
                            if indicator.lower() in response.lower())
        structure_bonus = min(structure_count * 0.02, 0.1)
        
        return min(base_confidence + length_bonus + structure_bonus, 0.95)
    
    def _error_result(self, error_msg: str, file_path: str = None) -> Dict[str, Any]:
        """Create error result"""
        return {
            'text': '',
            'confidence': 0.0,
            'success': False,
            'error': error_msg,
            'method': 'error',
            'file_path': str(file_path) if file_path else None,
            'processing_time_seconds': 0.0
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.model_manager.cleanup()

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]: 
        """Backward compatible PDF extraction"""
        result = self.extract_text_from_file(pdf_path)
        return {
            'text': result.get('text', ''),
            'success': result.get('success', False),
            'confidence': result.get('confidence', 0.0),
            'method': result.get('method', 'unknown'),
            'error': result.get('error'),
            'processing_time': result.get('processing_time_seconds', 0.0)
        }