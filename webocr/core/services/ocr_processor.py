#services/ocr_processor.py - for ocr extraction
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
from django.utils import timezone
import pytz

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

# Philippines timezone
PH_TZ = pytz.timezone('Asia/Manila')

def get_ph_time():
    """Get current Philippines time"""
    return timezone.now().astimezone(PH_TZ)

def format_ph_time(dt=None):
    """Format Philippines time for logging"""
    if dt is None:
        dt = get_ph_time()
    return dt.strftime('%Y-%m-%d %H:%M:%S')

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
        init_start_time = get_ph_time()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.temp_dir = tempfile.mkdtemp()
        
        if self.device == "cuda":
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"[{format_ph_time()}] Using GPU with {memory_gb:.1f}GB memory")
        else:
            logger.info(f"[{format_ph_time()}] Using CPU")
        
        init_complete_time = get_ph_time()
        init_duration = (init_complete_time - init_start_time).total_seconds()
        logger.info(f"[{format_ph_time(init_complete_time)}] ModelManager initialized in {init_duration:.3f}s")
    
    def get_qwen_model(self, model_size="7B"):
        """Get Qwen2-VL model"""
        if not QWEN_AVAILABLE:
            logger.warning(f"[{format_ph_time()}] Qwen2-VL not available")
            return None, None
            
        model_key = f"qwen2vl_{model_size}"
        
        if model_key not in self.models:
            with self._lock:
                if model_key not in self.models:
                    try:
                        load_start_time = get_ph_time()
                        logger.info(f"[{format_ph_time(load_start_time)}] Loading Qwen2-VL {model_size}")
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
                        
                        load_complete_time = get_ph_time()
                        load_duration = (load_complete_time - load_start_time).total_seconds()
                        logger.info(f"[{format_ph_time(load_complete_time)}] Qwen2-VL loaded successfully in {load_duration:.2f}s")
                        
                    except Exception as e:
                        error_time = get_ph_time()
                        logger.error(f"[{format_ph_time(error_time)}] Failed to load Qwen2-VL: {e}")
                        self.models[model_key] = (None, None)
        
        return self.models.get(model_key, (None, None))
    
    def get_easyocr_reader(self, languages=['en']):
        """Get EasyOCR reader"""
        model_key = f"easyocr_{'_'.join(languages)}"
        
        if model_key not in self.models:
            with self._lock:
                if model_key not in self.models:
                    try:
                        load_start_time = get_ph_time()
                        logger.info(f"[{format_ph_time(load_start_time)}] Loading EasyOCR")
                        reader = easyocr.Reader(
                            languages, gpu=True, download_enabled=True,
                            model_storage_directory=os.path.join(self.temp_dir, 'easyocr')
                        )
                        self.models[model_key] = reader
                        
                        load_complete_time = get_ph_time()
                        load_duration = (load_complete_time - load_start_time).total_seconds()
                        logger.info(f"[{format_ph_time(load_complete_time)}] EasyOCR loaded in {load_duration:.2f}s")
                    except Exception as e:
                        error_time = get_ph_time()
                        logger.error(f"[{format_ph_time(error_time)}] Failed to load EasyOCR: {e}")
                        self.models[model_key] = None
        
        return self.models.get(model_key)
    
    def cleanup(self):
        """Cleanup resources"""
        cleanup_start_time = get_ph_time()
        logger.info(f"[{format_ph_time(cleanup_start_time)}] Starting ModelManager cleanup...")
        
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
            
            cleanup_complete_time = get_ph_time()
            cleanup_duration = (cleanup_complete_time - cleanup_start_time).total_seconds()
            logger.info(f"[{format_ph_time(cleanup_complete_time)}] ModelManager cleanup completed in {cleanup_duration:.2f}s")


def cleanup_models():
    """Clean up GPU memory and unload models to prevent memory leaks"""
    try:
        cleanup_start_time = get_ph_time()
        logger.info(f"[{format_ph_time(cleanup_start_time)}] Starting global model cleanup...")
        ModelManager().cleanup()
        cleanup_complete_time = get_ph_time()
        cleanup_duration = (cleanup_complete_time - cleanup_start_time).total_seconds()
        logger.info(f"[{format_ph_time(cleanup_complete_time)}] Global model cleanup completed in {cleanup_duration:.2f}s")
    except Exception as e:
        error_time = get_ph_time()
        logger.warning(f"[{format_ph_time(error_time)}] Cleanup failed: {e}")


class OCRProcessor:
    def __init__(self, languages=['en'], model_size="7B", device="auto", gpu=None, verbose=False, confidence_threshold=0.5, **kwargs):
        init_start_time = get_ph_time()
        logger.info(f"[{format_ph_time(init_start_time)}] Initializing OCRProcessor...")
        
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
        
        init_complete_time = get_ph_time()
        init_duration = (init_complete_time - init_start_time).total_seconds()
        logger.info(f"[{format_ph_time(init_complete_time)}] OCRProcessor initialized in {init_duration:.3f}s")
    
    def _strip_llm_preamble(self, s: str) -> str:
        s = s.strip()
        s = re.sub(r'^\s*(here\s+is\s+the\s+extracted\s+text.*?:)\s*', '', s, flags=re.I)
        s = re.sub(r'^\s*(here\s+is\s+the\s+text.*?:)\s*', '', s, flags=re.I)
        return s

    def extract_text_from_file(self, file_path: str) -> Dict[str, Any]:
        """Main extraction method with lean pipeline"""
        extract_start_time = get_ph_time()
        start_time = time.time()

        logger.info(f"[{format_ph_time(extract_start_time)}] Starting text extraction from: **{Path(file_path).name}**")

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

            processing_time = time.time() - start_time
            extract_complete_time = get_ph_time()
            
            result.update({
                'processing_time_seconds': processing_time,
                'file_path': str(file_path),
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'text': cleaned_text
            })
            
            logger.info(f"[{format_ph_time(extract_complete_time)}] ✅ Text extraction completed in {processing_time:.2f}s - {len(cleaned_text)} chars, {result.get('confidence', 0):.1f}% confidence")
            
            return result
            
        except Exception as e:
            error_time = get_ph_time()
            processing_time = time.time() - start_time
            logger.error(f"[{format_ph_time(error_time)}] ❌ Processing failed in {processing_time:.2f}s: {e}")
            return self._error_result(str(e), file_path)
        
    def _remove_bold_syntax(self, text: str) -> str:
        """Remove markdown-like bold text syntax (e.g., '**bold**')"""
        return text.replace('**', '')
    
    def _process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Streamlined PDF processing pipeline"""
        pdf_start_time = get_ph_time()
        logger.debug(f"[{format_ph_time(pdf_start_time)}] Processing PDF: {Path(pdf_path).name}")
        
        # Step 1: Convert to images
        images = self._pdf_to_images(pdf_path)
        if not images:
            error_time = get_ph_time()
            logger.error(f"[{format_ph_time(error_time)}] No images extracted from PDF")
            return self._error_result("No images extracted from PDF", pdf_path)
        
        # Step 2: Try Qwen2-VL with layout preservation
        if QWEN_AVAILABLE:
            qwen_start_time = get_ph_time()
            result = self._try_qwen_extraction(images, use_layout_prompt=True)
            if result['success'] and len(result['text'].strip()) >= 50:
                result['method'] = 'qwen2vl_layout'
                qwen_complete_time = get_ph_time()
                qwen_duration = (qwen_complete_time - qwen_start_time).total_seconds()
                logger.debug(f"[{format_ph_time(qwen_complete_time)}] ✅ Qwen layout extraction succeeded in {qwen_duration:.2f}s")
                return result
            
        # Step 3: Retry Qwen with simple prompt if layout failed
        logger.debug(f"[{format_ph_time()}] Layout extraction insufficient, trying simple prompt")
        result = self._try_qwen_extraction(images, use_layout_prompt=False) 
        if result['success'] and len(result['text'].strip()) >= 20:
            result['method'] = 'qwen2vl_simple'
            logger.debug(f"[{format_ph_time()}] ✅ Qwen simple extraction succeeded")
            return result
        
        # Step 4: Fallback to EasyOCR
        easyocr_start_time = get_ph_time()
        result = self._try_easyocr_extraction(images)
        if result['success']:
            result['method'] = 'easyocr'
            easyocr_complete_time = get_ph_time()
            easyocr_duration = (easyocr_complete_time - easyocr_start_time).total_seconds()
            logger.debug(f"[{format_ph_time(easyocr_complete_time)}] ✅ EasyOCR extraction succeeded in {easyocr_duration:.2f}s")
            return result
        
        # Step 5: Last resort - basic text extraction
        basic_start_time = get_ph_time()
        result = self._try_basic_extraction(pdf_path)
        result['method'] = 'basic'
        basic_complete_time = get_ph_time()
        basic_duration = (basic_complete_time - basic_start_time).total_seconds()
        logger.debug(f"[{format_ph_time(basic_complete_time)}] Basic extraction completed in {basic_duration:.2f}s")
        return result
    
    def _process_image(self, image_path: str) -> Dict[str, Any]:
        """Streamlined image processing"""
        image_start_time = get_ph_time()
        logger.debug(f"[{format_ph_time(image_start_time)}] Processing image: {Path(image_path).name}")
        
        try:
            image = Image.open(image_path)
            images = [self._enhance_image(image)]
            
            # Try Qwen first
            if QWEN_AVAILABLE:
                result = self._try_qwen_extraction(images, use_layout_prompt=True)
                if result['success']:
                    result['method'] = 'qwen2vl_layout'
                    image_complete_time = get_ph_time()
                    image_duration = (image_complete_time - image_start_time).total_seconds()
                    logger.debug(f"[{format_ph_time(image_complete_time)}] ✅ Image Qwen extraction succeeded in {image_duration:.2f}s")
                    return result
            
            # Fallback to EasyOCR
            result = self._try_easyocr_extraction(images)
            result['method'] = 'easyocr'
            image_complete_time = get_ph_time()
            image_duration = (image_complete_time - image_start_time).total_seconds()
            logger.debug(f"[{format_ph_time(image_complete_time)}] Image EasyOCR extraction completed in {image_duration:.2f}s")
            return result
            
        except Exception as e:
            error_time = get_ph_time()
            logger.error(f"[{format_ph_time(error_time)}] Image processing error: {e}")
            return self._error_result(str(e), image_path)
    
    def _try_qwen_extraction(self, images: List[Image.Image], use_layout_prompt: bool) -> Dict[str, Any]:
        """Try Qwen2-VL extraction"""
        qwen_try_start_time = get_ph_time()
        
        try:
            model, processor = self.model_manager.get_qwen_model(self.model_size)
            if not model or not processor:
                logger.debug(f"[{format_ph_time()}] Qwen2-VL not available")
                return self._error_result("Qwen2-VL not available")
            
            all_text = []
            all_confidences = []
            
            prompt = self.layout_prompt if use_layout_prompt else self.simple_prompt
            prompt_type = "layout" if use_layout_prompt else "simple"
            
            for page_num, image in enumerate(images):
                page_start_time = get_ph_time()
                try:
                    result = self._qwen_inference(image, model, processor, prompt)
                    page_text = (result.get('text') or '').strip()
                    if result.get('success') and len(page_text) >= 5:
                        # Add page header
                        page_block = f"--- Page {page_num + 1} ---\n{page_text}"
                        all_text.append(page_block)
                        all_confidences.append(result.get('confidence', 0.75))
                        
                        page_complete_time = get_ph_time()
                        page_duration = (page_complete_time - page_start_time).total_seconds()
                        logger.debug(f"[{format_ph_time(page_complete_time)}] Page {page_num + 1} Qwen {prompt_type} extraction: {len(page_text)} chars in {page_duration:.2f}s")
                except Exception as e:
                    error_time = get_ph_time()
                    logger.warning(f"[{format_ph_time(error_time)}] Qwen page {page_num + 1} failed: {e}")
                    continue
            
            if all_text:
                combined_text = "\n\n".join(all_text)
                avg_confidence = sum(all_confidences) / len(all_confidences)
                
                qwen_try_complete_time = get_ph_time()
                qwen_try_duration = (qwen_try_complete_time - qwen_try_start_time).total_seconds()
                logger.debug(f"[{format_ph_time(qwen_try_complete_time)}] ✅ Qwen {prompt_type} extraction completed in {qwen_try_duration:.2f}s - {len(combined_text)} chars")
                
                return {
                    'text': combined_text,
                    'confidence': avg_confidence,
                    'success': True,
                    'error': None,
                    'pages_processed': len(all_text)
                }
            
            logger.debug(f"[{format_ph_time()}] No text extracted by Qwen")
            return self._error_result("No text extracted by Qwen")
            
        except Exception as e:
            error_time = get_ph_time()
            logger.error(f"[{format_ph_time(error_time)}] Qwen extraction failed: {e}")
            return self._error_result(f"Qwen extraction failed: {e}")

    
    def _qwen_inference(self, image: Image.Image, model, processor, prompt: str) -> Dict[str, Any]:
        """Fast Qwen inference without temp files"""
        inference_start_time = get_ph_time()
        
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
            
            inference_complete_time = get_ph_time()
            inference_duration = (inference_complete_time - inference_start_time).total_seconds()
            logger.debug(f"[{format_ph_time(inference_complete_time)}] Qwen inference completed in {inference_duration:.3f}s (generation: {generation_time:.3f}s)")
            
            return {
                'text': response.strip(),
                'confidence': confidence,
                'success': bool(response.strip()) and len(response.strip()) >= 3,
                'error': None
            }
            
        except Exception as e:
            error_time = get_ph_time()
            logger.error(f"[{format_ph_time(error_time)}] Qwen inference failed: {e}")
            return self._error_result(f"Qwen inference failed: {e}")
    
    def _try_easyocr_extraction(self, images: List[Image.Image]) -> Dict[str, Any]:
        """Try EasyOCR extraction with layout preservation"""
        easyocr_try_start_time = get_ph_time()
        
        try:
            reader = self.model_manager.get_easyocr_reader(self.languages)
            if not reader:
                logger.debug(f"[{format_ph_time()}] EasyOCR not available")
                return self._error_result("EasyOCR not available")
            
            all_text = []
            all_confidences = []
            
            for page_num, image in enumerate(images):
                page_start_time = get_ph_time()
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
                        all_confidences.append(page_confidence)
                        
                        page_complete_time = get_ph_time()
                        page_duration = (page_complete_time - page_start_time).total_seconds()
                        logger.debug(f"[{format_ph_time(page_complete_time)}] Page {page_num+1} EasyOCR: {len(page_text)} chars in {page_duration:.2f}s")
                
                except Exception as e:
                    error_time = get_ph_time()
                    logger.warning(f"[{format_ph_time(error_time)}] EasyOCR page {page_num + 1} failed: {e}")
                    continue
            
            if all_text:
                combined_text = "\n\n--- Page Break ---\n\n".join(all_text)
                avg_confidence = sum(all_confidences) / len(all_confidences)
                
                easyocr_try_complete_time = get_ph_time()
                easyocr_try_duration = (easyocr_try_complete_time - easyocr_try_start_time).total_seconds()
                logger.debug(f"[{format_ph_time(easyocr_try_complete_time)}] ✅ EasyOCR extraction completed in {easyocr_try_duration:.2f}s - {len(combined_text)} chars")
                
                return {
                    'text': combined_text,
                    'confidence': avg_confidence,
                    'success': True,
                    'error': None,
                    'pages_processed': len(all_text)
                }
            
            logger.debug(f"[{format_ph_time()}] No text extracted by EasyOCR")
            return self._error_result("No text extracted by EasyOCR")
            
        except Exception as e:
            error_time = get_ph_time()
            logger.error(f"[{format_ph_time(error_time)}] EasyOCR failed: {e}")
            return self._error_result(f"EasyOCR failed: {e}")
    
    def _try_basic_extraction(self, pdf_path: str) -> Dict[str, Any]:
        """Basic PDF text extraction"""
        basic_start_time = get_ph_time()
        
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
                    logger.warning(f"[{format_ph_time()}] Basic extraction page {page_num} failed: {e}")
                    continue
            
            doc.close()
            
            if all_text:
                combined_text = "\n\n--- Page Break ---\n\n".join(all_text)
                
                basic_complete_time = get_ph_time()
                basic_duration = (basic_complete_time - basic_start_time).total_seconds()
                logger.debug(f"[{format_ph_time(basic_complete_time)}] ✅ Basic extraction completed in {basic_duration:.2f}s - {len(combined_text)} chars")
                
                return {
                    'text': combined_text,
                    'confidence': 0.6,
                    'success': True,
                    'error': None,
                    'pages_processed': len(all_text)
                }
            
            logger.debug(f"[{format_ph_time()}] No text found in PDF")
            return self._error_result("No text found in PDF")
            
        except Exception as e:
            error_time = get_ph_time()
            logger.error(f"[{format_ph_time(error_time)}] Basic extraction failed: {e}")
            return self._error_result(f"Basic extraction failed: {e}")
    
    def _pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF to images efficiently"""
        convert_start_time = get_ph_time()
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
                    logger.warning(f"[{format_ph_time()}] Failed to extract page {page_num}: {e}")
                    continue
            
            doc.close()
            
            convert_complete_time = get_ph_time()
            convert_duration = (convert_complete_time - convert_start_time).total_seconds()
            logger.debug(f"[{format_ph_time(convert_complete_time)}] PDF to images conversion completed in {convert_duration:.2f}s - {len(images)} pages")
            
        except Exception as e:
            error_time = get_ph_time()
            logger.error(f"[{format_ph_time(error_time)}] PDF processing failed: {e}")
            
            # Fallback to pdf2image if available
            if PDF2IMAGE_AVAILABLE:
                try:
                    fallback_start_time = get_ph_time()
                    logger.info(f"[{format_ph_time(fallback_start_time)}] Trying pdf2image fallback")
                    images = convert_from_path(pdf_path, dpi=220, fmt='RGB')
                    # Convert to PIL format expected by rest of pipeline
                    images = [img.convert('RGB') for img in images if img.size[0] >= 100 and img.size[1] >= 100]
                    
                    fallback_complete_time = get_ph_time()
                    fallback_duration = (fallback_complete_time - fallback_start_time).total_seconds()
                    logger.info(f"[{format_ph_time(fallback_complete_time)}] ✅ pdf2image fallback succeeded in {fallback_duration:.2f}s - {len(images)} pages")
                except Exception as e2:
                    error_time = get_ph_time()
                    logger.error(f"[{format_ph_time(error_time)}] pdf2image fallback failed: {e2}")
        
        return images
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Light image enhancement"""
        enhance_start_time = get_ph_time()
        
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            original_size = image.size
            
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
            
            enhance_complete_time = get_ph_time()
            enhance_duration = (enhance_complete_time - enhance_start_time).total_seconds()
            logger.debug(f"[{format_ph_time(enhance_complete_time)}] Image enhanced in {enhance_duration:.3f}s ({original_size} -> {image.size})")
            
            return image
            
        except Exception as e:
            error_time = get_ph_time()
            logger.warning(f"[{format_ph_time(error_time)}] Image enhancement failed: {e}")
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
        cleanup_start_time = get_ph_time()
        logger.info(f"[{format_ph_time(cleanup_start_time)}] Starting OCRProcessor cleanup...")
        self.model_manager.cleanup()
        cleanup_complete_time = get_ph_time()
        cleanup_duration = (cleanup_complete_time - cleanup_start_time).total_seconds()
        logger.info(f"[{format_ph_time(cleanup_complete_time)}] OCRProcessor cleanup completed in {cleanup_duration:.2f}s")

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