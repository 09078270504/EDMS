import os
import io
import gc
import time
import logging
import tempfile
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import re
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import easyocr
import fitz  # PyMuPDF

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SharedModelManager:
    """Manages shared model instances with proper synchronization"""
    
    def __init__(self):
        self._qwen_model = None
        self._qwen_processor = None
        self._easyocr_reader = None
        self._model_lock = threading.Lock()
        self._gpu_lock = threading.Lock()
        self._temp_dir = tempfile.mkdtemp()
        
    def get_qwen_model(self, device="auto", model_size="2B"):
        """Get shared Qwen2-VL model instance"""
        if self._qwen_model is None:
            with self._model_lock:
                if self._qwen_model is None:  # Double-check locking
                    self._load_qwen_model(device, model_size)
        return self._qwen_model, self._qwen_processor
    
    def get_easyocr_reader(self, languages=['en']):
        """Get shared EasyOCR reader instance"""
        if self._easyocr_reader is None:
            with self._model_lock:
                if self._easyocr_reader is None:  # Double-check locking
                    self._load_easyocr_reader(languages)
        return self._easyocr_reader
    
    def _load_qwen_model(self, device, model_size):
        """Load Qwen2-VL model once"""
        try:
            model_name = f"Qwen/Qwen2-VL-{model_size}-Instruct"
            logger.info(f"Loading shared Qwen2-VL model: {model_name}")
            
            # Setup device
            if device == "auto":
                if torch.cuda.is_available():
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    device = "cuda" if total_memory >= 8 else "cpu"
                else:
                    device = "cpu"
            
            # Load model with appropriate settings
            if device == "cuda" and model_size in ["7B", "72B"]:
                # Use quantization for larger models
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self._qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            elif device == "cuda":
                self._qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self._qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True
                )
            
            self._qwen_processor = AutoProcessor.from_pretrained(model_name)
            self._qwen_model.eval()
            
            logger.info(f"Qwen2-VL model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen2-VL model: {e}")
            self._qwen_model = None
            self._qwen_processor = None
    
    def _load_easyocr_reader(self, languages):
        """Load EasyOCR reader once"""
        try:
            logger.info("Loading shared EasyOCR reader...")
            # Use CPU for EasyOCR to avoid GPU conflicts
            self._easyocr_reader = easyocr.Reader(
                languages,
                gpu=False,  # Use CPU to avoid GPU memory conflicts
                download_enabled=True,
                model_storage_directory=os.path.join(self._temp_dir, 'easyocr_models')
            )
            logger.info("EasyOCR reader loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load EasyOCR reader: {e}")
            self._easyocr_reader = None
    
    def cleanup(self):
        """Cleanup resources"""
        with self._model_lock:
            if self._qwen_model:
                del self._qwen_model
                self._qwen_model = None
            if self._qwen_processor:
                del self._qwen_processor
                self._qwen_processor = None
            if self._easyocr_reader:
                del self._easyocr_reader
                self._easyocr_reader = None
        
        # Cleanup temp directory
        if os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# Global shared model manager
_model_manager = None
_manager_lock = threading.Lock()


def get_model_manager():
    """Get global model manager instance"""
    global _model_manager
    if _model_manager is None:
        with _manager_lock:
            if _model_manager is None:
                _model_manager = SharedModelManager()
    return _model_manager


class OCRProcessor:
    """Improved OCR processor with better PDF handling and debugging"""
    
    def __init__(self, device="auto", confidence_threshold=0.5, model_size="2B", languages=['en']):
        self.device = device
        self.confidence_threshold = confidence_threshold  # Lowered from 0.7 to 0.5
        self.model_size = model_size
        self.languages = languages
        self.model_manager = get_model_manager()
        
        logger.info(f"OCR Processor initialized - Model: {model_size}, Device: {device}")
    
    def extract_text_from_file(self, file_path: str) -> Dict[str, Any]:
        """Main entry point - extract text from PDF or image file"""
        start_time = time.time()
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            logger.debug(f"Processing file: {file_path.name}, Size: {file_path.stat().st_size} bytes")
            
            # Process based on file type
            if file_path.suffix.lower() == '.pdf':
                result = self._process_pdf(str(file_path))
            elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                result = self._process_image(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            # Add metadata
            processing_time = time.time() - start_time
            result.update({
                'processing_time_seconds': processing_time,
                'file_path': str(file_path),
                'file_size_mb': file_path.stat().st_size / (1024 * 1024)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"File processing failed for {file_path}: {e}")
            return self._create_error_result(str(e), file_path)
        finally:
            # Light cleanup after each file
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF file with improved error handling"""
        logger.debug(f"Starting PDF processing: {pdf_path}")
        
        try:
            # Try multiple extraction methods
            images = []
            extraction_methods = []
            
            # Method 1: PyMuPDF
            try:
                images = self._pdf_to_images_pymupdf(pdf_path)
                if images:
                    extraction_methods.append("PyMuPDF")
                    logger.debug(f"PyMuPDF extracted {len(images)} images")
            except Exception as e:
                logger.warning(f"PyMuPDF failed: {e}")
            
            # Method 2: pdf2image (if first method failed or got no images)
            if not images and PDF2IMAGE_AVAILABLE:
                try:
                    images = self._pdf_to_images_pdf2image(pdf_path)
                    if images:
                        extraction_methods.append("pdf2image")
                        logger.debug(f"pdf2image extracted {len(images)} images")
                except Exception as e:
                    logger.warning(f"pdf2image failed: {e}")
            
            if not images:
                # Try to repair and re-extract PDF
                logger.warning("No images extracted, attempting PDF repair...")
                repaired_images = self._extract_with_repair(pdf_path)
                if repaired_images:
                    images = repaired_images
                    extraction_methods.append("repaired")
                else:
                    raise ValueError("No readable images extracted from PDF after all methods attempted")
            
            logger.debug(f"Total images extracted: {len(images)} using methods: {', '.join(extraction_methods)}")
            
            # Process all pages with better error handling
            all_text = []
            all_confidences = []
            page_results = []
            
            for page_num, image in enumerate(images):
                try:
                    logger.debug(f"Processing page {page_num + 1}, image size: {image.size}")
                    page_result = self._process_single_image(image, page_num + 1)
                    
                    # More lenient text acceptance
                    text = page_result.get('text', '').strip()
                    if text and len(text) >= 5:  # Lowered threshold from 20 to 5
                        all_text.append(text)
                        all_confidences.append(page_result['confidence'])
                        page_results.append(page_result)
                        logger.debug(f"Page {page_num + 1}: Extracted {len(text)} characters")
                    else:
                        logger.debug(f"Page {page_num + 1}: Insufficient text extracted ({len(text)} chars)")
                        
                except Exception as e:
                    logger.warning(f"Failed to process page {page_num + 1}: {e}")
                    continue
            
            # Combine results with more lenient acceptance
            combined_text = "\n\n".join(all_text)
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            
            logger.debug(f"Combined text length: {len(combined_text)}, confidence: {avg_confidence:.2f}")
            
            # Success if we have ANY meaningful text (lowered threshold)
            success = len(combined_text.strip()) >= 10  # Lowered from implicit higher threshold
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'method': f'pdf_multi_page_{"+".join(extraction_methods)}',
                'success': success,
                'error': None if success else f"Only extracted {len(combined_text)} characters",
                'pages_processed': len(page_results),
                'total_pages': len(images)
            }
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return self._create_error_result(str(e))
    
    def _extract_with_repair(self, pdf_path: str) -> List[Image.Image]:
        """Attempt to repair PDF and extract images"""
        try:
            logger.debug("Attempting PDF repair...")
            
            # Try to open with repair flag
            doc = fitz.open(pdf_path)
            if doc.needs_pass:
                logger.warning("PDF requires password, skipping repair attempt")
                doc.close()
                return []
            
            # Save to temporary file with repair
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Save with repair
                doc.save(temp_path, garbage=4, deflate=True)
                doc.close()
                
                # Try to extract from repaired PDF
                repaired_images = self._pdf_to_images_pymupdf(temp_path)
                logger.debug(f"Repair attempt yielded {len(repaired_images)} images")
                return repaired_images
                
            finally:
                # Cleanup temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.warning(f"PDF repair failed: {e}")
            return []
    
    def _process_image(self, image_path: str) -> Dict[str, Any]:
        """Process single image file"""
        try:
            image = Image.open(image_path)
            logger.debug(f"Processing image: {image_path}, size: {image.size}, mode: {image.mode}")
            return self._process_single_image(image)
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return self._create_error_result(str(e))
    
    def _process_single_image(self, image: Image.Image, page_num: int = 1) -> Dict[str, Any]:
        """Process a single PIL Image with improved debugging"""
        try:
            # Clean and validate image
            logger.debug(f"Page {page_num}: Original image size: {image.size}, mode: {image.mode}")
            
            cleaned_image = self._prepare_image(image)
            if not cleaned_image:
                logger.warning(f"Page {page_num}: Image preparation failed")
                return self._create_error_result("Image preparation failed")
            
            logger.debug(f"Page {page_num}: Prepared image size: {cleaned_image.size}")
            
            # Always try both methods and pick the best result
            results = []
            
            # Try Qwen2-VL
            try:
                qwen_result = self._extract_with_qwen2vl(cleaned_image)
                if qwen_result.get('success'):
                    results.append(('qwen2vl', qwen_result))
                    logger.debug(f"Page {page_num}: Qwen2-VL extracted {len(qwen_result.get('text', ''))} chars, confidence: {qwen_result.get('confidence', 0):.2f}")
            except Exception as e:
                logger.warning(f"Page {page_num}: Qwen2-VL failed: {e}")
            
            # Try EasyOCR
            try:
                easyocr_result = self._extract_with_easyocr(cleaned_image)
                if easyocr_result.get('success'):
                    results.append(('easyocr', easyocr_result))
                    logger.debug(f"Page {page_num}: EasyOCR extracted {len(easyocr_result.get('text', ''))} chars, confidence: {easyocr_result.get('confidence', 0):.2f}")
            except Exception as e:
                logger.warning(f"Page {page_num}: EasyOCR failed: {e}")
            
            # Pick best result
            if not results:
                logger.warning(f"Page {page_num}: Both OCR methods failed")
                return self._create_error_result("Both OCR methods failed")
            
            # Choose result with more text, or higher confidence if similar length
            best_method, best_result = max(results, key=lambda x: (len(x[1].get('text', '')), x[1].get('confidence', 0)))
            best_result['method'] = f"{best_method}_primary"
            best_result['page_number'] = page_num
            
            logger.debug(f"Page {page_num}: Selected {best_method} result with {len(best_result.get('text', ''))} chars")
            return best_result
            
        except Exception as e:
            logger.error(f"Single image processing failed: {e}")
            return self._create_error_result(str(e))
    
    def _prepare_image(self, image: Image.Image) -> Optional[Image.Image]:
        """Prepare and validate image for OCR with improved debugging"""
        try:
            original_size = image.size
            original_mode = image.mode
            
            # Convert to RGB
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
                logger.debug(f"Converted image from {original_mode} to RGB")
            
            # Check size constraints
            width, height = image.size
            if width < 10 or height < 10:  # More lenient minimum size
                logger.warning(f"Image too small: {width}x{height}")
                return None
            
            # Resize if too large
            max_size = 2048  # Reduced max size for better performance
            if width > max_size or height > max_size:
                original_size = image.size
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                logger.debug(f"Resized image from {original_size} to {image.size}")
            
            # More lenient blank check
            if self._is_blank_image(image):
                logger.warning("Image appears to be blank")
                # Try to enhance before giving up
                enhanced = self._enhance_image_aggressive(image)
                if enhanced and not self._is_blank_image(enhanced):
                    logger.debug("Recovered image after aggressive enhancement")
                    return enhanced
                return None
            
            # Enhance for OCR
            enhanced_image = self._enhance_image(image)
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Image preparation failed: {e}")
            return None
    
    def _is_blank_image(self, image: Image.Image) -> bool:
        """More lenient blank image detection"""
        try:
            gray = image.convert('L')
            extrema = gray.getextrema()
            
            # Check contrast - more lenient threshold
            if abs(extrema[1] - extrema[0]) < 5:  # Reduced from 20 to 5
                return True
            
            # Check if mostly white - more lenient
            histogram = gray.histogram()
            total_pixels = sum(histogram)
            white_pixels = sum(histogram[240:])  # Only very white pixels
            
            return (white_pixels / total_pixels) > 0.95  # More lenient threshold
            
        except:
            return False
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Standard image enhancement for OCR"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast and sharpness
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)  # Slightly more contrast
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)  # Slightly more sharpness
            
            return image
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image
    
    def _enhance_image_aggressive(self, image: Image.Image) -> Image.Image:
        """Aggressive image enhancement for difficult images"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # More aggressive enhancements
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)
            
            return image
        except Exception as e:
            logger.error(f"Aggressive image enhancement failed: {e}")
            return image
    
    def _extract_with_qwen2vl(self, image: Image.Image) -> Dict[str, Any]:
        """Extract text using shared Qwen2-VL model"""
        try:
            qwen_model, qwen_processor = self.model_manager.get_qwen_model(
                self.device, self.model_size
            )
            
            if not qwen_model or not qwen_processor:
                return {'text': '', 'confidence': 0.0, 'success': False, 'error': 'Qwen2-VL model not available'}
            
            # Use GPU lock for thread safety
            with self.model_manager._gpu_lock:
                return self._qwen_inference(image, qwen_model, qwen_processor)
                
        except Exception as e:
            logger.error(f"Qwen2-VL extraction failed: {e}")
            return {'text': '', 'confidence': 0.0, 'success': False, 'error': str(e)}
    
    def _qwen_inference(self, image: Image.Image, model, processor) -> Dict[str, Any]:
        """Perform Qwen2-VL inference with better prompting"""
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            image.save(temp_file.name, 'PNG', optimize=True)
            temp_path = temp_file.name
        
        try:
            # More specific prompting for better results
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": temp_path},
                    {"type": "text", "text": "Please read and extract ALL text content from this document image. Include numbers, dates, names, addresses, and any other visible text. Maintain the original structure and formatting as much as possible. Return only the extracted text without any explanations."}
                ]
            }]
            
            # Process inputs
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            
            # Generate with adjusted parameters
            start_time = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=3072,  # Increased token limit
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    repetition_penalty=1.1  # Reduce repetition
                )
            
            generation_time = time.time() - start_time
            
            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Calculate confidence with more lenient scoring
            confidence = self._calculate_confidence(response, generation_time)
            
            return {
                'text': response.strip(),
                'confidence': confidence,
                'success': bool(response.strip()) and len(response.strip()) >= 3,  # More lenient success criteria
                'error': None,
                'generation_time': generation_time
            }
            
        finally:
            # Cleanup
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def _extract_with_easyocr(self, image: Image.Image) -> Dict[str, Any]:
        """Extract text using shared EasyOCR reader with better settings"""
        try:
            easyocr_reader = self.model_manager.get_easyocr_reader(self.languages)
            
            if not easyocr_reader:
                return {'text': '', 'confidence': 0.0, 'success': False, 'error': 'EasyOCR reader not available'}
            
            # Convert to numpy array
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)
            
            # Extract text with better settings
            start_time = time.time()
            results = easyocr_reader.readtext(
                img_array, 
                detail=1, 
                paragraph=False,
                width_ths=0.7,  # More lenient width threshold
                height_ths=0.7  # More lenient height threshold
            )
            extraction_time = time.time() - start_time
            
            # Process results with more lenient filtering
            text_parts = []
            confidences = []
            
            for (bbox, text, conf) in results:
                if conf > 0.2:  # Lowered confidence threshold
                    text_parts.append(text)
                    confidences.append(conf)
            
            extracted_text = "\n".join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            scaled_confidence = min(avg_confidence * 1.1, 0.95)  # Less aggressive scaling
            
            return {
                'text': extracted_text,
                'confidence': scaled_confidence,
                'success': bool(extracted_text.strip()) and len(extracted_text.strip()) >= 3,
                'error': None,
                'extraction_time': extraction_time
            }
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return {'text': '', 'confidence': 0.0, 'success': False, 'error': str(e)}
    
    def _calculate_confidence(self, response: str, generation_time: float) -> float:
        """More lenient confidence calculation"""
        if not response or len(response.strip()) < 3:
            return 0.1
        
        base_confidence = 0.7  # Lowered base confidence
        
        # Length bonus - more generous
        length_bonus = min(len(response) / 500, 0.15)  # Easier to get bonus
        
        # Structure indicators
        structure_indicators = ['date', 'amount', 'total', 'company', 'address', ':', 'receipt', 'invoice']
        structure_count = sum(1 for indicator in structure_indicators 
                            if indicator.lower() in response.lower())
        structure_bonus = min(structure_count * 0.02, 0.1)  # More generous bonus
        
        # Reduce speed penalty
        speed_penalty = 0.05 if generation_time < 0.2 else 0.0
        
        final_confidence = base_confidence + length_bonus + structure_bonus - speed_penalty
        return min(max(final_confidence, 0.0), 0.95)
    
    def _pdf_to_images_pymupdf(self, pdf_path: str) -> List[Image.Image]:
        """Extract images using PyMuPDF with better error handling"""
        images = []
        doc = None
        
        try:
            # Open with error recovery
            doc = fitz.open(pdf_path)
            logger.debug(f"PDF opened: {len(doc)} pages, needs_pass: {doc.needs_pass}")
            
            if doc.needs_pass:
                logger.warning("PDF requires password")
                return []
            
            for page_num in range(min(len(doc), 50)):  # Increased page limit
                try:
                    page = doc.load_page(page_num)
                    
                    # Try multiple rendering settings
                    render_settings = [
                        {"scale": 2.0, "alpha": False},
                        {"scale": 1.5, "alpha": False},
                        {"scale": 3.0, "alpha": False},  # Higher quality
                        {"scale": 1.0, "alpha": False}
                    ]
                    
                    for settings in render_settings:
                        try:
                            mat = fitz.Matrix(settings["scale"], settings["scale"])
                            pix = page.get_pixmap(matrix=mat, alpha=settings["alpha"])
                            
                            img_data = pix.pil_tobytes(format="PNG")
                            img = Image.open(io.BytesIO(img_data))
                            
                            if img.size[0] > 50 and img.size[1] > 50:  # More lenient size check
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                images.append(img)
                                logger.debug(f"Page {page_num + 1}: Extracted image {img.size} at scale {settings['scale']}")
                                break
                                
                        except Exception as e:
                            logger.debug(f"Render settings {settings} failed for page {page_num}: {e}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Failed to process page {page_num}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
        finally:
            if doc:
                doc.close()
        
        return images
    
    def _pdf_to_images_pdf2image(self, pdf_path: str) -> List[Image.Image]:
        """Extract images using pdf2image with better settings"""
        try:
            images = convert_from_path(
                pdf_path, 
                dpi=200,  # Higher DPI for better quality
                first_page=1, 
                last_page=50,  # Increased page limit
                fmt='RGB',
                thread_count=1  # Single thread to avoid conflicts
            )
            
            valid_images = []
            for i, img in enumerate(images):
                if img.size[0] > 50 and img.size[1] > 50:  # More lenient size check
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    valid_images.append(img)
                    logger.debug(f"pdf2image page {i+1}: {img.size}")
            
            return valid_images
            
        except Exception as e:
            logger.error(f"pdf2image extraction failed: {e}")
            return []
    
    def _create_error_result(self, error_msg: str, file_path: str = None) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'text': '',
            'confidence': 0.0,
            'success': False,
            'error': error_msg,
            'method': 'error',
            'file_path': str(file_path) if file_path else None,
            'processing_time_seconds': 0.0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            'device': self.device,
            'model_size': self.model_size,
            'confidence_threshold': self.confidence_threshold,
            'languages': self.languages
        }
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass


def cleanup_models():
    """Cleanup shared models"""
    global _model_manager
    if _model_manager:
        _model_manager.cleanup()
        _model_manager = None