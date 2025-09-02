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

<<<<<<< Updated upstream

# PDF processing imports
try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    try:
        import fitz  # PyMuPDF
        PDF_SUPPORT = True
    except ImportError:
        PDF_SUPPORT = False
        print("Warning: PDF support not available. Install pdf2image or PyMuPDF for PDF processing.")


class OCRProcessor:
    def __init__(self, languages: List[str] = ['en'], gpu: bool = True, verbose: bool = True, 
                 auto_fallback_cpu: bool = True):
        """
        Initialize OCR processor with EasyOCR
        
        Args:
            languages: List of language codes (e.g., ['en', 'es', 'fr'])
            gpu: Whether to use GPU acceleration
            verbose: Whether to show initialization progress
            auto_fallback_cpu: Automatically fallback to CPU if GPU memory issues
        """
        self.languages = languages
        self.gpu = gpu
        self.verbose = verbose
        self.auto_fallback_cpu = auto_fallback_cpu
        self.reader = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize EasyOCR reader with error handling
        self._initialize_reader()
        
    def _initialize_reader(self):
        """Initialize EasyOCR reader with memory management"""
        try:
            if self.gpu and torch.cuda.is_available():
                # Check available GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_reserved = torch.cuda.memory_reserved(0)
                gpu_allocated = torch.cuda.memory_allocated(0)
                gpu_free = gpu_memory - gpu_reserved
                
                self.logger.info(f"GPU Memory - Total: {gpu_memory/1e9:.1f}GB, "
                               f"Reserved: {gpu_reserved/1e9:.1f}GB, "
                               f"Allocated: {gpu_allocated/1e9:.1f}GB, "
                               f"Free: {gpu_free/1e9:.1f}GB")
                
                # If less than 2GB free, consider CPU fallback
                if gpu_free < 2e9 and self.auto_fallback_cpu:
                    self.logger.warning("Limited GPU memory available, switching to CPU")
                    self.gpu = False
            
            print(f"Initializing EasyOCR with languages: {self.languages}, GPU: {self.gpu}")
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=self.verbose)
            
        except Exception as e:
            if self.gpu and self.auto_fallback_cpu:
                self.logger.warning(f"GPU initialization failed: {e}. Falling back to CPU")
                self.gpu = False
                try:
                    self.reader = easyocr.Reader(self.languages, gpu=False, verbose=self.verbose)
                except Exception as cpu_e:
                    raise RuntimeError(f"Both GPU and CPU initialization failed: {cpu_e}")
            else:
                raise e
    
    def _get_optimal_dpi(self, pdf_path: str, target_dpi: int = 200) -> int:
        """
        Determine optimal DPI based on available memory and PDF size
        """
        if not self.gpu:
            return min(target_dpi, 150)  # Lower DPI for CPU processing
        
        try:
            # Check available GPU memory
            if torch.cuda.is_available():
                gpu_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                # If less than 3GB free, use lower DPI
                if gpu_free < 3e9:
                    return min(target_dpi, 150)
                elif gpu_free < 5e9:
                    return min(target_dpi, 200)
        except:
            pass
        
        return target_dpi
    
    def _clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
    def preprocess_image(self, image_path: str, enhance: bool = True) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image_path: Path to the image file
            enhance: Whether to apply image enhancement
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        if enhance:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply noise reduction
            denoised = cv2.medianBlur(gray, 3)
            
            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            return enhanced
        
        return image
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 200, max_pages: int = None) -> List[np.ndarray]:
        """
        Convert PDF pages to images with memory management
        
        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for conversion
            max_pages: Maximum number of pages to process (None for all)
            
        Returns:
            List of images as numpy arrays
        """
        if not PDF_SUPPORT:
            raise ImportError("PDF support not available. Install pdf2image or PyMuPDF")
        
        images = []
        
        try:
            # Try pdf2image first (usually more memory efficient)
            try:
                from pdf2image import convert_from_path
                
                # Process PDF in chunks to manage memory
                try:
                    # Get total page count first
                    import fitz
                    doc = fitz.open(pdf_path)
                    total_pages = doc.page_count
                    doc.close()
                except:
                    total_pages = None
                
                if total_pages and max_pages:
                    total_pages = min(total_pages, max_pages)
                
                # Process pages in chunks of 5 to manage memory
                chunk_size = 5
                page_start = 1
                
                while True:
                    page_end = page_start + chunk_size - 1
                    if total_pages and page_end > total_pages:
                        page_end = total_pages
                    
                    try:
                        pil_images = convert_from_path(
                            pdf_path, 
                            dpi=dpi,
                            first_page=page_start,
                            last_page=page_end
                        )
                        
                        for pil_image in pil_images:
                            # Convert PIL to OpenCV format
                            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                            images.append(opencv_image)
                            # Clear PIL image from memory
                            del pil_image
                        
                        # Clear chunk from memory
                        del pil_images
                        gc.collect()
                        
                        if total_pages and page_end >= total_pages:
                            break
                        page_start = page_end + 1
                        
                    except Exception as chunk_error:
                        self.logger.error(f"Error processing pages {page_start}-{page_end}: {chunk_error}")
                        break
                
            except ImportError:
                # Fallback to PyMuPDF
                import fitz
                pdf_document = fitz.open(pdf_path)
                
                total_pages = pdf_document.page_count
                if max_pages:
                    total_pages = min(total_pages, max_pages)
                
                for page_num in range(total_pages):
                    try:
                        page = pdf_document.load_page(page_num)
                        # Convert to pixmap with specified resolution
                        matrix = fitz.Matrix(dpi/72, dpi/72)
                        pix = page.get_pixmap(matrix=matrix)
                        
                        # Convert to numpy array
                        img_data = pix.tobytes("ppm")
                        nparr = np.frombuffer(img_data, np.uint8)
                        opencv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if opencv_image is not None:
                            images.append(opencv_image)
                        
                        # Clean up
                        del pix, img_data, nparr
                        
                    except Exception as page_error:
                        self.logger.error(f"Error processing page {page_num + 1}: {page_error}")
                        continue
                
                pdf_document.close()
                
        except Exception as e:
            self.logger.error(f"Error converting PDF {pdf_path}: {str(e)}")
            raise
            
        return images
    
    def extract_text_from_pdf(self, pdf_path: str, enhance: bool = True, 
                            min_confidence: float = 0.5, dpi: int = 200) -> Dict:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to the PDF file
            enhance: Whether to preprocess images
            min_confidence: Minimum confidence threshold
            dpi: Resolution for PDF to image conversion
            
        Returns:
            Dictionary containing extracted data for all pages
        """
        try:
            # Clear GPU memory before processing
            self._clear_gpu_memory()
            
            # Determine optimal DPI for current memory situation
            optimal_dpi = self._get_optimal_dpi(pdf_path, dpi)
            if optimal_dpi != dpi:
                self.logger.info(f"Reducing DPI from {dpi} to {optimal_dpi} due to memory constraints")
            
            # Convert PDF to images
            images = self.pdf_to_images(pdf_path, dpi=optimal_dpi)
            
            if not images:
                return {
                    'file_path': pdf_path,
                    'error': f"Failed to convert PDF to images: {pdf_path}",
                    'total_pages': 0,
                    'all_text_data': [],
                    'pages': {},
                    'text_only': '',
                    'total_detections': 0,
                    'avg_confidence': 0.0
                }
            
            all_results = []
            page_results = {}
            
            for page_num, image in enumerate(images, 1):
                try:
                    # Apply enhancement if requested
                    if enhance:
                        # Convert to grayscale for enhancement
                        if len(image.shape) == 3:
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        else:
                            gray = image
                        denoised = cv2.medianBlur(gray, 3)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        enhanced_image = clahe.apply(denoised)
                    else:
                        enhanced_image = image
                    
                    # Clear memory before OCR
                    self._clear_gpu_memory()
                    
                    # Perform OCR on the page
                    results = self.reader.readtext(enhanced_image)
                    
                    # Process results for this page
                    page_data = []
                    for (bbox, text, confidence) in results:
                        if confidence >= min_confidence:
                            page_data.append({
                                'text': text,
                                'confidence': round(confidence, 3),
                                'bbox': bbox,
                                'bbox_normalized': self._normalize_bbox(bbox, enhanced_image.shape),
                                'page': page_num
                            })
                    
                    page_results[f'page_{page_num}'] = page_data
                    all_results.extend(page_data)
                    
                    # Clear memory after each page
                    self._clear_gpu_memory()
                    
                except Exception as page_error:
                    self.logger.error(f"Error processing page {page_num} of {pdf_path}: {str(page_error)}")
                    continue
            
            result_data = {
                'file_path': pdf_path,
                'total_pages': len(images),
                'all_text_data': all_results,
                'pages': page_results,
                'text_only': '\n'.join([item['text'] for item in all_results]),
                'total_detections': len(all_results),
                'avg_confidence': self._calculate_average_confidence(all_results)
            }
            
            return result_data
            
        except Exception as e:
            error_msg = f"Error processing PDF {pdf_path}: {str(e)}"
            self.logger.error(error_msg)
            
            # Try CPU fallback if GPU error
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                if self.gpu and self.auto_fallback_cpu:
                    self.logger.warning("GPU memory error, attempting CPU fallback")
                    try:
                        # Reinitialize with CPU
                        self.gpu = False
                        self._initialize_reader()
                        # Retry with CPU
                        return self.extract_text_from_pdf(pdf_path, enhance, min_confidence, dpi)
                    except Exception as cpu_error:
                        return {
                            'file_path': pdf_path,
                            'error': f"Both GPU and CPU processing failed: {str(cpu_error)}",
                            'total_pages': 0,
                            'all_text_data': [],
                            'pages': {},
                            'text_only': '',
                            'total_detections': 0,
                            'avg_confidence': 0.0
                        }
            
            return {
                'file_path': pdf_path,
                'error': error_msg,
                'total_pages': 0,
                'all_text_data': [],
                'pages': {},
                'text_only': '',
                'total_detections': 0,
                'avg_confidence': 0.0
            }
    
    def extract_text(self, image_path: str, enhance: bool = True, 
                    min_confidence: float = 0.5) -> List[Dict]:
        """
        Extract text from a single image
        
        Args:
            image_path: Path to the image file
            enhance: Whether to preprocess the image
            min_confidence: Minimum confidence threshold for text detection
            
        Returns:
            List of dictionaries containing text, confidence, and bounding box info
        """
        try:
            # Preprocess image if requested
            if enhance:
                image = self.preprocess_image(image_path, enhance=True)
            else:
                image = cv2.imread(image_path)
            
            # Perform OCR
            results = self.reader.readtext(image)
            
            # Process results
            extracted_data = []
            for (bbox, text, confidence) in results:
                if confidence >= min_confidence:
                    extracted_data.append({
                        'text': text,
                        'confidence': round(confidence, 3),
                        'bbox': bbox,
                        'bbox_normalized': self._normalize_bbox(bbox, image.shape)
                    })
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return []
    
    def _normalize_bbox(self, bbox: List, image_shape: Tuple) -> List:
        """Normalize bounding box coordinates to 0-1 range"""
        height, width = image_shape[:2]
        normalized = []
        for point in bbox:
            normalized.append([point[0]/width, point[1]/height])
        return normalized
    
    def _calculate_average_confidence(self, text_results: List[Dict]) -> float:
        """
        Calculate average confidence from OCR results
        
        Args:
            text_results: List of text detection results with confidence scores
            
        Returns:
            Average confidence as percentage (0-100)
        """
        if not text_results:
            return 0.0
        
        total_confidence = sum(item.get('confidence', 0) for item in text_results)
        avg_confidence = total_confidence / len(text_results)
        
        # Convert to percentage if needed (EasyOCR returns 0-1, we want 0-100)
        if avg_confidence <= 1.0:
            avg_confidence *= 100
            
        return round(avg_confidence, 1)
    
    def extract_text_simple(self, image_path: str, min_confidence: float = 0.5) -> str:
        """
        Extract text from image and return as simple string
        
        Args:
            image_path: Path to the image file
            min_confidence: Minimum confidence threshold
            
        Returns:
            Extracted text as a single string
        """
        data = self.extract_text(image_path, min_confidence=min_confidence)
        return '\n'.join([item['text'] for item in data])
    
    def extract_text_from_pdf_simple(self, pdf_path: str, min_confidence: float = 0.5) -> str:
        """
        Extract text from PDF and return as simple string
        
        Args:
            pdf_path: Path to the PDF file
            min_confidence: Minimum confidence threshold
            
        Returns:
            Extracted text as a single string
        """
        result_data = self.extract_text_from_pdf(pdf_path, min_confidence=min_confidence)
        return result_data.get('text_only', '')
    
    # Additional methods for different return formats
    def extract_text_from_pdf_with_status(self, pdf_path: str, enhance: bool = True, 
                                        min_confidence: float = 0.5, dpi: int = 200) -> Tuple[bool, Union[Dict, str]]:
        """
        Extract text from PDF file with success status
        
        Returns:
            Tuple of (success: bool, results: Dict or error_message: str)
        """
        result_data = self.extract_text_from_pdf(pdf_path, enhance, min_confidence, dpi)
        
        # Check if there was an error
        if result_data.get('error'):
            return False, result_data['error']
        else:
            return True, result_data
    
    def process_batch(self, input_dir: str, output_dir: str = None, 
                     file_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf'],
                     min_confidence: float = 0.5, dpi: int = 200) -> Dict:
        """
        Process multiple images in a directory
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory to save results (optional)
            file_extensions: Supported image file extensions
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with processing results
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all supported files
        supported_files = []
        for ext in file_extensions:
            supported_files.extend(input_path.glob(f"*{ext}"))
            supported_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        results = {}
        total_files = len(supported_files)
        
        print(f"Processing {total_files} files...")
        
        for i, file_path in enumerate(supported_files, 1):
            print(f"Processing {i}/{total_files}: {file_path.name}")
            
            # Determine file type and process accordingly
            if file_path.suffix.lower() == '.pdf':
                # Process PDF
                extracted_data = self.extract_text_from_pdf(str(file_path), min_confidence=min_confidence, dpi=dpi)
                results[file_path.name] = extracted_data
            else:
                # Process image
                extracted_data = self.extract_text(str(file_path), min_confidence=min_confidence)
                results[file_path.name] = {
                    'file_path': str(file_path),
                    'extracted_data': extracted_data,
                    'text_only': '\n'.join([item['text'] for item in extracted_data]),
                    'total_detections': len(extracted_data),
                    'avg_confidence': self._calculate_average_confidence(extracted_data)
                }
            
            # Save individual result if output directory specified
            if output_dir:
                output_file = output_path / f"{file_path.stem}_ocr.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results[file_path.name], f, indent=2, ensure_ascii=False)
        
        # Save summary
        if output_dir:
            summary_file = output_path / "ocr_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Create CSV summary
            self._create_csv_summary(results, output_path / "ocr_summary.csv")
        
        return results
    
    def _create_csv_summary(self, results: Dict, csv_path: Path):
        """Create a CSV summary of OCR results"""
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['File', 'Total_Detections', 'Extracted_Text'])
            
            for filename, data in results.items():
                writer.writerow([
                    filename,
                    data['total_detections'],
                    data['text_only'].replace('\n', ' | ')
                ])
    
    def save_results(self, results: Dict, output_file: str, format: str = 'json'):
        """
        Save OCR results to file
        
        Args:
            results: OCR results dictionary
            output_file: Output file path
            format: Output format ('json', 'txt', 'csv')
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                if isinstance(results, dict) and 'extracted_data' in results:
                    # Single image result
                    for item in results['extracted_data']:
                        f.write(f"{item['text']}\n")
                elif isinstance(results, dict):
                    # Batch results
                    for filename, data in results.items():
                        f.write(f"=== {filename} ===\n")
                        f.write(f"{data['text_only']}\n\n")
        
        elif format.lower() == 'csv':
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                if isinstance(results, dict) and 'extracted_data' in results:
                    # Single image result
                    writer.writerow(['Text', 'Confidence', 'BBox'])
                    for item in results['extracted_data']:
                        writer.writerow([item['text'], item['confidence'], str(item['bbox'])])
                else:
                    # Batch results
                    self._create_csv_summary(results, output_path)


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='OCR Processor using EasyOCR - Process images and PDFs')
    parser.add_argument('input', help='Input image/PDF file or directory')
    parser.add_argument('-o', '--output', help='Output file or directory')
    parser.add_argument('-l', '--languages', nargs='+', default=['en'], 
                       help='Language codes (e.g., en es fr)')
    parser.add_argument('-c', '--confidence', type=float, default=0.5,
                       help='Minimum confidence threshold (0.0-1.0)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--no-enhance', action='store_true', help='Skip image enhancement')
    parser.add_argument('-f', '--format', choices=['json', 'txt', 'csv'], default='json',
                       help='Output format')
    parser.add_argument('--batch', action='store_true', help='Process directory of images/PDFs')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for PDF to image conversion')
    
    args = parser.parse_args()
    
    # Initialize OCR processor
    processor = OCRProcessor(
        languages=args.languages,
        gpu=not args.no_gpu,
        verbose=True
    )
    
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        # Batch processing
        if not input_path.is_dir():
            print("Error: Input must be a directory for batch processing")
            sys.exit(1)
        
        output_dir = args.output or f"{input_path.name}_ocr_results"
        results = processor.process_batch(
            str(input_path),
            output_dir,
            min_confidence=args.confidence,
            dpi=args.dpi
        )
        
        print(f"\nBatch processing complete!")
        print(f"Processed {len(results)} files")
        print(f"Results saved to: {output_dir}")
        
    else:
        # Single file processing
        if not input_path.exists():
            print(f"Error: File not found: {input_path}")
            sys.exit(1)
        
        print(f"Processing: {input_path}")
        
        # Check if it's a PDF or image
        if input_path.suffix.lower() == '.pdf':
            # Process PDF
            results = processor.extract_text_from_pdf(
                str(input_path),
                enhance=not args.no_enhance,
                min_confidence=args.confidence,
                dpi=args.dpi
            )
            
            # Display results
            print(f"\nExtracted text from PDF ({results['total_pages']} pages, {results['total_detections']} detections):")
            print("-" * 50)
            
            if results.get('error'):
                print(f"Error: {results['error']}")
            else:
                # Show page-by-page breakdown
                for page_key, page_data in results['pages'].items():
                    if page_data:
                        print(f"\n--- {page_key.upper()} ---")
                        for item in page_data:
                            print(f"Text: {item['text']}")
                            print(f"Confidence: {item['confidence']:.3f}")
                            print("-" * 20)
            
            # Save results if output specified
            if args.output:
                processor.save_results(results, args.output, args.format)
                print(f"Results saved to: {args.output}")
                
        else:
            # Process image
            results = processor.extract_text(
                str(input_path),
                enhance=not args.no_enhance,
                min_confidence=args.confidence
            )
            
            # Display results
            print(f"\nExtracted text ({len(results)} detections):")
            print("-" * 50)
            for item in results:
                print(f"Text: {item['text']}")
                print(f"Confidence: {item['confidence']:.3f}")
                print("-" * 30)
            
            # Save results if output specified
            if args.output:
                save_data = {
                    'file_path': str(input_path),
                    'extracted_data': results,
                    'text_only': '\n'.join([item['text'] for item in results])
                }
                processor.save_results(save_data, args.output, args.format)
                print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
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
=======
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
>>>>>>> Stashed changes


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
<<<<<<< Updated upstream
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
                   {"type": "text", "text": """Extract ALL text from this document image with high accuracy. Follow these guidelines:


STRUCTURE: Preserve the original layout and hierarchy. Keep headers, sections, and line breaks intact.


INCLUDE EVERYTHING: Extract all visible text including:
- Company names, addresses, contact details
- Document numbers, reference IDs, invoice/receipt numbers
- All dates (issue dates, due dates, service periods)
- All monetary amounts with currency symbols (₱, $, PHP, USD)
- Names of people, signatories, customers
- Item descriptions, quantities, prices
- Terms, conditions, notes, and fine print
- Stamps, watermarks, and handwritten text


FORMATTING RULES:
- Keep monetary amounts with their currency: ₱1,234.56 or $100.00
- Preserve date formats exactly as shown: 01/15/2024 or January 15, 2024
- Maintain table structures with clear spacing
- Use line breaks to separate distinct sections
- Keep related information grouped together


ACCURACY: Read carefully - numbers, addresses, and names must be exact. If text is unclear but visible, include your best reading.


OUTPUT: Return only the extracted text content, no explanations or descriptions.""" }
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


=======
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
>>>>>>> Stashed changes
