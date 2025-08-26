import easyocr
import cv2
import numpy as np
import argparse
import json
import csv
import os
import sys
import gc
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging


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