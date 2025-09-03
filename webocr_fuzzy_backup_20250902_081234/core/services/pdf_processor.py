import torch
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import subprocess
import tempfile
import os
from PIL import Image, ImageFilter, ImageEnhance
import logging
import io
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class RobustPDFProcessor:
    """Handles corrupted and problematic PDFs with multiple fallback strategies"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def extract_images_from_pdf(self, pdf_path: str, max_pages: int = 10) -> List[Image.Image]:
        """
        Extract images from PDF using multiple methods and repair strategies
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Strategy 1: Try PyMuPDF first
        images = self._extract_with_pymupdf(pdf_path, max_pages)
        if images:
            logger.info(f"PyMuPDF extracted {len(images)} images")
            return images
        
        # Strategy 2: Try PDF repair + PyMuPDF
        images = self._extract_with_repair(pdf_path, max_pages)
        if images:
            logger.info(f"Repaired PDF extracted {len(images)} images")
            return images
        
        # Strategy 3: Try pdf2image
        images = self._extract_with_pdf2image(pdf_path, max_pages)
        if images:
            logger.info(f"pdf2image extracted {len(images)} images")
            return images
        
        # Strategy 4: Try Ghostscript conversion
        images = self._extract_with_ghostscript(pdf_path, max_pages)
        if images:
            logger.info(f"Ghostscript extracted {len(images)} images")
            return images
        
        # Strategy 5: Try poppler-utils
        images = self._extract_with_poppler(pdf_path, max_pages)
        if images:
            logger.info(f"Poppler extracted {len(images)} images")
            return images
        
        logger.error(f"All extraction methods failed for {pdf_path}")
        return []
    
    def _extract_with_pymupdf(self, pdf_path: str, max_pages: int) -> List[Image.Image]:
        """Extract using PyMuPDF with error recovery"""
        images = []
        try:
            # Try to open with recovery
            doc = fitz.open(pdf_path)
            
            for page_num in range(min(len(doc), max_pages)):
                try:
                    page = doc.load_page(page_num)
                    
                    # Try multiple rendering approaches
                    for scale_factor in [1.5, 1.0, 2.0]:  # Try different DPI
                        try:
                            mat = fitz.Matrix(scale_factor, scale_factor)
                            pix = page.get_pixmap(matrix=mat, alpha=False)
                            
                            # Convert to PNG bytes
                            png_data = pix.tobytes("png")
                            
                            # Create PIL image
                            img = Image.open(io.BytesIO(png_data))
                            
                            # Validate image
                            if self._validate_image(img):
                                images.append(img)
                                break
                        except Exception as e:
                            logger.debug(f"Scale {scale_factor} failed for page {page_num}: {e}")
                            continue
                    
                except Exception as e:
                    logger.warning(f"Failed to process page {page_num}: {e}")
                    continue
            
            doc.close()
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
        
        return images
    
    def _extract_with_repair(self, pdf_path: str, max_pages: int) -> List[Image.Image]:
        """Try to repair PDF first, then extract"""
        try:
            repaired_path = os.path.join(self.temp_dir, "repaired.pdf")
            
            # Try to repair with PyMuPDF
            try:
                doc = fitz.open(pdf_path)
                doc.save(repaired_path, garbage=4, deflate=True, clean=True)
                doc.close()
                
                # Now try to extract from repaired PDF
                return self._extract_with_pymupdf(repaired_path, max_pages)
            except:
                pass
            
            # Try to repair with Ghostscript
            try:
                cmd = [
                    'gs', '-o', repaired_path, '-sDEVICE=pdfwrite', 
                    '-dPDFSETTINGS=/prepress', '-dCompatibilityLevel=1.4',
                    pdf_path
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                if result.returncode == 0:
                    return self._extract_with_pymupdf(repaired_path, max_pages)
            except:
                pass
                
        except Exception as e:
            logger.error(f"PDF repair failed: {e}")
        
        return []
    
    def _extract_with_pdf2image(self, pdf_path: str, max_pages: int) -> List[Image.Image]:
        """Extract using pdf2image library"""
        images = []
        try:
            # Try with different settings
            for dpi in [150, 200, 100]:  # Try different DPI
                try:
                    images = convert_from_path(
                        pdf_path,
                        dpi=dpi,
                        first_page=1,
                        last_page=min(max_pages, 10),
                        timeout=30
                    )
                    
                    # Validate images
                    valid_images = [img for img in images if self._validate_image(img)]
                    
                    if valid_images:
                        return valid_images
                        
                except Exception as e:
                    logger.debug(f"pdf2image DPI {dpi} failed: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"pdf2image extraction failed: {e}")
        
        return []
    
    def _extract_with_ghostscript(self, pdf_path: str, max_pages: int) -> List[Image.Image]:
        """Extract using Ghostscript"""
        images = []
        try:
            for page_num in range(1, min(max_pages + 1, 11)):
                output_path = os.path.join(self.temp_dir, f"page_{page_num}.png")
                
                cmd = [
                    'gs', '-dNOPAUSE', '-dBATCH', '-dSAFER',
                    f'-dFirstPage={page_num}', f'-dLastPage={page_num}',
                    '-sDEVICE=png16m', '-r150',
                    f'-sOutputFile={output_path}',
                    pdf_path
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, timeout=30)
                    if result.returncode == 0 and os.path.exists(output_path):
                        img = Image.open(output_path)
                        if self._validate_image(img):
                            images.append(img.copy())
                        img.close()
                except Exception as e:
                    logger.debug(f"Ghostscript page {page_num} failed: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Ghostscript extraction failed: {e}")
        
        return images
    
    def _extract_with_poppler(self, pdf_path: str, max_pages: int) -> List[Image.Image]:
        """Extract using poppler-utils (pdftoppm)"""
        images = []
        try:
            output_prefix = os.path.join(self.temp_dir, "page")
            
            cmd = [
                'pdftoppm', '-png', '-r', '150',
                '-f', '1', '-l', str(min(max_pages, 10)),
                pdf_path, output_prefix
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            if result.returncode == 0:
                # Find generated PNG files
                import glob
                png_files = glob.glob(f"{output_prefix}*.png")
                png_files.sort()
                
                for png_file in png_files:
                    try:
                        img = Image.open(png_file)
                        if self._validate_image(img):
                            images.append(img.copy())
                        img.close()
                    except Exception as e:
                        logger.debug(f"Failed to load {png_file}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Poppler extraction failed: {e}")
        
        return images
    
    def _validate_image(self, img: Image.Image) -> bool:
        """Validate that image is usable for OCR"""
        try:
            # Check basic properties
            if not img or img.size[0] < 100 or img.size[1] < 100:
                return False
            
            # Check if image has content (not blank)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Simple check: image should not be completely white or black
            extrema = img.getextrema()
            if all(e[0] == e[1] for e in extrema):  # All pixels same color
                return False
            
            # Check if image has reasonable contrast
            grayscale = img.convert('L')
            histogram = grayscale.histogram()
            
            # If too much of image is one color, likely blank
            max_pixels = max(histogram)
            total_pixels = sum(histogram)
            
            if max_pixels / total_pixels > 0.95:  # 95% same color
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Image validation failed: {e}")
            return False
    
    def enhance_image_for_ocr(self, img: Image.Image) -> Image.Image:
        """Enhance image quality for better OCR"""
        try:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.1)
            
            # Convert to grayscale for OCR
            img = img.convert('L')
            
            return img
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return img
    
    def __del__(self):
        """Cleanup temporary files"""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass


# Update your OCRProcessor to use the robust PDF processor
def _pdf_to_images_robust(self, pdf_path: str) -> list:
    """Updated method for OCRProcessor using robust PDF processing"""
    try:
        pdf_processor = RobustPDFProcessor()
        images = pdf_processor.extract_images_from_pdf(pdf_path, max_pages=10)
        
        if not images:
            logger.error(f"No images could be extracted from {pdf_path}")
            return []
        
        # Enhance images for OCR
        enhanced_images = []
        for img in images:
            try:
                enhanced = pdf_processor.enhance_image_for_ocr(img)
                enhanced_images.append(enhanced)
            except Exception as e:
                logger.warning(f"Failed to enhance image: {e}")
                enhanced_images.append(img)  # Use original if enhancement fails
        
        logger.info(f"Successfully extracted and enhanced {len(enhanced_images)} images")
        return enhanced_images
        
    except Exception as e:
        logger.error(f"Robust PDF processing failed: {e}")
        return []