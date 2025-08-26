#!/usr/bin/env python3
"""
PDF Diagnostic Script - Analyze why OCR is failing
Usage: python pdf_diagnostic.py path/to/your.pdf
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pdf2image
import fitz
from pathlib import Path

def analyze_pdf(pdf_path):
    """Comprehensive PDF analysis"""
    print(f"🔍 ANALYZING PDF: {pdf_path}")
    print("=" * 60)
    
    # Basic file info
    file_size = os.path.getsize(pdf_path) / 1024 / 1024  # MB
    print(f"📁 File size: {file_size:.1f} MB")
    
    # PDF document analysis
    try:
        doc = fitz.open(pdf_path)
        print(f"📄 PDF Pages: {len(doc)}")
        print(f"🔒 Encrypted: {doc.is_encrypted}")
        print(f"📝 PDF Version: {doc.pdf_version()}")
        print(f"🏷️  Metadata: {doc.metadata}")
        
        # Analyze first page in detail
        if len(doc) > 0:
            page = doc[0]
            rect = page.rect
            print(f"📐 Page 1 dimensions: {rect.width:.0f} x {rect.height:.0f}")
            
            # Check for text
            text = page.get_text()
            print(f"📝 Direct text extraction: {len(text)} characters")
            if text.strip():
                print(f"📖 Text preview: {text[:200]}...")
                print("✅ This PDF has extractable text - OCR not needed!")
                doc.close()
                return
            else:
                print("❌ No extractable text found - need OCR")
            
            # Check for images/drawings
            image_list = page.get_images()
            drawings = page.get_drawings()
            print(f"🖼️  Images on page: {len(image_list)}")
            print(f"✏️  Drawings on page: {len(drawings)}")
            
        doc.close()
        
    except Exception as e:
        print(f"❌ PDF analysis error: {e}")
        return
    
    print("\n" + "=" * 60)
    print("🖼️  IMAGE CONVERSION ANALYSIS")
    
    # Convert to image and analyze
    try:
        # Try different DPI settings
        for dpi in [150, 200, 300]:
            print(f"\n📊 Testing DPI: {dpi}")
            try:
                images = pdf2image.convert_from_path(
                    pdf_path, 
                    dpi=dpi,
                    first_page=1,
                    last_page=1
                )
                
                if not images:
                    print(f"   ❌ No images generated at {dpi} DPI")
                    continue
                
                image = images[0]
                print(f"   📏 Image size: {image.size}")
                print(f"   🎨 Image mode: {image.mode}")
                
                # Save debug image
                debug_path = Path(pdf_path).parent / f"debug_{Path(pdf_path).stem}_dpi{dpi}.png"
                image.save(debug_path)
                print(f"   💾 Saved: {debug_path}")
                
                # Analyze image content
                analyze_image_content(image, dpi)
                
            except Exception as e:
                print(f"   ❌ DPI {dpi} failed: {e}")
    
    except Exception as e:
        print(f"❌ Image conversion failed: {e}")

def analyze_image_content(image, dpi):
    """Analyze the content of the converted image"""
    try:
        # Convert to numpy for analysis
        img_array = np.array(image)
        
        print(f"   📊 Array shape: {img_array.shape}")
        print(f"   📊 Array dtype: {img_array.dtype}")
        
        # Calculate statistics
        if len(img_array.shape) == 3:  # Color image
            mean_values = np.mean(img_array, axis=(0,1))
            overall_mean = np.mean(img_array)
            print(f"   📊 RGB means: R={mean_values[0]:.1f}, G={mean_values[1]:.1f}, B={mean_values[2]:.1f}")
        else:  # Grayscale
            overall_mean = np.mean(img_array)
        
        print(f"   📊 Overall brightness: {overall_mean:.1f}/255")
        
        # Check if image is mostly empty
        if overall_mean > 250:
            print("   ⚠️  IMAGE IS MOSTLY WHITE - Likely empty or very faint")
        elif overall_mean < 10:
            print("   ⚠️  IMAGE IS MOSTLY BLACK - Likely corrupted")
        elif 200 < overall_mean < 250:
            print("   ⚠️  IMAGE IS VERY LIGHT - Text might be too faint")
        else:
            print("   ✅ Image brightness looks reasonable")
        
        # Check for contrast
        std_dev = np.std(img_array)
        print(f"   📊 Contrast (std dev): {std_dev:.1f}")
        
        if std_dev < 10:
            print("   ⚠️  VERY LOW CONTRAST - Image might be uniform")
        elif std_dev < 30:
            print("   ⚠️  LOW CONTRAST - Might need enhancement")
        else:
            print("   ✅ Contrast looks adequate")
        
        # Test OCR on this image
        test_ocr_on_image(image, dpi)
        
    except Exception as e:
        print(f"   ❌ Image analysis error: {e}")

def test_ocr_on_image(image, dpi):
    """Test OCR on the image with different preprocessing"""
    print(f"   🔍 Testing OCR on {dpi} DPI image...")
    
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
        # Test 1: Original image
        test_single_ocr(ocr, image, "Original")
        
        # Test 2: Enhanced contrast
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(2.0)  # Double contrast
        test_single_ocr(ocr, enhanced, "High Contrast")
        
        # Test 3: Grayscale
        grayscale = image.convert('L').convert('RGB')
        test_single_ocr(ocr, grayscale, "Grayscale")
        
        # Test 4: Inverted (if image is mostly dark)
        from PIL import ImageOps
        inverted = ImageOps.invert(image.convert('RGB'))
        test_single_ocr(ocr, inverted, "Inverted")
        
    except ImportError:
        print("   ❌ PaddleOCR not available for testing")
    except Exception as e:
        print(f"   ❌ OCR test error: {e}")

def test_single_ocr(ocr, image, method_name):
    """Test OCR on a single image variant"""
    try:
        img_array = np.array(image, dtype=np.uint8)
        result = ocr.ocr(img_array)
        
        if result and result[0]:
            text_parts = []
            confidences = []
            
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    confidence = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 0.5
                    
                    if text and len(text.strip()) > 0:
                        text_parts.append(text.strip())
                        confidences.append(float(confidence))
            
            if text_parts:
                combined_text = ' '.join(text_parts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                print(f"      {method_name}: {len(combined_text)} chars, {avg_confidence:.2f} confidence")
                print(f"      Preview: {combined_text[:100]}...")
                
                # Check for scattered letters
                words = combined_text.split()
                single_chars = sum(1 for word in words if len(word) == 1)
                if len(words) > 0 and single_chars / len(words) > 0.5:
                    print(f"      ⚠️  High single character ratio: {single_chars}/{len(words)} - likely OCR artifacts")
            else:
                print(f"      {method_name}: No text extracted")
        else:
            print(f"      {method_name}: No OCR results")
            
    except Exception as e:
        print(f"      {method_name} OCR error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdf_diagnostic.py path/to/your.pdf")
        print("\nThis script will:")
        print("- Analyze your PDF structure")
        print("- Test different DPI conversions") 
        print("- Try various OCR preprocessing methods")
        print("- Save debug images for manual inspection")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    analyze_pdf(pdf_path)