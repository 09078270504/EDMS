#commands/openai.py - my first ever file processing. peace -- Quinn --
#perfect but expensive :()
# To run just type "python manage.py openai", the whole file processing code is already included in this code.
import os
import shutil
import json
import logging
import re
import time
import base64
import tempfile
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple
from multiprocessing import cpu_count
import fitz 
from PIL import Image
import io
from openai import OpenAI
from django.conf import settings
from django.utils import timezone
from django.db import transaction
from django.core.management.base import BaseCommand
import pytz
from core.models import DocumentCategory, ProcessedDocument
logger = logging.getLogger(__name__)

# ============================================================================
# ARCHIVE MANAGER
# ============================================================================
class ArchiveManager:
    """Manages the organized archive folder structure"""
    def __init__(self):
        self.upload_folder = Path(settings.UPLOAD_FOLDER)
        self.archive_folder = Path(getattr(settings, 'ARCHIVE_FOLDER', Path.cwd() / "archive"))
    
    def create_archive_structure(self, category_name, document_name, pdf_path, ocr_text, metadata, classification='successful'):
        """Create archive structure for successfully processed files"""
        if classification == 'partial':
            archive_category_name = f"{{PARTIAL}} - {category_name}"
            logger.info(f"📁 Creating PARTIAL archive for {document_name}")
        else:
            archive_category_name = category_name
            logger.info(f"📁 Creating SUCCESSFUL archive for {document_name}")
        
        # Create folder structure
        archive_folder = self.archive_folder / self._sanitize_folder_name(archive_category_name) / document_name
        original_dir = archive_folder / "original"
        ocr_dir = archive_folder / "ocr"
        metadata_dir = archive_folder / "metadata"
        
        for directory in [original_dir, ocr_dir, metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        original_filename = Path(pdf_path).name
        original_file_path = original_dir / original_filename
        ocr_file_path = ocr_dir / f"{document_name}.txt"
        metadata_file_path = metadata_dir / f"{document_name}.json"
        
        try:
            # Copy original PDF
            shutil.copy2(pdf_path, original_file_path)
            logger.info(f"Copied original file to: {original_file_path}")
            
            # Save OCR text
            with open(ocr_file_path, 'w', encoding='utf-8') as f:
                f.write(ocr_text)
            logger.info(f"Saved OCR text to: {ocr_file_path}")
            
            # Save metadata
            metadata_with_info = {
                **metadata,
                "processing_info": {
                    "category": category_name,
                    "classification": classification,
                    "document_name": document_name,
                    "original_filename": original_filename,
                    "processed_at": timezone.now().astimezone(pytz.timezone('Asia/Manila')).isoformat()
                }
            }
            
            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_with_info, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metadata to: {metadata_file_path}")
            
            return {
                'original_file_path': str(original_file_path),
                'ocr_file_path': str(ocr_file_path),
                'metadata_file_path': str(metadata_file_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to create archive structure: {e}")
            raise

    def cleanup_upload_file(self, file_path):
        """Remove processed file from upload folder"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up upload file: {file_path}")
                
                parent_dir = os.path.dirname(file_path)                # Remove empty parent directory if possible
                try:
                    if not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
                        logger.info(f"Cleaned up empty directory: {parent_dir}")
                except OSError:
                    pass
       
        except Exception as e:
            logger.error(f"Failed to cleanup upload file {file_path}: {e}")

    def scan_uploads_categories(self) -> List[Dict[str, any]]:
        """Scan upload folder for categories and PDF files"""
        categories = []
        try:
            for category_path in self.upload_folder.iterdir():
                if not category_path.is_dir():
                    continue
                    
                # Find PDF files in category
                pdf_files = []
                for file_path in category_path.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() == '.pdf':
                        pdf_files.append({
                            'filename': file_path.name,
                            'full_path': str(file_path),
                            'size': file_path.stat().st_size
                        })

                # Only include categories with PDF files
                if pdf_files:
                    categories.append({
                        'category_name': category_path.name,
                        'category_path': str(category_path),
                        'pdf_count': len(pdf_files),
                        'pdf_files': pdf_files
                    })
                    logger.info(f"Found category '{category_path.name}' with {len(pdf_files)} PDFs")
            
            return categories
            
        except Exception as e:
            logger.error(f"Error scanning upload categories: {e}")
            return []
    
    def generate_document_name(self, pdf_filename: str, category_name: str) -> str:
        """Generate clean document name for folder creation"""
        base_name = Path(pdf_filename).stem
        clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', base_name)
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')
        return clean_name.lower() if clean_name else "document"
    
    def _sanitize_folder_name(self, folder_name):
        """Sanitize folder name for file system"""
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', folder_name)
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        return sanitized

# ============================================================================
# OCR SERVICE
# ============================================================================
class OCRService:
    """Simple OCR service using OpenAI Vision"""
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            total_confidence = 0
            page_count = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Try direct text extraction first
                direct_text = page.get_text().strip()
                if len(direct_text) > 50:
                    full_text += f"--- Page {page_num + 1} ---\n{direct_text}\n\n"
                    total_confidence += 95.0
                    page_count += 1
                    continue
                
                # OCR for image-based content
                page_text, page_confidence = self._ocr_page(page, page_num + 1)
                if page_text:
                    full_text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
                    total_confidence += page_confidence
                    page_count += 1
            
            doc.close()
            
            # Calculate final confidence
            if page_count > 0:
                overall_confidence = total_confidence / page_count
                logger.info(f"✅ OCR SUCCESS: {overall_confidence:.1f}% confidence")
                return full_text.strip(), overall_confidence
            else:
                return "", 0.0
                
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return "", 0.0

    def _ocr_page(self, page, page_num):
        """OCR a single page using OpenAI Vision"""
        try:
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Optimize image
            if img.width > 1024 or img.height > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            base64_image = base64.b64encode(buffered.getvalue()).decode()
            
            # Call OpenAI Vision API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "You are a precise OCR text extraction engine. Extract only the actual visible text from this document image, exactly as it appears. CRITICAL RULES: 1. NEVER output repetitive patterns like endless zeros (00000...) or underscores (_____) - these are OCR artifacts, not real text 2. When you see form fields with printed underscores for writing spaces, output only 3-5 underscores maximum to represent the field 3. For permit numbers, authorization numbers, or ID fields that appear as long strings of zeros, either extract the actual visible number or mark as [FORM_FIELD] if it's clearly a blank field 4. If text is faint or blurry, apply maximum effort to recover it using context, character shapes, and document structure before marking as [UNREADABLE] 5. Preserve exact formatting: tables, spacing, line breaks, indentation 6. Stop immediately when you reach the natural end of document content - do not continue with repetitive patterns QUALITY TARGETS: - Interpret unclear characters using document context (business names, common terms, numbers) - Cross-reference similar text elsewhere in document for consistency - Use logical document structure to validate extracted content - For financial documents: ensure amounts, dates, and reference numbers are accurate Output only the clean, accurate text content. No explanations, no refusals, no repetitive artifacts."
                    }, {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }]
                }],
                max_tokens=4000,
                temperature=0
            )
          
            page_text = response.choices[0].message.content.strip()
            page_text = self._clean_ai_responses(page_text)
            
            if page_text and not self._is_ai_response(page_text):
                confidence = min(95.0, 85 + len(page_text) / 50)
                return page_text, confidence
            else:
                return "", 0.0
                
        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")
            return "", 0.0

    def _clean_ai_responses(self, text):
        """Remove common AI response patterns"""
        patterns = [
            r"^(Sure,?\s*)?(Here\s+is\s+the\s+)?(extracted\s+)?text(\s+content)?(\s+from\s+(the\s+)?(image|document|page))?\s*:?\s*",
            r"^```.*?\n",
            r"\n```\s*$",
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
        
        return text.strip()

    def _is_ai_response(self, text):
        """Check if text looks like an AI explanation"""
        if len(text) < 10:
            return True
            
        ai_indicators = [
            "I'm unable to extract", "I cannot extract",
            "OCR (Optical Character Recognition)",
            "tools like Adobe Acrobat"
        ]
        
        text_lower = text.lower()
        return any(indicator.lower() in text_lower for indicator in ai_indicators)
    
# ============================================================================
# ML DATA EXTRACTOR
# ============================================================================
class MLDataExtractor:
    """Simple ML data extraction using OpenAI"""
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    def extract_structured_data(self, ocr_text, document_name):
        """Extract structured data from OCR text"""
        try:
            structured_data, confidence = self._extract_with_gpt4o(ocr_text)
            
            if confidence >= 90.0:
                logger.info(f"✅ ML SUCCESS: {confidence:.1f}% confidence")
            else:
                logger.info(f"⚠️ ML partial: {confidence:.1f}% confidence")
                
            return structured_data
            
        except Exception as e:
            logger.error(f"ML extraction failed: {e}")
            return self._create_fallback_result()

    def _extract_with_gpt4o(self, ocr_text):
        """Extract data using GPT-4o"""
        system_prompt = "You are an expert document analyzer. Extract structured data accurately from document text."
        
        user_prompt = f"""Analyze this document and extract structured information:

    {ocr_text[:3000]}

    Return JSON with:
    - document_type (invoice/receipt/contract/certificate/statement/other)
    - confidence (0.0-1.0, your confidence in the analysis)
    - entities (people, companies, locations found)
    - key_information (important details like amounts, dates, reference numbers)
    - dates_found (all dates in YYYY-MM-DD format)
    - amounts_found (all monetary amounts with currency)

    Return only valid JSON."""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        
        return self._parse_response(response)

    def _parse_response(self, response):
        """Parse OpenAI response and calculate confidence"""
        content = response.choices[0].message.content.strip()
        try:
            # Extract JSON from response
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                json_content = content[start:end].strip()
            else:
                json_content = content
            
            structured_data = json.loads(json_content)
            base_confidence = structured_data.get('confidence', 0.8)       # Calculate confidence
            confidence = min(94.0, base_confidence * 100)
            
            structured_data.update({                       # Add processing metadata
                'processing_confidence': confidence,
                'extraction_method': 'openai_gpt4o'
            })
            
            return structured_data, confidence
            
        except Exception as e:
            logger.error(f"ML extraction error: {e}")
            return self._create_fallback_result(), 30.0

    def _create_fallback_result(self):
        """Create fallback result when extraction fails"""
        return {
            'document_type': 'other',
            'confidence': 0.3,
            'entities': [],
            'key_information': [],
            'dates_found': [],
            'amounts_found': [],
            'processing_confidence': 30.0,
            'extraction_method': 'fallback'
        }

# ============================================================================
# DOCUMENT PROCESSOR (MAIN ORCHESTRATOR)
# ============================================================================
class DocumentProcessor:
    """Document processor - orchestrates everything"""
    def __init__(self):
        self.max_workers = min(cpu_count() * 2, 8)
        self.archive_manager = ArchiveManager()
        self.ocr_service = OCRService()
        self.ml_extractor = MLDataExtractor()
        self.retry_counts = {}  # Track retry attempts per file

    def process_single_file(self, category, file_path, filename):
        """Process one file - failed files stay in upload directory"""
        try:
            start_time = time.time()
            document_name = self.archive_manager.generate_document_name(filename, category.name)
            ocr_text, ocr_conf = self.ocr_service.extract_text_from_pdf(file_path)      # OCR extraction
            if not ocr_text or ocr_conf < 10:
                error_message = f"OCR failed - confidence: {ocr_conf:.1f}%"             # File stays in upload directory for retry
                processing_time = time.time() - start_time
                logger.info(f"⚠️ {filename}: Skipped - {error_message} in {processing_time:.1f}s (file remains in upload)")
                return {
                    'status': 'skipped',
                    'classification': 'failed',
                    'ocr_confidence': ocr_conf,
                    'ml_confidence': 0,
                    'processing_time': processing_time,
                    'error': error_message
                }
            
            # ML extraction
            structured_data = self.ml_extractor.extract_structured_data(ocr_text, document_name)
            ml_conf = structured_data.get('processing_confidence', structured_data.get('confidence', 50))
            
            # Classify result
            classification = self._classify(ocr_conf, ml_conf)
            
            # Handle retry logic for low confidence files
            file_key = str(file_path)
            current_retries = self.retry_counts.get(file_key, 0)

            if classification == 'failed' and ocr_conf >= 10:  # Has some OCR but low confidence
                if current_retries < 3:
                    # Increment retry count and leave in upload
                    self.retry_counts[file_key] = current_retries + 1
                    error_message = f"Low confidence (attempt {current_retries + 1}/3) - OCR: {ocr_conf:.1f}%, ML: {ml_conf:.1f}%"
                    processing_time = time.time() - start_time
                    logger.info(f"🔄 {filename}: Retry {current_retries + 1}/3 - {error_message} in {processing_time:.1f}s (file remains in upload)")
                    
                    return {
                        'status': 'skipped',
                        'classification': 'retry',
                        'ocr_confidence': ocr_conf,
                        'ml_confidence': ml_conf,
                        'processing_time': processing_time,
                        'error': error_message
                    }
                else:
                    # 3 attempts done, force to partial
                    classification = 'partial'
                    logger.info(f"⚠️ {filename}: Forcing to partial after 3 attempts - OCR: {ocr_conf:.1f}%, ML: {ml_conf:.1f}%")
                    # Remove from retry tracking
                    self.retry_counts.pop(file_key, None)
            
            if classification in ['successful', 'partial']:    # Only archive and cleanup successful/partial files
                archive_paths = self.archive_manager.create_archive_structure(
                    category_name=category.name,
                    document_name=document_name,
                    pdf_path=file_path,
                    ocr_text=ocr_text,
                    metadata=structured_data,
                    classification=classification
                )
                
                # Create DB record
                db_record = self._create_db_record(
                    filename, category, document_name, classification,
                    ocr_conf, ml_conf, archive_paths
                )
                
                # Remove from upload folder and retry tracking
                self.archive_manager.cleanup_upload_file(file_path)
                self.retry_counts.pop(file_key, None)
                
                processing_time = time.time() - start_time
                logger.info(f"✅ {filename}: {classification} in {processing_time:.1f}s (OCR: {ocr_conf:.1f}%, ML: {ml_conf:.1f}%)")
                
                return {
                    'status': 'completed',
                    'classification': classification,
                    'ocr_confidence': ocr_conf,
                    'ml_confidence': ml_conf,
                    'processing_time': processing_time,
                    'db_record': db_record
                }
            else:  # Low confidence - leave in upload for retry
                error_message = f"Low confidence - OCR: {ocr_conf:.1f}%, ML: {ml_conf:.1f}%"
                processing_time = time.time() - start_time
                logger.info(f"⚠️ {filename}: Skipped - {error_message} in {processing_time:.1f}s (file remains in upload)")
                
                return {
                    'status': 'skipped',
                    'classification': 'failed',
                    'ocr_confidence': ocr_conf,
                    'ml_confidence': ml_conf,
                    'processing_time': processing_time,
                    'error': error_message
                }
            
        except Exception as e:             # Exception - leave file in upload
            error_message = f"Processing exception: {str(e)}"
            processing_time = time.time() - start_time
            logger.error(f"⚠️ {filename}: Skipped - {error_message} in {processing_time:.1f}s (file remains in upload)")
            return {
                'status': 'skipped',
                'classification': 'failed',
                'ocr_confidence': 0,
                'ml_confidence': 0,
                'processing_time': processing_time,
                'error': error_message
            }

    def process_category_parallel(self, category_name: str) -> dict:
        """Process category with parallel processing"""
        logger.info(f"🚀 Processing: {category_name}")
        categories_data = self.archive_manager.scan_uploads_categories()
        category_data = next((c for c in categories_data if c['category_name'] == category_name), None)
        if not category_data:
            return {'total': 0, 'successful': 0, 'partial': 0, 'failed': 0, 'skipped': 0}
        
        pdf_files = category_data['pdf_files']
        results = {'total': len(pdf_files), 'successful': 0, 'partial': 0, 'failed': 0, 'skipped': 0}
        if not pdf_files:
            return results

        category, _ = DocumentCategory.objects.get_or_create(name=category_name)         # Get category object
        db_records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_single_file, category, pdf['full_path'], pdf['filename']): pdf
                for pdf in pdf_files
            }
            for future in concurrent.futures.as_completed(future_to_file, timeout=1800):
                pdf_info = future_to_file[future]
                try:
                    result = future.result(timeout=180)
                    
                    if result:
                        if result.get('status') == 'completed':
                            # Successfully processed and archived
                            classification = result.get('classification', 'failed')
                            results[classification] += 1
                            
                            if result.get('db_record'):
                                db_records.append(result['db_record'])
                        elif result.get('status') == 'skipped':
                            # File was skipped and remains in upload directory
                            results['skipped'] += 1
                        else:
                            results['failed'] += 1
                    else:
                        results['failed'] += 1
                        
                except concurrent.futures.TimeoutError:
                    logger.error(f"⏰ Timeout: {pdf_info['filename']} - file remains in upload")
                    results['skipped'] += 1
                except Exception as e:
                    logger.error(f"⚠️ Error: {pdf_info['filename']}: {e} - file remains in upload")
                    results['skipped'] += 1
        
        if db_records:       # Save database records (only successful files)
            try:
                with transaction.atomic():
                    ProcessedDocument.objects.bulk_create(db_records, ignore_conflicts=True)
                logger.info(f"📝 Saved {len(db_records)} records to database")
            except Exception as e:
                logger.error(f"DB batch insert failed: {e}")
        
        category.processed_count = getattr(category, 'processed_count', 0) + results['successful']         # Update category stats
        category.save()
        logger.info(f"🎯 {category_name} complete: {results}")
        return results

    def reprocess_partials(self) -> dict:
        """Reprocess partial files to try reaching 0.9+ confidence"""
        logger.info("🔄 Scanning for partial files to reprocess...")
        results = {'reprocessed': 0, 'upgraded_to_successful': 0, 'remained_partial': 0}
        partial_folders = []              # Find all partial archive folders
        for folder in self.archive_manager.archive_folder.iterdir():
            if folder.is_dir() and folder.name.startswith("{PARTIAL}"):
                partial_folders.append(folder)
        
        if not partial_folders:
            logger.info("📭 No partial files found")
            return results
        
        logger.info(f"🔍 Found {len(partial_folders)} partial categories to check")
        for partial_folder in partial_folders:
            # Extract original category name
            original_category_name = partial_folder.name.replace("{PARTIAL} - ", "")
            
            for doc_folder in partial_folder.iterdir():
                if not doc_folder.is_dir():
                    continue
                    
                original_path = doc_folder / "original"
                if not original_path.exists():
                    continue
                    
                # Find the original PDF
                pdf_files = list(original_path.glob("*.pdf"))
                if not pdf_files:
                    continue
                    
                pdf_path = pdf_files[0]
                logger.info(f"🔄 Reprocessing partial: {doc_folder.name}")
                
                try:
                    # Re-run OCR and ML
                    ocr_text, ocr_conf = self.ocr_service.extract_text_from_pdf(str(pdf_path))
                    structured_data = self.ml_extractor.extract_structured_data(ocr_text, doc_folder.name)
                    ml_conf = structured_data.get('processing_confidence', structured_data.get('confidence', 50))
                    results['reprocessed'] += 1
                    if ocr_conf >= 90 and ml_conf >= 90:                     # Check if now qualifies as successful
                        logger.info(f"✅ Upgraded to successful: {doc_folder.name} (OCR: {ocr_conf:.1f}%, ML: {ml_conf:.1f}%)")
                        
                        # Move from partial to successful archive
                        new_archive_folder = self.archive_manager.archive_folder / original_category_name / doc_folder.name
                        new_archive_folder.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Move the entire document folder
                        shutil.move(str(doc_folder), str(new_archive_folder))
                        
                        # Update metadata files with new confidence scores
                        metadata_file = new_archive_folder / "metadata" / f"{doc_folder.name}.json"
                        if metadata_file.exists():
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            
                            metadata['processing_info']['classification'] = 'successful'
                            metadata['processing_info']['reprocessed_at'] = timezone.now().astimezone(pytz.timezone('Asia/Manila')).isoformat()
                            metadata['processing_confidence'] = ml_conf
                            
                            with open(metadata_file, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=2, ensure_ascii=False)
                        
                        results['upgraded_to_successful'] += 1
                    else:
                        logger.info(f"⚠️ Still partial: {doc_folder.name} (OCR: {ocr_conf:.1f}%, ML: {ml_conf:.1f}%)")
                        results['remained_partial'] += 1
                        
                except Exception as e:
                    logger.error(f"❌ Error reprocessing {doc_folder.name}: {e}")
                    results['remained_partial'] += 1
            
            # Clean up empty partial category folder
            try:
                if not any(partial_folder.iterdir()):
                    partial_folder.rmdir()
                    logger.info(f"🗑️ Removed empty partial folder: {partial_folder.name}")
            except:
                pass
        
        logger.info(f"🎯 Partial reprocessing complete: {results}")
        return results
    
    def _classify(self, ocr_conf, ml_conf): # feel free to change this logic this is for accepting success files
        """Classify processing result"""
        if ocr_conf >= 90 and ml_conf >= 90:
            return 'successful'
        elif ocr_conf > 0 and ml_conf > 0:
            return 'partial'
        else:
            return 'failed'

    def _create_db_record(self, filename, category, document_name, classification, ocr_conf, ml_conf, archive_paths):
        """Create database record for successful files only"""
        try:
            gmt_plus_8 = pytz.timezone('Asia/Manila')
            processed_at = timezone.now().astimezone(gmt_plus_8)
            model_fields = {field.name for field in ProcessedDocument._meta.get_fields()}       # Get model fields to avoid errors
            
            kwargs = {   # Build record
                'original_filename': filename,
                'document_name': document_name,
                'category': category,
                'status': classification,
                'processed_at': processed_at,
            }
            
            # Add optional fields if they exist
            if 'ocr_confidence' in model_fields:
                kwargs['ocr_confidence'] = ocr_conf
            if 'ml_confidence' in model_fields:
                kwargs['ml_confidence'] = ml_conf
            if 'archive_original_path' in model_fields:
                kwargs['archive_original_path'] = archive_paths.get('original_file_path', '')
            if 'archive_ocr_path' in model_fields:
                kwargs['archive_ocr_path'] = archive_paths.get('ocr_file_path', '')
            if 'archive_metadata_path' in model_fields:
                kwargs['archive_metadata_path'] = archive_paths.get('metadata_file_path', '')
            if 'processing_notes' in model_fields:
                kwargs['processing_notes'] = f'OCR: {ocr_conf:.1f}%, ML: {ml_conf:.1f}%'
            
            return ProcessedDocument(**kwargs)
            
        except Exception as e:
            logger.error(f"Failed to create DB record for {filename}: {e}")
            return ProcessedDocument(
                original_filename=filename,
                document_name=document_name,
                category=category,
                status=classification,
                processed_at=processed_at
            )

    def process_category_fast(self, category_name: str) -> dict:     # Legacy compatibility
        return self.process_category_parallel(category_name)

# ============================================================================
# MANAGEMENT COMMAND
# ============================================================================
class Command(BaseCommand):
    help = 'Process documents with OCR and metadata extraction'
    def add_arguments(self, parser):
        parser.add_argument('--category', type=str, help='Process specific category only')
        parser.add_argument('--list', action='store_true', help='List available categories')
        parser.add_argument('--watch', action='store_true', help='Watch upload folder for new files')
        parser.add_argument('--interval', type=int, default=30, help='Watch interval in seconds (default: 30)')

    def handle(self, *args, **options):
        processor = DocumentProcessor()
        archive_manager = ArchiveManager()
        if options.get('list'):
            self.list_categories(archive_manager)
            return
        if options.get('watch'):
            self.watch_mode(processor, archive_manager, options)
            return
        
        gmt_plus_8 = pytz.timezone('Asia/Manila')         # Process documents
        start_time = timezone.now().astimezone(gmt_plus_8)
        if options.get('category'):
            category_name = options['category']
            self.stdout.write(f"Processing category: {category_name}")
            results = processor.process_category_parallel(category_name)
            category_results = [{'name': category_name, 'results': results}]
        else:
            self.stdout.write("Processing all categories...")
            categories_data = archive_manager.scan_uploads_categories()
            category_results = []
            
            for category_data in categories_data:
                category_name = category_data['category_name']
                self.stdout.write(f"Processing: {category_name}")
                results = processor.process_category_parallel(category_name)
                category_results.append({'name': category_name, 'results': results})
        
        self.show_results(category_results, start_time)

    def list_categories(self, archive_manager):
        """List available categories"""
        self.stdout.write("Available categories:")
        categories_data = archive_manager.scan_uploads_categories()
        
        if not categories_data:
            self.stdout.write("No categories found")
            return
        
        for category_data in categories_data:
            name = category_data['category_name']
            count = category_data['pdf_count']
            self.stdout.write(f"  📁 {name} ({count} PDFs)")

    def show_results(self, category_results, start_time):
        """Show processing results"""
        gmt_plus_8 = pytz.timezone('Asia/Manila')
        end_time = timezone.now().astimezone(gmt_plus_8)
        duration = (end_time - start_time).total_seconds() 
        self.stdout.write("\n" + "="*50)
        self.stdout.write("📊 PROCESSING COMPLETE!")
        self.stdout.write("="*50)
        total_all = {'total': 0, 'successful': 0, 'partial': 0, 'failed': 0, 'skipped': 0}
        for category_result in category_results:
            name = category_result['name']
            results = category_result['results']
            
            total = results['total']
            successful = results['successful']
            partial = results.get('partial', 0)
            failed = results.get('failed', 0)
            skipped = results.get('skipped', 0)
            
            total_all['total'] += total
            total_all['successful'] += successful
            total_all['partial'] += partial
            total_all['failed'] += failed
            total_all['skipped'] += skipped
            
            processed_count = successful + partial
            success_rate = (successful / total * 100) if total > 0 else 0  # Only successful, not partial
            self.stdout.write(f"📂 {name}:")
            self.stdout.write(f"   ✅ {processed_count}/{total} processed ({success_rate:.1f}%)")
            if successful > 0:
                self.stdout.write(f"      🎯 {successful} successful")
            if partial > 0:
                self.stdout.write(f"      ⚠️ {partial} partial")
            if skipped > 0:
                upload_path = Path(settings.UPLOAD_FOLDER) / name   # Path to upload folder for this categoryH
                self.stdout.write(f"   ⏭️ {skipped} skipped (remain in upload)")
                self.stdout.write(f"      📁 Files still in: {upload_path}")
            if failed > 0:
                self.stdout.write(f"   ❌ {failed} failed")
            self.stdout.write("")
        
        # Show summary for skipped files
        if total_all['skipped'] > 0:
            upload_folder = Path(settings.UPLOAD_FOLDER)
            self.stdout.write(f"🔄 {total_all['skipped']} files remain in upload directory for retry: {upload_folder}")
            self.stdout.write("   Run the processor again to retry these files")
            self.stdout.write("")
        
        # Summary
        if total_all['total'] > 0:
            processed_total = total_all['successful'] + total_all['partial']
            overall_success = (total_all['successful'] / total_all['total'] * 100)  # Only successful for success rate
            avg_time = duration / total_all['total']
            
            duration_str = self._format_duration(duration)
            avg_time_str = self._format_duration(avg_time)

            self.stdout.write("🎯 SUMMARY:")
            self.stdout.write(f"   Total: {total_all['total']} documents")
            self.stdout.write(f"   Success: {total_all['successful']} ({overall_success:.1f}%)")
            self.stdout.write(f"   Duration: {duration_str} ({avg_time_str} per file)")
            
            if duration > 60:
                files_per_hour = (total_all['total'] / duration) * 3600
                self.stdout.write(f"   Throughput: {files_per_hour:.0f} files/hour")
            
            if overall_success >= 90:
                self.stdout.write("🎉 Excellent results!")
            elif overall_success >= 70:
                self.stdout.write("👍 Good results!")
            else:
                self.stdout.write("⚠️ Some issues - check failed documents")
        
        self.stdout.write("="*50)

    def watch_mode(self, processor, archive_manager, options):
        """Watch upload folder for new files"""
        interval = options.get('interval', 30)
        category = options.get('category')
        self.stdout.write("🔄 WATCH MODE STARTED")
        self.stdout.write("=" * 50)
        self.stdout.write(f"⏰ Checking every {interval} seconds")
        self.stdout.write(f"📂 Watching category: {category}" if category else "📂 Watching all categories")
        self.stdout.write("🛑 Press Ctrl+C to stop")
        self.stdout.write("=" * 50)
        scan_count = 0
        try:
            while True:
                scan_count += 1
                gmt_plus_8 = pytz.timezone('Asia/Manila')
                current_time = timezone.now().astimezone(gmt_plus_8)
                self.stdout.write(f"\n🔍 Scan #{scan_count} at {current_time.strftime('%H:%M:%S')}")
                categories_data = archive_manager.scan_uploads_categories()
                new_files_found = False

                for category_data in categories_data:
                    category_name = category_data['category_name']
                    if category and category_name != category:
                        continue

                    if category_data['pdf_files']:
                        new_files_found = True
                        self.stdout.write(f"🔄 Processing {len(category_data['pdf_files'])} files in {category_name}")
                        try:
                            start_time = timezone.now().astimezone(gmt_plus_8)
                            results = processor.process_category_parallel(category_name)
                            self.show_results([{
                                'name': category_name, 
                                'results': results
                            }], start_time)
                        except Exception as e:
                            self.stdout.write(f"❌ Error processing {category_name}: {e}")

                if not new_files_found:
                    self.stdout.write("📭 No files found in upload directories")
                    self.stdout.write("🔄 Checking partial files for reprocessing...")
                    partial_results = processor.reprocess_partials()
                    if partial_results['reprocessed'] > 0:
                        self.stdout.write(f"📊 Partial reprocessing results:")
                        self.stdout.write(f"   🔄 {partial_results['reprocessed']} files reprocessed")
                        self.stdout.write(f"   ✅ {partial_results['upgraded_to_successful']} upgraded to successful")
                        self.stdout.write(f"   ⚠️ {partial_results['remained_partial']} remained partial")
                    else:
                        self.stdout.write("📭 No partial files found to reprocess")

                self.stdout.write(f"💤 Sleeping for {interval} seconds...")
                time.sleep(interval)

        except KeyboardInterrupt:
            self.stdout.write("\n🛑 Watch mode stopped by user")
            self.stdout.write(f"📊 Completed {scan_count} scans during this session")

    def _format_duration(self, seconds):
        """Format duration in a human-readable way"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            if hours < 24:
                return f"{hours:.1f}h"
            else:
                days = hours / 24
                return f"{days:.1f}d"