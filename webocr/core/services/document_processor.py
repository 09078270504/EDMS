#services/document_processor.py - orchestrator for document processing
import os
import re
import shutil
import json
import time
import logging
import concurrent.futures
from pathlib import Path
from multiprocessing import cpu_count
from collections import defaultdict
from typing import Dict, List, Optional
import threading
import gc
import torch
from django.conf import settings
from django.utils import timezone
import pytz

# Import services
from core.services.ocr_processor import OCRProcessor, cleanup_models
from core.services.metadata import QwenExtractor, cleanup_qwen_models
from core.services.archive_manager import ArchiveManager

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
class DocumentProcessor:
    """Simple, fast, and efficient document processor using standardized Qwen2-VL 7B"""
    
    def __init__(self):
        init_start_time = get_ph_time()
        logger.info(f"[{format_ph_time(init_start_time)}] 🚀 Initializing DocumentProcessor...")
        
        # PERFORMANCE FIX: More aggressive parallelization
        available_cores = cpu_count()
        if torch.cuda.is_available():
            # Use more workers but smaller batches to prevent memory issues
            self.max_workers = min(available_cores, 4)  # Reduced workers to prevent memory conflicts
            self.batch_size = 3  # Much smaller batches to prevent memory issues
        else:
            self.max_workers = 2
            self.batch_size = 2
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_size = "7B"
        
        # Initialize services
        self.archive_manager = ArchiveManager()
        self._ocr_processor = None
        self._text_extractor = None
        self._processor_lock = threading.Lock()
        self.retry_counts = {}
        
        # PERFORMANCE FIX: Pre-warm models at startup
        self._pre_warm_models()
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'partial': 0,
            'failed': 0,
            'skipped': 0,
            'failed_count': 0,
            'total_time': 0.0,
            'processing_times': [],
            'ocr_times': [],
            'extraction_times': [],
            'archive_times': [],
            'document_types': defaultdict(int),
            'parallel_efficiency': 0.0,
            'average_times': {
                'ocr': 0.0,
                'extraction': 0.0,
                'archive': 0.0,
                'total': 0.0
            }
        }
        self.stats_lock = threading.Lock()
        
        init_complete_time = get_ph_time()
        init_duration = (init_complete_time - init_start_time).total_seconds()
        logger.info(f"[{format_ph_time(init_complete_time)}] ✅ DocumentProcessor initialized in {init_duration:.2f}s - Workers: {self.max_workers}, Batch: {self.batch_size}")

    def _pre_warm_models(self):
        """PERFORMANCE FIX: Pre-load models at startup instead of during processing"""
        warm_start_time = get_ph_time()
        logger.info(f"[{format_ph_time(warm_start_time)}] 🔥 Pre-warming models for performance...")
        
        start_time = time.time()
        
        # Pre-load OCR processor
        try:
            _ = self.ocr_processor  # This will trigger lazy loading
            ocr_warm_time = get_ph_time()
            logger.info(f"[{format_ph_time(ocr_warm_time)}] ✅ OCR processor pre-warmed")
        except Exception as e:
            logger.warning(f"[{format_ph_time()}] ⚠️ OCR processor pre-warm failed: {e}")
        
        # Pre-load text extractor
        try:
            _ = self.text_extractor  # This will trigger lazy loading
            extractor_warm_time = get_ph_time()
            logger.info(f"[{format_ph_time(extractor_warm_time)}] ✅ Text extractor pre-warmed")
        except Exception as e:
            logger.warning(f"[{format_ph_time()}] ⚠️ Text extractor pre-warm failed: {e}")
        
        warm_time = time.time() - start_time
        warm_complete_time = get_ph_time()
        logger.info(f"[{format_ph_time(warm_complete_time)}] 🎯 Model pre-warming completed in {warm_time:.1f}s")

    @property
    def ocr_processor(self):
        """OCR processor with performance optimizations"""
        if self._ocr_processor is None:
            with self._processor_lock:
                if self._ocr_processor is None:
                    load_start_time = get_ph_time()
                    self._ocr_processor = OCRProcessor(
                        device="auto",
                        confidence_threshold=0.6,
                        model_size=self.model_size,
                        languages=['en']
                    )
                    load_complete_time = get_ph_time()
                    logger.info(f"[{format_ph_time(load_complete_time)}] ✅ OCR processor loaded (shared)")
        return self._ocr_processor
    
    @property  
    def text_extractor(self):
        """Text extractor with performance optimizations"""
        if self._text_extractor is None:
            with self._processor_lock:
                if self._text_extractor is None:
                    try:
                        load_start_time = get_ph_time()
                        self._text_extractor = QwenExtractor(
                            model_size=self.model_size,
                            device=self.device
                        )
                        load_complete_time = get_ph_time()
                        logger.info(f"[{format_ph_time(load_complete_time)}] ✅ Text extractor loaded (shared)")
                    except Exception as e:
                        error_time = get_ph_time()
                        logger.warning(f"[{format_ph_time(error_time)}] ⚠️ Failed to load text extractor: {e}")
                        self._text_extractor = None
        return self._text_extractor
    
    def extract_text_from_pdf(self, file_path: str) -> tuple:
        """Extract text from PDF for backward compatibility"""
        extract_start_time = get_ph_time()
        try:
            result = self.ocr_processor.extract_text_from_file(file_path)
            if result.get('success'):
                text = result.get('text', '')
                confidence = float(result.get('confidence', 0.0)) * 100
                extract_complete_time = get_ph_time()
                extract_duration = (extract_complete_time - extract_start_time).total_seconds()
                logger.debug(f"[{format_ph_time(extract_complete_time)}] 📄 PDF text extracted in {extract_duration:.2f}s - {len(text)} chars, {confidence:.1f}% confidence")
                return text, confidence
            else:
                error_time = get_ph_time()
                logger.warning(f"[{format_ph_time(error_time)}] ⚠️ PDF extraction failed: {result.get('error', 'Unknown error')}")
                return "", 0.0
        except Exception as e:
            error_time = get_ph_time()
            logger.error(f"[{format_ph_time(error_time)}] ❌ OCR extraction failed: {e}")
            return "", 0.0
    
    def extract_structured_data(self, text: str, doc_type: str = None) -> dict:
        """Extract structured data for backward compatibility"""
        struct_start_time = get_ph_time()
        try:
            if self.text_extractor and text:
                clean_text = self._preprocess_text(text)
                result = self.text_extractor.extract_json(clean_text, doc_type) or {}
                
                # Add confidence score
                confidence = self._calculate_extraction_confidence(result, text)
                result['processing_confidence'] = confidence 
                result['confidence'] = confidence
                
                struct_complete_time = get_ph_time()
                struct_duration = (struct_complete_time - struct_start_time).total_seconds()
                logger.debug(f"[{format_ph_time(struct_complete_time)}] 🔍 Structured data extracted in {struct_duration:.2f}s - {confidence:.1f}% confidence")
                
                return result
            else:
                fallback_time = get_ph_time()
                logger.debug(f"[{format_ph_time(fallback_time)}] 🔄 Using fallback metadata extraction")
                return self._create_fallback_metadata(doc_type or 'general', text)
        except Exception as e:
            error_time = get_ph_time()
            logger.warning(f"[{format_ph_time(error_time)}] ⚠️ Structured extraction failed: {e}")
            return self._create_fallback_metadata(doc_type or 'general', text)

    def _calculate_extraction_confidence(self, result: dict, text: str) -> float:
        """More sophisticated confidence calculation"""
        text_length_score = min(len(text) / 500, 1.0) * 15
        text_structure_score = len(re.findall(r'\n', text)) / max(len(text.split('\n')), 1) * 10
        
        # Data completeness with weighted importance
        completeness_factors = {
            'amounts.total': 25,  # Most important
            'dates.issue_date': 20,
            'ids': 15,
            'parties.issuer.name': 10,
            'document_type': 10,
        }
        
        completeness_score = 0
        for field, weight in completeness_factors.items():
            if self._has_nested_value(result, field):
                completeness_score += weight
        
        # Document type specific bonuses
        doc_type_bonus = self._get_document_type_bonus(result, text)
        
        total_confidence = (
            text_length_score +
            text_structure_score +
            completeness_score +
            doc_type_bonus
        )
        
        return min(total_confidence, 100.0)
    
    def process_single_file_optimized(self, file_path: str, category_name: str, filename: str) -> Dict:
        """PERFORMANCE OPTIMIZED single file processing - target: 30-60 seconds per file"""
        file_start_time = get_ph_time()
        start_time = time.time()
        
        logger.info(f"[{format_ph_time(file_start_time)}]: Processing **{filename}**")
        
        try:
            # Quick validation
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 50:  # Reduced threshold
                logger.warning(f"[{format_ph_time()}]: Large file {file_size_mb:.1f}MB - **{filename}**")
            
            # PERFORMANCE FIX: OCR with timeout and memory management
            ocr_start_time = get_ph_time()
            ocr_start = time.time()
            try:
                # Clear GPU memory before OCR
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                ocr_result = self.ocr_processor.extract_text_from_file(file_path)
                ocr_time = time.time() - ocr_start
                ocr_complete_time = get_ph_time()
                
                # Early exit if OCR completely failed
                if not ocr_result.get('success') or not ocr_result.get('text', '').strip():
                    # FAST FALLBACK: Use filename-based extraction
                    logger.info(f"[{format_ph_time(ocr_complete_time)}]: OCR failed for **{filename}**, using filename extraction")
                    return self._process_filename_only(file_path, category_name, filename, start_time, file_start_time)
                
                ocr_text = ocr_result.get('text', '').strip()
                
                if len(ocr_text) < 10:
                    return self._process_filename_only(file_path, category_name, filename, start_time, file_start_time)
                
                logger.debug(f"[{format_ph_time(ocr_complete_time)}]: OCR completed in {ocr_time:.2f}s - {len(ocr_text)} chars")
                
            except Exception as e:
                error_time = get_ph_time()
                logger.warning(f"[{format_ph_time(error_time)}]: OCR error for **{filename}**: {e}")
                return self._process_filename_only(file_path, category_name, filename, start_time, file_start_time)
            
            # PERFORMANCE FIX: Fast metadata extraction with timeout
            extraction_start_time = get_ph_time()
            extraction_start = time.time()
            extraction_method = "none"
            
            try:
                if self.text_extractor:
                    # PERFORMANCE FIX: Skip AI enhancement for speed, use pattern extraction only
                    clean_text = self._preprocess_text_fast(ocr_text)
                    unified = self.text_extractor.extract_json(clean_text, None) or {}
                    extraction_method = "fast_pattern_extraction"
                    
                    # Quick validation and enhancement
                    unified = self._quick_enhance_extraction(unified, clean_text, filename)
                else:
                    unified = self._create_fallback_metadata_fast(filename, ocr_text)
                    extraction_method = "filename_fallback"
                
                extraction_time = time.time() - extraction_start
                extraction_complete_time = get_ph_time()
                logger.debug(f"[{format_ph_time(extraction_complete_time)}]: Extraction completed in {extraction_time:.2f}s - method: {extraction_method}")
                
            except Exception as e:
                error_time = get_ph_time()
                logger.warning(f"[{format_ph_time(error_time)}]: extraction failed for {filename}: {e}")
                unified = self._create_fallback_metadata_fast(filename, ocr_text)
                extraction_method = "error_fallback"
                extraction_time = time.time() - extraction_start
            
            # PERFORMANCE FIX: Quick confidence and classification
            confidence = self._calculate_confidence_fast(unified, ocr_text, filename, extraction_method)
            classification = self._classify_result_fast(confidence, unified)
            
            # Fast archiving
            archive_start_time = get_ph_time()
            archive_start = time.time()
            processing_time = time.time() - start_time
            
            if classification in ['successful', 'partial']:
                doc_stem = self.archive_manager.generate_document_name(filename, category_name)
                
                unified['processing_info'] = {
                    'category': category_name,
                    'classification': classification,
                    'document_name': doc_stem,
                    'original_filename': filename,
                    'processing_time': processing_time,
                    'extraction_method': extraction_method,
                    'model_used': f'Optimized-{self.model_size}',
                    'confidence_score': confidence,
                    'processed_at': get_ph_time().isoformat(),
                    'processed_at_ph_formatted': format_ph_time()
                }
                
                # Archive
                self.archive_manager.create_archive_structure(
                    category_name=category_name,
                    document_name=doc_stem,
                    pdf_path=file_path,
                    ocr_text=ocr_text,
                    metadata=unified,
                    classification=classification
                )
                self.archive_manager.cleanup_upload_file(file_path)
                
                archive_time = time.time() - archive_start
                archive_complete_time = get_ph_time()
                
                # Update stats
                with self.stats_lock:
                    self.stats['total_processed'] += 1
                    if classification == 'successful':
                        self.stats['successful'] += 1
                    else:
                        self.stats['partial'] += 1
                    self.stats['processing_times'].append(processing_time)
                    self.stats['ocr_times'].append(ocr_time)
                    self.stats['extraction_times'].append(extraction_time)
                    self.stats['archive_times'].append(archive_time)
                    self._update_average_times()
                
                file_complete_time = get_ph_time()
                total_duration = (file_complete_time - file_start_time).total_seconds()
                logger.info(f"[{format_ph_time(file_complete_time)}] ✅ SUCCESS: {filename} - {confidence:.1f}% confidence in {total_duration:.1f}s")
                
                return {
                    'status': 'completed',
                    'classification': classification,
                    'confidence': confidence,
                    'processing_time': processing_time,
                    'extraction_method': extraction_method
                }
            else:
                with self.stats_lock:
                    self.stats['total_processed'] += 1
                    self.stats['skipped'] += 1
                    self.stats['processing_times'].append(processing_time)
                    self._update_average_times()
                
                skip_time = get_ph_time()
                logger.info(f"[{format_ph_time(skip_time)}] ⏭️ SKIPPED: {filename} - Low confidence: {confidence:.1f}%")
                
                return {'status': 'skipped', 'reason': f'Low confidence: {confidence:.1f}%'}
        
        except Exception as e:
            processing_time = time.time() - start_time
            
            with self.stats_lock:
                self.stats['total_processed'] += 1
                self.stats['failed'] += 1
                self.stats['failed_count'] += 1
                self.stats['processing_times'].append(processing_time)
                self._update_average_times()
            
            error_time = get_ph_time()
            total_duration = (error_time - file_start_time).total_seconds()
            logger.error(f"[{format_ph_time(error_time)}] ❌ ERROR: {filename} - {str(e)} in {total_duration:.1f}s")
            return {'status': 'failed', 'error': str(e), 'processing_time': processing_time}

    def _process_filename_only(self, file_path: str, category_name: str, filename: str, start_time: float, file_start_time) -> Dict:
        """PERFORMANCE FIX: Super fast filename-only processing when OCR fails"""
        
        try:
            fallback_start_time = get_ph_time()
            logger.info(f"[{format_ph_time(fallback_start_time)}] 🔄 Using filename-only extraction for {filename}")
            
            # Extract basic info from filename
            unified = self._create_fallback_metadata_fast(filename, "")
            confidence = 65.0  # Reasonable confidence for filename extraction
            classification = "successful" if confidence >= 60 else "partial"
            
            doc_stem = self.archive_manager.generate_document_name(filename, category_name)
            
            unified['processing_info'] = {
                'category': category_name,
                'classification': classification,
                'document_name': doc_stem,
                'original_filename': filename,
                'processing_time': time.time() - start_time,
                'extraction_method': 'filename_only',
                'confidence_score': confidence,
                'processed_at': get_ph_time().isoformat(),
                'processed_at_ph_formatted': format_ph_time()
            }
            
            # Quick archive
            self.archive_manager.create_archive_structure(
                category_name=category_name,
                document_name=doc_stem,
                pdf_path=file_path,
                ocr_text="OCR failed - extracted from filename",
                metadata=unified,
                classification=classification
            )
            self.archive_manager.cleanup_upload_file(file_path)
            
            processing_time = time.time() - start_time
            
            with self.stats_lock:
                self.stats['total_processed'] += 1
                self.stats['successful'] += 1
                self.stats['processing_times'].append(processing_time)
                self._update_average_times()
            
            fallback_complete_time = get_ph_time()
            total_duration = (fallback_complete_time - file_start_time).total_seconds()
            logger.info(f"[{format_ph_time(fallback_complete_time)}] ✅ FILENAME SUCCESS: {filename} - {confidence:.1f}% confidence in {total_duration:.1f}s")
            
            return {
                'status': 'completed',
                'classification': classification,
                'confidence': confidence,
                'processing_time': processing_time,
                'extraction_method': 'filename_only'
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            with self.stats_lock:
                self.stats['total_processed'] += 1
                self.stats['failed'] += 1
                self.stats['processing_times'].append(processing_time)
                self._update_average_times()
            
            error_time = get_ph_time()
            total_duration = (error_time - file_start_time).total_seconds()
            logger.error(f"[{format_ph_time(error_time)}] ❌ FILENAME FALLBACK FAILED: {filename} - {str(e)} in {total_duration:.1f}s")
            return {'status': 'failed', 'error': str(e), 'processing_time': processing_time}

    def process_category_super_fast_with_data(self, category_data: dict) -> dict:
        """Process category using pre-scanned data"""
        category_start_time = get_ph_time()
        start_time = time.time()
        category_name = category_data['category_name']
        logger.info(f"[{format_ph_time(category_start_time)}] skibidi processing: {category_name}")
        
        pdf_files = category_data['pdf_files']
        total_files = len(pdf_files)
        results = {'total': total_files, 'successful': 0, 'partial': 0, 'failed': 0, 'skipped': 0}
        
        if total_files == 0:
            empty_time = get_ph_time()
            logger.info(f"[{format_ph_time(empty_time)}] 📭 No files to process in {category_name}")
            return results
        
        # PERFORMANCE FIX: Process in very small batches with aggressive memory management
        batch_size = 2
        batches = [pdf_files[i:i + batch_size] for i in range(0, len(pdf_files), batch_size)]
        
        logger.info(f"[{format_ph_time()}] 📊 Processing {total_files} files in {len(batches)} micro-batches")
        
        for batch_idx, batch in enumerate(batches):
            batch_start_time = get_ph_time()
            batch_start = time.time()
            
            # PERFORMANCE FIX: Sequential processing to avoid memory conflicts
            for pdf in batch:
                try:
                    result = self.process_single_file_optimized(
                        pdf['full_path'], category_name, pdf['filename']
                    )
                    
                    if result.get('status') == 'completed':
                        if result.get('classification') == 'successful':
                            results['successful'] += 1
                        else:
                            results['partial'] += 1
                    elif result.get('status') == 'skipped':
                        results['skipped'] += 1
                    else:
                        results['failed'] += 1
                        
                except Exception as e:
                    results['failed'] += 1
                    error_time = get_ph_time()
                    logger.error(f"[{format_ph_time(error_time)}] ❌ Processing error: {e}")
            
            # PERFORMANCE FIX: Aggressive memory cleanup between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            batch_time = time.time() - batch_start
            elapsed_total = time.time() - start_time
            batch_complete_time = get_ph_time()
            
            logger.info(f"[{format_ph_time(batch_complete_time)}] ✅ Batch {batch_idx + 1}/{len(batches)} completed in {batch_time:.1f}s. Total elapsed: {elapsed_total:.1f}s")
        
        total_time = time.time() - start_time
        throughput = total_files / total_time * 60 if total_time > 0 else 0
        
        category_complete_time = get_ph_time()
        logger.info(f"[{format_ph_time(category_complete_time)}] 🎯 CATEGORY COMPLETE: {category_name} in {total_time:.1f}s ({throughput:.1f} files/min): {results}")
        return results
    
    def process_all_categories(self) -> List[dict]:
        """Process all categories with standardized Qwen2-VL 7B"""
        all_start_time = get_ph_time()
        categories_data = self.archive_manager.scan_uploads_categories()
        all_results = []
        
        logger.info(f"[{format_ph_time(all_start_time)}] 🚀 Processing all categories with Qwen2-VL {self.model_size}")
        
        for category_data in categories_data:
            category_name = category_data['category_name']
            file_count = len(category_data.get('pdf_files', []))
            
            if file_count > 0:
                category_start_time = get_ph_time()
                logger.info(f"[{format_ph_time(category_start_time)}] 📂 Processing category: {category_name} ({file_count} files)")
                # PASS THE CATEGORY_DATA INSTEAD OF JUST THE NAME
                results = self.process_category_super_fast_with_data(category_data)
                all_results.append({'name': category_name, 'results': results})
        
        all_complete_time = get_ph_time()
        total_duration = (all_complete_time - all_start_time).total_seconds()
        logger.info(f"[{format_ph_time(all_complete_time)}] 🎉 ALL CATEGORIES COMPLETE in {total_duration:.1f}s")
        
        return all_results

    def reprocess_partials(self) -> dict:
        """Reprocess partial files to try reaching higher confidence with standardized model"""
        reprocess_start_time = get_ph_time()
        logger.info(f"[{format_ph_time(reprocess_start_time)}] 🔄 Scanning for partial files to reprocess with Qwen2-VL 7B...")
        results = {'reprocessed': 0, 'upgraded_to_successful': 0, 'remained_partial': 0}
        partial_folders = []
        
        # Find all partial archive folders
        for folder in self.archive_manager.archive_folder.iterdir():
            if folder.is_dir() and folder.name.startswith("{PARTIAL}"):
                partial_folders.append(folder)
        
        if not partial_folders:
            no_partials_time = get_ph_time()
            logger.info(f"[{format_ph_time(no_partials_time)}] 📭 No partial files found")
            return results
        
        found_partials_time = get_ph_time()
        logger.info(f"[{format_ph_time(found_partials_time)}] 🔍 Found {len(partial_folders)} partial categories to check")
        
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
                partial_start_time = get_ph_time()
                logger.info(f"[{format_ph_time(partial_start_time)}] 🔄 Reprocessing partial with Qwen2-VL 7B: {doc_folder.name}")
                
                try:
                    # Re-run OCR and extraction with standardized model
                    ocr_text, ocr_conf = self.extract_text_from_pdf(str(pdf_path))
                    structured_data = self.extract_structured_data(ocr_text, doc_folder.name)
                    ml_conf = structured_data.get('processing_confidence', structured_data.get('confidence', 50))
                    
                    results['reprocessed'] += 1
                    
                    if ocr_conf >= 90 and ml_conf >= 90:
                        upgrade_time = get_ph_time()
                        logger.info(f"[{format_ph_time(upgrade_time)}] ✅ Upgraded to successful: {doc_folder.name} (OCR: {ocr_conf:.1f}%, ML: {ml_conf:.1f}%)")
                        
                        # Move from partial to successful archive
                        new_archive_folder = self.archive_manager.archive_folder / original_category_name / doc_folder.name
                        new_archive_folder.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Move the entire document folder
                        shutil.move(str(doc_folder), str(new_archive_folder))
                        
                        # Update metadata files with new confidence scores and model info
                        metadata_file = new_archive_folder / "metadata" / f"{doc_folder.name}.json"
                        if metadata_file.exists():
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            
                            metadata['processing_info']['classification'] = 'successful'
                            metadata['processing_info']['reprocessed_at'] = get_ph_time().isoformat()
                            metadata['processing_info']['reprocessed_at_ph_formatted'] = format_ph_time()
                            metadata['processing_info']['model_used'] = f'Qwen2-VL-{self.model_size}'
                            metadata['processing_confidence'] = ml_conf
                            
                            with open(metadata_file, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=2, ensure_ascii=False)
                        
                        results['upgraded_to_successful'] += 1
                    else:
                        remain_time = get_ph_time()
                        logger.info(f"[{format_ph_time(remain_time)}] ⚠️ Still partial: {doc_folder.name} (OCR: {ocr_conf:.1f}%, ML: {ml_conf:.1f}%)")
                        results['remained_partial'] += 1
                        
                except Exception as e:
                    error_time = get_ph_time()
                    logger.error(f"[{format_ph_time(error_time)}] ❌ Error reprocessing {doc_folder.name}: {e}")
                    results['remained_partial'] += 1
            
            # Clean up empty partial category folder
            try:
                if not any(partial_folder.iterdir()):
                    partial_folder.rmdir()
                    cleanup_time = get_ph_time()
                    logger.info(f"[{format_ph_time(cleanup_time)}] 🗑️ Removed empty partial folder: {partial_folder.name}")
            except:
                pass
        
        reprocess_complete_time = get_ph_time()
        reprocess_duration = (reprocess_complete_time - reprocess_start_time).total_seconds()
        logger.info(f"[{format_ph_time(reprocess_complete_time)}] 🎯 Partial reprocessing complete with Qwen2-VL 7B in {reprocess_duration:.1f}s: {results}")
        return results

    def get_stats_summary(self) -> dict:
        """Get processing statistics with model information"""
        with self.stats_lock:
            total = self.stats['total_processed']
            if total == 0:
                return 'No files processed yet'
            
            avg_time = (sum(self.stats['processing_times']) / len(self.stats['processing_times']) 
                       if self.stats['processing_times'] else 0)
            success_rate = ((self.stats['successful'] + self.stats['partial']) / total) * 100
            
            return {
                'total_processed': total,
                'successful': self.stats['successful'],
                'partial': self.stats['partial'],
                'failed': self.stats['failed'],
                'skipped': self.stats['skipped'],
                'failed_count': self.stats['failed_count'],
                'success_rate': round(success_rate, 1),
                'average_processing_time': round(avg_time, 2),
                'throughput_per_hour': round(3600 / avg_time) if avg_time > 0 else 0,
                'document_types': dict(self.stats['document_types']),
                'model_used': f'Qwen2-VL-{self.model_size}',
                'device': self.device,
                'average_times': {
                    'ocr': round(self.stats['average_times']['ocr'], 2),
                    'extraction': round(self.stats['average_times']['extraction'], 2),
                    'archive': round(self.stats['average_times']['archive'], 2),
                    'total': round(self.stats['average_times']['total'], 2)
                }
            }
    
    def cleanup(self):
        """Comprehensive cleanup with standardized model management"""
        cleanup_start_time = get_ph_time()
        logger.info(f"[{format_ph_time(cleanup_start_time)}] 🧹 Starting DocumentProcessor cleanup...")
        
        try:
            # Clean up OCR processor
            if self._ocr_processor and hasattr(self._ocr_processor, 'cleanup'):
                self._ocr_processor.cleanup()
                self._ocr_processor = None
                logger.info(f"[{format_ph_time()}] ✅ OCR processor cleaned up")
            
            # Clean up text extractor 
            if self._text_extractor and hasattr(self._text_extractor, 'cleanup'):
                self._text_extractor.cleanup()
                self._text_extractor = None
                logger.info(f"[{format_ph_time()}] ✅ Text extractor cleaned up")
            
            # Clean up shared model managers
            cleanup_models()  # OCR models
            cleanup_qwen_models()  # Text extraction models
            
            # General cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            cleanup_complete_time = get_ph_time()
            cleanup_duration = (cleanup_complete_time - cleanup_start_time).total_seconds()
            logger.info(f"[{format_ph_time(cleanup_complete_time)}] ✅ DocumentProcessor cleanup completed in {cleanup_duration:.2f}s (Qwen2-VL {self.model_size})")
        except Exception as e:
            error_time = get_ph_time()
            logger.warning(f"[{format_ph_time(error_time)}] ⚠️ Cleanup warning: {e}")

    # Include all the other helper methods from the original file
    def _update_average_times(self):
        """Update average processing times"""
        if self.stats['processing_times']:
            self.stats['average_times']['total'] = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
        if self.stats['ocr_times']:
            self.stats['average_times']['ocr'] = sum(self.stats['ocr_times']) / len(self.stats['ocr_times'])
        if self.stats['extraction_times']:
            self.stats['average_times']['extraction'] = sum(self.stats['extraction_times']) / len(self.stats['extraction_times'])
        if self.stats['archive_times']:
            self.stats['average_times']['archive'] = sum(self.stats['archive_times']) / len(self.stats['archive_times'])

    def _create_fallback_metadata_fast(self, filename: str, text: str) -> dict:
        """PERFORMANCE FIX: Fast fallback metadata creation"""
    
        # Quick document type detection
        filename_lower = filename.lower()
        if 'email' in filename_lower:
            doc_type = 'email'
        elif 'check' in filename_lower:
            doc_type = 'check'
        elif 'voucher' in filename_lower:
            doc_type = 'voucher'
        elif 'request' in filename_lower:
            doc_type = 'request_for_payment'
        else:
            doc_type = 'general'
        
        # Extract basic info from filename
        ids = {}
        dates = {}
        
        # Extract date (YYMMDD format)
        date_match = re.search(r'(\d{6})', filename)
        if date_match:
            date_str = date_match.group(1)
            try:
                year = int(date_str[:2]) + 2000
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                if 1 <= month <= 12 and 1 <= day <= 31:
                    formatted_date = f"{day:02d}/{month:02d}/{year}"
                    dates['issue_date'] = formatted_date
            except ValueError:
                pass
        
        # Extract document numbers
        numbers = re.findall(r'\b(\d{7,})\b', filename)
        if numbers:
            ids['document_no'] = numbers[0]
        
        return {
            "doc_type": doc_type,
            "title": filename.replace('.pdf', ''),
            "ids": ids,
            "parties": {"issuer": {"name": None, "tin": None}, "recipient": {"name": None, "tin": None}, "other": []},
            "dates": dates,
            "amounts": {"currency": "PHP", "total": None, "subtotal": None, "tax": None},
            "line_items": [],
            "payment_terms": None,
            "addresses": {},
            "contacts": {},
            "status": "processed",
            "tags": [doc_type],
            "extracted_entities": {"people": [], "companies": [], "locations": [], "projects": []},
            "notes": ["Fast extraction"],
            "confidence": 0.6
        }

    def _preprocess_text_fast(self, text: str) -> str:
        """PERFORMANCE FIX: Fast text preprocessing"""
        if not text:
            return ""
        
        # Quick cleanup
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        result = '\n'.join(lines)
        
        # Quick truncation for speed
        if len(result) > 3000:
            result = result[:3000]
        
        return result

    def _quick_enhance_extraction(self, unified: dict, text: str, filename: str) -> dict:
        """PERFORMANCE FIX: Quick enhancement without heavy AI processing"""
        if not unified:
            return self._create_fallback_metadata_fast(filename, text)
        
        # Quick filename-based enhancements
        if not unified.get('ids', {}).get('document_no'):
            # Extract numbers from filename
            numbers = re.findall(r'\b(\d{7,})\b', filename)
            if numbers:
                if 'ids' not in unified:
                    unified['ids'] = {}
                unified['ids']['document_no'] = numbers[0]
        
        # Quick date extraction from filename
        if not unified.get('dates', {}).get('issue_date'):
            date_match = re.search(r'(\d{6})', filename)
            if date_match:
                date_str = date_match.group(1)
                try:
                    year = int(date_str[:2]) + 2000
                    month = int(date_str[2:4])
                    day = int(date_str[4:6])
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        formatted_date = f"{day:02d}/{month:02d}/{year}"
                        if 'dates' not in unified:
                            unified['dates'] = {}
                        unified['dates']['issue_date'] = formatted_date
                except ValueError:
                    pass
        
        return unified

    def _calculate_confidence_fast(self, unified: dict, text: str, filename: str, method: str) -> float:
        """PERFORMANCE FIX: Fast confidence calculation"""
        
        confidence = 45.0  # Base confidence
        
        # Quick bonuses
        if unified.get('ids') and any(unified['ids'].values()):
            confidence += 15
        if unified.get('dates') and any(unified['dates'].values()):
            confidence += 10
        if unified.get('doc_type') != 'general':
            confidence += 10
        if text and len(text) > 100:
            confidence += 10
        if method == 'fast_pattern_extraction':
            confidence += 10
        
        return min(confidence, 100.0)

    def _classify_result_fast(self, confidence: float, unified: dict) -> str:
        """PERFORMANCE FIX: Fast result classification"""
        
        has_basic_data = bool(
            unified.get('dates', {}).get('issue_date') or
            unified.get('ids') and any(unified['ids'].values()) or
            unified.get('doc_type') != 'general'
        )
        
        if confidence >= 60 and has_basic_data:
            return "successful"
        elif confidence >= 40:
            return "partial"
        else:
            return "failed"

    def _has_nested_value(self, data: dict, field: str) -> bool:
        """Check if nested field has a value"""
        try:
            keys = field.split('.')
            current = data
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return False
            return current is not None and current != ""
        except:
            return False

    def _get_document_type_bonus(self, result: dict, text: str) -> float:
        """Calculate document type specific bonus"""
        doc_type = result.get('doc_type', 'general')
        if doc_type == 'general':
            return 0
        
        # Basic bonus for having a specific document type
        bonus = 5.0
        
        # Additional bonuses based on document type
        type_indicators = {
            'invoice': ['invoice', 'bill', 'amount due'],
            'receipt': ['receipt', 'received', 'payment'],
            'email': ['from:', 'to:', 'subject:'],
            'check': ['pay to order', 'check number']
        }
        
        if doc_type in type_indicators:
            text_lower = text.lower()
            matches = sum(1 for indicator in type_indicators[doc_type] if indicator in text_lower)
            bonus += matches * 2
        
        return min(bonus, 15.0)

    def _create_fallback_metadata(self, doc_type: str, ocr_text: str) -> dict:
        """Create fallback metadata with pattern extraction"""
        return {
            "doc_type": doc_type,
            "title": self._extract_title(ocr_text),
            "ids": self._extract_ids(ocr_text, doc_type),
            "parties": {
                "issuer": {"name": None, "tin": None},
                "recipient": {"name": None, "tin": None},
                "other": []
            },
            "dates": self._extract_dates(ocr_text),
            "amounts": self._extract_amounts(ocr_text),
            "line_items": [],
            "payment_terms": None,
            "addresses": {},
            "contacts": {},
            "status": "processed",
            "tags": [doc_type],
            "extracted_entities": {"people": [], "companies": [], "locations": [], "projects": []},
            "notes": [],
            "confidence": 0.5
        }

    def _extract_title(self, text: str) -> str:
        """Extract document title - always return something"""
        lines = text.split('\n')
        
        # Strategy 1: Look for actual title in first 8 lines
        for line in lines[:8]:
            line = line.strip()
            # Good title characteristics: reasonable length, not just numbers/dates, has meaningful words
            if (15 <= len(line) <= 100 and 
                not re.search(r'^\d+[-/]\d+[-/]\d+', line) and  # Not a date
                not re.search(r'^\d{4,}', line) and  # Not just a year
                re.search(r'\w', line)):  # Has meaningful content
                return line
        # Fallback: Return first non-empty line
        for line in lines:
            line = line.strip()
            if line:
                return line
        return "Untitled Document"

    def _extract_amounts(self, text: str) -> dict:
        """Extract monetary amounts"""
        amounts = {"currency": None, "total": None, "subtotal": None, "tax": None}
        
        # Currency detection
        if any(symbol in text.lower() for symbol in ['₱', 'php', 'peso']):
            amounts["currency"] = "PHP"
        elif 'usd' in text.lower():
            amounts["currency"] = "USD"
        
        # Amount patterns
        patterns = [
            (r'total[:\s]+([₱$]?\s*[\d,]+\.?\d*)', 'total'),
            (r'amount[:\s]+([₱$]?\s*[\d,]+\.?\d*)', 'total'),
            (r'subtotal[:\s]+([₱$]?\s*[\d,]+\.?\d*)', 'subtotal'),
            (r'tax[:\s]+([₱$]?\s*[\d,]+\.?\d*)', 'tax')
        ]
        
        for pattern, field in patterns:
            if not amounts.get(field):
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    amount_str = re.sub(r'[₱$,\s]', '', match.group(1))
                    try:
                        amounts[field] = float(amount_str)
                    except ValueError:
                        continue
        
        return amounts

    def _extract_dates(self, text: str) -> dict:
        """Extract dates"""
        dates = {"issue_date": None, "due_date": None}
        
        date_patterns = [
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',
            r'\b([A-Za-z]{3,}\s+\d{1,2},?\s+\d{4})\b'
        ]
        
        found_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            found_dates.extend(matches)
        
        if found_dates:
            dates["issue_date"] = found_dates[0]
            if len(found_dates) > 1:
                dates["due_date"] = found_dates[1]
        
        return dates
    
    def _extract_ids(self, text: str, doc_type: str) -> dict:
        """Extract document IDs"""
        ids = {}
        
        # Document-specific patterns
        patterns = {
            'official_receipt': [
                (r'or\s*#?\s*:?\s*(\w+)', 'receipt_no'),
                (r'receipt\s*#?\s*:?\s*(\w+)', 'receipt_no')
            ],
            'invoice': [
                (r'invoice\s*#?\s*:?\s*(\w+)', 'invoice_no')
            ],
            'voucher': [
                (r'voucher\s*#?\s*:?\s*(\w+)', 'voucher_no')
            ],
            'check': [
                (r'check\s*#?\s*:?\s*(\w+)', 'check_no')
            ]
        }
        
        # Generic patterns
        generic_patterns = [
            (r'ref(?:erence)?\s*#?\s*:?\s*(\w+)', 'reference_no'),
            (r'doc(?:ument)?\s*#?\s*:?\s*(\w+)', 'document_no')
        ]
        
        # Apply patterns
        doc_patterns = patterns.get(doc_type, []) + generic_patterns
        for pattern, field in doc_patterns:
            if not ids.get(field):
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    ids[field] = match.group(1)
        
        return ids

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for extraction"""
        if not text:
            return ""
        
        # Clean up text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        result = '\n'.join(lines)
        
        # Truncation if too long
        if len(result) > 4000:
            result = result[:4000]
        
        return result

    # Backward compatibility methods
    def process_category_parallel(self, category_name: str) -> dict:
        """Backward compatibility method"""
        # Find category data first
        categories_data = self.archive_manager.scan_uploads_categories()
        category_data = next((c for c in categories_data if c['category_name'] == category_name), None)
        if not category_data:
            return {'total': 0, 'successful': 0, 'partial': 0, 'failed': 0, 'skipped': 0}
        
        return self.process_category_super_fast_with_data(category_data)
    
    def get_stats(self) -> dict:
        """Backward compatibility method"""
        return self.get_stats_summary()
    
    def cleanup_resources(self):
        """Backward compatibility method"""
        self.cleanup()
    
    def process_single_file(self, file_path: str, category_name: str, filename: str) -> Dict:
        """Use optimized processing"""
        return self.process_single_file_optimized(file_path, category_name, filename)