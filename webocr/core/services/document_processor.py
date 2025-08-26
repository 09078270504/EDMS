import os
import re
import time
import logging
import concurrent.futures
from pathlib import Path
from multiprocessing import cpu_count
from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils import timezone
import pytz
import threading
import gc
import torch
from core.services.ocr_processor import OCRProcessor
from core.services.qwen_extractor import QwenExtractor
from core.services.archive_manager import ArchiveManager

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Enhanced document processor with maximum accuracy and comprehensive document type support"""
    
    def __init__(self):
        # Optimize workers for stability and accuracy
        self.max_workers = min(cpu_count(), 3)  # Conservative for model stability
        
        # Shared model instances with proper locking
        self._ocr_processor = None
        self._data_extractor = None
        self._model_lock = threading.Lock()
        
        self.archive_manager = ArchiveManager()
        self.retry_counts = {}
        self.progress_lock = threading.Lock()
        
        # Enhanced statistics tracking
        self.stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'partial_extractions': 0,
            'failed_extractions': 0,
            'total_ocr_time': 0,
            'total_extraction_time': 0,
            'total_archive_time': 0,
            'successful_files': [],
            'failed_files': [],
            'document_type_stats': {},
            'confidence_stats': {
                'ocr_confidence_sum': 0,
                'ml_confidence_sum': 0,
                'count': 0
            }
        }
        
        # Processing configuration
        self.processing_config = {
            'ocr': {
                'min_confidence': 0.4,  # Lower threshold for emails and complex documents
                'dpi': 200,
                'enhance_images': True,
                'post_process_text': True
            },
            'extraction': {
                'retry_on_failure': True,
                'max_retries': 2,
                'fallback_mode': True,
                'confidence_threshold': 0.2,
                'context_aware_extraction': True
            },
            'classification': {
                'email_bonus': 15,
                'meaningful_field_bonus': 5,
                'text_length_bonus': True,
                'document_type_bonus': True
            },
            'performance': {
                'memory_cleanup_interval': 2,
                'gpu_memory_limit': '6GiB',
                'batch_size': 1  # Process one at a time for maximum accuracy
            }
        }
        
        logger.info(f"Initialized Enhanced Document Processor with {self.max_workers} workers")
        logger.info("Initializing shared models for maximum accuracy...")
        
        # Pre-initialize models to avoid conflicts
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models with comprehensive error handling and fallback options"""
        try:
            logger.info("Loading Enhanced OCR processor...")
            with self._model_lock:
                if self._ocr_processor is None:
                    self._ocr_processor = OCRProcessor(
                        languages=['en'],
                        gpu=not os.environ.get('FORCE_CPU_ONLY', False),
                        verbose=False,
                        auto_fallback_cpu=True
                    )
            logger.info("Enhanced OCR processor loaded successfully")
            
            logger.info("Loading Enhanced Qwen extractor...")
            with self._model_lock:
                if self._data_extractor is None:
                    try:
                        device = "cuda" if torch.cuda.is_available() and not os.environ.get('FORCE_CPU_ONLY') else "cpu"
                        self._data_extractor = QwenExtractor(device=device)
                        
                        # Test the extractor
                        test_result = self._data_extractor._test_model_functionality()
                        if not test_result:
                            logger.warning("Qwen model test failed, but continuing with fallback mode")
                        else:
                            logger.info("Qwen model test passed - ready for extraction")
                            
                    except Exception as e:
                        logger.error(f"Qwen initialization failed: {e}")
                        logger.info("Creating fallback-only extractor")
                        self._data_extractor = QwenExtractor(force_cpu=True)
                        
            logger.info("All enhanced models loaded successfully")
            
        except Exception as e:
            logger.error(f"Critical failure in model initialization: {e}")
            raise
    
    def _get_ocr_processor(self):
        """Get shared OCR processor instance"""
        if self._ocr_processor is None:
            with self._model_lock:
                if self._ocr_processor is None:
                    self._ocr_processor = OCRProcessor(
                        languages=['en'],
                        gpu=not os.environ.get('FORCE_CPU_ONLY', False),
                        verbose=False,
                        auto_fallback_cpu=True
                    )
        return self._ocr_processor
    
    def _get_data_extractor(self):
        """Get shared data extractor instance"""
        if self._data_extractor is None:
            with self._model_lock:
                if self._data_extractor is None:
                    self._data_extractor = self._create_qwen_extractor()
        return self._data_extractor

    def _create_qwen_extractor(self):
        """Create Qwen extractor with proper device handling"""
        try:
            device = "cuda" if torch.cuda.is_available() and not os.environ.get('FORCE_CPU_ONLY') else "cpu"
            logger.info(f"Initializing Qwen on device: {device}")
            extractor = QwenExtractor(device=device)
            return extractor
        except Exception as e:
            logger.error(f"Failed to create Qwen extractor: {e}")
            try:
                logger.info("Falling back to CPU-only mode for Qwen")
                extractor = QwenExtractor(device="cpu", force_cpu=True)
                return extractor
            except Exception as e2:
                logger.error(f"CPU fallback also failed: {e2}")
                raise e2
    
    def _update_progress(self, completed, total, filename, status):
        """Thread-safe progress updates with enhanced information"""
        with self.progress_lock:
            progress_pct = (completed / total) * 100
            logger.info(f"Progress: {completed}/{total} ({progress_pct:.1f}%) - {status}: {filename}")
    
    def _cleanup_memory(self):
        """Enhanced memory cleanup"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
                
            logger.debug("Memory cleanup completed")
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    def process_single_file(self, file_path, category_name, filename):
        """Enhanced single file processing with maximum accuracy"""
        overall_start = time.time()
        
        try:
            document_name = self.archive_manager.generate_document_name(filename, category_name)
            
            # Step 1: Enhanced OCR extraction with post-processing
            ocr_start = time.time()
            try:
                ocr_processor = self._get_ocr_processor()
                ocr_result = ocr_processor.extract_text_from_pdf(
                    file_path,
                    enhance=self.processing_config['ocr']['enhance_images'],
                    min_confidence=self.processing_config['ocr']['min_confidence'],
                    dpi=self.processing_config['ocr']['dpi']
                )
                
                # Get both raw and processed text
                raw_ocr_text = ocr_result.get("raw_text", "")
                processed_ocr_text = ocr_result.get("text_only", "")
                ocr_confidence = ocr_result.get("avg_confidence", 0.0)
                
                # Use processed text for extraction
                ocr_text = processed_ocr_text if processed_ocr_text else raw_ocr_text
                ocr_time = time.time() - ocr_start
                
                # Enhanced OCR validation
                if not ocr_text or len(ocr_text.strip()) < 20:
                    raise Exception(f"OCR produced insufficient text: {len(ocr_text)} characters")
                
                logger.debug(f"OCR completed for {filename}: {len(ocr_text)} chars, {ocr_confidence:.1f}% confidence")
                
            except Exception as e:
                ocr_time = time.time() - ocr_start
                error_message = f"OCR failed: {str(e)}"
                logger.error(f"OCR ERROR {filename}: {error_message}")
                
                return {
                    'status': 'skipped',
                    'classification': 'failed',
                    'ocr_confidence': 0,
                    'ml_confidence': 0,
                    'processing_time': time.time() - overall_start,
                    'ocr_time': ocr_time,
                    'error': error_message,
                    'failure_stage': 'ocr'
                }
            
            # Step 2: Enhanced data extraction with multiple strategies
            extraction_start = time.time()
            extracted_data = {}
            ml_confidence = 0
            extraction_error = None
            extraction_method = "none"
            
            try:
                data_extractor = self._get_data_extractor()
                
                # Strategy 1: Standard extraction
                extraction_result = data_extractor.extract_structured_data(ocr_text, category_name.lower())
                extracted_data = extraction_result.get('data', {})
                ml_confidence = extraction_result.get('confidence', 0.0) * 100
                extraction_method = extraction_result.get('method', 'ai_extraction')
                
                # Strategy 2: Check if extraction was successful, if not try context-aware extraction
                meaningful_fields = self._count_meaningful_fields(extracted_data, ocr_text)
                if meaningful_fields == 0 and not data_extractor.fallback_mode:
                    logger.info(f"Retrying extraction for {filename} with enhanced context")
                    enhanced_result = self._retry_extraction_with_context(data_extractor, ocr_text, filename, category_name)
                    if enhanced_result and enhanced_result.get('data'):
                        extracted_data = enhanced_result.get('data', {})
                        ml_confidence = enhanced_result.get('confidence', 0.0) * 100
                        extraction_method = enhanced_result.get('method', 'context_retry')
                
                # Strategy 3: If still no meaningful data, try document-specific patterns
                if meaningful_fields == 0:
                    logger.info(f"Applying pattern-based extraction for {filename}")
                    pattern_result = self._apply_pattern_extraction(ocr_text, filename, category_name)
                    if pattern_result:
                        extracted_data.update(pattern_result)
                        ml_confidence = max(ml_confidence, 30)  # Give some confidence for pattern matching
                        extraction_method = "pattern_enhanced"
                
            except Exception as e:
                extraction_error = str(e)
                logger.error(f"EXTRACTION ERROR {filename}: {extraction_error}")
                # Continue with OCR-only processing
                
            extraction_time = time.time() - extraction_start
            
            # Step 3: Enhanced classification with comprehensive criteria
            classification = self._classify_result_enhanced(
                ocr_confidence, ml_confidence, len(ocr_text), 
                extracted_data, filename, extraction_error, category_name
            )
            
            # Update document type statistics
            doc_type = self._determine_document_type_from_content(ocr_text, filename)
            self._update_document_type_stats(doc_type, classification)
            
            # Step 4: Intelligent retry logic
            file_key = str(file_path)
            current_retries = self.retry_counts.get(file_key, 0)
            
            if classification == 'failed' and self._should_retry(ocr_confidence, ml_confidence, current_retries, filename, doc_type):
                self.retry_counts[file_key] = current_retries + 1
                processing_time = time.time() - overall_start
                error_message = f"Retry {current_retries + 1}/3 - OCR: {ocr_confidence:.1f}%, ML: {ml_confidence:.1f}%"
                logger.info(f"RETRY {filename}: {error_message}")
                
                return {
                    'status': 'retry',
                    'classification': 'retry',
                    'ocr_confidence': ocr_confidence,
                    'ml_confidence': ml_confidence,
                    'processing_time': processing_time,
                    'error': error_message,
                    'retry_count': current_retries + 1
                }
            
            # Step 5: Archive with comprehensive metadata
            if classification in ['successful', 'partial']:
                archive_start = time.time()
                
                # Comprehensive metadata
                metadata = {
                    'file_info': {
                        'original_filename': filename,
                        'file_size': os.path.getsize(file_path),
                        'category': category_name,
                        'document_name': document_name,
                        'detected_document_type': doc_type
                    },
                    'processing_results': {
                        'ocr_confidence': ocr_confidence,
                        'ml_confidence': ml_confidence,
                        'overall_confidence': self._calculate_overall_confidence(ocr_confidence, ml_confidence, extracted_data),
                        'processing_time_seconds': time.time() - overall_start,
                        'ocr_time_seconds': ocr_time,
                        'extraction_time_seconds': extraction_time,
                        'classification': classification,
                        'text_length': len(ocr_text),
                        'retry_count': current_retries,
                        'meaningful_fields_extracted': self._count_meaningful_fields(extracted_data, ocr_text),
                        'extraction_error': extraction_error,
                        'extraction_method': extraction_method,
                        'document_type_detected': doc_type
                    },
                    'quality_metrics': {
                        'text_density': len(ocr_text.split()) / max(len(ocr_text), 1),
                        'field_completeness': self._calculate_field_completeness(extracted_data, doc_type),
                        'confidence_score': self._calculate_confidence_score(ocr_confidence, ml_confidence, extracted_data)
                    },
                    'extracted_data': extracted_data,
                    'processing_info': {
                        'category': category_name,
                        'classification': classification,
                        'document_name': document_name,
                        'original_filename': filename,
                        'processed_at': timezone.now().astimezone(pytz.timezone('Asia/Manila')).isoformat(),
                        'worker_count': self.max_workers,
                        'processing_version': '3.0_enhanced_accuracy',
                        'config_used': self.processing_config
                    }
                }
                
                try:
                    archive_paths = self.archive_manager.create_archive_structure(
                        category_name=category_name,
                        document_name=document_name,
                        pdf_path=file_path,
                        ocr_text=ocr_text,
                        metadata=metadata,
                        classification=classification
                    )
                    archive_time = time.time() - archive_start
                    
                    # Cleanup
                    self.archive_manager.cleanup_upload_file(file_path)
                    self.retry_counts.pop(file_key, None)
                    
                except Exception as e:
                    archive_time = time.time() - archive_start
                    error_message = f"Archive failed: {str(e)}"
                    logger.error(f"ARCHIVE ERROR {filename}: {error_message}")
                    
                    return {
                        'status': 'skipped',
                        'classification': 'failed',
                        'ocr_confidence': ocr_confidence,
                        'ml_confidence': ml_confidence,
                        'processing_time': time.time() - overall_start,
                        'error': error_message,
                        'failure_stage': 'archive'
                    }
                
                # Update comprehensive statistics
                processing_time = time.time() - overall_start
                self._update_comprehensive_stats(processing_time, ocr_time, extraction_time, archive_time, filename, classification, ocr_confidence, ml_confidence, doc_type)
                
                logger.info(f"SUCCESS {filename}: {classification} ({doc_type}) in {processing_time:.1f}s "
                          f"(OCR: {ocr_confidence:.1f}%, ML: {ml_confidence:.1f}%, "
                          f"fields: {self._count_meaningful_fields(extracted_data, ocr_text)}, method: {extraction_method})")
                
                return {
                    'status': 'completed',
                    'classification': classification,
                    'document_type': doc_type,
                    'ocr_confidence': ocr_confidence,
                    'ml_confidence': ml_confidence,
                    'processing_time': processing_time,
                    'ocr_time': ocr_time,
                    'extraction_time': extraction_time,
                    'archive_time': archive_time,
                    'text_length': len(ocr_text),
                    'archive_paths': archive_paths,
                    'meaningful_fields': self._count_meaningful_fields(extracted_data, ocr_text),
                    'extraction_method': extraction_method
                }
            else:
                processing_time = time.time() - overall_start
                error_message = f"Classification failed - OCR: {ocr_confidence:.1f}%, ML: {ml_confidence:.1f}%, Type: {doc_type}"
                logger.warning(f"SKIPPED {filename}: {error_message} in {processing_time:.1f}s")
                
                self.stats['failed_files'].append({
                    'filename': filename,
                    'error': error_message,
                    'ocr_confidence': ocr_confidence,
                    'ml_confidence': ml_confidence,
                    'document_type': doc_type,
                    'extraction_method': extraction_method
                })
                
                return {
                    'status': 'skipped',
                    'classification': 'failed',
                    'document_type': doc_type,
                    'ocr_confidence': ocr_confidence,
                    'ml_confidence': ml_confidence,
                    'processing_time': processing_time,
                    'error': error_message,
                    'failure_stage': 'classification'
                }
        
        except Exception as e:
            processing_time = time.time() - overall_start
            error_message = f"Processing exception: {str(e)}"
            logger.error(f"ERROR {filename}: {error_message}")
            
            self.stats['failed_files'].append({
                'filename': filename,
                'error': error_message,
                'exception': str(e),
                'processing_time': processing_time
            })
            
            return {
                'status': 'skipped',
                'classification': 'failed',
                'ocr_confidence': 0,
                'ml_confidence': 0,
                'processing_time': processing_time,
                'error': error_message,
                'failure_stage': 'exception'
            }

    def _determine_document_type_from_content(self, text, filename):
        """Determine document type from content and filename"""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Check filename first
        filename_types = {
            'invoice': ['invoice', 'bill'],
            'voucher': ['voucher', 'check', 'payment'],
            'receipt': ['receipt', 'or'],
            'email': ['email', 'fwd', 'fw', 're:', 'reply'],
            'contract': ['contract', 'agreement', 'mou'],
            'purchase_order': ['po', 'purchase', 'order'],
            'statement': ['statement', 'soa'],
            'tax': ['tax', 'bir', '2307', 'withholding'],
            'delivery': ['delivery', 'dr', 'shipping']
        }
        
        for doc_type, keywords in filename_types.items():
            if any(keyword in filename_lower for keyword in keywords):
                return doc_type
        
        # Check content patterns
        content_patterns = {
            'email': ['from:', 'to:', 'subject:', 'sent:', '@'],
            'invoice': ['invoice', 'bill to', 'amount due', 'tax invoice'],
            'voucher': ['voucher', 'check number', 'batch number', 'accounts payable'],
            'receipt': ['receipt', 'received from', 'official receipt'],
            'purchase_order': ['purchase order', 'po number', 'supplier'],
            'contract': ['agreement', 'contract', 'terms and conditions'],
            'financial_statement': ['balance sheet', 'income statement', 'financial statement'],
            'bank_statement': ['bank statement', 'account statement', 'transaction'],
            'tax_document': ['tax return', 'bir form', 'withholding tax'],
            'delivery_receipt': ['delivery receipt', 'goods received', 'delivered']
        }
        
        for doc_type, patterns in content_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score >= 2:  # Need at least 2 pattern matches
                return doc_type
        
        return 'general'

    def _update_document_type_stats(self, doc_type, classification):
        """Update document type statistics"""
        if doc_type not in self.stats['document_type_stats']:
            self.stats['document_type_stats'][doc_type] = {
                'total': 0, 'successful': 0, 'partial': 0, 'failed': 0
            }
        
        self.stats['document_type_stats'][doc_type]['total'] += 1
        self.stats['document_type_stats'][doc_type][classification] += 1

    def _calculate_field_completeness(self, extracted_data, doc_type):
        """Calculate how complete the extracted data is for the document type"""
        if not extracted_data:
            return 0.0
        
        # Define expected fields for each document type
        expected_fields = {
            'invoice': ['invoice_number', 'vendor_name', 'total_amount', 'invoice_date'],
            'voucher': ['voucher_number', 'vendor_name', 'amount_due', 'batch_number'],
            'email': ['sender_email', 'subject', 'date_sent'],
            'receipt': ['receipt_number', 'total_amount', 'store_name'],
            'purchase_order': ['po_number', 'supplier_name', 'total_amount'],
            'contract': ['contract_number', 'party_a', 'party_b'],
            'general': ['document_number', 'date', 'amount']
        }
        
        required_fields = expected_fields.get(doc_type, expected_fields['general'])
        found_fields = sum(1 for field in required_fields if extracted_data.get(field))
        
        return found_fields / len(required_fields) if required_fields else 0.0

    def _calculate_confidence_score(self, ocr_confidence, ml_confidence, extracted_data):
        """Calculate overall confidence score"""
        base_score = (ocr_confidence + ml_confidence) / 2
        
        # Add bonus for extracted data
        field_bonus = min(len(extracted_data) * 2, 20)
        
        return min(base_score + field_bonus, 100)

    def _retry_extraction_with_context(self, data_extractor, ocr_text, filename, category):
        """Retry extraction with enhanced context awareness"""
        try:
            # Determine specific context from filename and content
            doc_type = self._determine_document_type_from_content(ocr_text, filename)
            
            # Use document-specific extraction
            result = data_extractor.extract_structured_data(ocr_text, doc_type)
            return result
            
        except Exception as e:
            logger.warning(f"Context-aware retry extraction failed for {filename}: {e}")
            return None

    def _apply_pattern_extraction(self, text, filename, category):
        """Apply pattern-based extraction as fallback"""
        try:
            patterns = {}
            
            # Common patterns for all documents
            common_patterns = {
                'amounts': r'(?:USD|PHP|EUR|\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                'dates': r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                'email_addresses': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                'phone_numbers': r'(\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9})',
                'reference_numbers': r'(?:no\.?|number|#)\s*:?\s*([A-Z0-9-]+)',
            }
            
            # Extract using patterns
            extracted = {}
            for key, pattern in common_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    if key == 'amounts':
                        extracted[key] = [{'amount': m, 'currency': 'USD'} for m in matches[:3]]
                    else:
                        extracted[key] = matches[:5]  # Limit to first 5 matches
            
            return extracted
            
        except Exception as e:
            logger.warning(f"Pattern extraction failed for {filename}: {e}")
            return {}

    def _classify_result_enhanced(self, ocr_confidence, ml_confidence, text_length, extracted_data, filename, extraction_error, category):
        """Enhanced classification with comprehensive criteria"""
        # Calculate meaningful field bonus
        meaningful_fields = self._count_meaningful_fields(extracted_data, "")
        field_bonus = min(meaningful_fields * 3, 15)  # Up to 15% bonus
        
        # Text length bonus for substantial content
        text_bonus = min(8, text_length / 250) if text_length > 100 else 0
        
        # Document type bonus
        doc_type = self._determine_document_type_from_content("", filename)
        type_bonus = self.processing_config['classification'].get('document_type_bonus', 0)
        if type_bonus and doc_type in ['email', 'invoice', 'voucher']:
            type_bonus = 10
        else:
            type_bonus = 0
        
        # Category bonus
        category_bonus = 5 if category and any(word in category.lower() for word in ['important', 'priority', 'urgent']) else 0
        
        # Adjust scores
        adjusted_ocr = ocr_confidence + text_bonus + type_bonus + category_bonus
        adjusted_ml = ml_confidence + field_bonus
        
        # Enhanced classification logic with more granular conditions
        if adjusted_ocr >= 85 and adjusted_ml >= 50:
            return 'successful'
        elif adjusted_ocr >= 75 and adjusted_ml >= 35:
            return 'successful'
        elif adjusted_ocr >= 65 and meaningful_fields >= 4:
            return 'successful'
        elif adjusted_ocr >= 55 and adjusted_ml >= 25:
            return 'successful'
        elif adjusted_ocr >= 45 and meaningful_fields >= 3:
            return 'partial'
        elif adjusted_ocr >= 35 and adjusted_ml >= 15:
            return 'partial'
        elif adjusted_ocr >= 50:  # High OCR confidence alone
            return 'partial'
        elif meaningful_fields >= 2:  # Some useful data extracted
            return 'partial'
        else:
            return 'failed'

    def _count_meaningful_fields(self, extracted_data, ocr_text):
        """Count meaningful extracted fields with enhanced validation"""
        if not extracted_data:
            return 0
        
        meaningful_count = 0
        
        for key, value in extracted_data.items():
            if not value:
                continue
                
            if isinstance(value, str):
                cleaned_value = value.strip()
                if len(cleaned_value) >= 2 and cleaned_value.lower() not in ['n/a', 'none', 'null', '']:
                    # Give higher weight to important fields
                    weight = 2 if any(important in key.lower() for important in ['number', 'amount', 'date', 'name', 'email']) else 1
                    meaningful_count += weight
            elif isinstance(value, (int, float)):
                if value > 0:
                    meaningful_count += 2  # Numbers are valuable
            elif isinstance(value, dict):
                if value:  # Non-empty dict
                    meaningful_count += 1
            elif isinstance(value, list):
                if value and len(value) > 0:  # Non-empty list
                    meaningful_count += len(value) if len(value) <= 3 else 3  # Cap contribution
        
        return meaningful_count

    def _should_retry(self, ocr_confidence, ml_confidence, current_retries, filename, doc_type):
        """Enhanced retry logic with document type awareness"""
        if current_retries >= self.processing_config['extraction']['max_retries']:
            return False
        
        # Retry if OCR is decent but extraction failed
        if ocr_confidence >= 60 and ml_confidence < 15:
            return True
        
        # Retry for important document types even with lower OCR
        important_types = ['invoice', 'voucher', 'contract', 'email']
        if doc_type in important_types and ocr_confidence >= 45:
            return True
        
        # Retry if it looks like a structured document
        if ocr_confidence >= 40 and any(indicator in filename.lower() for indicator in ['invoice', 'voucher', 'receipt', 'order']):
            return True
        
        return False

    def _calculate_overall_confidence(self, ocr_confidence, ml_confidence, extracted_data):
        """Calculate weighted overall confidence with field quality bonus"""
        meaningful_fields = self._count_meaningful_fields(extracted_data, "")
        
        # Weight OCR more heavily but give significant bonus for successful extraction
        if meaningful_fields >= 4:
            # Excellent extraction boosts confidence significantly
            base_confidence = (ocr_confidence * 0.5) + (ml_confidence * 0.5)
            extraction_bonus = min(meaningful_fields * 2, 20)
            return min(base_confidence + extraction_bonus, 100)
        elif meaningful_fields >= 2:
            # Good extraction
            base_confidence = (ocr_confidence * 0.6) + (ml_confidence * 0.4)
            extraction_bonus = min(meaningful_fields * 3, 15)
            return min(base_confidence + extraction_bonus, 100)
        else:
            # Poor extraction penalizes confidence
            return (ocr_confidence * 0.8) + (ml_confidence * 0.2)

    def _update_comprehensive_stats(self, processing_time, ocr_time, extraction_time, archive_time, filename, classification, ocr_confidence, ml_confidence, doc_type):
        """Update comprehensive processing statistics"""
        self.stats['total_processed'] += 1
        self.stats['total_ocr_time'] += ocr_time
        self.stats['total_extraction_time'] += extraction_time
        self.stats['total_archive_time'] += archive_time
        
        # Update classification stats
        if classification == 'successful':
            self.stats['successful_extractions'] += 1
        elif classification == 'partial':
            self.stats['partial_extractions'] += 1
        else:
            self.stats['failed_extractions'] += 1
        
        # Update confidence stats
        self.stats['confidence_stats']['ocr_confidence_sum'] += ocr_confidence
        self.stats['confidence_stats']['ml_confidence_sum'] += ml_confidence
        self.stats['confidence_stats']['count'] += 1
        
        self.stats['successful_files'].append({
            'filename': filename,
            'classification': classification,
            'document_type': doc_type,
            'processing_time': processing_time,
            'ocr_time': ocr_time,
            'extraction_time': extraction_time,
            'ocr_confidence': ocr_confidence,
            'ml_confidence': ml_confidence
        })

    def process_category_parallel(self, category_name):
        """Enhanced parallel processing with comprehensive error handling"""
        logger.info(f"Processing category: {category_name}")
        
        categories_data = self.archive_manager.scan_uploads_categories()
        category_data = next((c for c in categories_data if c['category_name'] == category_name), None)
        
        if not category_data:
            return {'total': 0, 'successful': 0, 'partial': 0, 'failed': 0, 'skipped': 0}
        
        pdf_files = category_data['pdf_files']
        results = {'total': len(pdf_files), 'successful': 0, 'partial': 0, 'failed': 0, 'skipped': 0, 'retries': 0}
        
        if not pdf_files:
            return results
        
        logger.info(f"Processing {len(pdf_files)} files with {self.max_workers} workers")
        
        completed_count = 0
        start_time = time.time()
        
        # Use conservative number of workers for accuracy
        actual_workers = min(self.max_workers, len(pdf_files), 2)
        logger.info(f"Using {actual_workers} workers for maximum accuracy")

# changed ehrerehre========================================================
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
            future_to_file = {
                executor.submit(self.process_single_file, pdf['full_path'], category_name, pdf['filename']): pdf
                for pdf in pdf_files
            }

            try:
                for future in concurrent.futures.as_completed(future_to_file, timeout=10800):  # 3 hour timeout
                    pdf_info = future_to_file[future]
                    completed_count += 1

                    try:
                        result = future.result(timeout=1200)  # 20 min per file timeout

                        if result:
                            if result.get('status') == 'completed':
                                classification = result.get('classification', 'failed')
                                results[classification] += 1
                                self._update_progress(completed_count, len(pdf_files), pdf_info['filename'], f'COMPLETED-{classification.upper()}')
                            elif result.get('status') == 'retry':
                                results['retries'] += 1
                                self._update_progress(completed_count, len(pdf_files), pdf_info['filename'], 'RETRY')
                            elif result.get('status') == 'skipped':
                                results['skipped'] += 1
                                self._update_progress(completed_count, len(pdf_files), pdf_info['filename'], 'SKIPPED')
                            else:
                                results['failed'] += 1
                        else:
                            results['failed'] += 1

                    except concurrent.futures.TimeoutError:
                        logger.error(f"TIMEOUT: {pdf_info['filename']}")
                        results['skipped'] += 1
                        self._update_progress(completed_count, len(pdf_files), pdf_info['filename'], 'TIMEOUT')
                    except Exception as e:
                        logger.error(f"EXECUTOR ERROR: {pdf_info['filename']}: {e}")
                        results['skipped'] += 1
                        self._update_progress(completed_count, len(pdf_files), pdf_info['filename'], 'ERROR')

                    # Enhanced progress summary every 2 files
                    if completed_count % 2 == 0 or completed_count == len(pdf_files):
                        elapsed = time.time() - start_time
                        avg_time = elapsed / completed_count
                        remaining = len(pdf_files) - completed_count
                        eta = remaining * avg_time

                        successful_so_far = results['successful'] + results['partial']
                        success_rate = (successful_so_far / completed_count) * 100 if completed_count > 0 else 0

                        logger.info(
                            f"Batch progress: {completed_count}/{len(pdf_files)} "
                            f"({success_rate:.1f}% success rate, {results['retries']} retries, ETA: {eta:.0f}s)"
                        )

            except KeyboardInterrupt:
                logger.warning("Interrupted by user. Cancelling outstanding tasks…")
                for fut in future_to_file.keys():
                    fut.cancel()
                # Re-raise so the caller (management command) exits promptly
                raise

        
        # Enhanced memory cleanup after batch
        self._cleanup_memory()
        
        # Add timing information
        total_time = time.time() - start_time
        results['processing_time'] = total_time
        results['avg_time_per_file'] = total_time / len(pdf_files) if pdf_files else 0
        
        logger.info(f"Category {category_name} complete: {results}")
        return results
    
    def process_all_categories(self):
        """Process all categories with enhanced monitoring and statistics"""
        categories_data = self.archive_manager.scan_uploads_categories()
        category_results = []
        total_start_time = time.time()
        
        logger.info(f"Starting enhanced processing of {len(categories_data)} categories")
        
        for i, category_data in enumerate(categories_data, 1):
            category_name = category_data['category_name']
            file_count = len(category_data.get('pdf_files', []))
            
            logger.info(f"Category {i}/{len(categories_data)}: {category_name} ({file_count} files)")
            
            if file_count == 0:
                logger.info(f"Skipping empty category: {category_name}")
                category_results.append({
                    'name': category_name, 
                    'results': {'total': 0, 'successful': 0, 'partial': 0, 'failed': 0, 'skipped': 0}
                })
                continue
            
            category_start = time.time()
            results = self.process_category_parallel(category_name)
            category_time = time.time() - category_start
            
            results['processing_time'] = category_time
            category_results.append({'name': category_name, 'results': results})
            
            # Enhanced category summary
            successful = results['successful'] + results['partial']
            success_rate = (successful / results['total']) * 100 if results['total'] > 0 else 0
            avg_time_per_file = category_time / results['total'] if results['total'] > 0 else 0
            
            logger.info(f"Category {category_name} completed: {successful}/{results['total']} "
                      f"({success_rate:.1f}% success, {results.get('retries', 0)} retries) "
                      f"in {category_time:.1f}s ({avg_time_per_file:.1f}s/file)")
            
            # Cleanup between categories
            self._cleanup_memory()
        
        total_time = time.time() - total_start_time
        logger.info(f"All categories completed in {total_time:.1f}s")
        
        # Log comprehensive statistics
        self._log_comprehensive_performance_summary()
        
        return category_results

    def watch_mode(self, interval=30, category=None):
        """Enhanced watch mode with adaptive intervals and comprehensive monitoring"""
        logger.info(f"ENHANCED WATCH MODE STARTED - checking every {interval} seconds")
        if category:
            logger.info(f"Watching category: {category}")
        else:
            logger.info("Watching all categories")
        
        scan_count = 0
        consecutive_empty_scans = 0
        dynamic_interval = interval
        
        try:
            while True:
                scan_count += 1
                gmt_plus_8 = pytz.timezone('Asia/Manila')
                current_time = timezone.now().astimezone(gmt_plus_8)
                logger.info(f"Enhanced Scan #{scan_count} at {current_time.strftime('%H:%M:%S')} "
                          f"(interval: {dynamic_interval}s)")
                
                categories_data = self.archive_manager.scan_uploads_categories()
                new_files_found = False
                total_processed = 0
                
                for category_data in categories_data:
                    category_name = category_data['category_name']
                    
                    if category and category_name != category:
                        continue
                    
                    if category_data['pdf_files']:
                        new_files_found = True
                        file_count = len(category_data['pdf_files'])
                        logger.info(f"Processing {file_count} files in {category_name}")
                        
                        try:
                            batch_start = time.time()
                            results = self.process_category_parallel(category_name)
                            batch_time = time.time() - batch_start
                            
                            successful = results['successful'] + results['partial']
                            total_processed += successful
                            
                            logger.info(f"Batch results for {category_name}: {successful}/{results['total']} "
                                      f"processed in {batch_time:.1f}s (success rate: {(successful/results['total']*100):.1f}%)")
                        except Exception as e:
                            logger.error(f"Error processing {category_name}: {e}")
                
                # Adaptive interval adjustment
                if new_files_found:
                    consecutive_empty_scans = 0
                    dynamic_interval = max(15, interval // 2)  # Faster when busy
                    logger.info(f"Processed {total_processed} files this cycle")
                else:
                    consecutive_empty_scans += 1
                    if consecutive_empty_scans >= 3:
                        dynamic_interval = min(interval * 2, 300)  # Slower when idle, max 5 min
                    logger.info("No files found in upload directories")
                
                # Performance summary every 10 scans
                if scan_count % 10 == 0 and self.stats['total_processed'] > 0:
                    self._log_comprehensive_performance_summary()
                
                # Memory cleanup every 2 scans
                if scan_count % 2 == 0:
                    self._cleanup_memory()
                
                logger.info(f"Sleeping for {dynamic_interval} seconds...")
                time.sleep(dynamic_interval)
        
        except KeyboardInterrupt:
            logger.info(f"Enhanced watch mode stopped by user after {scan_count} scans")
            self._log_comprehensive_performance_summary()

    def _log_comprehensive_performance_summary(self):
        """Log detailed performance statistics with document type breakdown"""
        if self.stats['total_processed'] == 0:
            return
        
        total_files = self.stats['total_processed']
        avg_ocr = self.stats['total_ocr_time'] / total_files
        avg_extraction = self.stats['total_extraction_time'] / total_files
        avg_archive = self.stats['total_archive_time'] / total_files
        
        # Calculate average confidences
        conf_stats = self.stats['confidence_stats']
        avg_ocr_conf = conf_stats['ocr_confidence_sum'] / conf_stats['count'] if conf_stats['count'] > 0 else 0
        avg_ml_conf = conf_stats['ml_confidence_sum'] / conf_stats['count'] if conf_stats['count'] > 0 else 0
        
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total files processed: {total_files}")
        logger.info(f"Successful extractions: {self.stats['successful_extractions']} ({(self.stats['successful_extractions']/total_files)*100:.1f}%)")
        logger.info(f"Partial extractions: {self.stats['partial_extractions']} ({(self.stats['partial_extractions']/total_files)*100:.1f}%)")
        logger.info(f"Failed extractions: {self.stats['failed_extractions']} ({(self.stats['failed_extractions']/total_files)*100:.1f}%)")
        logger.info("")
        logger.info(f"Average OCR confidence: {avg_ocr_conf:.1f}%")
        logger.info(f"Average ML confidence: {avg_ml_conf:.1f}%")
        logger.info("")
        logger.info(f"Average OCR time: {avg_ocr:.2f}s ({(avg_ocr/(avg_ocr+avg_extraction+avg_archive))*100:.1f}%)")
        logger.info(f"Average extraction time: {avg_extraction:.2f}s ({(avg_extraction/(avg_ocr+avg_extraction+avg_archive))*100:.1f}%)")
        logger.info(f"Average archive time: {avg_archive:.2f}s ({(avg_archive/(avg_ocr+avg_extraction+avg_archive))*100:.1f}%)")
        logger.info("")
        
        # Document type breakdown
        if self.stats['document_type_stats']:
            logger.info("DOCUMENT TYPE BREAKDOWN:")
            for doc_type, type_stats in self.stats['document_type_stats'].items():
                total_type = type_stats['total']
                success_rate = ((type_stats['successful'] + type_stats['partial']) / total_type) * 100 if total_type > 0 else 0
                logger.info(f"  {doc_type}: {total_type} files ({success_rate:.1f}% success rate)")
        
        logger.info("")
        logger.info(f"Failed files: {len(self.stats['failed_files'])}")
        
        if len(self.stats['failed_files']) > 0:
            logger.info("Recent failures:")
            for failure in self.stats['failed_files'][-5:]:  # Last 5 failures
                logger.info(f"  - {failure['filename']}: {failure.get('error', 'Unknown error')}")
        
        logger.info("=" * 80)

    def get_comprehensive_stats_summary(self):
        """Get comprehensive statistics summary"""
        if self.stats['total_processed'] == 0:
            return "No files processed yet"
        
        total_files = self.stats['total_processed']
        avg_ocr = self.stats['total_ocr_time'] / total_files
        avg_extraction = self.stats['total_extraction_time'] / total_files
        avg_archive = self.stats['total_archive_time'] / total_files
        total_avg = avg_ocr + avg_extraction + avg_archive
        
        # Calculate confidence averages
        conf_stats = self.stats['confidence_stats']
        avg_ocr_conf = conf_stats['ocr_confidence_sum'] / conf_stats['count'] if conf_stats['count'] > 0 else 0
        avg_ml_conf = conf_stats['ml_confidence_sum'] / conf_stats['count'] if conf_stats['count'] > 0 else 0
        
        return {
            'total_processed': total_files,
            'success_breakdown': {
                'successful': self.stats['successful_extractions'],
                'partial': self.stats['partial_extractions'],
                'failed': self.stats['failed_extractions']
            },
            'confidence_averages': {
                'ocr_confidence': round(avg_ocr_conf, 1),
                'ml_confidence': round(avg_ml_conf, 1)
            },
            'average_times': {
                'ocr': round(avg_ocr, 2),
                'extraction': round(avg_extraction, 2),
                'archive': round(avg_archive, 2),
                'total': round(total_avg, 2)
            },
            'throughput_per_hour': round(3600 / total_avg) if total_avg > 0 else 0,
            'document_type_stats': self.stats['document_type_stats'],
            'failed_count': len(self.stats['failed_files']),
            'overall_success_rate': round(((self.stats['successful_extractions'] + self.stats['partial_extractions']) / total_files) * 100, 1)
        }
    

    def get_stats_summary(self):
        """Return a comprehensive stats summary compatible with process_documents.py."""
        # If nothing processed yet, return zeros with expected shape
        total = self.stats.get('total_processed', 0)
        total_ocr_time = self.stats.get('total_ocr_time', 0.0)
        total_extraction_time = self.stats.get('total_extraction_time', 0.0)
        total_archive_time = self.stats.get('total_archive_time', 0.0)

        avg_ocr = (total_ocr_time / total) if total > 0 else 0.0
        avg_extraction = (total_extraction_time / total) if total > 0 else 0.0
        avg_archive = (total_archive_time / total) if total > 0 else 0.0
        total_avg = avg_ocr + avg_extraction + avg_archive

        conf_stats = self.stats.get('confidence_stats', {'ocr_confidence_sum': 0.0, 'ml_confidence_sum': 0.0, 'count': 0})
        count = conf_stats.get('count', 0)
        avg_ocr_conf = (conf_stats.get('ocr_confidence_sum', 0.0) / count) if count > 0 else 0.0
        avg_ml_conf = (conf_stats.get('ml_confidence_sum', 0.0) / count) if count > 0 else 0.0

        successful = self.stats.get('successful_extractions', 0)
        partial = self.stats.get('partial_extractions', 0)
        failed = self.stats.get('failed_extractions', 0)
        success_rate = ((successful + partial) / total * 100) if total > 0 else 0.0

        return {
            'total_processed': total,
            'success_rate': round(success_rate, 1),
            'failed_count': len(self.stats.get('failed_files', [])),
            'confidence_averages': {
                'ocr_confidence': round(avg_ocr_conf, 1),
                'ml_confidence': round(avg_ml_conf, 1),
            },
            'average_times': {
                'ocr': round(avg_ocr, 2),
                'extraction': round(avg_extraction, 2),
                'archive': round(avg_archive, 2),
                'total': round(total_avg, 2),
            },
            'throughput_per_hour': round(3600 / total_avg) if total_avg > 0 else 0,
            'document_type_stats': self.stats.get('document_type_stats', {}),
        }

        
# ... after getting extracted_data from Qwen ...


    def _clean_extracted_data(self, data, ocr_text):
        """Validate and clean the AI-extracted data."""
        if not data:
            return {}
        
        # 1. Validate key fields aren't nonsense
        if 'company_name' in data:
            # If the company name is a long fragment of sentence, it's probably wrong
            if len(data['company_name'].split()) > 6:
                # Try to find a better company name using a simpler pattern
                better_name = self._find_company_name_pattern(ocr_text)
                if better_name:
                    data['company_name'] = better_name
                else:
                    data.pop('company_name', None) # Remove garbage data
        
        # 2. Ensure dates are properly formatted
        if 'date_issued' in data:
            data['date_issued'] = self._standardize_date(data['date_issued'])
        
        # 3. Cross-check extracted amounts with amounts found in text
        if 'total_amount' in data:
            # Check if this amount appears in the text as a number
            if data['total_amount'].replace(',', '').replace('.', '').isdigit():
                # Check if it's in the list of amounts we found via patterns
                all_amounts = self._extract_amounts(ocr_text)
                if data['total_amount'] not in [a['amount'] for a in all_amounts]:
                    # Might be a false positive, lower confidence
                    pass
                    
        return data