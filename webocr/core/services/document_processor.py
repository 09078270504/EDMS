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


class DocumentProcessor:
    """Simple, fast, and efficient document processor using standardized Qwen2-VL 7B"""
    
    def __init__(self):
        # Simple worker allocation
        self.max_workers = min(cpu_count() // 2, 3) if torch.cuda.is_available() else min(cpu_count(), 4)
        self.batch_size = 6 if torch.cuda.is_available() else 8
        
        # Use standardized Qwen2-VL 7B for both OCR and text extraction
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_size = "7B"  # Standardized model size
        
        # Initialize services with standardized model
        self.archive_manager = ArchiveManager()
        self._ocr_processor = None
        self._text_extractor = None
        self._processor_lock = threading.Lock()
        self.retry_counts = {}
        
        # Enhanced statistics tracking
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'partial': 0,
            'failed': 0,
            'skipped': 0,
            'failed_count': 0,  # For backward compatibility
            'total_time': 0.0,
            'processing_times': [],
            'ocr_times': [],
            'extraction_times': [],
            'archive_times': [],
            'document_types': defaultdict(int),
            'average_times': {
                'ocr': 0.0,
                'extraction': 0.0,
                'archive': 0.0,
                'total': 0.0
            }
        }
        self.stats_lock = threading.Lock()
        
        logger.info(f"Document Processor initialized - Workers: {self.max_workers}, Batch: {self.batch_size}, Model: Qwen2-VL {self.model_size}")
    
    @property
    def ocr_processor(self):
        """Lazy-loaded OCR processor with standardized Qwen2-VL 7B"""
        if self._ocr_processor is None:
            with self._processor_lock:
                if self._ocr_processor is None:
                    self._ocr_processor = OCRProcessor(
                        device="auto",
                        confidence_threshold=0.6,
                        model_size=self.model_size,  # Use standardized 7B
                        languages=['en']
                    )
                    logger.info(f"OCR processor loaded with Qwen2-VL {self.model_size}")
        return self._ocr_processor
    
    @property  
    def text_extractor(self):
        """Lazy-loaded text extractor with standardized Qwen2-VL 7B"""
        if self._text_extractor is None:
            with self._processor_lock:
                if self._text_extractor is None:
                    try:
                        # Always use standardized 7B model
                        self._text_extractor = QwenExtractor(
                            model_size=self.model_size,  # Use standardized 7B
                            device=self.device
                        )
                        logger.info(f"Text extractor loaded with Qwen2-VL {self.model_size} on {self.device}")
                    except Exception as e:
                        logger.warning(f"Failed to load text extractor: {e}")
                        self._text_extractor = None
        return self._text_extractor
    
    def extract_text_from_pdf(self, file_path: str) -> tuple:
        """Extract text from PDF for backward compatibility"""
        try:
            result = self.ocr_processor.extract_text_from_file(file_path)
            if result.get('success'):
                text = result.get('text', '')
                confidence = float(result.get('confidence', 0.0)) * 100
                return text, confidence
            else:
                return "", 0.0
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return "", 0.0
    
    def extract_structured_data(self, text: str, doc_type: str = None) -> dict:
        """Extract structured data for backward compatibility"""
        try:
            if self.text_extractor and text:
                clean_text = self._preprocess_text(text)
                result = self.text_extractor.extract_json(clean_text, doc_type) or {}
                
                # Add confidence score
                confidence = self._calculate_extraction_confidence(result, text)
                result['processing_confidence'] = confidence
                result['confidence'] = confidence
                
                return result
            else:
                return self._create_fallback_metadata(doc_type or 'general', text)
        except Exception as e:
            logger.warning(f"Structured extraction failed: {e}")
            return self._create_fallback_metadata(doc_type or 'general', text)

    def _calculate_extraction_confidence(self, file_path: str, unified: dict, text: str) -> float:
        """More sophisticated confidence calculation"""
        ocr_result = self.ocr_processor.extract_text_from_file(file_path)
        ocr_text = ocr_result.get('text', '').strip()
        base_conf = float(ocr_result.get('confidence', 0.0)) * 100
        
        # Text quality metrics
        text_length_score = min(len(ocr_text) / 500, 1.0) * 15
        text_structure_score = len(re.findall(r'\n', ocr_text)) / max(len(ocr_text.split('\n')), 1) * 10
        
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
            if self._has_nested_value(unified, field):
                completeness_score += weight
        
        # Document type specific bonuses
        doc_type_bonus = self._get_document_type_bonus(unified, ocr_text)
        
        total_confidence = (
            base_conf * 0.3 +
            text_length_score +
            text_structure_score +
            completeness_score +
            doc_type_bonus
        )
        
        return min(total_confidence, 100.0)
    
    def process_single_file(self, file_path: str, category_name: str, filename: str) -> Dict:
        """Process single file with standardized Qwen2-VL 7B pipeline"""
        start_time = time.time()
        
        try:
            # Basic validation
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 100:  # Skip very large files
                raise ValueError(f"File too large: {file_size_mb:.1f}MB")
            
            # OCR Processing with standardized model
            ocr_start = time.time()
            ocr_result = self.ocr_processor.extract_text_from_file(file_path)
            ocr_time = time.time() - ocr_start
            
            if not ocr_result.get('success'):
                raise Exception(f"OCR failed: {ocr_result.get('error', 'Unknown error')}")
            
            ocr_text = ocr_result.get('text', '').strip()
            if len(ocr_text) < 20:
                raise Exception("Insufficient OCR text extracted")
            
            # Document type detection
            doc_type = None
            
            # Metadata extraction with standardized model
            extraction_start = time.time()
            unified = {}
            extraction_method = "none"
            
            if self.text_extractor:
                try:
                    # Preprocess text for better extraction
                    clean_text = self._preprocess_text(ocr_text)
                    unified = self.text_extractor.extract_json(clean_text, doc_type) or {}
                    extraction_method = f"enhanced_qwen2vl_{self.model_size}"

                    # Get the detected document type
                    detected_doc_type = unified.get('doc_type', 'general')
                    logger.info(f"Document type detected: {detected_doc_type} for {filename}")
        
                    
                except Exception as e:
                    logger.warning(f"Enhanced extraction failed for {filename}: {e}")
                    unified = self._create_fallback_metadata('general', ocr_text)
                    extraction_method = "fallback"
                    detected_doc_type = 'general'
            else:
                unified = self._create_fallback_metadata('general', ocr_text)
                extraction_method = "no_extractor"
                detected_doc_type = 'general'
            
            extraction_time = time.time() - extraction_start
            doc_type = detected_doc_type
            
            # Calculate confidence and classify
            confidence = self._calculate_confidence(ocr_result, unified, ocr_text, extraction_method)
            classification = self._classify_result(confidence, unified, ocr_text)
            
            # Archive processing
            archive_start = time.time()
            processing_time = time.time() - start_time
            
            # Archive if successful or partial
            if classification in ['successful', 'partial']:
                doc_stem = self.archive_manager.generate_document_name(filename, category_name)
                
                # Add processing info with model information
                unified['processing_info'] = {
                    'category': category_name,
                    'classification': classification,
                    'document_name': doc_stem,
                    'original_filename': filename,
                    'processing_time': processing_time,
                    'extraction_method': extraction_method,
                    'model_used': f'Qwen2-VL-{self.model_size}',
                    'confidence_score': confidence,
                    'processed_at': timezone.now().astimezone(pytz.timezone('Asia/Manila')).isoformat()
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
                
                # Update stats
                with self.stats_lock:
                    self.stats['total_processed'] += 1
                    if classification == 'successful':
                        self.stats['successful'] += 1
                    else:
                        self.stats['partial'] += 1
                    self.stats['total_time'] += processing_time
                    self.stats['processing_times'].append(processing_time)
                    self.stats['ocr_times'].append(ocr_time)
                    self.stats['extraction_times'].append(extraction_time)
                    self.stats['archive_times'].append(archive_time)
                    self.stats['document_types'][doc_type] += 1
                    
                    # Update averages
                    self._update_average_times()
                
                logger.info(f"SUCCESS: {filename} ({doc_type}) - {confidence:.1f}% confidence in {processing_time:.1f}s [Qwen2-VL {self.model_size}]")
                
                return {
                    'status': 'completed',
                    'classification': classification,
                    'confidence': confidence,
                    'processing_time': processing_time,
                    'extraction_method': extraction_method,
                    'model_used': f'Qwen2-VL-{self.model_size}'
                }
            else:
                # Update stats for skipped
                with self.stats_lock:
                    self.stats['total_processed'] += 1
                    self.stats['skipped'] += 1
                    self.stats['processing_times'].append(processing_time)
                    self._update_average_times()
                
                logger.warning(f"SKIPPED: {filename} - {confidence:.1f}% confidence too low")
                return {'status': 'skipped', 'reason': f'Low confidence: {confidence:.1f}%'}
        
        except Exception as e:
            processing_time = time.time() - start_time
            
            with self.stats_lock:
                self.stats['total_processed'] += 1
                self.stats['failed'] += 1
                self.stats['failed_count'] += 1  # For backward compatibility
                self.stats['processing_times'].append(processing_time)
                self._update_average_times()
                
            logger.error(f"ERROR: {filename} - {str(e)}")
            return {'status': 'failed', 'error': str(e), 'processing_time': processing_time}
    
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
    
    def _detect_document_type(self, text: str, filename: str) -> str:
        """Enhanced document type detection with confidence"""
        
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Weighted scoring system
        type_scores = defaultdict(float)
        
        # Filename patterns (high confidence)
        filename_patterns = {
            'official_receipt': (['receipt', 'or'], 0.8),
            'invoice': (['invoice', 'bill'], 0.8),
            'check': (['check', 'cheque'], 0.9),
            'voucher': (['voucher'], 0.7),
        }
        
        for doc_type, (patterns, weight) in filename_patterns.items():
            if any(pattern in filename_lower for pattern in patterns):
                type_scores[doc_type] += weight
        
        # Content patterns (medium confidence)
        content_patterns = {
            'official_receipt': ([
                'official receipt', 'received from', 'or number',
                'bir form', 'receipt number'
            ], 0.6),
            'invoice': ([
                'invoice number', 'bill to', 'amount due',
                'invoice date', 'payment terms'
            ], 0.6),
        }
        
        for doc_type, (patterns, weight) in content_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in text_lower)
            type_scores[doc_type] += (matches / len(patterns)) * weight
        
        if type_scores:
            best_type = max(type_scores.keys(), key=lambda k: type_scores[k])
            confidence = min(type_scores[best_type], 1.0)
            return best_type, confidence
        
        return 'general', 0.3
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and optimize text for extraction"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = []
        for line in text.split('\n'):
            line = re.sub(r'\s+', ' ', line.strip())
            if line:
                lines.append(line)
        
        # Limit length but keep structure
        result = '\n'.join(lines)
        if len(result) > 6000:
            result = result[:6000] + "..."
        
        return result
    
    def _enhance_extraction(self, unified: dict, ocr_text: str, doc_type: str) -> dict:
        """Enhance extraction with fallback information"""
        if not unified:
            return self._create_fallback_metadata(doc_type, ocr_text)
        
        enhanced = unified.copy()
        
        # Ensure doc_type is correct
        enhanced['doc_type'] = doc_type
        
        # Extract missing key information using patterns
        if not enhanced.get('amounts', {}).get('total'):
            amounts = self._extract_amounts(ocr_text)
            if amounts:
                if 'amounts' not in enhanced:
                    enhanced['amounts'] = {}
                enhanced['amounts'].update(amounts)
        
        if not enhanced.get('dates', {}).get('issue_date'):
            dates = self._extract_dates(ocr_text)
            if dates:
                if 'dates' not in enhanced:
                    enhanced['dates'] = {}
                enhanced['dates'].update(dates)
        
        if not any(v for v in enhanced.get('ids', {}).values() if v):
            ids = self._extract_ids(ocr_text, doc_type)
            if ids:
                if 'ids' not in enhanced:
                    enhanced['ids'] = {}
                enhanced['ids'].update(ids)
        
        return enhanced
    
    def reprocess_partials(self) -> dict:
        """Reprocess partial files to try reaching higher confidence with standardized model"""
        logger.info("Scanning for partial files to reprocess with Qwen2-VL 7B...")
        results = {'reprocessed': 0, 'upgraded_to_successful': 0, 'remained_partial': 0}
        partial_folders = []
        
        # Find all partial archive folders
        for folder in self.archive_manager.archive_folder.iterdir():
            if folder.is_dir() and folder.name.startswith("{PARTIAL}"):
                partial_folders.append(folder)
        
        if not partial_folders:
            logger.info("No partial files found")
            return results
        
        logger.info(f"Found {len(partial_folders)} partial categories to check")
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
                logger.info(f"Reprocessing partial with Qwen2-VL 7B: {doc_folder.name}")
                
                try:
                    # Re-run OCR and extraction with standardized model
                    ocr_text, ocr_conf = self.extract_text_from_pdf(str(pdf_path))
                    structured_data = self.extract_structured_data(ocr_text, doc_folder.name)
                    ml_conf = structured_data.get('processing_confidence', structured_data.get('confidence', 50))
                    
                    results['reprocessed'] += 1
                    
                    if ocr_conf >= 90 and ml_conf >= 90:
                        logger.info(f"Upgraded to successful: {doc_folder.name} (OCR: {ocr_conf:.1f}%, ML: {ml_conf:.1f}%)")
                        
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
                            metadata['processing_info']['reprocessed_at'] = timezone.now().astimezone(pytz.timezone('Asia/Manila')).isoformat()
                            metadata['processing_info']['model_used'] = f'Qwen2-VL-{self.model_size}'
                            metadata['processing_confidence'] = ml_conf
                            
                            with open(metadata_file, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=2, ensure_ascii=False)
                        
                        results['upgraded_to_successful'] += 1
                    else:
                        logger.info(f"Still partial: {doc_folder.name} (OCR: {ocr_conf:.1f}%, ML: {ml_conf:.1f}%)")
                        results['remained_partial'] += 1
                        
                except Exception as e:
                    logger.error(f"Error reprocessing {doc_folder.name}: {e}")
                    results['remained_partial'] += 1
            
            # Clean up empty partial category folder
            try:
                if not any(partial_folder.iterdir()):
                    partial_folder.rmdir()
                    logger.info(f"Removed empty partial folder: {partial_folder.name}")
            except:
                pass
        
        logger.info(f"Partial reprocessing complete with Qwen2-VL 7B: {results}")
        return results

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
        elif 'usd' in text.lower():  # Fixed the broken condition here
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
    
    def _calculate_confidence(self, ocr_result: dict, unified: dict, ocr_text: str, method: str) -> float:
        """Calculate processing confidence"""
        base_conf = float(ocr_result.get('confidence', 0.0) or 0.0) * 100
        
        # Text quality
        text_score = min(len(ocr_text) / 300, 1.0) * 20
        
        # Data completeness
        completeness_score = 0
        if unified.get('amounts', {}).get('total'):
            completeness_score += 15
        if unified.get('dates', {}).get('issue_date'):
            completeness_score += 10
        if any(v for v in unified.get('ids', {}).values() if v):
            completeness_score += 10
        
        # Method bonus - higher for standardized Qwen2-VL 7B
        method_bonus = {f'qwen2vl_{self.model_size}_extraction': 15, 'fallback': 5, 'no_extractor': 0}.get(method, 0)
        
        # Structure indicators
        indicators = ['total', 'amount', 'date', 'number', 'name']
        structure_score = sum(5 for ind in indicators if ind.lower() in ocr_text.lower())
        
        total_confidence = base_conf * 0.4 + text_score + completeness_score + method_bonus + structure_score
        return min(total_confidence, 100.0)
    
    def _classify_result(self, confidence: float, unified: dict, ocr_text: str) -> str:
        """Classify processing result"""
        has_key_data = bool(
            unified.get('dates', {}).get('issue_date') or
            unified.get('amounts', {}).get('total') or
            any(v for v in unified.get('ids', {}).values() if v)
        )
        
        if confidence >= 75 and has_key_data:
            return "successful"
        elif confidence >= 50 and (has_key_data or len(ocr_text) > 100):
            return "partial"
        else:
            return "failed"
    
    def process_category(self, category_name: str) -> dict:
        """Process category with simple batching using standardized Qwen2-VL 7B"""
        logger.info(f"Processing category: {category_name} with Qwen2-VL {self.model_size}")
    
        # Adaptive batch size based on available memory
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            memory_per_doc = 500 * 1024 * 1024  # ~500MB per document
            max_batch_from_memory = int(free_memory / memory_per_doc)
            adaptive_batch_size = min(self.batch_size, max_batch_from_memory, 4)
        else:
            adaptive_batch_size = min(self.batch_size, 2)  # Conservative for CPU
        
        logger.info(f"Using adaptive batch size: {adaptive_batch_size}")


        # Get files
        categories_data = self.archive_manager.scan_uploads_categories()
        category_data = next((c for c in categories_data if c['category_name'] == category_name), None)
        
        if not category_data or not category_data.get('pdf_files'):
            return {'total': 0, 'successful': 0, 'partial': 0, 'failed': 0, 'skipped': 0}
        
        pdf_files = category_data['pdf_files']
        results = {'total': len(pdf_files), 'successful': 0, 'partial': 0, 'failed': 0, 'skipped': 0}
        
        # Process in batches
        batches = [pdf_files[i:i + self.batch_size] for i in range(0, len(pdf_files), self.batch_size)]
        
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} files) with Qwen2-VL {self.model_size}")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.process_single_file, pdf['full_path'], category_name, pdf['filename']): pdf
                    for pdf in batch
                }
                
                for future in concurrent.futures.as_completed(futures, timeout=1200):  # 20 min timeout
                    try:
                        result = future.result(timeout=180)  # 3 min per file
                        
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
                        logger.error(f"Processing error: {e}")
            
            # Simple cleanup between batches
            if batch_idx < len(batches) - 1:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        logger.info(f"Category {category_name} completed with Qwen2-VL {self.model_size}: {results}")
        return results
    
    def process_all_categories(self) -> List[dict]:
        """Process all categories with standardized Qwen2-VL 7B"""
        categories_data = self.archive_manager.scan_uploads_categories()
        all_results = []
        
        logger.info(f"Processing all categories with Qwen2-VL {self.model_size}")
        
        for category_data in categories_data:
            category_name = category_data['category_name']
            file_count = len(category_data.get('pdf_files', []))
            
            if file_count > 0:
                logger.info(f"Processing category: {category_name} ({file_count} files)")
                results = self.process_category(category_name)
                all_results.append({'name': category_name, 'results': results})
        
        return all_results
    
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
        try:
            # Clean up OCR processor
            if self._ocr_processor and hasattr(self._ocr_processor, 'cleanup'):
                self._ocr_processor.cleanup()
                self._ocr_processor = None
            
            # Clean up text extractor 
            if self._text_extractor and hasattr(self._text_extractor, 'cleanup'):
                self._text_extractor.cleanup()
                self._text_extractor = None
            
            # Clean up shared model managers
            cleanup_models()  # OCR models
            cleanup_qwen_models()  # Text extraction models
            
            # General cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"DocumentProcessor cleanup completed (Qwen2-VL {self.model_size})")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    # Backward compatibility methods
    def process_category_parallel(self, category_name: str) -> dict:
        """Backward compatibility method"""
        return self.process_category(category_name)
    
    def get_stats(self) -> dict:
        """Backward compatibility method"""
        return self.get_stats_summary()
    
    def cleanup_resources(self):
        """Backward compatibility method"""
        self.cleanup()



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

    def process_with_fallback_strategies(self, file_path: str, category_name: str, filename: str) -> Dict:
        """Process with multiple fallback strategies"""
        
        strategies = [
            ('primary', self._process_primary_strategy),
            ('reduced_batch', self._process_reduced_strategy),
            ('cpu_fallback', self._process_cpu_fallback),
            ('minimal', self._process_minimal_strategy)
        ]
        
        last_error = None
        
        for strategy_name, strategy_func in strategies:
            try:
                logger.info(f"Attempting {strategy_name} strategy for {filename}")
                result = strategy_func(file_path, category_name, filename)
                
                if result.get('status') in ['completed', 'partial']:
                    logger.info(f"Success with {strategy_name} strategy")
                    return result
                    
            except Exception as e:
                logger.warning(f"{strategy_name} strategy failed: {e}")
                last_error = e
                
                # Cleanup between strategies
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # All strategies failed
        return {
            'status': 'failed', 
            'error': f'All strategies failed. Last error: {last_error}',
            'strategies_attempted': [s[0] for s in strategies]
        }

    def format_extraction_output(self, unified: dict, ocr_text: str, processing_info: dict) -> dict:
        """Format output to match expected structure"""
        
        return {
            "document_type": unified.get('doc_type', 'unknown'),
            "confidence": unified.get('confidence', 0.0),
            "entities": {
                "people": self._extract_people_entities(unified, ocr_text),
                "companies": self._extract_company_entities(unified, ocr_text),
                "locations": self._extract_location_entities(unified, ocr_text)
            },
            "key_information": self._extract_key_information(unified),
            "dates_found": self._extract_all_dates(unified, ocr_text),
            "amounts_found": self._extract_all_amounts(unified, ocr_text),
            "processing_confidence": processing_info.get('confidence', 0.0),
            "extraction_method": f"qwen2vl_{self.model_size}",
            "processing_info": processing_info
        }

class MemoryManager:
    """Memory management for large batch processing"""
    
    def __init__(self):
        self.memory_threshold = 0.8  # 80% memory usage threshold
        
    def check_memory_usage(self):
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            return memory_used > self.memory_threshold
        return False
    
    def cleanup_if_needed(self):
        if self.check_memory_usage():
            torch.cuda.empty_cache()
            gc.collect()
            return True
        return False