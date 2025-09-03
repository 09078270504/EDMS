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
from decimal import Decimal, InvalidOperation

# Updated import - using refactored OCR processor
from core.services.ocr_processor import OCRProcessor, cleanup_models
from core.services.qwen_text_extractor import QwenTextExtractor
from core.services.archive_manager import ArchiveManager

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Enhanced document processor with shared model instances for better batch processing"""
    
    def __init__(self):
        # Reduce max workers to prevent resource conflicts
        # Use 2 workers max when using GPU, 4 when CPU-only
        self.max_workers = 2 if torch.cuda.is_available() and not os.environ.get('FORCE_CPU_ONLY') else min(cpu_count(), 4)
        
        # Single shared OCR processor instance
        self.ocr_processor = None
        self.text_extractor = None
        self.archive_manager = ArchiveManager()
        
        # Thread-safe statistics
        self.stats_lock = threading.Lock()
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_time': 0,
            'document_types': {}
        }
        
        # Initialize shared processors
        self._initialize_processors()
        
        logger.info(f"Initialized Enhanced Document Processor with {self.max_workers} workers")
    
    def _initialize_processors(self):
        """Initialize shared processor instances"""
        try:
            # OCR Processor
            device = "cpu" if os.environ.get('FORCE_CPU_ONLY') else "auto"
            model_size = os.environ.get('QWEN_MODEL_SIZE', "2B")
            
            self.ocr_processor = OCRProcessor(
                device=device,
                confidence_threshold=0.7,
                model_size=model_size,
                languages=['en']
            )
            logger.info(f"OCR processor initialized - Model: {model_size}, Device: {device}")
            
            # Text Extractor
            if torch.cuda.is_available() and not os.environ.get('FORCE_CPU_ONLY'):
                text_device = "cuda"
            else:
                text_device = "cpu"
            
            text_model = "Qwen/Qwen2.5-7B-Instruct"
            self.text_extractor = QwenTextExtractor(
                model_name=text_model,
                device=text_device
            )
            logger.info(f"Text extractor initialized - Model: {text_model}, Device: {text_device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize processors: {e}")
            raise
    
    def process_single_file(self, file_path, category_name, filename):
        """Process a single file with shared processors"""
        start_time = time.time()
        
        try:
            # 1) OCR with shared processor
            if not self.ocr_processor:
                raise Exception("OCR processor not initialized")
                
            ocr_result = self.ocr_processor.extract_text_from_file(file_path)
            if not ocr_result.get('success'):
                raise Exception(f"OCR failed: {ocr_result.get('error', 'Unknown error')}")
            
            ocr_text = ocr_result.get('text', '')
            if len(ocr_text.strip()) < 20:
                raise Exception("Insufficient OCR text extracted")
            
            # 2) Document type determination
            document_type = self._determine_document_type(ocr_text, filename)
            
            # 3) LLM extraction for unified JSON
            unified = {}
            ai_confidence = 0.0
            used_method = "qwen_text_extraction"
            
            try:
                if self.text_extractor:
                    unified = self.text_extractor.extract_json(ocr_text, document_type) or {}
                    
                    # Ensure document type is set
                    if not unified.get("doc_type") or unified.get("doc_type") == "general":
                        unified["doc_type"] = document_type
                    
                    ai_confidence = float(unified.get("confidence", 0.0) or 0.0)
                else:
                    raise Exception("Text extractor not available")
                    
            except Exception as e:
                logger.warning(f"Qwen text extraction failed for {filename}: {e}")
                used_method = "qwen_ocr_only"
                
                # Minimal base schema
                unified = self._create_minimal_schema(document_type)
            
            # 4) Calculate processing confidence
            base_conf = float(ocr_result.get('confidence', 0.0) or 0.0) * 100.0
            ai_conf = max(0.0, min(1.0, ai_confidence)) * 100.0
            
            # Count entities
            ee = unified.get("extracted_entities", {}) or {}
            entity_count = sum(len(v) for v in ee.values() if isinstance(v, list))
            bonus = min(entity_count * 3, 30)
            
            processing_confidence = min(base_conf + (ai_conf * 0.5) + bonus, 100.0)
            classification = (
                "successful" if processing_confidence >= 70
                else "partial" if processing_confidence >= 40
                else "failed"
            )
            
            processing_time = time.time() - start_time
            
            # 5) Archive successful/partial results
            if classification in ['successful', 'partial']:
                doc_stem = self.archive_manager.generate_document_name(filename, category_name)
                self.archive_manager.create_archive_structure(
                    category_name=category_name,
                    document_name=doc_stem,
                    pdf_path=file_path,
                    ocr_text=ocr_text,
                    metadata=unified,
                    classification=classification
                )
                self.archive_manager.cleanup_upload_file(file_path)
                
                # Update stats thread-safely
                with self.stats_lock:
                    self.stats['total_processed'] += 1
                    self.stats['successful'] += 1
                    self.stats['total_time'] += processing_time
                    doc_type_key = unified.get("doc_type", document_type)
                    self.stats['document_types'][doc_type_key] = self.stats['document_types'].get(doc_type_key, 0) + 1
                
                logger.info(
                    f"SUCCESS: {filename} ({unified.get('doc_type', document_type)}) - "
                    f"{processing_confidence:.1f}% confidence, {entity_count} entities "
                    f"in {processing_time:.1f}s | method={used_method}"
                )
                
                return {
                    'status': 'completed',
                    'classification': classification,
                    'document_type': unified.get("doc_type", document_type),
                    'confidence': processing_confidence,
                    'processing_time': processing_time,
                    'entities_found': entity_count,
                    'extraction_method': used_method
                }
            else:
                # Update stats for failed
                with self.stats_lock:
                    self.stats['total_processed'] += 1
                    self.stats['failed'] += 1
                
                logger.warning(f"LOW CONFIDENCE: {filename} - {processing_confidence:.1f}% confidence")
                return {'status': 'skipped', 'reason': 'Low confidence'}
                
        except Exception as e:
            # Update stats for error
            with self.stats_lock:
                self.stats['total_processed'] += 1
                self.stats['failed'] += 1
            
            logger.error(f"ERROR: {filename} - {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def _create_minimal_schema(self, document_type):
        """Create minimal schema when LLM extraction fails"""
        return {
            "doc_type": document_type or "unknown",
            "title": None,
            "ids": {
                "document_no": None, "reference_no": None, "po_no": None, "so_no": None,
                "jo_no": None, "invoice_no": None, "voucher_no": None, "ticket_no": None
            },
            "parties": {
                "issuer": {"name": None, "tin": None},
                "recipient": {"name": None, "tin": None},
                "other": []
            },
            "dates": {
                "issue_date": None, "due_date": None,
                "service_period": {"from": None, "to": None},
                "coverage": {"from": None, "to": None}
            },
            "amounts": {
                "currency": None, "subtotal": None, "tax": None,
                "total": None, "other_charges": []
            },
            "line_items": [],
            "payment_terms": None,
            "addresses": {"issuer": None, "recipient": None, "delivery": None, "billing": None},
            "contacts": {"email": None, "phone": None},
            "status": "unknown",
            "tags": [],
            "extracted_entities": {"people": [], "companies": [], "locations": [], "projects": []},
            "notes": [],
            "confidence": 0.0
        }
    
    def _determine_document_type(self, text, filename):
        """Determine document type from text and filename"""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Check filename first
        filename_indicators = {
            'check': ['check', 'cheque', 'chk'],
            'invoice': ['invoice', 'inv', 'bill'],
            'receipt': ['receipt', 'or', 'official'],
            'voucher': ['voucher', 'vp'],
            'purchase_order': ['purchase', 'po'],
            'contract': ['contract', 'agreement'],
            'email': ['email', 'message'],
            'statement': ['statement', 'soa'],
            'itinerary': ['itinerary', 'booking', 'flight', 'travel']
        }
        
        for doc_type, indicators in filename_indicators.items():
            if any(indicator in filename_lower for indicator in indicators):
                return doc_type
        
        # Check text content
        text_indicators = {
            'check': ['pay to the order of', 'account number', 'routing', 'memo'],
            'invoice': ['invoice number', 'bill to', 'amount due', 'vat'],
            'receipt': ['official receipt', 'received from', 'or number'],
            'voucher': ['voucher payable', 'batch number', 'check voucher'],
            'purchase_order': ['purchase order', 'supplier', 'delivery date'],
            'contract': ['terms and conditions', 'agreement', 'whereas'],
            'email': ['from:', 'to:', 'subject:', 'sent:'],
            'statement': ['statement of account', 'balance', 'transaction'],
            'itinerary': ['booking reference', 'flight', 'itinerary', 'passenger', 'boarding']
        }
        
        for doc_type, indicators in text_indicators.items():
            if sum(1 for indicator in indicators if indicator in text_lower) >= 2:
                return doc_type
        
        return 'general'
    
    def process_category(self, category_name):
        """Process a category with optimized parallel execution"""
        logger.info(f"Processing category: {category_name}")
        
        categories_data = self.archive_manager.scan_uploads_categories()
        category_data = next((c for c in categories_data if c['category_name'] == category_name), None)
        
        if not category_data or not category_data.get('pdf_files'):
            return {'total': 0, 'successful': 0, 'failed': 0}
        
        pdf_files = category_data['pdf_files']
        results = {'total': len(pdf_files), 'successful': 0, 'failed': 0}
        
        # Process files with controlled parallelism
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file, pdf['full_path'], category_name, pdf['filename']): pdf
                for pdf in pdf_files
            }
            
            # Collect results with timeout
            for future in concurrent.futures.as_completed(future_to_file, timeout=1800):  # 30 min total timeout
                try:
                    result = future.result(timeout=600)  # 10 min per file timeout
                    
                    if result and result.get('status') == 'completed':
                        results['successful'] += 1
                    else:
                        results['failed'] += 1
                        
                except concurrent.futures.TimeoutError:
                    results['failed'] += 1
                    file_info = future_to_file[future]
                    logger.error(f"Processing timeout for {file_info['filename']}")
                    
                except Exception as e:
                    results['failed'] += 1
                    logger.error(f"Processing failed: {e}")
        
        # Cleanup after batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Category {category_name} completed: {results}")
        return results
    
    def process_all_categories(self):
        """Process all categories"""
        categories_data = self.archive_manager.scan_uploads_categories()
        all_results = []
        
        for category_data in categories_data:
            category_name = category_data['category_name']
            file_count = len(category_data.get('pdf_files', []))
            
            if file_count > 0:
                logger.info(f"Starting category: {category_name} ({file_count} files)")
                results = self.process_category(category_name)
                all_results.append({'name': category_name, 'results': results})
        
        return all_results
    
    def get_stats_summary(self):
        """Get processing statistics"""
        with self.stats_lock:
            if self.stats['total_processed'] == 0:
                return {
                    'total_processed': 0,
                    'success_rate': 0.0,
                    'failed_count': 0,
                    'average_times': {'total': 0.0},
                    'throughput_per_hour': 0,
                    'document_types': {}
                }
            
            success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
            avg_time = self.stats['total_time'] / self.stats['total_processed']
            
            return {
                'total_processed': self.stats['total_processed'],
                'success_rate': round(success_rate, 1),
                'failed_count': self.stats['failed'],
                'average_times': {'total': round(avg_time, 2)},
                'throughput_per_hour': round(3600 / avg_time) if avg_time > 0 else 0,
                'document_types': dict(sorted(self.stats['document_types'].items()))
            }
    
    def get_stats(self):
        """Backward compatibility"""
        return self.get_stats_summary()
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.ocr_processor:
                del self.ocr_processor
                self.ocr_processor = None
            
            if self.text_extractor:
                del self.text_extractor
                self.text_extractor = None
            
            # Cleanup shared models
            cleanup_models()
            
            # GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("DocumentProcessor cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.cleanup()
        except:
            pass


class Command(BaseCommand):
    """Django management command for document processing with optimized Qwen2-VL OCR"""
    
    help = 'Process uploaded documents using optimized Qwen2-VL OCR with batch processing'
    
    def add_arguments(self, parser):
        parser.add_argument('--category', help='Process specific category only')
        parser.add_argument('--workers', type=int, help='Number of worker threads (max 2 for GPU, 4 for CPU)')
        parser.add_argument('--cpu-only', action='store_true', help='Force CPU processing')
        parser.add_argument('--model-size', choices=['2B', '7B'], default='2B', help='Qwen model size')
    
    def handle(self, *args, **options):
        # Set environment variables
        if options['cpu_only']:
            os.environ['FORCE_CPU_ONLY'] = '1'
        
        if options['model_size']:
            os.environ['QWEN_MODEL_SIZE'] = options['model_size']
        
        processor = None
        
        try:
            processor = DocumentProcessor()
            
            # Override workers if specified
            if options['workers']:
                max_allowed = 2 if not options['cpu_only'] else 4
                processor.max_workers = min(options['workers'], max_allowed)
            
            start_time = time.time()
            
            self.stdout.write(f"Using Qwen2-VL OCR with {options.get('model_size', '2B')} model")
            self.stdout.write(f"Processing mode: {'CPU-only' if options['cpu_only'] else 'GPU-accelerated'}")
            self.stdout.write(f"Workers: {processor.max_workers}")
            self.stdout.write(f"Shared model architecture: Enabled")
            self.stdout.write("")
            
            if options['category']:
                results = processor.process_category(options['category'])
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Category '{options['category']}': "
                        f"{results['successful']} successful, {results['failed']} failed"
                    )
                )
            else:
                self.stdout.write("Processing all categories...")
                all_results = processor.process_all_categories()
                
                total_successful = sum(r['results']['successful'] for r in all_results)
                total_failed = sum(r['results']['failed'] for r in all_results)
                
                self.stdout.write("\n" + "="*60)
                self.stdout.write("PROCESSING COMPLETE (FIXED VERSION)")
                self.stdout.write("="*60)
                
                for result in all_results:
                    stats = result['results']
                    success_rate = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
                    self.stdout.write(f"{result['name']}:")
                    self.stdout.write(f"   {stats['successful']}/{stats['total']} processed ({success_rate:.1f}% success)")
                    if stats['successful'] > 0:
                        self.stdout.write(f"      ✓ {stats['successful']} successful")
                
                self.stdout.write(f"\nSUMMARY:")
                self.stdout.write(f"   Total: {total_successful + total_failed} documents")
                self.stdout.write(f"   Successful: {total_successful} ({total_successful/(total_successful+total_failed)*100:.1f}%)")
                self.stdout.write(f"   Partial: 0")
                self.stdout.write(f"   Skipped: 0")
                
                # Performance stats
                stats = processor.get_stats()
                total_time = time.time() - start_time
                
                self.stdout.write(f"   Duration: {total_time:.1f}s ({stats['average_times']['total']:.1f}s per file)")
                self.stdout.write(f"   Workers: {processor.max_workers} (optimized)")
                self.stdout.write(f"   Throughput: {int(3600/stats['average_times']['total']) if stats['average_times']['total'] > 0 else 0} files/hour")
                
                self.stdout.write(f"\nPERFORMANCE BREAKDOWN:")
                
                if stats.get('document_types'):
                    for doc_type, count in stats['document_types'].items():
                        self.stdout.write(f"   {doc_type}: {count} documents")
        
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('\nInterrupted by user'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Failed to initialize processor: {str(e)}'))
            self.stdout.write(f"ERROR: {str(e)}")
            self.stdout.write("Try using --cpu-only flag if you're having GPU issues")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            if processor:
                processor.cleanup()