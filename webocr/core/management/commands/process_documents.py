import os
import time
import logging
import json
import re
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

# Import your existing services
from core.services.ocr_processor import OCRProcessor
from core.services.qwen_extractor import QwenExtractor
from core.services.archive_manager import ArchiveManager
from core.services.document_processor import DocumentProcessor
logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Process documents with enhanced parallel PaddleOCR + Qwen (Fixed)'
    
    def add_arguments(self, parser):
        parser.add_argument('--category', type=str, help='Process specific category only')
        parser.add_argument('--list', action='store_true', help='List available categories')
        parser.add_argument('--watch', action='store_true', help='Watch upload folder for new files')
        parser.add_argument('--interval', type=int, default=30, help='Watch interval in seconds (default: 30)')
        parser.add_argument('--workers', type=int, help='Number of parallel workers (default: auto, max 4)')
        parser.add_argument('--stats', action='store_true', help='Show performance statistics')
        parser.add_argument('--retry-failed', action='store_true', help='Clear retry counts and reprocess failed files')
        parser.add_argument('--cpu-only', action='store_true', help='Force CPU-only mode for Qwen')
    
    def handle(self, *args, **options):
        # Set CPU-only mode if requested
        if options.get('cpu_only'):
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            logger.info("Forced CPU-only mode")
        
        try:
            processor = DocumentProcessor()
            
            if options.get('workers'):
                requested_workers = min(options['workers'], 4)  # Cap at 4
                processor.max_workers = requested_workers
                logger.info(f"Using {processor.max_workers} workers (manual override, capped at 4)")
            
            if options.get('list'):
                self.list_categories(processor.archive_manager)
                return
            
            if options.get('stats'):
                self.show_stats(processor)
                return
            
            if options.get('retry_failed'):
                self.retry_failed_files(processor)
                return
            
            if options.get('watch'):
                self.watch_mode(processor, options)
                return
            
            # Process documents
            start_time = timezone.now().astimezone(pytz.timezone('Asia/Manila'))
            
            if options.get('category'):
                category_name = options['category']
                self.stdout.write(f"Processing category: {category_name}")
                results = processor.process_category_parallel(category_name)
                category_results = [{'name': category_name, 'results': results}]
            else:
                self.stdout.write("Processing all categories...")
                category_results = processor.process_all_categories()
            
            self.show_results(category_results, start_time, processor)
            
        except Exception as e:
            logger.error(f"Failed to initialize processor: {e}")
            self.stdout.write(f"ERROR: {e}")
            self.stdout.write("Try using --cpu-only flag if you're having GPU issues")
    
    def list_categories(self, archive_manager):
        """Enhanced category listing"""
        self.stdout.write("Available categories:")
        self.stdout.write("=" * 50)
        categories_data = archive_manager.scan_uploads_categories()
        
        if not categories_data:
            self.stdout.write("No categories found")
            return
        
        total_files = 0
        for category_data in categories_data:
            name = category_data['category_name']
            count = category_data['pdf_count']
            
            # Calculate total size
            total_size = 0
            if 'pdf_files' in category_data:
                for pdf in category_data['pdf_files']:
                    try:
                        total_size += os.path.getsize(pdf['full_path'])
                    except:
                        pass
            
            size_mb = total_size / (1024 * 1024)
            
            self.stdout.write(f"  {name} ({count} PDFs, {size_mb:.1f} MB)")
            total_files += count
        
        self.stdout.write("=" * 50)
        self.stdout.write(f"Total: {total_files} files across {len(categories_data)} categories")
    
    def show_stats(self, processor):
        """Show performance statistics"""
        stats = processor.get_stats_summary()
        if isinstance(stats, str):
            self.stdout.write(stats)
            return
        
        self.stdout.write("Performance Statistics:")
        self.stdout.write("=" * 50)
        self.stdout.write(f"Files processed: {stats['total_processed']}")
        self.stdout.write(f"Success rate: {stats['success_rate']}%")
        self.stdout.write(f"Failed files: {stats['failed_count']}")
        self.stdout.write("")
        self.stdout.write("Average processing times:")
        self.stdout.write(f"  OCR: {stats['average_times']['ocr']}s")
        self.stdout.write(f"  Extraction: {stats['average_times']['extraction']}s")
        self.stdout.write(f"  Archive: {stats['average_times']['archive']}s")
        self.stdout.write(f"  Total: {stats['average_times']['total']}s")
        self.stdout.write("")
        self.stdout.write(f"Throughput: {stats['throughput_per_hour']} files/hour")
        self.stdout.write("=" * 50)
    
    def retry_failed_files(self, processor):
        """Clear retry tracking and reprocess failed files"""
        self.stdout.write("Clearing retry counts and reprocessing failed files...")
        processor.retry_counts.clear()
        
        start_time = timezone.now().astimezone(pytz.timezone('Asia/Manila'))
        category_results = processor.process_all_categories()
        self.show_results(category_results, start_time, processor)
    
    def watch_mode(self, processor, options):
        """Watch mode wrapper with enhanced output"""
        interval = options.get('interval', 30)
        category = options.get('category')
        
        self.stdout.write("ENHANCED WATCH MODE STARTED (FIXED)")
        self.stdout.write("=" * 50)
        self.stdout.write(f"Initial interval: {interval} seconds (adaptive)")
        self.stdout.write(f"Workers: {processor.max_workers} (resource-optimized)")
        self.stdout.write(f"Category: {category}" if category else "Watching all categories")
        self.stdout.write("Features: Shared models, proper error handling, resource optimization")
        self.stdout.write("Press Ctrl+C to stop")
        self.stdout.write("=" * 50)
        
        processor.watch_mode(interval, category)
    
    def show_results(self, category_results, start_time, processor):
        """Enhanced results display"""
        end_time = timezone.now().astimezone(pytz.timezone('Asia/Manila'))
        duration = (end_time - start_time).total_seconds()
        
        self.stdout.write("\n" + "="*60)
        self.stdout.write("PROCESSING COMPLETE (FIXED VERSION)")
        self.stdout.write("="*60)
        
        total_all = {'total': 0, 'successful': 0, 'partial': 0, 'failed': 0, 'skipped': 0}
        
        for category_result in category_results:
            name = category_result['name']
            results = category_result['results']
            
            for key in total_all:
                total_all[key] += results.get(key, 0)
            
            total = results['total']
            successful = results['successful']
            partial = results.get('partial', 0)
            skipped = results.get('skipped', 0)
            
            if total == 0:
                continue
            
            processed_count = successful + partial
            success_rate = (successful / total * 100) if total > 0 else 0
            
            self.stdout.write(f"{name}:")
            self.stdout.write(f"   {processed_count}/{total} processed ({success_rate:.1f}% success)")
            
            if successful > 0:
                self.stdout.write(f"      ✓ {successful} successful")
            if partial > 0:
                self.stdout.write(f"      ~ {partial} partial")
            if skipped > 0:
                upload_path = Path(settings.UPLOAD_FOLDER) / name
                self.stdout.write(f"   ↻ {skipped} skipped (remain in upload)")
            
            # Show category timing if available
            if 'processing_time' in results:
                category_time = results['processing_time']
                avg_time = category_time / total if total > 0 else 0
                self.stdout.write(f"   ⏱ {category_time:.1f}s total ({avg_time:.1f}s/file)")
            
            self.stdout.write("")
        
        # Enhanced summary
        if total_all['total'] > 0:
            processed_total = total_all['successful'] + total_all['partial']
            overall_success = (total_all['successful'] / total_all['total'] * 100)
            avg_time = duration / total_all['total']
            
            self.stdout.write("SUMMARY:")
            self.stdout.write(f"   Total: {total_all['total']} documents")
            self.stdout.write(f"   Successful: {total_all['successful']} ({overall_success:.1f}%)")
            self.stdout.write(f"   Partial: {total_all['partial']}")
            self.stdout.write(f"   Skipped: {total_all['skipped']}")
            self.stdout.write(f"   Duration: {duration:.1f}s ({avg_time:.1f}s per file)")
            self.stdout.write(f"   Workers: {processor.max_workers} (optimized)")
            
            if duration > 60:
                files_per_hour = (total_all['total'] / duration) * 3600
                self.stdout.write(f"   Throughput: {files_per_hour:.0f} files/hour")
            
            # Show performance breakdown
            stats = processor.get_stats_summary()
            if isinstance(stats, dict):
                self.stdout.write("\nPERFORMANCE BREAKDOWN:")
                self.stdout.write(f"   OCR: {stats['average_times']['ocr']}s/file")
                self.stdout.write(f"   Extraction: {stats['average_times']['extraction']}s/file")
                self.stdout.write(f"   Archive: {stats['average_times']['archive']}s/file")
                
                if stats['failed_count'] > 0:
                    self.stdout.write(f"\n⚠ {stats['failed_count']} files failed processing")
                    self.stdout.write("   Use --retry-failed to reprocess them")
                    self.stdout.write("   Use --cpu-only if you have GPU issues")
        
        self.stdout.write("="*60)

        