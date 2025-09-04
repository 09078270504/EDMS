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
from core.services.metadata import QwenExtractor
from core.services.archive_manager import ArchiveManager
from core.services.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Process documents with enhanced parallel PaddleOCR + Qwen (Compatible)'
    
    def add_arguments(self, parser):
        parser.add_argument('--category', type=str, help='Process specific category only')
        parser.add_argument('--list', action='store_true', help='List available categories')
        parser.add_argument('--watch', action='store_true', help='Watch upload folder for new files')
        parser.add_argument('--interval', type=int, default=30, help='Watch interval in seconds (default: 30)')
        parser.add_argument('--workers', type=int, help='Number of parallel workers (default: auto, max 4)')
        parser.add_argument('--stats', action='store_true', help='Show performance statistics')
        parser.add_argument('--retry-failed', action='store_true', help='Clear retry counts and reprocess failed files')
        parser.add_argument('--cpu-only', action='store_true', help='Force CPU-only mode for Qwen')
        parser.add_argument('--ocr', dest='ocr', action='store_true', default=True, help='Enable OCR (default: on)')
        parser.add_argument('--no-ocr', dest='ocr', action='store_false', help='Disable OCR')

    def handle(self, *args, **options):
        use_ocr = bool(options.get('ocr', True))
        
        if options.get('cpu_only'):
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            logger.info("Forced CPU-only mode")
        
        try:
            processor = DocumentProcessor()
            archive_manager = processor.archive_manager
            
            if options.get('workers'):
                requested_workers = min(options['workers'], 4)  # Cap at 4
                processor.max_workers = requested_workers
                logger.info(f"Using {processor.max_workers} workers (manual override, capped at 4)")
            
            if options.get('list'):
                self.list_categories(archive_manager)
                return
            
            if options.get('stats'):
                self.show_stats(processor)
                return
            
            if options.get('retry_failed'):
                self.retry_failed_files(processor)
                return
            
            if options.get('watch'):
                self.watch_mode(processor, archive_manager, options)
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
        self.stdout.write(f"Successful files: {stats['successful']}")
        self.stdout.write(f"Partial files: {stats['partial']}")
        self.stdout.write(f"Failed files: {stats['failed']}")
        self.stdout.write(f"Skipped files: {stats['skipped']}")
        self.stdout.write("")
        self.stdout.write("Average processing times:")
        
        avg_times = stats.get('average_times', {})
        self.stdout.write(f"  OCR: {avg_times.get('ocr', 0):.2f}s")
        self.stdout.write(f"  Extraction: {avg_times.get('extraction', 0):.2f}s")
        self.stdout.write(f"  Archive: {avg_times.get('archive', 0):.2f}s")
        self.stdout.write(f"  Total: {avg_times.get('total', 0):.2f}s")
        self.stdout.write("")
        self.stdout.write(f"Throughput: {stats['throughput_per_hour']} files/hour")
        
        if stats.get('document_types'):
            self.stdout.write("\nDocument types processed:")
            for doc_type, count in stats['document_types'].items():
                self.stdout.write(f"  {doc_type}: {count}")
        
        self.stdout.write("=" * 50)
    
    def retry_failed_files(self, processor):
        """Clear retry tracking and reprocess failed files"""
        self.stdout.write("Clearing retry counts and reprocessing failed files...")
        processor.retry_counts.clear()
        
        start_time = timezone.now().astimezone(pytz.timezone('Asia/Manila'))
        category_results = processor.process_all_categories()
        self.show_results(category_results, start_time, processor)
    
    def watch_mode(self, processor, archive_manager, options):
        """Watch mode for continuous processing"""
        interval = options.get('interval', 30)
        category = options.get('category')
        self.stdout.write("WATCH MODE STARTED")
        self.stdout.write("=" * 50)
        self.stdout.write(f"Checking every {interval} seconds")
        self.stdout.write(f"Watching category: {category}" if category else "Watching all categories")
        self.stdout.write("Press Ctrl+C to stop")
        self.stdout.write("=" * 50)
        scan_count = 0
        
        try:
            while True:
                scan_count += 1
                gmt_plus_8 = pytz.timezone('Asia/Manila')
                current_time = timezone.now().astimezone(gmt_plus_8)
                self.stdout.write(f"\nScan #{scan_count} at {current_time.strftime('%H:%M:%S')}")
                categories_data = archive_manager.scan_uploads_categories()
                new_files_found = False

                for category_data in categories_data:
                    category_name = category_data['category_name']
                    if category and category_name != category:
                        continue

                    if category_data['pdf_files']:
                        new_files_found = True
                        self.stdout.write(f"Processing {len(category_data['pdf_files'])} files in {category_name}")
                        try:
                            start_time = timezone.now().astimezone(gmt_plus_8)
                            results = processor.process_category_parallel(category_name)
                            self.show_results([{
                                'name': category_name, 
                                'results': results
                            }], start_time, processor)
                        except Exception as e:
                            self.stdout.write(f"Error processing {category_name}: {e}")

                if not new_files_found:
                    self.stdout.write("No files found in upload directories")
                    self.stdout.write("Checking partial files for reprocessing...")
                    partial_results = processor.reprocess_partials()
                    if partial_results['reprocessed'] > 0:
                        self.stdout.write(f"Partial reprocessing results:")
                        self.stdout.write(f"   {partial_results['reprocessed']} files reprocessed")
                        self.stdout.write(f"   {partial_results['upgraded_to_successful']} upgraded to successful")
                        self.stdout.write(f"   {partial_results['remained_partial']} remained partial")
                    else:
                        self.stdout.write("No partial files found to reprocess")

                self.stdout.write(f"Sleeping for {interval} seconds...")
                time.sleep(interval)

        except KeyboardInterrupt:
            self.stdout.write("\nWatch mode stopped by user")
            self.stdout.write(f"Completed {scan_count} scans during this session")
    
    def show_results(self, category_results, start_time, processor):
        """Display comprehensive processing results"""
        gmt_plus_8 = pytz.timezone('Asia/Manila')
        end_time = timezone.now().astimezone(gmt_plus_8)
        duration = (end_time - start_time).total_seconds() 
        
        self.stdout.write("\n" + "="*50)
        self.stdout.write("PROCESSING COMPLETE!")
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
            
            self.stdout.write(f"Category: {name}")
            self.stdout.write(f"   {processed_count}/{total} processed ({success_rate:.1f}% success rate)")
            if successful > 0:
                self.stdout.write(f"      {successful} successful")
            if partial > 0:
                self.stdout.write(f"      {partial} partial")
            if skipped > 0:
                upload_path = Path(settings.UPLOAD_FOLDER) / name
                self.stdout.write(f"   {skipped} skipped (remain in upload)")
                self.stdout.write(f"      Files still in: {upload_path}")
            if failed > 0:
                self.stdout.write(f"   {failed} failed")
            self.stdout.write("")
        
        # Show summary for skipped files
        if total_all['skipped'] > 0:
            upload_folder = Path(settings.UPLOAD_FOLDER)
            self.stdout.write(f"{total_all['skipped']} files remain in upload directory for retry: {upload_folder}")
            self.stdout.write("   Run the processor again to retry these files")
            self.stdout.write("")
        
        # Summary
        if total_all['total'] > 0:
            processed_total = total_all['successful'] + total_all['partial']
            overall_success = (total_all['successful'] / total_all['total'] * 100)  # Only successful for success rate
            avg_time = duration / total_all['total']
            
            duration_str = self._format_duration(duration)
            avg_time_str = self._format_duration(avg_time)

            self.stdout.write("SUMMARY:")
            self.stdout.write(f"   Total: {total_all['total']} documents")
            self.stdout.write(f"   Successful: {total_all['successful']} ({overall_success:.1f}%)")
            self.stdout.write(f"   Partial: {total_all['partial']}")
            self.stdout.write(f"   Failed: {total_all['failed']}")
            self.stdout.write(f"   Skipped: {total_all['skipped']}")
            self.stdout.write(f"   Duration: {duration_str} ({avg_time_str} per file)")
            
            if duration > 60:
                files_per_hour = (total_all['total'] / duration) * 3600
                self.stdout.write(f"   Throughput: {files_per_hour:.0f} files/hour")
            
            # Performance evaluation
            if overall_success >= 90:
                self.stdout.write("Excellent results!")
            elif overall_success >= 70:
                self.stdout.write("Good results!")
            else:
                self.stdout.write("Some issues - check failed documents")
            
            # Show processor statistics
            stats = processor.get_stats_summary()
            if isinstance(stats, dict):
                self.stdout.write(f"   Overall session stats: {stats['total_processed']} files processed")
        
        self.stdout.write("="*50)

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