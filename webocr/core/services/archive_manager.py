#services/archive_manager.py - manages the organized archive folder structure
import os
import shutil
import json
import logging
import re
from pathlib import Path
from django.conf import settings
from django.utils import timezone
import pytz

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

class ArchiveManager:
    """Manages the organized archive folder structure"""
    def __init__(self):
        self.upload_folder = Path(settings.UPLOAD_FOLDER)
        self.archive_folder = Path(getattr(settings, 'ARCHIVE_FOLDER', Path.cwd() / "archive"))
        self.logger = logging.getLogger(__name__)
   
    def create_archive_structure(self, category_name, document_name, pdf_path, ocr_text, metadata, classification='successful'):
        """Create archive structure for successfully processed files"""
        archive_start_time = get_ph_time()
        
        if classification == 'partial':
            archive_category_name = f"{{PARTIAL}} - {category_name}"
            self.logger.info(f"[{format_ph_time(archive_start_time)}]: Creating PARTIAL archive for {document_name}")
        else:
            archive_category_name = category_name
            self.logger.info(f"[{format_ph_time(archive_start_time)}]: Creating SUCCESSFUL archive for {document_name}")
       
        # Create folder structure
        archive_folder = self.archive_folder / self._sanitize_folder_name(archive_category_name) / document_name
        original_dir = archive_folder / "original"
        ocr_dir = archive_folder / "ocr"
        metadata_dir = archive_folder / "metadata"
       
        folder_start_time = get_ph_time()
        for directory in [original_dir, ocr_dir, metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        folder_complete_time = get_ph_time()
        folder_duration = (folder_complete_time - folder_start_time).total_seconds()
        self.logger.debug(f"[{format_ph_time(folder_complete_time)}]: Created folder structure in {folder_duration:.3f}s")
       
        # Define file paths
        original_filename = Path(pdf_path).name
        original_file_path = original_dir / original_filename
        ocr_file_path = ocr_dir / f"{document_name}.txt"
        metadata_file_path = metadata_dir / f"{document_name}.json"
       
        try:
            # Copy original PDF
            copy_start_time = get_ph_time()
            shutil.copy2(pdf_path, original_file_path)
            copy_complete_time = get_ph_time()
            copy_duration = (copy_complete_time - copy_start_time).total_seconds()
            self.logger.debug(f"[{format_ph_time(copy_complete_time)}]: Copied original file in {copy_duration:.3f}s to {original_file_path.name}")
           
            # Save OCR text
            ocr_save_start_time = get_ph_time()
            with open(ocr_file_path, 'w', encoding='utf-8') as f:
                f.write(ocr_text)
            ocr_save_complete_time = get_ph_time()
            ocr_save_duration = (ocr_save_complete_time - ocr_save_start_time).total_seconds()
            self.logger.debug(f"[{format_ph_time(ocr_save_complete_time)}]: Saved OCR text in {ocr_save_duration:.3f}s - {len(ocr_text)} chars")
           
            # Save metadata
            metadata_save_start_time = get_ph_time()
            processed_at_ph = get_ph_time()
            metadata_with_info = {
                **metadata,
                "processing_info": {
                    "category": category_name,
                    "classification": classification,
                    "document_name": document_name,
                    "original_filename": original_filename,
                    "processed_at": processed_at_ph.isoformat(),
                    "processed_at_ph_formatted": format_ph_time(processed_at_ph)
                }
            }
           
            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_with_info, f, indent=2, ensure_ascii=False)
            
            metadata_save_complete_time = get_ph_time()
            metadata_save_duration = (metadata_save_complete_time - metadata_save_start_time).total_seconds()
            self.logger.debug(f"[{format_ph_time(metadata_save_complete_time)}]: Saved metadata in {metadata_save_duration:.3f}s")
           
            archive_complete_time = get_ph_time()
            archive_duration = (archive_complete_time - archive_start_time).total_seconds()
            self.logger.info(f"[{format_ph_time(archive_complete_time)}]: Archive structure created in {archive_duration:.2f}s for {document_name}")
            
            return {
                'original_file_path': str(original_file_path),
                'ocr_file_path': str(ocr_file_path),
                'metadata_file_path': str(metadata_file_path)
            }
           
        except Exception as e:
            error_time = get_ph_time()
            self.logger.error(f"[{format_ph_time(error_time)}] Failed to create archive structure: {e}")
            raise

    def cleanup_upload_file(self, file_path):
        """Remove processed file from upload folder"""
        cleanup_start_time = get_ph_time()
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.info(f"[{format_ph_time()}] Cleaned up upload file: {file_path}")
               
                parent_dir = os.path.dirname(file_path)
                # Remove empty parent directory if possible
                try:
                    if not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
                        cleanup_complete_time = get_ph_time()
                        self.logger.info(f"[{format_ph_time(cleanup_complete_time)}] Cleaned up empty directory: {parent_dir}")
                except OSError:
                    pass
      
        except Exception as e:
            error_time = get_ph_time()
            self.logger.error(f"[{format_ph_time(error_time)}] Failed to cleanup upload file {file_path}: {e}")

    def scan_uploads_categories(self):
        """Scan upload folder for categories and PDF files"""
        scan_start_time = get_ph_time()
        self.logger.info(f"[{format_ph_time(scan_start_time)}] 🔍 Scanning upload folder for categories...")
        
        categories = []
        try:
            for category_path in self.upload_folder.iterdir():
                if not category_path.is_dir():
                    continue
                   
                # Find PDF and image files in category
                pdf_files = []
                supported_exts = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
                for file_path in category_path.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in supported_exts:
                        pdf_files.append({
                            'filename': file_path.name,
                            'full_path': str(file_path),
                            'size': file_path.stat().st_size,
                            'is_image': file_path.suffix.lower() != '.pdf'
                        })

                # Only include categories with supported files
                if pdf_files:
                    categories.append({
                        'category_name': category_path.name,
                        'category_path': str(category_path),
                        'pdf_count': len(pdf_files),
                        'pdf_files': pdf_files
                    })
                    found_time = get_ph_time()
                    self.logger.info(f"[{format_ph_time(found_time)}] Found category '{category_path.name}' with {len(pdf_files)} PDFs")
           
            scan_complete_time = get_ph_time()
            scan_duration = (scan_complete_time - scan_start_time).total_seconds()
            self.logger.info(f"[{format_ph_time(scan_complete_time)}] ✅ Scan completed in {scan_duration:.2f}s - Found {len(categories)} categories")
            
            return categories
           
        except Exception as e:
            error_time = get_ph_time()
            self.logger.error(f"[{format_ph_time(error_time)}] Error scanning upload categories: {e}")
            return []
   
    def generate_document_name(self, pdf_filename, category_name):
        """Generate clean document name for folder creation"""
        name_gen_time = get_ph_time()
        base_name = Path(pdf_filename).stem
        clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', base_name)
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')
        result = clean_name.lower() if clean_name else "document"
        
        self.logger.debug(f"[{format_ph_time(name_gen_time)}] Generated document name: '{result}' from '{pdf_filename}'")
        return result
   
    def _sanitize_folder_name(self, folder_name):
        """Sanitize folder name for file system"""
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', folder_name)
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        return sanitized