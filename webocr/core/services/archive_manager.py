import os
import shutil
import json
import logging
import re
from pathlib import Path
from django.conf import settings
from django.utils import timezone
import pytz

class ArchiveManager:
    """Manages the organized archive folder structure"""
    def __init__(self):
        self.upload_folder = Path(settings.UPLOAD_FOLDER)
        self.archive_folder = Path(getattr(settings, 'ARCHIVE_FOLDER', Path.cwd() / "archive"))
        self.logger = logging.getLogger(__name__)
   
    def create_archive_structure(self, category_name, document_name, pdf_path, ocr_text, metadata, classification='successful'):
        """Create archive structure for successfully processed files"""
        if classification == 'partial':
            archive_category_name = f"{{PARTIAL}} - {category_name}"
            self.logger.info(f"📁 Creating PARTIAL archive for {document_name}")
        else:
            archive_category_name = category_name
            self.logger.info(f"📁 Creating SUCCESSFUL archive for {document_name}")
       
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
            self.logger.info(f"Copied original file to: {original_file_path}")
           
            # Save OCR text
            with open(ocr_file_path, 'w', encoding='utf-8') as f:
                f.write(ocr_text)
            self.logger.info(f"Saved OCR text to: {ocr_file_path}")
           
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
            self.logger.info(f"Saved metadata to: {metadata_file_path}")
           
            return {
                'original_file_path': str(original_file_path),
                'ocr_file_path': str(ocr_file_path),
                'metadata_file_path': str(metadata_file_path)
            }
           
        except Exception as e:
            self.logger.error(f"Failed to create archive structure: {e}")
            raise

    def cleanup_upload_file(self, file_path):
        """Remove processed file from upload folder"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.info(f"Cleaned up upload file: {file_path}")
               
                parent_dir = os.path.dirname(file_path)
                # Remove empty parent directory if possible
                try:
                    if not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
                        self.logger.info(f"Cleaned up empty directory: {parent_dir}")
                except OSError:
                    pass
      
        except Exception as e:
            self.logger.error(f"Failed to cleanup upload file {file_path}: {e}")

    def scan_uploads_categories(self):
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
                    self.logger.info(f"Found category '{category_path.name}' with {len(pdf_files)} PDFs")
           
            return categories
           
        except Exception as e:
            self.logger.error(f"Error scanning upload categories: {e}")
            return []
   
    def generate_document_name(self, pdf_filename, category_name):
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