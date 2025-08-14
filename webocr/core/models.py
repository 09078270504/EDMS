# core/models.py
from django.db import models
from django.utils import timezone
from datetime import timedelta

class LoginAttempt(models.Model):
    ip_address = models.GenericIPAddressField()
    username = models.CharField(max_length=150, blank=True)
    attempt_time = models.DateTimeField(auto_now_add=True)
    failures_count = models.IntegerField(default=1)
    
    class Meta:
        unique_together = ('ip_address', 'username')
    
    def is_locked_out(self):
        if self.failures_count >= 5:
            lockout_time = self.attempt_time + timedelta(seconds=300)  # 5 minutes
            return timezone.now() < lockout_time
        return False
    
    def remaining_lockout_time(self):
        if self.is_locked_out():
            lockout_time = self.attempt_time + timedelta(seconds=300)
            return int((lockout_time - timezone.now()).total_seconds())
        return 0
    
class DocumentCategory(models.Model):
    """Document categories like 'ACME Corp 2024'"""
    name = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    processed_count = models.IntegerField(default=0)
    
    def __str__(self):
        return self.name

class ProcessedDocument(models.Model):
    """Processed document record"""
    
    # Basic Information
    category = models.ForeignKey(DocumentCategory, on_delete=models.CASCADE)
    original_filename = models.CharField(max_length=255)
    document_name = models.CharField(max_length=255)
    
    # Processing Results
    status = models.CharField(max_length=20, default='completed')
    processed_at = models.DateTimeField(auto_now_add=True)
    
    # Confidence Scores (used by your code)
    ocr_confidence = models.FloatField(default=0.0)
    ml_confidence = models.FloatField(default=0.0)
    
    # Archive Paths (used by your code)
    archive_original_path = models.TextField(blank=True, default='')
    archive_ocr_path = models.TextField(blank=True, default='')
    archive_metadata_path = models.TextField(blank=True, default='')
    
    # Processing Notes
    processing_notes = models.TextField(blank=True, default='')
    
    class Meta:
        ordering = ['-processed_at']
    
    def __str__(self):
        return f"{self.category.name}/{self.document_name}"