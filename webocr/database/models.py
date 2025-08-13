from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone
from datetime import timedelta

# model for user
class User(AbstractUser):
    def __str__(self):
        return self.username

# model for document
class Document(models.Model):
    # document info
    client_name = models.CharField(max_length=255)
    document_name = models.CharField(max_length=255)
    upload_date = models.DateField(auto_now_add=True)

    # flattened file paths (only filenames, not the full path)
    metadata_filename = models.CharField(max_length=255, blank=True, null=True)
    ocr_filename = models.CharField(max_length=255, blank=True, null=True)

    # original file location (full path)
    original_file_path = models.CharField(max_length=500)

    # status
    STATUS_CHOICES = [
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('error', 'Error'),
    ]

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='processing')

    # get the full path for metadata
    def get_metadata_path(self):
        if self.metadata_filename:
            return f"/webocr/metadata/{self.metadata_filename}"
        return None
    
    # get the full path for ocr
    def get_ocr_path(self):
        if self.ocr_filename:
            return f"/webocr/ocr/{self.ocr_filename}"
        return None
    
    # for readability
    def __str__(self):
        return f"{self.client_name} - {self.document_name}"
    
    # for query performance
    class Meta:
        ordering = ['-upload_date']
        indexes = (
            models.Index(fields=['client_name']),
            models.Index(fields=['status']),
            models.Index(fields=['upload_date']),
        )
    