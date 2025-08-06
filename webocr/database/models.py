from django.db import models
from django.contrib.auth.models import AbstractUser

# Create your models here.
class User(AbstractUser):
    def __str__(self):
        return self.username

class Documents(models.Model):
    # file identification
    filename = models.CharField(max_length=255)
    document_name = models.CharField(max_length=255)

    # status
    STATUS_CHOICES = [
        ('updated', 'Updated'),
        ('processing', 'Processing'),
        ('ready', 'Ready'),
        ('error', 'Error'),
    ]

    status = models.CharField(max_length=15, choices=STATUS_CHOICES, default='updated')

    # file path
    original_path = models.CharField(max_length=500)
    ocr_path = models.CharField(max_length=500)
    metadata_path = models.CharField(max_length=500)
    

    # file properties
    ocr_text = models.TextField(blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_size = models.BigIntegerField()
    file_type = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.document_name}"
    
    class Meta: # para yung order niya from recent to old
        ordering = ['-uploaded_at']

class Document_Metadata(models.Model):
    document = models.OneToOneField(Documents, on_delete=models.CASCADE, related_name = 'metadata')

    # financial data
    amount = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    currency = models.CharField(max_length=10, default='PHP')

    # document details
    date_issued = models.DateField(blank=True, null=True)
    client_name = models.CharField(max_length=255, blank=True)
    vendor_name = models.CharField(max_length=255, blank=True)
    invoice_number = models.CharField(max_length=100, blank=True)

    # Flexible data (for our future customization and others)
    additional_data = models.JSONField(default=dict, blank=True)

    class Meta: # for faster queries
        indexes = [
            models.Index(fields=['amount']),
            models.Index(fields=['currency']),
            models.Index(fields=['date_issued']),
            models.Index(fields=['client_name']),
            models.Index(fields=['vendor_name']),
            models.Index(fields=['invoice_number']),
        ]

    def __str__(self):
        return f"Metadata for {self.document.document_name}"
    
class Search_Log(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    search_query = models.TextField()
    result_count = models.IntegerField()
    response_time_ms = models.IntegerField()
    search_at = models.DateTimeField(auto_now_add=True)

    class Meta: # para yung order niya from recent to old
        ordering = ['-search_at']