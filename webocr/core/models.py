# core/models.py
from django.db import models
from django.utils import timezone
from datetime import timedelta
from django.contrib.auth import get_user_model # For security event 

User = get_user_model()


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

class SecurityEvent(models.Model):
    """Comprehensive security event logging"""
    EVENT_TYPES = [
        ('login_success', 'Login Success'),
        ('login_failure', 'Login Failure'),
        ('logout', 'Logout'),
        ('password_change', 'Password Change'),
        ('password_reset_request', 'Password Reset Request'),
        ('password_reset_complete', 'Password Reset Complete'),
        ('account_locked', 'Account Locked'),
        ('suspicious_activity', 'Suspicious Activity'),
        ('admin_access', 'Admin Access'),
        ('permission_denied', 'Permission Denied'),
    ]
    
    RISK_LEVELS = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]
    
    event_type = models.CharField(max_length=30, choices=EVENT_TYPES)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    username_attempted = models.CharField(max_length=150, blank=True)
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    risk_level = models.CharField(max_length=10, choices=RISK_LEVELS, default='low')
    session_key = models.CharField(max_length=40, blank=True)
    extra_data = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['event_type', '-timestamp']),
            models.Index(fields=['user', '-timestamp']),
            models.Index(fields=['ip_address', '-timestamp']),
            models.Index(fields=['risk_level', '-timestamp']),
        ]
    
    def __str__(self):
        user_info = self.user.username if self.user else self.username_attempted
        return f"{self.get_event_type_display()} - {user_info} ({self.ip_address})"

class UserSession(models.Model):
    """Track active user sessions"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    session_key = models.CharField(max_length=40, unique=True)
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField(blank=True)
    login_time = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.ip_address}"

class SuspiciousActivity(models.Model):
    """Track suspicious activities"""
    ACTIVITY_TYPES = [
        ('multiple_failed_logins', 'Multiple Failed Logins'),
        ('unusual_location', 'Unusual Location'),
        ('multiple_sessions', 'Multiple Active Sessions'),
        ('rapid_requests', 'Rapid Requests'),
        ('privilege_escalation_attempt', 'Privilege Escalation Attempt'),
    ]
    
    activity_type = models.CharField(max_length=30, choices=ACTIVITY_TYPES)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    ip_address = models.GenericIPAddressField()
    description = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    is_resolved = models.BooleanField(default=False)
    resolved_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='resolved_activities')
    
    class Meta:
        verbose_name = "Suspicious Activity"
        verbose_name_plural = "Suspicious Activities"  # ← This fixes the admin display
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.get_activity_type_display()} - {self.ip_address}"

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