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