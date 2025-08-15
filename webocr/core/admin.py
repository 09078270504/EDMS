# Register your models here.
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from database.models import User  # Import your custom user model
from django.utils.html import format_html
from django.db.models import Count
from django.utils import timezone
from datetime import timedelta
from .models import LoginAttempt, SecurityEvent, UserSession, SuspiciousActivity

# Register your models here.
class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'is_active', 'last_login') 
    list_filter = ('is_active', 'last_login')

admin.site.register(User, CustomUserAdmin)  # Register your user model

@admin.register(SecurityEvent)
class SecurityEventAdmin(admin.ModelAdmin):
    list_display = [
        'timestamp', 'event_type', 'user_display', 'ip_address', 
        'risk_level_badge', 'session_key'
    ]
    list_filter = [
        'event_type', 'risk_level', 'timestamp',
        ('user', admin.RelatedOnlyFieldListFilter)
    ]
    search_fields = ['user__username', 'username_attempted', 'ip_address']
    readonly_fields = ['timestamp', 'session_key', 'extra_data']
    date_hierarchy = 'timestamp'
    
    def user_display(self, obj):
        return obj.user.username if obj.user else obj.username_attempted
    user_display.short_description = 'User'
    
    def risk_level_badge(self, obj):
        colors = {
            'low': 'green',
            'medium': 'orange', 
            'high': 'red',
            'critical': 'darkred'
        }
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            colors.get(obj.risk_level, 'black'),
            obj.get_risk_level_display()
        )
    risk_level_badge.short_description = 'Risk Level'

@admin.register(UserSession)
class UserSessionAdmin(admin.ModelAdmin):
    list_display = [
        'user', 'ip_address', 'login_time', 'last_activity', 
        'session_duration', 'is_active_badge'
    ]
    list_filter = ['is_active', 'login_time']
    search_fields = ['user__username', 'ip_address']
    readonly_fields = ['login_time', 'session_key']
    
    def session_duration(self, obj):
        if obj.is_active:
            duration = timezone.now() - obj.login_time
        else:
            duration = obj.last_activity - obj.login_time
        
        hours = duration.total_seconds() // 3600
        minutes = (duration.total_seconds() % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"
    session_duration.short_description = 'Duration'
    
    def is_active_badge(self, obj):
        if obj.is_active:
            return format_html('<span style="color: green;">●</span> Active')
        else:
            return format_html('<span style="color: red;">●</span> Inactive')
    is_active_badge.short_description = 'Status'

@admin.register(SuspiciousActivity)
class SuspiciousActivityAdmin(admin.ModelAdmin):
    list_display = [
        'timestamp', 'activity_type', 'user', 'ip_address', 
        'is_resolved_badge', 'resolved_by'
    ]
    list_filter = ['activity_type', 'is_resolved', 'timestamp']
    search_fields = ['user__username', 'ip_address', 'description']
    readonly_fields = ['timestamp']
    actions = ['mark_resolved']
    
    def is_resolved_badge(self, obj):
        if obj.is_resolved:
            return format_html('<span style="color: green;">✓</span> Resolved')
        else:
            return format_html('<span style="color: red;">⚠</span> Open')
    is_resolved_badge.short_description = 'Status'
    
    def mark_resolved(self, request, queryset):
        queryset.update(is_resolved=True, resolved_by=request.user)
        self.message_user(request, f"{queryset.count()} activities marked as resolved.")
    mark_resolved.short_description = "Mark selected activities as resolved"

@admin.register(LoginAttempt)
class LoginAttemptAdmin(admin.ModelAdmin):
    list_display = [
        'username', 'ip_address', 'attempt_time', 
        'failures_count', 'lockout_status'
    ]
    list_filter = ['attempt_time']
    search_fields = ['username', 'ip_address']
    readonly_fields = ['attempt_time']
    
    def lockout_status(self, obj):
        if obj.is_locked_out():
            remaining = obj.remaining_lockout_time()
            return format_html(
                '<span style="color: red;">🔒 Locked ({} sec remaining)</span>',
                remaining
            )
        else:
            return format_html('<span style="color: green;">✓ Not Locked</span>')
    lockout_status.short_description = 'Lockout Status'

# Custom admin site title
admin.site.site_header = "WebOCR Security Administration"
admin.site.site_title = "WebOCR Security Admin"
admin.site.index_title = "Security Monitoring Dashboard"