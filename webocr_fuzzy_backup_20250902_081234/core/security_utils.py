# core/security_utils.py
import logging
from django.utils import timezone
from .models import SecurityEvent, UserSession, SuspiciousActivity

# Set up loggers
security_logger = logging.getLogger('security')
auth_logger = logging.getLogger('authentication')

def get_client_ip(request):
    """Get client IP address from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def get_user_agent(request):
    """Get user agent from request"""
    return request.META.get('HTTP_USER_AGENT', '')

def log_security_event(event_type, user=None, username_attempted='', request=None, 
                      extra_data=None, risk_level='low', ip_address=None):
    """Log security events to database and file"""
    
    # Use provided ip_address or get from request
    if ip_address is None:
        ip_address = get_client_ip(request) if request else '127.0.0.1'
    
    user_agent = get_user_agent(request) if request else ''
    
    # Handle session key - it might not exist yet
    session_key = ''
    if request and hasattr(request, 'session'):
        try:
            # Ensure session is created
            if not request.session.session_key:
                request.session.create()
            session_key = request.session.session_key or ''
        except:
            session_key = ''
    
    # Create database record
    security_event = SecurityEvent.objects.create(
        event_type=event_type,
        user=user,
        username_attempted=username_attempted,
        ip_address=ip_address,
        user_agent=user_agent,
        risk_level=risk_level,
        session_key=session_key,
        extra_data=extra_data or {}
    )
    
    # Log to file
    log_message = f"{event_type}: User={user.username if user else username_attempted}, IP={ip_address}"
    
    if risk_level in ['high', 'critical']:
        security_logger.warning(log_message)
    else:
        security_logger.info(log_message)
    
    # Log to auth logger for authentication events
    if event_type in ['login_success', 'login_failure', 'logout']:
        auth_logger.info(log_message)
    
    return security_event

def track_user_session(user, request):
    """Track user session - with proper session handling"""
    ip_address = get_client_ip(request)
    user_agent = get_user_agent(request)
    
    # Ensure session exists and get session key
    session_key = ''
    if hasattr(request, 'session'):
        try:
            # Create session if it doesn't exist
            if not request.session.session_key:
                request.session.create()
            session_key = request.session.session_key
        except:
            # If session creation fails, generate a placeholder
            import uuid
            session_key = str(uuid.uuid4())[:40]
    
    # If we still don't have a session key, skip session tracking
    if not session_key:
        security_logger.warning(f"Could not create session for user {user.username}")
        return None
    
    # End any existing active sessions for this user
    UserSession.objects.filter(user=user, is_active=True).update(is_active=False)
    
    # Create new session record
    try:
        session = UserSession.objects.create(
            user=user,
            session_key=session_key,
            ip_address=ip_address,
            user_agent=user_agent
        )
        return session
    except Exception as e:
        security_logger.error(f"Failed to create UserSession: {e}")
        return None

def detect_suspicious_activity(user=None, ip_address='', activity_type='', description=''):
    """Detect and log suspicious activities"""
    
    # Ensure we have a valid IP address
    if not ip_address:
        ip_address = '127.0.0.1'  # fallback for localhost
    
    suspicious_activity = SuspiciousActivity.objects.create(
        activity_type=activity_type,
        user=user,
        ip_address=ip_address,
        description=description
    )
    
    # Log high-risk event with IP address parameter
    log_security_event(
        event_type='suspicious_activity',
        user=user,
        ip_address=ip_address,  # Pass IP address explicitly
        extra_data={'activity_type': activity_type, 'description': description},
        risk_level='high'
    )
    
    return suspicious_activity

def check_multiple_failed_logins(ip_address, username):
    """Check for multiple failed login attempts"""
    from .models import LoginAttempt
    
    try:
        attempt = LoginAttempt.objects.get(ip_address=ip_address, username=username)
        if attempt.failures_count >= 3:  # Detect at 3 failures, lock at 5
            detect_suspicious_activity(
                ip_address=ip_address,
                activity_type='multiple_failed_logins',
                description=f"Multiple failed login attempts for {username} from {ip_address}"
            )
    except LoginAttempt.DoesNotExist:
        pass