# core/middleware.py
from django.utils import timezone
from .models import UserSession  
from .security_utils import log_security_event

class SecurityMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Update last activity for authenticated users
        if request.user.is_authenticated and hasattr(request, 'session'):
            UserSession.objects.filter(
                user=request.user,
                session_key=request.session.session_key,
                is_active=True
            ).update(last_activity=timezone.now())
        
        response = self.get_response(request)
        return response