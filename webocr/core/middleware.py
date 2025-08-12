# core/middleware.py
from django.shortcuts import redirect
from django.urls import reverse, resolve
from database.models import LoginAttempt

class LockoutMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        current_url = resolve(request.path_info).url_name
        
        # Block access to login if locked out
        if current_url == 'login':
            ip_address = request.META.get('REMOTE_ADDR', '')
            attempt = LoginAttempt.objects.filter(ip_address=ip_address).order_by('-attempt_time').first()
            
            if attempt and attempt.is_locked_out():
                return redirect('account_locked')
        
        response = self.get_response(request)
        return response