#From models app
from database.models import Document
#Core Django shortcut
from django.shortcuts import render, redirect, get_object_or_404
#Database query
from django.db.models import Q
#For Authentication System
from django.contrib.auth import login, logout, get_user_model, update_session_auth_hash, authenticate
#Authentication Forms
from django.contrib.auth.forms import AuthenticationForm, PasswordChangeForm
#For Access Control
from django.contrib.auth.decorators import login_required
#For User Interface
from django.contrib import messages
from django import forms
#For Email Functionality
from django.core.mail import send_mail
from django.conf import settings
#For Log in attempts 
from django.utils import timezone
from .models import LoginAttempt
from datetime import timedelta

from database.models import Document

# Forgot Password Form
class ForgotPasswordForm(forms.Form):
    email = forms.EmailField(
        label='Email Address',
        widget=forms.EmailInput(attrs={
            'placeholder': 'Enter your registered email address',
            'class': 'w-full px-4 py-3 border border-gray-300 rounded-xl'
        }),
        help_text='Enter the email address you used to register your account.'
    )

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if not get_user_model().objects.filter(email=email).exists():
            raise forms.ValidationError("No account found with this email address.")
        return email

def base(request):
    return render(request, 'base.html')

# Login Form
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        ip_address = request.META.get('REMOTE_ADDR', '')
        
        # Check if locked out
        try:
            attempt = LoginAttempt.objects.get(ip_address=ip_address, username=username)
            if attempt.is_locked_out():
                return redirect('account_locked')
        except LoginAttempt.DoesNotExist:
            pass
        
        # Try to authenticate
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            # Success - clear any failed attempts
            LoginAttempt.objects.filter(ip_address=ip_address, username=username).delete()
            login(request, user)
            return redirect('dashboard')  # Change to your success page
        else:
            # Failed login - record attempt
            attempt, created = LoginAttempt.objects.get_or_create(
                ip_address=ip_address,
                username=username,
                defaults={'failures_count': 1}
            )
            if not created:
                attempt.failures_count += 1
                attempt.attempt_time = timezone.now()
                attempt.save()
            
            # Check if now locked out
            if attempt.failures_count >= 5:
                return redirect('account_locked')
            
            messages.error(request, 'Invalid username or password')
    
    return render(request, 'auth/login.html')

def account_locked_view(request):
    ip_address = request.META.get('REMOTE_ADDR', '')
    
    # Get the most recent lockout attempt for this IP
    attempt = LoginAttempt.objects.filter(ip_address=ip_address).order_by('-attempt_time').first()
    
    if attempt and attempt.is_locked_out():
        remaining_seconds = attempt.remaining_lockout_time()
    else:
        # No lockout or expired - redirect to login
        if attempt:
            attempt.delete()
        return redirect('login')
    
    context = {
        'remaining_seconds': remaining_seconds,
        'lockout_expired': remaining_seconds <= 0
    }
    
    return render(request, 'auth/account_locked.html', context)

# Forgot Password Form
def forgot_password_view(request):
    if request.method == 'POST':
        form = ForgotPasswordForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            
            try:
                User = get_user_model()
                user = User.objects.get(email=email)
                
                # User password set to default "comfac123" when reset
                user.set_password("comfac123")
                user.save()
                
                subject = 'Password Reset - Archive System'
                message = f'''
Dear {user.first_name or user.username},

We have successfully processed your password reset request for your Archive System account.

Your password has been reset to: comfac123

For your security, we recommend that you:
1. Log in immediately using this new password
2. Change your password to something more personal and secure after logging in
3. If you did not request this password reset, please contact our support team immediately


Best regards,
Archive System Team

---
This is an automated message. Please do not reply to this email.
                '''
                
                send_mail(
                    subject,
                    message,
                    settings.DEFAULT_FROM_EMAIL,
                    [user.email],
                    fail_silently=False,
                )
                
                messages.success(request, f'Password reset successful! A new password has been sent to {email}.')
                return redirect('login')
                
            except Exception as e:
                messages.error(request, 'An error occurred while sending the email. Please try again later.')
                
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = ForgotPasswordForm()

    return render(request, 'auth/forgot_password.html', {'form': form})

# Dashboard Access
@login_required
def dashboard(request):
    total_documents = Document.objects.count()
    
    stats = {
        'total_documents': total_documents,
    }
    
    context = {
        'user': request.user,
        'stats': stats,
    }
    return render(request, 'auth/dashboard.html', context)

# Logout 
def logout_view(request):
    logout(request)
    return redirect('base')

# Search Form
@login_required
def search_form(request):
    document = None
    search_query = request.GET.get('search_query', '')
    if search_query:
        pass
    return render(request, 'search/search_form.html', {
        'document': document,
        'search_query': search_query,
    })

# Documents Search Functionality
@login_required
def search_documents(request):
    search_query = request.GET.get('search_query', '')
    document = []

    if search_query:
        document = Document.objects.filter(
            Q(document_name__icontains=search_query) |
            Q(ocr_text__icontains=search_query)
        )

    return render(request, 'search/search_documents.html', {'document': document, 'search_query': search_query})

# Document Search Output
@login_required
def document_detail(request, document_id):
    document = get_object_or_404(Document, id=document_id)
    return render(request, 'search/document_detail.html', {'document': document})

# Document View 
@login_required
def documents_view(request):
    document = Document.objects.all()  # Fetch all documents
    return render(request, 'documents/documents_view.html', {'document': document})

# Password Change
def change_password(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)
            messages.success(request, 'Your password has been successfully updated!')
            return redirect('dashboard')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, error)
    else:
        form = PasswordChangeForm(request.user)
    return render(request, 'auth/change_password.html', {'form': form})