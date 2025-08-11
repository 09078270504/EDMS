#From models app
from .models import Documents
#Core Django shortcut
from django.shortcuts import render, redirect, get_object_or_404
#Database query
from django.db.models import Q
#For Authentication System
from django.contrib.auth import login, logout, get_user_model, update_session_auth_hash
#Authentication Forms
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.forms import PasswordChangeForm
#For Access Control
from django.contrib.auth.decorators import login_required
#For User Interface
from django.contrib import messages
from django import forms
#For Email Functionality
from django.core.mail import send_mail
from django.conf import settings

# User Creation Form
class CustomUserCreationForm(forms.ModelForm):
    email = forms.EmailField(
        label='Email Address',
        widget=forms.EmailInput(attrs={'placeholder': 'Enter your email address'}),
        help_text='Required. Please enter a valid email address.'
    )
    password1 = forms.CharField(
        label='Password',
        widget=forms.PasswordInput,
        help_text='Enter a password.'
    )
    password2 = forms.CharField(
        label='Password confirmation',
        widget=forms.PasswordInput,
        help_text='Enter the same password as before, for verification.'
    )

    class Meta:
        model = get_user_model()
        fields = ('username', 'email')

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if get_user_model().objects.filter(email=email).exists():
            raise forms.ValidationError("A user with this email already exists.")
        return email

    def clean_password2(self):
        password1 = self.cleaned_data.get("password1")
        password2 = self.cleaned_data.get("password2")
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("Passwords don't match")
        return password2

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password1"])
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
        return user

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
    return render(request, 'auth/login.html')

# Login Form
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)

            return redirect('search_form')
        else:
            return render(request, 'auth/login.html', {'form': form})
    else:
        form = AuthenticationForm()

    return render(request, 'auth/login.html', {'form': form})

# Register Form
def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, 'Registration successful! You can now log in.')
            return redirect('login')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CustomUserCreationForm()

    return render(request, 'auth/register.html', {'form': form})

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
    total_documents = Documents.objects.count()
    
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
    documents = None
    search_query = request.GET.get('search_query', '')
    if search_query:
        pass
    return render(request, 'search/search_form.html', {
        'documents': documents,
        'search_query': search_query,
    })

# Documents Search Functionality
@login_required
def search_documents(request):
    search_query = request.GET.get('search_query', '')
    documents = []

    if search_query:
        documents = Documents.objects.filter(
            Q(document_name__icontains=search_query) |
            Q(ocr_text__icontains=search_query)
        )

    return render(request, 'search/search_documents.html', {'documents': documents, 'search_query': search_query})

# Document Search Output
@login_required
def document_detail(request, document_id):
    document = get_object_or_404(Documents, id=document_id)
    return render(request, 'search/document_detail.html', {'document': document})

# Document View 
@login_required
def documents_view(request):
    documents = Documents.objects.all()  # Fetch all documents
    return render(request, 'documents/documents_view.html', {'documents': documents})

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