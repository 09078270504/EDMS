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
#Handles ML decoding/encoding
import requests
import json
from django.http import JsonResponse


class SearchAPIClient:
    """Client to communicate with ML Search API"""
    
    def __init__(self):
        # Conf ML search API URL
        self.api_base_url = getattr(settings, 'SEARCH_API_URL', 'http://localhost:5000/api')
        self.timeout = 30
    
    def search_documents(self, query, client_filter=None, max_results=20):
        """Call the ML search API"""
        try:
            url = f"{self.api_base_url}/search"
            payload = {
                "query": query,
                "client_filter": client_filter,
                "max_results": max_results
            }
            
            response = requests.post(
                url, 
                json=payload, 
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"API returned status {response.status_code}"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Failed to connect to search API: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Search error: {str(e)}"
            }
    
    def get_document_content(self, document_id, content_type='ocr'):
        """Get document content from ML API"""
        try:
            url = f"{self.api_base_url}/document/{document_id}/content"
            params = {'type': content_type}
            
            response = requests.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"API returned status {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Error getting document content: {str(e)}"
            }

# Initialize the search client
search_client = SearchAPIClient()


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
    """Enhanced search form with ML integration"""
    documents = None
    search_query = request.GET.get('search_query', '')
    
    if search_query:
        # If there's a query, show quick preview (first 3 results)
        print(f"🔍 Quick search preview for: '{search_query}'")
        api_results = search_client.search_documents(
            query=search_query,
            max_results=3
        )
        
        if api_results.get('success'):
            documents = api_results['data']['results'][:3]
        else:
            # Fallback to basic search for preview
            django_docs = Documents.objects.filter(
                Q(document_name__icontains=search_query) |
                Q(ocr_text__icontains=search_query)
            )[:3]
            documents = list(django_docs)
    
    # Get available clients for filter dropdown
    try:
        available_clients = Documents.objects.values_list('client_name', flat=True).distinct()
    except:
        available_clients = []
    
    return render(request, 'search/search_form.html', {
        'documents': documents,
        'search_query': search_query,
        'available_clients': available_clients,
    })

# Documents Search Functionality
@login_required
def search_documents(request):
    """Enhanced search with ML capabilities"""
    search_query = request.GET.get('search_query', '')
    client_filter = request.GET.get('client_filter', '')
    use_ml_search = request.GET.get('smart_search', 'true') == 'true'  # Default to smart search
    
    documents = []
    total_results = 0
    search_type = 'basic'
    ml_results = None
    
    if search_query:
        if use_ml_search:
            # Try ML Search first
            print(f"🔍 Using ML search for: '{search_query}'")
            api_results = search_client.search_documents(
                query=search_query,
                client_filter=client_filter if client_filter else None,
                max_results=50
            )
            
            if api_results.get('success'):
                search_type = 'ml'
                ml_results = api_results['data']['results']
                total_results = api_results['data']['total_results']
                print(f"ML search found {total_results} results")
                
                # Convert ML results to Django-compatible format
                documents = []
                for result in ml_results:
                    # Create a document-like object
                    doc_obj = {
                        'id': result['document_id'],
                        'document_name': result['document_name'],
                        'client_name': result['client_name'],
                        'document_type': result['document_type'],
                        'relevance_score': result['relevance_score'],
                        'match_type': result['match_type'],
                        'preview': result['preview'],
                        'file_paths': result['file_paths'],
                        'is_ml_result': True
                    }
                    documents.append(doc_obj)
            else:
                print(f"ML search failed: {api_results.get('error')}")
                messages.error(request, f"Smart search unavailable. Using basic search.")
                use_ml_search = False
        
        if not use_ml_search or (use_ml_search and not ml_results):
            # Fallback to basic Django search
            print(f"Using basic search for: '{search_query}'")
            search_type = 'basic'
            django_documents = Documents.objects.filter(
                Q(document_name__icontains=search_query) |
                Q(ocr_text__icontains=search_query)
            )
            
            if client_filter:
                django_documents = django_documents.filter(client_name__icontains=client_filter)
            
            documents = list(django_documents)
            total_results = len(documents)
            print(f"Basic search found {total_results} results")

    context = {
        'documents': documents,
        'search_query': search_query,
        'client_filter': client_filter,
        'total_results': total_results,
        'search_type': search_type,
        'use_ml_search': use_ml_search,
        'ml_results': ml_results
    }
    
    return render(request, 'search/search_documents.html', context)

#Check API Status
@login_required
def check_ml_search_status(request):
    """Check if ML Search API is available"""
    try:
        url = f"{search_client.api_base_url}/test"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            return JsonResponse({
                'available': True,
                'status': 'Search is active'
            })
        else:
            return JsonResponse({
                'available': False,
                'status': 'Search unavailable'
            })
            
    except Exception as e:
        return JsonResponse({
            'available': False,
            'status': 'Search disconnected'
        })


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


