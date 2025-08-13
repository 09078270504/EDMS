#From models app
from database.models import Document  # Fixed: Use consistent model name
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
#For SEARCH API
import json
import os
import re
from django.http import JsonResponse
#For Log in attempts 
from django.utils import timezone
from .models import LoginAttempt
from datetime import timedelta

# User Creation Form (aint sure)
"""class CustomUserCreationForm(forms.ModelForm):
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
        return user """

# ======TWO-STAGE SEARCH ENGINE======
class TwoStageSearchEngine:
    """
    Two-stage search system:
    Stage 1: Search metadata files (fast)
    Stage 2: Search full OCR text files (deep)
    """
    
    def __init__(self):
        self.stage_1_results = []
        self.stage_2_results = []
    
    def stage_1_search(self, query, client_filter=None):
        """
        Stage 1: Search only metadata files
        Returns: List of matching documents with metadata preview
        """
        print(f"🔍 Stage 1: Searching metadata for '{query}'")
        
        # Get all documents from database with file paths
        documents = Document.objects.all()  # Fixed: Use Document instead of Documents
        
        if client_filter:
            documents = documents.filter(client_name__icontains=client_filter)
        
        stage_1_matches = []
        search_terms = query.lower().split()
        
        for doc in documents:
            # Get metadata file path from database
            metadata_path = self._get_metadata_path(doc)
            
            if not metadata_path or not os.path.exists(metadata_path):
                continue
            
            try:
                # Load and search metadata
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Search through metadata fields
                match_score, matched_fields = self._search_metadata(metadata, search_terms)
                
                if match_score > 0:
                    result = {
                        'document_id': doc.id,
                        'document_name': doc.document_name,
                        'client_name': getattr(doc, 'client_name', 'Unknown'),
                        'document_type': metadata.get('document_type', 'Unknown'),
                        'match_score': match_score,
                        'matched_fields': matched_fields,
                        'metadata_preview': self._create_metadata_preview(metadata),
                        'file_paths': {
                            'metadata': metadata_path,
                            'ocr': self._get_ocr_path(doc),
                            'original': self._get_original_path(doc)
                        },
                        'search_stage': 1
                    }
                    stage_1_matches.append(result)
                    
            except Exception as e:
                print(f"Error reading metadata for {doc.document_name}: {e}")
                continue
        
        # Sort by match score
        stage_1_matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        print(f"Stage 1 found {len(stage_1_matches)} results")
        return stage_1_matches
    
    def stage_2_search(self, query, documents_to_search, client_filter=None):
        """
        Stage 2: Search full OCR text files for deeper results
        documents_to_search: List of document IDs to search, or 'all'
        """
        print(f"Stage 2: Deep search in OCR text for '{query}'")
        
        if documents_to_search == 'all':
            documents = Document.objects.all()  # Fixed: Use Document
            if client_filter:
                documents = documents.filter(client_name__icontains=client_filter)
        else:
            documents = Document.objects.filter(id__in=documents_to_search)  # Fixed: Use Document
        
        stage_2_matches = []
        search_terms = query.lower().split()
        
        for doc in documents:
            ocr_path = self._get_ocr_path(doc)
            
            if not ocr_path or not os.path.exists(ocr_path):
                continue
            
            try:
                # Load and search full OCR text
                with open(ocr_path, 'r', encoding='utf-8') as f:
                    ocr_text = f.read()
                
                # Search through OCR text
                match_score, matched_snippets = self._search_ocr_text(ocr_text, search_terms)
                
                if match_score > 0:
                    # Also get metadata for preview
                    metadata_path = self._get_metadata_path(doc)
                    metadata = {}
                    if metadata_path and os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                        except:
                            pass
                    
                    result = {
                        'document_id': doc.id,
                        'document_name': doc.document_name,
                        'client_name': getattr(doc, 'client_name', 'Unknown'),
                        'document_type': metadata.get('document_type', 'Unknown'),
                        'match_score': match_score,
                        'matched_snippets': matched_snippets,
                        'metadata_preview': self._create_metadata_preview(metadata),
                        'file_paths': {
                            'metadata': metadata_path,
                            'ocr': ocr_path,
                            'original': self._get_original_path(doc)
                        },
                        'search_stage': 2
                    }
                    stage_2_matches.append(result)
                    
            except Exception as e:
                print(f"Error reading OCR text for {doc.document_name}: {e}")
                continue
        
        # Sort by match score
        stage_2_matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        print(f"Stage 2 found {len(stage_2_matches)} results")
        return stage_2_matches
    
    def _get_metadata_path(self, doc):
        """Get metadata file path from document record"""
        # Check if your model has a metadata_path field
        if hasattr(doc, 'metadata_path') and doc.metadata_path:
            return doc.metadata_path
        
        # Fallback: construct path based on naming convention
        base_name = os.path.splitext(doc.document_name)[0]
        client_name = getattr(doc, 'client_name', 'unknown')
        return f"archive/{client_name}/{base_name}/metadata/{base_name}.json"
    
    def _get_ocr_path(self, doc):
        """Get OCR text file path from document record"""
        if hasattr(doc, 'ocr_path') and doc.ocr_path:
            return doc.ocr_path
        
        # Fallback: construct path
        base_name = os.path.splitext(doc.document_name)[0]
        client_name = getattr(doc, 'client_name', 'unknown')
        return f"archive/{client_name}/{base_name}/ocr/{base_name}.txt"
    
    def _get_original_path(self, doc):
        """Get original PDF file path"""
        if hasattr(doc, 'original_path') and doc.original_path:
            return doc.original_path
        
        # Fallback: construct path
        base_name = os.path.splitext(doc.document_name)[0]
        client_name = getattr(doc, 'client_name', 'unknown')
        return f"archive/{client_name}/{base_name}/original/{base_name}.pdf"
    
    def _search_metadata(self, metadata, search_terms):
        """Search through metadata fields and return score + matched fields"""
        matched_fields = []
        total_score = 0
        
        # Define field weights (higher = more important)
        field_weights = {
            'names': 3,
            'invoice_numbers': 4,
            'financial': 2,
            'emails': 2,
            'phones': 2,
            'document_type': 3,
            'keywords': 1,
            'addresses': 1
        }
        
        # Search each metadata field
        for field_name, field_data in metadata.items():
            if not isinstance(field_data, (list, str)):
                continue
            
            field_text = ""
            if isinstance(field_data, list):
                field_text = " ".join(str(item) for item in field_data)
            else:
                field_text = str(field_data)
            
            field_text_lower = field_text.lower()
            
            # Check if any search terms match this field
            field_matches = 0
            for term in search_terms:
                if term in field_text_lower:
                    field_matches += 1
            
            if field_matches > 0:
                weight = field_weights.get(field_name, 1)
                field_score = field_matches * weight
                total_score += field_score
                
                matched_fields.append({
                    'field': field_name,
                    'matches': field_matches,
                    'sample_data': field_text[:100] + "..." if len(field_text) > 100 else field_text
                })
        
        return total_score, matched_fields
    
    def _search_ocr_text(self, ocr_text, search_terms):
        """Search through OCR text and return score + matched snippets"""
        ocr_lower = ocr_text.lower()
        matched_snippets = []
        total_score = 0
        
        for term in search_terms:
            if term in ocr_lower:
                total_score += 1
                
                # Find snippets around matches
                snippets = self._extract_snippets(ocr_text, term)
                matched_snippets.extend(snippets)
        
        # Remove duplicate snippets
        unique_snippets = []
        for snippet in matched_snippets:
            if snippet not in unique_snippets:
                unique_snippets.append(snippet)
        
        return total_score, unique_snippets[:5]  # Limit to 5 snippets
    
    def _extract_snippets(self, text, search_term, context_chars=150):
        """Extract text snippets around search term matches"""
        snippets = []
        text_lower = text.lower()
        term_lower = search_term.lower()
        
        start = 0
        while True:
            pos = text_lower.find(term_lower, start)
            if pos == -1:
                break
            
            # Extract snippet with context
            snippet_start = max(0, pos - context_chars//2)
            snippet_end = min(len(text), pos + len(search_term) + context_chars//2)
            
            snippet = text[snippet_start:snippet_end].strip()
            
            # Highlight the search term
            highlighted_snippet = re.sub(
                re.escape(search_term), 
                f"<mark>{search_term}</mark>", 
                snippet, 
                flags=re.IGNORECASE
            )
            
            snippets.append(highlighted_snippet)
            start = pos + 1
        
        return snippets
    
    def _create_metadata_preview(self, metadata):
        """Create a preview of key metadata for display"""
        preview = {}
        
        # Extract key fields for preview
        preview_fields = ['financial', 'names', 'dates', 'document_type', 'invoice_numbers']
        
        for field in preview_fields:
            if field in metadata and metadata[field]:
                if isinstance(metadata[field], list):
                    preview[field] = metadata[field][:3]  # First 3 items
                else:
                    preview[field] = metadata[field]
        
        return preview

# Initialize search engine
search_engine = TwoStageSearchEngine()

# ===================================
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

# Register Form (aint no way im sure)
"""def register_view(request):
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

    return render(request, 'auth/register.html', {'form': form})"""

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
            return redirect('search_form')
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

# =====================================================
# TWO-STAGE SEARCH VIEWS 
# =====================================================

@login_required
def search_form(request):
    """Main search form - redirects to results page"""
    return render(request, 'search/search_form.html', {
        'available_clients': Document.objects.values_list('client_name', flat=True).distinct()
    })

def convert_stage_1_results(stage_1_results):
    """Convert Stage 1 results to template format"""
    results = []
    for result in stage_1_results:
        doc_obj = type('obj', (object,), {
            'id': result['document_id'],
            'document_name': result['document_name'],
            'filename': result['document_name'],
            'client_name': result['client_name'],
            'document_type': result['document_type'],
            'match_score': result['match_score'],
            'matched_fields': result['matched_fields'],
            'search_stage': 1,
            'file_paths': result['file_paths'],
            'metadata': result['metadata_preview']
        })
        results.append(doc_obj)
    return results

def convert_stage_2_results(stage_2_results):
    """Convert Stage 2 results to template format"""
    results = []
    for result in stage_2_results:
        doc_obj = type('obj', (object,), {
            'id': result['document_id'],
            'document_name': result['document_name'],
            'filename': result['document_name'],
            'client_name': result['client_name'],
            'document_type': result['document_type'],
            'match_score': result['match_score'],
            'matched_snippets': result['matched_snippets'],
            'search_stage': 2,
            'file_paths': result['file_paths'],
            'metadata': result['metadata_preview']
        })
        results.append(doc_obj)
    return results

# =====================================================
# FUNCTIONS (UPDATED)
# =====================================================

@login_required
def search_documents(request):
    """
    Automatic two-stage search:
    1. Try fast metadata search first
    2. If no results, automatically try deep OCR search
    3. Show results or "no documents found"
    """
    search_query = request.GET.get('search_query', '')
    client_filter = request.GET.get('client_filter', '')
    
    results = []
    total_results = 0
    search_performed = False
    search_stage_used = 1
    search_type = 'metadata'
    
    if search_query:
        search_performed = True
        
        print(f"Starting automatic search for: '{search_query}'")
        
        # STAGE 1: Try metadata search first (fast)
        print(f"Stage 1: Searching metadata...")
        stage_1_results = search_engine.stage_1_search(search_query, client_filter)
        
        if stage_1_results:
            # Found results in metadata - use them
            print(f"Stage 1 found {len(stage_1_results)} results - using metadata search")
            results = convert_stage_1_results(stage_1_results)
            total_results = len(results)
            search_stage_used = 1
            search_type = 'metadata'
            
        else:
            # No results in metadata - automatically try deep search
            print(f"Stage 1 found 0 results - trying Stage 2...")
            print(f"Stage 2: Searching full OCR content...")
            
            stage_2_results = search_engine.stage_2_search(search_query, 'all', client_filter)
            
            if stage_2_results:
                print(f"Stage 2 found {len(stage_2_results)} results - using deep search")
                results = convert_stage_2_results(stage_2_results)
                total_results = len(results)
                search_stage_used = 2
                search_type = 'full_content'
            else:
                print(f"Stage 2 found 0 results - no documents found")
                results = []
                total_results = 0
                search_stage_used = 2
                search_type = 'full_content'
    
    context = {
        'search_query': search_query,
        'client_filter': client_filter,
        'results': results,
        'total_results': total_results,
        'search_performed': search_performed,
        'search_stage_used': search_stage_used,  # Which stage actually found results
        'search_type': search_type,
        'available_clients': Document.objects.values_list('client_name', flat=True).distinct()
    }
    
    return render(request, 'search/search_documents.html', context)

@login_required
def document_detail(request, document_id):
    """Enhanced document detail view with metadata"""
    document = get_object_or_404(Document, id=document_id)  # Fixed: Use Document
    
    # Load metadata if available
    metadata_path = search_engine._get_metadata_path(document)
    metadata = {}
    
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
    
    # Load OCR text preview
    ocr_path = search_engine._get_ocr_path(document)
    ocr_preview = ""
    
    if ocr_path and os.path.exists(ocr_path):
        try:
            with open(ocr_path, 'r', encoding='utf-8') as f:
                ocr_text = f.read()
                # First 500 characters as preview
                ocr_preview = ocr_text[:500] + ("..." if len(ocr_text) > 500 else "")
        except Exception as e:
            print(f"Error loading OCR text: {e}")
    
    context = {
        'document': document,
        'metadata': metadata,
        'ocr_preview': ocr_preview,
        'file_paths': {
            'metadata': metadata_path,
            'ocr': ocr_path,
            'original': search_engine._get_original_path(document)
        }
    }
    
    return render(request, 'search/document_detail.html', context)

@login_required
def documents_view(request):
    """Document listing view"""
    document = Document.objects.all()  # Fixed: Use Document
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