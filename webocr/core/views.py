from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_GET
from django.contrib.auth import login, logout, get_user_model, authenticate
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django import forms
from django.core.mail import send_mail
from django.conf import settings
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View

# ===========================
# PROJECT IMPORTS
# ===========================
from .models import LoginAttempt, SecurityEvent, UserSession, SuspiciousActivity, ChatConversation, ChatMessage
from database.models import Document
from database.serializer import DocumentListSerializer
from .security_utils import log_security_event, track_user_session, check_multiple_failed_logins
from .services.llm_search import answer_from_context

# ===========================
# STANDARD LIBRARY IMPORTS
# ===========================
import json
import os
import re
from datetime import timedelta

# ===========================
# FUZZY SEARXGH
# ===========================
from rapidfuzz import fuzz, process


# ===========================
# API VIEWS
# ===========================
@method_decorator(csrf_exempt, name='dispatch')
class SecurityEventsAPI(View):
    def get(self, request):
        events = SecurityEvent.objects.filter(
            timestamp__gte=timezone.now() - timedelta(hours=24)
        ).values(
            'event_type', 'user__username', 'ip_address', 
            'timestamp', 'risk_level'
        )[:100]
        
        return JsonResponse({
            'status': 'success',
            'events': list(events),
            'count': len(events)
        })


@method_decorator(csrf_exempt, name='dispatch') 
class SuspiciousActivitiesAPI(View):
    def get(self, request):
        activities = SuspiciousActivity.objects.filter(
            is_resolved=False
        ).values(
            'activity_type', 'user__username', 'ip_address',
            'description', 'timestamp'
        )
        
        return JsonResponse({
            'status': 'success',
            'suspicious_activities': list(activities),
            'count': len(activities)
        })


# ===========================
# SEARCH ENGINE
# ===========================
class EnhancedTwoStageSearchEngine:
    def __init__(self):
        self.stage_1_results = []
        self.stage_2_results = []
        self.fuzzy_results = []
        
        # Fuzzy search configuration
        self.fuzzy_threshold = getattr(settings, 'FUZZY_SEARCH_THRESHOLD', 70)  # Minimum similarity score
        self.fuzzy_limit = getattr(settings, 'FUZZY_SEARCH_LIMIT', 10)  # Max fuzzy results per stage
        self.enable_fuzzy = getattr(settings, 'ENABLE_FUZZY_SEARCH', True)
        
    def stage_1_search(self, query, client_filter=None, use_fuzzy=True):
        """
        Stage 1: Search metadata with exact matching + fuzzy search
        """
        print(f"Stage 1: Searching metadata for '{query}' (fuzzy: {use_fuzzy and self.enable_fuzzy})")
        
        documents = Document.objects.all()
        if client_filter:
            documents = documents.filter(client_name__icontains=client_filter)
        
        exact_matches = []
        fuzzy_candidates = []
        search_terms = query.lower().split()
        
        for doc in documents:
            metadata_path = self._get_metadata_path(doc)
            
            if not metadata_path or not os.path.exists(metadata_path):
                continue
            
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Exact matching (existing logic)
                exact_score, exact_fields = self._search_metadata_exact(metadata, search_terms)
                
                if exact_score > 0:
                    result = self._create_stage1_result(doc, metadata, exact_score, exact_fields, 'exact')
                    exact_matches.append(result)
                elif use_fuzzy and self.enable_fuzzy:
                    # Fuzzy matching for documents without exact matches
                    fuzzy_score, fuzzy_fields = self._search_metadata_fuzzy(metadata, query, search_terms)
                    
                    if fuzzy_score >= self.fuzzy_threshold:
                        result = self._create_stage1_result(doc, metadata, fuzzy_score, fuzzy_fields, 'fuzzy')
                        fuzzy_candidates.append(result)
                        
            except Exception as e:
                print(f"Error reading metadata for {doc.document_name}: {e}")
                continue
        
        # Combine and sort results (exact matches first, then fuzzy)
        all_results = exact_matches + sorted(fuzzy_candidates, key=lambda x: x['match_score'], reverse=True)[:self.fuzzy_limit]
        
        print(f"Stage 1 found {len(exact_matches)} exact matches, {len(fuzzy_candidates)} fuzzy matches")
        return all_results
    
    def stage_2_search(self, query, documents_to_search, client_filter=None, use_fuzzy=True):
        """
        Stage 2: Deep search in OCR text with exact matching + fuzzy search
        """
        print(f"Stage 2: Deep search in OCR text for '{query}' (fuzzy: {use_fuzzy and self.enable_fuzzy})")
        
        if documents_to_search == 'all':
            documents = Document.objects.all()
            if client_filter:
                documents = documents.filter(client_name__icontains=client_filter)
        else:
            documents = Document.objects.filter(id__in=documents_to_search)
        
        exact_matches = []
        fuzzy_candidates = []
        search_terms = query.lower().split()
        
        for doc in documents:
            ocr_path = self._get_ocr_path(doc)
            
            if not ocr_path or not os.path.exists(ocr_path):
                continue
            
            try:
                with open(ocr_path, 'r', encoding='utf-8') as f:
                    ocr_text = f.read()
                
                # Exact matching (existing logic)
                exact_score, exact_snippets = self._search_ocr_text_exact(ocr_text, search_terms)
                
                if exact_score > 0:
                    result = self._create_stage2_result(doc, exact_score, exact_snippets, 'exact')
                    exact_matches.append(result)
                elif use_fuzzy and self.enable_fuzzy:
                    # Fuzzy matching for documents without exact matches
                    fuzzy_score, fuzzy_snippets = self._search_ocr_text_fuzzy(ocr_text, query, search_terms)
                    
                    if fuzzy_score >= self.fuzzy_threshold:
                        result = self._create_stage2_result(doc, fuzzy_score, fuzzy_snippets, 'fuzzy')
                        fuzzy_candidates.append(result)
                        
            except Exception as e:
                print(f"Error reading OCR text for {doc.document_name}: {e}")
                continue
        
        # Combine and sort results
        all_results = exact_matches + sorted(fuzzy_candidates, key=lambda x: x['match_score'], reverse=True)[:self.fuzzy_limit]
        
        print(f"Stage 2 found {len(exact_matches)} exact matches, {len(fuzzy_candidates)} fuzzy matches")
        return all_results
    
    def fuzzy_search_stage(self, query, client_filter=None):
        """
        Stage 3: Pure fuzzy search across all content (fallback option)
        """
        print(f"Stage 3: Pure fuzzy search for '{query}'")
        
        documents = Document.objects.all()
        if client_filter:
            documents = documents.filter(client_name__icontains=client_filter)
        
        fuzzy_results = []
        
        for doc in documents:
            # Combine metadata and OCR content for comprehensive fuzzy search
            combined_text = self._get_combined_document_text(doc)
            
            if combined_text:
                fuzzy_score = self._calculate_fuzzy_score(query, combined_text)
                
                if fuzzy_score >= self.fuzzy_threshold:
                    snippets = self._extract_fuzzy_snippets(combined_text, query)
                    result = self._create_fuzzy_result(doc, fuzzy_score, snippets)
                    fuzzy_results.append(result)
        
        fuzzy_results.sort(key=lambda x: x['match_score'], reverse=True)
        print(f"Stage 3 found {len(fuzzy_results)} fuzzy matches")
        return fuzzy_results[:self.fuzzy_limit]
    
    def _search_metadata_exact(self, metadata, search_terms):
        """Existing exact metadata search logic"""
        matched_fields = []
        total_score = 0
        
        field_weights = {
            'names': 3, 'invoice_numbers': 4, 'financial': 2, 'emails': 2,
            'phones': 2, 'document_type': 3, 'keywords': 1, 'addresses': 1
        }
        
        for field_name, field_data in metadata.items():
            if not isinstance(field_data, (list, str)):
                continue
            
            field_text = ""
            if isinstance(field_data, list):
                field_text = " ".join(str(item) for item in field_data)
            else:
                field_text = str(field_data)
            
            field_text_lower = field_text.lower()
            field_matches = sum(1 for term in search_terms if term in field_text_lower)
            
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
    
    def _search_metadata_fuzzy(self, metadata, original_query, search_terms):
        """Fuzzy metadata search using rapidfuzz"""
        matched_fields = []
        total_score = 0
        
        field_weights = {
            'names': 3, 'invoice_numbers': 4, 'financial': 2, 'emails': 2,
            'phones': 2, 'document_type': 3, 'keywords': 1, 'addresses': 1
        }
        
        for field_name, field_data in metadata.items():
            if not isinstance(field_data, (list, str)):
                continue
            
            field_text = ""
            if isinstance(field_data, list):
                field_text = " ".join(str(item) for item in field_data)
            else:
                field_text = str(field_data)
            
            # Calculate fuzzy similarity
            similarity = fuzz.partial_ratio(original_query.lower(), field_text.lower())
            
            if similarity >= self.fuzzy_threshold:
                weight = field_weights.get(field_name, 1)
                field_score = (similarity / 100.0) * weight * 50  # Scale to match exact search scores
                total_score += field_score
                
                matched_fields.append({
                    'field': field_name,
                    'similarity': similarity,
                    'match_type': 'fuzzy',
                    'sample_data': field_text[:100] + "..." if len(field_text) > 100 else field_text
                })
        
        return total_score, matched_fields
    
    def _search_ocr_text_exact(self, ocr_text, search_terms):
        """Existing exact OCR search logic"""
        ocr_lower = ocr_text.lower()
        matched_snippets = []
        total_score = 0
        
        for term in search_terms:
            if term in ocr_lower:
                total_score += 1
                snippets = self._extract_snippets(ocr_text, term)
                matched_snippets.extend(snippets)
        
        # Remove duplicates
        unique_snippets = []
        for snippet in matched_snippets:
            if snippet not in unique_snippets:
                unique_snippets.append(snippet)
        
        return total_score, unique_snippets[:5]
    
    def _search_ocr_text_fuzzy(self, ocr_text, original_query, search_terms):
        """Fuzzy OCR search using rapidfuzz"""
        
        # Split OCR text into chunks for better fuzzy matching
        chunks = self._split_text_into_chunks(ocr_text, chunk_size=200)
        matched_snippets = []
        best_similarity = 0
        
        for chunk in chunks:
            similarity = fuzz.partial_ratio(original_query.lower(), chunk.lower())
            
            if similarity >= self.fuzzy_threshold:
                if similarity > best_similarity:
                    best_similarity = similarity
                
                # Extract snippet with fuzzy highlighting
                highlighted_chunk = self._highlight_fuzzy_match(chunk, original_query)
                matched_snippets.append(highlighted_chunk)
        
        # Score based on best similarity found
        total_score = (best_similarity / 100.0) * 10  # Scale to match exact search scores
        
        return total_score, matched_snippets[:5]
    
    def _split_text_into_chunks(self, text, chunk_size=200):
        """Split text into overlapping chunks for better fuzzy matching"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size // 2):  # 50% overlap
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def _highlight_fuzzy_match(self, text, query):
        """Highlight fuzzy matches in text"""
        # Find the best matching substring
        words = text.split()
        query_words = query.split()
        
        best_match = ""
        best_score = 0
        
        # Try different combinations of consecutive words
        for i in range(len(words)):
            for j in range(i + 1, min(i + len(query_words) + 2, len(words) + 1)):
                substring = " ".join(words[i:j])
                score = fuzz.ratio(query.lower(), substring.lower())
                
                if score > best_score:
                    best_score = score
                    best_match = substring
        
        # Highlight the best match
        if best_match and best_score >= self.fuzzy_threshold:
            highlighted_text = text.replace(best_match, f"<mark>{best_match}</mark>")
            return highlighted_text
        
        return text
    
    def _calculate_fuzzy_score(self, query, combined_text):
        """Calculate overall fuzzy similarity score"""
        # Use different fuzzy algorithms and take the best score
        partial_ratio = fuzz.partial_ratio(query.lower(), combined_text.lower())
        token_sort_ratio = fuzz.token_sort_ratio(query.lower(), combined_text.lower())
        token_set_ratio = fuzz.token_set_ratio(query.lower(), combined_text.lower())
        
        # Return the highest score
        return max(partial_ratio, token_sort_ratio, token_set_ratio)
    
    def _get_combined_document_text(self, doc):
        """Get combined metadata and OCR text for comprehensive fuzzy search"""
        combined_parts = []
        
        # Add metadata text
        metadata_path = self._get_metadata_path(doc)
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                for key, value in metadata.items():
                    if isinstance(value, list):
                        combined_parts.extend(str(item) for item in value)
                    elif isinstance(value, str):
                        combined_parts.append(value)
            except:
                pass
        
        # Add OCR text (first 1000 characters to avoid memory issues)
        ocr_path = self._get_ocr_path(doc)
        if ocr_path and os.path.exists(ocr_path):
            try:
                with open(ocr_path, 'r', encoding='utf-8') as f:
                    ocr_text = f.read()[:1000]  # Limit size for fuzzy search
                    combined_parts.append(ocr_text)
            except:
                pass
        
        return " ".join(combined_parts)
    
    def _extract_fuzzy_snippets(self, text, query):
        """Extract relevant snippets for fuzzy matches"""
        chunks = self._split_text_into_chunks(text, chunk_size=150)
        relevant_snippets = []
        
        for chunk in chunks:
            similarity = fuzz.partial_ratio(query.lower(), chunk.lower())
            
            if similarity >= self.fuzzy_threshold:
                highlighted_chunk = self._highlight_fuzzy_match(chunk, query)
                relevant_snippets.append(highlighted_chunk)
        
        return relevant_snippets[:3]
    
    def _create_stage1_result(self, doc, metadata, score, fields, match_type):
        """Create result object for stage 1 search"""
        return {
            'document_id': doc.id,
            'document_name': doc.document_name,
            'client_name': getattr(doc, 'client_name', 'Unknown'),
            'document_type': metadata.get('document_type', 'Unknown'),
            'match_score': score,
            'matched_fields': fields,
            'match_type': match_type,
            'metadata_preview': self._create_metadata_preview(metadata),
            'file_paths': {
                'metadata': self._get_metadata_path(doc),
                'ocr': self._get_ocr_path(doc),
                'original': self._get_original_path(doc)
            },
            'search_stage': 1
        }
    
    def _create_stage2_result(self, doc, score, snippets, match_type):
        """Create result object for stage 2 search"""
        metadata_path = self._get_metadata_path(doc)
        metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except:
                pass
        
        return {
            'document_id': doc.id,
            'document_name': doc.document_name,
            'client_name': getattr(doc, 'client_name', 'Unknown'),
            'document_type': metadata.get('document_type', 'Unknown'),
            'match_score': score,
            'matched_snippets': snippets,
            'match_type': match_type,
            'metadata_preview': self._create_metadata_preview(metadata),
            'file_paths': {
                'metadata': metadata_path,
                'ocr': self._get_ocr_path(doc),
                'original': self._get_original_path(doc)
            },
            'search_stage': 2
        }
    
    def _create_fuzzy_result(self, doc, score, snippets):
        """Create result object for fuzzy search"""
        metadata_path = self._get_metadata_path(doc)
        metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except:
                pass
        
        return {
            'document_id': doc.id,
            'document_name': doc.document_name,
            'client_name': getattr(doc, 'client_name', 'Unknown'),
            'document_type': metadata.get('document_type', 'Unknown'),
            'match_score': score,
            'matched_snippets': snippets,
            'match_type': 'fuzzy',
            'metadata_preview': self._create_metadata_preview(metadata),
            'file_paths': {
                'metadata': metadata_path,
                'ocr': self._get_ocr_path(doc),
                'original': self._get_original_path(doc)
            },
            'search_stage': 3
        }
    
    # Keep all your existing helper methods (_get_metadata_path, _get_ocr_path, etc.)
    def _get_metadata_path(self, doc):
        if hasattr(doc, 'metadata_path') and doc.metadata_path:
            return doc.metadata_path
        
        base_name = os.path.splitext(doc.document_name)[0]
        client_name = getattr(doc, 'client_name', 'unknown')
        return f"archive/{client_name}/{base_name}/metadata/{base_name}.json"
    
    def _get_ocr_path(self, doc):
        if hasattr(doc, 'ocr_path') and doc.ocr_path:
            return doc.ocr_path
        
        base_name = os.path.splitext(doc.document_name)[0]
        client_name = getattr(doc, 'client_name', 'unknown')
        return f"archive/{client_name}/{base_name}/ocr/{base_name}.txt"
    
    def _get_original_path(self, doc):
        if hasattr(doc, 'original_path') and doc.original_path:
            return doc.original_path
        
        base_name = os.path.splitext(doc.document_name)[0]
        client_name = getattr(doc, 'client_name', 'unknown')
        return f"archive/{client_name}/{base_name}/original/{base_name}.pdf"
    
    def _extract_snippets(self, text, search_term, context_chars=150):
        snippets = []
        text_lower = text.lower()
        term_lower = search_term.lower()
        
        start = 0
        while True:
            pos = text_lower.find(term_lower, start)
            if pos == -1:
                break
            
            snippet_start = max(0, pos - context_chars//2)
            snippet_end = min(len(text), pos + len(search_term) + context_chars//2)
            
            snippet = text[snippet_start:snippet_end].strip()
            
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
        preview = {}
        preview_fields = ['financial', 'names', 'dates', 'document_type', 'invoice_numbers']
        
        for field in preview_fields:
            if field in metadata and metadata[field]:
                if isinstance(metadata[field], list):
                    preview[field] = metadata[field][:3]
                else:
                    preview[field] = metadata[field]
        
        return preview


enhanced_search_engine = EnhancedTwoStageSearchEngine()


# ===========================
# AUTHENTICATION VIEWS
# ===========================
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
    if request.user.is_authenticated:
        return redirect('search_form')
    return render(request, 'base.html')


def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        ip_address = request.META.get('REMOTE_ADDR', '')
        
        try:
            attempt = LoginAttempt.objects.get(ip_address=ip_address, username=username)
            if attempt.is_locked_out():
                log_security_event(
                    event_type='account_locked',
                    username_attempted=username,
                    request=request,
                    risk_level='medium'
                )
                return redirect('account_locked')
        except LoginAttempt.DoesNotExist:
            pass
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            LoginAttempt.objects.filter(ip_address=ip_address, username=username).delete()
            
            log_security_event(
                event_type='login_success',
                user=user,
                request=request,
                risk_level='low'
            )
            
            track_user_session(user, request)
            login(request, user)
            
            return redirect('search_form')
        else:
            attempt, created = LoginAttempt.objects.get_or_create(
                ip_address=ip_address,
                username=username,
                defaults={'failures_count': 1}
            )
            if not created:
                attempt.failures_count += 1
                attempt.attempt_time = timezone.now()
                attempt.save()
            
            log_security_event(
                event_type='login_failure',
                username_attempted=username,
                request=request,
                extra_data={'failures_count': attempt.failures_count},
                risk_level='medium' if attempt.failures_count >= 3 else 'low'
            )
            
            check_multiple_failed_logins(ip_address, username)
            
            if attempt.failures_count >= 5:
                log_security_event(
                    event_type='account_locked',
                    username_attempted=username,
                    request=request,
                    risk_level='high'
                )
                return redirect('account_locked')
            
            messages.error(request, 'Invalid username or password')
    
    return render(request, 'auth/login.html')


def account_locked_view(request):
    ip_address = request.META.get('REMOTE_ADDR', '')
    
    attempt = LoginAttempt.objects.filter(ip_address=ip_address).order_by('-attempt_time').first()
    
    if attempt and attempt.is_locked_out():
        remaining_seconds = attempt.remaining_lockout_time()
    else:
        if attempt:
            attempt.delete()
        return redirect('login')
    
    context = {
        'remaining_seconds': remaining_seconds,
        'lockout_expired': remaining_seconds <= 0
    }
    
    return render(request, 'auth/account_locked.html', context)


def forgot_password_view(request):
    if request.method == 'POST':
        form = ForgotPasswordForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            
            try:
                User = get_user_model()
                user = User.objects.get(email=email)
                
                log_security_event(
                    event_type='password_reset_request',
                    user=user,
                    request=request,
                    risk_level='medium'
                )
                
                user.set_password("comfac123")
                user.save()
                
                log_security_event(
                    event_type='password_reset_complete',
                    user=user,
                    request=request,
                    risk_level='medium'
                )
                
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


@login_required
def dashboard(request, conversation_id=None):
    total_documents = Document.objects.count()
    
    # Get all conversations for the current user
    conversations = ChatConversation.objects.filter(user=request.user).order_by('-created_at')
    print(f"DEBUG: Found {conversations.count()} conversations for user {request.user.username}")
    
    # Get messages for current conversation if specified
    messages = []
    current_conversation = None
    if conversation_id:
        try:
            current_conversation = ChatConversation.objects.get(id=conversation_id, user=request.user)
            messages = current_conversation.messages.all()
            print(f"DEBUG: Found {messages.count()} messages for conversation {conversation_id}")
        except ChatConversation.DoesNotExist:
            print(f"DEBUG: Conversation {conversation_id} not found")
            pass
    
    stats = {
        'total_documents': total_documents,
    }
    
    context = {
        'user': request.user,
        'stats': stats,
        'messages': messages,
        'conversations': conversations,
        'current_conversation': current_conversation,
    }
    return render(request, 'auth/dashboard.html', context)


def logout_view(request):
    user = request.user
    
    if user.is_authenticated:
        log_security_event(
            event_type='logout',
            user=user,
            request=request,
            risk_level='low'
        )
        
        UserSession.objects.filter(
            user=user,
            session_key=request.session.session_key
        ).update(is_active=False)
    
    logout(request)
    return redirect('login')


@login_required
def change_password(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, 'Password changed successfully! Please log in again.')
            logout(request)
            return redirect('login')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, error)
    else:
        form = PasswordChangeForm(request.user)
    
    # Get conversations for sidebar
    conversations = ChatConversation.objects.filter(user=request.user).order_by('-created_at')
    
    return render(request, 'auth/change_password.html', {
        'form': form,
        'conversations': conversations,
    })


# ===========================
# SEARCH VIEWS
# ===========================
@login_required
def search_form(request):
    return render(request, 'search/search_form.html', {
        'available_clients': Document.objects.values_list('client_name', flat=True).distinct()
    })


def convert_stage_1_results(stage_1_results):
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


@login_required
def search_documents(request):
    search_query = request.GET.get('search_query', '')
    client_filter = request.GET.get('client_filter', '')
    enable_fuzzy = request.GET.get('fuzzy', 'true').lower() == 'true'  # Allow disabling fuzzy search
    
    results = []
    total_results = 0
    search_performed = False
    search_stage_used = 1
    search_type = 'metadata'
    search_modes_used = []
    
    if search_query:
        search_performed = True
        print(f"Starting enhanced search: '{search_query}' (fuzzy: {enable_fuzzy})")
        
        # Stage 1: Enhanced metadata search with fuzzy
        stage_1_results = enhanced_search_engine.stage_1_search(
            search_query, client_filter, use_fuzzy=enable_fuzzy
        )
        
        if stage_1_results:
            results = convert_enhanced_stage_1_results(stage_1_results)
            total_results = len(results)
            search_stage_used = 1
            search_type = 'metadata'
            
            # Track which search modes found results
            exact_count = len([r for r in stage_1_results if r.get('match_type') == 'exact'])
            fuzzy_count = len([r for r in stage_1_results if r.get('match_type') == 'fuzzy'])
            
            if exact_count > 0:
                search_modes_used.append(f"{exact_count} exact")
            if fuzzy_count > 0:
                search_modes_used.append(f"{fuzzy_count} fuzzy")
                
        else:
            print("No Stage 1 matches - trying Stage 2 with fuzzy search")
            
            # Stage 2: Enhanced OCR search with fuzzy
            stage_2_results = enhanced_search_engine.stage_2_search(
                search_query, 'all', client_filter, use_fuzzy=enable_fuzzy
            )
            
            if stage_2_results:
                results = convert_enhanced_stage_2_results(stage_2_results)
                total_results = len(results)
                search_stage_used = 2
                search_type = 'full_content'
                
                exact_count = len([r for r in stage_2_results if r.get('match_type') == 'exact'])
                fuzzy_count = len([r for r in stage_2_results if r.get('match_type') == 'fuzzy'])
                
                if exact_count > 0:
                    search_modes_used.append(f"{exact_count} exact")
                if fuzzy_count > 0:
                    search_modes_used.append(f"{fuzzy_count} fuzzy")
            else:
                # Stage 3: Pure fuzzy search as last resort (only if fuzzy is enabled)
                if enable_fuzzy:
                    print("No Stage 2 matches - trying pure fuzzy search")
                    
                    fuzzy_results = enhanced_search_engine.fuzzy_search_stage(
                        search_query, client_filter
                    )
                    
                    if fuzzy_results:
                        results = convert_enhanced_fuzzy_results(fuzzy_results)
                        total_results = len(results)
                        search_stage_used = 3
                        search_type = 'fuzzy_comprehensive'
                        search_modes_used.append(f"{len(fuzzy_results)} comprehensive fuzzy")
    
    # Create search summary for display
    search_summary = " + ".join(search_modes_used) if search_modes_used else ""
    
    context = {
        'search_query': search_query,
        'client_filter': client_filter,
        'results': results,
        'total_results': total_results,
        'search_performed': search_performed,
        'search_stage_used': search_stage_used,
        'search_type': search_type,
        'search_summary': search_summary,
        'fuzzy_enabled': enable_fuzzy,
        'available_clients': Document.objects.values_list('client_name', flat=True).distinct()
    }
    
    return render(request, 'search/search_documents.html', context)

def convert_enhanced_stage_1_results(stage_1_results):
    """Convert enhanced stage 1 results with fuzzy match information"""
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
            'match_type': result.get('match_type', 'exact'),
            'search_stage': 1,
            'file_paths': result['file_paths'],
            'metadata': result['metadata_preview']
        })
        results.append(doc_obj)
    return results

def convert_enhanced_stage_2_results(stage_2_results):
    """Convert enhanced stage 2 results with fuzzy match information"""
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
            'match_type': result.get('match_type', 'exact'),
            'search_stage': 2,
            'file_paths': result['file_paths'],
            'metadata': result['metadata_preview']
        })
        results.append(doc_obj)
    return results

def convert_enhanced_fuzzy_results(fuzzy_results):
    """Convert pure fuzzy search results"""
    results = []
    for result in fuzzy_results:
        doc_obj = type('obj', (object,), {
            'id': result['document_id'],
            'document_name': result['document_name'],
            'filename': result['document_name'],
            'client_name': result['client_name'],
            'document_type': result['document_type'],
            'match_score': result['match_score'],
            'matched_snippets': result['matched_snippets'],
            'match_type': 'fuzzy',
            'search_stage': 3,
            'file_paths': result['file_paths'],
            'metadata': result['metadata_preview']
        })
        results.append(doc_obj)
    return results

@login_required
def document_detail(request, document_id):
    document = get_object_or_404(Document, id=document_id)
    
    metadata_path = search_engine._get_metadata_path(document) # type: ignore
    metadata = {}
    
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
    
    ocr_path = search_engine._get_ocr_path(document) # type: ignore
    ocr_preview = ""
    
    if ocr_path and os.path.exists(ocr_path):
        try:
            with open(ocr_path, 'r', encoding='utf-8') as f:
                ocr_text = f.read()
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
            'original': search_engine._get_original_path(document) # type: ignore
        }
    }
    
    return render(request, 'search/document_detail.html', context)


@login_required
def documents_view(request):
    document = Document.objects.all()
    return render(request, 'documents/documents_view.html', {'document': document})


@require_GET
@login_required
def document_list_api(request):
    documents = Document.objects.all()
    serializer = DocumentListSerializer(documents, many=True)
    return JsonResponse(serializer.data, safe=False)


# ===========================
# CHAT VIEWS
# ===========================
@login_required
def chat_message(request):
    """Handle chat messages and create/update conversations"""
    if request.method == 'POST':
        message_content = request.POST.get('message', '').strip()
        conversation_id = request.POST.get('conversation_id', '')
        
        if not message_content:
            return JsonResponse({'error': 'Message cannot be empty'}, status=400)
        
        # Get or create conversation
        conversation = None
        if conversation_id:
            try:
                conversation = ChatConversation.objects.get(id=conversation_id, user=request.user)
            except ChatConversation.DoesNotExist:
                pass
        
        if not conversation:
            # Create new conversation with first message as title
            title = message_content[:50] + '...' if len(message_content) > 50 else message_content
            conversation = ChatConversation.objects.create(
                user=request.user,
                title=title
            )
        
        # Save user message
        user_message = ChatMessage.objects.create(
            conversation=conversation,
            message_type='user',
            content=message_content
        )
        
        # Here you would integrate with your search/AI logic
        # For now, let's create simple, natural chat responses
        responses = [
            f"Thank you for your message about '{message_content}'. I'm here to help with any questions you might have.",
            f"I understand you're asking about '{message_content}'. Could you provide more details so I can assist you better?",
            f"That's an interesting point about '{message_content}'. What specific information are you looking for?",
            f"I see you mentioned '{message_content}'. How can I help you with this topic?",
            f"I'd be happy to help you with '{message_content}'. What would you like to know more about?",
            f"Thanks for bringing up '{message_content}'. Is there something specific you'd like assistance with?"
        ]
        import random
        response_content = random.choice(responses)
        
        assistant_message = ChatMessage.objects.create(
            conversation=conversation,
            message_type='assistant',
            content=response_content
        )
        
        return JsonResponse({
            'success': True,
            'conversation_id': conversation.id,
            'is_new_conversation': not bool(conversation_id),
            'user_message': {
                'content': user_message.content,
                'timestamp': user_message.timestamp.isoformat()
            },
            'assistant_message': {
                'content': assistant_message.content,
                'timestamp': assistant_message.timestamp.isoformat()
            }
        })
    
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)


@login_required
def delete_conversation(request, conversation_id):
    """Delete a conversation"""
    if request.method in ['POST', 'DELETE']:
        try:
            conversation = ChatConversation.objects.get(id=conversation_id, user=request.user)
            conversation.delete()
            return JsonResponse({'success': True})
        except ChatConversation.DoesNotExist:
            return JsonResponse({'error': 'Conversation not found'}, status=404)
    
    return JsonResponse({'error': 'Only POST/DELETE method allowed'}, status=405)


@login_required
def rename_conversation(request, conversation_id):
    """Rename a conversation"""
    if request.method == 'POST':
        try:
            import json
            data = json.loads(request.body)
            new_title = data.get('title', '').strip()
            
            if not new_title:
                return JsonResponse({'error': 'Title cannot be empty'}, status=400)
            
            conversation = ChatConversation.objects.get(id=conversation_id, user=request.user)
            conversation.title = new_title
            conversation.save()
            
            return JsonResponse({'success': True, 'title': new_title})
        except ChatConversation.DoesNotExist:
            return JsonResponse({'error': 'Conversation not found'}, status=404)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)


# ===========================
# LLM SEARCH VIEWS
# ===========================
@login_required
def llm_search_documents(request):
    search_query = request.GET.get('search_query', '').strip()
    client_filter = request.GET.get('client_filter', '').strip()
    enable_fuzzy = request.GET.get('fuzzy', 'true').lower() == 'true'

    results = []
    total_results = 0
    search_performed = False
    llm_answer = ""
    error_message = ""

    if search_query:
        search_performed = True
        
        try:
            # Use enhanced search engine for better context gathering
            stage_1_results = enhanced_search_engine.stage_1_search(
                search_query, client_filter, use_fuzzy=enable_fuzzy
            )
            docs_for_context = stage_1_results

            # Fallback to enhanced stage 2 if needed
            if not docs_for_context:
                stage_2_results = enhanced_search_engine.stage_2_search(
                    search_query, 'all', client_filter, use_fuzzy=enable_fuzzy
                )
                docs_for_context = stage_2_results
                
            # Final fallback to pure fuzzy search
            if not docs_for_context and enable_fuzzy:
                fuzzy_results = enhanced_search_engine.fuzzy_search_stage(
                    search_query, client_filter
                )
                docs_for_context = fuzzy_results

            if docs_for_context:
                # Build context with enhanced information
                context_blocks = []
                for i, doc_result in enumerate(docs_for_context[:5]):
                    meta = doc_result.get('metadata_preview') or {}
                    
                    context_lines = [
                        f"=== Document {i+1}: {doc_result.get('document_name')} ===",
                        f"Client: {doc_result.get('client_name', 'Unknown')}",
                        f"Type: {doc_result.get('document_type', 'Unknown')}",
                        f"Match type: {doc_result.get('match_type', 'exact')}",
                        f"Match score: {doc_result.get('match_score', 0):.1f}"
                    ]
                    
                    # Add metadata information
                    if meta:
                        for key, value in meta.items():
                            if value and key in ['financial', 'names', 'dates', 'invoice_numbers']:
                                if isinstance(value, list):
                                    context_lines.append(f"{key.title()}: {', '.join(map(str, value[:3]))}")
                                else:
                                    context_lines.append(f"{key.title()}: {value}")
                    
                    # Add content snippet
                    if 'matched_snippets' in doc_result and doc_result['matched_snippets']:
                        clean_snippet = re.sub(r'<[^>]+>', '', doc_result['matched_snippets'][0])
                        context_lines.append(f"Content: {clean_snippet[:300]}...")
                    
                    context_blocks.append("\n".join(context_lines))

                full_context = "\n\n".join(context_blocks)
                
                # Send to LLM
                llm_answer = answer_from_context(search_query, full_context, temperature=0.2)
                
                # Convert results for display
                if 'matched_snippets' in docs_for_context[0] if docs_for_context else False:
                    results = convert_enhanced_stage_2_results(docs_for_context[:10])
                elif docs_for_context[0].get('search_stage') == 3:
                    results = convert_enhanced_fuzzy_results(docs_for_context[:10])
                else:
                    results = convert_enhanced_stage_1_results(docs_for_context[:10])

                total_results = len(results)
                
            else:
                llm_answer = "I couldn't find any documents that match your query, even with fuzzy search enabled. This could be because the database is empty or your search terms are too different from the document content."
                
        except Exception as e:
            print(f"Enhanced LLM Search Error: {e}")
            error_message = f"An error occurred during AI search: {str(e)[:100]}"
            llm_answer = ""

    context = {
        'search_query': search_query,
        'client_filter': client_filter,
        'results': results,
        'total_results': total_results,
        'search_performed': search_performed,
        'search_stage_used': 3,
        'search_type': 'llm_enhanced',
        'llm_answer': llm_answer,
        'error_message': error_message,
        'fuzzy_enabled': enable_fuzzy,
        'available_clients': Document.objects.values_list('client_name', flat=True).distinct()
    }
    
    return render(request, 'search/search_documents.html', context)


@login_required  
def llm_test_connection(request):
    """
    Test endpoint to verify LLM service is working
    """
    try:
        test_query = "What is a test?"
        test_context = "This is a test document with sample information."
        result = answer_from_context(test_query, test_context, temperature=0.1)
        
        return JsonResponse({
            'status': 'success',
            'message': 'LLM service is working',
            'test_response': result[:100] + "..." if len(result) > 100 else result
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'LLM service error: {str(e)}'
        }, status=500)


def llm_ping(request):
    """
    Simple ping endpoint to test if LLM imports work
    """
    try:
        # Test if we can import the LLM service
        from .services.llm_search import answer_from_context
        
        # Try a very simple test
        result = answer_from_context("ping", "test", 0.1)
        status = "LLM service loaded successfully"
        
        return HttpResponse(f"LLM Status: {status}\nTest result: {result[:50]}...")
        
    except ImportError as e:
        return HttpResponse(f"LLM Import Error: {e}", status=500)
    except Exception as e:
        return HttpResponse(f"LLM Error: {e}", status=500)