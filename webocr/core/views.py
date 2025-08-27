from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
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
from .models import LoginAttempt, SecurityEvent, UserSession, SuspiciousActivity, ChatConversation, ChatMessage  # Local models
from database.models import Document  # Document model
from database.serializer import DocumentListSerializer  # Serializer
from .security_utils import log_security_event, track_user_session, check_multiple_failed_logins # For enhanced log in view

# ===========================
# STANDARD LIBRARY IMPORTS
# ===========================
import json  # JSON handling
import os  # File system
import re  # Regex
from datetime import timedelta  # Time delta


# For team integrations
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

class TwoStageSearchEngine:
    def __init__(self):
        self.stage_1_results = []
        self.stage_2_results = []
    
    def stage_1_search(self, query, client_filter=None):
        print(f"Stage 1: Searching metadata for '{query}'")
        
        documents = Document.objects.all()
        if client_filter:
            documents = documents.filter(client_name__icontains=client_filter)
        
        stage_1_matches = []
        search_terms = query.lower().split()
        
        for doc in documents:
            metadata_path = self._get_metadata_path(doc)
            
            if not metadata_path or not os.path.exists(metadata_path):
                continue
            
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
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
        
        stage_1_matches.sort(key=lambda x: x['match_score'], reverse=True)
        print(f"Stage 1 found {len(stage_1_matches)} results")
        return stage_1_matches
    
    def stage_2_search(self, query, documents_to_search, client_filter=None):
        print(f"Stage 2: Deep search in OCR text for '{query}'")
        
        if documents_to_search == 'all':
            documents = Document.objects.all()
            if client_filter:
                documents = documents.filter(client_name__icontains=client_filter)
        else:
            documents = Document.objects.filter(id__in=documents_to_search)
        
        stage_2_matches = []
        search_terms = query.lower().split()
        
        for doc in documents:
            ocr_path = self._get_ocr_path(doc)
            
            if not ocr_path or not os.path.exists(ocr_path):
                continue
            
            try:
                with open(ocr_path, 'r', encoding='utf-8') as f:
                    ocr_text = f.read()
                
                match_score, matched_snippets = self._search_ocr_text(ocr_text, search_terms)
                
                if match_score > 0:
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
        
        stage_2_matches.sort(key=lambda x: x['match_score'], reverse=True)
        print(f"Stage 2 found {len(stage_2_matches)} results")
        return stage_2_matches
    
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
    
    def _search_metadata(self, metadata, search_terms):
        matched_fields = []
        total_score = 0
        
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
        
        for field_name, field_data in metadata.items():
            if not isinstance(field_data, (list, str)):
                continue
            
            field_text = ""
            if isinstance(field_data, list):
                field_text = " ".join(str(item) for item in field_data)
            else:
                field_text = str(field_data)
            
            field_text_lower = field_text.lower()
            
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
        ocr_lower = ocr_text.lower()
        matched_snippets = []
        total_score = 0
        
        for term in search_terms:
            if term in ocr_lower:
                total_score += 1
                snippets = self._extract_snippets(ocr_text, term)
                matched_snippets.extend(snippets)
        
        unique_snippets = []
        for snippet in matched_snippets:
            if snippet not in unique_snippets:
                unique_snippets.append(snippet)
        
        return total_score, unique_snippets[:5]
    
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

search_engine = TwoStageSearchEngine()

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
    
    # Get messages for current conversation if specified
    messages = []
    if conversation_id:
        try:
            conversation = ChatConversation.objects.get(id=conversation_id, user=request.user)
            messages = conversation.messages.all()
        except ChatConversation.DoesNotExist:
            pass
    
    stats = {
        'total_documents': total_documents,
    }
    
    context = {
        'user': request.user,
        'stats': stats,
        'messages': messages,
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
    
    results = []
    total_results = 0
    search_performed = False
    search_stage_used = 1
    search_type = 'metadata'
    
    if search_query:
        search_performed = True
        print(f"Starting search: '{search_query}'")
        
        stage_1_results = search_engine.stage_1_search(search_query, client_filter)
        
        if stage_1_results:
            results = convert_stage_1_results(stage_1_results)
            total_results = len(results)
            search_stage_used = 1
            search_type = 'metadata'
        else:
            print("No Stage 1 matches - falling back to Stage 2")
            stage_2_results = search_engine.stage_2_search(search_query, 'all', client_filter)
            
            if stage_2_results:
                results = convert_stage_2_results(stage_2_results)
                total_results = len(results)
                search_stage_used = 2
                search_type = 'full_content'
    
    context = {
        'search_query': search_query,
        'client_filter': client_filter,
        'results': results,
        'total_results': total_results,
        'search_performed': search_performed,
        'search_stage_used': search_stage_used,
        'search_type': search_type,
        'available_clients': Document.objects.values_list('client_name', flat=True).distinct()
    }
    
    return render(request, 'search/search_documents.html', context)

@login_required
def document_detail(request, document_id):
    document = get_object_or_404(Document, id=document_id)
    
    metadata_path = search_engine._get_metadata_path(document)
    metadata = {}
    
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
    
    ocr_path = search_engine._get_ocr_path(document)
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
            'original': search_engine._get_original_path(document)
        }
    }
    
    return render(request, 'search/document_detail.html', context)

@login_required
def documents_view(request):
    document = Document.objects.all()
    return render(request, 'documents/documents_view.html', {'document': document})

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
    return render(request, 'auth/change_password.html', {'form': form, 'hide_sidebar': True})

@require_GET
@login_required
def document_list_api(request):
    documents = Document.objects.all()
    serializer = DocumentListSerializer(documents, many=True)
    return JsonResponse(serializer.data, safe=False)


# Chat-related views
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