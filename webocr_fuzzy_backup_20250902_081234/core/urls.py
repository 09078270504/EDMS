# core/urls.py (your app's URLs)
from django.urls import path
from .import views
from .views import change_password, SecurityEventsAPI, SuspiciousActivitiesAPI

urlpatterns = [
    # Authentication URLs
    path('', views.base, name='base'),
    path('login/', views.login_view, name='login'),
    path('forgot-password/', views.forgot_password_view, name='forgot_password'),
    path('logout/', views.logout_view, name='logout'),
    path('change-password/', views.change_password, name='change_password'),
    path('account-locked/', views.account_locked_view, name='account_locked'),
    
    # Chat URLs
    path('chat/', views.dashboard, name='dashboard'),
    path('chat/<int:conversation_id>/', views.dashboard, name='dashboard_conversation'),
    path('chat/message/', views.chat_message, name='chat_message'),
    path('chat/delete/<int:conversation_id>/', views.delete_conversation, name='delete_conversation'),
    path('chat/rename/<int:conversation_id>/', views.rename_conversation, name='rename_conversation'),
    
    # MAIN SEARCH PAGE - Entry point for both search modes
    path('documents/', views.search_form, name='search_form'),
    
    # DUAL SEARCH ENGINES - Two different result pages
    path('search/documents/', views.search_documents, name='search_documents'),        # Keyword search
    path('search/llm/', views.llm_search_documents, name='llm_search_documents'),     # LLM search
    
    # Document URLs
    path('document/<int:document_id>/', views.document_detail, name='document_detail'),
    path('documents/view/', views.documents_view, name='documents_view'),
    
    # API URLs
    path('api/security-events/', SecurityEventsAPI.as_view(), name='security_events_api'),
    path('api/suspicious-activities/', SuspiciousActivitiesAPI.as_view(), name='suspicious_activities_api'),
    path('api/document-list/', views.document_list_api, name='document_list_api'),
    
    # LLM Testing URLs
    path('search/llm/ping/', views.llm_ping, name='llm_ping'),                       # Basic connectivity test
    path('search/llm/test/', views.llm_test_connection, name='llm_test'),            # Full LLM test
]

# Main project urls.py (webocr/urls.py)
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('core.urls')),  # Make sure this points to your app
]

# Serve static and media files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
"""