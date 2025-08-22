#For URL routes
from django.urls import path
#Imports views.py
from .import views
#Imports change_password from views.py
from .views import change_password, SecurityEventsAPI, SuspiciousActivitiesAPI

urlpatterns = [
    # Authentication URLs
    path('', views.base, name='base'),
    path('login/', views.login_view, name='login'),
    #path('register/', views.register_view, name='register'),
    path('forgot-password/', views.forgot_password_view, name='forgot_password'),
    path('logout/', views.logout_view, name='logout'),
    path('chat/', views.dashboard, name='dashboard'),
    path('change-password/', views.change_password, name='change_password'),
    
    # Search URLs (Simplified)
    path('documents/', views.search_form, name='search_form'),                    # Search form
    path('search/documents/', views.search_documents, name='search_documents'), # Results page (automatic search)
    
    # Document URLs
    path('document/<int:document_id>/', views.document_detail, name='document_detail'),
    path('documents/', views.documents_view, name='documents_view'),
    # Password Change URL
    path('change_password/', change_password, name='change_password'),
    # Forgot password functionality
    path('forgot-password/', views.forgot_password_view, name='forgot_password'),
    # Account lock
    path('account-locked/', views.account_locked_view, name='account_locked'),
    # For team integrations
    path('api/security-events/', SecurityEventsAPI.as_view(), name='security_events_api'),
    path('api/suspicious-activities/', SuspiciousActivitiesAPI.as_view(), name='suspicious_activities_api'),
]


