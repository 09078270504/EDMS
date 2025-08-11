#For URL routes
from django.urls import path
#Imports views.py
from . import views
#Imports change_password from views.py
from .views import change_password

urlpatterns = [
    # Root URL of site
    path('search/', views.search_form, name='search_form'),
    # Base URL of the site
    path('', views.base, name='base'),
    # Authentication URLs
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    # Home page / Dashboard
    path('dashboard/', views.dashboard, name='dashboard'),
    # Search URLs
    path('search_documents/', views.search_documents, name='search_documents'),
    path('documents_view/', views.dashboard, name='documents_view'),
    path('document/<int:document_id>/', views.document_detail, name='document_detail'),
    # Password Change URL
    path('change_password/', change_password, name='change_password'),
    # Forgot password functionality
    path('forgot-password/', views.forgot_password_view, name='forgot_password'),
]