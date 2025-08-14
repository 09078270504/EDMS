from django.urls import path
from .views import (
    DocumentCreateView, 
    MetadataSearchView, 
    DocumentListView, 
    DocumentDetailView
)

urlpatterns = [
    path('documents/create/', DocumentCreateView.as_view(), name='document-create'),
    path('documents/search/', MetadataSearchView.as_view(), name='metadata-search'),
    path('documents/', DocumentListView.as_view(), name='document-list'),
    path('documents/<int:document_id>/', DocumentDetailView.as_view(), name='document-detail'),
]