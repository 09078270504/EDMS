from rest_framework import serializers
from .models import Document

# for ml team to insert processed documents
class DocumentCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document # this model will be serialize
        fields = [
            'client_name',
            'document_name',
            'metadata_filename',
            'ocr_filename',
            'original_file_path',
        ]

# for search backend to get file paths
class DocumentListSerializer(serializers.ModelSerializer):
    metadata_path = serializers.SerializerMethodField()
    ocr_path = serializers.SerializerMethodField()
    class Meta:
        model = Document 
        fields = [
            'client_name',
            'document_name',
            'metadata_filename',
            'ocr_filename',
            'metadata_path',
            'ocr_path',
            'original_file_path',
        ]

    def get_metadata_path(self, obj):
        return obj.get_metadata_path()

    def get_ocr_path(self, obj):
        return obj.get_ocr_path()