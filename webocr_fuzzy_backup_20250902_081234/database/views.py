from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from django.core.exceptions import ValidationError
from .models import Document
from .serializer import DocumentCreateSerializer, DocumentListSerializer

# for ml team to insert processed documents
class DocumentCreateView(APIView):
    def post(self, request):
        # deserialize the incoming data
        serializer = DocumentCreateSerializer(data=request.data)

        if serializer.is_valid():
            try:
                document = serializer.save(status='completed') # this will save the document
                return Response({
                    'id': document.id,
                    'message': 'Document inserted successfully!',
                    'metadata_path': document.get_metadata_path(),
                    'ocr_path': document.get_ocr_path()
                }, status=status.HTTP_201_CREATED)
            except ValidationError as e:
                return Response({
                    'error': 'Validation error occurred',
                    'details': str(e)
                }, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                return Response({
                    'error': 'An unexpected error occurred',
                    'details': str(e)
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response({
                'errors': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
# for search backend - primary search
class MetadataSearchView(APIView):
    def get(self, request):
        # GET optional client filter
        client_filter = request.GET.get('client', '')

        # only complete data with metadata
        queryset = Document.objects.filter(
            status = 'completed',
            metadata_filename__isnull=False
        )

        if client_filter:
            queryset = queryset.filter(client_name__icontains=client_filter)

        # Exclude document with empty metadata filename
        documents = queryset.exclude(metadata_filename='')

        # extract OCR filenames
        ocr_files = [doc.ocr_filename for doc in documents if doc.ocr_filename]

        return Response({
            'ocr_directory': '/webocr/ocr/',
            'files': ocr_files,
            'total_count': len(ocr_files)
        })

# Get all documents with their file paths (for debugging/admin)
class DocumentListView(APIView):

    def get(self, request):
        # Get optional filters
        client_filter = request.GET.get('client', '')
        status_filter = request.GET.get('status', '')

        # Start with all documents
        queryset = Document.objects.all()

        # Apply client filter if provided
        if client_filter:
            queryset = queryset.filter(client_name__icontains=client_filter)

        # Apply status filter if provided
        if status_filter:
            queryset = queryset.filter(status=status_filter)

        # Serialize the queryset
        serializer = DocumentListSerializer(queryset, many=True)

        # Return serialized data and count
        return Response({
            'documents': serializer.data,
            'total_count': queryset.count()
        })

# Get specific document details
class DocumentDetailView(APIView):
    def get(self, request, document_id):
        try:
            # Try to retrieve the document by ID
            document = Document.objects.get(id=document_id)

            # Serialize and return the document
            serializer = DocumentListSerializer(document)
            return Response(serializer.data)

        except Document.DoesNotExist:
            # Return 404 if document is not found
            return Response(
                {'error': 'Document not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )