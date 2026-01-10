import os
from django.conf import settings
from rapidfuzz import fuzz

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
        query = (request.GET.get('q') or '').strip()
        enable_fuzzy = os.environ.get('ENABLE_FUZZY_SEARCH', 'True').lower() == 'true'
        fuzzy_threshold = int(os.environ.get('FUZZY_SEARCH_THRESHOLD', '70'))
        fuzzy_limit = int(os.environ.get('FUZZY_SEARCH_LIMIT', '10'))
        fuzzy_chunk_size = int(os.environ.get('FUZZY_CHUNK_SIZE', '200'))
        fuzzy_max_text_len = int(os.environ.get('FUZZY_MAX_TEXT_LENGTH', '5000'))

        # base queryset
        queryset = Document.objects.filter(
            status='completed',
            metadata_filename__isnull=False
        )

        if client_filter:
            queryset = queryset.filter(client_name__icontains=client_filter)

        queryset = queryset.exclude(metadata_filename='')

        # If no search query provided, return basic listing (backwards compatible)
        if not query:
            ocr_files = [doc.ocr_filename for doc in queryset if doc.ocr_filename]
            return Response({
                'ocr_directory': '/webocr/ocr/',
                'files': ocr_files,
                'total_count': len(ocr_files)
            })

        # Search OCR files for query
        results = []
        for doc in queryset:
            if not doc.ocr_filename:
                continue

            # Build archive OCR file path:
            # ARCHIVE_FOLDER/<client_name>/<document_name>/ocr/<ocr_filename>
            archive_root = Path(getattr(settings, 'ARCHIVE_FOLDER', Path.cwd() / 'archive'))
            ocr_path = archive_root / doc.client_name / doc.document_name / 'ocr' / doc.ocr_filename

            if not ocr_path.exists():
                # skip missing OCR files
                continue

            try:
                text = ocr_path.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                # skip files we cannot read
                continue

            text_sample = text[:fuzzy_max_text_len]  # cap for performance
            lower_text = text_sample.lower()
            q_lower = query.lower()

            # Exact substring match (fast)
            idx = lower_text.find(q_lower)
            if idx != -1:
                # Create a snippet around the match
                start = max(0, idx - 80)
                end = min(len(text_sample), idx + len(query) + 80)
                snippet = text_sample[start:end].replace('\n', ' ')
                results.append({
                    'client_name': doc.client_name,
                    'document_name': doc.document_name,
                    'ocr_filename': doc.ocr_filename,
                    'ocr_path': str(ocr_path),
                    'match_type': 'exact',
                    'snippet': snippet,
                    'score': 100
                })
                continue

            # Fuzzy fallback (optional)
            if enable_fuzzy:
                # Split into chunks of N words to avoid single large string comparisons
                words = text_sample.split()
                chunks = [' '.join(words[i:i+fuzzy_chunk_size]) for i in range(0, len(words), fuzzy_chunk_size)]
                best_score = 0
                best_snippet = None
                for chunk in chunks:
                    score = fuzz.partial_ratio(query, chunk)
                    if score > best_score:
                        best_score = score
                        # Pick a snippet centered near the fuzzy match
                        # naive: use start of chunk as snippet
                        best_snippet = chunk[:300].replace('\n', ' ')
                if best_score >= fuzzy_threshold:
                    results.append({
                        'client_name': doc.client_name,
                        'document_name': doc.document_name,
                        'ocr_filename': doc.ocr_filename,
                        'ocr_path': str(ocr_path),
                        'match_type': 'fuzzy',
                        'snippet': best_snippet or '',
                        'score': int(best_score)
                    })

            # Respect limit
            if len(results) >= fuzzy_limit:
                break

        # Sort by score (fuzzy first), then return
        results = sorted(results, key=lambda r: (-int(r.get('score', 0))))
        return Response({
            'query': query,
            'results': results,
            'total_count': len(results)
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