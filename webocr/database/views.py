import os, json
from pathlib import Path
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
        client_filter = request.GET.get('client', '')
        query = (request.GET.get('q') or '').strip()
        enable_fuzzy = os.environ.get('ENABLE_FUZZY_SEARCH', 'True').lower() == 'true'
        fuzzy_threshold = int(os.environ.get('FUZZY_SEARCH_THRESHOLD', '70'))
        fuzzy_limit = int(os.environ.get('FUZZY_SEARCH_LIMIT', '10'))
        fuzzy_max_text_len = int(os.environ.get('FUZZY_MAX_TEXT_LENGTH', '5000'))

        queryset = Document.objects.filter(
            status='completed',
            metadata_filename__isnull=False
        )
        if client_filter:
            queryset = queryset.filter(client_name__icontains=client_filter)
        queryset = queryset.exclude(metadata_filename='')

        # No query: keep backward compatible output
        if not query:
            ocr_files = [doc.ocr_filename for doc in queryset if doc.ocr_filename]
            return Response({
                'ocr_directory': '/webocr/ocr/',
                'files': ocr_files,
                'total_count': len(ocr_files)
            })

        # Search DB-backed documents first
        results = []
        archive_root = Path(getattr(settings, 'ARCHIVE_FOLDER', Path.cwd() / 'archive'))
        q_low = query.lower()

        for doc in queryset:
            if not doc.ocr_filename and not doc.metadata_filename:
                continue

            doc_folder = archive_root / doc.client_name / doc.document_name
            ocr_path = doc_folder / 'ocr' / (doc.ocr_filename or "")
            meta_path = doc_folder / 'metadata' / (doc.metadata_filename or "")

            # Read and search OCR
            for path, kind in ((ocr_path, 'ocr'), (meta_path, 'metadata')):
                if not path.exists():
                    continue
                try:
                    txt = path.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    continue

                txt_sample = txt[:fuzzy_max_text_len].lower()
                idx = txt_sample.find(q_low)
                if idx != -1:
                    start = max(0, idx - 80); end = min(len(txt_sample), idx + len(query) + 80)
                    snippet = txt[start:end].replace('\n', ' ')
                    results.append({
                        'client_name': doc.client_name,
                        'document_name': doc.document_name,
                        'ocr_filename': doc.ocr_filename,
                        'metadata_filename': doc.metadata_filename,
                        'match_type': 'exact',
                        'file_kind': kind,
                        'path': str(path),
                        'snippet': snippet,
                        'score': 100
                    })
                    break  # found in this doc
                # fuzzy fallback
                if enable_fuzzy:
                    score = fuzz.partial_ratio(query, txt_sample)
                    if score >= fuzzy_threshold:
                        snippet = txt[:300].replace('\n', ' ')
                        results.append({
                            'client_name': doc.client_name,
                            'document_name': doc.document_name,
                            'ocr_filename': doc.ocr_filename,
                            'metadata_filename': doc.metadata_filename,
                            'match_type': 'fuzzy',
                            'file_kind': kind,
                            'path': str(path),
                            'snippet': snippet,
                            'score': int(score)
                        })
                        break
            if len(results) >= fuzzy_limit:
                break

        # If no results from DB records, fallback to scanning archive for matching files
        if not results:
            for path in archive_root.rglob('*.txt'):
                try:
                    txt = path.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    continue
                txt_low = txt[:fuzzy_max_text_len].lower()
                if q_low in txt_low:
                    results.append({
                        'client_name': path.parts[-4] if len(path.parts) >= 4 else '',
                        'document_name': path.parts[-3] if len(path.parts) >= 3 else '',
                        'file_kind': 'ocr',
                        'path': str(path),
                        'match_type': 'exact',
                        'snippet': txt_low[max(0, txt_low.find(q_low)-80):txt_low.find(q_low)+len(q_low)+80].replace('\n',' '),
                        'score': 100
                    })
                elif enable_fuzzy:
                    score = fuzz.partial_ratio(query, txt_low)
                    if score >= fuzzy_threshold:
                        results.append({
                            'client_name': path.parts[-4] if len(path.parts) >= 4 else '',
                            'document_name': path.parts[-3] if len(path.parts) >= 3 else '',
                            'file_kind': 'ocr',
                            'path': str(path),
                            'match_type': 'fuzzy',
                            'snippet': txt_low[:300].replace('\n', ' '),
                            'score': int(score)
                        })
                if len(results) >= fuzzy_limit:
                    break

        results = sorted(results, key=lambda r: -int(r.get('score', 0)))
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