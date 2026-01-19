# WebOCR (Arkayb)
A Django-based document OCR, search, and chat assistant. It ingests files from the upload/ folder, extracts text, indexes it, and lets you search and chat with document context.

Features:
- OCR + indexing for PDFs/images
- Keyword and optional fuzzy search
- Document views: original preview + extracted text
- Chat assistant using top-matching document snippets
- Auth + admin; detailed logs in webocr/logs/search.log

Quick Start:
1) Create env: conda create -n webocr python=3.11 -y; conda activate webocr
2) Install deps: pip install -r requirements.txt
3) Configure webocr/.env: set DEBUG, SECRET_KEY, DATABASE_URL=sqlite:///db.sqlite3, ALLOWED_HOSTS, ENABLE_FUZZY_SEARCH=True, FUZZY_* values, OPENAI_API_KEY/GROQ_API_KEY if using chat
4) Migrate and run: cd webocr; python manage.py migrate; python manage.py createsuperuser; python manage.py runserver

Ingestion:
Place files in upload/ then run: cd webocr; python manage.py start

Key URLs:
- admin/
- login/, forgot-password/, logout/
- documents/ (search form)
- search/documents/ (results)
- document/<document_id>/, original/, ocr/
- documents/view/
- chat/, chat/<conversation_id>/, chat/message/

Troubleshooting:
- Use route URLs (e.g., documents/view/) not template filenames
- Close all template blocks with {% endblock %}
- Use SQLite locally if MySQL plugin errors occur
- If search is empty, ingest files and re-run start