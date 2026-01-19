import os
import sys
from pathlib import Path

# Ensure project root and app are on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "webocr"))

# Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webocr.settings")

from django.core.wsgi import get_wsgi_application  # noqa: E402
from vercel_wsgi import make_lambda_handler  # noqa: E402

application = get_wsgi_application()
handler = make_lambda_handler(application)
