import sys
import importlib

print("=" * 60)
print("VERIFICATION OF ALL PACKAGES")
print("=" * 60)

packages = [
    ('Django', 'django'),
    ('DRF', 'rest_framework'),
    ('PyMySQL', 'pymysql'),
    ('PyTorch', 'torch'),
    ('Transformers', 'transformers'),
    ('EasyOCR', 'easyocr'),
    ('OpenCV', 'cv2'),
    ('OpenAI', 'openai'),
    ('PyMuPDF', 'fitz'),
    ('spaCy', 'spacy'),
    ('Pandas', 'pandas'),
    ('NumPy', 'numpy'),
    ('Pillow', 'PIL'),
    ('scikit-image', 'skimage'),
]

all_ok = True
for display_name, import_name in packages:
    try:
        if import_name == 'fitz':
            import fitz
        elif import_name == 'cv2':
            import cv2
        elif import_name == 'PIL':
            from PIL import Image
        else:
            importlib.import_module(import_name)
        print(f"✓ {display_name:15} - OK")
    except Exception as e:
        print(f"✗ {display_name:15} - FAILED: {str(e)[:50]}...")
        all_ok = False

print("=" * 60)
if all_ok:
    print("SUCCESS: All packages installed correctly!")
else:
    print("WARNING: Some packages failed to import")

# GPU check
try:
    import torch
    print(f"\nGPU Status:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except:
    print("\nGPU Status: Could not check")