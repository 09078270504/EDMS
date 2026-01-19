#!/bin/bash
# Vercel build script to collect static files

echo "Collecting static files..."
python webocr/manage.py collectstatic --noinput
