"""Compatibility wrapper: process_documents -> start
This command provides backward compatibility for scripts that call
`python manage.py process_documents` by delegating to the `start` command
which is the active document processing command in this project.
"""
from .start import Command as StartCommand

# Reuse start.Command directly so all flags/behavior remain identical
class Command(StartCommand):
    help = 'Compatibility wrapper for `start` (process_documents -> start)'
