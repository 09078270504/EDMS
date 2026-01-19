import os
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.db import OperationalError


class Command(BaseCommand):
    help = "Create a default superuser from env vars if it doesn't exist"

    def handle(self, *args, **options):
        username = os.environ.get('DJANGO_SUPERUSER_USERNAME')
        email = os.environ.get('DJANGO_SUPERUSER_EMAIL')
        password = os.environ.get('DJANGO_SUPERUSER_PASSWORD')

        if not username or not email or not password:
            self.stdout.write(self.style.WARNING(
                'Skipping superuser creation: DJANGO_SUPERUSER_* env vars not all set'
            ))
            return

        User = get_user_model()
        try:
            if User.objects.filter(username=username).exists():
                self.stdout.write(self.style.SUCCESS(
                    f'Superuser "{username}" already exists. Skipping.'
                ))
                return

            User.objects.create_superuser(username=username, email=email, password=password)
            self.stdout.write(self.style.SUCCESS(
                f'Superuser "{username}" created successfully.'
            ))
        except OperationalError as e:
            self.stdout.write(self.style.ERROR(
                f'Database error while creating superuser: {e}'
            ))
        except Exception as e:
            self.stdout.write(self.style.ERROR(
                f'Unexpected error while creating superuser: {e}'
            ))