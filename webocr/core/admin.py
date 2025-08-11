# Register your models here.
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User  # Import your custom user model

# Register your models here.
class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'is_active', 'last_login')
    list_filter = ('is_active', 'last_login')

admin.site.register(User, CustomUserAdmin)  # Register your user model