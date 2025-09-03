# core/validators.py
import re
from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _

class ComplexityValidator:
    """
    Validate that the password contains at least:
    - 1 uppercase letter
    - 1 lowercase letter  
    - 1 digit
    - 1 special character
    """
    
    def validate(self, password, user=None):
        errors = []
        
        # Check for uppercase letter
        if not re.search(r'[A-Z]', password):
            errors.append(_("Password must contain at least one uppercase letter."))
        
        # Check for lowercase letter
        if not re.search(r'[a-z]', password):
            errors.append(_("Password must contain at least one lowercase letter."))
        
        # Check for digit
        if not re.search(r'\d', password):
            errors.append(_("Password must contain at least one number."))
        
        # Check for special character
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append(_("Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)."))
        
        if errors:
            raise ValidationError(errors)
    
    def get_help_text(self):
        return _(
            "Your password must contain at least one uppercase letter, "
            "one lowercase letter, one number, and one special character."
        )