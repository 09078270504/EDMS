// change_password.js
(function () {
  'use strict';

  // Password visibility toggle
  document.querySelectorAll('.toggle-password').forEach(button => {
    button.addEventListener('click', function () {
      const input = this.parentElement.querySelector('input');
      const icon  = this.querySelector('svg');

      if (!input || !icon) return;

      if (input.type === 'password') {
        input.type = 'text';
        icon.innerHTML = `
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
        `;
      } else {
        input.type = 'password';
        icon.innerHTML = `
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
        `;
      }
    });
  });

  // Basic password strength checks
  function validatePassword(password) {
    const errors = [];
    if (password.length < 12) errors.push('Password must be at least 12 characters');
    if (!/[A-Z]/.test(password)) errors.push('Password must contain at least one uppercase letter');
    if (!/[a-z]/.test(password)) errors.push('Password must contain at least one lowercase letter');
    if (!/\d/.test(password)) errors.push('Password must contain at least one number');
    if (!/[!@#$%^&*(),.?":{}|<>]/.test(password)) errors.push('Password must contain at least one special character');
    if (/^\d+$/.test(password)) errors.push('Password cannot be entirely numeric');
    const common = ['password','12345678','qwerty','password123','qwertyui'];
    if (common.includes(password.toLowerCase())) errors.push('This password is too common');
    return errors;
  }

  function showFeedback(id, msg, isError) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = msg || '';
    el.className = 'mt-1 text-sm ' + (isError ? 'text-red-600' : 'text-green-600');
    if (!msg) el.classList.add('hidden'); else el.classList.remove('hidden');
  }

  const p1 = document.getElementById('new_password1');
  const p2 = document.getElementById('new_password2');

  p1?.addEventListener('input', function () {
    const val = this.value || '';
    if (!val) return showFeedback('password1-feedback', '', false);
    const errs = validatePassword(val);
    if (errs.length) showFeedback('password1-feedback', errs[0], true);
    else showFeedback('password1-feedback', 'Password meets requirements ✓', false);
  });

  p2?.addEventListener('input', function () {
    const v1 = p1?.value || '';
    const v2 = this.value || '';
    if (!v2) return showFeedback('password2-feedback', '', false);
    showFeedback('password2-feedback', v1 === v2 ? 'Passwords match ✓' : 'Passwords do not match', v1 !== v2);
  });
})();
