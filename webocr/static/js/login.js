document.addEventListener('DOMContentLoaded', function() {
    const toggleButton = document.getElementById('toggle-password');
    
    if (toggleButton) {
        toggleButton.addEventListener('click', togglePassword);
        // Initialize with password hidden
        initializePasswordToggle();
    }
});

function initializePasswordToggle() {
    const passwordInput = document.getElementById('id_password');
    const eyeOpen = document.getElementById('eye-open');
    const eyeClosed = document.getElementById('eye-closed');
    
    if (passwordInput && eyeOpen && eyeClosed) {
        passwordInput.type = 'password';
        setEyeIcon(false);
    }
}

function togglePassword() {
    const passwordInput = document.getElementById('id_password');
    
    if (!passwordInput) return;
    
    const isPasswordVisible = passwordInput.type === 'text';
    passwordInput.type = isPasswordVisible ? 'password' : 'text';
    setEyeIcon(!isPasswordVisible);
    
    // Focus back to password input after toggle
    passwordInput.focus();
}

function setEyeIcon(isVisible) {
    const eyeOpen = document.getElementById('eye-open');
    const eyeClosed = document.getElementById('eye-closed');
    
    if (!eyeOpen || !eyeClosed) return;
    
    if (isVisible) {
        // Password is visible, show eye-closed icon
        eyeOpen.classList.add('hidden');
        eyeClosed.classList.remove('hidden');
    } else {
        // Password is hidden, show eye-open icon
        eyeOpen.classList.remove('hidden');
        eyeClosed.classList.add('hidden');
    }
}