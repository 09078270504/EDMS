(function() {
    'use strict';
    
    let timeLeft = 300; // Default 5 minutes
    let lockoutExpired = false;
    
    try {
        // These will be replaced by Django template variables when properly configured
        timeLeft = parseInt("{{ remaining_seconds|default:300 }}") || 300;
        lockoutExpired = "{{ lockout_expired|yesno:'true,false' }}" === "true";
    } catch (e) {
        console.log("Using default timer values");
    }
    
    function updateCountdown() {
        const minutes = Math.floor(timeLeft / 60);
        const seconds = timeLeft % 60;
        
        document.getElementById('countdown').textContent = 
            `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        
        if (timeLeft > 0) {
            timeLeft--;
        } else {
            handleLockoutExpired();
        }
    }
    
    function handleLockoutExpired() {
        document.getElementById('countdown').textContent = "00:00";
        document.querySelector('.countdown-text').textContent = "You can now try logging in again";
        document.querySelector('.countdown').style.background = "#e8f5e9";
        document.querySelector('.countdown').style.borderColor = "#4caf50";
        
        document.getElementById('tryAgainBtn').style.display = 'inline-block';
        document.querySelector('a[href="/password-reset/"]').style.display = 'none';
        document.querySelector('.message').innerHTML = 
            'The lockout period has ended. You can now attempt to log in again.';
            
        document.querySelector('.countdown-timer').style.color = '#4caf50';
    }
    
    function initializeTimer() {
        if (lockoutExpired) {
            timeLeft = 0;
            updateCountdown();
        } else {
            setInterval(updateCountdown, 1000);
            updateCountdown();
        }
    }
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeTimer);
    } else {
        initializeTimer();
    }
})();