// --- START CONFIGURATION ---
const SIDEBAR_SELECTOR = '#logo-sidebar';
// --- END CONFIGURATION ---

document.addEventListener('DOMContentLoaded', function() {
    // --- Element Selectors ---
    const getEl = (id) => document.getElementById(id);
    const sidebar = document.querySelector(SIDEBAR_SELECTOR);
    const chatForm = getEl('chatForm');
    const chatFormActive = getEl('chatFormActive');
    const messageInput = getEl('messageInput');
    const sendButton = getEl('sendButton');
    const messageInputActive = getEl('messageInputActive');
    const sendButtonActive = getEl('sendButtonActive');
    const fixedInputArea = getEl('fixedInputArea');

    // --- Main Setup ---
    if (!chatForm && !chatFormActive) return;

    setupSidebarObserver(sidebar, fixedInputArea);
    setupInputListeners(messageInput, sendButton, chatForm);
    setupInputListeners(messageInputActive, sendButtonActive, chatFormActive);
});

/**
 * Watches the sidebar for size changes and adjusts the chatbox position.
 * @param {Element} sidebarEl The sidebar element to observe.
 * @param {Element} inputAreaEl The chatbox element to move.
 */
function setupSidebarObserver(sidebarEl, inputAreaEl) {
    if (!sidebarEl) {
        console.error('Sidebar element not found for selector:', SIDEBAR_SELECTOR, '. Text box positioning will not work.');
        return;
    }
    if (!inputAreaEl) return;

    const resizeObserver = new ResizeObserver(entries => {
        for (let entry of entries) {
            const sidebarWidth = entry.contentRect.width;
            inputAreaEl.style.left = `${sidebarWidth}px`;
        }
    });
    resizeObserver.observe(sidebarEl);
}

/**
 * Sets up all necessary event listeners for a chat input.
 * @param {HTMLInputElement|HTMLTextAreaElement} inputEl The text input element.
 * @param {HTMLButtonElement} buttonEl The send button element.
 * @param {HTMLFormElement} formEl The form element.
 */
function setupInputListeners(inputEl, buttonEl, formEl) {
    if (!inputEl || !buttonEl || !formEl) return;

    const inputContainer = inputEl.closest('.flex.items-center, .flex.items-end');

    formEl.addEventListener('submit', handleChatSubmit);

    inputEl.addEventListener('input', () => {
        updateSendButtonState(inputEl, buttonEl);
        if (inputEl.tagName === 'TEXTAREA') {
            inputEl.style.height = 'auto';
            inputEl.style.height = `${inputEl.scrollHeight}px`;
        }
    });

    inputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && inputEl.value.trim()) {
            e.preventDefault();
            formEl.dispatchEvent(new Event('submit'));
        }
    });

    if (inputContainer) {
        inputEl.addEventListener('focus', () => inputContainer.classList.add('input-focused'));
        inputEl.addEventListener('blur', () => inputContainer.classList.remove('input-focused'));
    }
}

/**
 * Centralized function to control the send button's state.
 * @param {HTMLInputElement|HTMLTextAreaElement} inputEl
 * @param {HTMLButtonElement} buttonEl
 */
function updateSendButtonState(inputEl, buttonEl) {
    if (!inputEl || !buttonEl) return;
    const hasText = inputEl.value.trim().length > 0;
    buttonEl.disabled = !hasText;
    buttonEl.classList.toggle('bg-green-600', hasText);
    buttonEl.classList.toggle('bg-gray-300', !hasText);
}


// --- Core Chat Functions ---

async function handleChatSubmit(e) {
    e.preventDefault();
    const form = e.target;
    const formData = new FormData(form);
    const messageContent = formData.get('message').trim();
    if (!messageContent) return;

    let inputElement, buttonElement;
    
    if (form.id === 'chatForm') {
        inputElement = document.getElementById('messageInput');
        buttonElement = document.getElementById('sendButton');
    } else if (form.id === 'chatFormActive') {
        inputElement = document.getElementById('messageInputActive');
        buttonElement = document.getElementById('sendButtonActive');
    } else {
        // Fallback to generic selector
        inputElement = form.querySelector('input[name="message"], textarea[name="message"]');
        buttonElement = form.querySelector('button[type="submit"]');
    }

    console.log('Form ID:', form.id);
    console.log('Input element found:', inputElement);
    console.log('Input element ID:', inputElement?.id);
    console.log('Input current value:', inputElement?.value);

    if (!inputElement || !buttonElement) {
        console.error('Could not find input or button elements');
        return;
    }

    const shouldClearInput = Boolean(inputElement && messageContent);

    if (form.id === 'chatForm') {
        const greetingSection = document.getElementById('greetingSection');
        if (greetingSection) {
            greetingSection.classList.add('transitioning-out');
            setTimeout(() => {
                greetingSection.style.display = 'none';
                document.getElementById('conversationSection')?.classList.remove('hidden');
                document.getElementById('fixedInputArea')?.classList.remove('hidden');
            }, 600);
        }
    }

    addMessageToChat('user', messageContent);
    showTypingIndicator();

    try {
        const csrfToken = getCSRFToken();
        console.log('CSRF Token retrieved:', csrfToken);
        
        if (!csrfToken) {
            throw new Error('No CSRF token found');
        }
        
        const response = await fetch('/chat/message/', {
            method: 'POST',
            body: formData,
            headers: { 
                'X-CSRFToken': csrfToken,
                'X-Requested-With': 'XMLHttpRequest'
            },
        });
        
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        const data = await response.json();
        console.log('Response data:', data);
        
        if (data.success) {
            console.log('Success condition met, clearing input');
            console.log('About to clear - Input element:', inputElement);
            console.log('About to clear - Input value before:', inputElement.value);
            
            if (shouldClearInput && inputElement) {
                // Clear the input
                inputElement.value = '';
                
               
                if (inputElement.tagName === 'TEXTAREA') {
                    inputElement.style.height = 'auto';
                }
                
         
                const inputContainer = inputElement.closest('.flex.items-center, .flex.items-end');
                if (inputContainer) {
                    
                    if (document.activeElement === inputElement) {
                        inputContainer.classList.add('input-focused');
                    } else {   
                        inputContainer.classList.remove('input-focused');
                    }
                }
                
                // Update button state
                updateSendButtonState(inputElement, buttonElement);
                
                // Force a re-check of the value
                console.log('Input value after clearing:', inputElement.value);
                console.log('Input cleared successfully');
                
                inputElement.dispatchEvent(new Event('input', { bubbles: true }));
            }
            
            addMessageToChat('assistant', data.assistant_message.content);
            if (form.id === 'chatForm') {
                const hiddenInput = document.querySelector('input[name="conversation_id"]');
                if (hiddenInput) {
                    hiddenInput.value = data.conversation_id;
                }
                window.history.pushState({}, '', `/chat/${data.conversation_id}/`);
                if (data.is_new_conversation) updateSidebar();
            }
        } else {
            console.log('Success condition NOT met, data:', data);
            addMessageToChat('assistant', 'Error: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error:', error);
        addMessageToChat('assistant', 'Sorry, an error occurred.');
    } finally {
        hideTypingIndicator();
    }
}

function addMessageToChat(messageType, content) {
    const messagesList = document.getElementById('messagesList');
    const messageContainer = document.createElement('div');
    messageContainer.className = `message-container ${messageType} message-appear`;
    const currentTime = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
    
    if (messageType === 'typing-indicator') {
        messageContainer.id = 'typingIndicator';
        messageContainer.innerHTML = `<div class="message-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>`;
    } else {
        messageContainer.innerHTML = `<div class="message-bubble"><div class="message-content">${content}</div><div class="message-time">${currentTime}</div></div>`;
    }
    
    messagesList.appendChild(messageContainer);
    scrollToBottom();
}

function showTypingIndicator() {
    if (document.getElementById('typingIndicator')) return;
    addMessageToChat('typing-indicator', '');
}

function hideTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) indicator.remove();
}

function scrollToBottom() {
    const messagesList = document.getElementById('messagesList');
    const lastMessage = messagesList.lastElementChild;
    if (lastMessage) {
        setTimeout(() => {
            lastMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 100);
    }
}

function updateSidebar() {
    setTimeout(() => { location.reload(); }, 1000);
}

function getCSRFToken() {
    const metaTag = document.querySelector('meta[name="csrf-token"]');
    if (metaTag) {
        const token = metaTag.getAttribute('content');
        console.log('CSRF token from meta tag:', token);
        return token;
    }
    
    console.log('Meta tag not found, trying cookie...');
    const cookieToken = getCookie('csrftoken');
    console.log('CSRF token from cookie:', cookieToken);
    return cookieToken;
}

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}