// static/core/js/dashboard.js
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);

  class ChatInterface {
    constructor() {
      // Sections
      this.greetingSection     = $('greetingSection');
      this.conversationSection = $('conversationSection');
      this.fixedInputArea      = $('fixedInputArea');
      this.pageHeader          = $('pageHeader');

      // Initial form
      this.messageInput = $('messageInput');
      this.sendButton   = $('sendButton');
      this.chatForm     = $('chatForm');

      // Active chat
      this.messageInputActive = $('messageInputActive');
      this.sendButtonActive   = $('sendButtonActive');
      this.chatFormActive     = $('chatFormActive');
      this.messagesList       = $('messagesList');

      // Guard: only run on pages that have the chat UI
      if (!this.chatForm || !this.chatFormActive || !this.messagesList) return;

      this.isFirstMessage = true;
      this.isTyping = false;

      // Keep horizontal overflow off
      document.body.style.overflowX = 'hidden';

      this.bindEvents();
      this.updateSendButton(this.messageInput, this.sendButton);
      this.updateSendButton(this.messageInputActive, this.sendButtonActive);

      // Reflect fixed input height into CSS var used by layout
      this.updateInputHeightVar();
      window.addEventListener('resize', this.debounce(() => this.updateInputHeightVar(), 120), { passive: true });
    }

    // ========= utils =========
    debounce(fn, delay = 120) {
      let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn.apply(this, args), delay); };
    }

    updateInputHeightVar() {
      if (!this.fixedInputArea) return;
      const h = Math.ceil(this.fixedInputArea.getBoundingClientRect().height || 64);
      document.documentElement.style.setProperty('--input-h', `${h}px`);
    }

    autoResize(textarea) {
      textarea.style.height = 'auto';
      const newHeight = Math.min(textarea.scrollHeight, 128); // max-h-32
      textarea.style.height = `${newHeight}px`;
    }

    // ========= events =========
    bindEvents() {
      // Initial form submit
      this.chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        this.handleFirstMessage();
      });

      // Active chat submit
      this.chatFormActive.addEventListener('submit', (e) => {
        e.preventDefault();
        this.handleSendMessage();
      });

      // Initial input changes
      this.messageInput.addEventListener('input', () => {
        this.updateSendButton(this.messageInput, this.sendButton);
      });

      // Active input (enable/disable + autoresize + keep height var fresh)
      this.messageInputActive.addEventListener('input', () => {
        this.updateSendButton(this.messageInputActive, this.sendButtonActive);
        this.autoResize(this.messageInputActive);
        this.updateInputHeightVar();
      });

      // Enter sends (Shift+Enter = newline)
      this.messageInputActive.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          if (this.messageInputActive.value.trim() && !this.isTyping) {
            this.handleSendMessage();
          }
        }
      });

      // Header fade (only matters before the first message)
      window.addEventListener('scroll', this.debounce(() => this.handleScrollEffects(), 16), { passive: true });
    }

    handleScrollEffects() {
      if (this.isFirstMessage || !this.pageHeader) return;
      const y = window.scrollY;
      const h = window.innerHeight;
      const fadeStart = h * 0.3;
      const fadeEnd   = h * 0.8;

      if (y <= fadeStart) {
        this.pageHeader.style.opacity = '1';
        this.pageHeader.style.transform = 'translateY(0)';
      } else if (y >= fadeEnd) {
        this.pageHeader.style.opacity = '0.3';
        this.pageHeader.style.transform = 'translateY(-10px)';
      } else {
        const t = (y - fadeStart) / (fadeEnd - fadeStart);
        this.pageHeader.style.opacity = String(1 - t * 0.7);
        this.pageHeader.style.transform = `translateY(${-10 * t}px)`;
      }
    }

    updateSendButton(input, button) {
      const hasText = input.value.trim().length > 0;
      const enable = hasText && !this.isTyping;
      button.disabled = !enable;
      button.classList.toggle('bg-green-600', enable);
      button.classList.toggle('bg-gray-300', !enable);
    }

    // ========= flows =========
    handleFirstMessage() {
      const msg = this.messageInput.value.trim();
      if (!msg || this.isTyping) return;

      // Reveal conversation UI, hide hero
      this.conversationSection.classList.remove('hidden');
      this.fixedInputArea.classList.remove('hidden');
      this.greetingSection?.classList.add('hidden');
      this.pageHeader.classList.add('header-fade');
      this.updateInputHeightVar();

      // Seed first message
      this.addMessage(msg, 'user');

      // Focus active input soon
      setTimeout(() => this.messageInputActive?.focus(), 40);

      // Typing + simulated reply
      this.showTypingIndicator();
      this.simulateAIResponse();

      this.isFirstMessage = false;
      this.updateSendButton(this.messageInputActive, this.sendButtonActive);
    }

    async handleSendMessage() {
      const msg = this.messageInputActive.value.trim();
      if (!msg || this.isTyping) return;

      this.addMessage(msg, 'user');

      // Reset input box
      this.messageInputActive.value = '';
      this.messageInputActive.style.height = 'auto';
      this.updateSendButton(this.messageInputActive, this.sendButtonActive);

      // Typing + response
      this.showTypingIndicator();
      this.scrollToLatestMessage();
      await this.simulateAIResponse();
    }

    addMessage(text, type = 'user') {
      const row = document.createElement('div');
      row.className = `message-appear ${type === 'user' ? 'flex justify-end' : 'flex justify-start'}`;

      const bubble = document.createElement('div');
      bubble.className = (type === 'user')
        ? 'bg-green-600 text-white px-4 py-3 rounded-2xl max-w-xs lg:max-w-2xl break-words whitespace-pre-wrap'
        : 'bg-gray-100 border border-gray-300 text-gray-800 px-4 py-3 rounded-2xl max-w-xs lg:max-w-2xl shadow-sm break-words whitespace-pre-wrap';

      bubble.textContent = text;
      row.appendChild(bubble);
      this.messagesList.appendChild(row);

      this.scrollToLatestMessage();
    }

    showTypingIndicator() {
      this.isTyping = true;
      this.updateSendButton(this.messageInputActive, this.sendButtonActive);

      const wrapper = document.createElement('div');
      wrapper.id = 'typingIndicator';
      wrapper.className = 'flex justify-start message-appear';

      const bubble = document.createElement('div');
      bubble.className = 'bg-gray-100 border border-gray-300 px-4 py-3 rounded-2xl shadow-sm typing-indicator';

      const dots = document.createElement('div');
      dots.className = 'typing-dots flex space-x-1';
      for (let i = 0; i < 3; i++) {
        const dot = document.createElement('span');
        dot.className = 'w-2 h-2 bg-gray-400 rounded-full inline-block';
        dots.appendChild(dot);
      }

      bubble.appendChild(dots);
      wrapper.appendChild(bubble);
      this.messagesList.appendChild(wrapper);
      this.scrollToLatestMessage();
    }

    hideTypingIndicator() {
      const t = $('typingIndicator');
      if (t) t.remove();
      this.isTyping = false;
      this.updateSendButton(this.messageInputActive, this.sendButtonActive);
    }

    async simulateAIResponse() {
      await new Promise(r => setTimeout(r, 1200 + Math.random() * 1200));
      this.hideTypingIndicator();

      const responses = [
        'Yoko nga bala ka dyan',
        'Ganto muna placeholder since wala pa tayu bot',
        'Bla bla bla',
      ];
      const text = responses[Math.floor(Math.random() * responses.length)];
      this.addMessage(text, 'assistant');
      this.scrollToLatestMessage();
    }

    // Only the messages list scrolls (no whole-page jump)
    scrollToLatestMessage() {
      requestAnimationFrame(() => {
        this.messagesList.scrollTo({
          top: this.messagesList.scrollHeight,
          behavior: 'smooth'
        });
      });
    }
  }

  document.addEventListener('DOMContentLoaded', () => new ChatInterface());
})();
// Handle chat form submissions
document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const chatFormActive = document.getElementById('chatFormActive');
    
    if (chatForm) {
        chatForm.addEventListener('submit', handleChatSubmit);
    }
    
    if (chatFormActive) {
        chatFormActive.addEventListener('submit', function(e) {
            handleChatSubmit(e);
            // Clear the active input after sending
            setTimeout(() => {
                document.getElementById('messageInputActive').value = '';
                document.getElementById('sendButtonActive').disabled = true;
                document.getElementById('sendButtonActive').classList.add('bg-gray-300');
                document.getElementById('sendButtonActive').classList.remove('bg-green-600');
            }, 100);
        });
    }
    
    // Enable/disable send buttons based on input
    const messageInput = document.getElementById('messageInput');
    const messageInputActive = document.getElementById('messageInputActive');
    const sendButton = document.getElementById('sendButton');
    const sendButtonActive = document.getElementById('sendButtonActive');
    
    if (messageInput && sendButton) {
        messageInput.addEventListener('input', function() {
            sendButton.disabled = !this.value.trim();
            sendButton.classList.toggle('bg-green-600', this.value.trim());
            sendButton.classList.toggle('bg-gray-300', !this.value.trim());
        });
        
        // Add Enter key functionality for initial input
        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (this.value.trim()) {
                    chatForm.dispatchEvent(new Event('submit'));
                }
            }
        });
    }
    
    if (messageInputActive && sendButtonActive) {
        messageInputActive.addEventListener('input', function() {
            sendButtonActive.disabled = !this.value.trim();
            sendButtonActive.classList.toggle('bg-green-600', this.value.trim());
            sendButtonActive.classList.toggle('bg-gray-300', !this.value.trim());
        });
        
        // Add Enter key functionality for active chat input
        messageInputActive.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (this.value.trim()) {
                    chatFormActive.dispatchEvent(new Event('submit'));
                }
            }
        });
    }
});

function handleChatSubmit(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const messageContent = formData.get('message').trim();
    
    if (!messageContent) return;
    
    // If this is the initial form (greeting screen), transition to conversation view
    const isInitialForm = e.target.id === 'chatForm';
    
    if (isInitialForm) {
        // Hide greeting section and show conversation section
        document.getElementById('greetingSection').style.display = 'none';
        document.getElementById('conversationSection').classList.remove('hidden');
        document.getElementById('fixedInputArea').classList.remove('hidden');
        
        // Add the user message to the conversation immediately
        addMessageToChat('user', messageContent);
        
        // Clear the initial input
        document.getElementById('messageInput').value = '';
        document.getElementById('sendButton').disabled = true;
        document.getElementById('sendButton').classList.add('bg-gray-300');
                        document.getElementById('sendButton').classList.remove('bg-green-600');
    }
    
    // Send message to server
    fetch('/chat/message/', {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': formData.get('csrfmiddlewaretoken'),
        },
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            if (isInitialForm) {
                // For new conversation, add assistant response and update conversation ID
                addMessageToChat('assistant', data.assistant_message.content);
                document.querySelector('input[name="conversation_id"]').value = data.conversation_id;
                
                // Update the page URL without reload
                window.history.pushState({}, '', `/chat/${data.conversation_id}/`);
                
                // If this is a new conversation, refresh the sidebar to show it
                if (data.is_new_conversation) {
                    updateSidebar(data.conversation_id);
                }
            } else {
                // For existing conversation, just reload to show new messages
                location.reload();
            }
        } else {
            alert('Error sending message: ' + (data.error || 'Unknown error'));
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error sending message');
    });
}

function addMessageToChat(messageType, content) {
    const messagesList = document.getElementById('messagesList');
    const messageContainer = document.createElement('div');
    messageContainer.className = `message-container ${messageType}`;
    
    const currentTime = new Date().toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: false 
    });
    
    messageContainer.innerHTML = `
        <div class="message-bubble">
            <div class="message-content">${content}</div>
            <div class="message-time">${currentTime}</div>
        </div>
    `;
    
    messagesList.appendChild(messageContainer);
    
    // Scroll to bottom
    messageContainer.scrollIntoView({ behavior: 'smooth' });
}

function updateSidebar(conversationId) {
    // Reload the entire page to refresh the sidebar with new conversation
    // In a more sophisticated implementation, you could make an AJAX call to update just the sidebar
    setTimeout(() => {
        location.reload();
    }, 1000); // Give a small delay to ensure the conversation is saved
}

function handleChatSubmit(e) {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);
    const messageContent = (formData.get('message') || '').trim();
    if (!messageContent) return;

    const isInitialForm = form.id === 'chatForm';

    if (isInitialForm) {
        const greeting = document.getElementById('greetingSection');
        const convo = document.getElementById('conversationSection');
        const fixed = document.getElementById('fixedInputArea');

        // 1) Fade OUT the greeting
        greeting.classList.add('fade-out-up');

        const onFadeOutEnd = () => {
            // Hide greeting after animation
            greeting.style.display = 'none';
            greeting.classList.remove('fade-out-up');
            greeting.removeEventListener('animationend', onFadeOutEnd);

            // 2) Reveal chat UI and fade IN
            convo.classList.remove('hidden');
            fixed.classList.remove('hidden');

            // Start with a tiny visual guard to ensure layout is on
            // before animating (avoids rare flicker)
            requestAnimationFrame(() => {
                convo.classList.add('appearing', 'fade-in-up');
                fixed.classList.add('appearing', 'fade-in-up');

                const cleanup = (el) => {
                    el.classList.remove('appearing', 'fade-in-up');
                };
                // Remove classes after the fade-in completes
                convo.addEventListener('animationend', () => cleanup(convo), { once: true });
                fixed.addEventListener('animationend', () => cleanup(fixed), { once: true });
            });

            // Add the user's first message (feels instant)
            addMessageToChat('user', messageContent);

            // Reset input/button on the hero form
            const msgInput = document.getElementById('messageInput');
            const sendBtn = document.getElementById('sendButton');
            if (msgInput) msgInput.value = '';
            if (sendBtn) {
                sendBtn.disabled = true;
                sendBtn.classList.add('bg-gray-300');
                sendBtn.classList.remove('bg-green-600');
            }
        };

        greeting.addEventListener('animationend', onFadeOutEnd, { once: true });
    }

    // Send to server as before
    fetch('/chat/message/', {
        method: 'POST',
        body: formData,
        headers: { 'X-CSRFToken': formData.get('csrfmiddlewaretoken') },
    })
        .then((r) => r.json())
        .then((data) => {
            if (data.success) {
                if (isInitialForm) {
                    addMessageToChat('assistant', data.assistant_message.content);
                    const hiddenConvoId = document.querySelector('input[name="conversation_id"]');
                    if (hiddenConvoId) hiddenConvoId.value = data.conversation_id;

                    window.history.pushState({}, '', `/chat/${data.conversation_id}/`);
                    if (data.is_new_conversation) updateSidebar(data.conversation_id);
                } else {
                    location.reload();
                }
            } else {
                alert('Error sending message: ' + (data.error || 'Unknown error'));
            }
        })
        .catch((err) => {
            console.error(err);
            alert('Error sending message');
        });
}
