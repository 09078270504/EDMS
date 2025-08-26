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
