// base.js — ChatGPT-like sidebar behavior (persisted; brand opens-only; arrow closes)
(function () {
  'use strict';

  const STORAGE_KEY = 'sidebar:collapsed';
  const mSM = window.matchMedia('(min-width: 640px)');

  const $ = (id) => document.getElementById(id);
  const sidebar      = $('logo-sidebar');
  const pageContent  = $('pageContent');
  const brandToggle  = $('brandToggle');
  const collapseBtn  = $('collapseSidebar');
  const userMenuBtn  = $('userMenuButton');
  const userDropdown = $('userDropdown');

  if (!sidebar) return;

  // ---------- helpers ----------
  const getStored = () => {
    try { return localStorage.getItem(STORAGE_KEY) === '1'; } catch (_) { return false; }
  };
  const setStored = (collapsed) => {
    try { localStorage.setItem(STORAGE_KEY, collapsed ? '1' : '0'); } catch (_) {}
    document.documentElement.classList.toggle('sidebar-collapsed', collapsed);
  };

  const applyDesktop = (collapsed) => {
    sidebar.classList.toggle('collapsed', collapsed);
    pageContent && pageContent.classList.toggle('collapsed', collapsed);
    brandToggle && brandToggle.setAttribute('aria-pressed', String(!collapsed));
    collapseBtn && collapseBtn.setAttribute('aria-expanded', String(!collapsed));
    // ensure on-canvas for desktop
    sidebar.classList.remove('-translate-x-full', 'translate-x-0');
    document.body.classList.remove('overflow-hidden');
  };

  const openMobile = () => {
    sidebar.classList.remove('-translate-x-full');
    sidebar.classList.add('translate-x-0');
    document.body.classList.add('overflow-hidden');
  };
  const closeMobile = () => {
    sidebar.classList.add('-translate-x-full');
    sidebar.classList.remove('translate-x-0');
    document.body.classList.remove('overflow-hidden');
  };

  const showDropdown = () => { userDropdown?.classList.remove('hidden'); };
  const hideDropdown = () => { userDropdown?.classList.add('hidden'); };

  // ---------- initial state ----------
  const initialCollapsed =
    document.documentElement.classList.contains('sidebar-collapsed') || getStored();

  if (mSM.matches) {
    applyDesktop(initialCollapsed);
  } else {
    closeMobile(); // mobile starts closed
  }
  // persist the resolved initial state so every page keeps it
  setStored(initialCollapsed);

  // keep layout strict on bfcache return / forward-back nav
  window.addEventListener('pageshow', (e) => {
    if (e.persisted) {
      if (mSM.matches) {
        applyDesktop(getStored());
      } else {
        closeMobile();
        hideDropdown();
      }
    }
  });

  // Keep consistent when crossing the sm breakpoint
  mSM.addEventListener('change', (e) => {
    if (e.matches) {
      // transitioning to desktop: apply stored state
      applyDesktop(getStored());
      hideDropdown();
    } else {
      // transitioning to mobile: always start closed
      closeMobile();
      hideDropdown();
    }
  });

  // ---------- user dropdown ----------
  if (userMenuBtn && userDropdown) {
    userMenuBtn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();

      if (mSM.matches) {
        // DESKTOP: if collapsed, expand first (this *changes* state to open intentionally)
        if (sidebar.classList.contains('collapsed')) {
          applyDesktop(false);
          setStored(false);      // persist OPEN
          requestAnimationFrame(showDropdown);
        } else {
          userDropdown.classList.toggle('hidden');
        }
      } else {
        // MOBILE
        if (sidebar.classList.contains('-translate-x-full')) {
          openMobile();
          requestAnimationFrame(() => requestAnimationFrame(showDropdown));
        } else {
          userDropdown.classList.toggle('hidden');
        }
      }
    });

    document.addEventListener('click', (e) => {
      if (!userMenuBtn.contains(e.target) && !userDropdown.contains(e.target)) {
        hideDropdown();
      }
    }, { passive: true });
  }

  // ---------- controls ----------
  // Arrow CLOSES only (desktop collapse, mobile drawer close) and persists CLOSED
  collapseBtn && collapseBtn.addEventListener('click', () => {
    if (mSM.matches) {
      if (!sidebar.classList.contains('collapsed')) {
        applyDesktop(true);
        setStored(true);   // persist CLOSED
      }
    } else {
      closeMobile();
    }
    hideDropdown();
  });

  // Brand logo: **OPEN ONLY** (never closes)
  brandToggle && brandToggle.addEventListener('click', () => {
    if (mSM.matches) {
      if (sidebar.classList.contains('collapsed')) {
        applyDesktop(false);
        setStored(false);  // persist OPEN
      }
      // if already open, do nothing
    } else {
      if (sidebar.classList.contains('-translate-x-full')) {
        openMobile();
      }
    }
    hideDropdown();
  });

  // No keyboard toggle (e.g., Ctrl+B) — keeps state strictly user-controlled via logo/arrow
})();

function deleteConversation(conversationId) {
      if (confirm('Are you sure you want to delete this conversation?')) {
        fetch(`/chat/delete/${conversationId}/`, {
          method: 'POST',
          headers: {
            'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
            'Content-Type': 'application/json',
          },
        })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            location.reload(); // Reload to update the sidebar
          } else {
            alert('Error deleting conversation');
          }
        })
        .catch(error => {
          console.error('Error:', error);
          alert('Error deleting conversation');
        });
      }
    }
let currentConversationId = null;
    let currentConversationTitle = null;

    // Get CSRF token
    function getCSRFToken() {
        return document.querySelector('meta[name="csrf-token"]').getAttribute('content');
    }

    // Context menu functions
    function showContextMenu(event, element) {
        event.preventDefault();
        event.stopPropagation();
        
        currentConversationId = element.dataset.conversationId;
        currentConversationTitle = element.dataset.conversationTitle;
        
        console.log('Opening context menu for:', currentConversationId, currentConversationTitle);
        
        const contextMenu = document.getElementById('chatContextMenu');
        contextMenu.classList.remove('hidden');
        
        // Position the context menu near the button that was clicked
        const buttonRect = event.target.closest('button').getBoundingClientRect();
        contextMenu.style.left = (buttonRect.right - contextMenu.offsetWidth) + 'px';
        contextMenu.style.top = (buttonRect.bottom + 5) + 'px';
        
        // Hide context menu when clicking elsewhere
        setTimeout(() => {
            document.addEventListener('click', hideContextMenu);
        }, 10);
    }

    function hideContextMenu() {
        const contextMenu = document.getElementById('chatContextMenu');
        contextMenu.classList.add('hidden');
        document.removeEventListener('click', hideContextMenu);
    }

    // Rename functions
    function renameChatItem() {
        hideContextMenu();
        document.getElementById('renameInput').value = currentConversationTitle;
        document.getElementById('renameModal').classList.remove('hidden');
    }

    function cancelRename() {
        document.getElementById('renameModal').classList.add('hidden');
    }

    function confirmRename() {
        const newTitle = document.getElementById('renameInput').value.trim();
        if (newTitle && newTitle !== currentConversationTitle) {
            // Send rename request
            fetch(`/chat/rename/${currentConversationId}/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCSRFToken(),
                },
                body: JSON.stringify({ title: newTitle })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    location.reload(); // Refresh to show updated title
                } else {
                    alert('Error renaming conversation: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error renaming conversation');
            });
        }
        document.getElementById('renameModal').classList.add('hidden');
    }

    // Delete functions
    function deleteChatItem() {
        console.log('Delete clicked for conversation:', currentConversationId);
        hideContextMenu();
        document.getElementById('deleteConfirmModal').classList.remove('hidden');
    }

    function cancelDelete() {
        document.getElementById('deleteConfirmModal').classList.add('hidden');
    }

    function confirmDelete() {
        console.log('Confirming delete for conversation:', currentConversationId);
        if (!currentConversationId) {
            alert('No conversation selected for deletion');
            return;
        }
        
        // Send delete request
        fetch(`/chat/delete/${currentConversationId}/`, {
            method: 'DELETE',
            headers: {
                'X-CSRFToken': getCSRFToken(),
            }
        })
        .then(response => {
            console.log('Delete response status:', response.status);
            return response.json();
        })
        .then(data => {
            console.log('Delete response data:', data);
            if (data.success) {
                location.reload(); // Refresh to remove deleted item
            } else {
                alert('Error deleting conversation: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error deleting conversation');
        });
        document.getElementById('deleteConfirmModal').classList.add('hidden');
    }

    // Legacy delete function (if still needed)
    function deleteConversation(conversationId) {
        currentConversationId = conversationId;
        document.getElementById('deleteConfirmModal').classList.remove('hidden');
    }

    // Debug: Check if conversations are loaded
    document.addEventListener('DOMContentLoaded', function() {
        const historyItems = document.querySelectorAll('.chat-history-item');
        console.log('Found', historyItems.length, 'chat history items');
    });
