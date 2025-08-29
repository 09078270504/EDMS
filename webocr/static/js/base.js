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
  
})();

// Chat and UI functionality
(function() {
  'use strict'; 

  let currentConversationId = null;
  let currentConversationTitle = null;

  // Get CSRF token
  function getCSRFToken() {
    const token = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') ||
                  document.querySelector('[name=csrfmiddlewaretoken]')?.value;
    return token;
  }

  // Context menu functions
  function showContextMenu(event, element) {
    event.preventDefault();
    event.stopPropagation();
    
    currentConversationId = element.dataset.conversationId;
    currentConversationTitle = element.dataset.conversationTitle;
    
    console.log('Opening context menu for:', currentConversationId, currentConversationTitle);
    
    const contextMenu = document.getElementById('chatContextMenu');
    if (contextMenu) {
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
  }

  function hideContextMenu() {
    const contextMenu = document.getElementById('chatContextMenu');
    if (contextMenu) {
      contextMenu.classList.add('hidden');
    }
    document.removeEventListener('click', hideContextMenu);
  }

  // Rename functions
  function renameChatItem() {
    hideContextMenu();
    const renameInput = document.getElementById('renameInput');
    const renameModal = document.getElementById('renameModal');
    if (renameInput && renameModal) {
      renameInput.value = currentConversationTitle;
      renameModal.classList.remove('hidden');
    }
  }

  function cancelRename() {
    const renameModal = document.getElementById('renameModal');
    if (renameModal) {
      renameModal.classList.add('hidden');
    }
  }

  function confirmRename() {
    const renameInput = document.getElementById('renameInput');
    const renameModal = document.getElementById('renameModal');
    
    if (renameInput) {
      const newTitle = renameInput.value.trim();
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
    }
    
    if (renameModal) {
      renameModal.classList.add('hidden');
    }
  }

  // Delete functions
  function deleteChatItem() {
    console.log('Delete clicked for conversation:', currentConversationId);
    hideContextMenu();
    const deleteModal = document.getElementById('deleteConfirmModal');
    if (deleteModal) {
      deleteModal.classList.remove('hidden');
    }
  }

  function cancelDelete() {
    const deleteModal = document.getElementById('deleteConfirmModal');
    if (deleteModal) {
      deleteModal.classList.add('hidden');
    }
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
    
    const deleteModal = document.getElementById('deleteConfirmModal');
    if (deleteModal) {
      deleteModal.classList.add('hidden');
    }
  }

  // Legacy delete function (if still needed)
  function deleteConversation(conversationId) {
    if (confirm('Are you sure you want to delete this conversation?')) {
      fetch(`/chat/delete/${conversationId}/`, {
        method: 'POST',
        headers: {
          'X-CSRFToken': getCSRFToken(),
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

  // Initialize chat/search panels
  function initializePanels() {
    const chatPanel = document.getElementById("chatPanel");
    const searchPanel = document.getElementById("searchPanel");
    const chatButton = document.getElementById("chatButton");
    const searchButton = document.getElementById("searchButton");

    if (chatPanel) {
      // Show Chat panel by default
      chatPanel.classList.remove("hidden");
    }

    function setActiveButton(activeBtn, inactiveBtn) {
      // Active button styles (green background, white text)
      activeBtn.classList.remove('text-gray-600', 'bg-transparent');
      activeBtn.classList.add('text-white', 'bg-green-600');
      
      // Inactive button styles (gray text, transparent background)
      inactiveBtn.classList.remove('text-white', 'bg-green-600');
      inactiveBtn.classList.add('text-gray-600', 'bg-transparent');
    }

    if (chatButton) {
      // Add event listeners for Chat/Search buttons
      chatButton.addEventListener("click", function() {
        // Show Chat, hide Search
        if (searchPanel) searchPanel.classList.add("hidden");
        if (chatPanel) chatPanel.classList.remove("hidden");

        // Set Chat as active, Search as inactive
        setActiveButton(chatButton, searchButton);
      });
    }

    if (searchButton) {
      searchButton.addEventListener("click", function() {
        // Show Search, hide Chat
        if (chatPanel) chatPanel.classList.add("hidden");
        if (searchPanel) searchPanel.classList.remove("hidden");

        // Set Search as active, Chat as inactive
        setActiveButton(searchButton, chatButton);
      });
    }
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializePanels);
  } else {
    initializePanels();
  }

  // Make functions globally available
  window.showContextMenu = showContextMenu;
  window.hideContextMenu = hideContextMenu;
  window.renameChatItem = renameChatItem;
  window.cancelRename = cancelRename;
  window.confirmRename = confirmRename;
  window.deleteChatItem = deleteChatItem;
  window.cancelDelete = cancelDelete;
  window.confirmDelete = confirmDelete;
  window.deleteConversation = deleteConversation;

})();
