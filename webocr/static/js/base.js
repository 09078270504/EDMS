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
