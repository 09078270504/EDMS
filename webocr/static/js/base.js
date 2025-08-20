document.addEventListener('DOMContentLoaded', function () {
  const userMenuButton = document.getElementById('userMenuButton');
  const userDropdown   = document.getElementById('userDropdown');
  const sidebar        = document.getElementById('logo-sidebar');
  const brandToggle    = document.getElementById('brandToggle');
  const collapseBtn    = document.getElementById('collapseSidebar');
  const pageContent    = document.getElementById('pageContent');

  // User dropdown toggle
  if (userMenuButton && userDropdown) {
    userMenuButton.addEventListener('click', function (e) {
      e.stopPropagation();
      userDropdown.classList.toggle('hidden');
      const caret = userMenuButton.querySelector('svg');
      caret?.classList.toggle('transform');
      caret?.classList.toggle('rotate-180');
    });
    document.addEventListener('click', function (e) {
      if (!userMenuButton.contains(e.target) && !userDropdown.contains(e.target)) {
        userDropdown.classList.add('hidden');
        const caret = userMenuButton.querySelector('svg');
        caret?.classList.remove('rotate-180');
      }
    });
  }

  // Ensure sidebar visible on ≥sm screens and content aligned
  if (sidebar && window.matchMedia('(min-width: 640px)').matches) {
    sidebar.classList.remove('-translate-x-full');
    pageContent.classList.toggle('collapsed', sidebar.classList.contains('collapsed'));
  }

  function syncContentShift() {
    const collapsed = sidebar.classList.contains('collapsed');
    pageContent.classList.toggle('collapsed', collapsed);
    brandToggle?.setAttribute('aria-pressed', String(collapsed));
  }

  // Collapse via arrow (only closes)
  collapseBtn && collapseBtn.addEventListener('click', () => {
    if (!sidebar.classList.contains('collapsed')) {
      sidebar.classList.add('collapsed');
      userDropdown?.classList.add('hidden');
      const caret = userMenuButton?.querySelector('svg');
      caret?.classList.remove('rotate-180');
      syncContentShift();
    }
  });

  // Logo toggles (open/close)
  brandToggle && brandToggle.addEventListener('click', () => {
    if (sidebar.classList.contains('collapsed')) {
      sidebar.classList.remove('collapsed');
      syncContentShift();
    }
  });

  // Optional: Ctrl+B toggle
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key.toLowerCase() === 'b') {
      e.preventDefault();
      sidebar.classList.toggle('collapsed');
      userDropdown?.classList.add('hidden');
      const caret = userMenuButton?.querySelector('svg');
      caret?.classList.remove('rotate-180');
      syncContentShift();
    }
  });
});
