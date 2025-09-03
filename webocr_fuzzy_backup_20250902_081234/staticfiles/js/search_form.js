(function() {
    // Elements
    const searchInput = document.getElementById('search_query');
    const clearBtn = document.getElementById('clearBtn');
    const filterBtn = document.getElementById('filterBtn');
    const filterModal = document.getElementById('filterModal');
    const closeModalBtn = document.getElementById('closeModalBtn');
    const resetFiltersBtn = document.getElementById('resetFiltersBtn');
    const applyFiltersBtn = document.getElementById('applyFiltersBtn');
    const searchForm = document.getElementById('searchForm');
    const pageHeader = document.getElementById('pageHeader');
    
    // Filter form elements
    const clientFilter = document.getElementById('client_filter');
    const itemName = document.getElementById('item_name');
    const includesWords = document.getElementById('includes_words');
    const dateFrom = document.getElementById('date_from');
    const dateTo = document.getElementById('date_to');
    
    // Hidden form inputs
    const clientFilterHidden = document.getElementById('client_filter_hidden');
    const dateFromHidden = document.getElementById('date_from_hidden');
    const dateToHidden = document.getElementById('date_to_hidden');

    // Header fade effect on scroll
    function handleScrollFade() {
        const scrollY = window.scrollY;
        const fadeStart = 100; // Start fading after 100px
        const fadeEnd = 300;   // Completely faded at 300px
        
        if (scrollY <= fadeStart) {
            // Fully visible
            pageHeader.style.opacity = '1';
            pageHeader.style.transform = 'translateY(0)';
        } else if (scrollY >= fadeEnd) {
            // Fully faded
            pageHeader.style.opacity = '0';
            pageHeader.style.transform = 'translateY(-20px)';
        } else {
            // Gradually fade
            const fadeProgress = (scrollY - fadeStart) / (fadeEnd - fadeStart);
            const opacity = 1 - fadeProgress;
            const translateY = fadeProgress * -20;
            
            pageHeader.style.opacity = opacity;
            pageHeader.style.transform = `translateY(${translateY}px)`;
        }
    }

    // Throttle scroll events for better performance
    let scrollTimeout;
    function throttledScrollHandler() {
        if (!scrollTimeout) {
            scrollTimeout = setTimeout(() => {
                handleScrollFade();
                scrollTimeout = null;
            }, 16); // ~60fps
        }
    }

    // Add scroll event listener
    window.addEventListener('scroll', throttledScrollHandler);

    // Clear button functionality
    function syncClear() {
        if (searchInput.value && searchInput.value.trim().length > 0) {
            clearBtn.classList.remove('hidden');
        } else {
            clearBtn.classList.add('hidden');
        }
    }
    
    syncClear();
    searchInput.addEventListener('input', syncClear);

    clearBtn.addEventListener('click', () => {
        searchInput.value = '';
        searchInput.focus();
        syncClear();
    });

    // Modal functionality
    function openModal() {
        filterModal.classList.remove('hidden');
        filterModal.classList.add('show');
        document.body.style.overflow = 'hidden';
        console.log('Modal opened'); // Debug log
    }

    function closeModal() {
        filterModal.classList.add('hidden');
        filterModal.classList.remove('show');
        document.body.style.overflow = 'auto';
        console.log('Modal closed'); // Debug log
    }

    // Event listeners for modal
    filterBtn.addEventListener('click', openModal);
    
    // Close button event listener with debugging
    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('Close button clicked'); // Debug log
            closeModal();
        });
    } else {
        console.error('Close button not found!');
    }
    
    // Close modal when clicking outside
    filterModal.addEventListener('click', (e) => {
        if (e.target === filterModal) {
            closeModal();
        }
    });

    // Close modal with Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !filterModal.classList.contains('hidden')) {
            closeModal();
        }
    });

    // Reset filters
    resetFiltersBtn.addEventListener('click', () => {
        // Just clear the form fields in the modal - don't submit
        clientFilter.value = '';
        itemName.value = '';
        includesWords.value = '';
        dateFrom.value = '';
        dateTo.value = '';
        
        console.log('Filters reset'); // Debug log
    });

    // Apply filters
    applyFiltersBtn.addEventListener('click', () => {
        // Update hidden inputs with modal values
        clientFilterHidden.value = clientFilter.value;
        dateFromHidden.value = dateFrom.value;
        dateToHidden.value = dateTo.value;
        
        // Close modal and submit form
        closeModal();
        searchForm.submit();
    });

    // Enter key in search input submits form
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            searchForm.submit();
        }
    });
    

    // Initialize scroll fade effect
    handleScrollFade();
})();