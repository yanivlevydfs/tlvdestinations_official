document.addEventListener('DOMContentLoaded', () => {
  // ---------- Theme (Bootswatch swap + persist) ----------
  const themeBtn  = document.getElementById('theme-toggle');
  const themeIcon = document.getElementById('theme-icon');
  const themeLink = document.getElementById('bs-theme-link');
  const mapModal  = document.getElementById("mapModal");
  const iframe    = document.getElementById("map-frame");
  const THEMES = {
    light: 'https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/litera/bootstrap.min.css',
    dark:  'https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/darkly/bootstrap.min.css'
  };
  const savedTheme = localStorage.getItem('fe-theme') || 'light';
  if (savedTheme === 'dark') {
    document.body.classList.add('dark-mode');
    themeLink.href = THEMES.dark;
    themeIcon.className = 'bi bi-sun';
  }
  themeBtn.addEventListener('click', () => {
    const dark = !document.body.classList.contains('dark-mode');
    document.body.classList.toggle('dark-mode', dark);
    themeLink.href = dark ? THEMES.dark : THEMES.light;
    themeIcon.className = dark ? 'bi bi-sun' : 'bi bi-moon-stars';
    localStorage.setItem('fe-theme', dark ? 'dark' : 'light');
  });

  // ---------- DataTable init/destroy helper ----------
  let table; // global ref
  function dtButtonsFor(lang) {
    const isMobile = window.innerWidth <= 768;
    if (isMobile) return []; // skip buttons on mobile
    const b = LANG[lang].dtButtons;
    return [
		{ extend:'copyHtml5',  className:'btn btn-primary mobile-small-btn', text:`<i class="bi bi-clipboard me-1"></i> ${b.copy}` },
		 //{ extend:'csvHtml5',   className:'btn btn-primary mobile-small-btn', text:`<i class="bi bi-filetype-csv me-1"></i> ${b.csv}` },
		{ extend:'excelHtml5', className:'btn btn-primary mobile-small-btn', text:`<i class="bi bi-file-earmark-excel me-1"></i> ${b.excel}` },
		{ extend:'pdfHtml5',   className:'btn btn-primary mobile-small-btn', text:`<i class="bi bi-file-earmark-pdf me-1"></i> ${b.pdf}` },
		{ extend:'print',      className:'btn btn-primary mobile-small-btn', text:`<i class="bi bi-printer me-1"></i> ${b.print}` }
    ];
  }

  function initDataTable(lang) {
    return $('#airports-table').DataTable({
      dom:
        "<'row mb-2'<'col-sm-12 col-md-6'B><'col-sm-12 col-md-6'f>>" +
        "<'row'<'col-12'tr>>" +
        "<'row mt-2'<'col-sm-12 col-md-5'i><'col-sm-12 col-md-7'p>>",
      buttons: dtButtonsFor(lang),
      responsive: true,
      fixedHeader: true,
      pageLength: 10,
      lengthMenu: [5, 10, 25, 50],
      columnDefs: [
        { targets: [0,1,2], responsivePriority: 1 },   // IATA, Name, City always visible
        { targets: [3],     responsivePriority: 2 },   // Country
        { targets: [4,5,6], responsivePriority: 10000 },
		{ targets: [7], visible: false, searchable: true }		// Airlines, Distance, Flight Time collapse first
      ],
      order: [[0, 'asc']],
      language: LANG[lang].dt,
      pagingType: "simple",
      initComplete: function () {
        $('#airports-table_filter input')
          .attr('id', 'airports-search')
          .attr('name', 'airports-search')
          .attr('placeholder', LANG[lang].placeholderSearch || 'Search…');
        const info = document.querySelector('#airports-table_info');
        if (info) {
          info.classList.add('last-update-badge');
        }
      }
    });
  }

  function escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  // 🆕 updateActiveFilters now includes city
  function updateActiveFilters(dict) {
    const lbl = document.getElementById('active-filters');
    const country = document.getElementById('country-filter').value;
    const q = document.getElementById('query-filter').value.trim();
    const parts = [];
    if (country && country !== 'All') {
      parts.push(`${dict.country}: ${country}`);
    }
    if (q) {
      parts.push(`${dict.search}: "${q}"`);
    }
    if (lbl) lbl.textContent = parts.length ? parts.join(' • ') : dict.noFilters;
  }

  // ---------- Language ----------
  let currentLang = localStorage.getItem('fe-lang') || 'en';
  const langBtn   = document.getElementById('lang-toggle');

  function safeSet(id, value, isHTML=false) {
    const el = document.getElementById(id);
    if (!el) return;
    if (isHTML) el.innerHTML = value;
    else el.textContent = value;
  }

  function applyUnitsInCells(dict) {
    const rows = document.querySelectorAll('#airports-table tbody tr');
    const updates = [];
    rows.forEach(tr => {
      const d = tr.children[6];
      const t = tr.children[7];
      if (!d || !t) return;
      let newD = d.textContent;
      let newT = t.textContent;
      if (newD.includes('Kilometer')) newD = newD.replace('Kilometer', dict.units.km);
      if (newD.includes('קילומטר') && dict.units.km === 'Kilometer') newD = newD.replace('קילומטר', 'Kilometer');
      if (newT.includes('Hours')) newT = newT.replace('Hours', dict.units.hr);
      if (newT.includes('שעות') && dict.units.hr === 'Hours') newT = newT.replace('שעות', 'Hours');
      if (newD !== d.textContent || newT !== t.textContent) {
        updates.push(() => {
          if (newD !== d.textContent) d.textContent = newD;
          if (newT !== t.textContent) t.textContent = newT;
        });
      }
    });
    updates.forEach(fn => fn());
  }

  function applyLanguage(lang, reinitDT=true) {
    const d = LANG[lang];
    document.documentElement.lang = (lang==='he' ? 'he' : 'en');
    document.body.dir = (lang==='he' ? 'rtl' : 'ltr');
    const canvas = document.getElementById('filtersCanvas');
    if (canvas) {
      if (lang==='he') { canvas.classList.remove('offcanvas-start'); canvas.classList.add('offcanvas-end'); }
      else { canvas.classList.remove('offcanvas-end'); canvas.classList.add('offcanvas-start'); }
    }
    safeSet('brand-title', `<i class="bi bi-airplane-engines me-2"></i> ${d.brand}`, true);
    safeSet('view-map-btn', `<i class="bi bi-globe-americas me-1"></i> ${d.viewMap}`, true);
    safeSet('theme-toggle', `<i id="theme-icon" class="bi bi-moon-stars me-1"></i> ${d.theme}`, true);
    safeSet('lang-label', d.langToggleLabelOther);
    safeSet('filters-title', `<i class="bi bi-sliders me-2"></i>${d.filters}`, true);
    safeSet('lbl-country', d.country);
    safeSet('lbl-search', d.search);
    safeSet('clear-filters', `<i class="bi bi-x-circle me-1"></i> ${d.clear}`, true);
    safeSet('active-filters', d.noFilters);
    safeSet('map-title', d.mapTitle);
    const qf = document.getElementById('query-filter');
    if (qf) qf.placeholder = d.placeholderSearch;
    const ths = document.querySelectorAll("#airports-table thead th");
    const headVals = [
      d.table?.iata      || "IATA",
      d.table?.name      || "Airport",
	  d.table?.country   || "Country",
      d.table?.city      || "City",      
      d.table?.airlines  || "Airlines",
      d.table?.distance  || "Distance",
      d.table?.flightTime|| "Flight Time",
	  d.table?.direction || "Direction"
    ];
    headVals.forEach((t,i)=>ths[i] && (ths[i].textContent = t));
    if (reinitDT) {
      const prevSearch = $('#query-filter').val();
      const prevCountry = $('#country-filter').val();
      if (table) table.destroy();
      table = initDataTable(lang);
      if (prevSearch) table.search(prevSearch).draw();
      if (prevCountry && prevCountry !== 'All') {
        table.column(2).search(escapeRegex(prevCountry), true, false).draw();
      }
    }
    applyUnitsInCells(d);
    currentLang = lang;
    localStorage.setItem('fe-lang', lang);
    updateActiveFilters(d);
  }

  if (langBtn) {
    langBtn.addEventListener('click', () => {
      const u = new URL(window.location.href);
      const isHe = u.searchParams.get('lang') === 'he';
      if (isHe) {
        u.searchParams.delete('lang');
        localStorage.setItem('fe-lang', 'en');
      } else {
        u.searchParams.set('lang', 'he');
        localStorage.setItem('fe-lang', 'he');
      }
      window.location.href = u.toString();
    });
  }

// ---------- Mobile filters ----------
// Country filter (MOBILE)
$('#country-filter-mobile').on('change', function () {
  const val = this.value === "All" ? '' : this.value;
  $('#query-filter-mobile').val('');
  table.search('', true, false);
  table.column(2).search(escapeRegex(val), true, false).draw();
  updateActiveFilters(LANG[currentLang]);
});


// Query text search (MOBILE)
let debounceTimerMobile;
$('#query-filter-mobile').on('input', function () {
  clearTimeout(debounceTimerMobile);
  const val = this.value;
  $('#country-filter-mobile').val('All');
  table.column(2).search('', true, false);

  debounceTimerMobile = setTimeout(() => {
    table.search(escapeRegex(val), true, false).draw();
    updateActiveFilters(LANG[currentLang]);
  }, 250);
});


// Clear button (MOBILE)
$('#clear-filters-mobile').on('click', () => {
  $('#country-filter-mobile').val('All').trigger('change');
  $('#query-filter-mobile').val('').trigger('input');
});

  document.addEventListener('click', (e) => {
    const btn = e.target.closest('#ai-chat-btn');
    if (btn) {
      window.location.href = '/chat';
    }
  });

  // ---------- DataTable (first init with saved lang) ----------
  const savedLang2 = localStorage.getItem('fe-lang') || 'en';
  table = initDataTable(savedLang2);
  applyLanguage(savedLang2, false);

  // ---------- Filters ----------
$('#country-filter').on('change', function () {
  const val = this.value === "All" ? '' : this.value;
  $('#query-filter').val('');
  table.search('', true, false); // Clear global search
  table.column(2).search(escapeRegex(val), true, false).draw();
  updateActiveFilters(LANG[currentLang]);
});


let debounceTimer;
$('#query-filter').on('input', function () {
  clearTimeout(debounceTimer);
  const val = this.value;
  $('#country-filter').val('All');
  table.column(2).search('', true, false);

  debounceTimer = setTimeout(() => {
    table.search(escapeRegex(val), true, false).draw();
    updateActiveFilters(LANG[currentLang]);
  }, 250);
});


  const clearFiltersBtn = document.getElementById('clear-filters');
  if (clearFiltersBtn) {
    clearFiltersBtn.addEventListener('click', () => {
      $('#country-filter').val('All').trigger('change');
      $('#query-filter').val('').trigger('input');
    });
  }
  updateActiveFilters(LANG[currentLang]);

  // ---------- Buttons ----------
  const viewMapBtn = document.getElementById('view-map-btn');
  if (viewMapBtn) {
    viewMapBtn.addEventListener('click', () => {
      if (!iframe.src) {
        iframe.src = iframe.dataset.src;
      }
      const modal = new bootstrap.Modal(mapModal);
      modal.show();
      iframe.contentWindow?.postMessage("modal-shown", "*");
    });
  }
  // ---------- Install App Banner (Mobile Only) ----------
  const installContainer = document.getElementById('install-app-container');
  const installBtn = document.getElementById('install-app-btn');
  let deferredPrompt = null;
  let dismissed = false;
  let scrollShown = false;

  if (window.innerWidth <= 768) {
    // Show banner with animation
    function showInstallBanner() {
      if (dismissed) return;
      installContainer?.classList.remove('d-none');
      setTimeout(() => installContainer?.classList.add('show'), 50);

      setTimeout(() => {
        hideInstallBanner();
      }, 10000);
    }

    function hideInstallBanner() {
      installContainer?.classList.remove('show');
      setTimeout(() => installContainer?.classList.add('d-none'), 400);
    }

    window.addEventListener('beforeinstallprompt', (e) => {
      e.preventDefault();
      deferredPrompt = e;
      showInstallBanner();
    });

    installBtn?.addEventListener('click', async () => {
      if (!deferredPrompt) return;
      deferredPrompt.prompt();
      const { outcome } = await deferredPrompt.userChoice;
      console.log(`User response: ${outcome}`);
      deferredPrompt = null;
      hideInstallBanner();
    });

    window.addEventListener('scroll', () => {
      if (scrollShown || dismissed || deferredPrompt) return;
      if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 100) {
        scrollShown = true;
        showInstallBanner();
      }
    });
  }  // end mobile js
  const directionSwitch = document.getElementById("direction-switch");
  const directionLabel = document.getElementById("direction-switch-label");
  const toggleHandleLabel = document.querySelector(".toggle-label");

  if (directionSwitch && $.fn.dataTable.isDataTable('#airports-table')) {
    const table = $('#airports-table').DataTable();
    const isHebrew = document.documentElement.lang === "he";

    function applyDirectionFilter(checked) {
      const regex = checked ? "^D$" : "^A$";
      table.column(7).search(regex, true, false).draw();

      directionLabel.textContent = checked
        ? (isHebrew ? "המראות (מתל אביב)" : "TLV Departures")
        : (isHebrew ? "נחיתות (לתל אביב)" : "TLV Arrivals");
    }
    directionSwitch.checked = true;  // Force switch ON
    applyDirectionFilter(directionSwitch.checked);

    directionSwitch.addEventListener("change", () => {
      applyDirectionFilter(directionSwitch.checked);
    });
  }
});
