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
	document.addEventListener('click', (e) => {
	  const btn = e.target.closest('#ai-chat-btn');
	  if (btn) {
		console.log('AI Chat button clicked — navigating to /chat');
		window.location.href = '/chat';
	  }
	});
  // ---------- DataTable init/destroy helper ----------
  let table; // global ref
	function dtButtonsFor(lang) {
	  const isMobile = window.innerWidth <= 768; // or use a stricter check if needed
	  if (isMobile) return []; // ❌ Skip buttons on mobile!

	  const b = LANG[lang].dtButtons;
	  return [
		{ extend:'copyHtml5',  className:'btn btn-primary', text:`<i class="bi bi-clipboard me-1"></i> ${b.copy}` },
		{ extend:'csvHtml5',   className:'btn btn-primary', text:`<i class="bi bi-filetype-csv me-1"></i> ${b.csv}` },
		{ extend:'excelHtml5', className:'btn btn-primary', text:`<i class="bi bi-file-earmark-excel me-1"></i> ${b.excel}` },
		{ extend:'pdfHtml5',   className:'btn btn-primary', text:`<i class="bi bi-file-earmark-pdf me-1"></i> ${b.pdf}` },
		{ extend:'print',      className:'btn btn-primary', text:`<i class="bi bi-printer me-1"></i> ${b.print}` }
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
        { targets: [4,5,6], responsivePriority: 10000 } // Airlines, Distance, Flight Time collapse first
      ],
      order: [[0, 'asc']],
      language: LANG[lang].dt,
      initComplete: function () {
        $('#airports-table_filter input')
          .attr('id', 'airports-search')
          .attr('name', 'airports-search')
          .attr('placeholder', LANG[lang].placeholderSearch || 'Search…');
      }
    });
  }

  // ---------- Toast Helper ----------
  const toastEl   = document.getElementById('app-toast');
  const toastText = document.getElementById('toast-text');
  const bsToast   = new bootstrap.Toast(toastEl, { delay: 2500 });

  function pingToast(msg, variant='dark') {
    toastEl.className = `toast align-items-center text-bg-${variant} border-0`;
    toastText.textContent = msg;
    bsToast.show();
  }

  function escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

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

    // Distance replacements
    if (newD.includes('Kilometer')) newD = newD.replace('Kilometer', dict.units.km);
    if (newD.includes('קילומטר') && dict.units.km === 'Kilometer') newD = newD.replace('קילומטר', 'Kilometer');

    // Flight time replacements
    if (newT.includes('Hours')) newT = newT.replace('Hours', dict.units.hr);
    if (newT.includes('שעות') && dict.units.hr === 'Hours') newT = newT.replace('שעות', 'Hours');

    // Only queue an update if something changed
    if (newD !== d.textContent || newT !== t.textContent) {
      updates.push(() => {
        if (newD !== d.textContent) d.textContent = newD;
        if (newT !== t.textContent) t.textContent = newT;
      });
    }
  });

  // Apply all updates in one pass
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
    safeSet('refresh-btn', `<i class="bi bi-arrow-repeat me-1"></i> ${d.refresh}`, true);
    safeSet('theme-toggle', `<i id="theme-icon" class="bi bi-moon-stars me-1"></i> ${d.theme}`, true);
    safeSet('filters-toggle-mobile', `<i class="bi bi-sliders me-1"></i> ${d.filters}`, true);
    safeSet('lang-label', d.langToggleLabelOther);
    safeSet('filters-title', `<i class="bi bi-sliders me-2"></i>${d.filters}`, true);
    safeSet('filters-header', d.filtersHeader);
    safeSet('lbl-country', d.country);
    safeSet('lbl-search', d.search);
    safeSet('clear-filters', `<i class="bi bi-x-circle me-1"></i> ${d.clear}`, true);
    safeSet('active-filters', d.noFilters);
    safeSet('main-title', d.tableTitle);
    safeSet('subheader-text', d.subtitle);
    safeSet('export-label', `<i class="bi bi-database me-1"></i>${d.export}`, true);
    safeSet('map-title', d.mapTitle);
    const qf = document.getElementById('query-filter');
    if (qf) qf.placeholder = d.placeholderSearch;

    const ths = document.querySelectorAll("#airports-table thead th");
    const headVals = [
      d.table?.iata      || "IATA",
      d.table?.name      || "Name",
      d.table?.city      || "City",
      d.table?.country   || "Country",
      d.table?.airlines  || "Airlines",
      d.table?.distance  || "Distance",
      d.table?.flightTime|| "Flight Time"
    ];
    headVals.forEach((t,i)=>ths[i] && (ths[i].textContent = t));

    if (reinitDT) {
      const prevSearch = $('#query-filter').val();
      const prevCountry = $('#country-filter').val();
      if (table) table.destroy();
      table = initDataTable(lang);
      if (prevSearch) table.search(prevSearch).draw();
      if (prevCountry && prevCountry !== 'All') {
        table.column(3).search(escapeRegex(prevCountry), true, false).draw();
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
$('#country-filter-mobile').on('change', function() {
  const val = this.value === "All" ? '' : this.value;
  table.column(3).search(escapeRegex(val), true, false).draw();
  updateActiveFilters(LANG[currentLang]); // reuse same updater
});

let debounceTimerMobile;
$('#query-filter-mobile').on('input', function() {
  clearTimeout(debounceTimerMobile);
  const val = this.value;
  debounceTimerMobile = setTimeout(() => {
    table.search(escapeRegex(val), true, false).draw();
    updateActiveFilters(LANG[currentLang]); // reuse same updater
  }, 250);
});

$('#clear-filters-mobile').on('click', () => {
  $('#country-filter-mobile').val('All').trigger('change');
  $('#query-filter-mobile').val('').trigger('input');
});



  // ---------- DataTable (first init with saved lang) ----------
  const savedLang = localStorage.getItem('fe-lang') || 'en';
  table = initDataTable(savedLang);
  applyLanguage(savedLang, false);

  // ---------- Filters ----------
  $('#country-filter').on('change', function() {
    const val = this.value === "All" ? '' : this.value;
    table.column(3).search(escapeRegex(val), true, false).draw();
    updateActiveFilters(LANG[currentLang]);
  });

  let debounceTimer;
  $('#query-filter').on('input', function() {
    clearTimeout(debounceTimer);
    const val = this.value;
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
  const refreshBtn = document.getElementById('refresh-btn');
  const viewMapBtn = document.getElementById('view-map-btn');

  if (refreshBtn) {
    refreshBtn.addEventListener('click', async () => {
      pingToast(LANG[currentLang].overlayStart, 'info');
      try {
        const res = await fetch('/admin/refresh?force=true', { method: 'POST' });
        if (!res.ok) throw new Error('Refresh failed');
        pingToast(LANG[currentLang].toastRefOk, 'success');
      } catch (err) {
        pingToast(LANG[currentLang].toastRefErr, 'danger');
      }
    });
  }

  if (viewMapBtn) {
    viewMapBtn.addEventListener('click', () => {
      pingToast(LANG[currentLang].overlayLoadingMap, 'info');
      const modal = new bootstrap.Modal(mapModal);
      modal.show();
      iframe.contentWindow.postMessage("modal-shown", "*");
    });
  }
});

