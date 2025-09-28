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
    const city = document.getElementById('city-filter')?.value || '';
    const q = document.getElementById('query-filter').value.trim();
    const parts = [];
    if (country && country !== 'All') {
      parts.push(`${dict.country}: ${country}`);
    }
    if (city && city !== 'All') {
      parts.push(`City: ${city}`);
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
    updateActiveFilters(LANG[currentLang]);
  });

  // 🆕 Mobile city filter
  $('#city-filter-mobile').on('change', function() {
    const val = this.value === "All" ? '' : this.value;
    table.column(2).search(escapeRegex(val), true, false).draw();
    updateActiveFilters(LANG[currentLang]);
  });

  let debounceTimerMobile;
  $('#query-filter-mobile').on('input', function() {
    clearTimeout(debounceTimerMobile);
    const val = this.value;
    debounceTimerMobile = setTimeout(() => {
      table.search(escapeRegex(val), true, false).draw();
      updateActiveFilters(LANG[currentLang]);
    }, 250);
  });

  $('#clear-filters-mobile').on('click', () => {
    $('#country-filter-mobile').val('All').trigger('change');
    $('#city-filter-mobile').val('All').trigger('change');
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

  // 🆕 populate city dropdowns after table loaded
  function populateDropdownFromTable(columnIndex, selectId) {
    const uniqueVals = new Set();
    table.column(columnIndex).data().each(function (val) {
      uniqueVals.add(val.trim());
    });
    const select = document.getElementById(selectId);
    if (!select) return;
    select.innerHTML = '<option value="All">All</option>';
    [...uniqueVals].sort().forEach(val => {
      const opt = document.createElement('option');
      opt.value = val;
      opt.textContent = val;
      select.appendChild(opt);
    });
  }
  populateDropdownFromTable(2, 'city-filter');
  populateDropdownFromTable(2, 'city-filter-mobile');

  applyLanguage(savedLang2, false);

  // ---------- Filters ----------
  $('#country-filter').on('change', function() {
    const val = this.value === "All" ? '' : this.value;
    table.column(3).search(escapeRegex(val), true, false).draw();
    updateActiveFilters(LANG[currentLang]);
  });

  // 🆕 City filter desktop
  $('#city-filter').on('change', function() {
    const val = this.value === "All" ? '' : this.value;
    table.column(2).search(escapeRegex(val), true, false).draw();
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
      $('#city-filter').val('All').trigger('change');
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
});
