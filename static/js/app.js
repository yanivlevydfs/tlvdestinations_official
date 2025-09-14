document.addEventListener('DOMContentLoaded', () => {
  // ---------- Theme (Bootswatch swap + persist) ----------
  const themeBtn  = document.getElementById('theme-toggle');
  const themeIcon = document.getElementById('theme-icon');
  const themeLink = document.getElementById('bs-theme-link');
  const modal = document.getElementById("mapModal");
  const iframe = document.getElementById("map-frame");
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

    responsive: {
      details: {
        type: 'column',     // adds the ► / ▼ control in its own column
        target: 0           // use column 0 as the control
      },
      breakpoints: [
        { name: 'desktop', width: Infinity },
        { name: 'tablet',  width: 992 },
        { name: 'mobile',  width: 768 }
      ]
    },

    fixedHeader: true,
    pageLength: 10,
    lengthMenu: [5, 10, 25, 50],
    columnDefs: [
      { className: 'dtr-control', orderable: false, targets: 0 }, // expand arrow column
      { targets: [1,2,3], responsivePriority: 1 },    // IATA, Name, City always visible
      { targets: [4],     responsivePriority: 2 },    // Country
      { targets: [5,6,7], responsivePriority: 10000 } // Airlines, Distance, FlightTime collapse first
    ],
    order: [1, 'asc'],

    language: LANG[lang].dt,

    initComplete: function () {
      $('#airports-table_filter input')
        .attr('id', 'airports-search')
        .attr('name', 'airports-search')
        .attr('placeholder', LANG[lang].placeholderSearch || 'Search…');
    }
  });
}


  // ---------- Helpers ----------
  const overlay       = document.getElementById('loadingOverlay');
  const loadingBar    = document.getElementById('loadingBar');
  const loadingDetail = document.getElementById('loadingDetail');
  const toastEl       = document.getElementById('app-toast');
  const toastText     = document.getElementById('toast-text');
  const bsToast       = new bootstrap.Toast(toastEl, { delay: 2500 });

  function showOverlay(msg) {
    overlay.style.display = 'flex';
    loadingBar.style.width = '0%';
    loadingBar.textContent = '0%';
    loadingDetail.textContent = msg;
  }
  function hideOverlay() { overlay.style.display = 'none'; }
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

	  lbl.textContent = parts.length ? parts.join(' • ') : dict.noFilters;
	}
  // ---------- SSE Progress ----------
  function updateProgress(p, dict) {
    if (!p) return;
    if (p.running) {
      const total = Math.max(1, p.total || 1);
      const done  = p.done || 0;
      const pct   = Math.min(100, Math.round((done/total)*100));
      overlay.style.display = 'flex';
      loadingBar.style.width = pct + '%';
      loadingBar.textContent = pct + '%';
      loadingDetail.textContent = dict.progress(done,total);
    } else {
      hideOverlay();
      pingToast(dict.toastRefOk, 'success');
    }
  }
  try {
    const evtSource = new EventSource("/api/progress/stream");
    // we'll attach handlers after language is applied (see below)
    window._sseSource = evtSource;
  } catch (e) { /* non-fatal */ }

  // ---------- Language ----------
  let currentLang = localStorage.getItem('fe-lang') || 'en';
  const langBtn   = document.getElementById('lang-toggle');
  const langLabel = document.getElementById('lang-label');

  function applyUnitsInCells(dict) {
    // Replace unit labels in Distance/Flight Time columns (7th & 8th)
    document.querySelectorAll('#airports-table tbody tr').forEach(tr=>{
      const d = tr.children[6]; const t = tr.children[7];
      if (d && d.textContent.includes('Kilometer')) d.textContent = d.textContent.replace('Kilometer', dict.units.km);
      if (t && t.textContent.includes('Hours'))    t.textContent = t.textContent.replace('Hours', dict.units.hr);
      if (d && d.textContent.includes('קילומטר') && dict.units.km==='Kilometer') d.textContent = d.textContent.replace('קילומטר', 'Kilometer');
      if (t && t.textContent.includes('שעות') && dict.units.hr==='Hours') t.textContent = t.textContent.replace('שעות', 'Hours');
    });
  }

  function applyLanguage(lang, reinitDT=true) {
    const d = LANG[lang];

    // Document direction + html lang
    document.documentElement.lang = (lang==='he' ? 'he' : 'en');
    document.body.dir = (lang==='he' ? 'rtl' : 'ltr');

    // Offcanvas side for RTL
    const canvas = document.getElementById('filtersCanvas');
    if (lang==='he') { canvas.classList.remove('offcanvas-start'); canvas.classList.add('offcanvas-end'); }
    else { canvas.classList.remove('offcanvas-end'); canvas.classList.add('offcanvas-start'); }

    // Nav & headings
    document.getElementById('brand-title').innerHTML = `<i class="bi bi-airplane-engines me-2"></i> ${d.brand}`;
    document.getElementById('view-map-btn').innerHTML = `<i class="bi bi-globe-americas me-1"></i> ${d.viewMap}`;
    document.getElementById('refresh-btn').innerHTML = `<i class="bi bi-arrow-repeat me-1"></i> ${d.refresh}`;
    document.getElementById('theme-toggle').innerHTML = `<i id="theme-icon" class="bi bi-moon-stars me-1"></i> ${d.theme}`;
    document.getElementById('filters-toggle-mobile').innerHTML = `<i class="bi bi-sliders me-1"></i> ${d.filters}`;
    document.getElementById('lang-label').textContent = d.langToggleLabelOther;

    document.getElementById('filters-title').innerHTML = `<i class="bi bi-sliders me-2"></i>${d.filters}`;
    document.getElementById('filters-header').textContent = d.filtersHeader;
    document.getElementById('lbl-country').textContent = d.country;
    document.getElementById('lbl-search').textContent  = d.search;
    document.getElementById('clear-filters').innerHTML = `<i class="bi bi-x-circle me-1"></i> ${d.clear}`;
    document.getElementById('active-filters').textContent = d.noFilters;
    document.getElementById('main-title').textContent = d.tableTitle;
    document.getElementById('subheader-text').textContent = d.subtitle;
    document.getElementById('export-label').innerHTML = `<i class="bi bi-database me-1"></i>${d.export}`;
    document.getElementById('map-title').textContent = d.mapTitle;
    document.getElementById('overlay-title').textContent = d.overlayTitle;
    document.getElementById('query-filter').placeholder = d.placeholderSearch;

    // Table headers
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

    // Re-Init DataTable with language (preserve filters/search)
    if (reinitDT) {
      const prevSearch = $('#query-filter').val();
      const prevCountry = $('#country-filter').val();

      if (table) table.destroy();
      table = initDataTable(lang);

      // Re-apply custom filters
      if (prevSearch) {
        table.search(prevSearch).draw();
      }
      if (prevCountry && prevCountry !== 'All') {
        table.column(3).search(escapeRegex(prevCountry), true, false).draw();
      }
    }

    // Units in cells
    applyUnitsInCells(d);

    // Update stored lang
    currentLang = lang;
    localStorage.setItem('fe-lang', lang);

    // Rebind SSE handlers with current language texts
    if (window._sseSource) {
      window._sseSource.onmessage = null;
      window._sseSource.addEventListener("progress", (e) => updateProgress(JSON.parse(e.data), d));
      window._sseSource.addEventListener("ping", () => {});
    }

    // Update active filters summary label
    updateActiveFilters(d);
  }

	langBtn.addEventListener('click', () => {
	  const u = new URL(window.location.href);
	  const isHe = u.searchParams.get('lang') === 'he';

	  if (isHe) {
		u.searchParams.delete('lang');       // back to EN (canonical /)
		localStorage.setItem('fe-lang', 'en');
	  } else {
		u.searchParams.set('lang', 'he');    // switch to HE
		localStorage.setItem('fe-lang', 'he');
	  }

	  // reload with new URL so server renders correct <html lang>, meta, title etc.
	  window.location.href = u.toString();
	});

  // ---------- DataTable (first init with saved lang) ----------
  const savedLang = localStorage.getItem('fe-lang') || 'en';
  table = initDataTable(savedLang);
  applyLanguage(savedLang, false); // apply texts, units, etc., without re-init loop

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

 document.getElementById('clear-filters').addEventListener('click', () => {
    $('#country-filter').val('All').trigger('change');
    $('#query-filter').val('').trigger('input');
  });

  updateActiveFilters(LANG[currentLang]);

 // ---------- Buttons ----------
  const refreshBtn = document.getElementById('refresh-btn');
  const viewMapBtn = document.getElementById('view-map-btn');
  const mapFrame   = document.getElementById('map-frame');
  const mapModalEl = document.getElementById('mapModal');

  refreshBtn.addEventListener('click', async () => {
    showOverlay(LANG[currentLang].overlayStart);
    try {
      const res  = await fetch('/admin/refresh?force=true', { method: 'POST' });
      if (!res.ok) throw new Error('Refresh failed');
    } catch (err) {
      hideOverlay();
      pingToast(LANG[currentLang].toastRefErr, 'danger');
    }
  });

  viewMapBtn.addEventListener('click', () => {
    showOverlay(LANG[currentLang].overlayLoadingMap);
    const modal = new bootstrap.Modal(mapModalEl);
    modal.show();
    hideOverlay();
    mapFrame.contentWindow.postMessage("modal-shown", "*");
  });
  
});