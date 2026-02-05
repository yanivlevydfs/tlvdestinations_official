document.addEventListener('DOMContentLoaded', () => {

  // ---------- Constants & DOM refs ----------
  const THEMES = {
    light: 'https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/litera/bootstrap.min.css',
    dark: 'https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/darkly/bootstrap.min.css'
  };

  let table;  // will hold DataTable instance
  let currentLang = localStorage.getItem('fe-lang') || 'en';

  const themeBtn = document.getElementById('theme-toggle');
  const themeIcon = document.getElementById('theme-icon');
  const themeLink = document.getElementById('bs-theme-link');
  const mapModal = document.getElementById('mapModal');
  const iframe = document.getElementById('map-frame');
  const langBtn = document.getElementById('lang-toggle');
  const clearFiltersBtn = document.getElementById('clear-filters');
  const viewMapBtn = document.getElementById('view-map-btn');
  const installContainer = document.getElementById('install-app-container');
  const installBtn = document.getElementById('install-app-btn');
  const directionSelect = document.getElementById('direction-select');
  const airlineSelect = document.getElementById('airline-filter');
  const airlineSelectMobile = document.getElementById('airline-filter-mobile');

  // ---------- Theme (Bootswatch + persist) ----------
  const savedTheme = localStorage.getItem('fe-theme') || 'light';
  if (savedTheme === 'dark') {
    document.body.classList.add('dark-mode');
    if (themeLink) themeLink.href = THEMES.dark;
    if (themeIcon) themeIcon.className = 'bi bi-sun';
  }
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const dark = !document.body.classList.contains('dark-mode');
      document.body.classList.toggle('dark-mode', dark);
      if (themeLink) themeLink.href = dark ? THEMES.dark : THEMES.light;
      if (themeIcon) themeIcon.className = dark ? 'bi bi-sun' : 'bi bi-moon-stars';
      localStorage.setItem('fe-theme', dark ? 'dark' : 'light');
    });
  }

  // ---------- DataTable init / destroy helper ----------
  function dtButtonsFor(lang) {
    // 🧱 Safe fallback if LANG or dtButtons missing
    const safeLang = (typeof LANG !== "undefined" && LANG[lang]) ? LANG[lang] : LANG?.en || {};
    const b = safeLang.dtButtons || { copy: "Copy", excel: "Excel", pdf: "PDF", print: "Print" };

    return [
      {
        extend: "copyHtml5",
        className: "btn btn-primary btn-sm mobile-small-btn",
        text: `<i class="bi bi-clipboard me-1"></i> ${b.copy}`,
      },
      {
        extend: "excelHtml5",
        className: "btn btn-primary btn-sm mobile-small-btn",
        text: `<i class="bi bi-file-earmark-excel me-1"></i> ${b.excel}`,
      },
      {
        extend: "pdfHtml5",
        className: "btn btn-primary btn-sm mobile-small-btn",
        text: `<i class="bi bi-file-earmark-pdf me-1"></i> ${b.pdf}`,
        customize: function (doc) {
          const now = new Date();
          const timestamp = now
            .toLocaleString("en-GB", {
              year: "numeric",
              month: "2-digit",
              day: "2-digit",
              hour: "2-digit",
              minute: "2-digit",
            })
            .replace(",", "");

          // 🕒 Header timestamp
          doc.header = function () {
            return {
              text: `Exported: ${timestamp}`,
              alignment: "right",
              margin: [0, 10, 10, 0],
              fontSize: 8,
            };
          };

          // 🌍 RTL/LTR handling
          doc.defaultStyle = {
            font: "DejaVuSans",
            alignment: lang === "he" ? "right" : "left",
            rtl: lang === "he",
          };

          const table = doc.content?.[1]?.table;
          if (table?.body) {
            // 🪞 Apply RTL text fixes
            if (lang === "he") {
              table.body.forEach((row) => {
                row.forEach((cell, idx) => {
                  if (typeof cell === "string") {
                    row[idx] = { text: cell, alignment: "right", rtl: true };
                  } else if (typeof cell === "object") {
                    cell.alignment = "right";
                    cell.rtl = true;
                  }
                });
              });
            }

            // 🔍 Dynamically detect “Direction” column index
            const headerRow = table.body[0] || [];
            const dirIndex = headerRow.findIndex((c) =>
              typeof c === "string"
                ? /direction/i.test(c)
                : /direction/i.test(c.text || "")
            );

            if (dirIndex >= 0) {
              table.body.forEach((row) => row.splice(dirIndex, 1));
              if (table.widths && table.widths.length > dirIndex) {
                table.widths.splice(dirIndex, 1);
              }
            }
          }
        },
      },
      {
        extend: "print",
        className: "btn btn-primary btn-sm mobile-small-btn",
        text: `<i class="bi bi-printer me-1"></i> ${b.print}`,
      },
    ];
  }

  function initDataTable(lang) {
    // 🧱 Guard language object safely
    const safeLang =
      (typeof LANG !== "undefined" && LANG[lang])
        ? LANG[lang]
        : LANG?.en || {};

    // 🧩 Detect "Direction" column index dynamically
    const dirHeader = document.querySelectorAll('#airports-table thead th');
    let dirIndex = 7; // fallback default
    dirHeader.forEach((th, i) => {
      const text = th.textContent?.trim().toLowerCase();
      if (text.includes('direction') || text.includes('כיוון')) dirIndex = i;
    });

    return $('#airports-table').DataTable({
      dom:
        "<'row mb-2'<'col-sm-12 col-md-6'B><'col-sm-12 col-md-6'f>>" +
        "<'row'<'col-12'tr>>" +
        "<'row mt-2'<'col-sm-12 col-md-5'i><'col-sm-12 col-md-7'p>>",
      buttons: dtButtonsFor(lang),
      responsive: true,
      fixedHeader: true,
      pageLength: 25,
      lengthMenu: [5, 10, 25, 50],
      columnDefs: [
        { targets: [0, 1, 2], responsivePriority: 1 },
        { targets: [3], responsivePriority: 2 },
        { targets: [4, 5, 6], responsivePriority: 10000 },
        { targets: [dirIndex], visible: false, searchable: true } // ✅ dynamic
      ],
      order: [[1, 'asc']],
      language: safeLang.dt || {},
      pagingType: "simple",
      initComplete: function () {
        $('#airports-table_filter input')
          .attr('id', 'airports-search')
          .attr('name', 'airports-search')
          .attr('placeholder', safeLang.placeholderSearch || 'Search…');

        const info = document.querySelector('#airports-table_info');
        if (info) info.classList.add('last-update-badge');
      }
    });
  }


  function escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  // 🆕 updateActiveFilters includes city + search
  function updateActiveFilters(dict) {
    const lbl = document.getElementById('active-filters');
    const countryEl = document.getElementById('country-filter');
    const queryEl = document.getElementById('query-filter');
    const country = countryEl ? countryEl.value : '';
    const q = queryEl ? queryEl.value.trim() : '';
    const parts = [];
    if (country && country !== 'All') {
      parts.push(`${dict.country}: ${country}`);
    }
    if (q) {
      parts.push(`${dict.search}: "${q}"`);
    }
    // Airline
    const airlineEl = document.getElementById('airline-filter');
    const airline = airlineEl ? airlineEl.value : '';
    if (airline && airline !== 'All') {
      parts.push(`${dict.airline}: ${airline}`);
    }

    if (lbl) {
      lbl.textContent = parts.length ? parts.join(' • ') : dict.noFilters;
    }
  }

  // ---------- Language / UI updates ----------
  function safeSet(id, value, isHTML = false) {
    const el = document.getElementById(id);
    if (!el) return;
    if (isHTML) el.innerHTML = value;
    else el.textContent = value;
  }

  function applyUnitsInCells(dict) {
    const rows = document.querySelectorAll('#airports-table tbody tr');
    rows.forEach(tr => {
      const d = tr.children[6];
      const t = tr.children[7];
      if (!d || !t) return;
      let newD = d.textContent;
      let newT = t.textContent;
      if (newD.includes('Kilometer')) newD = newD.replace('Kilometer', dict.units.km);
      if (newD.includes('קילומטר') && dict.units.km === 'Kilometer') {
        newD = newD.replace('קילומטר', 'Kilometer');
      }
      if (newT.includes('Hours')) newT = newT.replace('Hours', dict.units.hr);
      if (newT.includes('שעות') && dict.units.hr === 'Hours') {
        newT = newT.replace('שעות', 'Hours');
      }
      if (newD !== d.textContent) d.textContent = newD;
      if (newT !== t.textContent) t.textContent = newT;
    });
  }

  function applyLanguage(lang, reinitDT = true) {
    const d = LANG[lang];
    document.documentElement.lang = (lang === 'he' ? 'he' : 'en');
    document.body.dir = (lang === 'he' ? 'rtl' : 'ltr');

    const canvas = document.getElementById('filtersCanvas');
    if (canvas) {
      if (lang === 'he') {
        canvas.classList.remove('offcanvas-start');
        canvas.classList.add('offcanvas-end');
      } else {
        canvas.classList.remove('offcanvas-end');
        canvas.classList.add('offcanvas-start');
      }
    }

    const isDesktop = window.matchMedia("(min-width: 992px)").matches;
    safeSet("brand-title", isDesktop ? d.brand_desktop : d.brand_mobile, true);
    safeSet('view-map-btn', `<i class="bi bi-globe-americas me-1"></i> ${d.viewMap}`, true);
    safeSet('theme-toggle', `<i id="theme-icon" class="bi bi-moon-stars me-1"></i> ${d.theme}`, true);
    safeSet('lang-label', d.langToggleLabelOther);
    safeSet('filters-title', `<i class="bi bi-sliders me-2"></i>${d.filters}`, true);
    safeSet('lbl-country', d.country);
    safeSet('lbl-search', d.search);
    // safeSet('lbl-airline', d.airline); // If label has ID
    safeSet('clear-filters', `<i class="bi bi-x-circle me-1"></i> ${d.clear}`, true);
    safeSet('active-filters', d.noFilters);
    safeSet('map-title', d.mapTitle);

    const qf = document.getElementById('query-filter');
    if (qf) qf.placeholder = d.placeholderSearch;

    const ths = document.querySelectorAll('#airports-table thead th');
    const headVals = [
      d.table?.iata || "IATA",
      d.table?.name || "Airport",
      d.table?.country || "Country",
      d.table?.city || "City",
      d.table?.airlines || "Airlines",
      d.table?.distance || "Distance",
      d.table?.flightTime || "Flight Time",
      d.table?.direction || "Direction"
    ];
    headVals.forEach((t, i) => {
      if (ths[i]) ths[i].textContent = t;
    });

    if (reinitDT) {
      const prevSearch = $('#query-filter').val();
      const prevCountry = $('#country-filter').val();
      if (table) table.destroy();
      table = initDataTable(lang);
      if (prevSearch) table.search(prevSearch).draw();
      if (prevCountry && prevCountry !== 'All') {
        table.column(2).search(escapeRegex(prevCountry), true, false).draw();
      }
      populateAirlineFilter(lang);
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

  // ---------- Populate Airline Filter ----------
  function populateAirlineFilter(lang) {
    if (!table) return;
    const airlines = new Set();
    // Column 4 is "Airlines"
    // DataTables().rows().every(...) iterates over all rows
    table.rows().every(function () {
      const data = this.data()[4]; // string content of the cell
      if (!data) return;
      // Extract from data-airline="..."
      const matches = data.match(/data-airline="([^"]*)"/g);
      if (matches) {
        matches.forEach(m => {
          const name = m.match(/data-airline="([^"]*)"/)[1];
          if (name) airlines.add(name);
        });
      }
    });

    const sorted = Array.from(airlines).sort();

    [airlineSelect, airlineSelectMobile].forEach(sel => {
      if (!sel) return;
      const currentVal = sel.value;
      // Keep first option (All)
      while (sel.options.length > 1) {
        sel.remove(1);
      }
      sorted.forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        sel.appendChild(opt);
      });
      // Restore selection if valid
      if (sorted.includes(currentVal)) {
        sel.value = currentVal;
      } else {
        sel.value = "All"; // reset if not found (or on init)
      }
    });
  }

  // ---------- Mobile filters ----------
  $('#country-filter-mobile').on('change', function () {
    const val = this.value === 'All' ? '' : this.value;
    $('#query-filter-mobile').val('');
    table.search('', true, false);
    table.column(2).search(escapeRegex(val), true, false).draw();
    updateActiveFilters(LANG[currentLang]);
  });

  $('#airline-filter-mobile').on('change', function () {
    const val = this.value === 'All' ? '' : this.value;
    $('#query-filter-mobile').val('');
    table.search('', true, false); // clear global search
    // Assuming Airline is column 4
    table.column(4).search(escapeRegex(val), true, false).draw();
    updateActiveFilters(LANG[currentLang]);
  });

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

  // ---------- Chat button redirect ----------
  document.addEventListener('click', e => {
    const btn = e.target.closest('#ai-chat-btn');
    if (btn) {
      window.location.href = '/chat';
    }
  });

  // ---------- Initialize DataTable + language safely ----------
  const savedLang2 = localStorage.getItem('fe-lang') || 'en';
  const safeLangObj =
    (typeof LANG !== "undefined" && LANG[savedLang2])
      ? LANG[savedLang2]
      : LANG?.en || {};

  if (!LANG || !LANG[savedLang2]) {
    console.warn(`⚠️ Missing LANG data for '${savedLang2}', falling back to English.`);
  }

  try {
    table = initDataTable(savedLang2);
    populateAirlineFilter(savedLang2);
    // Defer applyLanguage slightly to allow DOM + DT render
    setTimeout(() => applyLanguage(savedLang2, false), 50);
  } catch (err) {
    console.error("❌ Failed to initialize DataTable or language:", err);
  }

  // ---------- Desktop filters ----------
  $('#country-filter').on('change', function () {
    const val = this.value === 'All' ? '' : this.value;
    $('#query-filter').val('');
    table.search('', true, false);
    table.column(2).search(escapeRegex(val), true, false).draw();
    updateActiveFilters(LANG[currentLang]);
  });

  $('#airline-filter').on('change', function () {
    const val = this.value === 'All' ? '' : this.value;
    $('#query-filter').val('');
    table.search('', true, false);
    table.column(4).search(escapeRegex(val), true, false).draw();
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

  if (clearFiltersBtn) {
    clearFiltersBtn.addEventListener('click', () => {
      $('#country-filter').val('All').trigger('change');
      $('#airline-filter').val('All').trigger('change');
      $('#query-filter').val('').trigger('input');
    });
  }
  updateActiveFilters(LANG[currentLang]);

  // Init Mobile Filters (now that table is ready)
  if (table) {
    $('#country-filter-mobile').val('All').trigger('change');
    $('#airline-filter-mobile').val('All').trigger('change');
    $('#query-filter-mobile').val('').trigger('input');
  }

  // ---------- View Map button logic ----------
  if (viewMapBtn && mapModal && iframe) {
    viewMapBtn.addEventListener('click', () => {
      //console.log('🗺️ View map clicked. iframe:', iframe, 'mapModal:', mapModal);

      // If iframe has no src, set from data-src
      const dataSrc = iframe.dataset?.src;
      if (!iframe.src && dataSrc) {
        iframe.src = dataSrc;
      }

      try {
        const modal = new bootstrap.Modal(mapModal);
        modal.show();
        iframe.contentWindow?.postMessage('modal-shown', '*');
      } catch (err) {
        console.error('❌ Error showing map modal:', err);
      }
    });
  } else if (viewMapBtn || mapModal || iframe) {
    // ⚠️ Warn only if SOME map elements exist, but not all
    console.warn('⚠️ Some view-map elements missing:', { viewMapBtn, mapModal, iframe });
  } else {
    // ✅ No map section at all → no warning needed
    // console.debug('ℹ️ This page has no map section, skipping map logic.');
  }

  // ---------- Install App Banner (mobile) ----------
  if (window.innerWidth <= 768) {
    let deferredPrompt = null;
    let dismissed = false;
    let scrollShown = false;

    function showInstallBanner() {
      if (dismissed) return;
      installContainer?.classList.remove('d-none');
      setTimeout(() => installContainer?.classList.add('show'), 50);
      setTimeout(hideInstallBanner, 10000);
    }
    function hideInstallBanner() {
      installContainer?.classList.remove('show');
      setTimeout(() => installContainer?.classList.add('d-none'), 400);
    }

    window.addEventListener('beforeinstallprompt', e => {
      e.preventDefault();
      deferredPrompt = e;
      showInstallBanner();
    });

    installBtn?.addEventListener('click', async () => {
      if (!deferredPrompt) return;
      deferredPrompt.prompt();
      await deferredPrompt.userChoice;
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
  }

  // ---------- Direction filter ----------
  if (directionSelect && $.fn.dataTable.isDataTable('#airports-table')) {
    const dt = table; // IMPORTANT: reuse existing instance

    function applyDirectionFilter(value) {
      if (!value || value === 'all') {
        dt.column(7).search('', true, false).draw();
        return;
      }
      const regex = value === 'outbound' ? '^D$' : '^A$';
      dt.column(7).search(regex, true, false).draw();
    }

    applyDirectionFilter(directionSelect.value);
    directionSelect.addEventListener('change', () => {
      applyDirectionFilter(directionSelect.value);
    });
  }

  (() => {
    const COOKIE_KEY = 'cookie_consent_choice';
    const PREFS_KEY = 'cookie_prefs';
    const cookieLang = document.documentElement.lang || 'en';
    const DEV_SHOW_COOKIE_BANNER = false;
    if (DEV_SHOW_COOKIE_BANNER) localStorage.removeItem(COOKIE_KEY);

    const TEXT = {
      en: {
        banner: {
          message: "We use cookies to enhance your experience. Choose what you allow.",
          accept: "Accept All",
          reject: "Reject All",
          customize: "Customize"
        },
        modal: {
          title: "Customize Your Preferences",
          desc: "Choose which types of cookies you allow. You can always change this later.",
          essential: "Essential – Required for site to function.",
          analytics: "Analytics – Help us understand usage.",
          marketing: "Marketing – Personalized ads and campaigns.",
          cancel: "Cancel",
          save: "Save Preferences"
        }
      },
      he: {
        banner: {
          message: "אנו משתמשים בעוגיות כדי לשפר את חוויית השימוש שלך. באפשרותך לבחור מה לאפשר.",
          accept: "אשר הכול",
          reject: "דחה הכול",
          customize: "התאם אישית"
        },
        modal: {
          title: "התאם את ההעדפות שלך",
          desc: "בחר אילו קוקיות לאפשר. תוכל לשנות זאת בכל עת.",
          essential: "חיוני – נדרש לפעולת האתר.",
          analytics: "ניתוח – עוזר לנו להבין את השימוש באתר.",
          marketing: "שיווק – פרסומות מותאמות אישית.",
          cancel: "ביטול",
          save: "שמור העדפות"
        }
      }
    };

    const t = TEXT[cookieLang] || TEXT.en;

    const cookieBanner = document.getElementById('cookie-banner');
    const acceptBtn = document.getElementById('cookie-accept-btn');
    const rejectBtn = document.getElementById('cookie-reject-btn');
    const customizeBtn = document.getElementById('cookie-customize-btn');
    const cookieMsg = document.getElementById('cookie-msg');
    const customizeModalEl = document.getElementById('cookieCustomizeModal');
    const savePrefsBtn = document.getElementById('save-cookie-prefs');
    const analyticsToggle = document.getElementById('pref-analytics');
    const marketingToggle = document.getElementById('pref-marketing');
    const labelTitle = document.getElementById('cookieCustomizeLabel');
    const labelDesc = document.getElementById('cookie-pref-desc');
    const labelEssential = document.getElementById('label-essential');
    const labelAnalytics = document.getElementById('label-analytics');
    const labelMarketing = document.getElementById('label-marketing');
    const cancelBtn = document.getElementById('modal-cancel-btn');

    // Set texts
    cookieMsg.textContent = t.banner.message;
    acceptBtn.textContent = t.banner.accept;
    rejectBtn.textContent = t.banner.reject;
    customizeBtn.textContent = t.banner.customize;
    labelTitle.textContent = t.modal.title;
    labelDesc.textContent = t.modal.desc;
    labelEssential.textContent = t.modal.essential;
    labelAnalytics.textContent = t.modal.analytics;
    labelMarketing.textContent = t.modal.marketing;
    cancelBtn.textContent = t.modal.cancel;
    savePrefsBtn.textContent = t.modal.save;

    // Show banner only if not accepted
    if (!localStorage.getItem(COOKIE_KEY)) {
      cookieBanner.classList.remove('d-none');
    }

    acceptBtn.addEventListener('click', () => {
      localStorage.setItem(COOKIE_KEY, 'all');
      localStorage.setItem(PREFS_KEY, JSON.stringify({ essential: true, analytics: true, marketing: true }));
      cookieBanner.classList.add('d-none');
    });

    rejectBtn.addEventListener('click', () => {
      localStorage.setItem(COOKIE_KEY, 'essential');
      localStorage.setItem(PREFS_KEY, JSON.stringify({ essential: true, analytics: false, marketing: false }));
      cookieBanner.classList.add('d-none');
    });

    customizeBtn.addEventListener('click', () => {
      const stored = localStorage.getItem(PREFS_KEY);
      if (stored) {
        const prefs = JSON.parse(stored);
        analyticsToggle.checked = !!prefs.analytics;
        marketingToggle.checked = !!prefs.marketing;
      }
      if (customizeModalEl) {
        const modal = new bootstrap.Modal(customizeModalEl);
        modal.show();
      }
    });

    savePrefsBtn.addEventListener('click', () => {
      const prefs = {
        essential: true,
        analytics: analyticsToggle.checked,
        marketing: marketingToggle.checked
      };
      localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
      localStorage.setItem(COOKIE_KEY, 'custom');
      cookieBanner.classList.add('d-none');
      bootstrap.Modal.getInstance(customizeModalEl)?.hide();
    });
  })();

  // ---------- 🔺 Triangle blink until user opens one ----------
  if (window.innerWidth <= 768) {
    if (!localStorage.getItem('triangle_learned')) {
      // show animation
      document.body.classList.remove('user-learned');

      // detect first open on any row
      document.addEventListener('click', e => {
        const td = e.target.closest('td.dtr-control');
        if (td) {
          localStorage.setItem('triangle_learned', '1');
          document.body.classList.add('user-learned');
        }
      });
    } else {
      // already interacted before
      document.body.classList.add('user-learned');
    }
  }

  // ---------- 🔧 Fix ResizeObserver + DataTable recalcs ----------
  (function fixResizeObserverAndDT() {
    // 1) Silence Chrome's harmless ResizeObserver warning
    const re = /ResizeObserver loop completed with undelivered notifications/;
    const origErr = console.error;
    console.error = (...args) => {
      if (args[0] && re.test(String(args[0]))) return; // ignore only this warning
      origErr.apply(console, args);
    };

    // 2) Safe DataTable adjust helper
    const adjustDT = () => {
      if (!window.jQuery || !$.fn.DataTable) return;
      const $t = $('#airports-table');
      if (!$t.length) return;
      const dt = $t.DataTable();
      dt.columns.adjust().responsive.recalc();
    };

    // 3) Run after load/layout settle
    window.addEventListener('load', () => setTimeout(adjustDT, 250));

    // 4) Recalc after major layout changes
    window.addEventListener('resize', () => requestAnimationFrame(adjustDT));
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') setTimeout(adjustDT, 150);
    });

    $('#mapModal').on('shown.bs.modal', () => setTimeout(adjustDT, 300));
  })();

  // ---------- Destination click interception ----------
  // Run only on home or stats pages to prevent loops on destination pages
  // ---------- Destination click interception ----------
  // Run only on home or stats pages to prevent loops on destination pages
  if (!window.location.pathname.startsWith("/destinations/")) {
    document.addEventListener("click", function (e) {
      const a = e.target.closest("a[href^='/destinations/']");
      if (!a) return;

      e.preventDefault();

      // --- Analytics (non-blocking) ---
      const iata = (a.dataset.iata || a.href.split("/").pop()).toUpperCase();
      const city = a.dataset.city || "";
      const country = a.dataset.country || "";
      fetch("/api/analytics/click", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ iata, city, country }),
      }).catch(() => { });

      // --- Loader ---
      const loader = document.getElementById("global-loader");
      const textEl = document.querySelector("#global-loader .loader-text");
      const lang = document.documentElement.lang || "en";

      if (loader && textEl) {
        textEl.textContent =
          lang === "he" ? "טוען את נתוני היעד…" : "Destination loading…";

        loader.style.display = "flex";
        // ✅ Force paint on mobile browsers
        loader.style.transform = "translateZ(0)";
        void loader.offsetHeight;

        // ✅ Double rAF ensures repaint before navigation
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            const url = new URL(a.href, window.location.origin);
            if (lang === "he") url.searchParams.set("lang", "he");
            window.location.href = url.toString();
          });
        });
      } else {
        // Fallback: normal redirect if loader missing
        window.location.href = a.href;
      }
    });
  }
});
// ---------- Handle BFCache restore + Android Back ----------
window.addEventListener("pageshow", (event) => {
  const loader = document.getElementById("global-loader");
  const navEntry = performance.getEntriesByType("navigation")[0];

  // Hide loader ONLY when coming back (not on normal load)
  const isBackRestore =
    event.persisted || (navEntry && navEntry.type === "back_forward");

  if (isBackRestore && loader) loader.style.display = "none";
});


