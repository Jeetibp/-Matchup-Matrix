/* UI enhancements — tooltips, filter chips, sortable headers,
   quick filter, CSV export, number formatting, performance badges.
   Pure presentation layer. Does not call any backend.
*/
(function () {
  'use strict';

  const GLOSSARY = {
    'Runs': 'Total runs scored',
    'Balls': 'Total balls faced/bowled',
    'Matches': 'Number of matches',
    'Innings': 'Number of innings',
    'Avg': 'Average — runs per dismissal',
    'AVG': 'Average — runs per dismissal (batting) or per wicket (bowling)',
    'SR': 'Strike Rate — runs per 100 balls (batting) or balls per wicket (bowling)',
    '100s': 'Centuries (100+ scores)',
    '50s': 'Half-centuries (50–99 scores)',
    'HS': 'Highest individual / team score',
    'LS': 'Lowest team score',
    'HC': 'Highest successful chase',
    'LD': 'Lowest defended total',
    'RPI': 'Runs Per Innings',
    'Dot%': 'Percentage of dot balls',
    'BPD': 'Balls Per Dismissal',
    'BPB': 'Balls Per Boundary',
    'ECO': 'Economy — runs conceded per over',
    'Wickets': 'Total wickets taken',
    'Best': 'Best bowling figures in an innings',
    'wickets_1': '1st-innings wickets',
    'wickets_2': '2nd-innings wickets',
    'five_wkts': '5-wicket hauls',
    '4s': 'Fours hit',
    '6s': 'Sixes hit',
    'PP1': 'Powerplay (Overs 1–6)',
    'PP2': 'Middle 1 (Overs 7–10)',
    'PP3': 'Middle 2 (Overs 11–15)',
    'PP4': 'Death (Overs 16–20)',
    'Bowler': 'Bowler name',
    'Batsman': 'Batsman name',
    'Team': 'Team name',
    'Venue': 'Match venue',
    'Boundary%': 'Percentage of balls that went for a boundary',
    'win_pct': 'Win percentage',
    'win_pct_1st': 'Win % when batting first',
    'win_pct_2nd': 'Win % when batting second',
  };

  // ---------- 1. Tooltips on stat headers ----------
  function applyTooltips() {
    document.querySelectorAll('.table thead th').forEach((th) => {
      const text = th.textContent.trim();
      const def = GLOSSARY[text];
      if (def && !th.hasAttribute('title')) {
        th.setAttribute('title', def);
        th.setAttribute('data-bs-toggle', 'tooltip');
        th.style.cursor = 'help';
      }
    });
    if (window.bootstrap && window.bootstrap.Tooltip) {
      document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach((el) => {
        if (!window.bootstrap.Tooltip.getInstance(el)) {
          new window.bootstrap.Tooltip(el, { container: 'body', trigger: 'hover focus' });
        }
      });
    }
  }

  // ---------- 2. Number formatting (add commas to large ints) ----------
  function formatNumbers() {
    document.querySelectorAll('.table tbody td').forEach((td) => {
      const t = td.textContent.trim();
      if (/^\d{4,}$/.test(t)) {
        td.textContent = parseInt(t, 10).toLocaleString('en-IN');
      }
    });
  }

  // ---------- 3. Performance colour badges ----------
  function colorBadges() {
    document.querySelectorAll('.table').forEach((tbl) => {
      const headers = Array.from(tbl.querySelectorAll('thead th')).map((h) => h.textContent.trim());
      if (!headers.length) return;
      const isBat = headers.includes('SR') && (headers.includes('100s') || headers.includes('Batsman'));
      const isBowl = headers.includes('ECO') && (headers.includes('Wickets') || headers.includes('Bowler'));
      const findIdx = (name) => headers.indexOf(name);
      const idxAvg = findIdx('AVG') >= 0 ? findIdx('AVG') : findIdx('Avg');
      const idxSR = findIdx('SR');
      const idxECO = findIdx('ECO');

      tbl.querySelectorAll('tbody tr').forEach((tr) => {
        const cells = tr.querySelectorAll('td');
        if (!cells.length) return;
        if (isBat) {
          if (idxAvg >= 0) tagPerf(cells[idxAvg], 40, 25, false);
          if (idxSR >= 0) tagPerf(cells[idxSR], 140, 110, false);
        }
        if (isBowl) {
          if (idxAvg >= 0) tagPerf(cells[idxAvg], 25, 35, true);
          if (idxECO >= 0) tagPerf(cells[idxECO], 7, 9, true);
        }
      });
    });
  }
  function tagPerf(cell, goodTh, badTh, lowerIsBetter) {
    if (!cell || cell.classList.contains('perf-tagged')) return;
    const val = parseFloat(cell.textContent.replace(/,/g, ''));
    if (isNaN(val) || val === 0) return;
    cell.classList.add('perf-tagged');
    let cls = '';
    if (lowerIsBetter) {
      if (val <= goodTh) cls = 'perf-good';
      else if (val >= badTh) cls = 'perf-bad';
    } else {
      if (val >= goodTh) cls = 'perf-good';
      else if (val <= badTh) cls = 'perf-bad';
    }
    if (cls) cell.classList.add(cls);
  }

  // ---------- 4. Sortable column headers ----------
  function makeSortable() {
    document.querySelectorAll('.table').forEach((tbl) => {
      const tbody = tbl.querySelector('tbody');
      if (!tbody || tbody.querySelectorAll('tr').length < 2) return;
      const headers = tbl.querySelectorAll('thead th');
      if (!headers.length) return;
      headers.forEach((th, idx) => {
        if (th.dataset.sortAttached) return;
        th.dataset.sortAttached = '1';
        th.classList.add('sortable-th');
        const ind = document.createElement('span');
        ind.className = 'sort-ind';
        ind.textContent = '⇅';
        th.appendChild(document.createTextNode(' '));
        th.appendChild(ind);
        let dir = 0;
        th.addEventListener('click', () => {
          dir = dir === 1 ? -1 : 1;
          headers.forEach((h) => {
            const i = h.querySelector('.sort-ind');
            if (i) i.textContent = '⇅';
            h.classList.remove('sorted-asc', 'sorted-desc');
          });
          ind.textContent = dir === 1 ? '▲' : '▼';
          th.classList.add(dir === 1 ? 'sorted-asc' : 'sorted-desc');
          const rows = Array.from(tbody.querySelectorAll('tr'));
          rows.sort((a, b) => {
            const av = (a.children[idx] && a.children[idx].textContent || '').trim();
            const bv = (b.children[idx] && b.children[idx].textContent || '').trim();
            const an = parseFloat(av.replace(/,/g, ''));
            const bn = parseFloat(bv.replace(/,/g, ''));
            const numeric = !isNaN(an) && !isNaN(bn);
            const cmp = numeric ? an - bn : av.localeCompare(bv);
            return dir === 1 ? cmp : -cmp;
          });
          rows.forEach((r) => tbody.appendChild(r));
        });
      });
    });
  }

  // ---------- 5. Quick-filter + CSV export toolbar ----------
  function addTableToolbars() {
    document.querySelectorAll('.table-responsive').forEach((wrap) => {
      if (wrap.dataset.toolbarAttached) return;
      const tbl = wrap.querySelector('table.table');
      if (!tbl) return;
      const tbody = tbl.querySelector('tbody');
      if (!tbody) return;
      const rows = tbody.querySelectorAll('tr');
      const headers = tbl.querySelectorAll('thead th');
      if (rows.length < 5 || headers.length < 3) return;
      wrap.dataset.toolbarAttached = '1';

      const toolbar = document.createElement('div');
      toolbar.className = 'table-toolbar d-flex gap-2 mb-2 flex-wrap align-items-center';
      toolbar.innerHTML = '' +
        '<input type="text" class="form-control form-control-sm quick-filter-input" placeholder="🔍 Filter rows..." style="max-width:240px">' +
        '<button class="btn btn-sm btn-outline-secondary csv-export-btn" type="button" title="Download as CSV">⬇ CSV</button>' +
        '<small class="text-muted ms-auto row-count"></small>';
      wrap.parentNode.insertBefore(toolbar, wrap);

      const input = toolbar.querySelector('.quick-filter-input');
      const countEl = toolbar.querySelector('.row-count');
      const updateCount = () => {
        const all = tbody.querySelectorAll('tr');
        const visible = Array.from(all).filter((r) => r.style.display !== 'none').length;
        countEl.textContent = visible + ' / ' + all.length + ' rows';
      };
      updateCount();
      input.addEventListener('input', () => {
        const q = input.value.toLowerCase().trim();
        tbody.querySelectorAll('tr').forEach((r) => {
          const txt = r.textContent.toLowerCase();
          r.style.display = !q || txt.indexOf(q) !== -1 ? '' : 'none';
        });
        updateCount();
      });
      toolbar.querySelector('.csv-export-btn').addEventListener('click', () => exportCsv(tbl));
    });
  }
  function exportCsv(tbl) {
    const rows = [];
    tbl.querySelectorAll('thead tr').forEach((tr) => {
      rows.push(Array.from(tr.children).map((c) =>
        csvEscape(c.textContent.replace(/[⇅▲▼]/g, '').trim())
      ));
    });
    tbl.querySelectorAll('tbody tr').forEach((tr) => {
      if (tr.style.display === 'none') return;
      rows.push(Array.from(tr.children).map((c) => csvEscape(c.textContent.trim())));
    });
    const csv = rows.map((r) => r.join(',')).join('\n');
    const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'matchup-matrix-' + new Date().toISOString().slice(0, 10) + '.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function csvEscape(s) {
    if (/[",\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
    return s;
  }

  // ---------- 6. Active filter chips ----------
  function renderFilterChips() {
    const params = new URLSearchParams(window.location.search);
    const meaningful = ['min_innings', 'innings_filter', 'season', 'venue', 'team',
                        'team_a', 'team_b', 'compare_teams'];
    const chips = [];
    meaningful.forEach((key) => {
      const vals = params.getAll(key);
      vals.forEach((v) => {
        if (!v || v === 'all' || v === '0') return;
        if (key === 'min_innings' && v === '1') return;
        let label = v;
        if (key === 'innings_filter') label = v === '1' ? '1st Innings' : v === '2' ? '2nd Innings' : v;
        if (key === 'min_innings') label = 'Min ' + v + ' innings';
        chips.push({ key, val: v, display: prettyKey(key) + ': ' + label });
      });
    });
    if (!chips.length) return;
    const container = document.querySelector('.container.mt-4');
    if (!container) return;
    const bar = document.createElement('div');
    bar.className = 'filter-chips d-flex flex-wrap align-items-center gap-2 mb-3';
    bar.innerHTML = chips.map((c) =>
      '<span class="chip">' + escapeHtml(c.display) +
      '<button class="chip-x" data-key="' + c.key + '" data-val="' + encodeURIComponent(c.val) + '" aria-label="Remove">×</button></span>'
    ).join('') + '<button class="btn btn-sm btn-link chips-clear-all">Clear all filters</button>';

    // Insert after the "Tip" alert if present, else at top
    const tip = container.querySelector('#league-hint');
    if (tip && tip.nextSibling) container.insertBefore(bar, tip.nextSibling);
    else container.insertBefore(bar, container.firstChild);

    bar.querySelectorAll('.chip-x').forEach((btn) => {
      btn.addEventListener('click', () => {
        const k = btn.dataset.key;
        const v = decodeURIComponent(btn.dataset.val);
        const p = new URLSearchParams(window.location.search);
        const remaining = p.getAll(k).filter((x) => x !== v);
        p.delete(k);
        remaining.forEach((x) => p.append(k, x));
        window.location.search = p.toString();
      });
    });
    bar.querySelector('.chips-clear-all').addEventListener('click', () => {
      const p = new URLSearchParams();
      const lg = new URLSearchParams(window.location.search).get('league');
      if (lg) p.set('league', lg);
      window.location.search = p.toString();
    });
  }
  function prettyKey(k) {
    return ({
      min_innings: 'Filter', innings_filter: 'Innings', season: 'Season',
      venue: 'Venue', team: 'Team', team_a: 'Team A', team_b: 'Team B',
      compare_teams: 'Compare',
    })[k] || k;
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (m) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
  }

  // ---------- Init ----------
  function init() {
    try { renderFilterChips(); } catch (e) { console.warn('chips:', e); }
    try { applyTooltips(); } catch (e) { console.warn('tooltips:', e); }
    try { formatNumbers(); } catch (e) { console.warn('numfmt:', e); }
    try { colorBadges(); } catch (e) { console.warn('badges:', e); }
    try { makeSortable(); } catch (e) { console.warn('sort:', e); }
    try { addTableToolbars(); } catch (e) { console.warn('toolbar:', e); }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
