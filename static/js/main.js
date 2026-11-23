(function () {
  'use strict';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var el = function (tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };

  /* ---------------- theme ---------------- */
  $('#themeBtn').addEventListener('click', function () {
    var root = document.documentElement;
    var cur = root.getAttribute('data-theme') ||
      (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    var next = cur === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('mm-theme', next); } catch (e) {}
  });

  /* ---------------- generic toggle helper ---------------- */
  function bindToggle(group, onChange) {
    $$('button', group).forEach(function (b) {
      b.addEventListener('click', function () {
        $$('button', group).forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
        onChange(b.getAttribute('data-v'));
      });
    });
  }

  /* ---------------- videos: play when visible, keep rate ---------------- */
  function keepRate(v) {
    var apply = function () { var r = parseFloat(v.getAttribute('data-rate') || '1'); if (v.playbackRate !== r) v.playbackRate = r; };
    v.addEventListener('loadedmetadata', apply);
    v.addEventListener('play', apply);
    apply();
  }
  var io = 'IntersectionObserver' in window ? new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      var v = en.target;
      if (en.isIntersecting) { var p = v.play(); if (p && p.catch) p.catch(function () {}); }
      else v.pause();
    });
  }, { threshold: 0.25 }) : null;
  function watch(v) {
    v.muted = true; keepRate(v);
    if (io) { v.removeAttribute('autoplay'); io.observe(v); }
  }

  /* ---------------- problem section: film strip ---------------- */
  var film = $('#filmAll');
  for (var i = 0; i < 24; i++) {
    var im = el('img'); im.src = 'static/images/seal/s' + (i % 10) + '.jpg'; im.alt = ''; im.loading = 'lazy';
    film.appendChild(im);
  }
  var stream = $('#streamImgs');
  [0, 1, 2, 3].forEach(function (k) { var im = el('img'); im.src = 'static/images/frames/f' + k + '.jpg'; im.alt = ''; stream.appendChild(im); });

  /* ---------------- VMBench explorer ---------------- */
  var CATS = {
    spatial: { name: 'Spatial', color: 'var(--spatial)' },
    appearance: { name: 'Appearance', color: 'var(--appearance)' },
    both: { name: 'Spatial & Appearance', color: 'var(--both)' }
  };
  var TASKS = [
    { id: 'spring_compression', name: 'Spring Compression', cat: 'spatial', n: 3, seg: 1,
      goal: 'Compress the spring back to its deepest previously reached position.',
      mem: 'How deep the spring was pushed', use: 'Push it to the same depth' },
    { id: 'flash_beacon', name: 'Flash Beacon', cat: 'spatial', n: 4, seg: 2,
      goal: 'Move the cube to where each beacon briefly appeared.',
      mem: 'Two beacon locations, each shown briefly', use: 'Visit both spots in order' },
    { id: 'explore_cube', name: 'Explore Cube', cat: 'spatial', n: 4, seg: 3,
      goal: 'Search all the cups and retrieve the hidden cube.',
      mem: 'What was under each cup', use: 'Retrieve the hidden cube' },
    { id: 'hidden_pattern', name: 'Hidden Pattern', cat: 'appearance', n: 3, seg: 2,
      goal: 'Push the box whose hidden pattern matches the pattern on the card.',
      mem: 'Which pattern is in which box, and the card', use: 'Push the matching box' },
    { id: 'drawer_match', name: 'Drawer Match', cat: 'appearance', n: 4, seg: 3,
      goal: 'Open the upper or middle drawer whose pattern matches the bottom drawer.',
      mem: 'The pattern inside each drawer', use: 'Reopen the matching drawer' },
    { id: 'scroll_discovery', name: 'Scroll Discovery', cat: 'appearance', n: 4, seg: 3,
      goal: 'Reveal the scroll patterns, then stack the two matching scrolls.',
      mem: 'The pattern on each scroll', use: 'Stack the matching pair' },
    { id: 'placement_imitation', name: 'Placement Imitation', cat: 'both', n: 3, seg: 2,
      goal: 'Move each cube to the position shown in the demonstration.',
      mem: 'Where each colored cube paused', use: 'Put each cube back there' },
    { id: 'mine_sweeper', name: 'Mine Sweeper', cat: 'both', n: 4, seg: 2,
      goal: 'Place each cube where the detector briefly showed the same color.',
      mem: 'Which color flashed where', use: 'Place each cube on its color' },
    { id: 'seal_checking', name: 'Seal Checking', cat: 'both', n: 4, seg: 2,
      goal: 'Lift the stamps, then place each cube on the matching revealed color.',
      mem: 'The color under each stamp', use: 'Place each cube on its color' }
  ];
  var curTask = 0, curLook = 'std';
  var list = $('#taskList');
  ['spatial', 'appearance', 'both'].forEach(function (c) {
    list.appendChild(el('div', 'cat-h', '<i style="background:' + CATS[c].color + '"></i>' + CATS[c].name));
    TASKS.forEach(function (t, idx) {
      if (t.cat !== c) return;
      var b = el('button', 'task-btn', t.name);
      b.setAttribute('role', 'tab'); b.setAttribute('data-i', idx);
      b.addEventListener('click', function () { curTask = idx; renderTask(); });
      list.appendChild(b);
    });
  });
  function renderTask() {
    var t = TASKS[curTask];
    $$('.task-btn', list).forEach(function (b) { b.setAttribute('aria-selected', +b.getAttribute('data-i') === curTask ? 'true' : 'false'); });
    var chip = $('#tCat'); chip.textContent = CATS[t.cat].name; chip.style.background = CATS[t.cat].color;
    $('#tName').textContent = t.name;
    $('#tGoal').textContent = t.goal;
    $('#tMem').textContent = t.mem;
    $('#tUse').textContent = t.use;
    $('#tSeg').textContent = t.seg;
    var seq = $('#tSeq'); seq.innerHTML = '';
    if (curLook === 'ood') {
      var im = el('img'); im.src = 'static/images/tasks/' + t.id + '_ood.jpg'; im.alt = t.name + ' with unseen appearance';
      im.style.maxWidth = '320px'; seq.appendChild(im);
      seq.appendChild(el('span', 'muted small', 'New room, table, lighting and distractors. The task and its key moments are unchanged.'));
      return;
    }
    for (var k = 1; k <= t.n; k++) {
      if (k > 1) seq.appendChild(el('span', 'arr', '→'));
      var img = el('img'); img.src = 'static/images/tasks/' + t.id + '_0' + k + '.jpg'; img.alt = t.name + ' step ' + k;
      seq.appendChild(img);
    }
  }
  bindToggle($('#oodToggle'), function (v) { curLook = v; renderTask(); });
  renderTask();

  /* ---------------- segment widget ---------------- */
  var SEG = [2, 7], NFR = 10, anchor = 4;
  var frames = $('#frames');
  function renderFrames() {
    var miss = $('#missChk').checked;
    frames.innerHTML = '';
    var band = el('div', 'segband', 'key segment: cube visible');
    band.style.left = 'calc(' + (SEG[0] * 10) + '% + 1px)';
    band.style.width = 'calc(' + ((SEG[1] - SEG[0] + 1) * 10) + '% - 4px)';
    frames.appendChild(band);
    var caught = miss ? Math.min(anchor + 1, SEG[1] + 1) : anchor; // frame where it gets stored
    for (var i = 0; i < NFR; i++) {
      var f = el('div', 'fr');
      var inSeg = i >= SEG[0] && i <= SEG[1];
      if (inSeg) f.classList.add('inseg');
      var img = el('img'); img.src = 'static/images/frames/f' + i + '.jpg'; img.alt = 'frame ' + (i + 1);
      f.appendChild(img);
      var lab;
      if (!inSeg) lab = el('div', 'lab neg', 'skip');
      else if (i < anchor) { lab = el('div', 'lab mask', '–'); f.classList.add('masked'); }
      else if (i === anchor) {
        if (miss) lab = el('div', 'lab miss', 'missed');
        else { lab = el('div', 'lab pos', 'KEEP'); f.classList.add('keep'); }
      }
      else if (miss && i === caught) { lab = el('div', 'lab pos', 'KEEP'); f.classList.add('keep'); }
      else lab = el('div', 'lab neg', 'skip');
      f.appendChild(lab);
      frames.appendChild(f);
    }
    var a = anchor + 1;
    var msg;
    if (!miss) {
      msg = 'This epoch the anchor is frame <b>' + a + '</b>. Earlier frames in the segment are not penalized. Later ones are negative, because the moment is already in memory. Next epoch a different anchor is drawn, so every frame of the moment can be a valid pick.';
    } else if (anchor < SEG[1]) {
      msg = 'With <b>scheduled sampling</b>, the memory history comes from the model\'s own predictions. Here it skipped frame ' + a + '. The cube is still visible, so frame <b>' + (a + 1) + '</b> stays positive: the memorizer learns to recover from a missed anchor.';
    } else {
      msg = 'The memorizer skipped frame ' + a + ', the last frame of the moment. No valid frame is left, so this case shows why catching the moment <b>early</b> matters.';
    }
    $('#explain').innerHTML = msg;
  }
  $('#resample').addEventListener('click', function () {
    var prev = anchor;
    while (anchor === prev) anchor = SEG[0] + Math.floor(Math.random() * (SEG[1] - SEG[0] + 1));
    renderFrames();
  });
  $('#missChk').addEventListener('change', renderFrames);
  renderFrames();

  /* ---------------- tooltip ---------------- */
  var tip = $('#tip');
  function showTip(html, ev) {
    tip.innerHTML = html; tip.classList.add('show'); moveTip(ev);
  }
  function moveTip(ev) {
    var x = ev.clientX + 14, y = ev.clientY + 14;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - 14;
    if (y + h > window.innerHeight - 8) y = ev.clientY - h - 14;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  function hideTip() { tip.classList.remove('show'); }

  /* ---------------- results chart ---------------- */
  var VM_TASKS = ['Spring Compression', 'Flash Beacon', 'Explore Cube', 'Hidden Pattern', 'Drawer Match', 'Scroll Discovery', 'Placement Imitation', 'Mine Sweeper', 'Seal Checking'];
  var VM_CAT = ['Spatial', 'Spatial', 'Spatial', 'Appearance', 'Appearance', 'Appearance', 'Spatial & Appearance', 'Spatial & Appearance', 'Spatial & Appearance'];
  // paper column order: Explore, Mine, Hidden, Seal, Spring, Drawer, Flash, Scroll, Placement
  var PAPER_ORDER = ['Explore Cube', 'Mine Sweeper', 'Hidden Pattern', 'Seal Checking', 'Spring Compression', 'Drawer Match', 'Flash Beacon', 'Scroll Discovery', 'Placement Imitation'];
  function reorder(arr) { var m = {}; PAPER_ORDER.forEach(function (t, i) { m[t] = arr[i]; }); return VM_TASKS.map(function (t) { return m[t]; }); }
  var DATA = {
    std: {
      note: 'Standard appearance. 100 rollouts per task, all methods trained jointly on the same 100 demos per task.',
      tasks: VM_TASKS, cats: VM_CAT,
      methods: [
        ['π0.5', reorder([0, 1, 24, 0, 0, 21, 0, 16, 1]), 7.0],
        ['Mem-0', reorder([0, 0, 34, 0, 25, 58, 0, 17, 2]), 15.1],
        ['EventVLA', reorder([2, 1, 24, 2, 34, 51, 15, 29, 6]), 18.2],
        ['NativeMEM', reorder([0, 0, 25, 0, 44, 44, 9, 19, 1]), 15.8],
        ['MomentMem', reorder([2, 93, 32, 64, 52, 47, 38, 22, 99]), 49.9]
      ]
    },
    ood: {
      note: 'Unseen room, table, lighting and distractors. Memory components are reused from the standard appearance; executors are trained on target-appearance demos.',
      tasks: VM_TASKS, cats: VM_CAT,
      methods: [
        ['NativeMEM', reorder([0, 0, 31, 0, 31, 49, 8, 25, 3]), 16.3],
        ['Mem-0', reorder([0, 0, 24, 0, 28, 21, 0, 1, 0]), 8.2],
        ['EventVLA', reorder([0, 1, 40, 0, 30, 48, 6, 4, 4]), 14.8],
        ['MomentMem', reorder([12, 79, 35, 55, 79, 51, 59, 28, 99]), 55.2]
      ]
    },
    rm: {
      note: 'RMBench, an existing memory benchmark. Same architecture, no changes. Mem-0 and EventVLA numbers are from their papers.',
      tasks: ['Observe and Pick Up', 'Rearrange Blocks', 'Put Back Block', 'Swap Blocks', 'Swap T', 'Battery Try', 'Blocks Ranking Try', 'Cover Blocks', 'Press Button'],
      cats: null,
      methods: [
        ['π0.5', [9, 13, 11, 24, 15, 16, 6, 0, 0], 10.4],
        ['Mem-0', [4, 89, 90, 67, 14, 28, 18, 68, 0], 42.0],
        ['EventVLA', [21, 96, 95, 96, 87, 35, 81, 97, 3], 67.9],
        ['NativeMEM', [0, 3, 0, 62, 0, 13, 0, 0, 0], 8.7],
        ['MomentMem', [82, 78, 100, 99, 97, 56, 22, 84, 10], 69.8]
      ]
    }
  };
  var fmt = function (v) { return (Math.round(v * 10) / 10) + '%'; };
  function tipHtml(title, d, getV) {
    var rows = d.methods.map(function (m) { return [m[0], getV(m)]; })
      .sort(function (a, b) { return b[1] - a[1]; });
    return '<h4>' + title + '</h4>' + rows.map(function (r) {
      return '<div class="r' + (r[0] === 'MomentMem' ? ' me' : '') + '"><span>' + r[0] + '</span><span>' + fmt(r[1]) + '</span></div>';
    }).join('');
  }
  // fixed color per method, never re-assigned when the method set changes
  var ORDER = ['MomentMem', 'π0.5', 'Mem-0', 'EventVLA', 'NativeMEM'];
  var SLOT = { 'MomentMem': 'm1', 'π0.5': 'm2', 'Mem-0': 'm3', 'EventVLA': 'm4', 'NativeMEM': 'm5' };
  function sorted(d) {
    return d.methods.slice().sort(function (a, b) { return ORDER.indexOf(a[0]) - ORDER.indexOf(b[0]); });
  }
  function makeRow(name, sub, d, getV, isAvg) {
    var row = el('div', 'brow' + (isAvg ? ' avg' : ''));
    row.appendChild(el('div', 'name', name + (sub ? '<small>' + sub + '</small>' : '')));
    var tr = el('div', 'track');
    var top = Math.max.apply(null, d.methods.map(getV));
    sorted(d).forEach(function (m) {
      var v = getV(m);
      var line = el('div', 'bar-line ' + SLOT[m[0]] + (m[0] === 'MomentMem' ? ' o' : ''));
      var bar = el('div', 'bar'); bar.style.width = '0%';
      bar.setAttribute('data-w', v);
      line.appendChild(bar);
      line.appendChild(el('span', 'v' + (v === top ? ' top' : ''), fmt(v)));
      line.setAttribute('aria-label', m[0] + ' ' + fmt(v));
      tr.appendChild(line);
    });
    row.appendChild(tr);
    row.addEventListener('mouseenter', function (e) { showTip(tipHtml(name, d, getV), e); });
    row.addEventListener('mousemove', moveTip);
    row.addEventListener('mouseleave', hideTip);
    return row;
  }
  function renderLegend(d) {
    $('#benchLegend').innerHTML = sorted(d).map(function (m) {
      return '<span><i class="sw ' + SLOT[m[0]] + '"></i>' + m[0] + '</span>';
    }).join('');
  }
  function renderBars(key) {
    var d = DATA[key], box = $('#bars');
    $('#benchNote').textContent = d.note;
    renderLegend(d);
    box.innerHTML = '';
    d.tasks.forEach(function (t, i) {
      box.appendChild(makeRow(t, d.cats ? d.cats[i] : '', d, function (m) { return m[1][i]; }, false));
    });
    box.appendChild(makeRow('Average', '', d, function (m) { return m[2]; }, true));
    requestAnimationFrame(function () {
      $$('.bar', box).forEach(function (b) { b.style.width = 'calc((100% - 52px) * ' + (b.getAttribute('data-w') / 100) + ')'; });
    });
  }
  bindToggle($('#benchToggle'), renderBars);
  renderBars('std');

  /* ---------------- small charts ---------------- */
  function hbars(target, rows) {
    var box = $(target);
    rows.forEach(function (r) {
      var row = el('div', 'hb' + (r.me ? ' me' : ''));
      row.appendChild(el('span', '', r.label));
      var tr = el('div', 'tr');
      var bar = el('div', 'bar ' + (r.me ? 'o' : 'g')); bar.style.width = (r.v * 0.8) + '%';
      tr.appendChild(bar); tr.appendChild(el('span', 'v', r.v.toFixed(1) + '%'));
      row.appendChild(tr); box.appendChild(row);
    });
  }
  hbars('#ablation', [
    { label: 'Point annotation', v: 6.9 },
    { label: 'w/o scheduled sampling', v: 19.0 },
    { label: 'Full MomentMem', v: 49.9, me: true }
  ]);
  hbars('#fewshot', [
    { label: 'From Qwen · 5', v: 29.0 },
    { label: 'Ours · 5', v: 43.7, me: true },
    { label: 'Full · 100', v: 44.3 }
  ]);
  $('#fewshot').appendChild(el('p', 'small muted', 'Task success (%); numbers are memory-annotated demos. “From Qwen” adapts directly from pretrained Qwen3-VL; “Ours” starts from the memorizer trained on the other eight tasks; “Full” annotates all 100 demonstrations.'));
  $('#fewshot p:last-child').style.margin = '12px 0 0';

  var waff = $('#waffles');
  [['Placement Imitation', 3, 14], ['Explore Cube', 11, 17]].forEach(function (t) {
    waff.appendChild(el('div', 'small', '<b>' + t[0] + '</b>'));
    [['MomentMem', t[2], 'o'], ['π0.5', t[1], 'b']].forEach(function (m) {
      var r = el('div', 'waffle-row');
      r.appendChild(el('div', 'lbl', '<span>' + m[0] + '</span><b>' + m[1] + '/30</b>'));
      var w = el('div', 'waffle ' + m[2]);
      for (var i = 0; i < 30; i++) w.appendChild(el('i', i < m[1] ? 'on' : ''));
      r.appendChild(w); waff.appendChild(r);
    });
  });

  /* ---------------- real-robot videos ---------------- */
  var OURS = [
    { f: 'ours_1C', title: 'Cube under the 1st cup', dur: '0:50' },
    { f: 'ours_2C', title: 'Cube under the 2nd cup', dur: '1:11' },
    { f: 'ours_3C', title: 'Cube under the 3rd cup', dur: '2:15' }
  ];
  var PI = [
    { f: 'pi_dup', title: 'Re-checks cups it already lifted', dur: '4:39' },
    { f: 'pi_change', title: 'Switches target mid-search', dur: '1:43' },
    { f: 'pi_return', title: 'Returns to a checked cup', dur: '2:00' }
  ];
  var speed = 4;
  function addVids(target, items, tagCls, tagTxt) {
    var g = $(target);
    items.forEach(function (it) {
      var c = el('div', 'card vid-card');
      var v = el('video');
      v.src = 'static/videos/real/' + it.f + '.mp4';
      v.poster = 'static/videos/real/' + it.f + '.jpg';
      v.muted = true; v.loop = true; v.controls = true; v.playsInline = true; v.preload = 'metadata';
      v.setAttribute('playsinline', ''); v.setAttribute('data-rate', speed); v.setAttribute('data-real', '');
      c.appendChild(v);
      c.appendChild(el('div', 'cap', '<span class="tag ' + tagCls + '">' + tagTxt + '</span><div class="txt"><b>' + it.title + '</b></div>'));
      g.appendChild(c);
      watch(v);
    });
  }
  addVids('#oursGrid', OURS, 'ours', 'Ours');
  addVids('#piGrid', PI, 'base', 'π0.5');
  /* hero videos */
  $$('.hero-videos video').forEach(watch);

  /* citation: copy BibTeX */
  var copyBtn = $('#copyBib');
  copyBtn.addEventListener('click', function () {
    var text = $('#bibtex').textContent;
    var done = function () { copyBtn.textContent = 'Copied'; setTimeout(function () { copyBtn.textContent = 'Copy'; }, 1500); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () { fallback(); });
    } else fallback();
    function fallback() {
      var ta = el('textarea'); ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select();
      try { document.execCommand('copy'); done(); } catch (e) {}
      document.body.removeChild(ta);
    }
  });

})();
