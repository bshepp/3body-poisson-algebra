// bracket_widget.js — §2 Poisson bracket calculator
// Pick f(q,p) and g(q,p) from a small menu; show {f,g} as a heatmap.
// Bracket computed numerically: {f,g} = ∂_q f · ∂_p g − ∂_p f · ∂_q g

(function () {
  const cf = document.getElementById('br-f');
  const cg = document.getElementById('br-g');
  const cb = document.getElementById('br-fg');
  if (!cf || !cg || !cb) return;
  const W = cf.width, H = cf.height;

  // World window
  const Q_MIN = -2.5, Q_MAX = 2.5;
  const P_MIN = -2.5, P_MAX = 2.5;

  // Observable library — each entry: { label (TeX), fn(q,p), bracketLabel (optional override) }
  const OBS = {
    q:        { tex: 'q',                fn: (q,p) => q },
    p:        { tex: 'p',                fn: (q,p) => p },
    q2:       { tex: '\\tfrac12 q^2',    fn: (q,p) => 0.5*q*q },
    p2:       { tex: '\\tfrac12 p^2',    fn: (q,p) => 0.5*p*p },
    qp:       { tex: 'q\\,p',            fn: (q,p) => q*p },
    Hosc:     { tex: '\\tfrac12(q^2+p^2)', fn: (q,p) => 0.5*(q*q + p*p) },
    Hpend:    { tex: '\\tfrac12 p^2 - \\cos q', fn: (q,p) => 0.5*p*p - Math.cos(q) },
    Lz:       { tex: 'q^2 - p^2',        fn: (q,p) => q*q - p*p }
  };

  // Pretty-print a known bracket, when we have a closed form. Falls back to "{f,g}".
  // Keys: "f|g" with f,g from the OBS keys.
  const KNOWN = {
    'q|p':       '1',
    'p|q':       '-1',
    'q|q':       '0',
    'p|p':       '0',
    'q|q2':      '0',
    'q|p2':      'p',
    'p|q2':      '-q',
    'p|p2':      '0',
    'q2|p2':     '2\\,q\\,p',
    'q|qp':      'q',
    'p|qp':      '-p',
    'q|Hosc':    'p',
    'p|Hosc':    '-q',
    'q|Hpend':   'p',
    'p|Hpend':   '-\\sin q',
    'Hosc|Hpend':'p\\,(\\sin q - q)',
    'q2|qp':     '-q^2',
    'p2|qp':     'p^2',
    'q|Lz':      '0',
    'p|Lz':      '-2q',
    'Hosc|Lz':   '-4\\,q\\,p'
  };

  function bracketName(fk, gk) {
    if (fk === gk) return '0';
    if (KNOWN[fk + '|' + gk]) return KNOWN[fk + '|' + gk];
    if (KNOWN[gk + '|' + fk]) return '-(' + KNOWN[gk + '|' + fk] + ')';
    return '\\{f,g\\}';  // unknown closed form
  }

  // ---- numerical bracket ----
  const EPS = 1e-3;
  function bracket(fn, gn, q, p) {
    const fq = (fn(q+EPS, p) - fn(q-EPS, p)) / (2*EPS);
    const fp = (fn(q, p+EPS) - fn(q, p-EPS)) / (2*EPS);
    const gq = (gn(q+EPS, p) - gn(q-EPS, p)) / (2*EPS);
    const gp = (gn(q, p+EPS) - gn(q, p-EPS)) / (2*EPS);
    return fq*gp - fp*gq;
  }

  // ---- rendering ----
  function paint(canvas, valFn) {
    const ctx = canvas.getContext('2d');
    // first pass: compute values + range
    const NX = 80, NY = 80;
    const grid = new Float32Array(NX*NY);
    let vmin = Infinity, vmax = -Infinity;
    for (let j = 0; j < NY; j++) {
      const p = P_MIN + (j + 0.5) / NY * (P_MAX - P_MIN);
      for (let i = 0; i < NX; i++) {
        const q = Q_MIN + (i + 0.5) / NX * (Q_MAX - Q_MIN);
        const v = valFn(q, p);
        grid[j*NX + i] = v;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      }
    }
    // symmetric scale around 0
    const M = Math.max(Math.abs(vmin), Math.abs(vmax)) || 1;

    // draw cell-by-cell
    const cw = W / NX, ch = H / NY;
    for (let j = 0; j < NY; j++) {
      for (let i = 0; i < NX; i++) {
        const v = grid[j*NX + i] / M;  // in [-1, 1]
        ctx.fillStyle = colormap(v);
        ctx.fillRect(i*cw, H - (j+1)*ch, Math.ceil(cw)+1, Math.ceil(ch)+1);
      }
    }

    // axes
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, H/2); ctx.lineTo(W, H/2);
    ctx.moveTo(W/2, 0); ctx.lineTo(W/2, H);
    ctx.stroke();
  }

  // diverging colormap: blue (-1) → black (0) → orange (+1)
  function colormap(v) {
    v = Math.max(-1, Math.min(1, v));
    if (v >= 0) {
      const t = v;
      const r = Math.round(17  + t * (184 - 17));
      const g = Math.round(17  + t * (65  - 17));
      const b = Math.round(17  + t * (14  - 17));
      return `rgb(${r},${g},${b})`;
    } else {
      const t = -v;
      const r = Math.round(17  + t * (58  - 17));
      const g = Math.round(17  + t * (160 - 17));
      const b = Math.round(17  + t * (255 - 17));
      return `rgb(${r},${g},${b})`;
    }
  }

  function update() {
    const fk = cf.dataset.choice || 'q';
    const gk = cg.dataset.choice || 'p';
    const F = OBS[fk], G = OBS[gk];
    paint(cf, F.fn);
    paint(cg, G.fn);
    paint(cb, (q,p) => bracket(F.fn, G.fn, q, p));
    // labels (KaTeX)
    const labF = document.getElementById('br-flabel');
    const labG = document.getElementById('br-glabel');
    const labB = document.getElementById('br-fglabel');
    labF.innerHTML = `$f(q,p) = ${F.tex}$`;
    labG.innerHTML = `$g(q,p) = ${G.tex}$`;
    labB.innerHTML = `$\\{f,g\\} = ${bracketName(fk, gk)}$`;
    if (window.renderMathInElement) {
      [labF, labG, labB].forEach(el =>
        renderMathInElement(el, { delimiters: [
          { left: '$', right: '$', display: false }
        ]}));
    }
  }

  // ---- UI: dropdowns ----
  function buildSelect(sel, current) {
    sel.innerHTML = '';
    Object.keys(OBS).forEach(k => {
      const opt = document.createElement('option');
      opt.value = k;
      opt.textContent = OBS[k].tex.replace(/\\tfrac12 /g,'½').replace(/\\,/g,'').replace(/\\cos/g,'cos').replace(/\\sin/g,'sin').replace(/\\/g,'').replace(/\^2/g,'²');
      if (k === current) opt.selected = true;
      sel.appendChild(opt);
    });
  }
  const selF = document.getElementById('br-fsel');
  const selG = document.getElementById('br-gsel');
  buildSelect(selF, 'q');
  buildSelect(selG, 'p');
  cf.dataset.choice = 'q';
  cg.dataset.choice = 'p';
  selF.addEventListener('change', () => { cf.dataset.choice = selF.value; update(); });
  selG.addEventListener('change', () => { cg.dataset.choice = selG.value; update(); });

  update();
})();
