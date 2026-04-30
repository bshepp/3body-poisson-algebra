// potentials_widget.js — §3 pairwise potential viewer
// Plots V(r) for several choices and shows the three-body "bond graph":
// three particles wiggling on a circle with H_ij values updating live.

(function () {
  const canvas = document.getElementById('pot-canvas');
  const bond   = document.getElementById('pot-bond');
  const sel    = document.getElementById('pot-sel');
  if (!canvas || !bond || !sel) return;

  const ctx = canvas.getContext('2d');
  const bctx = bond.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const BW = bond.width, BH = bond.height;

  // Potential menu
  const POTS = {
    inv1: { tex: '1/r',         V: r => 1/r,           dom: [0.4, 4] },
    inv2: { tex: '1/r^2',       V: r => 1/(r*r),       dom: [0.4, 4] },
    inv3: { tex: '1/r^3',       V: r => 1/(r*r*r),     dom: [0.5, 4] },
    log:  { tex: '\\log r',     V: r => Math.log(r),   dom: [0.2, 4] },
    r2:   { tex: 'r^2',         V: r => r*r,           dom: [0.2, 4] },
    yuk:  { tex: 'e^{-r}/r',    V: r => Math.exp(-r)/r,dom: [0.4, 4] }
  };

  ['inv1','inv2','inv3','log','r2','yuk'].forEach(k => {
    const opt = document.createElement('option');
    opt.value = k;
    opt.textContent = POTS[k].tex
      .replace(/\^2/g,'²').replace(/\^3/g,'³')
      .replace(/\\log/g,'log').replace(/e\^\{-r\}/g,'e^(-r)');
    sel.appendChild(opt);
  });
  sel.value = 'inv1';

  function plotPotential(key) {
    const P = POTS[key];
    ctx.fillStyle = '#fafaf7';
    ctx.fillRect(0, 0, W, H);

    // sample
    const N = 400;
    const [r0, r1] = P.dom;
    const ys = new Float32Array(N);
    let ymin = Infinity, ymax = -Infinity;
    for (let i = 0; i < N; i++) {
      const r = r0 + (r1 - r0) * i / (N-1);
      const v = P.V(r);
      ys[i] = v;
      if (isFinite(v)) {
        if (v < ymin) ymin = v;
        if (v > ymax) ymax = v;
      }
    }
    // clip top for divergent potentials
    const cap = ymin + (ymax - ymin) * 1.0;
    ymax = Math.min(ymax, cap);

    const padL = 38, padR = 14, padT = 14, padB = 28;
    const px = i => padL + (W - padL - padR) * i / (N-1);
    const py = v => padT + (H - padT - padB) * (1 - (v - ymin) / (ymax - ymin || 1));

    // axes
    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB);
    ctx.lineTo(W - padR, H - padB);
    ctx.stroke();
    ctx.fillStyle = '#666';
    ctx.font = '11px Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('r', W/2, H - 6);
    ctx.save();
    ctx.translate(12, H/2); ctx.rotate(-Math.PI/2);
    ctx.fillText('V(r)', 0, 0);
    ctx.restore();
    ctx.textAlign = 'right';
    ctx.fillText(r0.toFixed(1), padL, H - padB + 14);
    ctx.fillText(r1.toFixed(1), W - padR, H - padB + 14);

    // curve
    ctx.strokeStyle = '#b8410e';
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < N; i++) {
      const v = ys[i];
      if (!isFinite(v) || v > cap) { started = false; continue; }
      const x = px(i), y = py(v);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // sample markers at three example pair distances
    return { px, py, ymin, ymax: cap, V: P.V };
  }

  // --- bond graph: three orbiting bodies on a slow circular wobble ---
  // Geometry: equilateral triangle that breathes and rotates so distances vary.
  let t = 0;
  const masses = [1, 1, 1];

  function drawBond(meta) {
    bctx.fillStyle = '#fafaf7';
    bctx.fillRect(0, 0, BW, BH);

    const cx = BW/2, cy = BH/2;
    const baseR = 60;
    // three bodies, slight asymmetric breathing
    const wobble = [
      { phase: 0,                 r: baseR + 20*Math.sin(t*0.7) },
      { phase: 2*Math.PI/3,       r: baseR + 18*Math.sin(t*0.7 + 1.7) },
      { phase: 4*Math.PI/3,       r: baseR + 22*Math.sin(t*0.7 + 3.1) }
    ];
    const rot = t * 0.15;
    const pts = wobble.map(w => [
      cx + w.r * Math.cos(w.phase + rot),
      cy + w.r * Math.sin(w.phase + rot)
    ]);

    // pair distances in "world units" (px / 40)
    const pairs = [[0,1], [0,2], [1,2]];
    const labels = ['H₁₂', 'H₁₃', 'H₂₃'];
    const colors = ['#3aa0ff', '#7bc94c', '#ffcb3a'];

    pairs.forEach(([i, j], k) => {
      const dx = pts[i][0] - pts[j][0], dy = pts[i][1] - pts[j][1];
      const dpx = Math.hypot(dx, dy);
      const r = dpx / 40;          // world distance
      const v = meta.V(r);          // potential value
      const Hij = masses[i] * masses[j] * v;

      // bond line
      bctx.strokeStyle = colors[k];
      bctx.lineWidth = 2;
      bctx.beginPath();
      bctx.moveTo(pts[i][0], pts[i][1]);
      bctx.lineTo(pts[j][0], pts[j][1]);
      bctx.stroke();

      // label at midpoint
      const mx = (pts[i][0] + pts[j][0])/2;
      const my = (pts[i][1] + pts[j][1])/2;
      bctx.fillStyle = '#1a1a1a';
      bctx.font = '12px Georgia, serif';
      bctx.textAlign = 'left';
      bctx.fillText(`${labels[k]} = ${isFinite(Hij) ? Hij.toFixed(2) : '∞'}`,
                    mx + 4, my - 2);
    });

    // bodies
    for (let i = 0; i < 3; i++) {
      bctx.fillStyle = '#1a1a1a';
      bctx.beginPath();
      bctx.arc(pts[i][0], pts[i][1], 7, 0, 2*Math.PI);
      bctx.fill();
      bctx.fillStyle = '#fff';
      bctx.textAlign = 'center';
      bctx.font = 'bold 11px Georgia, serif';
      bctx.fillText(String(i+1), pts[i][0], pts[i][1] + 4);
    }

    // mark current pair distances on the V(r) plot too
    bctx.fillStyle = '#666';
    bctx.font = '11px Georgia, serif';
    bctx.textAlign = 'left';
    bctx.fillText('three-body bond view',  8, BH - 8);
  }

  let meta = plotPotential(sel.value);
  sel.addEventListener('change', () => { meta = plotPotential(sel.value); });

  function loop() {
    t += 0.016;
    drawBond(meta);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
})();
