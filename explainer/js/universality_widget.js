// §6 Universality: morph between potentials, dimension stays locked at 116
(function () {
  const cv = document.getElementById('uni-canvas');
  const al = document.getElementById('uni-alpha');
  const be = document.getElementById('uni-beta');
  const ga = document.getElementById('uni-gamma');
  const lbl = document.getElementById('uni-formula');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;

  function V(r, a, b, g) {
    return a / r + b / (r * r) + g * Math.log(r);
  }

  function draw() {
    const a = +al.value, b = +be.value, g = +ga.value;
    ctx.clearRect(0, 0, W, H);
    // axes
    ctx.strokeStyle = '#bbb';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(40, 10); ctx.lineTo(40, H - 28); ctx.lineTo(W - 10, H - 28);
    ctx.stroke();
    ctx.fillStyle = '#666';
    ctx.font = '12px Georgia';
    ctx.fillText('r', W - 16, H - 12);
    ctx.fillText('V(r)', 6, 18);

    // sample
    const rmin = 0.4, rmax = 4.0;
    const N = 400;
    let vals = [];
    for (let i = 0; i < N; i++) {
      const r = rmin + (rmax - rmin) * i / (N - 1);
      vals.push(V(r, a, b, g));
    }
    let vmin = Infinity, vmax = -Infinity;
    for (const v of vals) { if (isFinite(v)) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; } }
    if (vmax - vmin < 1e-6) { vmax = vmin + 1; }
    // clamp range so spikes don't dominate
    const span = vmax - vmin;
    vmin = vmin; vmax = vmin + Math.min(span, 8);

    ctx.strokeStyle = '#b8410e';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const r = rmin + (rmax - rmin) * i / (N - 1);
      const x = 40 + (W - 50) * (r - rmin) / (rmax - rmin);
      const v = Math.min(Math.max(vals[i], vmin), vmax);
      const y = (H - 28) - (H - 38) * (v - vmin) / (vmax - vmin);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // dimension stamp
    ctx.fillStyle = '#1a1a1a';
    ctx.font = 'bold 14px Georgia';
    ctx.fillText('dim L₃ = 116', W - 130, 28);

    // formula label
    const fmt = (x) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(2);
    lbl.textContent = `V(r) = ${a.toFixed(2)}/r ${fmt(b)}/r² ${fmt(g)} log r`;
  }

  [al, be, ga].forEach(s => s.addEventListener('input', draw));
  draw();
})();
