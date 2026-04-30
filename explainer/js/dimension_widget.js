// §7 Spatial dimension d ∈ {1,2,3}: toggle d, sequence stays the same
(function () {
  const stage = document.getElementById('dim-stage');
  const out = document.getElementById('dim-out');
  if (!stage) return;

  const SEQ = [3, 6, 17, 116];

  function render(d) {
    stage.querySelectorAll('button').forEach(b => {
      b.classList.toggle('active', +b.dataset.d === d);
    });
    // body count widgets per d
    const cv = document.getElementById('dim-bodies');
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);

    // draw bodies on a line / plane / 3-axis sketch
    ctx.strokeStyle = '#bbb';
    ctx.fillStyle = '#1a1a1a';
    if (d === 1) {
      ctx.beginPath(); ctx.moveTo(20, H / 2); ctx.lineTo(W - 20, H / 2); ctx.stroke();
      [0.25, 0.5, 0.75].forEach((u, i) => {
        const x = 20 + (W - 40) * u, y = H / 2;
        ctx.beginPath(); ctx.arc(x, y, 8, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.font = '11px Georgia';
        ctx.fillText(i + 1, x - 3, y + 4); ctx.fillStyle = '#1a1a1a';
      });
    } else if (d === 2) {
      ctx.strokeRect(20, 20, W - 40, H - 40);
      const pts = [[0.3, 0.3], [0.7, 0.35], [0.5, 0.75]];
      pts.forEach((p, i) => {
        const x = 20 + (W - 40) * p[0], y = 20 + (H - 40) * p[1];
        ctx.beginPath(); ctx.arc(x, y, 8, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.font = '11px Georgia';
        ctx.fillText(i + 1, x - 3, y + 4); ctx.fillStyle = '#1a1a1a';
      });
    } else {
      // pseudo-3d box
      const ox = 60, oy = H - 40;
      ctx.beginPath();
      ctx.moveTo(ox, oy); ctx.lineTo(ox + 200, oy);
      ctx.moveTo(ox, oy); ctx.lineTo(ox, oy - 140);
      ctx.moveTo(ox, oy); ctx.lineTo(ox + 90, oy - 70);
      ctx.stroke();
      const pts3 = [[40, 60, 30], [120, 90, 60], [80, 30, 90]];
      pts3.forEach((p, i) => {
        const x = ox + p[0] + 0.5 * p[2];
        const y = oy - p[1] - 0.5 * p[2];
        ctx.beginPath(); ctx.arc(x, y, 8, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.font = '11px Georgia';
        ctx.fillText(i + 1, x - 3, y + 4); ctx.fillStyle = '#1a1a1a';
      });
    }

    out.innerHTML = `d = ${d} &nbsp;⟶&nbsp; <strong>[${SEQ.join(', ')}]</strong>`;
  }

  stage.querySelectorAll('button').forEach(b => {
    b.addEventListener('click', () => render(+b.dataset.d));
  });
  render(2);
})();
