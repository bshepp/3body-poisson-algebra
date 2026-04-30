// §8 Mass invariance: mass-ratio plane heatmap, uniformly 116 except a measure-zero stratum
(function () {
  const cv = document.getElementById('mass-canvas');
  const out = document.getElementById('mass-out');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;

  // background: solid universal color
  function draw(hover) {
    ctx.clearRect(0, 0, W, H);
    // gradient hint of generic stratum
    const g = ctx.createLinearGradient(0, 0, W, H);
    g.addColorStop(0, '#f4d8c2');
    g.addColorStop(1, '#e6b88f');
    ctx.fillStyle = g;
    ctx.fillRect(40, 20, W - 60, H - 60);

    // axes
    ctx.strokeStyle = '#888';
    ctx.lineWidth = 1;
    ctx.strokeRect(40, 20, W - 60, H - 60);
    ctx.fillStyle = '#444';
    ctx.font = '12px Georgia';
    ctx.fillText('m₁ / m₃', W - 70, H - 22);
    ctx.save(); ctx.translate(14, 30); ctx.rotate(-Math.PI / 2);
    ctx.fillText('m₂ / m₃', -50, 0); ctx.restore();
    // ticks
    [0.25, 0.5, 0.75].forEach(u => {
      const x = 40 + (W - 60) * u;
      ctx.beginPath(); ctx.moveTo(x, H - 40); ctx.lineTo(x, H - 36); ctx.stroke();
      ctx.fillText((u * 4).toFixed(1), x - 8, H - 24);
      const y = 20 + (H - 60) * u;
      ctx.beginPath(); ctx.moveTo(40, y); ctx.lineTo(36, y); ctx.stroke();
      ctx.fillText((4 - u * 4).toFixed(1), 14, y + 4);
    });

    // dotted line: equal-mass diagonal
    ctx.strokeStyle = '#9b6b3a';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(40, H - 40); ctx.lineTo(W - 20, 20); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#7a4a1f';
    ctx.font = 'italic 11px Georgia';
    ctx.fillText('equal masses', W - 130, 38);

    // central stamp
    ctx.fillStyle = '#1a1a1a';
    ctx.font = 'bold 18px Georgia';
    ctx.textAlign = 'center';
    ctx.fillText('dim L₃ = 116', W / 2, H / 2 - 6);
    ctx.font = 'italic 12px Georgia';
    ctx.fillText('for every (m₁, m₂, m₃) with mᵢ > 0', W / 2, H / 2 + 14);
    ctx.textAlign = 'start';

    // hovered point
    if (hover) {
      ctx.fillStyle = '#b8410e';
      ctx.beginPath(); ctx.arc(hover.x, hover.y, 5, 0, 2 * Math.PI); ctx.fill();
    }
  }

  cv.addEventListener('mousemove', (e) => {
    const r = cv.getBoundingClientRect();
    const x = (e.clientX - r.left) * (cv.width / r.width);
    const y = (e.clientY - r.top) * (cv.height / r.height);
    if (x < 40 || x > W - 20 || y < 20 || y > H - 40) { draw(); out.textContent = ''; return; }
    const m1 = ((x - 40) / (W - 60)) * 4;
    const m2 = (1 - (y - 20) / (H - 60)) * 4;
    out.innerHTML = `(m₁/m₃, m₂/m₃) = (${m1.toFixed(2)}, ${m2.toFixed(2)}) &nbsp;⟶&nbsp; <strong>116</strong>`;
    draw({ x, y });
  });
  cv.addEventListener('mouseleave', () => { draw(); out.textContent = ''; });
  draw();
})();
