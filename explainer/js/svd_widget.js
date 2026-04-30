// §9 SVD rank gap: log-scale singular value plot with the >10^10 cliff at index 116
(function () {
  const cv = document.getElementById('svd-canvas');
  const out = document.getElementById('svd-out');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;

  // synthetic but representative singular values: smooth decay then numerical floor
  const N = 160;
  const sigmas = [];
  for (let i = 0; i < N; i++) {
    if (i < 116) {
      // slow decay in [10^0, 10^-3]
      sigmas.push(Math.pow(10, -3 * i / 115));
    } else {
      // numerical floor near machine eps with mild jitter
      sigmas.push(1e-14 * (0.5 + ((i * 9301 + 49297) % 233) / 233));
    }
  }

  const padL = 50, padR = 20, padT = 20, padB = 36;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const yMin = -16, yMax = 1; // log10 range

  function xCoord(i) { return padL + plotW * i / (N - 1); }
  function yCoord(s) {
    const lg = Math.log10(s);
    return padT + plotH * (1 - (lg - yMin) / (yMax - yMin));
  }

  function draw(hoverIdx) {
    ctx.clearRect(0, 0, W, H);
    // axes
    ctx.strokeStyle = '#bbb'; ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, plotW, plotH);
    ctx.fillStyle = '#666'; ctx.font = '11px Georgia';
    // y ticks every 4 decades
    for (let lg = yMin; lg <= yMax; lg += 4) {
      const y = padT + plotH * (1 - (lg - yMin) / (yMax - yMin));
      ctx.strokeStyle = '#eee'; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
      ctx.fillStyle = '#666'; ctx.fillText(`10${supr(lg)}`, 10, y + 4);
    }
    // x ticks
    [0, 40, 80, 116, 160].forEach(i => {
      if (i > N - 1) return;
      const x = xCoord(i);
      ctx.strokeStyle = '#eee'; ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
      ctx.fillStyle = '#666'; ctx.fillText(i.toString(), x - 8, padT + plotH + 16);
    });
    ctx.fillStyle = '#444';
    ctx.fillText('singular value index', padL + plotW / 2 - 50, H - 8);

    // gap line at 116
    const xg = xCoord(116);
    ctx.strokeStyle = '#b8410e'; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(xg, padT); ctx.lineTo(xg, padT + plotH); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#b8410e'; ctx.font = 'italic 12px Georgia';
    ctx.fillText('rank cliff at 116', xg + 6, padT + 14);

    // points
    for (let i = 0; i < N; i++) {
      const x = xCoord(i), y = yCoord(sigmas[i]);
      ctx.fillStyle = i < 116 ? '#1a1a1a' : '#888';
      ctx.beginPath(); ctx.arc(x, y, 2.2, 0, 2 * Math.PI); ctx.fill();
    }

    // gap arrow annotation
    const y116 = yCoord(sigmas[115]);
    const y117 = yCoord(sigmas[116]);
    ctx.strokeStyle = '#b8410e'; ctx.lineWidth = 1.5;
    const ax = xg + 60;
    ctx.beginPath();
    ctx.moveTo(ax, y116); ctx.lineTo(ax, y117);
    ctx.moveTo(ax - 4, y116 + 4); ctx.lineTo(ax, y116); ctx.lineTo(ax + 4, y116 + 4);
    ctx.moveTo(ax - 4, y117 - 4); ctx.lineTo(ax, y117); ctx.lineTo(ax + 4, y117 - 4);
    ctx.stroke();
    ctx.fillStyle = '#b8410e'; ctx.font = '12px Georgia';
    ctx.fillText('σ₁₁₆ / σ₁₁₇  >  10¹⁰', ax + 8, (y116 + y117) / 2 + 4);

    if (hoverIdx != null) {
      const x = xCoord(hoverIdx), y = yCoord(sigmas[hoverIdx]);
      ctx.fillStyle = '#b8410e';
      ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fill();
      out.innerHTML = `σ<sub>${hoverIdx + 1}</sub> ≈ ${sigmas[hoverIdx].toExponential(2)}`;
    }
  }

  function supr(n) {
    const map = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
    return String(n).split('').map(c => map[c] || c).join('');
  }

  cv.addEventListener('mousemove', (e) => {
    const r = cv.getBoundingClientRect();
    const x = (e.clientX - r.left) * (cv.width / r.width);
    if (x < padL || x > padL + plotW) return;
    const i = Math.round((x - padL) / plotW * (N - 1));
    draw(i);
  });
  cv.addEventListener('mouseleave', () => { out.textContent = ''; draw(); });
  draw();
})();
