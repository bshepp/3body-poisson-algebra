// §10 The frontier: a(4) ≥ 5604, climbing-bar timeline of the lower bound
(function () {
  const cv = document.getElementById('front-canvas');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;

  // milestones: real convergence data, website/data/datasets/level4_convergence.json
  // (global_highsample config; float64 SVD lower bound on a(4) at each sample count)
  const MS = [
    { label: '30K samples', v: 4501, note: 'numerical SVD' },
    { label: '50K samples', v: 5009, note: 'numerical SVD' },
    { label: '100K samples', v: 5449, note: 'numerical SVD' },
    { label: '150K samples', v: 5588, note: 'numerical SVD' },
    { label: '200K samples (AWS)', v: 5604, note: 'current lower bound' }
  ];

  const padL = 70, padR = 30, padT = 30, padB = 60;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const VMAX = 6500;
  const barW = plotW / MS.length * 0.65;
  const slot = plotW / MS.length;

  ctx.clearRect(0, 0, W, H);
  // axis
  ctx.strokeStyle = '#bbb'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH); ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();
  ctx.fillStyle = '#666'; ctx.font = '11px Georgia';
  for (let v = 0; v <= 6000; v += 1000) {
    const y = padT + plotH * (1 - v / VMAX);
    ctx.strokeStyle = '#eee'; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
    ctx.fillStyle = '#666'; ctx.fillText(v.toString(), 30, y + 4);
  }
  ctx.fillStyle = '#444'; ctx.font = '12px Georgia';
  ctx.save(); ctx.translate(16, padT + plotH / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('lower bound on a(4)', -60, 0); ctx.restore();

  MS.forEach((m, i) => {
    const x = padL + slot * i + (slot - barW) / 2;
    const h = plotH * m.v / VMAX;
    const y = padT + plotH - h;
    ctx.fillStyle = i === MS.length - 1 ? '#b8410e' : '#d8a274';
    ctx.fillRect(x, y, barW, h);
    ctx.fillStyle = '#1a1a1a'; ctx.font = 'bold 12px Georgia';
    ctx.textAlign = 'center';
    ctx.fillText(m.v.toString(), x + barW / 2, y - 6);
    ctx.font = '11px Georgia'; ctx.fillStyle = '#444';
    // wrap label
    ctx.fillText(m.label, x + barW / 2, padT + plotH + 16);
    ctx.font = 'italic 10px Georgia'; ctx.fillStyle = '#777';
    ctx.fillText(m.note, x + barW / 2, padT + plotH + 32);
    ctx.textAlign = 'start';
  });

  // ceiling line (unknown true value)
  const yC = padT + 10;
  ctx.strokeStyle = '#888'; ctx.setLineDash([6, 4]);
  ctx.beginPath(); ctx.moveTo(padL, yC); ctx.lineTo(padL + plotW, yC); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#555'; ctx.font = 'italic 11px Georgia';
  ctx.fillText('a(4) = ?  (true value unknown)', padL + 10, yC - 6);
})();
