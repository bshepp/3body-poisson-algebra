// dimseq_widget.js — §5 the dimension sequence as a bar chart
// Compares dimension sequences across potentials, highlighting the universal
// 3,6,17,116 vs the harmonic exception 3,6,15.

(function () {
  const canvas = document.getElementById('dim-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  // Each row: {label, color, seq, note}
  const ROWS = [
    { label: '1/r  (Newton)',          color: '#b8410e', seq: [3, 6, 17, 116], note: 'universal' },
    { label: '1/r² (Calogero–Moser)',  color: '#d77a2c', seq: [3, 6, 17, 116], note: 'integrable, same dim' },
    { label: '1/r³',                   color: '#c66',    seq: [3, 6, 17, 116], note: 'same dim' },
    { label: 'log r',                  color: '#888',    seq: [3, 6, 17, 116], note: 'same dim' },
    { label: 'e^(-r)/r (Yukawa)',      color: '#3aa0ff', seq: [3, 6, 17, 116], note: 'same dim' },
    { label: 'r²  (harmonic)',         color: '#7bc94c', seq: [3, 6, 15],       note: 'closes at level 2' }
  ];

  const LEVELS = 4; // show columns L_0..L_3

  function draw() {
    ctx.fillStyle = '#fafaf7';
    ctx.fillRect(0, 0, W, H);

    const padL = 180, padR = 30, padT = 30, padB = 36;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const rowH = plotH / ROWS.length;

    // header
    ctx.fillStyle = '#1a1a1a';
    ctx.font = 'bold 13px Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('potential', 12, padT - 10);
    ctx.textAlign = 'center';
    for (let k = 0; k < LEVELS; k++) {
      const x = padL + plotW * (k + 0.5) / LEVELS;
      ctx.fillText('L' + String.fromCharCode(0x2080 + k), x, padT - 10);
    }

    // bars (log-ish height: linear up to 116 for visual clarity)
    const MAX = 116;
    for (let i = 0; i < ROWS.length; i++) {
      const R = ROWS[i];
      const yTop = padT + i * rowH;
      const yMid = yTop + rowH / 2;

      // label
      ctx.fillStyle = R.color;
      ctx.fillRect(8, yMid - 6, 12, 12);
      ctx.fillStyle = '#1a1a1a';
      ctx.font = '12px Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText(R.label, 26, yMid + 4);
      ctx.fillStyle = '#888';
      ctx.font = 'italic 11px Georgia, serif';
      ctx.fillText(R.note, 26, yMid + 18);

      // bars per level
      for (let k = 0; k < LEVELS; k++) {
        const cellL = padL + plotW * k / LEVELS;
        const cellW = plotW / LEVELS - 6;
        const v = R.seq[k];
        if (v === undefined) {
          // closed-out: draw a hollow stub
          ctx.strokeStyle = '#aaa';
          ctx.setLineDash([3, 3]);
          ctx.strokeRect(cellL + 3, yMid - 8, cellW, 16);
          ctx.setLineDash([]);
          ctx.fillStyle = '#888';
          ctx.font = 'italic 10px Georgia, serif';
          ctx.textAlign = 'center';
          ctx.fillText('closed', cellL + cellW/2 + 3, yMid + 4);
          continue;
        }
        const h = Math.max(2, Math.min(1, v / MAX) * (rowH - 14));
        ctx.fillStyle = R.color;
        ctx.fillRect(cellL + 3, yMid + 8 - h, cellW, h);
        ctx.fillStyle = '#1a1a1a';
        ctx.font = 'bold 11px Georgia, serif';
        ctx.textAlign = 'center';
        ctx.fillText(String(v), cellL + cellW/2 + 3, yMid + 8 - h - 3);
      }
    }

    // baseline
    ctx.strokeStyle = '#ccc';
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH);
    ctx.stroke();
  }

  draw();
})();
