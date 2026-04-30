// pendulum_widget.js — §1 phase-space portrait of a planar pendulum
// Hamiltonian: H(q,p) = p^2/2 - cos(q)
// Click to launch a trajectory from that initial condition.

(function () {
  const canvas = document.getElementById('pendulum-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  // World window: q in [-pi, pi], p in [-3, 3]
  const Q_MIN = -Math.PI, Q_MAX = Math.PI;
  const P_MIN = -3, P_MAX = 3;
  const toScreen = (q, p) => [
    (q - Q_MIN) / (Q_MAX - Q_MIN) * W,
    H - (p - P_MIN) / (P_MAX - P_MIN) * H
  ];
  const fromScreen = (sx, sy) => [
    Q_MIN + sx / W * (Q_MAX - Q_MIN),
    P_MIN + (H - sy) / H * (P_MAX - P_MIN)
  ];

  const hamiltonian = (q, p) => 0.5 * p * p - Math.cos(q);

  // Pre-render energy contour background once
  const bg = document.createElement('canvas');
  bg.width = W; bg.height = H;
  (function paintBackground() {
    const bctx = bg.getContext('2d');
    bctx.fillStyle = '#111';
    bctx.fillRect(0, 0, W, H);

    // contour levels (energy)
    const levels = [-0.95, -0.7, -0.4, -0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0];
    const img = bctx.createImageData(W, H);
    const d = img.data;
    for (let sy = 0; sy < H; sy++) {
      for (let sx = 0; sx < W; sx++) {
        const [q, p] = fromScreen(sx + 0.5, sy + 0.5);
        const Hv = hamiltonian(q, p);
        // Draw a faint contour band around each level
        let near = false;
        for (const L of levels) {
          if (Math.abs(Hv - L) < 0.025) { near = true; break; }
        }
        const i = 4 * (sy * W + sx);
        if (Math.abs(Hv - 1.0) < 0.012) {
          // separatrix highlighted
          d[i] = 184; d[i+1] = 65; d[i+2] = 14; d[i+3] = 200;
        } else if (near) {
          d[i] = 80; d[i+1] = 90; d[i+2] = 110; d[i+3] = 180;
        } else {
          d[i] = 17; d[i+1] = 17; d[i+2] = 17; d[i+3] = 255;
        }
      }
    }
    bctx.putImageData(img, 0, 0);

    // axes
    bctx.strokeStyle = '#444';
    bctx.lineWidth = 1;
    bctx.beginPath();
    const [zx, zy] = toScreen(0, 0);
    bctx.moveTo(0, zy); bctx.lineTo(W, zy);
    bctx.moveTo(zx, 0); bctx.lineTo(zx, H);
    bctx.stroke();

    // labels
    bctx.fillStyle = '#888';
    bctx.font = '12px Georgia, serif';
    bctx.textAlign = 'left';
    bctx.fillText('q (angle)', W - 70, zy - 6);
    bctx.fillText('p (momentum)', zx + 6, 14);
    bctx.textAlign = 'center';
    bctx.fillText('-π', 12, zy + 14);
    bctx.fillText('π', W - 10, zy + 14);
  })();

  // ---- trajectories ----
  const TRAJS = []; // each: { q0, p0, pts: [[q,p],...] , hue }
  const MAX_POINTS = 1500;
  const dt = 0.02;

  function rk4(q, p, h) {
    // dq = p, dp = -sin(q)
    const k1q = p,           k1p = -Math.sin(q);
    const k2q = p + 0.5*h*k1p, k2p = -Math.sin(q + 0.5*h*k1q);
    const k3q = p + 0.5*h*k2p, k3p = -Math.sin(q + 0.5*h*k2q);
    const k4q = p +     h*k3p, k4p = -Math.sin(q +     h*k3q);
    return [
      q + (h/6)*(k1q + 2*k2q + 2*k3q + k4q),
      p + (h/6)*(k1p + 2*k2p + 2*k3p + k4p)
    ];
  }

  function wrapQ(q) {
    while (q >  Math.PI) q -= 2*Math.PI;
    while (q < -Math.PI) q += 2*Math.PI;
    return q;
  }

  function step() {
    for (const tr of TRAJS) {
      let q = tr.q, p = tr.p;
      for (let k = 0; k < 4; k++) [q, p] = rk4(q, p, dt);
      tr.q = q; tr.p = p;
      tr.pts.push([wrapQ(q), p]);
      if (tr.pts.length > MAX_POINTS) tr.pts.shift();
    }
  }

  function draw() {
    ctx.drawImage(bg, 0, 0);
    for (const tr of TRAJS) {
      ctx.strokeStyle = tr.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let prevSx = null;
      for (let i = 0; i < tr.pts.length; i++) {
        const [q, p] = tr.pts[i];
        const [sx, sy] = toScreen(q, p);
        // break the path when q wraps (jump > pi)
        if (prevSx !== null && Math.abs(sx - prevSx) > W/2) {
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(sx, sy);
        } else if (prevSx === null) {
          ctx.moveTo(sx, sy);
        } else {
          ctx.lineTo(sx, sy);
        }
        prevSx = sx;
      }
      ctx.stroke();
      // current point
      const [hx, hy] = toScreen(wrapQ(tr.q), tr.p);
      ctx.fillStyle = tr.color;
      ctx.beginPath();
      ctx.arc(hx, hy, 3.5, 0, 2*Math.PI);
      ctx.fill();
    }
    // legend
    ctx.fillStyle = '#aaa';
    ctx.font = '12px Georgia, serif';
    ctx.textAlign = 'left';
    ctx.fillText('click to launch a trajectory · orange ring = separatrix (H=1)',
                 8, H - 8);
  }

  let playing = true;
  function loop() {
    if (playing) step();
    draw();
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  // ---- interaction ----
  const PALETTE = ['#3aa0ff', '#7bc94c', '#ffcb3a', '#ff5b3a', '#c074ff'];
  let colorIdx = 0;

  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const sx = (e.clientX - rect.left) * (W / rect.width);
    const sy = (e.clientY - rect.top)  * (H / rect.height);
    const [q, p] = fromScreen(sx, sy);
    TRAJS.push({
      q, p, q0: q, p0: p,
      pts: [[wrapQ(q), p]],
      color: PALETTE[colorIdx++ % PALETTE.length]
    });
    document.getElementById('pend-h').textContent = hamiltonian(q, p).toFixed(3);
  });

  document.getElementById('pend-clear').addEventListener('click', () => {
    TRAJS.length = 0;
    document.getElementById('pend-h').textContent = '—';
  });
  document.getElementById('pend-pause').addEventListener('click', (e) => {
    playing = !playing;
    e.target.textContent = playing ? '\u2759\u2759 Pause' : '\u25b6 Play';
  });

  // Seed a few sample trajectories so the picture isn't empty
  function seed(q, p) {
    TRAJS.push({
      q, p, q0: q, p0: p,
      pts: [[wrapQ(q), p]],
      color: PALETTE[colorIdx++ % PALETTE.length]
    });
  }
  seed(0.5, 0);          // small libration
  seed(2.5, 0);          // large libration near separatrix
  seed(0, 2.2);          // rotation
  document.getElementById('pend-h').textContent = '0.420 / 0.801 / 1.42';
})();
