// orbit_widget.js — §0 three-body playground
// Drives the canvas + sliders + presets, talks to orbit_worker.js.

(function () {
  const canvas = document.getElementById('orbit-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // ---- presets (G = 1) ----
  // Each: { masses:[m1,m2,m3], state:[x1,y1,x2,y2,x3,y3, vx1,vy1,vx2,vy2,vx3,vy3], view:scale }
  const PRESETS = {
    // Equilateral Lagrange L4: equal masses on a unit triangle, rotating rigidly.
    // For G=m=1, omega = sqrt(3) on a triangle of circumradius R=1/sqrt(3) -> side 1.
    lagrange: (() => {
      const R = 1 / Math.sqrt(3);
      const omega = Math.sqrt(3);  // for m=1, side=1, G=1
      const ang = [Math.PI/2, Math.PI/2 + 2*Math.PI/3, Math.PI/2 + 4*Math.PI/3];
      const px = ang.map(a => R*Math.cos(a));
      const py = ang.map(a => R*Math.sin(a));
      const vx = ang.map(a => -omega*R*Math.sin(a));
      const vy = ang.map(a =>  omega*R*Math.cos(a));
      return {
        masses: [1, 1, 1],
        state: [px[0],py[0], px[1],py[1], px[2],py[2],
                vx[0],vy[0], vx[1],vy[1], vx[2],vy[2]],
        view: 2.0
      };
    })(),

    // Chenciner-Montgomery figure-8 choreography (Chenciner & Montgomery 2000).
    // Standard initial conditions, units G=m=1.
    figure8: {
      masses: [1, 1, 1],
      state: [
         0.97000436, -0.24308753,
        -0.97000436,  0.24308753,
         0.0,         0.0,
         0.466203685, 0.43236573,
         0.466203685, 0.43236573,
        -0.93240737, -0.86473146
      ],
      view: 2.5
    },

    // Random — generated at click time
    chaos: null
  };

  function randomChaos() {
    function rand(a, b) { return a + Math.random()*(b-a); }
    return {
      masses: [rand(0.5, 2), rand(0.5, 2), rand(0.5, 2)],
      state: [
        rand(-1, 1), rand(-1, 1),
        rand(-1, 1), rand(-1, 1),
        rand(-1, 1), rand(-1, 1),
        rand(-0.5, 0.5), rand(-0.5, 0.5),
        rand(-0.5, 0.5), rand(-0.5, 0.5),
        rand(-0.5, 0.5), rand(-0.5, 0.5)
      ],
      view: 3.0
    };
  }

  // ---- worker ----
  const worker = new Worker('js/orbit_worker.js');

  // ---- view state ----
  let viewScale = 2.0;       // world half-width shown
  let trails = true;
  let trailBufs = [[], [], []];
  const TRAIL_MAX = 800;
  const COLORS = ['#ff5b3a', '#3aa0ff', '#7bc94c'];
  let lastPos = [0,0,0,0,0,0];
  let radii = [1, 1, 1];
  let offscreen = false;
  let lastPreset = 'figure8';

  function loadPreset(name) {
    let p = PRESETS[name];
    if (name === 'chaos') p = randomChaos();
    if (!p) return;
    lastPreset = name;
    viewScale = p.view;
    trailBufs = [[], [], []];
    radii = p.masses.map(m => Math.cbrt(m));
    // Subtract COM motion so the system stays centered
    const m = p.masses, s = p.state.slice();
    const M = m[0]+m[1]+m[2];
    const cx = (m[0]*s[0]+m[1]*s[2]+m[2]*s[4])/M;
    const cy = (m[0]*s[1]+m[1]*s[3]+m[2]*s[5])/M;
    const vcx = (m[0]*s[6]+m[1]*s[8]+m[2]*s[10])/M;
    const vcy = (m[0]*s[7]+m[1]*s[9]+m[2]*s[11])/M;
    for (let i = 0; i < 3; i++) {
      s[2*i]   -= cx;
      s[2*i+1] -= cy;
      s[6+2*i]   -= vcx;
      s[6+2*i+1] -= vcy;
    }
    // Update mass sliders
    document.getElementById('m1').value = Math.log10(m[0]);
    document.getElementById('m2').value = Math.log10(m[1]);
    document.getElementById('m3').value = Math.log10(m[2]);
    document.getElementById('m1v').value = m[0].toFixed(2);
    document.getElementById('m2v').value = m[1].toFixed(2);
    document.getElementById('m3v').value = m[2].toFixed(2);
    worker.postMessage({ type: 'init', masses: m, state: s, dt: 0.003, stepsPerFrame: 6 });
    worker.postMessage({ type: 'play' });
    lastPos = [s[0],s[1],s[2],s[3],s[4],s[5]];
  }

  function draw(pos) {
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, W, H);
    const scale = Math.min(W, H) / (2 * viewScale);
    const cx = W/2, cy = H/2;
    const toScreen = (x, y) => [cx + x*scale, cy - y*scale];

    // trails
    if (trails) {
      for (let i = 0; i < 3; i++) {
        const buf = trailBufs[i];
        if (buf.length < 2) continue;
        ctx.strokeStyle = COLORS[i] + '88';
        ctx.lineWidth = 1;
        ctx.beginPath();
        const [sx0, sy0] = toScreen(buf[0][0], buf[0][1]);
        ctx.moveTo(sx0, sy0);
        for (let k = 1; k < buf.length; k++) {
          const [sx, sy] = toScreen(buf[k][0], buf[k][1]);
          ctx.lineTo(sx, sy);
        }
        ctx.stroke();
      }
    }

    // bodies
    for (let i = 0; i < 3; i++) {
      const [sx, sy] = toScreen(pos[2*i], pos[2*i+1]);
      ctx.fillStyle = COLORS[i];
      ctx.beginPath();
      ctx.arc(sx, sy, 4 + 3*radii[i], 0, 2*Math.PI);
      ctx.fill();
    }

    // off-screen notice
    let allOff = true;
    for (let i = 0; i < 3; i++) {
      const [sx, sy] = toScreen(pos[2*i], pos[2*i+1]);
      if (sx >= 0 && sx <= W && sy >= 0 && sy <= H) { allOff = false; break; }
    }
    offscreen = allOff;
    if (allOff) {
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(0, H/2 - 28, W, 56);
      ctx.fillStyle = '#fafaf7';
      ctx.font = '16px Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('all bodies off screen — click to reset', W/2, H/2);
    }
  }

  worker.onmessage = (ev) => {
    if (ev.data.type !== 'frame') return;
    const p = ev.data.pos;
    lastPos = p;
    if (trails) {
      for (let i = 0; i < 3; i++) {
        trailBufs[i].push([p[2*i], p[2*i+1]]);
        if (trailBufs[i].length > TRAIL_MAX) trailBufs[i].shift();
      }
    }
    draw(p);
  };

  // Drive the worker via animation frames
  function loop() {
    worker.postMessage({ type: 'step' });
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  // ---- UI wiring ----
  document.querySelectorAll('.presets button').forEach(b => {
    b.addEventListener('click', () => loadPreset(b.dataset.preset));
  });
  canvas.addEventListener('click', () => {
    if (offscreen) loadPreset(lastPreset);
  });
  document.getElementById('play').addEventListener('click',
    () => worker.postMessage({ type: 'play' }));
  document.getElementById('pause').addEventListener('click',
    () => worker.postMessage({ type: 'pause' }));
  document.getElementById('reset').addEventListener('click',
    () => loadPreset('lagrange'));
  document.getElementById('trails').addEventListener('click', () => {
    trails = !trails;
    if (!trails) trailBufs = [[],[],[]];
  });

  function readMass(id) {
    const v = parseFloat(document.getElementById(id).value);
    return Math.pow(10, v);
  }
  function pushMasses() {
    const m = [readMass('m1'), readMass('m2'), readMass('m3')];
    document.getElementById('m1v').value = m[0].toFixed(2);
    document.getElementById('m2v').value = m[1].toFixed(2);
    document.getElementById('m3v').value = m[2].toFixed(2);
    radii = m.map(x => Math.cbrt(x));
    worker.postMessage({ type: 'masses', masses: m });
  }
  ['m1','m2','m3'].forEach(id =>
    document.getElementById(id).addEventListener('input', pushMasses));

  document.getElementById('speed').addEventListener('input', (e) => {
    const s = parseFloat(e.target.value);
    document.getElementById('speedv').value = s.toFixed(1) + '\u00d7';
    worker.postMessage({ type: 'speed', speed: s });
  });

  // boot
  loadPreset('figure8');
})();
