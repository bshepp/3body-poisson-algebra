// orbit_worker.js
// RK4 integrator for the planar 3-body problem in a Web Worker.
// State vector y = [x1,y1,x2,y2,x3,y3, vx1,vy1,vx2,vy2,vx3,vy3]
// G = 1 (units chosen so masses are O(1)).

let G = 1.0;
let SOFT2 = 1e-4;     // softening^2 to prevent singular acceleration on close approach
let masses = [1, 1, 1];
let state = null;
let dt = 0.005;
let speed = 1.0;
let running = false;
let stepsPerFrame = 8;

function accel(y, m) {
  const a = new Float64Array(6);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      if (i === j) continue;
      const dx = y[2*j]   - y[2*i];
      const dy = y[2*j+1] - y[2*i+1];
      const r2 = dx*dx + dy*dy + SOFT2;
      const inv = 1 / (r2 * Math.sqrt(r2));
      a[2*i]   += G * m[j] * dx * inv;
      a[2*i+1] += G * m[j] * dy * inv;
    }
  }
  return a;
}

function deriv(y, m) {
  // dy/dt = [v..., a...]
  const dydt = new Float64Array(12);
  for (let k = 0; k < 6; k++) dydt[k] = y[6+k];
  const a = accel(y, m);
  for (let k = 0; k < 6; k++) dydt[6+k] = a[k];
  return dydt;
}

function rk4Step(y, m, h) {
  const k1 = deriv(y, m);
  const y2 = new Float64Array(12);
  for (let i = 0; i < 12; i++) y2[i] = y[i] + 0.5*h*k1[i];
  const k2 = deriv(y2, m);
  const y3 = new Float64Array(12);
  for (let i = 0; i < 12; i++) y3[i] = y[i] + 0.5*h*k2[i];
  const k3 = deriv(y3, m);
  const y4 = new Float64Array(12);
  for (let i = 0; i < 12; i++) y4[i] = y[i] + h*k3[i];
  const k4 = deriv(y4, m);
  const out = new Float64Array(12);
  for (let i = 0; i < 12; i++)
    out[i] = y[i] + (h/6)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
  return out;
}

function tick() {
  if (!running || !state) return;
  const h = dt * speed;
  for (let s = 0; s < stepsPerFrame; s++) {
    state = rk4Step(state, masses, h);
  }
  // post current positions back
  postMessage({ type: 'frame', pos: [
    state[0], state[1], state[2], state[3], state[4], state[5]
  ]});
}

self.onmessage = (ev) => {
  const msg = ev.data;
  switch (msg.type) {
    case 'init':
      masses = msg.masses.slice();
      state = new Float64Array(msg.state);
      dt = msg.dt ?? 0.005;
      stepsPerFrame = msg.stepsPerFrame ?? 8;
      break;
    case 'masses':
      masses = msg.masses.slice();
      break;
    case 'speed':
      speed = msg.speed;
      break;
    case 'play':
      running = true;
      break;
    case 'pause':
      running = false;
      break;
    case 'step':
      tick();
      break;
  }
};
