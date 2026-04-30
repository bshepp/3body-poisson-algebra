// algebra_widget.js — §4 build the algebra step by step
// Visualize L_0 → L_1 → L_2 → L_3 with cumulative dimensions 3, 6, 17, 116.

(function () {
  const root = document.getElementById('algebra-stage');
  const btn  = document.getElementById('alg-step');
  const reset = document.getElementById('alg-reset');
  const output = document.getElementById('alg-dim');
  if (!root) return;

  // Each level: cumulative dimension and a one-line description of new generators.
  const LEVELS = [
    {
      label: 'L₀',
      dim: 3,
      desc: 'seed: the three pairwise Hamiltonians',
      items: ['H₁₂', 'H₁₃', 'H₂₃']
    },
    {
      label: 'L₁',
      dim: 6,
      desc: 'add {Hᵢⱼ, Hₖₗ} for sharing a body — three new mixed generators',
      items: ['{H₁₂, H₁₃}', '{H₁₂, H₂₃}', '{H₁₃, H₂₃}']
    },
    {
      label: 'L₂',
      dim: 17,
      desc: 'iterate again — eleven independent new observables emerge',
      items: ['{H₁₂, {H₁₂, H₁₃}}', '{H₁₃, {H₁₂, H₂₃}}', '… (11 new)']
    },
    {
      label: 'L₃',
      dim: 116,
      desc: 'level 3: ninety-nine new generators — the algebra is non-trivial',
      items: ['(99 new symbolic expressions, ranks verified by SVD)']
    },
    {
      label: 'L₄',
      dim: '≥ 5604',
      desc: 'level 4 frontier: lower bound from 4096-sample SVD on AWS',
      items: ['(thousands more — exact dimension still open)']
    }
  ];

  let shown = 0;

  function render() {
    root.innerHTML = '';
    for (let k = 0; k <= shown; k++) {
      const L = LEVELS[k];
      const row = document.createElement('div');
      row.className = 'alg-row alg-row-' + k;
      row.innerHTML = `
        <div class="alg-tag">${L.label}</div>
        <div class="alg-body">
          <div class="alg-dim-line">dim = <strong>${L.dim}</strong>
            <span class="alg-desc"> — ${L.desc}</span>
          </div>
          <ul class="alg-items">
            ${L.items.map(s => `<li>${s}</li>`).join('')}
          </ul>
        </div>
      `;
      root.appendChild(row);
    }
    if (output) {
      const dims = LEVELS.slice(0, shown+1).map(l => l.dim).join(', ');
      output.textContent = '[' + dims + (shown < LEVELS.length-1 ? ', …' : '') + ']';
    }
    if (btn) {
      btn.disabled = (shown >= LEVELS.length - 1);
      btn.textContent = btn.disabled ? 'Reached the frontier' : 'Compute next level →';
    }
  }

  if (btn) btn.addEventListener('click', () => {
    if (shown < LEVELS.length - 1) { shown++; render(); }
  });
  if (reset) reset.addEventListener('click', () => { shown = 0; render(); });

  render();
})();
