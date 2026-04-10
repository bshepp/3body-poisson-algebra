#!/usr/bin/env python3
"""
Energy Bound Search — AWS-ready with checkpointing
====================================================

Investigate whether the negative semi-definite 117th generator can be
combined with other algebra elements to produce a conserved quantity
with definite sign, yielding an operator inequality on the quantum
three-body problem.

Steps:
1. Verify g_q commutes with total momentum P and center of mass
2. Build all 117 quantum generators (N=3, d=1, level 3)
3. Compute the commutant of H_total: ker(ad_{H_total}) exactly over QQ[hbar]
4. Compare commutant dimension at classical (116/QQ) vs quantum (117/QQ[hbar])
5. Construct quadratic Casimir from level-2 structure constants
6. Analyze conserved combinations for sign definiteness

Features:
  - Per-step pickle checkpointing with atomic writes
  - Completion manifest (skip-if-done on restart)
  - SIGTERM handling for spot instance reclamation
  - S3 heartbeat and periodic log/checkpoint sync
  - Per-bracket progress reporting (every bracket, not every 20th)

Usage:
    # Local (no S3)
    python -u energy_bound_search.py

    # AWS (with S3 sync)
    S3_BUCKET=3body-compute-290318 python -u energy_bound_search.py
"""

import os
import sys
import json
import pickle
import signal
import subprocess
from time import time, strftime

os.environ["PYTHONUNBUFFERED"] = "1"
sys.setrecursionlimit(500000)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import sympy as sp
from sympy import (Symbol, Integer, Rational, Add, Poly, expand, cancel,
                   diff, symbols)
from sympy.polys.matrices import DomainMatrix
from sympy.polys.domains import QQ

from exact_growth_nbody import NBodyAlgebra
from quantum_algebra import QuantumNBodyAlgebra, hbar_sym
from identify_117th import build_generators_both, separate_hbar_orders

# ---------------------------------------------------------------------------
# Infrastructure: S3, checkpointing, SIGTERM
# ---------------------------------------------------------------------------

S3_BUCKET = os.environ.get("S3_BUCKET", "")
RESULTS_PREFIX = "results/energy_bound"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints_energy_bound")
MANIFEST_PATH = os.path.join(CHECKPOINT_DIR, "manifest.json")

_shutdown_requested = False


def _sigterm_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n  [SIGTERM] Shutdown requested at {strftime('%H:%M:%S')} "
          f"-- finishing current step...", flush=True)


signal.signal(signal.SIGTERM, _sigterm_handler)


def s3_sync(local_dir, s3_prefix):
    if not S3_BUCKET:
        return
    dest = f"s3://{S3_BUCKET}/{s3_prefix}"
    try:
        subprocess.run(
            ["aws", "s3", "sync", local_dir, dest, "--size-only"],
            capture_output=True, timeout=300,
        )
    except Exception as e:
        print(f"  [S3 sync warning: {e}]", flush=True)


def s3_cp(local_path, s3_key):
    if not S3_BUCKET:
        return
    dest = f"s3://{S3_BUCKET}/{s3_key}"
    try:
        subprocess.run(
            ["aws", "s3", "cp", local_path, dest],
            capture_output=True, timeout=60,
        )
    except Exception as e:
        print(f"  [S3 cp warning: {e}]", flush=True)


def s3_pull(s3_prefix, local_dir):
    if not S3_BUCKET:
        return
    src = f"s3://{S3_BUCKET}/{s3_prefix}"
    try:
        subprocess.run(
            ["aws", "s3", "sync", src, local_dir, "--size-only"],
            capture_output=True, timeout=300,
        )
    except Exception as e:
        print(f"  [S3 pull warning: {e}]", flush=True)


def heartbeat(step_name):
    if not S3_BUCKET:
        return
    hb = {
        "job": "energy_bound_search",
        "step": step_name,
        "time": strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        p = subprocess.run(
            ["aws", "s3", "cp", "-",
             f"s3://{S3_BUCKET}/{RESULTS_PREFIX}/heartbeat.json"],
            input=json.dumps(hb).encode(), capture_output=True, timeout=30,
        )
    except Exception:
        pass


def save_checkpoint(tag, data):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"{tag}.pkl")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    sz = os.path.getsize(path) / 1024 / 1024
    print(f"  [ckpt] Saved {tag} ({sz:.1f} MB)", flush=True)


def load_checkpoint(tag):
    path = os.path.join(CHECKPOINT_DIR, f"{tag}.pkl")
    if not os.path.exists(path):
        return None
    print(f"  [ckpt] Loading {tag}...", flush=True)
    with open(path, "rb") as f:
        data = pickle.load(f)
    sz = os.path.getsize(path) / 1024 / 1024
    print(f"  [ckpt] Loaded {tag} ({sz:.1f} MB)", flush=True)
    return data


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    if S3_BUCKET:
        s3_cp(MANIFEST_PATH, f"{RESULTS_PREFIX}/manifest.json")


def mark_complete(step, result_summary):
    manifest = load_manifest()
    manifest[step] = {
        "status": "complete",
        "result": result_summary,
        "completed_at": strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    save_manifest(manifest)


def is_complete(step):
    manifest = load_manifest()
    return step in manifest and manifest[step].get("status") == "complete"


def sync_all():
    """Sync checkpoints + results to S3."""
    s3_sync(CHECKPOINT_DIR, f"{RESULTS_PREFIX}/checkpoints")
    results_dir = os.path.join(SCRIPT_DIR, "..", "results", "energy_bound")
    if os.path.isdir(results_dir):
        s3_sync(results_dir, "results/energy_bound")


def check_shutdown(step_name=""):
    if _shutdown_requested:
        print(f"\n  [SHUTDOWN] Graceful exit after {step_name}", flush=True)
        sync_all()
        sys.exit(0)


# ---------------------------------------------------------------------------
# Step 1: Symmetry verification
# ---------------------------------------------------------------------------

def step1_symmetry_check():
    """Verify g_q commutes with total momentum and center of mass."""
    if is_complete("step1"):
        print("=" * 70)
        print("STEP 1: SYMMETRY VERIFICATION [cached]")
        print("=" * 70)
        ckpt = load_checkpoint("step1_gq")
        return ckpt["g_q"], ckpt["alg"]

    print("=" * 70)
    print("STEP 1: SYMMETRY VERIFICATION")
    print("=" * 70)
    t0 = time()
    heartbeat("step1")

    alg = QuantumNBodyAlgebra(3, 1, "1/r")
    H12, H13, H23 = alg.hamiltonian_list
    x1, x2, x3 = alg.q_vars
    px1, px2, px3 = alg.p_vars

    b1 = cancel(alg.commutator(H12, H13))
    b2 = cancel(alg.commutator(b1, H12))
    b3 = cancel(alg.commutator(b2, b1))
    orders = separate_hbar_orders(b3)
    g_q = expand(orders.get(2, Integer(0)))

    print(f"  g_q: {len(Add.make_args(g_q))} terms")

    P_total = px1 + px2 + px3
    print(f"\n  Total momentum P = {P_total}")

    comm_gq_P = alg.commutator(g_q, P_total)
    comm_gq_P = cancel(comm_gq_P)
    p_zero = (comm_gq_P == 0)
    print(f"  [g_q, P]/(i*hbar) = {'ZERO' if p_zero else 'NONZERO'}")

    X_cm = (x1 + x2 + x3) / 3
    print(f"\n  Center of mass X_cm = (x1 + x2 + x3)/3")

    comm_gq_Xcm = alg.commutator(g_q, X_cm)
    comm_gq_Xcm = cancel(comm_gq_Xcm)
    xcm_zero = (comm_gq_Xcm == 0)
    print(f"  [g_q, X_cm]/(i*hbar) = {'ZERO' if xcm_zero else 'NONZERO'}")

    elapsed = time() - t0
    print(f"\n  Step 1 completed in {elapsed:.1f}s")

    save_checkpoint("step1_gq", {"g_q": g_q, "alg": alg})
    mark_complete("step1", {
        "g_q_terms": len(Add.make_args(g_q)),
        "commutes_with_P": p_zero,
        "commutes_with_Xcm": xcm_zero,
        "time_s": round(elapsed, 1),
    })
    sync_all()
    return g_q, alg


# ---------------------------------------------------------------------------
# Generator building (with checkpoint)
# ---------------------------------------------------------------------------

def build_all_generators():
    """Build classical and quantum generators, with checkpoint."""
    ckpt = load_checkpoint("generators")
    if ckpt is not None:
        print("\n  Generators loaded from checkpoint.")
        return ckpt

    print("\n")
    result = build_generators_both()
    save_checkpoint("generators", result)
    sync_all()
    return result


# ---------------------------------------------------------------------------
# Step 2: Quantum commutant
# ---------------------------------------------------------------------------

def step2_quantum_commutant(alg_q, q_exprs, q_names, q_levels):
    """Compute the commutant of H_total in the quantum algebra exactly."""
    if is_complete("step2_quantum"):
        print("\n" + "=" * 70)
        print("STEP 2: COMMUTANT OF H_total (QUANTUM) [cached]")
        print("=" * 70)
        ckpt = load_checkpoint("step2_quantum_result")
        return ckpt["kernel"], ckpt["rank"], ckpt["brackets"], ckpt["domain"]

    print("\n" + "=" * 70)
    print("STEP 2: COMMUTANT OF H_total (QUANTUM, QQ[hbar])")
    print("=" * 70)
    t0 = time()
    heartbeat("step2_quantum")

    H_total = q_exprs[0] + q_exprs[1] + q_exprs[2]
    phase_vars = list(alg_q.all_vars)
    domain = QQ[hbar_sym]

    n_gen = len(q_exprs)

    # Phase A: Compute brackets [e_i, H_total] with per-bracket checkpointing
    brackets_ckpt = load_checkpoint("step2_q_brackets")
    if brackets_ckpt is not None:
        brackets = brackets_ckpt["brackets"]
        start_idx = len(brackets)
        print(f"  Resumed {start_idx} brackets from checkpoint")
    else:
        brackets = []
        start_idx = 0

    print(f"  Computing Moyal brackets [e_i, H_total]... "
          f"({start_idx}/{n_gen} done)")

    for i in range(start_idx, n_gen):
        check_shutdown("step2_quantum brackets")
        t1 = time()
        comm = alg_q.commutator(q_exprs[i], H_total)
        comm = cancel(comm)
        brackets.append(comm)
        n_terms = len(Add.make_args(expand(comm))) if comm != 0 else 0
        elapsed_b = time() - t1
        print(f"    [{i+1}/{n_gen}] {elapsed_b:.1f}s, "
              f"{'ZERO' if comm == 0 else f'{n_terms} terms'} "
              f"(level {q_levels[i]})", flush=True)

        if (i + 1) % 10 == 0:
            save_checkpoint("step2_q_brackets", {"brackets": brackets})
            heartbeat(f"step2_quantum_bracket_{i+1}")

    save_checkpoint("step2_q_brackets", {"brackets": brackets})

    n_zero = sum(1 for b in brackets if b == 0)
    print(f"\n  {n_zero} generators commute with H_total exactly")

    # Phase B: Build monomial-coefficient matrix
    print(f"\n  Building monomial-coefficient matrix over QQ[hbar]...")
    t_matrix = time()
    all_monoms = set()
    bracket_polys = []
    for comm in brackets:
        if comm == 0:
            bracket_polys.append({})
            continue
        expanded = expand(comm)
        p = Poly(expanded, *phase_vars, domain='QQ')
        md = p.as_dict()
        bracket_polys.append(md)
        all_monoms.update(md.keys())

    monom_list = sorted(all_monoms)
    monom_to_idx = {m: i for i, m in enumerate(monom_list)}
    n_mon = len(monom_list)
    print(f"  Matrix dimensions: {n_gen} generators x {n_mon} monomials")

    rows = []
    for md in bracket_polys:
        row = [domain.zero] * n_mon
        for monom, coeff in md.items():
            row[monom_to_idx[monom]] = domain.convert(coeff)
        rows.append(row)

    M = DomainMatrix(rows, (n_gen, n_mon), domain)
    print(f"  Matrix built in {time()-t_matrix:.1f}s")

    # Phase C: Exact rank
    print(f"  Computing exact rank over QQ[hbar]...")
    heartbeat("step2_quantum_rank")
    t_rank = time()
    M_rank = M.rank()
    print(f"  rank(M) = {M_rank} [{time()-t_rank:.1f}s]")
    print(f"  => Commutant dimension = {n_gen} - {M_rank} = {n_gen - M_rank}")

    # Phase D: Exact nullspace
    print(f"\n  Computing exact nullspace...")
    heartbeat("step2_quantum_nullspace")
    t_null = time()
    Mt = M.transpose()
    kernel = Mt.nullspace()
    print(f"  Nullspace computed in {time()-t_null:.1f}s")
    print(f"  Kernel dimension: {len(kernel)}")

    assert len(kernel) == n_gen - M_rank, \
        f"Kernel dim {len(kernel)} != {n_gen} - {M_rank} = {n_gen - M_rank}"

    elapsed = time() - t0
    print(f"\n  Step 2 completed in {elapsed:.1f}s")

    save_checkpoint("step2_quantum_result", {
        "kernel": kernel, "rank": M_rank,
        "brackets": brackets, "domain": domain,
    })
    mark_complete("step2_quantum", {
        "n_generators": n_gen,
        "rank": M_rank,
        "commutant_dim": n_gen - M_rank,
        "n_zero_brackets": n_zero,
        "n_monomials": n_mon,
        "time_s": round(elapsed, 1),
    })
    sync_all()
    return kernel, M_rank, brackets, domain


# ---------------------------------------------------------------------------
# Step 2b: Classical commutant
# ---------------------------------------------------------------------------

def step2b_classical_commutant(alg_c, c_exprs, c_names, c_levels):
    """Compute the commutant of H_total in the CLASSICAL algebra over QQ."""
    if is_complete("step2b_classical"):
        print("\n" + "=" * 70)
        print("STEP 2b: COMMUTANT OF H_total (CLASSICAL) [cached]")
        print("=" * 70)
        ckpt = load_checkpoint("step2b_classical_result")
        return ckpt["kernel"], ckpt["rank"]

    print("\n" + "=" * 70)
    print("STEP 2b: COMMUTANT OF H_total (CLASSICAL, QQ)")
    print("=" * 70)
    t0 = time()
    heartbeat("step2b_classical")

    H_total = c_exprs[0] + c_exprs[1] + c_exprs[2]
    phase_vars = list(alg_c.all_vars)
    n_gen = len(c_exprs)

    # Per-bracket checkpointing
    brackets_ckpt = load_checkpoint("step2b_cl_brackets")
    if brackets_ckpt is not None:
        brackets_cl = brackets_ckpt["brackets"]
        start_idx = len(brackets_cl)
        print(f"  Resumed {start_idx} brackets from checkpoint")
    else:
        brackets_cl = []
        start_idx = 0

    print(f"  Computing Poisson brackets {{e_i, H_total}}... "
          f"({start_idx}/{n_gen} done)")

    for i in range(start_idx, n_gen):
        check_shutdown("step2b_classical brackets")
        t1 = time()
        comm = alg_c.poisson_bracket(c_exprs[i], H_total)
        comm = cancel(comm)
        brackets_cl.append(comm)
        n_terms = len(Add.make_args(expand(comm))) if comm != 0 else 0
        elapsed_b = time() - t1
        print(f"    [{i+1}/{n_gen}] {elapsed_b:.1f}s, "
              f"{'ZERO' if comm == 0 else f'{n_terms} terms'} "
              f"(level {c_levels[i]})", flush=True)

        if (i + 1) % 10 == 0:
            save_checkpoint("step2b_cl_brackets", {"brackets": brackets_cl})

    save_checkpoint("step2b_cl_brackets", {"brackets": brackets_cl})

    n_zero_cl = sum(1 for b in brackets_cl if b == 0)
    print(f"\n  {n_zero_cl} generators have {{e_i, H_total}} = 0 exactly")

    print(f"\n  Building monomial-coefficient matrix over QQ...")
    t_matrix = time()
    all_monoms_cl = set()
    bracket_polys_cl = []
    for comm in brackets_cl:
        if comm == 0:
            bracket_polys_cl.append({})
            continue
        expanded = expand(comm)
        p = Poly(expanded, *phase_vars, domain='QQ')
        md = p.as_dict()
        bracket_polys_cl.append(md)
        all_monoms_cl.update(md.keys())

    monom_list_cl = sorted(all_monoms_cl)
    monom_to_idx_cl = {m: i for i, m in enumerate(monom_list_cl)}
    n_mon_cl = len(monom_list_cl)
    print(f"  Matrix: {n_gen} x {n_mon_cl}")

    rows_cl = []
    for md in bracket_polys_cl:
        row = [QQ.zero] * n_mon_cl
        for monom, coeff in md.items():
            row[monom_to_idx_cl[monom]] = QQ.convert(coeff)
        rows_cl.append(row)

    M_cl = DomainMatrix(rows_cl, (n_gen, n_mon_cl), QQ)
    print(f"  Matrix built in {time()-t_matrix:.1f}s")

    print(f"  Computing exact rank over QQ...")
    heartbeat("step2b_classical_rank")
    t_rank = time()
    M_cl_rank = M_cl.rank()
    print(f"  rank(M_cl) = {M_cl_rank} [{time()-t_rank:.1f}s]")
    print(f"  => Classical commutant dimension = "
          f"{n_gen} - {M_cl_rank} = {n_gen - M_cl_rank}")

    print(f"\n  Computing exact nullspace...")
    heartbeat("step2b_classical_nullspace")
    t_null = time()
    Mt_cl = M_cl.transpose()
    kernel_cl = Mt_cl.nullspace()
    print(f"  Nullspace computed in {time()-t_null:.1f}s")
    print(f"  Classical kernel dimension: {len(kernel_cl)}")

    elapsed = time() - t0
    print(f"\n  Step 2b completed in {elapsed:.1f}s")

    save_checkpoint("step2b_classical_result", {
        "kernel": kernel_cl, "rank": M_cl_rank,
    })
    mark_complete("step2b_classical", {
        "n_generators": n_gen,
        "rank": M_cl_rank,
        "commutant_dim": n_gen - M_cl_rank,
        "n_zero_brackets": n_zero_cl,
        "n_monomials": n_mon_cl,
        "time_s": round(elapsed, 1),
    })
    sync_all()
    return kernel_cl, M_cl_rank


# ---------------------------------------------------------------------------
# Step 2 comparison
# ---------------------------------------------------------------------------

def step2_comparison(kernel_q, rank_q, n_gen_q,
                     kernel_cl, rank_cl, n_gen_cl):
    """Compare classical and quantum commutant dimensions."""
    print("\n" + "=" * 70)
    print("STEP 2 COMPARISON: CLASSICAL vs QUANTUM COMMUTANT")
    print("=" * 70)

    dim_cl = len(kernel_cl)
    dim_q = len(kernel_q)

    print(f"\n  Classical algebra: {n_gen_cl} generators, rank {rank_cl}")
    print(f"    Commutant dimension: {dim_cl}")

    print(f"\n  Quantum algebra: {n_gen_q} generators, rank {rank_q}")
    print(f"    Commutant dimension: {dim_q}")

    if dim_q > dim_cl:
        result = "LARGER"
        print(f"\n  *** QUANTUM COMMUTANT IS LARGER: {dim_q} > {dim_cl} ***")
        print(f"  The quantum algebra has {dim_q - dim_cl} MORE conserved")
        print(f"  combinations than the classical one.")
        print(f"  => The extra hbar-dependent direction PARTICIPATES in")
        print(f"     a conserved quantity. Energy bound may be possible.")
    elif dim_q == dim_cl:
        result = "EQUAL"
        print(f"\n  Quantum and classical commutants have EQUAL dimension: {dim_q}")
        print(f"  The 117th direction does NOT participate in any conserved")
        print(f"  combination. Energy bound from this approach is NOT possible.")
    else:
        result = "SMALLER"
        print(f"\n  *** UNEXPECTED: quantum commutant SMALLER "
              f"({dim_q} < {dim_cl}) ***")
        print(f"  This should not happen — needs investigation.")

    mark_complete("step2_comparison", {
        "classical_commutant_dim": dim_cl,
        "quantum_commutant_dim": dim_q,
        "result": result,
    })
    return result


# ---------------------------------------------------------------------------
# Step 3: Analyze kernel vectors
# ---------------------------------------------------------------------------

def analyze_kernel_vectors(kernel_q, q_exprs, q_names, q_levels, alg_q, domain):
    """Analyze the conserved combinations for sign definiteness."""
    print("\n" + "=" * 70)
    print("STEP 3: ANALYZE CONSERVED COMBINATIONS")
    print("=" * 70)
    heartbeat("step3_analyze")

    if not kernel_q:
        print("  No kernel vectors — no conserved combinations exist.")
        mark_complete("step3_analyze", {"n_kernel_vectors": 0})
        return

    analysis = []

    for ki, kvec in enumerate(kernel_q):
        print(f"\n  --- Kernel vector {ki+1}/{len(kernel_q)} ---")

        coeffs = []
        for i in range(kvec.shape[0]):
            c = kvec[i, 0].element
            coeffs.append(c)

        nonzero = [(i, c) for i, c in enumerate(coeffs) if c != domain.zero]
        print(f"  {len(nonzero)} nonzero coefficients out of {len(coeffs)}")

        if len(nonzero) <= 20:
            for i, c in nonzero:
                print(f"    e_{i} ({q_names[i]}, level {q_levels[i]}): {c}")
        else:
            print(f"    (showing first 10)")
            for i, c in nonzero[:10]:
                print(f"    e_{i} ({q_names[i]}, level {q_levels[i]}): {c}")
            print(f"    ... and {len(nonzero)-10} more")

        print(f"\n  Checking hbar structure of coefficients...")
        has_hbar_coeffs = any(
            'hbar' in str(c) for _, c in nonzero
        )
        if has_hbar_coeffs:
            print(f"    Coefficients DEPEND on hbar")
        else:
            print(f"    Coefficients are rational (hbar-independent)")

        level_counts = {}
        for i, c in nonzero:
            lv = q_levels[i]
            level_counts[lv] = level_counts.get(lv, 0) + 1
        print(f"  Levels contributing: {dict(sorted(level_counts.items()))}")

        analysis.append({
            "n_nonzero": len(nonzero),
            "has_hbar_coeffs": has_hbar_coeffs,
            "level_counts": level_counts,
        })

    mark_complete("step3_analyze", {
        "n_kernel_vectors": len(kernel_q),
        "vectors": analysis,
    })
    sync_all()
    return kernel_q


# ---------------------------------------------------------------------------
# Step 4: Casimir from level-2 structure constants
# ---------------------------------------------------------------------------

def step4_casimir(alg_c, c_exprs, c_names, c_levels):
    """Construct quadratic Casimir from level-2 structure constants."""
    if is_complete("step4_casimir"):
        print("\n" + "=" * 70)
        print("STEP 4: CASIMIR [cached]")
        print("=" * 70)
        ckpt = load_checkpoint("step4_casimir_result")
        return ckpt["K"], ckpt["C_exact"], ckpt["r"]

    print("\n" + "=" * 70)
    print("STEP 4: CASIMIR FROM LEVEL-2 STRUCTURE CONSTANTS")
    print("=" * 70)
    t0 = time()
    heartbeat("step4_casimir")

    alg_l2 = NBodyAlgebra(3, 1, "1/r")
    phase_vars = list(alg_l2.all_vars)

    l2_exprs = list(alg_l2.hamiltonian_list)
    l2_names = list(alg_l2.hamiltonian_names)
    l2_levels = [0] * len(l2_exprs)

    n_l0 = len(l2_exprs)
    for i in range(n_l0):
        for j in range(i + 1, n_l0):
            expr = alg_l2.poisson_bracket(l2_exprs[i], l2_exprs[j])
            expr = cancel(expr)
            l2_exprs.append(expr)
            l2_names.append(f"{{{l2_names[i]},{l2_names[j]}}}")
            l2_levels.append(1)

    computed = set()
    for i in range(n_l0):
        for j in range(i + 1, n_l0):
            computed.add(frozenset({i, j}))

    frontier_1 = [i for i, lv in enumerate(l2_levels) if lv == 1]
    n_before = len(l2_exprs)
    for i in frontier_1:
        for j in range(n_before):
            if i == j:
                continue
            pair = frozenset({i, j})
            if pair in computed:
                continue
            computed.add(pair)
            expr = alg_l2.poisson_bracket(l2_exprs[i], l2_exprs[j])
            expr = cancel(expr)
            l2_exprs.append(expr)
            l2_names.append(f"{{{l2_names[i]},{l2_names[j]}}}")
            l2_levels.append(2)

    print(f"  Built {len(l2_exprs)} generators through level 2")

    all_monoms = set()
    poly_list = []
    for expr in l2_exprs:
        p = Poly(expand(expr), *phase_vars, domain='QQ')
        md = p.as_dict()
        poly_list.append(md)
        all_monoms.update(md.keys())

    monom_list = sorted(all_monoms)
    monom_to_idx = {m: i for i, m in enumerate(monom_list)}
    n_mon = len(monom_list)

    rows = []
    for md in poly_list:
        row = [QQ.zero] * n_mon
        for m, c in md.items():
            row[monom_to_idx[m]] = QQ.convert(c)
        rows.append(row)

    dm = DomainMatrix(rows, (len(rows), n_mon), QQ)
    rank = dm.rank()
    print(f"  Level-2 rank: {rank}")

    basis_indices = []
    for i in range(len(l2_exprs)):
        trial = [rows[j] for j in basis_indices] + [rows[i]]
        trial_dm = DomainMatrix(trial, (len(trial), n_mon), QQ)
        if trial_dm.rank() > len(basis_indices):
            basis_indices.append(i)
        if len(basis_indices) == rank:
            break

    print(f"  Basis indices: {basis_indices}")
    r = len(basis_indices)

    print(f"  Computing {r*(r-1)//2} brackets for structure constants...")
    heartbeat("step4_structure_constants")

    basis_dm_rows = [rows[idx] for idx in basis_indices]
    basis_dm = DomainMatrix(basis_dm_rows, (r, n_mon), QQ)

    C_exact = [[[Rational(0) for _ in range(r)]
                for _ in range(r)] for _ in range(r)]

    for a in range(r):
        i = basis_indices[a]
        for b in range(a + 1, r):
            j = basis_indices[b]
            bracket = alg_l2.poisson_bracket(l2_exprs[i], l2_exprs[j])
            bracket = cancel(bracket)
            expanded = expand(bracket)
            p = Poly(expanded, *phase_vars, domain='QQ')
            md = p.as_dict()

            rhs = [QQ.zero] * n_mon
            for m, c in md.items():
                if m in monom_to_idx:
                    rhs[monom_to_idx[m]] = QQ.convert(c)

            rhs_dm = DomainMatrix([[v] for v in rhs], (n_mon, 1), QQ)
            system = basis_dm.transpose()
            aug = system.hstack(rhs_dm)
            aug_rref, pivots = aug.rref()

            for k in range(r):
                val = QQ.zero
                for row_idx in range(min(r, aug_rref.shape[0])):
                    if row_idx < len(pivots) and pivots[row_idx] == k:
                        val = aug_rref[row_idx, r].element
                        break
                frac = Rational(int(val.numerator), int(val.denominator)) \
                    if hasattr(val, 'numerator') else Rational(val)
                C_exact[a][b][k] = frac
                C_exact[b][a][k] = -frac

    print(f"  Structure constants computed.")

    print(f"\n  Building Killing form over QQ (exact)...")
    K = [[Rational(0) for _ in range(r)] for _ in range(r)]
    for i in range(r):
        for j in range(i, r):
            val = Rational(0)
            for k in range(r):
                for l in range(r):
                    val += C_exact[i][k][l] * C_exact[j][l][k]
            K[i][j] = val
            K[j][i] = val

    print(f"  Killing form ({r}x{r}):")
    n_nonzero_K = sum(1 for i in range(r) for j in range(r) if K[i][j] != 0)
    print(f"    Nonzero entries: {n_nonzero_K}")

    K_rows = [[QQ.convert(K[i][j]) for j in range(r)] for i in range(r)]
    K_dm = DomainMatrix(K_rows, (r, r), QQ)
    K_rank = K_dm.rank()
    print(f"    Rank: {K_rank}")
    print(f"    Null space dimension: {r - K_rank}")

    if K_rank > 0:
        print(f"\n  Non-degenerate subspace has dimension {K_rank}")
        print(f"  Center dimension: {r - K_rank}")
    else:
        print(f"\n  Killing form is identically zero — "
              f"abelian or nilpotent algebra")

    elapsed = time() - t0
    print(f"\n  Step 4 completed in {elapsed:.1f}s")

    save_checkpoint("step4_casimir_result", {
        "K": K, "C_exact": C_exact, "r": r,
    })
    mark_complete("step4_casimir", {
        "rank": rank,
        "killing_rank": K_rank,
        "center_dim": r - K_rank,
        "n_nonzero_killing": n_nonzero_K,
        "time_s": round(elapsed, 1),
    })
    sync_all()
    return K, C_exact, r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time()

    print("=" * 70)
    print("ENERGY BOUND SEARCH")
    print("=" * 70)
    print(f"  SymPy version: {sp.__version__}")
    print(f"  S3 bucket: {S3_BUCKET or '(none — local mode)'}")
    print(f"  Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"  Started: {strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Restore checkpoints from S3 if available
    if S3_BUCKET:
        print(f"\n  Pulling checkpoints from S3...")
        s3_pull(f"{RESULTS_PREFIX}/checkpoints", CHECKPOINT_DIR)

    manifest = load_manifest()
    if manifest:
        completed = [k for k, v in manifest.items()
                     if v.get("status") == "complete"]
        print(f"  Previously completed steps: {completed}")

    # Step 1
    g_q, alg_single = step1_symmetry_check()
    check_shutdown("step1")

    # Build generators
    (alg_c, c_exprs, c_names, c_levels,
     alg_q, q_exprs, q_names, q_levels) = build_all_generators()

    n_gen_cl = len(c_exprs)
    n_gen_q = len(q_exprs)
    print(f"\n  Classical: {n_gen_cl} generators")
    print(f"  Quantum: {n_gen_q} generators")
    check_shutdown("generators")

    # Step 2: Quantum commutant
    kernel_q, rank_q, brackets_q, domain_q = \
        step2_quantum_commutant(alg_q, q_exprs, q_names, q_levels)
    check_shutdown("step2_quantum")

    # Step 2b: Classical commutant
    kernel_cl, rank_cl = \
        step2b_classical_commutant(alg_c, c_exprs, c_names, c_levels)
    check_shutdown("step2b_classical")

    # Comparison
    result = step2_comparison(kernel_q, rank_q, n_gen_q,
                              kernel_cl, rank_cl, n_gen_cl)

    # Step 3: Analyze kernel vectors
    if kernel_q:
        analyze_kernel_vectors(kernel_q, q_exprs, q_names, q_levels,
                               alg_q, domain_q)

    # Step 4: Casimir
    try:
        step4_casimir(alg_c, c_exprs, c_names, c_levels)
    except Exception as e:
        print(f"\n  Casimir construction failed: {e}")
        import traceback
        traceback.print_exc()

    # Final summary
    total_time = time() - t_start
    print(f"\n\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Classical commutant dimension: {len(kernel_cl)}")
    print(f"    ({n_gen_cl} generators, rank {rank_cl} under ad_H)")
    print(f"  Quantum commutant dimension: {len(kernel_q)}")
    print(f"    ({n_gen_q} generators, rank {rank_q} under ad_H)")
    print(f"  Comparison result: {result}")
    print(f"\n  Total time: {total_time:.1f}s")

    # Save final results JSON
    results_dir = os.path.join(SCRIPT_DIR, "..", "results", "energy_bound")
    os.makedirs(results_dir, exist_ok=True)
    final = {
        "classical_generators": n_gen_cl,
        "quantum_generators": n_gen_q,
        "classical_rank_ad_H": rank_cl,
        "quantum_rank_ad_H": rank_q,
        "classical_commutant_dim": len(kernel_cl),
        "quantum_commutant_dim": len(kernel_q),
        "comparison": result,
        "total_time_s": round(total_time, 1),
        "completed_at": strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest": load_manifest(),
    }
    final_path = os.path.join(results_dir, "energy_bound_results.json")
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n  Results saved to: {final_path}")

    mark_complete("final", final)
    sync_all()

    if S3_BUCKET:
        s3_cp(final_path, f"results/energy_bound/energy_bound_results.json")


if __name__ == "__main__":
    main()
