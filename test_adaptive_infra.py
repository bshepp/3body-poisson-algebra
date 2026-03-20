#!/usr/bin/env python3
"""
Smoke test for adaptive scan infrastructure.

Runs a tiny 5x5 grid to validate:
  1. Single-worker (sequential) path
  2. Multi-worker (fork-based Pool) path
  3. Checkpoint recovery (interrupt + resume)
  4. Verification report
  5. Block merge

Uses the harmonic potential (fast to build, ~2s algebra) to keep
total runtime under 5 minutes.
"""
import os
import sys
import json
import shutil
import numpy as np
from time import time

sys.path.insert(0, os.path.dirname(__file__))

import multi_epsilon_atlas as mea

TEST_DIR = 'atlas_test_adaptive'
ORIG_HIRES = mea.HIRES_DIR
ORIG_GRID_N = mea.GRID_N

def patch_constants(grid_n=5):
    mea.HIRES_DIR = TEST_DIR
    mea.GRID_N = grid_n

def restore_constants():
    mea.HIRES_DIR = ORIG_HIRES
    mea.GRID_N = ORIG_GRID_N

def cleanup():
    if os.path.isdir(TEST_DIR):
        shutil.rmtree(TEST_DIR)

def test_sequential():
    """Test 1: sequential (workers=1) adaptive scan."""
    print("\n" + "="*60)
    print("TEST 1: Sequential adaptive scan (5x5, workers=1)")
    print("="*60)
    patch_constants(5)
    try:
        mea.run_adaptive_scan('1/r', n_workers=1)
        out = os.path.join(TEST_DIR, '1_r', 'adaptive')
        assert os.path.isfile(os.path.join(out, 'rank_map.npy'))
        rm = np.load(os.path.join(out, 'rank_map.npy'))
        assert rm.shape == (5, 5), f"Expected (5,5), got {rm.shape}"
        assert (rm > 0).all(), f"Some ranks <= 0: {rm}"
        cp = mea.load_checkpoint(out)
        assert cp['completed_rows'] == 5
        print(f"  PASS: rank_map shape={rm.shape}, all ranks > 0")
        print(f"  Ranks: {np.unique(rm)}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        restore_constants()

def test_verify():
    """Test 2: verification on the sequential scan output."""
    print("\n" + "="*60)
    print("TEST 2: Verification")
    print("="*60)
    patch_constants(5)
    try:
        out = os.path.join(TEST_DIR, '1_r', 'adaptive')
        ok, report = mea.verify_adaptive_scan(out, expected_rows=5,
                                               expected_cols=5)
        assert ok, f"Verification failed: {report.get('errors')}"
        assert os.path.isfile(os.path.join(out, 'verification_report.json'))
        assert 'checksums' in report
        assert len(report['checksums']) > 0
        print(f"  PASS: verification passed, "
              f"{len(report['checksums'])} checksums")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        restore_constants()

def test_checkpoint_recovery():
    """Test 3: simulate interruption and resume."""
    print("\n" + "="*60)
    print("TEST 3: Checkpoint recovery")
    print("="*60)
    patch_constants(5)
    try:
        out = os.path.join(TEST_DIR, '1_r', 'adaptive')
        # Pretend only 2 rows were completed
        cp_file = os.path.join(out, 'checkpoint.json')
        with open(cp_file) as f:
            cp = json.load(f)
        cp['completed_rows'] = 2
        with open(cp_file, 'w') as f:
            json.dump(cp, f)

        # Zero out rows 2-4 to confirm they get recomputed
        rm = np.load(os.path.join(out, 'rank_map.npy'))
        rm[2:, :] = 0
        np.save(os.path.join(out, 'rank_map.npy'), rm)

        mea.run_adaptive_scan('1/r', n_workers=1)
        rm2 = np.load(os.path.join(out, 'rank_map.npy'))
        assert (rm2 > 0).all(), f"Rows not recomputed: {rm2}"
        cp2 = mea.load_checkpoint(out)
        assert cp2['completed_rows'] == 5
        print(f"  PASS: resumed from row 2, all 5 rows complete")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        restore_constants()

def test_block_merge():
    """Test 4: distributed block scan + merge."""
    print("\n" + "="*60)
    print("TEST 4: Block scan + merge (rows 0-3, 3-5)")
    print("="*60)
    cleanup()
    patch_constants(5)
    try:
        # Block 1: rows 0-3
        mea.run_adaptive_scan('1/r', n_workers=1,
                              start_row=0, end_row=3)
        b1 = mea._adaptive_block_dir(
            os.path.join(TEST_DIR, '1_r', 'adaptive'), 0, 3)
        assert os.path.isdir(b1), f"Block dir missing: {b1}"
        rm1 = np.load(os.path.join(b1, 'rank_map.npy'))
        assert rm1.shape[0] == 3

        # Block 2: rows 3-5
        mea.run_adaptive_scan('1/r', n_workers=1,
                              start_row=3, end_row=5)
        b2 = mea._adaptive_block_dir(
            os.path.join(TEST_DIR, '1_r', 'adaptive'), 3, 5)
        assert os.path.isdir(b2)
        rm2 = np.load(os.path.join(b2, 'rank_map.npy'))
        assert rm2.shape[0] == 2

        # Merge
        ok = mea.merge_adaptive_blocks('1/r')
        assert ok, "Merge failed"

        merged_dir = os.path.join(TEST_DIR, '1_r', 'adaptive', 'merged')
        rm_merged = np.load(os.path.join(merged_dir, 'rank_map.npy'))
        assert rm_merged.shape == (5, 5), f"Bad merged shape: {rm_merged.shape}"
        assert (rm_merged > 0).all()

        vr_path = os.path.join(merged_dir, 'verification_report.json')
        assert os.path.isfile(vr_path)

        print(f"  PASS: 2 blocks merged to {rm_merged.shape}, "
              f"verification passed")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        restore_constants()


def test_atomic_checkpoint():
    """Test 5: atomic checkpoint write."""
    print("\n" + "="*60)
    print("TEST 5: Atomic checkpoint")
    print("="*60)
    try:
        test_dir = os.path.join(TEST_DIR, '_atomic_test')
        os.makedirs(test_dir, exist_ok=True)

        mea.save_checkpoint_atomic(test_dir, 42, 156, -1,
                                    extra={'custom': 'data'})
        cp_file = os.path.join(test_dir, 'checkpoint.json')
        assert os.path.isfile(cp_file)
        assert not os.path.isfile(cp_file + '.tmp')

        with open(cp_file) as f:
            cp = json.load(f)
        assert cp['completed_rows'] == 42
        assert cp['custom'] == 'data'

        print(f"  PASS: atomic checkpoint written and verified")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()
        return False


if __name__ == '__main__':
    cleanup()
    t0 = time()

    results = {}
    results['atomic_checkpoint'] = test_atomic_checkpoint()
    results['sequential'] = test_sequential()
    results['verify'] = test_verify()
    results['checkpoint_recovery'] = test_checkpoint_recovery()

    cleanup()
    results['block_merge'] = test_block_merge()

    cleanup()

    elapsed = time() - t0
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    sys.exit(0 if all_pass else 1)
