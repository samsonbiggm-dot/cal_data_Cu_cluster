# -*- coding: utf-8 -*-
"""
stage4_convergence_fixed.py  v5
=================================
v5 修改记录（对应 config.py v2）：
  FIX A - CO2* 使用两阶段约束优化（frozen O → full relax），确保化学吸附
  FIX B - 优化后 check_co2_angle 验证 O-C-O < 160°
  FIX C - SPEED 字典统一速度参数（max_cycle=150，conv_tol=1e-8，
          hess_stepsize=0.003，geo_maxsteps=150）
  继承 v4 所有修复：
    auto_spin / CLUSTERS 统一 / 腔极化动态提取 / 振动后重建 SCF /
    断点续算 / CHE 估算势垒
"""
import numpy as np
import warnings, json, os
warnings.filterwarnings('ignore')

from pyscf import gto
from pyscf.dft import rks as pyscf_rks, uks as pyscf_uks
from pyscf.geomopt import geometric_solver
from nqeddft import Cavity, QEDRKS, QEDUKS
from nqeddft.phonon import QEDPhonon
from config import (CLUSTERS as CFG_CLUSTERS, ADS_OFFSETS,
                    XC_FUNCTIONAL, BASIS, LAMBDA_VALS, SPEED,
                    cm_to_au, AU2EV, build_mol_geom,
                    check_co2_angle, get_co_polarization)
import stage0_reference as s0


# ═══════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════

def _get_spin(cluster_name, ads_name=None):
    cl     = CFG_CLUSTERS[cluster_name]
    n_cu   = len([l for l in cl['geom'].strip().split('\n')])
    n_cu_e = n_cu * 11
    total  = n_cu_e if ads_name is None else n_cu_e + ADS_OFFSETS[ads_name]['n_electrons']
    return total % 2


def make_mol(cluster_name, ads_name=None):
    geom, spin = build_mol_geom(cluster_name, ads_name)
    spin = _get_spin(cluster_name, ads_name)
    mol  = gto.M(
        atom    = geom,
        basis   = BASIS,
        spin    = spin,
        charge  = 0,
        unit    = 'Angstrom',
        verbose = 0,
    )
    return mol


def make_mf(mol, cavity, xc, use_uks):
    mf = (QEDUKS if use_uks else QEDRKS)(mol, cavity)
    mf.xc          = xc
    mf.verbose     = 0
    mf.max_cycle   = SPEED['max_cycle']
    mf.conv_tol    = SPEED['conv_tol']
    mf.level_shift = SPEED['level_shift']
    mf.init_guess  = 'atom'
    return mf


def safe_kernel(mf, label=""):
    try:
        e  = mf.kernel()
        if getattr(mf, 'converged', True):
            return float(e), True
        print(f"  ⚠ [{label}] 未收敛，level_shift=0.3 重试...")
        mf.level_shift = 0.3
        mf.max_cycle   = SPEED['max_cycle']
        e  = mf.kernel()
        mf.level_shift = 0.0
        ok = getattr(mf, 'converged', True)
        if not ok:
            print(f"  ⚠ [{label}] 仍未收敛")
        return float(e), ok
    except Exception as ex:
        print(f"  ✗ [{label}]: {type(ex).__name__}: {ex}")
        return float('nan'), False


def get_co_stretch(freqs_list, min_freq=1000.0, max_freq=3000.0):
    real = [float(f) for f in freqs_list
            if min_freq < float(f) < max_freq]
    return max(real) if real else float('nan')


def match_and_shift(freqs_free, freqs_cav,
                    min_freq=100.0, max_shift=500.0):
    ref = [(i, float(f)) for i, f in enumerate(freqs_free)
           if float(f) > min_freq]
    cav = [(i, float(f)) for i, f in enumerate(freqs_cav)
           if float(f) > min_freq]
    shifts, used = {}, set()
    for i_r, f_r in ref:
        candidates = [(i_c, f_c) for i_c, f_c in cav if i_c not in used]
        if not candidates:
            break
        i_c, f_c = min(candidates, key=lambda x: abs(x[1] - f_r))
        shift = f_c - f_r
        shifts[i_r] = {
            'freq_free': f_r,
            'freq_cav':  f_c  if abs(shift) < max_shift else float('nan'),
            'shift':     shift if abs(shift) < max_shift else float('nan'),
        }
        used.add(i_c)
    return shifts


# ═══════════════════════════════════════════════════════════════════════
# CO2* 两阶段约束优化
# ═══════════════════════════════════════════════════════════════════════

def _opt_co2_twophase(mol0, cavity, use_uks, xc):
    """
    两阶段 CO2* 优化（QED 腔内）：
    Phase 1: 冻结 O 原子 → Phase 2: 全局弛豫
    确保优化器不逃到线性物理吸附极小值。
    """
    syms   = [mol0.atom_symbol(i) for i in range(mol0.natm)]
    O_idxs = [i for i, s in enumerate(syms) if s == 'O']

    mf = make_mf(mol0, cavity, xc, use_uks)

    # 阶段 1
    print('  [CO2* opt] Phase 1: frozen O...')
    try:
        mol_p1 = geometric_solver.optimize(
            mf.Gradients(),
            cartesian   = True,
            maxsteps    = 60,
            constraints = {'freeze': {'atoms': O_idxs}},
            assert_convergence = False,
        )
    except Exception as e:
        print(f'  Phase 1 fallback: {e}')
        mol_p1 = mol0

    # 阶段 2
    print('  [CO2* opt] Phase 2: full relax...')
    mf2 = make_mf(mol_p1, cavity, xc, use_uks)
    mol_opt = geometric_solver.optimize(
        mf2.Gradients(),
        cartesian          = True,
        maxsteps           = SPEED['geo_maxsteps'],
        assert_convergence = False,
    )

    check_co2_angle(mol_opt)
    mf2.mol = mol_opt
    mf2.mol.build()
    mf2.kernel()
    return mol_opt, mf2


# ═══════════════════════════════════════════════════════════════════════
# 计算参数
# ═══════════════════════════════════════════════════════════════════════

XC            = XC_FUNCTIONAL
LAMBDA        = 0.02
CLUSTER_SIZES = list(CFG_CLUSTERS.keys())
ADSORBATES    = list(ADS_OFFSETS.keys())

# ═══════════════════════════════════════════════════════════════════════
# 断点续算
# ═══════════════════════════════════════════════════════════════════════

CHECKPOINT = 'stage4_results.json'

def _save(results):
    tmp = CHECKPOINT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x:
                  None if (isinstance(x, float) and np.isnan(x)) else x)
    os.replace(tmp, CHECKPOINT)

if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        results = json.load(f)
    print(f'[RESUME] Loaded {CHECKPOINT}')
else:
    results = {}


# ═══════════════════════════════════════════════════════════════════════
# 参考气相分子能量（λ=0）
# ═══════════════════════════════════════════════════════════════════════

OMEGA_DUMMY = 0.1
cav0 = Cavity().add_mode(OMEGA_DUMMY, 0.0, [0, 0, 1])

from config import GAS_REFS
GAS_MOL_MAP = {'CO2*': 'CO2', 'COOH*': 'COOH', 'CO*': 'CO'}

E_ref = results.get('_gas_refs', {})
if not E_ref:
    print("=" * 65)
    print("参考分子能量（lambda=0）")
    print("=" * 65)
    from config import GAS_REFS as _GR
    _GR_ext = dict(_GR)
    _GR_ext['COOH'] = ("C  0.000  0.000  0.000\n"
                        "O  1.250  0.000  0.000\n"
                        "O -0.550  1.140  0.000\n"
                        "H -1.530  1.100  0.000")
    _ATOMIC_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'Cu': 29}
    for name, atom_str in _GR_ext.items():
        n_e    = sum(_ATOMIC_Z.get(l.split()[0], 0)
                     for l in atom_str.replace(';', '\n').split('\n') if l.strip())
        spin_g = n_e % 2
        mol_r  = gto.M(atom=atom_str, basis=BASIS,
                       spin=spin_g, charge=0, unit='Angstrom', verbose=0)
        mf_r   = (QEDUKS if spin_g else QEDRKS)(mol_r, cav0)
        mf_r.xc        = XC
        mf_r.verbose   = 0
        mf_r.max_cycle = SPEED['max_cycle']
        mf_r.conv_tol  = SPEED['conv_tol']
        mf_r.init_guess = 'atom'
        e_r, ok_r = safe_kernel(mf_r, name)
        E_ref[name] = e_r
        print(f"  {name:6s}  spin={spin_g}  e={e_r:.8f} Ha  "
              f"{'✓' if ok_r else '✗'}")
    results['_gas_refs'] = E_ref
    _save(results)


# ═══════════════════════════════════════════════════════════════════════
# 主计算循环
# ═══════════════════════════════════════════════════════════════════════

for size in CLUSTER_SIZES:
    if size not in results:
        results[size] = {}

    use_uks_slab = (_get_spin(size) != 0)
    print(f"\n{'='*65}\n团簇 {size}  (spin={_get_spin(size)})\n{'='*65}")

    # 裸团簇能量
    if '_slab_energy' not in results[size]:
        mol_slab = make_mol(size)
        mf_slab  = make_mf(mol_slab, cav0, XC, use_uks_slab)
        e_slab, ok_slab = safe_kernel(mf_slab, f"{size} slab")
        results[size]['_slab_energy'] = e_slab
        _save(results)
        print(f"  裸团簇: {e_slab:.8f} Ha  {'✓' if ok_slab else '✗'}")
    else:
        e_slab = results[size]['_slab_energy']
        print(f"  [SKIP] 裸团簇: {e_slab:.8f} Ha")

    for ads in ADSORBATES:
        if ads in results[size] and results[size][ads].get('_done'):
            print(f"\n  [SKIP] {ads}")
            continue

        if ads not in results[size]:
            results[size][ads] = {}

        use_uks = (_get_spin(size, ads) != 0)
        is_co2  = (ads == 'CO2*')
        print(f"\n  --- {ads} (spin={_get_spin(size, ads)}) ---")

        mol_ads = make_mol(size, ads)

        # ── λ=0 SCF（CO2* 使用两阶段优化）──────────────────────────
        if is_co2:
            mol_opt0, mf0 = _opt_co2_twophase(mol_ads, cav0, use_uks, XC)
        else:
            mf0 = make_mf(mol_ads, cav0, XC, use_uks)
            mol_opt0 = geometric_solver.optimize(
                mf0.Gradients(),
                cartesian          = True,
                maxsteps           = SPEED['geo_maxsteps'],
                assert_convergence = False,
            )
            mf0.mol = mol_opt0
            mf0.mol.build()

        e0, ok0 = safe_kernel(mf0, f"{size}+{ads} λ=0")

        gas_key  = GAS_MOL_MAP.get(ads)
        e_mol    = E_ref.get(gas_key, float('nan'))
        e_ads_eV = ((e0 - e_slab - e_mol) * AU2EV
                    if not any(np.isnan(x) for x in [e0, e_slab, e_mol])
                    else float('nan'))
        print(f"    吸附能 (λ=0): {e_ads_eV:+.3f} eV")
        results[size][ads]['e_ads_lam0'] = e_ads_eV

        # ── λ=0 振动 ──────────────────────────────────────────────────
        freq0, freqs0_list = float('nan'), []
        if ok0:
            ph0 = QEDPhonon(mf0)
            try:
                hess0 = ph0.numerical_hessian_fast(
                    stepsize=SPEED['hess_stepsize'], verbose=False)
                freqs0, _ = ph0.harmonic_analysis(hess0)
                freqs0_list = list(freqs0)
                freq0 = get_co_stretch(freqs0_list)
                print(f"    C-O 伸缩 (λ=0): {freq0:.1f} cm⁻¹")
            except Exception as ex:
                print(f"    ✗ 振动分析失败: {ex}")
        results[size][ads]['freq_lam0'] = freq0

        # FIX 3: 从优化构型提取极化方向
        pol = get_co_polarization(mol_opt0, ads)

        # ── 腔 SCF + 振动 ─────────────────────────────────────────────
        omega_res = cm_to_au(freq0) if not np.isnan(freq0) else OMEGA_DUMMY
        cav_res   = Cavity().add_mode(omega_res, LAMBDA, pol)

        mol_cav = make_mol(size, ads)

        if is_co2:
            mol_opt_cav, mf_cav = _opt_co2_twophase(mol_cav, cav_res, use_uks, XC)
        else:
            mf_cav = make_mf(mol_cav, cav_res, XC, use_uks)
            mol_opt_cav = geometric_solver.optimize(
                mf_cav.Gradients(),
                cartesian          = True,
                maxsteps           = SPEED['geo_maxsteps'],
                assert_convergence = False,
            )
            mf_cav.mol = mol_opt_cav
            mf_cav.mol.build()

        e_cav, ok_cav = safe_kernel(mf_cav, f"{size}+{ads} λ={LAMBDA}")

        # 振动分析前重建
        if ok_cav:
            mf_cav.mol.build()
            mf_cav.kernel()

        freq_cav, shift = float('nan'), float('nan')
        if ok_cav and freqs0_list:
            ph_cav = QEDPhonon(mf_cav)
            try:
                hess_cav = ph_cav.numerical_hessian_fast(
                    stepsize=SPEED['hess_stepsize'], verbose=False)
                freqs_cav, _ = ph_cav.harmonic_analysis(hess_cav)
                freqs_cav_list = list(freqs_cav)
                freq_cav = get_co_stretch(freqs_cav_list)

                sm   = match_and_shift(freqs0_list, freqs_cav_list)
                best = min(
                    (v for v in sm.values()
                     if not np.isnan(v['freq_free'])
                     and not np.isnan(v['shift'])),
                    key=lambda x: abs(x['freq_free'] - freq0),
                    default=None,
                )
                shift = best['shift'] if best else float('nan')
                print(f"    C-O 伸缩 (λ={LAMBDA}): {freq_cav:.1f} cm⁻¹, "
                      f"Δω={shift:+.2f} cm⁻¹")
            except Exception as ex:
                print(f"    ✗ 腔中振动分析失败: {ex}")

        results[size][ads]['freq_lam_cav'] = freq_cav
        results[size][ads]['freq_shift']   = shift
        results[size][ads]['_done']        = True
        _save(results)


# ═══════════════════════════════════════════════════════════════════════
# 汇总表
# ═══════════════════════════════════════════════════════════════════════

def fmt(x, s="+.3f"):
    return f"{x:{s}}" if (x is not None and not np.isnan(x)) else "   nan"

print(f"\n{'='*65}")
for label, key, sfmt in [
    ("吸附能 (eV)，λ=0",              "e_ads_lam0",   "+.3f"),
    ("C-O 频率 (cm⁻¹)，λ=0",          "freq_lam0",    ".1f" ),
    (f"腔诱导频移 (cm⁻¹)，λ={LAMBDA}", "freq_shift",   "+.2f"),
]:
    print(f"\n{label}")
    print(f"{'团簇':>6}", end="")
    for ads in ADSORBATES:
        print(f"  {ads:>10}", end="")
    print()
    for size in CLUSTER_SIZES:
        print(f"{size:>6}", end="")
        for ads in ADSORBATES:
            v = results.get(size, {}).get(ads, {}).get(key, float('nan'))
            if v is None: v = float('nan')
            print(f"  {fmt(v, sfmt):>10}", end="")
        print()

print(f"\n{'='*65}")
print("反应能 CO2*→COOH* (meV)，CHE 近似 (U=0 V)")
print(f"{'团簇':>6}  {'ΔE (meV)':>12}")
print("-" * 22)
for size in CLUSTER_SIZES:
    e_co2  = results.get(size, {}).get('CO2*',  {}).get('e_ads_lam0', float('nan'))
    e_cooh = results.get(size, {}).get('COOH*', {}).get('e_ads_lam0', float('nan'))
    if e_co2 is None:  e_co2  = float('nan')
    if e_cooh is None: e_cooh = float('nan')
    bar = (e_cooh - e_co2) * 1000 if not np.isnan(e_co2 + e_cooh) else float('nan')
    print(f"{size:>6}  {fmt(bar, '+.1f'):>12}")

print("\nStage 4 v5 完成。")
