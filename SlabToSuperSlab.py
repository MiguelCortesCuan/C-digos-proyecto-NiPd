# -*- coding: utf-8 -*-
"""
SlabToSuperSlab.py — robust in-plane supercell builder (+ trimming + vacuum + pools)
====================================================================================
• Scans relaxed QE *.out in ROOT_DIR, picks lowest-energy T-variant.
• Optionally trims the number of interatomic planes (keep bottom N).
• Resets vacuum (either total or per-side) and recenters the slab.
• Builds an edge-safe in-plane supercell and freezes bottom planes
  (after trimming) carried over into the supercell.
• Writes: structures/*.xyz, *.cif and inputs/*.in, *.lsf
"""

from pathlib import Path
from datetime import date
import re
import math
import numpy as np

from ase.io import read, write
from ase import Atoms
from ase.constraints import FixAtoms

# ───────────── USER SETTINGS ───────────────────────────────────────────────
ROOT_DIR         = Path(r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\Supercell\SurfaceEnergy\111_2x2\relax")
OUTPUT_DIR       = Path(r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\Supercell\SurfaceEnergy\111_4x4")
PRESERVE_SUBDIRS = True

STRUCT_SUBDIR         = "Structures"
RELAX_INPUT_SUBDIR    = "Relax"
PRERELAX_INPUT_SUBDIR = "Prerelax"

# Supercell & freezing
REPEAT           = (2, 2, 1)     # (nx, ny, nz), nz must be 1 for slabs here
FROZEN_PLANES    = 2             # bottom N planes (of the TRIMMED slab) to freeze

# ### NEW: plane trimming & vacuum
TARGET_PLANES    = 4             # keep bottom N planes from the *.out; set None to keep all
VACUUM_MODE      = "total"       # "total" -> VACUUM_ANG total; "each" -> VACUUM_ANG each side
VACUUM_ANG       = 20.0          # Å; e.g., 20 (total) OR 10 with VACUUM_MODE="each"

# QE inputs
KPOINTS          = (3, 3, 1)     # automatic mesh (will be written to input)
KSHIFT           = (1, 1, 0)     # half-shifts to avoid Γ-only artifacts
ECUTWFC, ECUTRHO = 60, 480
DEGAUSS          = 0.0146997171
PSEUDO_DIR       = "/tmpu/isholek_g/isholek/MIGUEL/QE/PP"
PSEUDO_MAP       = {"Ni": "Ni.upf", "Pd": "Pd.upf"}
START_MAGN       = {"Ni": 0.75, "Pd": 0.1}

# LSF job defaults (+ pool flags auto-chosen from cores & k-mesh)
QUEUE_NAME       = "q_hpc"
QUEUE_CORES      = 48            # total MPI ranks (since OMP=1 below)
OMP_THREADS      = 1
RESOURCE_STRING  = 'span[ptile=16]'

# --- PRE-RELAX profile (quick fall-into-basin) ---
MAKE_PRERELAX   = True
PRE_SUFFIX      = "quick"     # filenames get *_quick.in / *_quick.lsf
PRE_KPOINTS     = (2, 2, 1)
PRE_KSHIFT      = (1, 1, 0)
PRE_SMEARING    = "mv"
PRE_DEGAUSS     = 0.03        # ~0.41 eV
PRE_CONV_THR    = 1.0e-6      # SCF
PRE_FORCE_THR   = 3.0e-3      # ionic
PRE_MIX_BETA    = 0.30
PRE_ELEC_STEPS  = 150


# ───────────── Internals ───────────────────────────────────────────────────
END_OK_RE        = re.compile(r"JOB\s*DONE", re.I)
ENERGY_RE_STRICT = re.compile(r"!\s*total energy\s*=\s*([\-0-9.]+)\s*Ry", re.I)
ENERGY_RE_FALLBK = re.compile(r"\btotal energy\s*=\s*([\-0-9.]+)\s*Ry", re.I)
T_SUFFIX_RE      = re.compile(r"(?i)(?:^|[ _-])T\d+(?=$|[ _-])")

# ─────────── helpers: energy parsing & grouping ────────────────────────────
def extract_final_energy_ry(text: str):
    m = ENERGY_RE_STRICT.findall(text)
    if m:
        try: return float(m[-1])
        except ValueError: pass
    m2 = ENERGY_RE_FALLBK.findall(text)
    if m2:
        try: return float(m2[-1])
        except ValueError: pass
    return None

def system_key(out_path: Path):
    parent = str(out_path.parent.resolve())
    base   = T_SUFFIX_RE.sub("", out_path.stem).strip("_- ")
    return (parent, base)

# ─────────── helpers: geometry / planes / replication ─────────────────────
def _n_hat_from_cell(cell):
    c = np.asarray(cell, float)[2]
    n = np.linalg.norm(c)
    return (c / n) if n > 1e-12 else np.array([0.0, 0.0, 1.0])

def group_planes_with_tol(zvals, tol_override=None):
    z = np.asarray(zvals, float)
    order = np.argsort(z)
    z_sorted = z[order]
    if len(z_sorted) <= 1:
        return [order.tolist()], np.array([float(z.mean())]), (1e-4 if tol_override is None else float(tol_override))
    dif = np.diff(z_sorted)
    if tol_override is None:
        nz = dif[dif > 1e-6]
        tol = max(1e-4, 0.30 * np.percentile(nz, 95)) if len(nz) else 1e-4
    else:
        tol = float(tol_override)
    planes = [[order[0]]]
    for k in range(1, len(z_sorted)):
        if (z_sorted[k] - z_sorted[k-1]) <= tol: planes[-1].append(order[k])
        else:                                     planes.append([order[k]])
    centers = np.array([z[p].mean() for p in planes], float)
    return planes, centers, tol

def supercell_expand_crop_with_map(slab1: Atoms, nx: int, ny: int):
    A = np.asarray(slab1.get_cell(), float)
    c = A[2]; cn = np.linalg.norm(c) or 1.0
    n_hat = c / cn
    a1 = A[0] - np.dot(A[0], n_hat) * n_hat
    a2 = A[1] - np.dot(A[1], n_hat) * n_hat
    if np.linalg.norm(a1) < 1e-10 or np.linalg.norm(a2) < 1e-10:
        a1, a2 = A[0], A[1]
    A_tgt = np.array([a1 * nx, a2 * ny, n_hat * cn], float)

    sfrac = slab1.get_scaled_positions(wrap=True)
    sfrac = np.minimum(sfrac, 1.0 - 1e-12)
    symbols = np.array(slab1.get_chemical_symbols())

    pos_list, sym_list, src_idx = [], [], []
    for i in range(nx + 1):
        for j in range(ny + 1):
            f = sfrac.copy()
            f[:, 0] = (f[:, 0] + i) / nx
            f[:, 1] = (f[:, 1] + j) / ny
            mask = (f[:, 0] < 1.0 - 1e-12) & (f[:, 1] < 1.0 - 1e-12)
            if not np.any(mask): continue
            r = f[mask] @ A_tgt
            pos_list.append(r)
            sym_list.append(symbols[mask])
            src_idx.append(np.nonzero(mask)[0])

    R = np.vstack(pos_list)
    S = np.concatenate(sym_list).tolist()
    src_map = np.concatenate(src_idx)
    scell = Atoms(symbols=S, positions=R, cell=A_tgt, pbc=(True, True, slab1.pbc[2]))
    frac = scell.get_scaled_positions(wrap=True)
    scell.set_scaled_positions(np.minimum(frac, 1.0 - 1e-12))
    return scell, src_map

def freeze_from_base_planes(slab1: Atoms, scell: Atoms, src_map, n_freeze: int):
    if n_freeze <= 0: return []
    z1 = (slab1.get_positions() @ _n_hat_from_cell(slab1.get_cell()))
    planes1, _, _ = group_planes_with_tol(z1)
    base_freeze = set(i for k in range(min(n_freeze, len(planes1))) for i in planes1[k])
    freeze_idx = [k for k, src in enumerate(src_map) if src in base_freeze]
    if freeze_idx:
        scell.set_constraint(FixAtoms(indices=sorted(set(freeze_idx))))
    return sorted(set(freeze_idx))

def format_plane_counts(counts):
    return "[" + ", ".join(str(x) for x in counts) + "]"

# ### NEW: trimming planes (keep bottom N) and setting vacuum
def trim_keep_bottom_planes(slab: Atoms, keep_n: int) -> Atoms:
    if keep_n is None:  # no trim
        return slab.copy()
    z = (slab.get_positions() @ _n_hat_from_cell(slab.get_cell()))
    planes, _, _ = group_planes_with_tol(z)
    keep_n = min(keep_n, len(planes))
    keep_idx = sorted([i for k in range(keep_n) for i in planes[k]])
    trimmed = slab[keep_idx]  # ASE indexing returns a new Atoms
    trimmed.set_cell(slab.get_cell(), scale_atoms=False)
    trimmed.set_pbc(slab.get_pbc())
    return trimmed

def set_vacuum_and_center(slab: Atoms, vac_value: float, mode: str = "total") -> Atoms:
    """mode='total' -> total vacuum thickness, 'each' -> per-side amount"""
    A = np.asarray(slab.get_cell(), float)
    n_hat = _n_hat_from_cell(A)
    pos = slab.get_positions()
    zproj = pos @ n_hat
    zmin, zmax = float(zproj.min()), float(zproj.max())
    thickness = zmax - zmin
    if mode.lower().startswith("each"):
        vac_each = float(vac_value)
        c_new = thickness + 2.0 * vac_each
        z_bottom_target = vac_each
    else:
        vac_total = float(vac_value)
        c_new = thickness + vac_total
        z_bottom_target = vac_total * 0.5
    # shift atoms so bottom sits at z_bottom_target
    shift = (z_bottom_target - zmin)
    pos_new = pos + np.outer(np.ones(len(slab)), n_hat) * shift
    slab.set_positions(pos_new)
    # set new c vector and keep a1,a2 as-is but orthogonalize to ĉ direction
    A_new = A.copy()
    A_new[2] = n_hat * c_new
    slab.set_cell(A_new, scale_atoms=False)
    return slab

# ─────────── QE input / LSF writers ───────────────────────────────────────
def write_pw_in(slab: Atoms, fn: Path, frozen_idx, kpts, kshift):
    symbols = [a.symbol for a in slab]
    species = sorted(set(symbols), key=symbols.index)
    stem = fn.stem
    L, A = [], lambda s: L.append(s)

    # &CONTROL
    A("&CONTROL")
    A(f"  title           = '{stem}',")
    A("  calculation     = 'relax',")
    A("  tstress         = .true.,")
    A("  tprnfor         = .true.,")
    A(f"  pseudo_dir      = '{PSEUDO_DIR}',")
    A(f"  prefix          = '{stem}',")
    A("  outdir          = '.',")
    A("  restart_mode    = 'from_scratch',")
    etot_thr_ry = len(slab) * 7.3498645e-7
    A(f"  etot_conv_thr   = {format(etot_thr_ry, '.6e').replace('e','d')}")
    A("  forc_conv_thr   = 7.8d-4")
    A("/\n")

    # &SYSTEM
    A("&SYSTEM")
    A("  ibrav        = 0,")
    A(f"  nat          = {len(slab)},")
    A(f"  ntyp         = {len(species)},")
    A("  nspin        = 2,")
    for i, s in enumerate(species, 1):
        A(f"  starting_magnetization({i}) = {START_MAGN.get(s, 0.0):.4f},")
    A(f"  ecutwfc      = {ECUTWFC},")
    A(f"  ecutrho      = {ECUTRHO},")
    A("  occupations  = 'smearing',")
    A("  smearing     = 'mv',")
    A("  nosym        = .true.,")
    A("  input_dft    = 'PBE',")
    A("  assume_isolated = '2D',")
    A("  noinv        = .true.,")
    A(f"  degauss      = {DEGAUSS},")
    A("  vdw_corr     = 'DFT-D3',")
    A("  dftd3_version = 6,")
    A("/\n")

    # &ELECTRONS
    A("&ELECTRONS")
    etot_thr_ry_conv = len(slab) * 7.3498645e-7 * 1e-3
    A(f"  conv_thr         = {format(etot_thr_ry_conv, '.6e').replace('e','d')}")
    A('  diagonalization  = "david"')
    A("  electron_maxstep = 512")
    A("  mixing_beta      = 0.2")
    A('  mixing_mode      = "local-TF"')
    A("  mixing_ndim      = 12")
    A("/\n")

    # &IONS
    A("&IONS")
    A("  ion_dynamics  = 'bfgs'")
    A("/\n")

    # ATOMIC_SPECIES
    A("ATOMIC_SPECIES")
    for s in species:
        mass = slab[symbols.index(s)].mass
        pseudo = PSEUDO_MAP.get(s, f"{s}.upf")
        A(f"  {s:2s} {mass:.4f} {pseudo}")
    A("")

    # ATOMIC_POSITIONS with flags
    frozen_set = set(frozen_idx)
    A("ATOMIC_POSITIONS {angstrom}")
    for a in slab:
        flag = "0 0 0" if a.index in frozen_set else "1 1 1"
        A(f"  {a.symbol:2s} {a.x: .10f} {a.y: .10f} {a.z: .10f} {flag}")
    A("")

    # CELL / K_POINTS (automatic, with shifts)
    A("CELL_PARAMETERS {angstrom}")
    for v in slab.get_cell():
        A(f"  {v[0]: .10f} {v[1]: .10f} {v[2]: .10f}")
    A("")
    kx, ky, kz = kpts
    sx, sy, sz = kshift
    A("K_POINTS {automatic}")
    A(f"  {kx} {ky} {kz}   {sx} {sy} {sz}")

    fn.write_text("\n".join(L), encoding="utf-8")
    
def write_pw_in_quick(slab: Atoms, fn: Path, frozen_idx, kpts, kshift):
    symbols = [a.symbol for a in slab]
    species = sorted(set(symbols), key=symbols.index)
    stem = fn.stem
    L, A = [], lambda s: L.append(s)

    # &CONTROL
    A("&CONTROL")
    A(f"  title           = '{stem}',")
    A("  calculation     = 'relax',")
    A("  tstress         = .true.,")
    A("  tprnfor         = .true.,")
    A(f"  pseudo_dir      = '{PSEUDO_DIR}',")
    A(f"  prefix          = '{stem}',")
    A("  outdir          = '.',")
    A("  restart_mode    = 'from_scratch',")
    etot_thr_ry = len(slab) * 7.3498645e-7
    A(f"  etot_conv_thr   = {format(etot_thr_ry, '.6e').replace('e','d')}")
    A(f"  forc_conv_thr   = {format(PRE_FORCE_THR, '.6e').replace('e','d')}")
    A("/\n")

    # &SYSTEM
    A("&SYSTEM")
    A("  ibrav        = 0,")
    A(f"  nat          = {len(slab)},")
    A(f"  ntyp         = {len(species)},")
    A("  nspin        = 2,")
    for i, s in enumerate(species, 1):
        A(f"  starting_magnetization({i}) = {START_MAGN.get(s, 0.0):.4f},")
    A(f"  ecutwfc      = {ECUTWFC},")
    A(f"  ecutrho      = {ECUTRHO},")
    A("  occupations  = 'smearing',")
    A(f"  smearing     = '{PRE_SMEARING}',")
    A(f"  degauss      = {PRE_DEGAUSS},")
    A("  nosym        = .true.,")
    A("  input_dft    = 'PBE',")
    A("  assume_isolated = '2D',")
    A("  noinv        = .true.,")
    A("  vdw_corr     = 'DFT-D3',")
    A("  dftd3_version = 6,")
    A("/\n")

    # &ELECTRONS
    A("&ELECTRONS")
    A(f"  conv_thr         = {format(PRE_CONV_THR, '.6e').replace('e','d')}")
    A('  diagonalization  = "david"')
    A(f"  electron_maxstep = {PRE_ELEC_STEPS}")
    A(f"  mixing_beta      = {PRE_MIX_BETA}")
    A('  mixing_mode      = "local-TF"')
    A("  mixing_ndim      = 12")
    A("/\n")

    # &IONS
    A("&IONS")
    A("  ion_dynamics  = 'bfgs'")
    A("/\n")

    # ATOMIC_SPECIES
    A("ATOMIC_SPECIES")
    for s in species:
        mass = slab[symbols.index(s)].mass
        pseudo = PSEUDO_MAP.get(s, f"{s}.upf")
        A(f"  {s:2s} {mass:.4f} {pseudo}")
    A("")

    # ATOMIC_POSITIONS with flags
    frozen_set = set(frozen_idx)
    A("ATOMIC_POSITIONS {angstrom}")
    for a in slab:
        flag = "0 0 0" if a.index in frozen_set else "1 1 1"
        A(f"  {a.symbol:2s} {a.x: .10f} {a.y: .10f} {a.z: .10f} {flag}")
    A("")

    # CELL / K_POINTS
    A("CELL_PARAMETERS {angstrom}")
    for v in slab.get_cell():
        A(f"  {v[0]: .10f} {v[1]: .10f} {v[2]: .10f}")
    A("")
    kx, ky, kz = kpts
    sx, sy, sz = kshift
    A("K_POINTS {automatic}")
    A(f"  {kx} {ky} {kz}   {sx} {sy} {sz}")

    fn.write_text("\n".join(L), encoding="utf-8")


# ### NEW: choose -nk, -nb, -ntg from cores & k-mesh
def choose_pool_flags(total_ranks: int, kpts: tuple):
    import math
    kx, ky, kz = kpts
    n_kpts = int(kx * ky * kz)

    # Pools: distribute k-points evenly and ensure nk | total_ranks
    nk = max(1, math.gcd(total_ranks, n_kpts))
    per_pool = total_ranks // nk

    # Conservative band groups to avoid "nyfft incompatible with nproc_bgrp":
    # prefer nb=1; allow nb=2 only if per_pool is even.
    nb = 2 if (per_pool % 2 == 0 and per_pool >= 4) else 1

    # FFT task groups: biggest safe divisor (keeps per_pool % ntg == 0)
    for cand in (8, 6, 4, 3, 2):
        if per_pool % cand == 0:
            ntg = cand
            break
    else:
        ntg = 1

    return nk, nb, ntg, n_kpts, per_pool

def write_lsf(lsf_path: Path, inp_name: str, kpts: tuple):
    job = inp_name.rsplit(".in", 1)[0]
    ranks = max(1, QUEUE_CORES // max(1, OMP_THREADS))
    nk, nb, ntg, n_kpts, per_pool = choose_pool_flags(ranks, kpts)
    lsf_path.write_text(f"""#!/bin/bash
#BSUB -J {job}
#BSUB -q {QUEUE_NAME}
#BSUB -n {QUEUE_CORES}
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -R "{RESOURCE_STRING}"
#BSUB -B
#BSUB -N
#BSUB -u ccuan@pceim.unam.mx
#

module load tbb/2021.6.0
module load compiler-rt/2022.1.0
module load intel/2022.1.0
module load mpi/intel-2021.6.0
module load mkl/2022.1.0
module load quantum/7.3

export OMP_NUM_THREADS={OMP_THREADS}
export MKL_NUM_THREADS={OMP_THREADS}
export OMP_DYNAMIC=false
export MKL_DYNAMIC=false
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=core
export I_MPI_FABRICS=shm:ofi
export I_MPI_DEBUG=0

mpirun -np {ranks} pw.x -nk {nk} -nb {nb} -ntg {ntg} -in {inp_name} > {job}.out 2>&1
""", encoding="utf-8")

# ─────────── per-file processing ───────────────────────────────────────────
def process_out(out_path: Path):
    txt = out_path.read_text(errors="ignore")
    if not END_OK_RE.search(txt):
        print(f"   · no JOB DONE, skipping: {out_path.name}")
        return

    slab_full = read(out_path, format="espresso-out", index=-1)  # relaxed 1×1
    if REPEAT[2] != 1:
        raise ValueError("This script is for in-plane supercells only (nz must be 1).")
    nx, ny, nz = REPEAT

    # Planes in the full slab (for reporting)
    z_all = (slab_full.get_positions() @ _n_hat_from_cell(slab_full.get_cell()))
    planes_all, _, tol_all = group_planes_with_tol(z_all)
    counts_all = [len(p) for p in planes_all]

    # ### NEW: trim planes and set vacuum
    slab1 = trim_keep_bottom_planes(slab_full, TARGET_PLANES)
    slab1 = set_vacuum_and_center(slab1, VACUUM_ANG, VACUUM_MODE)

    # Planes in the trimmed slab (for freezing + reporting)
    z1 = (slab1.get_positions() @ _n_hat_from_cell(slab1.get_cell()))
    planes1, _, tol1 = group_planes_with_tol(z1)
    base_counts = [len(p) for p in planes1]
    nat_1x1 = len(slab1)

    # Build supercell and map source indices
    scell, src_map = supercell_expand_crop_with_map(slab1, nx, ny)

    # Freeze bottom FROZEN_PLANES of the TRIMMED base slab
    frozen_idx = freeze_from_base_planes(slab1, scell, src_map, FROZEN_PLANES)

    # Sanity: plane counts in supercell (same tol)
    zS = (scell.get_positions() @ _n_hat_from_cell(scell.get_cell()))
    planesS, _, _ = group_planes_with_tol(zS, tol_override=tol1 * 1.05)
    sup_counts = [len(p) for p in planesS]

    expected_atoms = nat_1x1 * nx * ny
    ok_atoms = (len(scell) == expected_atoms)
    ok_planes = (len(base_counts) == len(sup_counts) and
                 all(a * nx * ny == b for a, b in zip(base_counts, sup_counts)))

    # destinations
    if OUTPUT_DIR:
        struct_root   = OUTPUT_DIR / STRUCT_SUBDIR
        relax_base    = OUTPUT_DIR / RELAX_INPUT_SUBDIR
        prerelax_base = relax_base / PRERELAX_INPUT_SUBDIR
        if PRESERVE_SUBDIRS:
            # Only preserve subdirs for STRUCTURES if you want; inputs stay flat
            rel_parent  = out_path.parent.relative_to(ROOT_DIR)
            struct_base = struct_root / rel_parent
        else:
            struct_base = struct_root
    else:
        struct_base   = out_path.parent / STRUCT_SUBDIR
        relax_base    = out_path.parent / RELAX_INPUT_SUBDIR
        prerelax_base = relax_base / PRERELAX_INPUT_SUBDIR

    # Ensure dirs exist
    struct_base.mkdir(parents=True, exist_ok=True)
    relax_base.mkdir(parents=True, exist_ok=True)
    prerelax_base.mkdir(parents=True, exist_ok=True)

    stem = f"{out_path.stem}_{REPEAT[0]}x{REPEAT[1]}x{REPEAT[2]}"
    # structures (kept where you want; flat or preserved per above)
    xyz = struct_base / f"{stem}.xyz"
    cif = struct_base / f"{stem}.cif"

    # --- write production (Relax/) ---
    inp_relax = relax_base / f"{stem}.in"
    lsf_relax = relax_base / f"{stem}.lsf"
    write(xyz, scell)
    write(cif, scell, format="cif", wrap=False)
    write_pw_in(scell, inp_relax, frozen_idx, KPOINTS, KSHIFT)
    write_lsf(lsf_relax, inp_relax.name, KPOINTS)

    # --- write pre-relax (Relax/PreRelax/) ---
    if MAKE_PRERELAX:
        inp_q = prerelax_base / f"{stem}_{PRE_SUFFIX}.in"
        lsf_q = prerelax_base / f"{stem}_{PRE_SUFFIX}.lsf"
        write_pw_in_quick(scell, inp_q, frozen_idx, PRE_KPOINTS, PRE_KSHIFT)
        write_lsf(lsf_q, inp_q.name, PRE_KPOINTS)

    # pretty print
    shown_struct = struct_base if OUTPUT_DIR is None else struct_base.relative_to(OUTPUT_DIR)
    shown_relax  = relax_base  if OUTPUT_DIR is None else relax_base.relative_to(OUTPUT_DIR)
    shown_pre    = prerelax_base if OUTPUT_DIR is None else prerelax_base.relative_to(OUTPUT_DIR)
    print(f"   √ {shown_struct}  |  Relax→ {shown_relax}  |  PreRelax→ {shown_pre}  (atoms: {len(scell)}, frozen: {len(frozen_idx)})")
    print(f"     Planes(full→trimmed): {len(counts_all)}→{len(base_counts)}   counts(full)={format_plane_counts(counts_all)}  trimmed={format_plane_counts(base_counts)}")
    tag1 = "OK" if ok_planes else "[! plane mismatch]"
    tag2 = "OK" if ok_atoms  else f"[! expected {expected_atoms}]"
    print(f"     Supercell planes: {format_plane_counts(sup_counts)} {tag1}")
    print(f"     Atom count: {len(scell)} {tag2}")

# ─────────── main ─────────────────────────────────────────────────────────
def main():
    outs = list(ROOT_DIR.rglob("*.out"))
    print(f"Scanning {len(outs)} *.out in {ROOT_DIR}")

    winners = {}
    skipped = 0

    for f in outs:
        txt = f.read_text(errors="ignore")
        if not END_OK_RE.search(txt):
            skipped += 1; continue
        e_ry = extract_final_energy_ry(txt)
        if e_ry is None:
            skipped += 1; continue
        key = system_key(f)
        if key not in winners or e_ry < winners[key][1]:
            winners[key] = (f, e_ry)

    print(f"Found {len(winners)} systems with a winner (skipped: {skipped}).")
    for (parent, base), (fbest, e) in winners.items():
        rel = fbest.relative_to(ROOT_DIR)
        print(f" → [{base}] winner: {rel.name}  E_final = {e:.6f} Ry")

    for (_, _), (fbest, _) in winners.items():
        print("Processing:", fbest.relative_to(ROOT_DIR))
        process_out(fbest)

    print(f"\nDone ({date.today()})")

if __name__ == "__main__":
    main()
