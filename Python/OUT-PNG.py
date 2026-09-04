from pathlib import Path
import re
import itertools
import math
import shutil
import traceback
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

import pyvista as pv
from ase import Atoms
from ase.io import read
from ase.data import covalent_radii, atomic_numbers


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Windows acepta forward slashes; evita problemas de escapes.
INPUT_DIR = "G:/My Drive/Work/UNAM/Doctorado/Proyecto/Resultados/Nanoparticles/QE/Supercell/Hydrogen/111_2x2-H/RELAX"
OUTPUT_DIR = "G:/My Drive/Work/UNAM/Doctorado/Proyecto/Resultados/Nanoparticles/QE/Supercell/Hydrogen/111_2x2-H/PNG"

# Para probar rápido:
# ONLY_CASES = {"Pd_111_H2-relax.out"}
# ONLY_CASES = {"Pd_111_B1-relax.out", "Ni_111_H1-relax.out"}
ONLY_CASES = set()

MODE = "auto"
TARGET_SPECIES = {"H"}
METALS = {"Ni", "Pd"}

BEGIN_FINAL = "Begin final coordinates"
END_FINAL = "End final coordinates"

IMAGE_SIZE = (1920, 1080)
ZOOM_SIZE = (900, 700)
COMPOSITE_SIZE = (2400, 1350)

COMPOSITE_TRANSPARENT = True
PYVISTA_BG = "black"
TRANSPARENT_BACKGROUND = True

KEEP_INTERMEDIATE_PNGS = False
STOP_ON_FIRST_FAILURE = True


# ============================================================
# ESTILO VISUAL
# ============================================================

ATOM_BASE_COLORS = {
    "Ni": "#9a9a9d",   # gris metálico
    "Pd": "#167f7a",   # verde-turquesa oscuro
    "H":  "#f2b6c6",   # rosa pálido
}
H_INTERACTION_COLOR = "#e7a4b6"

CELL_LINE_COLOR = "#4f6664"
CELL_LINE_WIDTH_MAIN = 0.75

BALL_STICK_SCALE = 0.36
MIN_RADIUS = 0.14
MAX_RADIUS = 0.42
H_RADIUS_OVERRIDE = 0.23
H_RADIUS_SIDE_BOOST = 1.18
ZOOM_H_RADIUS_BOOST = 1.20

# Profundidad suave: evita átomos casi negros.
TOP_LAYER_FACTORS = [1.18, 0.98, 0.88, 0.80, 0.74, 0.70]
LAYER_TOL = 0.75
TOP_BOND_MAX_LAYER = 2
ZOOM_BOND_MAX_LAYER = 2

# Enlaces metal-metal más gruesos.
# Se dibujan como líneas tubulares rápidas; no se usa pv.Cylinder por cada enlace.
BOND_LINE_WIDTH = 10

# Interacciones H-metal: 3D, segmentadas, no enlace completo.
H_INTERACTION_LINE_WIDTH = 8
H_DASHED_INTERACTIONS = True
H_INTERACTION_ALL_LOCAL = True
H_INTERACTION_SHELL_EXTRA = 0.55
H_INTERACTION_MAX_COUNT = 6
H_MAX_DIST_ABSOLUTE = 2.35
SLASH_SEG_LEN = 0.16
SLASH_GAP = 0.11

METAL_BOND_PADDING = 0.30
METAL_VISIBLE_PAIR_CUTOFF_SCALE = 1.04
DEFAULT_H_NEIGHBORS = 3


# ============================================================
# CONTROL DE CÁMARA / PANELES
# ============================================================

# Top view: más área visible para que los átomos queden comparables con side view.
TOP_MAIN_BOUNDS = (-0.85, 1.85, -0.85, 1.85, -0.08, 1.08)
TOP_MARGIN = 3.85

# Side view: más largo, estilo 1.png.
SIDE_PERIODIC_RANGE = 1
SIDE_X_HALF_WINDOW = 6.20
SIDE_X_RENDER_EXTRA = 0.85
SIDE_UP_EXTRA = 1.20
SIDE_DOWN_WINDOW = 4.15
SIDE_MARGIN = 0.94
SIDE_TOP_FRACTION = 0.76

# Zooms reales: no miniaturas.
ZOOM_TOP_FRAC_HALF_WIDTH = 0.34
ZOOM_SIDE_X_HALF_WINDOW = 1.75
ZOOM_SIDE_UP_EXTRA = 0.95
ZOOM_SIDE_DOWN_WINDOW = 1.65
ZOOM_TOP_PARALLEL_SCALE = 1.45
ZOOM_SIDE_PARALLEL_SCALE = 1.20
ZOOM_COMPLETION_SHELLS = 1


# ============================================================
# COMPOSICIÓN FINAL
# ============================================================

BRACKET_COLOR = "black"
BRACKET_WIDTH = 8
BRACKET_LEN = 70

ARROW_COLOR = "black"
ARROW_OUTLINE_COLOR = None
ARROW_WIDTH = 10
ARROW_OUTLINE_WIDTH = 14
ARROW_HEAD = 34

ZOOM_BORDER_COLOR = "black"
ZOOM_BORDER_WIDTH = 6


# ============================================================
# LECTURA QE / ASE
# ============================================================

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def has_complete_final_block(text: str) -> bool:
    return BEGIN_FINAL in text and END_FINAL in text and text.index(BEGIN_FINAL) < text.index(END_FINAL)


def extract_assume_isolated(input_text: str):
    match = re.search(
        r"assume_isolated\s*=\s*['\"]?([A-Za-z0-9_\-]+)['\"]?",
        input_text,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip().lower() if match else None


def infer_mode(input_text: str, requested_mode: str) -> str:
    if requested_mode in {"2d", "3d"}:
        return requested_mode
    assume_isolated = extract_assume_isolated(input_text)
    if assume_isolated is not None and "2d" in assume_isolated:
        return "2d"
    return "3d"


def periodic_axes_from_mode(mode: str):
    return (0, 1) if mode == "2d" else (0, 1, 2)


def set_pbc_from_mode(atoms: Atoms, mode: str):
    atoms.pbc = (True, True, False) if mode == "2d" else (True, True, True)


def read_final_atoms_from_out(out_file: Path) -> Atoms:
    text = read_text(out_file)
    if not has_complete_final_block(text):
        raise RuntimeError("No contiene bloque final completo.")

    try:
        atoms = read(str(out_file), format="espresso-out", index=-1)
        if isinstance(atoms, list):
            atoms = atoms[-1]
        if not isinstance(atoms, Atoms) or len(atoms) == 0:
            raise RuntimeError("ASE devolvió estructura vacía.")
        return atoms
    except Exception:
        frames = list(read(str(out_file), format="espresso-out", index=":"))
        valid = [f for f in frames if isinstance(f, Atoms) and len(f) > 0]
        if not valid:
            raise RuntimeError("ASE no encontró frames válidos.")
        return valid[-1]


def read_input_atoms(in_file: Path) -> Atoms:
    atoms = read(str(in_file), format="espresso-in")
    if isinstance(atoms, list):
        atoms = atoms[-1]
    if not isinstance(atoms, Atoms) or len(atoms) == 0:
        raise RuntimeError("No se pudo leer la estructura del .in.")
    return atoms


def ensure_valid_cell(final_atoms: Atoms, input_atoms: Atoms) -> Atoms:
    out_cell = np.array(final_atoms.cell)
    out_lengths = np.linalg.norm(out_cell, axis=1)
    if np.all(out_lengths > 1e-12):
        return final_atoms

    fixed = final_atoms.copy()
    fixed.set_cell(input_atoms.cell, scale_atoms=False)
    fixed.pbc = input_atoms.pbc
    return fixed


# ============================================================
# GEOMETRÍA
# ============================================================

def normalize(vec) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    n = np.linalg.norm(vec)
    if n < 1e-15:
        return vec
    return vec / n


def project_onto_plane(vec, normal):
    normal = normalize(normal)
    return np.asarray(vec, dtype=float) - np.dot(vec, normal) * normal


def reciprocal_vectors(cell):
    a = np.array(cell[0], dtype=float)
    b = np.array(cell[1], dtype=float)
    c = np.array(cell[2], dtype=float)
    vol = np.dot(a, np.cross(b, c))
    if abs(vol) < 1e-15:
        raise RuntimeError("Celda singular.")
    return np.cross(b, c) / vol, np.cross(c, a) / vol, np.cross(a, b) / vol


def periodic_mean(frac_values: np.ndarray) -> float:
    vals = np.mod(np.asarray(frac_values, dtype=float), 1.0)
    angles = 2.0 * np.pi * vals
    s = np.mean(np.sin(angles))
    c = np.mean(np.cos(angles))
    if abs(s) < 1e-15 and abs(c) < 1e-15:
        return float(vals[0])
    return float((np.arctan2(s, c) / (2.0 * np.pi)) % 1.0)


def get_base_scaled_positions(atoms: Atoms, periodic_axes):
    scaled = atoms.get_scaled_positions(wrap=False).copy()
    for ax in periodic_axes:
        scaled[:, ax] = np.mod(scaled[:, ax], 1.0)
    return scaled


def center_h_in_cell(atoms: Atoms):
    symbols = atoms.get_chemical_symbols()
    h_indices = [i for i, s in enumerate(symbols) if s in TARGET_SPECIES]
    if not h_indices:
        return atoms.copy(), (0.0, 0.0)

    scaled = atoms.get_scaled_positions(wrap=False).copy()
    hx = periodic_mean(scaled[h_indices, 0])
    hy = periodic_mean(scaled[h_indices, 1])

    sx = (0.5 - hx) % 1.0
    sy = (0.5 - hy) % 1.0

    shifted = atoms.copy()
    scaled[:, 0] = np.mod(scaled[:, 0] + sx, 1.0)
    scaled[:, 1] = np.mod(scaled[:, 1] + sy, 1.0)
    shifted.set_scaled_positions(scaled)
    return shifted, (sx, sy)


def atom_radius(symbol: str, view_kind: str = "main") -> float:
    if symbol == "H":
        r = H_RADIUS_OVERRIDE
        if view_kind in {"side", "side_zoom"}:
            r *= H_RADIUS_SIDE_BOOST
        if "zoom" in str(view_kind):
            r *= ZOOM_H_RADIUS_BOOST
        return r

    z = atomic_numbers.get(symbol, 0)
    if z > 0:
        r = covalent_radii[z] * BALL_STICK_SCALE
        return float(np.clip(r, MIN_RADIUS, MAX_RADIUS))
    return 0.20


def hex_to_rgb01(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return np.array(
        [int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)],
        dtype=float,
    ) / 255.0


def rgb01_to_hex(rgb):
    rgb = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    vals = (255.0 * rgb).round().astype(int)
    return "#{:02x}{:02x}{:02x}".format(*vals.tolist())


def shade_color(hex_color: str, factor: float) -> str:
    rgb = hex_to_rgb01(hex_color)
    if factor >= 1.0:
        rgb = 1.0 - (1.0 - rgb) / factor
    else:
        rgb = rgb * factor
    return rgb01_to_hex(rgb)


def blend_colors(c1: str, c2: str, t: float) -> str:
    rgb1 = hex_to_rgb01(c1)
    rgb2 = hex_to_rgb01(c2)
    return rgb01_to_hex((1.0 - t) * rgb1 + t * rgb2)


def assign_layers(atoms: Atoms, exclude_species={"H"}):
    symbols = atoms.get_chemical_symbols()
    z = atoms.get_positions()[:, 2]
    valid_idx = [i for i, s in enumerate(symbols) if s not in exclude_species]
    if not valid_idx:
        return {}

    order = sorted(valid_idx, key=lambda i: z[i], reverse=True)
    groups = []
    current = [order[0]]

    for idx in order[1:]:
        ref = np.mean([z[j] for j in current])
        if abs(z[idx] - ref) <= LAYER_TOL:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)

    layer_map = {}
    for lid, group in enumerate(groups):
        for idx in group:
            layer_map[idx] = lid
    return layer_map


def atom_base_color(symbol: str, atom_index: int, layer_map, use_layer_shading=True) -> str:
    base = ATOM_BASE_COLORS.get(symbol, "#b4b4b4")
    if symbol == "H" or not use_layer_shading:
        return base
    layer = layer_map.get(atom_index, len(TOP_LAYER_FACTORS) - 1)
    factor = TOP_LAYER_FACTORS[min(layer, len(TOP_LAYER_FACTORS) - 1)]
    return shade_color(base, factor)


def camera_vectors(plotter: pv.Plotter):
    cpos, focal, up = plotter.camera_position
    cpos = np.array(cpos, dtype=float)
    focal = np.array(focal, dtype=float)
    up = normalize(np.array(up, dtype=float))
    view = normalize(focal - cpos)
    right = normalize(np.cross(view, up))
    up = normalize(np.cross(right, view))
    return cpos, focal, view, right, up


def depth_field(points: np.ndarray, plotter: pv.Plotter):
    cpos, _, view, _, _ = camera_vectors(plotter)
    d = np.array([np.dot(p - cpos, view) for p in points], dtype=float)
    dmin, dmax = float(d.min()), float(d.max())
    if abs(dmax - dmin) < 1e-12:
        dmax = dmin + 1.0
    return d, dmin, dmax


def depth_factor(point: np.ndarray, plotter: pv.Plotter, dmin: float, dmax: float, near_gain=1.14, far_gain=0.58):
    cpos, _, view, _, _ = camera_vectors(plotter)
    d = float(np.dot(np.asarray(point) - cpos, view))
    t = (d - dmin) / (dmax - dmin)
    return near_gain * (1.0 - t) + far_gain * t


# ============================================================
# REGISTROS PERIÓDICOS
# ============================================================

def shift_range_for_bounds(bounds, periodic_axes):
    ranges = []
    for ax in range(3):
        if ax in periodic_axes:
            lo = int(np.floor(bounds[2 * ax]))
            hi = int(np.ceil(bounds[2 * ax + 1]))
            ranges.append(range(lo, hi + 1))
        else:
            ranges.append(range(0, 1))
    return ranges


def in_bounds(frac, bounds):
    return (
        bounds[0] <= frac[0] <= bounds[1]
        and bounds[2] <= frac[1] <= bounds[3]
        and bounds[4] <= frac[2] <= bounds[5]
    )


def build_records_in_bounds(atoms: Atoms, bounds, periodic_axes):
    cell = np.array(atoms.cell)
    scaled0 = get_base_scaled_positions(atoms, periodic_axes)
    ranges = shift_range_for_bounds(bounds, periodic_axes)

    records = []
    for i in range(len(atoms)):
        for shift in itertools.product(*ranges):
            frac = scaled0[i] + np.array(shift, dtype=float)
            if not in_bounds(frac, bounds):
                continue
            records.append(
                {
                    "index": i,
                    "shift": tuple(int(v) for v in shift),
                    "frac": frac,
                    "cart": frac @ cell,
                    "symbol": atoms[i].symbol,
                }
            )
    return records


def build_full_records(atoms: Atoms, periodic_axes, nrange=1):
    cell = np.array(atoms.cell)
    scaled0 = get_base_scaled_positions(atoms, periodic_axes)
    ranges = [range(-nrange, nrange + 1) if ax in periodic_axes else range(0, 1) for ax in range(3)]

    records = []
    for i in range(len(atoms)):
        for shift in itertools.product(*ranges):
            frac = scaled0[i] + np.array(shift, dtype=float)
            records.append(
                {
                    "index": i,
                    "shift": tuple(int(v) for v in shift),
                    "frac": frac,
                    "cart": frac @ cell,
                    "symbol": atoms[i].symbol,
                }
            )
    return records


def record_key(rec):
    return (rec["index"], rec["shift"])


def select_main_h_record(records):
    h_records = [r for r in records if r["symbol"] in TARGET_SPECIES]
    if not h_records:
        return None
    return min(h_records, key=lambda rec: (rec["frac"][0] - 0.5) ** 2 + (rec["frac"][1] - 0.5) ** 2)


def filter_only_main_h(records, main_h):
    """Conserva sólo el H central; elimina copias periódicas de H.

    Esto evita que aparezcan H rosados repetidos en bordes y que se dibujen
    interacciones H-metal falsas desde imágenes periódicas.
    """
    if main_h is None:
        return [r for r in records if r["symbol"] not in TARGET_SPECIES]
    main_key = record_key(main_h)
    out = [r for r in records if r["symbol"] not in TARGET_SPECIES]
    if main_key not in {record_key(r) for r in out}:
        out.append(main_h)
    return out


def infer_site_neighbor_count(name: str) -> int:
    s = Path(name).stem.upper()
    if re.search(r"(^|[_\-])T\d*($|[_\-])", s):
        return 1
    if re.search(r"(^|[_\-])B\d*($|[_\-])", s):
        return 2
    if "H_FCC" in s or "H_HCP" in s:
        return 3
    if re.search(r"(^|[_\-])H\d*($|[_\-])", s):
        return 3
    return DEFAULT_H_NEIGHBORS


def metal_pair_cutoffs(atoms: Atoms, periodic_axes):
    symbols = atoms.get_chemical_symbols()
    mins = {}
    for i in range(len(atoms)):
        if symbols[i] not in METALS:
            continue
        for j in range(i + 1, len(atoms)):
            if symbols[j] not in METALS:
                continue
            key = tuple(sorted((symbols[i], symbols[j])))
            d = atoms.get_distance(i, j, mic=True)
            if d > 1e-10:
                mins[key] = min(mins.get(key, 1e9), float(d))
    return {k: v + METAL_BOND_PADDING for k, v in mins.items()}


def select_side_records(records, main_h, right, up):
    keep = []
    main_key = record_key(main_h)
    for rec in records:
        if rec["symbol"] == "H":
            if record_key(rec) == main_key:
                keep.append(rec)
            continue

        rel = rec["cart"] - main_h["cart"]
        x = np.dot(rel, right)
        y = np.dot(rel, up)

        if abs(x) <= SIDE_X_HALF_WINDOW + SIDE_X_RENDER_EXTRA and -SIDE_DOWN_WINDOW <= y <= SIDE_UP_EXTRA:
            keep.append(rec)

    if main_key not in {record_key(r) for r in keep}:
        keep.append(main_h)
    return keep


def select_side_zoom_records(records, main_h, right, up):
    keep = []
    main_key = record_key(main_h)
    for rec in records:
        if rec["symbol"] == "H":
            if record_key(rec) == main_key:
                keep.append(rec)
            continue

        rel = rec["cart"] - main_h["cart"]
        x = np.dot(rel, right)
        y = np.dot(rel, up)

        if abs(x) <= ZOOM_SIDE_X_HALF_WINDOW and -ZOOM_SIDE_DOWN_WINDOW <= y <= ZOOM_SIDE_UP_EXTRA:
            keep.append(rec)

    if main_key not in {record_key(r) for r in keep}:
        keep.append(main_h)
    return keep


def complete_h_neighbors(records, all_records, main_h, h_neighbor_count):
    keys = {record_key(r) for r in records}
    out = list(records)
    pool = [r for r in all_records if r["symbol"] in METALS]

    candidates = []
    for rec in pool:
        d = float(np.linalg.norm(rec["cart"] - main_h["cart"]))
        if 1e-8 < d <= H_MAX_DIST_ABSOLUTE:
            candidates.append((d, rec))
    candidates.sort(key=lambda x: x[0])

    if not candidates:
        return out

    if H_INTERACTION_ALL_LOCAL:
        dmin_h = candidates[0][0]
        cutoff_h = min(dmin_h + H_INTERACTION_SHELL_EXTRA, H_MAX_DIST_ABSOLUTE)
        chosen = [(d, rec) for d, rec in candidates if d <= cutoff_h][:H_INTERACTION_MAX_COUNT]

        if len(chosen) < h_neighbor_count:
            for item in candidates:
                if item not in chosen and item[0] <= H_MAX_DIST_ABSOLUTE:
                    chosen.append(item)
                if len(chosen) >= h_neighbor_count:
                    break
    else:
        chosen = candidates[:max(1, h_neighbor_count)]

    for _, rec in chosen:
        if record_key(rec) not in keys:
            out.append(rec)
            keys.add(record_key(rec))
    return out


def complete_metal_neighbors(atoms, records, all_records, periodic_axes, shells=ZOOM_COMPLETION_SHELLS):
    cutoffs = {k: v * METAL_VISIBLE_PAIR_CUTOFF_SCALE for k, v in metal_pair_cutoffs(atoms, periodic_axes).items()}
    out = list(records)
    keys = {record_key(r) for r in out}
    pool = [r for r in all_records if r["symbol"] in METALS]

    for _ in range(max(0, int(shells))):
        added = []
        current = [r for r in out if r["symbol"] in METALS]

        for a in current:
            for b in pool:
                if a["index"] == b["index"] or record_key(b) in keys:
                    continue
                key = tuple(sorted((a["symbol"], b["symbol"])))
                cutoff = cutoffs.get(key)
                if cutoff is None:
                    continue
                d = float(np.linalg.norm(b["cart"] - a["cart"]))
                if 1e-8 < d <= cutoff:
                    added.append(b)
                    keys.add(record_key(b))

        if not added:
            break
        out.extend(added)

    return out


# ============================================================
# DIBUJO
# ============================================================

def add_line_tube(plotter, p1, p2, color, line_width):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    if np.linalg.norm(p2 - p1) <= 1e-8:
        return

    line = pv.Line(p1, p2, resolution=1)
    try:
        plotter.add_mesh(line, color=color, line_width=line_width, render_lines_as_tubes=True)
    except TypeError:
        plotter.add_mesh(line, color=color, line_width=line_width)


def bond_endpoints(p1, p2, sym1, sym2, view_kind="main"):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    vec = p2 - p1
    dist = np.linalg.norm(vec)
    if dist <= 1e-10:
        return None, None

    u = vec / dist
    r1 = atom_radius(sym1, view_kind)
    r2 = atom_radius(sym2, view_kind)

    if "H" in (sym1, sym2):
        s1 = 0.10 * r1 if sym1 == "H" else 0.18 * r1
        s2 = 0.10 * r2 if sym2 == "H" else 0.18 * r2
    else:
        s1 = 0.48 * r1
        s2 = 0.48 * r2

    q1 = p1 + u * s1
    q2 = p2 - u * s2

    if np.linalg.norm(q2 - q1) <= 0.03:
        return None, None
    return q1, q2


def draw_bond_or_interaction(plotter, p1, p2, color1, color2, dmin, dmax, view_kind, is_h=False):
    is_zoom = "zoom" in str(view_kind)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    length = float(np.linalg.norm(p2 - p1))
    if length <= 1e-8:
        return

    if is_h:
        near_gain, far_gain = (1.16, 0.76) if is_zoom else (1.10, 0.84)
        line_width = H_INTERACTION_LINE_WIDTH
    else:
        near_gain, far_gain = (1.14, 0.58) if is_zoom else (1.10, 0.62)
        line_width = BOND_LINE_WIDTH

    if is_h and H_DASHED_INTERACTIONS:
        starts = np.arange(0.0, length, SLASH_SEG_LEN + SLASH_GAP)
        for s in starts:
            e = min(s + SLASH_SEG_LEN, length)
            if e - s <= 0.025:
                continue

            q1 = p1 + (s / length) * (p2 - p1)
            q2 = p1 + (e / length) * (p2 - p1)
            tm = 0.5 * (s + e) / length

            base = blend_colors(color1, color2, tm)
            mid = 0.5 * (q1 + q2)
            factor = depth_factor(mid, plotter, dmin, dmax, near_gain=near_gain, far_gain=far_gain)
            add_line_tube(plotter, q1, q2, shade_color(base, factor), line_width)
        return

    mid = 0.5 * (p1 + p2)
    base = blend_colors(color1, color2, 0.5)
    factor = depth_factor(mid, plotter, dmin, dmax, near_gain=near_gain, far_gain=far_gain)
    add_line_tube(plotter, p1, p2, shade_color(base, factor), line_width)


def add_bonds(plotter, atoms, records, periodic_axes, layer_map, h_neighbor_count, view_kind):
    if not records:
        return

    pts = np.array([r["cart"] for r in records], dtype=float)
    _, dmin, dmax = depth_field(pts, plotter)

    cutoffs = {k: v * METAL_VISIBLE_PAIR_CUTOFF_SCALE for k, v in metal_pair_cutoffs(atoms, periodic_axes).items()}
    metals = [r for r in records if r["symbol"] in METALS]

    # Metal-metal
    drawn = set()
    for ia in range(len(metals)):
        a = metals[ia]
        for ib in range(ia + 1, len(metals)):
            b = metals[ib]
            if a["index"] == b["index"]:
                continue

            if "top" in str(view_kind):
                max_layer = max(layer_map.get(a["index"], 99), layer_map.get(b["index"], 99))
                allowed_layer = ZOOM_BOND_MAX_LAYER if "zoom" in str(view_kind) else TOP_BOND_MAX_LAYER
                if max_layer > allowed_layer:
                    continue

            key = tuple(sorted((a["symbol"], b["symbol"])))
            cutoff = cutoffs.get(key)
            if cutoff is None:
                continue

            d = float(np.linalg.norm(b["cart"] - a["cart"]))
            if not (1e-8 < d <= cutoff):
                continue

            pkey = tuple(sorted((record_key(a), record_key(b))))
            if pkey in drawn:
                continue
            drawn.add(pkey)

            q1, q2 = bond_endpoints(a["cart"], b["cart"], a["symbol"], b["symbol"], view_kind)
            if q1 is None:
                continue

            c1 = atom_base_color(a["symbol"], a["index"], layer_map)
            c2 = atom_base_color(b["symbol"], b["index"], layer_map)
            draw_bond_or_interaction(plotter, q1, q2, c1, c2, dmin, dmax, view_kind, is_h=False)

    # H-metal como interacción segmentada local.
    for h in [r for r in records if r["symbol"] in TARGET_SPECIES]:
        candidates = []
        for m in metals:
            d = float(np.linalg.norm(m["cart"] - h["cart"]))
            if 1e-8 < d <= H_MAX_DIST_ABSOLUTE:
                candidates.append((d, m))
        candidates.sort(key=lambda x: x[0])

        if not candidates:
            continue

        if H_INTERACTION_ALL_LOCAL:
            dmin_h = candidates[0][0]
            cutoff_h = min(dmin_h + H_INTERACTION_SHELL_EXTRA, H_MAX_DIST_ABSOLUTE)
            chosen = [(d, m) for d, m in candidates if d <= cutoff_h][:H_INTERACTION_MAX_COUNT]

            if len(chosen) < h_neighbor_count:
                for item in candidates:
                    if item not in chosen and item[0] <= H_MAX_DIST_ABSOLUTE:
                        chosen.append(item)
                    if len(chosen) >= h_neighbor_count:
                        break
        else:
            chosen = candidates[:max(1, h_neighbor_count)]

        for _, m in chosen:
            q1, q2 = bond_endpoints(h["cart"], m["cart"], h["symbol"], m["symbol"], view_kind)
            if q1 is None:
                continue
            draw_bond_or_interaction(
                plotter,
                q1,
                q2,
                H_INTERACTION_COLOR,
                H_INTERACTION_COLOR,
                dmin,
                dmax,
                view_kind,
                is_h=True,
            )


def add_atoms(plotter, records, layer_map, view_kind="main"):
    if not records:
        return

    pts = np.array([r["cart"] for r in records], dtype=float)
    _, dmin, dmax = depth_field(pts, plotter)
    is_zoom = "zoom" in str(view_kind)

    for rec in records:
        sym = rec["symbol"]

        if sym == "H":
            base = ATOM_BASE_COLORS["H"]
            near_gain, far_gain = (1.18, 0.78) if is_zoom else (1.10, 0.86)
            material = dict(ambient=0.08, diffuse=0.86, specular=0.58, specular_power=58)
        else:
            base = atom_base_color(sym, rec["index"], layer_map, use_layer_shading=True)
            near_gain, far_gain = (1.16, 0.55) if is_zoom else (1.14, 0.58)
            if sym == "Ni":
                material = dict(ambient=0.06, diffuse=0.78, specular=0.82, specular_power=78)
            elif sym == "Pd":
                material = dict(ambient=0.06, diffuse=0.80, specular=0.74, specular_power=72)
            else:
                material = dict(ambient=0.06, diffuse=0.82, specular=0.52, specular_power=52)

        factor = depth_factor(rec["cart"], plotter, dmin, dmax, near_gain=near_gain, far_gain=far_gain)
        sphere = pv.Sphere(
            radius=atom_radius(sym, view_kind),
            center=rec["cart"],
            theta_resolution=36,
            phi_resolution=36,
        )
        plotter.add_mesh(sphere, color=shade_color(base, factor), smooth_shading=True, **material)


def add_shifted_cell_box(plotter, atoms, shift_xy=(0, 0), line_width=CELL_LINE_WIDTH_MAIN):
    cell = np.array(atoms.cell)
    shift = np.array([shift_xy[0], shift_xy[1], 0.0], dtype=float)

    corners = np.array(list(itertools.product([0.0, 1.0], repeat=3)), dtype=float)
    pts = (corners + shift) @ cell

    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]

    for i, j in edges:
        add_line_tube(plotter, pts[i], pts[j], CELL_LINE_COLOR, line_width)


def add_side_cell_guides(plotter, records, right, up, line_width=CELL_LINE_WIDTH_MAIN):
    if not records:
        return

    pts = np.array([r["cart"] for r in records], dtype=float)
    center = pts.mean(axis=0)
    rel = pts - center

    xs = rel @ right
    ys = rel @ up

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    pad_y = 0.06 * max(ymax - ymin, 1.0)

    for x in (xmin, xmax):
        p1 = center + x * right + (ymin - pad_y) * up
        p2 = center + x * right + (ymax + pad_y) * up
        add_line_tube(plotter, p1, p2, CELL_LINE_COLOR, line_width)


# ============================================================
# CÁMARAS / CONTEXTOS
# ============================================================

def set_parallel_camera(plotter, focus, cam_dir, up_dir, half_height):
    focus = np.asarray(focus, dtype=float)
    cam_dir = normalize(cam_dir)
    up_dir = normalize(up_dir)

    cam_pos = focus + cam_dir * (8.0 * max(float(half_height), 1.0))
    plotter.camera_position = [tuple(cam_pos), tuple(focus), tuple(up_dir)]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = max(float(half_height), 0.25)


def project_world_to_pixel(plotter, point, image_size):
    _, focal, _, right, up = camera_vectors(plotter)

    rel = np.asarray(point, dtype=float) - focal
    half_h = float(plotter.camera.parallel_scale)
    half_w = half_h * image_size[0] / image_size[1]

    x = float(np.dot(rel, right))
    y = float(np.dot(rel, up))

    return (
        (x + half_w) / (2.0 * half_w) * image_size[0],
        (half_h - y) / (2.0 * half_h) * image_size[1],
    )


def fit_parallel_from_records(records, focus_guess, right, up, image_size, margin=1.05, top_fraction=None):
    right = normalize(right)
    up = normalize(up)
    focus_guess = np.asarray(focus_guess, dtype=float)

    xmin, xmax = 1e30, -1e30
    ymin, ymax = 1e30, -1e30

    for rec in records:
        r = atom_radius(rec["symbol"])
        rel = rec["cart"] - focus_guess
        x = np.dot(rel, right)
        y = np.dot(rel, up)

        xmin = min(xmin, x - r)
        xmax = max(xmax, x + r)
        ymin = min(ymin, y - r)
        ymax = max(ymax, y + r)

    aspect = image_size[0] / image_size[1]
    half_h_from_w = 0.5 * (xmax - xmin) / aspect
    shift_x = 0.5 * (xmin + xmax)

    if top_fraction is None:
        shift_y = 0.5 * (ymin + ymax)
        half_h = max(half_h_from_w, 0.5 * (ymax - ymin)) * margin
    else:
        half_h = max(half_h_from_w, (ymax - ymin) / (1.0 + top_fraction)) * margin
        shift_y = ymax - top_fraction * half_h

    focus = focus_guess + shift_x * right + shift_y * up
    return focus, half_h


def pca_in_plane(points, normal, fallback):
    normal = normalize(normal)
    fallback = normalize(project_onto_plane(fallback, normal))
    if np.linalg.norm(fallback) < 1e-12:
        fallback = np.array([1.0, 0.0, 0.0])

    center = np.mean(points, axis=0)
    proj = points - center
    proj = proj - np.outer(proj @ normal, normal)

    if len(proj) < 2 or np.linalg.norm(proj) < 1e-12:
        long_axis = fallback
    else:
        vals, vecs = np.linalg.eigh(proj.T @ proj)
        long_axis = normalize(project_onto_plane(vecs[:, np.argmax(vals)], normal))
        if np.linalg.norm(long_axis) < 1e-12:
            long_axis = fallback

    if np.dot(long_axis, fallback) < 0:
        long_axis = -long_axis

    short_axis = normalize(np.cross(normal, long_axis))
    return long_axis, short_axis


def rotate_basis_to_diagonal(long_axis, short_axis, aspect):
    phi = np.arctan2(1.0, aspect)
    right = normalize(np.cos(phi) * long_axis - np.sin(phi) * short_axis)
    up = normalize(np.sin(phi) * long_axis + np.cos(phi) * short_axis)
    return right, up


def get_top_context(atoms, periodic_axes, layer_map):
    records = build_records_in_bounds(atoms, TOP_MAIN_BOUNDS, periodic_axes)
    main_h = select_main_h_record(records)

    if main_h is None:
        wide = build_records_in_bounds(atoms, (-1.8, 2.8, -1.8, 2.8, -0.1, 1.1), periodic_axes)
        main_h = select_main_h_record(wide)
        if main_h is None:
            raise RuntimeError("No se encontró H para top view.")
        records.append(main_h)

    records = filter_only_main_h(records, main_h)

    cell = np.array(atoms.cell)
    a, b, c = normalize(cell[0]), normalize(cell[1]), normalize(cell[2])

    slab_center = np.mean(atoms.get_positions(), axis=0)
    cam_dir = c.copy()
    if np.dot(main_h["cart"] - slab_center, cam_dir) < 0:
        cam_dir = -cam_dir

    pts = np.array([r["cart"] for r in records], dtype=float)
    d1 = project_onto_plane(a + b, cam_dir)
    d2 = project_onto_plane(a - b, cam_dir)
    fallback = d1 if np.linalg.norm(d1) >= np.linalg.norm(d2) else d2

    long_axis, short_axis = pca_in_plane(pts, cam_dir, fallback)
    right, up = rotate_basis_to_diagonal(long_axis, short_axis, IMAGE_SIZE[0] / IMAGE_SIZE[1])

    return {
        "records": records,
        "main_h": main_h,
        "cam_dir": cam_dir,
        "right": right,
        "up": up,
        "a": a,
        "b": b,
        "c": c,
    }


def get_side_context(atoms, periodic_axes, layer_map):
    all_records = build_full_records(atoms, periodic_axes, SIDE_PERIODIC_RANGE)
    main_h = select_main_h_record(all_records)
    if main_h is None:
        raise RuntimeError("No se encontró H para side view.")

    cell = np.array(atoms.cell)
    a, b, c = normalize(cell[0]), normalize(cell[1]), normalize(cell[2])

    _, bstar, _ = reciprocal_vectors(cell)
    cam_dir = normalize(bstar)

    slab_center = np.mean(atoms.get_positions(), axis=0)
    desired_up = c.copy()
    if np.dot(main_h["cart"] - slab_center, desired_up) < 0:
        desired_up = -desired_up

    right = normalize(project_onto_plane(a, cam_dir))
    if np.linalg.norm(right) < 1e-12:
        right = normalize(project_onto_plane(c, cam_dir))

    up = normalize(np.cross(cam_dir, right))
    if np.dot(up, desired_up) < 0:
        right, up = -right, -up

    records = select_side_records(all_records, main_h, right, up)
    records = complete_h_neighbors(records, all_records, main_h, DEFAULT_H_NEIGHBORS)
    records = complete_metal_neighbors(atoms, records, all_records, periodic_axes, shells=1)

    return {
        "all_records": all_records,
        "records": records,
        "main_h": main_h,
        "cam_dir": cam_dir,
        "right": right,
        "up": up,
        "a": a,
        "b": b,
        "c": c,
    }


# ============================================================
# SCREENSHOTS Y VALIDACIÓN
# ============================================================

def make_background_transparent_png(path: Path, bg_color="white", tolerance=10):
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)

    rgb = arr[:, :, :3].astype(np.int16)
    bg_rgb = np.array(Image.new("RGB", (1, 1), bg_color).getpixel((0, 0))).astype(np.int16)
    candidate = np.max(np.abs(rgb - bg_rgb[None, None, :]), axis=2) <= tolerance

    h, w = candidate.shape
    visited = np.zeros((h, w), dtype=bool)
    q = deque()

    for x in range(w):
        for y in (0, h - 1):
            if candidate[y, x] and not visited[y, x]:
                q.append((y, x))
                visited[y, x] = True

    for y in range(h):
        for x in (0, w - 1):
            if candidate[y, x] and not visited[y, x]:
                q.append((y, x))
                visited[y, x] = True

    while q:
        y, x = q.popleft()
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < h and 0 <= xx < w and candidate[yy, xx] and not visited[yy, xx]:
                visited[yy, xx] = True
                q.append((yy, xx))

    arr[visited, 3] = 0
    Image.fromarray(arr, mode="RGBA").save(path)


def save_plotter_png(plotter: pv.Plotter, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    try:
        plotter.screenshot(str(out_png), transparent_background=TRANSPARENT_BACKGROUND)
    except TypeError:
        plotter.screenshot(str(out_png))
        if TRANSPARENT_BACKGROUND:
            make_background_transparent_png(out_png, bg_color=PYVISTA_BG, tolerance=12)


def validate_png_has_content(path: Path, label: str):
    if not path.exists():
        raise RuntimeError(f"No se generó {label}: {path}")

    arr = np.asarray(Image.open(path).convert("RGBA"))

    if arr[:, :, 3].min() < 255:
        ok = bool(np.any(arr[:, :, 3] > 5))
    else:
        ok = bool(np.any(np.any(arr[:, :, :3].astype(np.int16) < 245, axis=2)))

    if not ok:
        raise RuntimeError(f"{label} está vacío o fue completamente recortado: {path}")


# ============================================================
# RENDERS
# ============================================================

def new_plotter(size):
    plotter = pv.Plotter(off_screen=True, window_size=size)
    plotter.set_background(PYVISTA_BG)
    return plotter


def render_top_main(atoms, out_png, periodic_axes, layer_map, h_neighbor_count):
    ctx = get_top_context(atoms, periodic_axes, layer_map)
    plotter = new_plotter(IMAGE_SIZE)

    records = ctx["records"]
    focus_guess = np.mean(np.array([r["cart"] for r in records]), axis=0)
    focus, half_h = fit_parallel_from_records(
        records,
        focus_guess,
        ctx["right"],
        ctx["up"],
        IMAGE_SIZE,
        margin=TOP_MARGIN,
    )
    set_parallel_camera(plotter, focus, ctx["cam_dir"], ctx["up"], half_h)

    add_shifted_cell_box(plotter, atoms, shift_xy=(0, 0), line_width=CELL_LINE_WIDTH_MAIN)
    add_bonds(plotter, atoms, records, periodic_axes, layer_map, h_neighbor_count, view_kind="top")
    add_atoms(plotter, records, layer_map, view_kind="top")

    ctx["h_pixel"] = project_world_to_pixel(plotter, ctx["main_h"]["cart"], IMAGE_SIZE)
    save_plotter_png(plotter, out_png)
    plotter.close()
    return ctx


def render_side_main(atoms, out_png, periodic_axes, layer_map, h_neighbor_count):
    ctx = get_side_context(atoms, periodic_axes, layer_map)
    plotter = new_plotter(IMAGE_SIZE)

    records = ctx["records"]
    main_h = ctx["main_h"]

    fit_records = []
    for rec in records:
        rel = rec["cart"] - main_h["cart"]
        x = np.dot(rel, ctx["right"])
        y = np.dot(rel, ctx["up"])
        if abs(x) <= SIDE_X_HALF_WINDOW and -SIDE_DOWN_WINDOW <= y <= SIDE_UP_EXTRA:
            fit_records.append(rec)

    if not fit_records:
        fit_records = records

    focus, half_h = fit_parallel_from_records(
        fit_records,
        main_h["cart"],
        ctx["right"],
        ctx["up"],
        IMAGE_SIZE,
        margin=SIDE_MARGIN,
        top_fraction=SIDE_TOP_FRACTION,
    )
    set_parallel_camera(plotter, focus, ctx["cam_dir"], ctx["up"], half_h)

    add_side_cell_guides(plotter, records, ctx["right"], ctx["up"], line_width=CELL_LINE_WIDTH_MAIN)
    add_bonds(plotter, atoms, records, periodic_axes, layer_map, h_neighbor_count, view_kind="side")
    add_atoms(plotter, records, layer_map, view_kind="side")

    ctx["h_pixel"] = project_world_to_pixel(plotter, main_h["cart"], IMAGE_SIZE)
    save_plotter_png(plotter, out_png)
    plotter.close()
    return ctx


def render_top_zoom(atoms, out_png, periodic_axes, layer_map, h_neighbor_count, top_ctx):
    main_h = top_ctx["main_h"]

    bounds = (
        main_h["frac"][0] - ZOOM_TOP_FRAC_HALF_WIDTH,
        main_h["frac"][0] + ZOOM_TOP_FRAC_HALF_WIDTH,
        main_h["frac"][1] - ZOOM_TOP_FRAC_HALF_WIDTH,
        main_h["frac"][1] + ZOOM_TOP_FRAC_HALF_WIDTH,
        -0.08,
        1.08,
    )

    records = build_records_in_bounds(atoms, bounds, periodic_axes)
    if record_key(main_h) not in {record_key(r) for r in records}:
        records.append(main_h)

    records = filter_only_main_h(records, main_h)

    pool_bounds = (
        bounds[0] - 0.45,
        bounds[1] + 0.45,
        bounds[2] - 0.45,
        bounds[3] + 0.45,
        bounds[4],
        bounds[5],
    )
    all_records = build_records_in_bounds(atoms, pool_bounds, periodic_axes)

    records = complete_h_neighbors(records, all_records, main_h, h_neighbor_count)
    records = complete_metal_neighbors(atoms, records, all_records, periodic_axes, shells=1)

    plotter = new_plotter(ZOOM_SIZE)
    set_parallel_camera(
        plotter,
        np.asarray(main_h["cart"], dtype=float),
        top_ctx["cam_dir"],
        top_ctx["up"],
        ZOOM_TOP_PARALLEL_SCALE,
    )

    add_bonds(plotter, atoms, records, periodic_axes, layer_map, h_neighbor_count, view_kind="top_zoom")
    add_atoms(plotter, records, layer_map, view_kind="top_zoom")

    save_plotter_png(plotter, out_png)
    plotter.close()


def render_side_zoom(atoms, out_png, periodic_axes, layer_map, h_neighbor_count, side_ctx):
    main_h = side_ctx["main_h"]

    records = select_side_zoom_records(side_ctx["all_records"], main_h, side_ctx["right"], side_ctx["up"])
    records = complete_h_neighbors(records, side_ctx["all_records"], main_h, h_neighbor_count)
    records = complete_metal_neighbors(atoms, records, side_ctx["all_records"], periodic_axes, shells=1)

    plotter = new_plotter(ZOOM_SIZE)

    # H cerca del borde superior del zoom, como en 1.png.
    focus = np.asarray(main_h["cart"], dtype=float) - 0.50 * side_ctx["up"]
    set_parallel_camera(plotter, focus, side_ctx["cam_dir"], side_ctx["up"], ZOOM_SIDE_PARALLEL_SCALE)

    add_bonds(plotter, atoms, records, periodic_axes, layer_map, h_neighbor_count, view_kind="side_zoom")
    add_atoms(plotter, records, layer_map, view_kind="side_zoom")

    save_plotter_png(plotter, out_png)
    plotter.close()


# ============================================================
# COMPOSICIÓN FINAL
# ============================================================

def crop_background_margins(img: Image.Image, pad=18, return_bbox=False, bg=PYVISTA_BG):
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba)

    if arr[:, :, 3].min() < 255:
        mask = arr[:, :, 3] > 5
    else:
        rgb = arr[:, :, :3].astype(np.int16)
        if str(bg).lower() == "white":
            mask = np.any(rgb < 245, axis=2)
        else:
            mask = np.any(rgb > 18, axis=2)

    if not mask.any():
        bbox = (0, 0, img.width, img.height)
        cropped = rgba
    else:
        ys, xs = np.where(mask)
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad, img.width - 1)
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad, img.height - 1)
        bbox = (x0, y0, x1 + 1, y1 + 1)
        cropped = rgba.crop(bbox)

    return (cropped, bbox) if return_bbox else cropped


def resize_contain(img, box_w, box_h):
    img = img.convert("RGBA")
    scale = min(box_w / img.width, box_h / img.height)
    return img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)


def paste_center(canvas, img, box):
    x, y, w, h = box
    img = img.convert("RGBA")
    px = x + (w - img.width) // 2
    py = y + (h - img.height) // 2
    canvas.alpha_composite(img, (px, py))
    return (px, py, img.width, img.height)


def paste_left_center(canvas, img, box):
    """Pega alineado a la izquierda y centrado verticalmente.

    Para la side view evita que la slab quede flotando/centrada; debe arrancar
    como en 1.png y ocupar el bloque superior de izquierda a derecha.
    """
    x, y, w, h = box
    img = img.convert("RGBA")
    px = x
    py = y + (h - img.height) // 2
    canvas.alpha_composite(img, (px, py))
    return (px, py, img.width, img.height)


def map_original_pixel_to_canvas(pixel, crop_bbox, pasted_box):
    if pixel is None:
        return None

    x0, y0, x1, y1 = crop_bbox
    px, py, pw, ph = pasted_box

    return (
        px + (pixel[0] - x0) * (pw / max(x1 - x0, 1)),
        py + (pixel[1] - y0) * (ph / max(y1 - y0, 1)),
    )


def clamp_box_around(center, width, height, bounds_box):
    bx, by, bw, bh = bounds_box
    cx, cy = center

    width = int(min(width, 0.55 * bw))
    height = int(min(height, 0.55 * bh))

    x = int(round(cx - width / 2))
    y = int(round(cy - height / 2))

    x = max(bx + 8, min(x, bx + bw - width - 8))
    y = max(by + 8, min(y, by + bh - height - 8))

    return (x, y, width, height)


def draw_bracket(draw, box, color=BRACKET_COLOR, width=BRACKET_WIDTH, length=BRACKET_LEN):
    x, y, w, h = box
    x0, y0 = x, y
    x1, y1 = x + w, y + h

    segments = [
        [(x0, y0), (x0 + length, y0)], [(x0, y0), (x0, y0 + length)],
        [(x1, y0), (x1 - length, y0)], [(x1, y0), (x1, y0 + length)],
        [(x0, y1), (x0 + length, y1)], [(x0, y1), (x0, y1 - length)],
        [(x1, y1), (x1 - length, y1)], [(x1, y1), (x1, y1 - length)],
    ]

    for seg in segments:
        draw.line(seg, fill=color, width=width)


def draw_arrow(draw, start, end, color=ARROW_COLOR, width=ARROW_WIDTH, head=ARROW_HEAD):
    sx, sy = start
    ex, ey = end

    angle = math.atan2(ey - sy, ex - sx)
    left = angle + math.pi * 0.82
    right = angle - math.pi * 0.82

    p1 = (ex + head * math.cos(left), ey + head * math.sin(left))
    p2 = (ex + head * math.cos(right), ey + head * math.sin(right))

    if ARROW_OUTLINE_COLOR is not None:
        oh = head + 4
        op1 = (ex + oh * math.cos(left), ey + oh * math.sin(left))
        op2 = (ex + oh * math.cos(right), ey + oh * math.sin(right))
        draw.line([(sx, sy), (ex, ey)], fill=ARROW_OUTLINE_COLOR, width=ARROW_OUTLINE_WIDTH)
        draw.polygon([(ex, ey), op1, op2], fill=ARROW_OUTLINE_COLOR)

    draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
    draw.polygon([(ex, ey), p1, p2], fill=color)


def draw_zoom_border(draw, box):
    x, y, w, h = box
    for k in range(ZOOM_BORDER_WIDTH):
        draw.rectangle((x - k, y - k, x + w + k, y + h + k), outline=ZOOM_BORDER_COLOR)


def compose_final_figure(side_png, top_png, side_zoom_png, top_zoom_png, out_png, side_h_pixel=None, top_h_pixel=None):
    if COMPOSITE_TRANSPARENT:
        canvas = Image.new("RGBA", COMPOSITE_SIZE, (0, 0, 0, 0))
    else:
        canvas = Image.new("RGBA", COMPOSITE_SIZE, (255, 255, 255, 255))

    draw = ImageDraw.Draw(canvas, "RGBA")

    side_img, side_bbox = crop_background_margins(Image.open(side_png), pad=14, return_bbox=True)
    top_img, top_bbox = crop_background_margins(Image.open(top_png), pad=14, return_bbox=True)
    side_zoom = crop_background_margins(Image.open(side_zoom_png), pad=10)
    top_zoom = crop_background_margins(Image.open(top_zoom_png), pad=10)

    # Layout alineado con 1.png.
    side_box = (25, 20, 1620, 360)
    top_box = (25, 340, 1660, 985)
    side_zoom_box = (1715, 45, 620, 455)
    top_zoom_box = (1690, 585, 680, 715)

    side_actual = paste_left_center(canvas, resize_contain(side_img, side_box[2], side_box[3]), side_box)
    top_actual = paste_center(canvas, resize_contain(top_img, top_box[2], top_box[3]), top_box)
    side_zoom_actual = paste_center(canvas, resize_contain(side_zoom, side_zoom_box[2], side_zoom_box[3]), side_zoom_box)
    top_zoom_actual = paste_center(canvas, resize_contain(top_zoom, top_zoom_box[2], top_zoom_box[3]), top_zoom_box)

    draw_zoom_border(draw, side_zoom_actual)
    draw_zoom_border(draw, top_zoom_actual)

    sx, sy, sw, sh = side_actual
    tx, ty, tw, th = top_actual

    side_h_canvas = map_original_pixel_to_canvas(side_h_pixel, side_bbox, side_actual)
    top_h_canvas = map_original_pixel_to_canvas(top_h_pixel, top_bbox, top_actual)

    if side_h_canvas is None:
        side_h_canvas = (sx + 0.58 * sw, sy + 0.35 * sh)
    if top_h_canvas is None:
        top_h_canvas = (tx + 0.55 * tw, ty + 0.48 * th)

    side_bracket = clamp_box_around(
        (side_h_canvas[0], side_h_canvas[1] + 0.10 * sh),
        int(0.15 * sw),
        int(0.40 * sh),
        side_actual,
    )

    top_bracket = clamp_box_around(
        top_h_canvas,
        int(0.15 * tw),
        int(0.17 * th),
        top_actual,
    )

    draw_bracket(draw, side_bracket)
    draw_bracket(draw, top_bracket)

    sbx, sby, sbw, sbh = side_bracket
    zsx, zsy, zsw, zsh = side_zoom_actual
    draw_arrow(draw, (sbx + sbw, sby + sbh // 2), (zsx, zsy + zsh // 2))

    tbx, tby, tbw, tbh = top_bracket
    ztx, zty, ztw, zth = top_zoom_actual
    draw_arrow(draw, (tbx + tbw, tby + tbh // 2), (ztx, zty + zth // 2))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_png.with_name(out_png.stem + "__IN_PROGRESS__.png")
    canvas.save(tmp)
    tmp.replace(out_png)


# ============================================================
# PROCESAMIENTO
# ============================================================

def output_paths(root_dir: Path, out_file: Path, out_dir: Path):
    rel = out_file.relative_to(root_dir)
    target_dir = out_dir / rel.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    stem = out_file.stem
    part_dir = target_dir / "_tmp_render_parts" / stem

    if part_dir.exists():
        shutil.rmtree(part_dir, ignore_errors=True)
    part_dir.mkdir(parents=True, exist_ok=True)

    in_progress = target_dir / f"{stem}_figure__IN_PROGRESS__.png"
    if in_progress.exists():
        in_progress.unlink()

    return {
        "side": part_dir / f"{stem}_side.png",
        "top": part_dir / f"{stem}_top.png",
        "side_zoom": part_dir / f"{stem}_side_zoom.png",
        "top_zoom": part_dir / f"{stem}_top_zoom.png",
        "figure": target_dir / f"{stem}_figure.png",
        "part_dir": part_dir,
    }


def require_all_panel_files(paths):
    for label in ("side", "top", "side_zoom", "top_zoom"):
        if not paths[label].exists():
            raise RuntimeError(f"Falta panel {label}; no se acepta figura incompleta.")
        validate_png_has_content(paths[label], label)


def process_one(out_file: Path, root_dir: Path, out_dir: Path):
    in_file = out_file.with_suffix(".in")
    if not in_file.exists():
        print(f"[SKIP] {out_file.name} | no existe el .in correspondiente", flush=True)
        return

    out_text = read_text(out_file)
    if not has_complete_final_block(out_text):
        print(f"[SKIP] {out_file.name} | no tiene bloque final completo", flush=True)
        return

    input_text = read_text(in_file)
    mode = infer_mode(input_text, MODE)
    periodic_axes = periodic_axes_from_mode(mode)

    input_atoms = read_input_atoms(in_file)
    final_atoms = read_final_atoms_from_out(out_file)
    final_atoms = ensure_valid_cell(final_atoms, input_atoms)
    set_pbc_from_mode(final_atoms, mode)

    final_atoms, (sx, sy) = center_h_in_cell(final_atoms)
    layer_map = assign_layers(final_atoms, exclude_species={"H"})
    h_neighbor_count = infer_site_neighbor_count(out_file.stem)

    paths = output_paths(root_dir, out_file, out_dir)

    print(f"[RENDER] {out_file.relative_to(root_dir)} | side main", flush=True)
    side_ctx = render_side_main(final_atoms, paths["side"], periodic_axes, layer_map, h_neighbor_count)
    validate_png_has_content(paths["side"], "side main")

    print(f"[RENDER] {out_file.relative_to(root_dir)} | top main", flush=True)
    top_ctx = render_top_main(final_atoms, paths["top"], periodic_axes, layer_map, h_neighbor_count)
    validate_png_has_content(paths["top"], "top main")

    print(f"[RENDER] {out_file.relative_to(root_dir)} | side zoom", flush=True)
    render_side_zoom(final_atoms, paths["side_zoom"], periodic_axes, layer_map, h_neighbor_count, side_ctx)
    validate_png_has_content(paths["side_zoom"], "side zoom")

    print(f"[RENDER] {out_file.relative_to(root_dir)} | top zoom", flush=True)
    render_top_zoom(final_atoms, paths["top_zoom"], periodic_axes, layer_map, h_neighbor_count, top_ctx)
    validate_png_has_content(paths["top_zoom"], "top zoom")

    require_all_panel_files(paths)

    print(f"[COMPOSE] {out_file.relative_to(root_dir)} | final figure", flush=True)
    compose_final_figure(
        paths["side"],
        paths["top"],
        paths["side_zoom"],
        paths["top_zoom"],
        paths["figure"],
        side_h_pixel=side_ctx.get("h_pixel"),
        top_h_pixel=top_ctx.get("h_pixel"),
    )

    if not paths["figure"].exists():
        raise RuntimeError(f"No se creó la figura final: {paths['figure']}")

    validate_png_has_content(paths["figure"], "figura final")

    if not KEEP_INTERMEDIATE_PNGS:
        shutil.rmtree(paths["part_dir"], ignore_errors=True)

    print(f"[OK] {out_file.relative_to(root_dir)}", flush=True)
    print(f"     mode = {mode}", flush=True)
    print(f"     shift_xy_frac = ({sx:.5f}, {sy:.5f})", flush=True)
    print(f"     h_neighbor_count = {h_neighbor_count}", flush=True)
    print(f"     -> FIGURA FINAL COMPLETA: {paths['figure'].relative_to(out_dir)}", flush=True)


def main():
    root_dir = Path(INPUT_DIR).expanduser().resolve()
    out_dir = Path(OUTPUT_DIR).expanduser().resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {root_dir}")

    print(f"[INFO] Leyendo: {root_dir}", flush=True)
    print(f"[INFO] Guardando: {out_dir}", flush=True)
    print("[INFO] Salida final: *_figure.png", flush=True)

    if KEEP_INTERMEDIATE_PNGS:
        print("[INFO] Se conservarán paneles internos en _tmp_render_parts/", flush=True)

    found = False
    for out_file in root_dir.rglob("*.out"):
        if ONLY_CASES and out_file.name not in ONLY_CASES:
            continue

        found = True
        try:
            process_one(out_file, root_dir, out_dir)
        except Exception as e:
            try:
                part_dir = out_dir / out_file.relative_to(root_dir).parent / "_tmp_render_parts" / out_file.stem
                if not KEEP_INTERMEDIATE_PNGS:
                    shutil.rmtree(part_dir, ignore_errors=True)
            except Exception:
                pass

            print(f"[FAIL] {out_file.relative_to(root_dir)} | {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

            if STOP_ON_FIRST_FAILURE:
                raise

    if not found:
        print("[INFO] No se encontraron archivos .out", flush=True)


if __name__ == "__main__":
    main()
