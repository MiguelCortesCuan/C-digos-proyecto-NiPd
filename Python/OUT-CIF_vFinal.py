from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.io import read, write


LOGGER = logging.getLogger("qe_relax_to_vesta")


BEGIN_FINAL = "Begin final coordinates"
END_FINAL = "End final coordinates"


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def has_complete_final_block(text: str) -> bool:
    return BEGIN_FINAL in text and END_FINAL in text and text.index(BEGIN_FINAL) < text.index(END_FINAL)


def extract_assume_isolated(input_text: str) -> Optional[str]:
    match = re.search(
        r"assume_isolated\s*=\s*['\"]?([A-Za-z0-9_\-]+)['\"]?",
        input_text,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip().lower() if match else None


def infer_mode(input_text: str, requested_mode: str) -> str:
    """
    Devuelve '2d' o '3d'.
    """
    if requested_mode in {"2d", "3d"}:
        return requested_mode

    assume_isolated = extract_assume_isolated(input_text)
    if assume_isolated is not None and "2d" in assume_isolated:
        return "2d"
    return "3d"


def periodic_axes_from_mode(mode: str) -> Tuple[int, ...]:
    if mode == "2d":
        return (0, 1)
    return (0, 1, 2)


def set_pbc_from_mode(atoms: Atoms, mode: str) -> None:
    if mode == "2d":
        atoms.pbc = (True, True, False)
    else:
        atoms.pbc = (True, True, True)


def read_final_atoms_from_out(out_file: Path) -> Atoms:
    """
    Lee el último frame del .out usando ASE, pero solo después de verificar
    que existe un bloque final completo.
    """
    text = read_text(out_file)
    if not has_complete_final_block(text):
        raise RuntimeError("No contiene un bloque final completo de coordenadas.")

    # Intento rápido
    try:
        atoms = read(str(out_file), format="espresso-out", index=-1)
        if isinstance(atoms, list):
            atoms = atoms[-1]
        if not isinstance(atoms, Atoms) or len(atoms) == 0:
            raise RuntimeError("ASE devolvió una estructura vacía.")
        return atoms
    except Exception as exc_fast:
        LOGGER.debug("Fallo lectura rápida en %s: %s", out_file.name, exc_fast)

    # Fallback más robusto
    try:
        frames = list(read(str(out_file), format="espresso-out", index=":"))
        valid = [f for f in frames if isinstance(f, Atoms) and len(f) > 0]
        if not valid:
            raise RuntimeError("ASE no encontró frames válidos.")
        return valid[-1]
    except Exception as exc_all:
        raise RuntimeError(f"No se pudo leer la geometría final con ASE: {exc_all}") from exc_all


def read_input_atoms(in_file: Path) -> Atoms:
    atoms = read(str(in_file), format="espresso-in")
    if isinstance(atoms, list):
        atoms = atoms[-1]
    if not isinstance(atoms, Atoms) or len(atoms) == 0:
        raise RuntimeError("No se pudo leer la estructura de entrada.")
    return atoms


def ensure_valid_cell(final_atoms: Atoms, input_atoms: Atoms) -> Atoms:
    """
    Usa la celda del .out si es válida; en caso contrario, usa la del .in.
    """
    out_cell = np.array(final_atoms.cell)
    out_lengths = np.linalg.norm(out_cell, axis=1)

    if np.all(out_lengths > 1e-12):
        return final_atoms

    LOGGER.warning("Celda inválida en .out; se usará la celda del .in.")
    fixed = final_atoms.copy()
    fixed.set_cell(input_atoms.cell, scale_atoms=False)
    fixed.pbc = input_atoms.pbc
    return fixed


def largest_gap_shift_1d(frac_coords: np.ndarray) -> Tuple[float, float]:
    """
    Dadas coordenadas fraccionales 1D, calcula:
    - shift fraccional que pone el corte periódico en el centro del mayor hueco
    - tamaño de ese mayor hueco
    """
    coords = np.mod(np.asarray(frac_coords, dtype=float), 1.0)

    if coords.size == 0:
        return 0.0, 1.0

    coords = np.sort(coords)
    wrapped = np.r_[coords, coords[0] + 1.0]
    gaps = np.diff(wrapped)

    idx = int(np.argmax(gaps))
    start = coords[idx]
    gap = float(gaps[idx])

    midpoint = (start + 0.5 * gap) % 1.0
    shift = (-midpoint) % 1.0
    return float(shift), gap


def compute_fractional_pad(min_gap: float) -> float:
    """
    Pad automático muy pequeño en coordenadas fraccionales.
    No es fijo: depende del hueco disponible.
    """
    if min_gap <= 0.0:
        return 1e-10
    pad = 0.01 * min_gap
    return float(np.clip(pad, 1e-10, 1e-6))


def center_nonperiodic_axis_if_requested(
    atoms: Atoms,
    periodic_axes: Tuple[int, ...],
    enable: bool,
) -> Atoms:
    """
    Para slabs 2D puede ser útil centrar el espesor del slab en el eje no periódico.
    Solo aplica una traslación rígida.
    """
    if not enable:
        return atoms

    nonperiodic_axes = [ax for ax in (0, 1, 2) if ax not in periodic_axes]
    if len(nonperiodic_axes) != 1:
        return atoms

    ax = nonperiodic_axes[0]
    shifted = atoms.copy()

    scaled = shifted.get_scaled_positions(wrap=False)
    min_s = float(np.min(scaled[:, ax]))
    max_s = float(np.max(scaled[:, ax]))
    center_s = 0.5 * (min_s + max_s)

    scaled[:, ax] += 0.5 - center_s
    shifted.set_scaled_positions(scaled)
    return shifted


def shift_structure_for_vesta(
    atoms: Atoms,
    periodic_axes: Tuple[int, ...],
    apply_auto_pad: bool = True,
) -> Tuple[Atoms, np.ndarray, float, float]:
    """
    Aplica una traslación rígida para que el borde periódico de la celda
    caiga dentro del mayor hueco vacío de cada eje periódico.

    Devuelve:
    - estructura desplazada
    - vector de shift fraccional aplicado
    - mínimo mayor-hueco entre ejes periódicos
    - pad fraccional aplicado
    """
    shifted = atoms.copy()
    scaled = shifted.get_scaled_positions(wrap=False)

    shift_vec = np.zeros(3, dtype=float)
    gap_list = []

    for ax in periodic_axes:
        shift_ax, gap_ax = largest_gap_shift_1d(scaled[:, ax])
        shift_vec[ax] = shift_ax
        gap_list.append(gap_ax)

    min_gap = min(gap_list) if gap_list else 1.0
    pad = compute_fractional_pad(min_gap) if apply_auto_pad else 0.0

    for ax in periodic_axes:
        scaled[:, ax] = np.mod(scaled[:, ax] + shift_vec[ax] + pad, 1.0)

    shifted.set_scaled_positions(scaled)
    return shifted, shift_vec, float(min_gap), float(pad)


def relative_output_path(root_dir: Path, out_file: Path, out_dir: Path, suffix: str) -> Path:
    rel = out_file.relative_to(root_dir)
    return out_dir / rel.parent / f"{out_file.stem}{suffix}"


def iter_out_files(root_dir: Path) -> Iterable[Path]:
    yield from root_dir.rglob("*.out")


def process_one(
    out_file: Path,
    root_dir: Path,
    out_dir: Path,
    requested_mode: str,
    center_nonperiodic: bool,
    also_cif: bool,
) -> dict:
    in_file = out_file.with_suffix(".in")

    record = {
        "out_file": str(out_file),
        "in_file": str(in_file),
        "status": "",
        "mode": "",
        "reason": "",
        "shift_fx": "",
        "shift_fy": "",
        "shift_fz": "",
        "min_gap_frac": "",
        "pad_frac": "",
        "xsf_file": "",
        "cif_file": "",
    }

    if not in_file.exists():
        record["status"] = "skipped"
        record["reason"] = "No existe el archivo .in correspondiente."
        return record

    out_text = read_text(out_file)
    if not has_complete_final_block(out_text):
        record["status"] = "skipped"
        record["reason"] = "No existe bloque final completo (Begin/End final coordinates)."
        return record

    input_text = read_text(in_file)
    mode = infer_mode(input_text, requested_mode)
    periodic_axes = periodic_axes_from_mode(mode)

    input_atoms = read_input_atoms(in_file)
    final_atoms = read_final_atoms_from_out(out_file)
    final_atoms = ensure_valid_cell(final_atoms, input_atoms)
    set_pbc_from_mode(final_atoms, mode)

    shifted_atoms, shift_vec, min_gap, pad = shift_structure_for_vesta(
        final_atoms,
        periodic_axes=periodic_axes,
        apply_auto_pad=True,
    )

    shifted_atoms = center_nonperiodic_axis_if_requested(
        shifted_atoms,
        periodic_axes=periodic_axes,
        enable=center_nonperiodic,
    )

    xsf_path = relative_output_path(root_dir, out_file, out_dir, ".xsf")
    xsf_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(xsf_path), shifted_atoms, format="xsf")

    cif_path = ""
    if also_cif:
        cif_out = relative_output_path(root_dir, out_file, out_dir, ".cif")
        write(str(cif_out), shifted_atoms, format="cif")
        cif_path = str(cif_out)

    record.update(
        {
            "status": "written",
            "mode": mode,
            "reason": "OK",
            "shift_fx": f"{shift_vec[0]:.12g}",
            "shift_fy": f"{shift_vec[1]:.12g}",
            "shift_fz": f"{shift_vec[2]:.12g}",
            "min_gap_frac": f"{min_gap:.12g}",
            "pad_frac": f"{pad:.12g}",
            "xsf_file": str(xsf_path),
            "cif_file": cif_path,
        }
    )
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convierte archivos .out convergidos de QE a XSF para VESTA, "
            "sin repetir la celda y recolocando automáticamente el corte periódico."
        )
    )

    parser.add_argument(
        "root_dir",
        nargs="?",
        default=".",
        help="Carpeta raíz de entrada con archivos .out/.in",
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=None,
        help="Carpeta raíz de salida. Por defecto: <root_dir>/VESTA-ready",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "2d", "3d"),
        default="auto",
        help="Tratamiento periódico: auto, 2d o 3d",
    )
    parser.add_argument(
        "--center-nonperiodic",
        action="store_true",
        help="Centra el slab a lo largo del eje no periódico cuando aplique",
    )
    parser.add_argument(
        "--also-cif",
        action="store_true",
        help="Además del XSF, escribe también un CIF",
    )
    parser.add_argument(
        "--summary-name",
        default="qe_relax_to_vesta_summary.csv",
        help="Nombre del CSV resumen",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Activa logging detallado",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    root_dir = Path(args.root_dir).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir is not None
        else (root_dir / "VESTA-ready").resolve()
    )

    if not root_dir.exists():
        LOGGER.error("No existe la carpeta de entrada: %s", root_dir)
        return 1

    LOGGER.info("Escaneando: %s", root_dir)
    LOGGER.info("Salida: %s", out_dir)

    records = []
    found_any = False

    for out_file in iter_out_files(root_dir):
        found_any = True
        try:
            rec = process_one(
                out_file=out_file,
                root_dir=root_dir,
                out_dir=out_dir,
                requested_mode=args.mode,
                center_nonperiodic=args.center_nonperiodic,
                also_cif=args.also_cif,
            )
            records.append(rec)

            if rec["status"] == "written":
                LOGGER.info("OK  %s -> %s", out_file.name, Path(rec["xsf_file"]).name)
            else:
                LOGGER.warning("SKIP  %s  |  %s", out_file.name, rec["reason"])

        except Exception as exc:
            LOGGER.error("FAIL  %s  |  %s", out_file.name, exc)
            records.append(
                {
                    "out_file": str(out_file),
                    "in_file": str(out_file.with_suffix(".in")),
                    "status": "failed",
                    "mode": "",
                    "reason": str(exc),
                    "shift_fx": "",
                    "shift_fy": "",
                    "shift_fz": "",
                    "min_gap_frac": "",
                    "pad_frac": "",
                    "xsf_file": "",
                    "cif_file": "",
                }
            )

    if not found_any:
        LOGGER.warning("No se encontraron archivos .out.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / args.summary_name

    fieldnames = [
        "out_file",
        "in_file",
        "status",
        "mode",
        "reason",
        "shift_fx",
        "shift_fy",
        "shift_fz",
        "min_gap_frac",
        "pad_frac",
        "xsf_file",
        "cif_file",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    written = sum(r["status"] == "written" for r in records)
    skipped = sum(r["status"] == "skipped" for r in records)
    failed = sum(r["status"] == "failed" for r in records)

    LOGGER.info("Resumen: escritos=%d, omitidos=%d, fallidos=%d", written, skipped, failed)
    LOGGER.info("CSV resumen: %s", summary_path)

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())