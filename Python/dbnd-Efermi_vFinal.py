import os
import re
import glob
from typing import Optional, Tuple, List

import numpy as np


# =========================
# === CONFIGURACIÓN =======
# =========================
RAIZ = r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\Supercell\Hydrogen"

# Alineación al nivel de Fermi del NSCF
ALINEAR_EF = True

# Suavizado opcional para DOS antes de integrar
SIGMA = 0.0  # eV; 0 = apagado

# Si quieres calcular solo algunas carpetas, pon fragmentos del nombre de la hoja.
# Ejemplo: SOLO_LEAF_CONTIENE = ["Ni1Pd3", "Ni2Pd2"]
SOLO_LEAF_CONTIENE: List[str] = []


# =========================
# === ENERGÍA DE FERMI ==== 
# =========================
FERMI_PATTERNS = [
    re.compile(r"EFermi\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    re.compile(r"\bFermi\s+energy\b\s*[:=]?\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE),
    re.compile(r"\bthe\s+Fermi\s+energy\s+is\b\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE),
]


def extraer_ef_de_linea(linea: str) -> Optional[float]:
    """Extrae EF de una línea si contiene un patrón común de Quantum ESPRESSO."""
    if not linea or "fermi" not in linea.lower():
        return None

    for patron in FERMI_PATTERNS:
        m = patron.search(linea)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def buscar_ef_nscf(case_dir: str, leaf: str) -> Optional[float]:
    """Busca EF únicamente en la carpeta NSCF del caso correspondiente."""
    nscf_dir = os.path.join(case_dir, "NSCF")
    if not os.path.isdir(nscf_dir):
        return None

    candidatos = []

    directo = os.path.join(nscf_dir, f"{leaf}-NSCF.out")
    if os.path.isfile(directo):
        candidatos.append(directo)

    candidatos.extend(sorted(glob.glob(os.path.join(nscf_dir, f"*{leaf}*NSCF.out"))))
    candidatos.extend(sorted(glob.glob(os.path.join(nscf_dir, "*.out"))))

    # Quitar duplicados conservando el orden.
    candidatos = list(dict.fromkeys(candidatos))

    for path in candidatos:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for linea in f:
                    ef = extraer_ef_de_linea(linea)
                    if ef is not None:
                        return ef
        except Exception:
            continue

    return None


# =========================
# === LECTURA DE PDOS =====
# =========================
def cargar_tabla_pdos(path: str) -> np.ndarray:
    """Lee una tabla PDOS y devuelve datos numéricos ordenados por energía."""
    datos = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for linea in f:
            s = linea.strip()
            if not s or s.startswith("#"):
                continue

            try:
                datos.append([float(x) for x in s.split()])
            except Exception:
                continue

    if not datos:
        raise ValueError(f"Sin datos numéricos: {path}")

    arr = np.array(datos, dtype=float)
    idx = np.argsort(arr[:, 0])
    return arr[idx, :]


def gauss_suavizado(y: np.ndarray, E: np.ndarray, sigma: float) -> np.ndarray:
    """Aplica suavizado gaussiano opcional."""
    if sigma is None or sigma <= 0:
        return y

    dE = np.median(np.diff(E))
    if not np.isfinite(dE) or dE <= 0:
        return y

    half = int(np.ceil(4 * sigma / dE))
    if half < 1:
        return y

    xs = np.arange(-half, half + 1) * dE
    kernel = np.exp(-(xs**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


# =========================
# === PARSEO DE SISTEMAS ==
# =========================
def base_clean_leaf(leaf: str) -> str:
    """
    Quita sufijos de adsorción para obtener el nombre base.

    Ejemplos:
      Ni_111_B1              -> Ni_111
      Ni1Pd3_111-H_fcc_1     -> Ni1Pd3_111
      Ni2Pd2_111-H_hcp_2     -> Ni2Pd2_111
    """
    x = leaf.strip()
    x = re.sub(r"-H_(fcc|hcp)_\d+$", "", x, flags=re.IGNORECASE)
    x = re.sub(r"_H_(fcc|hcp)_\d+$", "", x, flags=re.IGNORECASE)
    x = re.sub(r"[-_](T\d+|B\d+|H\d+)$", "", x, flags=re.IGNORECASE)
    return x


def formula_desde_leaf(leaf: str) -> str:
    """Obtiene la fórmula química inicial desde el nombre de la carpeta PDOS."""
    base = base_clean_leaf(leaf)
    return base.split("_")[0] if base else leaf


def especies_desde_formula(formula: str) -> List[str]:
    """
    Extrae especies químicas reales desde fórmulas tipo:
      Ni, Pd, Ni1Pd3, Ni2Pd2, Ni3Pd1
    """
    especies = re.findall(r"[A-Z][a-z]?", formula)
    return list(dict.fromkeys(especies))


def pasa_filtro_leaf(leaf: str) -> bool:
    if not SOLO_LEAF_CONTIENE:
        return True
    return any(tok in leaf for tok in SOLO_LEAF_CONTIENE)


# =========================
# === BÚSQUEDA DE ARCHIVOS =
# =========================
def escanear_carpetas_pdos(raiz: str) -> List[str]:
    """Encuentra carpetas PDOS/<leaf> que contienen archivos .pdos_atm#*."""
    halladas = []

    for root, dirs, files in os.walk(raiz):
        if os.path.basename(root).lower() == "pdos":
            for d in sorted(dirs):
                hoja = os.path.join(root, d)
                if os.path.isdir(hoja) and glob.glob(os.path.join(hoja, "*.pdos_atm#*")):
                    halladas.append(hoja)

    return sorted(halladas)


def listar_pdos_d(pdos_dir: str, elemento: str) -> List[str]:
    """Lista archivos PDOS del orbital d para un elemento."""
    patron_d = os.path.join(pdos_dir, f"*.pdos_atm#*({elemento})_wfc#*(d)*")
    files_d = sorted(glob.glob(patron_d))
    if files_d:
        return files_d

    patron_d_laxo = os.path.join(pdos_dir, f"*.pdos_atm#*({elemento})_wfc#*d*")
    files_d = sorted(glob.glob(patron_d_laxo))
    if files_d:
        return files_d

    return []


# =========================
# === CÁLCULO DBAND =======
# =========================
def integrar_centro(E: np.ndarray, dos: np.ndarray) -> Optional[float]:
    """Calcula el centroide de la DOS por integración trapezoidal."""
    area = np.trapezoid(dos, E)
    if not np.isfinite(area) or abs(area) < 1e-12:
        return None

    momento = np.trapezoid(E * dos, E)
    if not np.isfinite(momento):
        return None

    return momento / area


def calcular_dband_en_folder(pdos_dir: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Suma las PDOS d del sistema y calcula epsilon_d.

    Devuelve:
      epsilon_d, EF_NSCF
    """
    leaf = os.path.basename(pdos_dir)
    case_dir = os.path.dirname(os.path.dirname(pdos_dir))

    formula = formula_desde_leaf(leaf)
    especies = especies_desde_formula(formula)

    if not especies:
        return None, None

    files_d = []
    for sp in especies:
        files_d.extend(listar_pdos_d(pdos_dir, sp))

    files_d = sorted(set(files_d))
    if not files_d:
        return None, buscar_ef_nscf(case_dir, leaf)

    ef_nscf = buscar_ef_nscf(case_dir, leaf)

    E_ref = None
    dsum = None

    for fp in files_d:
        arr = cargar_tabla_pdos(fp)
        if arr.shape[1] < 2:
            continue

        E = arr[:, 0]

        # Mantiene la misma lógica del código original:
        # si hay 3 columnas o más, suma columna 2 + columna 3 como up + down.
        if arr.shape[1] >= 3:
            contrib = arr[:, 1] + arr[:, 2]
        else:
            contrib = arr[:, 1]

        if E_ref is None:
            E_ref = E
            dsum = contrib.copy()
        else:
            if len(E) == len(E_ref) and np.allclose(E, E_ref, atol=1e-8):
                dsum += contrib
            else:
                raise ValueError(f"Mallas de energía distintas: {os.path.basename(fp)}")

    if E_ref is None or dsum is None:
        return None, ef_nscf

    E_work = E_ref.copy()
    if ALINEAR_EF and ef_nscf is not None:
        E_work = E_work - ef_nscf

    dsum = gauss_suavizado(dsum, E_work, SIGMA)
    epsilon_d = integrar_centro(E_work, dsum)

    return epsilon_d, ef_nscf


# =========================
# === MAIN ================
# =========================
def main() -> None:
    raiz = os.path.normpath(RAIZ)

    pdos_dirs = escanear_carpetas_pdos(raiz)
    pdos_dirs = [p for p in pdos_dirs if pasa_filtro_leaf(os.path.basename(p))]

    print("sistema\tepsilon_d_eV\tEF_NSCF_eV")

    for pdos_dir in pdos_dirs:
        leaf = os.path.basename(pdos_dir)
        case_dir = os.path.dirname(os.path.dirname(pdos_dir))
        case = os.path.basename(case_dir)
        sistema = f"{case}/{leaf}"

        try:
            epsilon_d, ef_nscf = calcular_dband_en_folder(pdos_dir)

            eps_txt = f"{epsilon_d:.6f}" if epsilon_d is not None else "NA"
            ef_txt = f"{ef_nscf:.6f}" if ef_nscf is not None else "NA"

            print(f"{sistema}\t{eps_txt}\t{ef_txt}")

        except Exception as exc:
            print(f"{sistema}\tERROR\t{exc}")


if __name__ == "__main__":
    main()
