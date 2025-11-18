# -*- coding: utf-8 -*-
r"""
Batch d-band center calculator for Quantum ESPRESSO PDOS trees.
- Recorrer RAIZ y detectar carpetas ...\PDOS\<Elemento_...>
- Leer EF (manual > DOS/*.out > NSCF/*.out > header de los pdos_atm)
- Sumar PDOS de orbital d del elemento (↑ + ↓) y calcular ε_d
- Imprimir resultados; sin tablas ni gráficas
- AHORA también imprime la energía de Fermi usada y su fuente
"""

import os
import re
import glob
from typing import Optional, Tuple, List
import numpy as np

# =========================
# === CONFIGURACIÓN =======
# =========================
RAIZ = r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\Supercell\Hydrogen"

# Alineación al nivel de Fermi
ALINEAR_EF = True
EF_MANUAL: Optional[float] = None  # si no es None, domina

# Suavizado (opcional) para el DOS previo a la integración
SIGMA = 0.0  # eV (0 = off)

# =========================
# === UTILIDADES ========== 
# =========================
HEADER_EF_PATTERNS = [
    re.compile(r"EFermi\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    re.compile(r"\bFermi\s+energy\b\s*[:=]?\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    re.compile(r"\bthe\s+Fermi\s+energy\s+is\b\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE),
]

SITE_MAP = {
    "T1": "Top site",
    "B1": "Bridge site",
    "H1": "Hollow fcc",
    "H2": "Hollow hcp",
}

def leer_ef_desde_header(lineas: List[str]) -> Optional[float]:
    for ln in lineas[:400]:
        if ("Fermi" in ln) or ("EFermi" in ln) or ln.strip().startswith("#") or ("Fermi" in ln.lower()):
            for pat in HEADER_EF_PATTERNS:
                m = pat.search(ln)
                if m:
                    try:
                        return float(m.group(1))
                    except Exception:
                        pass
    return None

def cargar_tabla_generica(path: str) -> Tuple[np.ndarray, Optional[float]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lineas = f.readlines()
    ef = leer_ef_desde_header(lineas)
    datos = []
    for ln in lineas:
        s = ln.strip()
        if (not s) or s.startswith("#"):
            continue
        parts = s.split()
        try:
            row = [float(x) for x in parts]
        except Exception:
            continue
        datos.append(row)
    if not datos:
        raise ValueError(f"Sin datos numéricos: {path}")
    arr = np.array(datos, float)
    idx = np.argsort(arr[:, 0])
    arr = arr[idx, :]
    return arr, ef

def gauss_suavizado(y: np.ndarray, E: np.ndarray, sigma: float) -> np.ndarray:
    if sigma is None or sigma <= 0:
        return y
    dE = np.median(np.diff(E))
    if not np.isfinite(dE) or dE <= 0:
        return y
    half = int(np.ceil(4 * sigma / dE))
    xs = np.arange(-half, half + 1) * dE
    kernel = np.exp(-(xs**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")

def listar_pdos_elemento_orbital(carpeta: str, elemento: str, orb_letra: str = "d") -> List[str]:
    """
    QE suele nombrar como: *.pdos_atm#*({elem})_wfc#*(d)*
    Este patrón filtra los wfc 'd'. Si no se encuentran,
    como fallback devolvemos todos los wfc del elemento.
    """
    patron_d = os.path.join(carpeta, f"*.pdos_atm#*({elemento})_wfc#*{orb_letra}*")
    files_d = sorted(glob.glob(patron_d))
    if files_d:
        return files_d
    # Fallback: devolver todos los wfc del elemento (se avisará)
    patron_all = os.path.join(carpeta, f"*.pdos_atm#*({elemento})_wfc#*")
    return sorted(glob.glob(patron_all))

def _leer_todo(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()

def _buscar_ef_en_archivo(path: str) -> Optional[float]:
    try:
        return leer_ef_desde_header(_leer_todo(path))
    except Exception:
        return None

def _buscar_ef_en_dos_out(dos_dir: str, leaf: str) -> Tuple[Optional[float], Optional[str]]:
    cand = os.path.join(dos_dir, f"{leaf}-dos.out")
    if os.path.isfile(cand):
        ef = _buscar_ef_en_archivo(cand)
        if ef is not None:
            return ef, cand
    globc = glob.glob(os.path.join(dos_dir, f"{leaf}*dos.out"))
    for g in globc:
        ef = _buscar_ef_en_archivo(g)
        if ef is not None:
            return ef, g
    return None, None

def _buscar_ef_en_nscf_out(case_dir: str, leaf: str) -> Tuple[Optional[float], Optional[str]]:
    nscf_dir = os.path.join(case_dir, "NSCF")
    if not os.path.isdir(nscf_dir):
        return None, None
    direct = os.path.join(nscf_dir, f"{leaf}-NSCF.out")
    if os.path.isfile(direct):
        ef = _buscar_ef_en_archivo(direct)
        if ef is not None:
            return ef, direct
    globc = glob.glob(os.path.join(nscf_dir, f"*{leaf}*NSCF.out"))
    for g in sorted(globc):
        ef = _buscar_ef_en_archivo(g)
        if ef is not None:
            return ef, g
    return None, None

def dos_candidatos_para(pdos_dir: str) -> List[str]:
    cand = [
        os.path.join(os.path.dirname(os.path.dirname(pdos_dir)), "DOS"),
        os.path.join(os.path.dirname(pdos_dir), "DOS"),
        os.path.join(pdos_dir, "DOS"),
    ]
    uniq = []
    for c in cand:
        c = os.path.normpath(c)
        if c not in uniq and os.path.isdir(c):
            uniq.append(c)
    return uniq

def leer_ef_desde_dos(dos_dir: str, leaf: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Buscar EF en:
    1) DOS\<leaf>.dos (header)  [si existiera]
    2) DOS\<leaf>-dos.out
    3) NSCF\<leaf>-NSCF.out
    """
    dos_exact = os.path.join(dos_dir, f"{leaf}.dos")
    if os.path.isfile(dos_exact):
        _, ef = cargar_tabla_generica(dos_exact)
        if ef is not None:
            return ef, dos_exact
    ef, src = _buscar_ef_en_dos_out(dos_dir, leaf)
    if ef is not None:
        return ef, src
    case_dir = os.path.dirname(dos_dir)
    ef, src = _buscar_ef_en_nscf_out(case_dir, leaf)
    if ef is not None:
        return ef, src
    return None, None

def parse_elemento_y_site(leaf: str) -> Tuple[str, Optional[str]]:
    parts = leaf.split("_")
    elem = re.match(r"^[A-Za-z0-9]+", parts[0]).group(0) if parts else leaf
    site = None
    if len(parts) >= 3:
        cand = parts[-1].upper()
        if cand in SITE_MAP:
            site = cand
    return elem, site

def facet_from_leaf(leaf: str) -> Optional[str]:
    parts = leaf.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return parts[1]
    return None

def escanear_carpetas_pdos(raiz: str) -> List[str]:
    halladas = []
    for root, dirs, files in os.walk(raiz):
        if os.path.basename(root).lower() == "pdos":
            for d in sorted(dirs):
                hoja = os.path.join(root, d)
                if os.path.isdir(hoja):
                    if glob.glob(os.path.join(hoja, "*.pdos_atm#*")):
                        halladas.append(hoja)
    return halladas

def preparar_series(E: np.ndarray, up: np.ndarray, dn: np.ndarray, ef: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    if ALINEAR_EF and (ef is not None):
        E = E - ef
    # Suavizado opcional
    up = gauss_suavizado(up, E, SIGMA)
    dn = gauss_suavizado(dn, E, SIGMA)
    # Para integrar, la densidad total d = up + dn (no se hace negativo)
    d_tot = up + dn
    return E, d_tot

def integrar_centro(E: np.ndarray, dos: np.ndarray) -> Optional[float]:
    w = np.trapz(dos, E)
    if not np.isfinite(w) or abs(w) < 1e-12:
        return None
    num = np.trapz(E * dos, E)
    return num / w

def calcular_dband_en_folder(pdos_dir: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    Suma los PDOS de orbital d para el elemento indicado por el nombre del folder leaf
    y calcula el centro de banda d. Devuelve (eps_d, ef_usada, fuente_ef).
    """
    leaf = os.path.basename(pdos_dir)          # p.ej., Ni_111_T1 o Ni_111
    elemento, _ = parse_elemento_y_site(leaf)

    # Archivos d del elemento
    files_d = listar_pdos_elemento_orbital(pdos_dir, elemento, "d")
    if not files_d:
        print(f"   [ERROR] No se encontraron pdos_atm para {elemento} (orbital d).")
        return None, None, "EF:NA"

    only_fallback = (not any("(d)" in os.path.basename(f) or "_d" in os.path.basename(f).lower() for f in files_d))
    if only_fallback:
        print(f"   [WARN] No se detectó '(d)' en nombres; usando TODOS los wfc de {elemento} (fallback).")

    # Intentar EF desde DOS/NSCF
    ef_dos = None
    src_dos = None
    for ddir in dos_candidatos_para(pdos_dir):
        ef_dos, src_dos = leer_ef_desde_dos(ddir, leaf=leaf)
        if ef_dos is not None:
            break

    # Sumar PDOS d y capturar posible EF de header
    E_ref = None
    dsum = None
    ef_header = None
    ef_header_src = None
    for fp in files_d:
        arr, ef = cargar_tabla_generica(fp)
        if arr.shape[1] < 3:
            # columnas esperadas: E  up  dn
            continue
        E = arr[:, 0]
        up = arr[:, 1]
        dn = arr[:, 2]
        if E_ref is None:
            E_ref = E
            dsum = up + dn
        else:
            if len(E) == len(E_ref) and np.allclose(E, E_ref, atol=1e-8):
                dsum += (up + dn)
            else:
                raise ValueError(f"Mallas de energía distintas: {os.path.basename(fp)}")
        if (ef_header is None) and (ef is not None):
            ef_header = ef
            ef_header_src = fp

    if E_ref is None or dsum is None:
        print("   [ERROR] No se pudieron acumular PDOS.")
        return None, None, "EF:NA"

    # Decidir EF final y su fuente
    if EF_MANUAL is not None:
        ef_final = EF_MANUAL
        ef_src = "EF_MANUAL"
    elif ef_dos is not None:
        ef_final = ef_dos
        ef_src = f"DOS/NSCF:{os.path.normpath(src_dos)}"
    else:
        ef_final = ef_header
        ef_src = f"PDOS header:{os.path.normpath(ef_header_src) if ef_header_src else 'NA'}"

    # Preparar series (alinear EF y suavizar si aplica)
    # reutilizamos preparar_series esperando up/dn; pasamos dsum/2, dsum/2
    E_aligned, d_tot = preparar_series(E_ref, dsum*0.5, dsum*0.5, ef_final)

    # Calcular ε_d
    eps_d = integrar_centro(E_aligned, d_tot)
    return eps_d, ef_final, ef_src

def construir_clean_counterpart(pdos_dir: str, raiz: str) -> Optional[str]:
    p = os.path.normpath(pdos_dir)                     # ...\111_2x2-H\PDOS\Ni_111_T1
    leaf = os.path.basename(p)                         # Ni_111_T1
    base_leaf = re.sub(r"_(T1|B1|H1|H2)$", "", leaf, flags=re.IGNORECASE)  # Ni_111

    pdos_parent = os.path.dirname(p)                   # ...\111_2x2-H\PDOS
    case_dir = os.path.dirname(pdos_parent)            # ...\111_2x2-H
    casos_parent = os.path.dirname(case_dir)           # ...\Hydrogen

    case_name = os.path.basename(case_dir)             # 111_2x2-H
    if case_name.endswith("-H"):
        clean_case = case_name[:-2]                    # 111_2x2
        cand = os.path.join(casos_parent, clean_case, "PDOS", base_leaf)
        if os.path.isdir(cand):
            return cand

    # fallback: buscar hermano exacto PDOS/<base_leaf> bajo casos hermanos
    try:
        for sib in os.listdir(casos_parent):
            cand = os.path.join(casos_parent, sib, "PDOS", base_leaf)
            if os.path.isdir(cand):
                return cand
    except Exception:
        pass
    return None

def main():
    raiz = os.path.normpath(RAIZ)
    print(f"\n[DBAND] Iniciando\n[DBAND] RAIZ: {raiz}\n[DBAND] Existe RAIZ? {os.path.isdir(raiz)}")

    # Diagnóstico rápido del primer nivel
    try:
        print("\n[DBAND] Primer nivel bajo RAIZ:")
        for d in sorted(os.listdir(raiz))[:50]:
            p = os.path.join(raiz, d)
            print("   ", "[D]" if os.path.isdir(p) else "[F]", d)
    except Exception as e:
        print("[DBAND] Error listando RAIZ:", e)

    # Buscar carpetas PDOS
    pdos_dirs = escanear_carpetas_pdos(raiz)
    print(f"\n[DBAND] Se hallaron {len(pdos_dirs)} carpeta(s) PDOS elegibles.")
    for p in pdos_dirs:
        print("   [PDOS] ", p)

    if not pdos_dirs:
        print("\n[WARN] No se hallaron carpetas PDOS bajo la RAIZ.")
        print("[HINT] Esperado algo como:")
        print(r"   G:\...\Hydrogen\111_2x2\PDOS\Ni_111")
        print(r"   G:\...\Hydrogen\111_2x2-H\PDOS\Ni_111_T1")
        return

    # Procesamiento
    for pdos_dir in pdos_dirs:
        leaf = os.path.basename(pdos_dir)                      # Ni_111_T1 o Ni_111
        caso_dir = os.path.dirname(os.path.dirname(pdos_dir))  # ...\<caso>
        caso = os.path.basename(caso_dir)                      # 111_2x2 o 111_2x2-H
        elemento, site = parse_elemento_y_site(leaf)
        facet = facet_from_leaf(leaf)
        es_h = caso.endswith("-H")

        header = f"[CASE] {caso} | Leaf={leaf} | elem={elemento} | facet={facet or '-'}"
        if es_h and site: header += f" | site={site}"
        print("\n" + header)

        if es_h:
            clean_dir = construir_clean_counterpart(pdos_dir, raiz)
            print("   [PAIR] CLEAN:", clean_dir if clean_dir else "no encontrado")

            # +H
            try:
                eps_h, ef_h, src_h = calcular_dband_en_folder(pdos_dir)
                if ef_h is not None:
                    print(f"   [EF  +H] {ef_h:.6f} eV  | fuente: {src_h}")
                if eps_h is not None:
                    print(f"   [ε_d +H] {eps_h:.6f} eV (relativo a E_F)")
                else:
                    print("   [ε_d +H] No calculable.")
            except Exception as ex:
                print("   [ERROR +H] ", ex)

            # CLEAN
            if clean_dir and os.path.isdir(clean_dir):
                try:
                    eps_c, ef_c, src_c = calcular_dband_en_folder(clean_dir)
                    if ef_c is not None:
                        print(f"   [EF  C ] {ef_c:.6f} eV  | fuente: {src_c}")
                    if eps_c is not None:
                        print(f"   [ε_d C ] {eps_c:.6f} eV (relativo a E_F)")
                        if (eps_h is not None) and (ef_h is not None) and (ef_c is not None):
                            print(f"   [Δε_d (+H - C)] {eps_h - eps_c:+.6f} eV")
                    else:
                        print("   [ε_d C ] No calculable.")
                except Exception as ex:
                    print("   [ERROR C ] ", ex)
            else:
                print("   [SKIP] Sin contraparte CLEAN.")
        else:
            # Sólo CLEAN
            try:
                eps_c, ef_c, src_c = calcular_dband_en_folder(pdos_dir)
                if ef_c is not None:
                    print(f"   [EF ] {ef_c:.6f} eV  | fuente: {src_c}")
                if eps_c is not None:
                    print(f"   [ε_d] {eps_c:.6f} eV (relativo a E_F)")
                else:
                    print("   [ε_d] No calculable.")
            except Exception as ex:
                print("   [ERROR] ", ex)

    print("\n[DONE] Cálculo de d-band centers y EF completado.")

if __name__ == "__main__":
    main()
