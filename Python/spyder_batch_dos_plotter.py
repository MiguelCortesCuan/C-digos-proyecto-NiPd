# -*- coding: utf-8 -*-
"""
Batch DOS/PDOS plotter (robusto por carpeta) – Quantum ESPRESSO

- Escanea recursivamente una RAÍZ y trata como "caso" CADA carpeta que contenga
  archivos PDOS (p.ej. *pdos_tot, *.pdos_atm#*).
- Para cada caso:
  * Grafica Total DOS (↑,↓) con relleno (↓ en negativo) + PDOS Ni y/o Pd (↓ negativo).
  * Guarda la figura en <PDOS_DIR>/Plots/.
  * Imprime rutas de PDOS y candidatos DOS, y de dónde salió EF.

- Si el caso está bajo un ancestro cuyo nombre termina en "-H", lo marca como +H
  e intenta comparar con su contraparte CLEAN (reemplazando el primer ancestro "-H"
  por su versión sin "-H", manteniendo el subpath).

Títulos:
- “Density of states of Ni(+Pd)[+H] – Top/Bridge/Hollow …” si el path contiene T1/B1/H1/H2
"""

import os, re, sys, glob
from typing import Optional, Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt

# =========================
# === CONFIGURACIÓN =======
# =========================
RAIZ = r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\Supercell\Hydrogen"

# Si existe +H pero NO existe clean, ¿graficar +H “solo”? (False = solo comparaciones)
PLOT_PLUS_H_SIN_CLEAN = True

# Orbitales a incluir por elemento
ORBITALES = ["s", "p", "d"]

# Ventanas (E eje Y; DOS eje X)
ALINEAR_EF = True
EMIN = -6
EMAX = 9
XMIN = None
XMAX = None

# ===== Estilos (exactos) =====
SIGMA = 0.0  # eV (0 = off)
LINEWIDTH = 1.0
ALPHA_LINE = 1.0
GRID = True
MARCAR_EF = True
LEGENDA = True
TITULO_GLOBAL = None

# TOTAL DOS (colores + contorno negro)
COLOR_TOTAL_UP_FILL   = "#1f77b4"  # azul
COLOR_TOTAL_DOWN_FILL = "#d62728"  # rojo
COLOR_TOTAL_LINE      = "black"
TOTAL_UP_LINESTYLE    = "-"
TOTAL_DOWN_LINESTYLE  = "dashdot"
ALPHA_FILL = 0.35

# PDOS Ni y Pd (negro, sin relleno)
COLOR_PDOS_NI_LINE = "black"
COLOR_PDOS_PD_LINE = "black"
NI_UP_LS  = "dashed"
NI_DW_LS  = ":"
PD_UP_LS  = "-"
PD_DW_LS  = (0, (3, 2))

# Línea de Fermi
FERMI_LINE_COLOR = "black"
FERMI_LINE_WIDTH = 0.8
FERMI_LINE_STYLE = "dashed"

# Tamaño imagen
USAR_PIXELES = True
ANCHO_PX = 1200
ALTO_PX  = 1200
ANCHO = 6.0
ALTO  = 8.0
DPI = 160
FORMATO = "png"   # "png" | "pdf" | "svg"

# Fuente
FUENTE_PREFERIDA = "overleaf"  # "overleaf" o "times"
FONT_SIZE_BASE   = 14
TITLE_SIZE       = None
LABEL_SIZE       = None
TICK_SIZE        = None
LEGEND_SIZE      = None

# Etiquetas de sitio
SITE_TOKENS = {"T1": "Top site", "B1": "Bridge site", "H1": "Hollow fcc", "H2": "Hollow hcp"}

# =========================
# === UTILIDADES ==========
# =========================
HEADER_EF_PATTERNS = [
    re.compile(r"\bEFermi\b\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?)", re.I),
    re.compile(r"\bFermi\s+energy\b\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)(?:\s*eV)?", re.I),
    re.compile(r"\bE[_\s-]*Fermi\b\s*[:=]\s*([+-]?\d+(?:\.\d+)?)(?:\s*eV)?", re.I),
]

def configurar_fuente(preferencia: str = "overleaf"):
    if (preferencia or "").strip().lower() == "times":
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        })
    else:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Latin Modern Roman", "CMU Serif",
                           "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        })

def aplicar_tamanos_fuente():
    base = FONT_SIZE_BASE if FONT_SIZE_BASE is not None else 12
    title = TITLE_SIZE if TITLE_SIZE is not None else base * 1.15
    label = LABEL_SIZE if LABEL_SIZE is not None else base * 1.00
    tick  = TICK_SIZE  if TICK_SIZE  is not None else base * 0.90
    leg   = LEGEND_SIZE if LEGEND_SIZE is not None else base * 0.90
    plt.rcParams.update({
        "font.size": base,
        "axes.titlesize": title,
        "axes.labelsize": label,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "legend.fontsize": leg,
    })

def leer_ef_desde_texto(lineas: List[str]) -> Optional[float]:
    for ln in lineas[:1000]:
        s = ln.strip()
        if not s:
            continue
        for pat in HEADER_EF_PATTERNS:
            m = pat.search(s)
            if m:
                try:
                    return float(m.group(1).replace("D","E").replace("d","E"))
                except Exception:
                    pass
    return None

def cargar_tabla_generica(path: str) -> Tuple[np.ndarray, Optional[float]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lineas = f.readlines()
    ef = leer_ef_desde_texto(lineas)
    datos = []
    for ln in lineas:
        s = ln.strip()
        if (not s) or s.startswith("#"):
            continue
        parts = s.split()
        try:
            row = [float(x.replace("D","E")) for x in parts]
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

def crear_figura():
    figsize = (ANCHO_PX / DPI, ALTO_PX / DPI) if USAR_PIXELES else (ANCHO, ALTO)
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    return fig, ax

def listar_elementos_en_dir(pdos_dir: str) -> List[str]:
    elems = set()
    for fp in glob.glob(os.path.join(pdos_dir, "*.pdos_atm#*")):
        m = re.search(r"\((\w+)\)\_wfc#", os.path.basename(fp))
        if m:
            elems.add(m.group(1))
    return sorted(elems)

def listar_pdos_elemento(pdos_dir: str, elemento: str, orbitales: List[str]) -> List[str]:
    files = []
    for orb in orbitales:
        patron = os.path.join(pdos_dir, f"*.pdos_atm#*({elemento})_wfc#*({orb})")
        files.extend(glob.glob(patron))
    return sorted(files)

def sumar_pdos(files: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    E_ref = None
    up_sum = None
    dn_sum = None
    for fp in files:
        arr, _ = cargar_tabla_generica(fp)
        if arr.shape[1] < 3:
            continue
        E = arr[:, 0]
        up = arr[:, 1]
        dn = arr[:, 2]
        if E_ref is None:
            E_ref = E
            up_sum = up.copy()
            dn_sum = dn.copy()
        else:
            if len(E) == len(E_ref) and np.allclose(E, E_ref, atol=1e-8):
                up_sum += up
                dn_sum += dn
            else:
                raise ValueError(f"Mallas de energía distintas: {os.path.basename(fp)}")
    if E_ref is None:
        raise ValueError("No se pudieron sumar PDOS (sin archivos válidos).")
    return E_ref, up_sum, dn_sum

def leer_total_dos(pdos_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    cand = glob.glob(os.path.join(pdos_dir, "*pdos_tot"))
    if not cand:
        return None, None, None, None
    total_path = cand[0]
    arr_total, _ = cargar_tabla_generica(total_path)
    if arr_total.shape[1] < 3:
        raise ValueError("pdos_tot sin columnas up/down suficientes.")
    E_total = arr_total[:, 0]
    up_total = arr_total[:, 1]
    dn_total = arr_total[:, 2]
    return E_total, up_total, dn_total, total_path

def detectar_site_label_en_ruta(path: str) -> Optional[str]:
    base = path.replace("\\", "/")
    for tok, lab in SITE_TOKENS.items():
        if re.search(rf"(?i)(^|[ _/\-\.+]){tok}($|[ _/\-\.+])", base):
            return lab
    return None

def detectar_es_h_y_anchor(pdos_dir: str, raiz: str) -> Tuple[bool, str]:
    """
    Devuelve (es_H, anchor_dir). 'anchor' = ancestro más cercano que "parece" ser
    el nombre del caso (p.ej. 111_2x2-H o 111_2x2). Si no encuentra, usa el
    primer nivel bajo RAIZ.
    """
    abs_raiz = os.path.abspath(raiz)
    cur = os.path.abspath(pdos_dir)
    es_H = False
    anchor = None
    pattern_anchor = re.compile(r"(\d{3}[_\-]?\dx\d(?:\-H)?)$", re.I)  # ej: 111_2x2, 111-2x2, 111_2x2-H
    while True:
        cur = os.path.dirname(cur)
        if len(cur) < len(abs_raiz) or os.path.normcase(cur) == os.path.normcase(abs_raiz):
            break
        base = os.path.basename(cur)
        if base.endswith("-H"):
            es_H = True
        if pattern_anchor.search(base):
            anchor = cur
            break
    if anchor is None:
        # fallback: primer nivel bajo RAIZ
        partes = os.path.relpath(pdos_dir, abs_raiz).split(os.sep)
        if partes:
            anchor = os.path.join(abs_raiz, partes[0])
            if os.path.basename(anchor).endswith("-H"):
                es_H = True
    return es_H, anchor

def construir_clean_counterpart(pdos_dir: str, raiz: str) -> Optional[str]:
    es_H, anchor = detectar_es_h_y_anchor(pdos_dir, raiz)
    if not es_H or not anchor:
        return None
    anchor_base = os.path.basename(anchor)
    if not anchor_base.endswith("-H"):
        return None
    anchor_clean_base = anchor_base[:-2]  # quitar "-H"
    parent = os.path.dirname(anchor)
    # subpath relativo desde anchor -> pdos_dir
    sub_rel = os.path.relpath(pdos_dir, anchor)
    # PDOS del clean en la misma subruta
    candidato = os.path.join(parent, anchor_clean_base, sub_rel)
    # Validar que contiene archivos PDOS
    if glob.glob(os.path.join(candidato, "*pdos_tot")) or glob.glob(os.path.join(candidato, "*.pdos_atm#*")):
        return candidato
    return None

def buscar_dos_cerca(pdos_dir: str, raiz: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Busca una carpeta DOS "cercana" (misma carpeta, padre, abuelo, anchor).
    Devuelve (dos_dir, EF, archivo_del_que_salió_EF) e imprime candidatos probados.
    """
    candidatos = []
    # misma carpeta
    candidatos.append(os.path.join(pdos_dir, "DOS"))
    # padre / abuelo
    padre = os.path.dirname(pdos_dir)
    candidatos.append(os.path.join(padre, "DOS"))
    abuelo = os.path.dirname(padre)
    candidatos.append(os.path.join(abuelo, "DOS"))
    # anchor
    _, anchor = detectar_es_h_y_anchor(pdos_dir, raiz)
    if anchor:
        candidatos.append(os.path.join(anchor, "DOS"))

    print("   [DOS] Candidatos:", *candidatos, sep="\n      ")

    for d in candidatos:
        if os.path.isdir(d):
            # busca EF en varios archivos
            patt = ["*dos*", "*.out", "*.dat", "*proj*dos*", "*pdos*", "*.txt"]
            archivos = []
            for p in patt:
                archivos.extend(glob.glob(os.path.join(d, p)))
            for fp in sorted(set(archivos)):
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        ef = leer_ef_desde_texto(f.readlines())
                    if ef is not None:
                        print(f"   [DOS] EF={ef:.6f} encontrado en: {fp}")
                        return d, ef, fp
                except Exception:
                    pass
    print("   [DOS] No se encontró EF en candidatos.")
    return None, None, None

def prep_curva(E: np.ndarray, y: np.ndarray, ef: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    if ALINEAR_EF and (ef is not None):
        E = E - ef
    y = gauss_suavizado(y, E, SIGMA)
    return E, y

def plot_vertical(ax, E_up, up, E_dn, dn, etiqueta_up=None, etiqueta_dn=None,
                  color_line="black", ls_up="-", ls_dn="dashdot",
                  fill_up=None, fill_dn=None, alpha_fill=0.35, lw=LINEWIDTH, alpha_line=ALPHA_LINE):
    # ↓ en negativo (Total y PDOS)
    if (fill_up is not None) and (E_up is not None) and (up is not None):
        ax.fill_betweenx(E_up, 0,  up, alpha=alpha_fill, color=fill_up)
    if (fill_dn is not None) and (E_dn is not None) and (dn is not None):
        ax.fill_betweenx(E_dn, 0, -dn, alpha=alpha_fill, color=fill_dn)
    if (E_up is not None) and (up is not None):
        ax.plot(up,  E_up, color=color_line, linestyle=ls_up, linewidth=lw,
                alpha=alpha_line, label=etiqueta_up)
    if (E_dn is not None) and (dn is not None):
        ax.plot(-dn, E_dn, color=color_line, linestyle=ls_dn, linewidth=lw,
                alpha=alpha_line, label=etiqueta_dn)

def titulo_auto(elementos: List[str], con_H: bool, path: str) -> str:
    cuerpo = "DOS" if not elementos else "+".join(elementos)
    base = f"Density of states of {cuerpo}{'+H' if con_H else ''}"
    site = detectar_site_label_en_ruta(path)
    if site:
        base += f" – {site}"
    return base

# =========================
# === GRAFICADORES ========
# =========================
def grafica_simple_en_dir(pdos_dir: str, ef: Optional[float]):
    print(f"[CASE] PDOS dir: {pdos_dir}")

    E_t, up_t, dn_t, tot_path = leer_total_dos(pdos_dir)
    if tot_path:
        print(f"   [TOTAL] pdos_tot: {tot_path}")
    else:
        print("   [TOTAL] *pdos_tot NO encontrado (se graficará solo PDOS elemental).")

    elems = [e for e in listar_elementos_en_dir(pdos_dir) if e in ("Ni", "Pd")]
    print(f"   [PDOS] Elementos detectados: {', '.join(elems) if elems else '(ninguno)'}")

    # EF (busca cerca e imprime candidatos)
    dos_dir, ef_auto, ef_file = buscar_dos_cerca(pdos_dir, RAIZ)
    ef_use = ef if ef is not None else ef_auto

    fig, ax = crear_figura()
    es_H, _ = detectar_es_h_y_anchor(pdos_dir, RAIZ)
    title = TITULO_GLOBAL if TITULO_GLOBAL else titulo_auto(elems, es_H, pdos_dir)
    ax.set_title(title)
    ax.set_xlabel("DOS (States/eV)")
    ax.set_ylabel(r"$E - E_{\mathrm{Fermi}}$ (eV)" if (ALINEAR_EF and ef_use is not None) else "Energy (eV)")
    if (EMIN is not None) or (EMAX is not None): ax.set_ylim(EMIN, EMAX)
    if (XMIN is not None) or (XMAX is not None):
        ax.set_xlim(XMIN if XMIN is not None else ax.get_xlim()[0],
                    XMAX if XMAX is not None else ax.get_xlim()[1])
    if GRID: ax.grid(True, linestyle=":")

    # Total
    if E_t is not None:
        Et_u, up_t_p = prep_curva(E_t.copy(), up_t.copy(), ef_use)
        Et_d, dn_t_p = prep_curva(E_t.copy(), dn_t.copy(), ef_use)
        plot_vertical(ax, Et_u, up_t_p, Et_d, dn_t_p,
                      etiqueta_up="Total DOS (↑)" if LEGENDA else None,
                      etiqueta_dn="Total DOS (↓)" if LEGENDA else None,
                      color_line=COLOR_TOTAL_LINE,
                      ls_up=TOTAL_UP_LINESTYLE, ls_dn=TOTAL_DOWN_LINESTYLE,
                      fill_up=COLOR_TOTAL_UP_FILL, fill_dn=COLOR_TOTAL_DOWN_FILL,
                      alpha_fill=ALPHA_FILL)

    # PDOS Ni/Pd
    for el in elems:
        files_el = listar_pdos_elemento(pdos_dir, el, ORBITALES)
        print(f"   [PDOS {el}] archivos: {len(files_el)}")
        if not files_el: continue
        E_e, up_e, dn_e = sumar_pdos(files_el)
        Ee_u, up_e_p = prep_curva(E_e.copy(), up_e.copy(), ef_use)
        Ee_d, dn_e_p = prep_curva(E_e.copy(), dn_e.copy(), ef_use)
        if el == "Ni":
            plot_vertical(ax, Ee_u, up_e_p, Ee_d, dn_e_p,
                          etiqueta_up=f"Ni (↑)" if LEGENDA else None,
                          etiqueta_dn=f"Ni (↓)" if LEGENDA else None,
                          color_line=COLOR_PDOS_NI_LINE, ls_up=NI_UP_LS, ls_dn=NI_DW_LS,
                          fill_up=None, fill_dn=None, alpha_fill=0.0, lw=LINEWIDTH*1.0)
        elif el == "Pd":
            plot_vertical(ax, Ee_u, up_e_p, Ee_d, dn_e_p,
                          etiqueta_up=f"Pd (↑)" if LEGENDA else None,
                          etiqueta_dn=f"Pd (↓)" if LEGENDA else None,
                          color_line=COLOR_PDOS_PD_LINE, ls_up=PD_UP_LS, ls_dn=PD_DW_LS,
                          fill_up=None, fill_dn=None, alpha_fill=0.0, lw=LINEWIDTH*0.95)

    if MARCAR_EF and (ef_use is not None):
        ax.axhline(0.0 if ALINEAR_EF else ef_use, lw=FERMI_LINE_WIDTH,
                   ls=FERMI_LINE_STYLE, color=FERMI_LINE_COLOR, alpha=1.0)

    if LEGENDA: ax.legend(loc="best")
    fig.tight_layout()
    outdir = os.path.join(pdos_dir, "Plots")
    os.makedirs(outdir, exist_ok=True)
    nombre = os.path.basename(os.path.normpath(pdos_dir))
    outpath = os.path.join(outdir, f"{nombre}_DOS_PDOS.{FORMATO}")
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)
    print(f"   [OK] Guardado: {outpath}")

def grafica_comparacion_en_dir(pdos_H_dir: str, pdos_clean_dir: str):
    print(f"[COMP] +H PDOS: {pdos_H_dir}")
    print(f"[COMP] CLEAN PDOS: {pdos_clean_dir}")

    # EF de cada uno (independientes)
    _, efC, _ = buscar_dos_cerca(pdos_clean_dir, RAIZ)
    _, efH, _ = buscar_dos_cerca(pdos_H_dir, RAIZ)

    # Total DOS
    E_t_C, up_t_C, dn_t_C, totC = leer_total_dos(pdos_clean_dir)
    E_t_H, up_t_H, dn_t_H, totH = leer_total_dos(pdos_H_dir)
    if totC: print(f"   [TOTAL clean] {totC}")
    if totH: print(f"   [TOTAL +H]    {totH}")

    # Elementos (unión)
    elems = sorted(set([e for e in listar_elementos_en_dir(pdos_clean_dir) if e in ("Ni", "Pd")]) |
                   set([e for e in listar_elementos_en_dir(pdos_H_dir) if e in ("Ni", "Pd")]))
    print(f"   [PDOS] Elementos: {', '.join(elems) if elems else '(ninguno)'}")

    fig, ax = crear_figura()
    title = TITULO_GLOBAL if TITULO_GLOBAL else titulo_auto(elems, True, pdos_H_dir)
    ax.set_title(title)
    ax.set_xlabel("DOS (States/eV)")
    ax.set_ylabel(r"$E - E_{\mathrm{Fermi}}$ (eV)")
    if (EMIN is not None) or (EMAX is not None): ax.set_ylim(EMIN, EMAX)
    if (XMIN is not None) or (XMAX is not None):
        ax.set_xlim(XMIN if XMIN is not None else ax.get_xlim()[0],
                    XMAX if XMAX is not None else ax.get_xlim()[1])
    if GRID: ax.grid(True, linestyle=":")

    # Clean tenue
    if E_t_C is not None:
        EtC_u, up_tC_p = prep_curva(E_t_C.copy(), up_t_C.copy(), efC)
        EtC_d, dn_tC_p = prep_curva(E_t_C.copy(), dn_t_C.copy(), efC)
        plot_vertical(ax, EtC_u, up_tC_p, EtC_d, dn_tC_p,
                      etiqueta_up="Total DOS (↑, clean)" if LEGENDA else None,
                      etiqueta_dn="Total DOS (↓, clean)" if LEGENDA else None,
                      color_line=COLOR_TOTAL_LINE, ls_up=TOTAL_UP_LINESTYLE, ls_dn=TOTAL_DOWN_LINESTYLE,
                      fill_up=COLOR_TOTAL_UP_FILL, fill_dn=COLOR_TOTAL_DOWN_FILL,
                      alpha_fill=max(0.15, ALPHA_FILL*0.5), lw=LINEWIDTH*0.95, alpha_line=0.7)

    # +H normal
    if E_t_H is not None:
        EtH_u, up_tH_p = prep_curva(E_t_H.copy(), up_t_H.copy(), efH)
        EtH_d, dn_tH_p = prep_curva(E_t_H.copy(), dn_t_H.copy(), efH)
        plot_vertical(ax, EtH_u, up_tH_p, EtH_d, dn_tH_p,
                      etiqueta_up="Total DOS (↑, +H)" if LEGENDA else None,
                      etiqueta_dn="Total DOS (↓, +H)" if LEGENDA else None,
                      color_line=COLOR_TOTAL_LINE, ls_up=TOTAL_UP_LINESTYLE, ls_dn=TOTAL_DOWN_LINESTYLE,
                      fill_up=COLOR_TOTAL_UP_FILL, fill_dn=COLOR_TOTAL_DOWN_FILL,
                      alpha_fill=ALPHA_FILL, lw=LINEWIDTH*1.05, alpha_line=1.0)

    # PDOS por elemento
    for el in elems:
        # clean
        files_C = listar_pdos_elemento(pdos_clean_dir, el, ORBITALES)
        if files_C:
            E_e_C, up_e_C, dn_e_C = sumar_pdos(files_C)
            EeC_u, up_eC_p = prep_curva(E_e_C.copy(), up_e_C.copy(), efC)
            EeC_d, dn_eC_p = prep_curva(E_e_C.copy(), dn_e_C.copy(), efC)
        else:
            EeC_u = EeC_d = up_eC_p = dn_eC_p = None
        # +H
        files_H = listar_pdos_elemento(pdos_H_dir, el, ORBITALES)
        if files_H:
            E_e_H, up_e_H, dn_e_H = sumar_pdos(files_H)
            EeH_u, up_eH_p = prep_curva(E_e_H.copy(), up_e_H.copy(), efH)
            EeH_d, dn_eH_p = prep_curva(E_e_H.copy(), dn_e_H.copy(), efH)
        else:
            EeH_u = EeH_d = up_eH_p = dn_eH_p = None

        if el == "Ni":
            plot_vertical(ax, EeC_u, up_eC_p, EeC_d, dn_eC_p,
                          etiqueta_up=f"Ni (↑, clean)" if LEGENDA else None,
                          etiqueta_dn=f"Ni (↓, clean)" if LEGENDA else None,
                          color_line=COLOR_PDOS_NI_LINE, ls_up=NI_UP_LS, ls_dn=NI_DW_LS,
                          lw=LINEWIDTH*0.95, alpha_line=0.6)
            plot_vertical(ax, EeH_u, up_eH_p, EeH_d, dn_eH_p,
                          etiqueta_up=f"Ni (↑, +H)" if LEGENDA else None,
                          etiqueta_dn=f"Ni (↓, +H)" if LEGENDA else None,
                          color_line=COLOR_PDOS_NI_LINE, ls_up=NI_UP_LS, ls_dn=NI_DW_LS,
                          lw=LINEWIDTH*1.05, alpha_line=1.0)
        elif el == "Pd":
            plot_vertical(ax, EeC_u, up_eC_p, EeC_d, dn_eC_p,
                          etiqueta_up=f"Pd (↑, clean)" if LEGENDA else None,
                          etiqueta_dn=f"Pd (↓, clean)" if LEGENDA else None,
                          color_line=COLOR_PDOS_PD_LINE, ls_up=PD_UP_LS, ls_dn=PD_DW_LS,
                          lw=LINEWIDTH*0.95, alpha_line=0.6)
            plot_vertical(ax, EeH_u, up_eH_p, EeH_d, dn_eH_p,
                          etiqueta_up=f"Pd (↑, +H)" if LEGENDA else None,
                          etiqueta_dn=f"Pd (↓, +H)" if LEGENDA else None,
                          color_line=COLOR_PDOS_PD_LINE, ls_up=PD_UP_LS, ls_dn=PD_DW_LS,
                          lw=LINEWIDTH*1.05, alpha_line=1.0)

    ax.axhline(0.0, lw=FERMI_LINE_WIDTH, ls=FERMI_LINE_STYLE, color=FERMI_LINE_COLOR, alpha=1.0)
    if LEGENDA: ax.legend(loc="best")
    fig.tight_layout()
    outdir = os.path.join(pdos_H_dir, "Plots")  # exporta en el PDOS de +H
    os.makedirs(outdir, exist_ok=True)
    nombre = os.path.basename(os.path.normpath(pdos_H_dir))
    outpath = os.path.join(outdir, f"{nombre}_DOS_PDOS_vs_clean.{FORMATO}")
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)
    print(f"   [OK] Guardado: {outpath}")

# =========================
# === ESCANEO RAÍZ ========
# =========================
def recolectar_pdos_dirs(raiz: str) -> List[str]:
    """
    Devuelve todas las carpetas que contienen archivos PDOS.
    (no requiere que la carpeta se llame 'PDOS')
    """
    pdos_dirs = set()
    for dirpath, dirnames, filenames in os.walk(raiz):
        # ¿hay archivos PDOS aquí?
        if glob.glob(os.path.join(dirpath, "*pdos_tot")) or glob.glob(os.path.join(dirpath, "*.pdos_atm#*")):
            pdos_dirs.add(os.path.abspath(dirpath))
    return sorted(pdos_dirs)

# =========================
# === MAIN ===============
# =========================
if __name__ == "__main__":
    # RAIZ por argv opcional
    if len(sys.argv) >= 2:
        RAIZ = sys.argv[1]

    print(f"[INFO] RAIZ = {RAIZ}")
    if not os.path.isdir(RAIZ):
        raise RuntimeError(f"No existe RAIZ: {RAIZ}")

    # Fuente y tamaños
    if (FUENTE_PREFERIDA or "").strip().lower() not in ("overleaf", "times"):
        FUENTE_PREFERIDA = "overleaf"
    configurar_fuente(FUENTE_PREFERIDA)
    aplicar_tamanos_fuente()

    # Recolectar todas las carpetas con PDOS
    pdos_dirs = recolectar_pdos_dirs(RAIZ)
    if not pdos_dirs:
        print("[WARN] No se hallaron archivos PDOS bajo la RAIZ.")
        # imprime una muestra de subdirectorios para depurar
        sub = [p for p in glob.glob(os.path.join(RAIZ, "**"), recursive=True) if os.path.isdir(p)]
        print(f"[INFO] Subdirectorios encontrados (muestra 10):")
        for s in sub[:10]:
            print("   ", s)
        sys.exit(0)

    print(f"[INFO] Carpetas con PDOS encontradas: {len(pdos_dirs)}")
    for p in pdos_dirs[:10]:
        print("   ", p)

    # Mapear +H → clean cuando sea posible
    procesado = set()
    for pdos_dir in pdos_dirs:
        if pdos_dir in procesado:
            continue
        es_H, _ = detectar_es_h_y_anchor(pdos_dir, RAIZ)
        if es_H:
            # Intentar contraparte clean
            pdos_clean = construir_clean_counterpart(pdos_dir, RAIZ)
            if pdos_clean and pdos_clean in pdos_dirs:
                try:
                    grafica_comparacion_en_dir(pdos_dir, pdos_clean)
                except Exception as e:
                    print(f"[ERROR] Comparación {pdos_dir} vs {pdos_clean}: {e}")
                procesado.add(pdos_dir)
                procesado.add(pdos_clean)
            else:
                if PLOT_PLUS_H_SIN_CLEAN:
                    try:
                        grafica_simple_en_dir(pdos_dir, ef=None)
                    except Exception as e:
                        print(f"[ERROR] +H solo {pdos_dir}: {e}")
                else:
                    print(f"[SKIP] {pdos_dir}: no se encontró contraparte CLEAN.")
                    procesado.add(pdos_dir)
        else:
            # Clean individual
            try:
                grafica_simple_en_dir(pdos_dir, ef=None)
            except Exception as e:
                print(f"[ERROR] Clean {pdos_dir}: {e}")
            procesado.add(pdos_dir)

    print("[DONE] Procesamiento completo.")
