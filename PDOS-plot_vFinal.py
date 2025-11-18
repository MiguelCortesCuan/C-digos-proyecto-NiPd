# -*- coding: utf-8 -*-
r"""
Batch DOS/PDOS plotter para Quantum ESPRESSO
- Escanea bajo RAIZ todas las carpetas ...\PDOS\<leaf> y grafica:
  * Caso limpio: Total DOS (↑/↓) con área bajo la curva y PDOS del elemento.
  * Pares +H vs CLEAN: compara Total y PDOS (↑/↓) para +H contra clean.
- Alinea en EF (opcional), invierte spin ↓ (se grafica negativo), y guarda en ...\PDOS\<leaf>\Plots.
- Leyenda “compuesta” para Total: misma línea + bloque debajo (alpha=ALPHA_FILL), con ~5 px de separación blanca y SIN borde.
"""

import os
import re
import glob
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase

# =========================
# === CONFIGURACIÓN =======
# =========================
RAIZ = r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\Supercell\Hydrogen"

# Alineación Fermi
ALINEAR_EF = True
EF_MANUAL: Optional[float] = None  # si no es None, domina

# Ventanas (E en Y)
EMIN = -6
EMAX = 9
XMIN = None
XMAX = None

# Suavizado gaussiano (eV); 0 = off
SIGMA = 0.0

# Estilo general
LINEWIDTH = 1.0
ALPHA_LINE = 1.0
GRID = True
MARCAR_EF = True
LEGENDA = True

# ===== Estilos gráficos =====
COLOR_TOTAL_UP_FILL   = "#1f77b4"   # azul
COLOR_TOTAL_DOWN_FILL = "#d62728"   # rojo
COLOR_TOTAL_LINE      = "black"     # contorno negro
TOTAL_UP_LINESTYLE    = "-"         # sólido
TOTAL_DOWN_LINESTYLE  = "dashdot"   # punteado
ALPHA_FILL            = 0.35        # alpha del área bajo curva (y de la leyenda compuesta)

# PDOS en negro
COLOR_PDOS_UP_LINE    = "black"
COLOR_PDOS_DOWN_LINE  = "black"
PDOS_UP_LINESTYLE     = "dashed"
PDOS_DOWN_LINESTYLE   = ":"

# Línea de Fermi
FERMI_LINE_COLOR = "black"
FERMI_LINE_WIDTH = 0.8
FERMI_LINE_STYLE = "dashed"

# Tamaño imagen
USAR_PIXELES = True
ANCHO_PX = 1200
ALTO_PX  = 1200
ANCHO = 6.0    # pulgadas si USAR_PIXELES = False
ALTO  = 8.0
DPI = 160
FORMATO = "png"   # "png" | "pdf" | "svg"

# Fuente (subida a 24)
FUENTE_PREFERIDA = "overleaf"  # "overleaf" o "times"
FONT_SIZE_BASE   = 20          # <<<<<< tamaño base pedido
TITLE_SIZE       = None
LABEL_SIZE       = None
TICK_SIZE        = None
LEGEND_SIZE      = None

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

def crear_figura():
    if USAR_PIXELES:
        figsize = (ANCHO_PX / DPI, ALTO_PX / DPI)
    else:
        figsize = (ANCHO, ALTO)
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    return fig, ax

def encontrar_total_pdos(carpeta: str) -> Optional[str]:
    candidatos = glob.glob(os.path.join(carpeta, "*pdos_tot"))
    return candidatos[0] if candidatos else None

def listar_pdos_elemento(carpeta: str, elemento: str) -> List[str]:
    patron = os.path.join(carpeta, f"*.pdos_atm#*({elemento})_wfc#*")
    return sorted(glob.glob(patron))

def sumar_pdos(files: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
    E_ref = None
    up_sum = None
    dn_sum = None
    ef_global = None
    for fp in files:
        arr, ef = cargar_tabla_generica(fp)
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
        if (ef_global is None) and (ef is not None):
            ef_global = ef
    if E_ref is None:
        raise ValueError("No se pudieron sumar PDOS (sin archivos válidos).")
    return E_ref, up_sum, dn_sum, ef_global

def parse_elemento_y_site(leaf: str) -> Tuple[str, Optional[str]]:
    parts = leaf.split("_")
    elem = re.match(r"^[A-Za-z0-9]+", parts[0]).group(0) if parts else leaf  # -> "Ni" (sin "(111)")
    site = None
    if len(parts) >= 3:
        cand = parts[-1].upper()
        if cand in SITE_MAP:
            site = cand
    return elem, site

def construir_clean_counterpart(pdos_dir: str, raiz: str) -> Optional[str]:
    r"""Mapea ...\111_2x2-H\PDOS\Ni_111_T1 -> ...\111_2x2\PDOS\Ni_111 (tu estructura real)."""
    p = os.path.normpath(pdos_dir)
    if "-H" not in p:
        return None

    # Carpeta del caso +H (por ejemplo "111_2x2-H")
    case_h_dir = os.path.basename(os.path.dirname(os.path.dirname(p)))
    # Quitar el sufijo "-H" → "111_2x2"
    clean_case_dir = case_h_dir.replace("-H", "")

    # Leaf (ej. "Ni_111_T1") y su base limpia sin el sufijo de sitio → "Ni_111"
    leaf = os.path.basename(p)
    base_leaf = re.sub(r"_(T1|B1|H1|H2)$", "", leaf, flags=re.IGNORECASE)

    # Construir la ruta CLEAN con la RAIZ correcta que ya definiste
    candidate = os.path.join(os.path.normpath(raiz), clean_case_dir, "PDOS", base_leaf)
    return candidate if os.path.isdir(candidate) else None

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

def leer_ef_desde_dos(dos_dir: str, leaf: str) -> Tuple[Optional[float], Optional[str]]:
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

def preparar_series(E: np.ndarray, up: np.ndarray, dn: np.ndarray, ef: Optional[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ALINEAR_EF and (ef is not None):
        E = E - ef
    up = gauss_suavizado(up, E, SIGMA)
    dn = gauss_suavizado(dn, E, SIGMA)
    return E, up, dn

def facet_from_leaf(leaf: str) -> Optional[str]:
    # Siempre ocultar la faceta en el título
    return None

def titulo_para(elemento: str, es_h: bool, site: Optional[str], facet: Optional[str]) -> str:
    sysname = f"{elemento}{' +H' if es_h else ''}"
    sitio = f", {SITE_MAP.get(site, site)}" if (es_h and site) else ""
    facet_txt = f" ({facet})" if facet else ""
    return f"Density of states of {sysname}{facet_txt}{sitio}"

# =========================
# === LEYENDA COMPUESTA ===
# =========================
# Forzar línea continua en la leyenda compuesta (como pediste)
COMPOSED_LEGEND_FORCE_SOLID = True
# Altura relativa del bloque dentro del handle
LEG_RECT_FRAC = 0.60
# Aproximar separación blanca de ~5 px entre bloque y línea
LEG_LINE_GAP_PX = 5.0  # se aproxima internamente

class FillLineLegend:
    """Proxy para una muestra de leyenda con BLOQUE (alpha=ALPHA_FILL) + LÍNEA encima."""
    def __init__(self, fill_color, line_color, linestyle="-", linewidth=1.0, alpha=0.35):
        self.fill_color = fill_color
        self.line_color = line_color
        self.linestyle  = linestyle
        self.linewidth  = linewidth
        self.alpha      = alpha

class HandlerFillLine(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        # bloque inferior
        rect_h = height * LEG_RECT_FRAC
        rect = mpatches.Rectangle(
            (x0, y0), width, rect_h,
            transform=trans, facecolor=orig_handle.fill_color,
            edgecolor="none", alpha=orig_handle.alpha
        )
        # “~5 px” de separación blanca (aprox). El parámetro height viene en puntos,
        # así que convertimos px -> puntos asumiendo 72 pt/in y dpi de la figura:
        fig = legend.axes.figure
        dpi = fig.dpi if fig is not None else 72.0
        gap_pts = (LEG_LINE_GAP_PX / dpi) * 72.0
        # Si por alguna razón es demasiado, lo limitamos a un 25% del espacio libre
        gap_pts = min(gap_pts, (height - rect_h) * 0.25)

        # línea centrada por encima del bloque + gap
        y_line = y0 + rect_h + gap_pts + (height - rect_h) * 0.35
        line = plt.Line2D(
            [x0, x0 + width], [y_line, y_line],
            transform=trans, color=orig_handle.line_color,
            linestyle="-" if COMPOSED_LEGEND_FORCE_SOLID else orig_handle.linestyle,
            linewidth=orig_handle.linewidth
        )
        return [rect, line]

def _aplicar_leyenda_solo_clean(ax, elemento: str):
    for l in ax.lines:
        l.set_label("_nolegend_")

    h_total_up   = FillLineLegend(
        fill_color=COLOR_TOTAL_UP_FILL, line_color=COLOR_TOTAL_LINE,
        linestyle=TOTAL_UP_LINESTYLE, linewidth=LINEWIDTH, alpha=ALPHA_FILL
    )
    h_total_down = FillLineLegend(
        fill_color=COLOR_TOTAL_DOWN_FILL, line_color=COLOR_TOTAL_LINE,
        linestyle=TOTAL_DOWN_LINESTYLE, linewidth=LINEWIDTH, alpha=ALPHA_FILL
    )
    h_pdos_up   = plt.Line2D([], [], color=COLOR_PDOS_UP_LINE,   ls=PDOS_UP_LINESTYLE,   lw=LINEWIDTH)
    h_pdos_down = plt.Line2D([], [], color=COLOR_PDOS_DOWN_LINE, ls=PDOS_DOWN_LINESTYLE, lw=LINEWIDTH)

    handles = [h_total_up, h_total_down, h_pdos_up, h_pdos_down]
    labels  = ["Total DOS (↑)", "Total DOS (↓)", f"{elemento} (↑)", f"{elemento} (↓)"]

    ax.legend(
        handles=handles, labels=labels, loc="best", frameon=True, handlelength=2.0, borderpad=0.6,
        handler_map={FillLineLegend: HandlerFillLine()}
    )

def _aplicar_leyenda_h_vs_clean(ax, elemento: str):
    for l in ax.lines:
        l.set_label("_nolegend_")

    h_total_up_h = FillLineLegend(
        fill_color=COLOR_TOTAL_UP_FILL, line_color=COLOR_TOTAL_LINE,
        linestyle=TOTAL_UP_LINESTYLE, linewidth=LINEWIDTH, alpha=ALPHA_FILL
    )
    h_total_dn_h = FillLineLegend(
        fill_color=COLOR_TOTAL_DOWN_FILL, line_color=COLOR_TOTAL_LINE,
        linestyle=TOTAL_DOWN_LINESTYLE, linewidth=LINEWIDTH, alpha=ALPHA_FILL
    )
    h_total_up_c = plt.Line2D([], [], color=COLOR_TOTAL_LINE, ls="-.", lw=LINEWIDTH)
    h_total_dn_c = plt.Line2D([], [], color=COLOR_TOTAL_LINE, ls=":",  lw=LINEWIDTH)
    h_pdos_up_h  = plt.Line2D([], [], color=COLOR_PDOS_UP_LINE,   ls=PDOS_UP_LINESTYLE,   lw=LINEWIDTH)
    h_pdos_dn_h  = plt.Line2D([], [], color=COLOR_PDOS_DOWN_LINE, ls=PDOS_DOWN_LINESTYLE, lw=LINEWIDTH)
    h_pdos_up_c  = plt.Line2D([], [], color="gray", ls="--", lw=LINEWIDTH)
    h_pdos_dn_c  = plt.Line2D([], [], color="gray", ls=":",  lw=LINEWIDTH)

    handles = [
        h_total_up_h, h_total_dn_h,
        h_total_up_c, h_total_dn_c,
        h_pdos_up_h,  h_pdos_dn_h,
        h_pdos_up_c,  h_pdos_dn_c,
    ]
    labels  = [
        "Total +H (↑)", "Total +H (↓)",
        "Total clean (↑)", "Total clean (↓)",
        f"{elemento}+H (↑)", f"{elemento}+H (↓)",
        f"{elemento} clean (↑)", f"{elemento} clean (↓)",
    ]

    ax.legend(
        handles=handles, labels=labels, loc="best", frameon=True, handlelength=2.0, borderpad=0.6,
        handler_map={FillLineLegend: HandlerFillLine()}
    )

# =========================
# === PLOTTING ============
# =========================
def plot_solo_clean(pdos_dir: str, elemento: str, facet: Optional[str]) -> Optional[str]:
    print(f"[CASE] PDOS dir: {pdos_dir}")
    total_path = encontrar_total_pdos(pdos_dir)
    if total_path is None:
        print("   [ERROR] No se encontró '*pdos_tot'.")
        return None
    print(f"   [TOTAL] pdos_tot: {total_path}")

    files_elem = listar_pdos_elemento(pdos_dir, elemento)
    if not files_elem:
        print(f"   [ERROR] No se encontraron pdos_atm para {elemento}.")
        return None
    print(f"   [PDOS {elemento}] archivos: {len(files_elem)}")

    cand_dos = dos_candidatos_para(pdos_dir)
    if cand_dos:
        print("   [DOS] Candidatos:")
        for c in cand_dos:
            print(f"      {c}")

    ef_dos = None
    dos_eligio = None
    leaf = os.path.basename(pdos_dir)
    for d in cand_dos:
        ef_dos, dos_eligio = leer_ef_desde_dos(d, leaf=leaf)
        if ef_dos is not None:
            print(f"   [DOS] EF={ef_dos:.6f} encontrado en: {dos_eligio}")
            break

    arr_tot, ef_tot = cargar_tabla_generica(total_path)
    if arr_tot.shape[1] < 3:
        print("   [ERROR] pdos_tot sin columnas up/down suficientes.")
        return None
    E_t, up_t, dn_t = arr_tot[:,0], arr_tot[:,1], arr_tot[:,2]
    E_e, up_e, dn_e, ef_e = sumar_pdos(files_elem)

    ef = EF_MANUAL if (EF_MANUAL is not None) else (ef_dos if ef_dos is not None else (ef_tot if ef_tot is not None else ef_e))

    Et, up_t, dn_t = preparar_series(E_t, up_t, dn_t, ef)
    Ee, up_e, dn_e = preparar_series(E_e, up_e, dn_e, ef)

    fig, ax = crear_figura()
    ax.set_title(titulo_para(elemento, es_h=False, site=None, facet=facet))
    ax.set_xlabel("DOS (States/eV)")
    ax.set_ylabel(r"$E - E_{\mathrm{Fermi}}$ (eV)" if (ALINEAR_EF and ef is not None) else "Energía (eV)")

    if (EMIN is not None) or (EMAX is not None):
        ax.set_ylim(EMIN, EMAX)
    if (XMIN is not None) or (XMAX is not None):
        if (XMIN is not None) and (XMAX is not None):
            ax.set_xlim(XMIN, XMAX)
        elif XMIN is not None:
            ax.set_xlim(left=XMIN)
        elif XMAX is not None:
            ax.set_xlim(right=XMAX)

    # Total DOS (↓ negativo) con relleno
    ax.fill_betweenx(Et, 0,  up_t, alpha=ALPHA_FILL, color=COLOR_TOTAL_UP_FILL,   label=None)
    ax.fill_betweenx(Et, 0, -dn_t, alpha=ALPHA_FILL, color=COLOR_TOTAL_DOWN_FILL, label=None)
    ax.plot(up_t,  Et, color=COLOR_TOTAL_LINE, linestyle=TOTAL_UP_LINESTYLE,   linewidth=LINEWIDTH, alpha=ALPHA_LINE)
    ax.plot(-dn_t, Et, color=COLOR_TOTAL_LINE, linestyle=TOTAL_DOWN_LINESTYLE, linewidth=LINEWIDTH, alpha=ALPHA_LINE)

    # PDOS elemento (↓ negativo)
    ax.plot(up_e,  Ee, color=COLOR_PDOS_UP_LINE,   linestyle=PDOS_UP_LINESTYLE, linewidth=LINEWIDTH, alpha=ALPHA_LINE)
    ax.plot(-dn_e, Ee, color=COLOR_PDOS_DOWN_LINE, linestyle=PDOS_DOWN_LINESTYLE, linewidth=LINEWIDTH, alpha=ALPHA_LINE)

    if GRID:
        ax.grid(True, linestyle=":")
    if MARCAR_EF:
        if ALINEAR_EF and (ef is not None):
            ax.axhline(0.0, lw=FERMI_LINE_WIDTH, ls=FERMI_LINE_STYLE, color=FERMI_LINE_COLOR, alpha=1.0)
        elif ef is not None:
            ax.axhline(ef, lw=FERMI_LINE_WIDTH, ls=FERMI_LINE_STYLE, color=FERMI_LINE_COLOR, alpha=1.0)
    if LEGENDA:
        _aplicar_leyenda_solo_clean(ax, elemento)

    fig.tight_layout()
    out_dir = os.path.join(pdos_dir, "Plots")
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, f"{os.path.basename(pdos_dir)}_DOS_PDOS.{FORMATO}")
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)
    print(f"   [OK] Guardado: {outpath}")
    return outpath

def plot_h_vs_clean(pdos_dir_h: str, pdos_dir_clean: str, elemento: str, site: Optional[str], facet: Optional[str]) -> Optional[str]:
    print(f"[CASE-H] PDOS dir (+H): {pdos_dir_h}")
    print(f"   [PAIR] Clean dir:     {pdos_dir_clean}")

    total_h = encontrar_total_pdos(pdos_dir_h)
    total_c = encontrar_total_pdos(pdos_dir_clean)
    if not total_h or not total_c:
        print("   [ERROR] Falta pdos_tot en +H o CLEAN.")
        return None
    print(f"   [TOTAL +H] {total_h}")
    print(f"   [TOTAL C ] {total_c}")

    f_h = listar_pdos_elemento(pdos_dir_h, elemento)
    f_c = listar_pdos_elemento(pdos_dir_clean, elemento)
    if (not f_h) or (not f_c):
        print(f"   [ERROR] No hay pdos_atm para {elemento} en +H o CLEAN.")
        return None
    print(f"   [PDOS {elemento} +H] archivos: {len(f_h)}")
    print(f"   [PDOS {elemento} C ] archivos: {len(f_c)}")

    cand_h = dos_candidatos_para(pdos_dir_h)
    cand_c = dos_candidatos_para(pdos_dir_clean)
    if cand_h:
        print("   [DOS +H] Candidatos:")
        for c in cand_h: print(f"      {c}")
    if cand_c:
        print("   [DOS C ] Candidatos:")
        for c in cand_c: print(f"      {c}")

    leaf_h = os.path.basename(pdos_dir_h)
    leaf_c = os.path.basename(pdos_dir_clean)

    ef_h = ef_c = None
    src_h = src_c = None
    for d in cand_h:
        ef_h, src_h = leer_ef_desde_dos(d, leaf=leaf_h)
        if ef_h is not None:
            print(f"   [DOS] EF(+H)={ef_h:.6f} encontrado en: {src_h}")
            break
    for d in cand_c:
        ef_c, src_c = leer_ef_desde_dos(d, leaf=leaf_c)
        if ef_c is not None:
            print(f"   [DOS] EF(C )={ef_c:.6f} encontrado en: {src_c}")
            break

    arr_th, ef_th = cargar_tabla_generica(total_h)
    arr_tc, ef_tc = cargar_tabla_generica(total_c)
    Et_h, up_t_h, dn_t_h = arr_th[:,0], arr_th[:,1], arr_th[:,2]
    Et_c, up_t_c, dn_t_c = arr_tc[:,0], arr_tc[:,1], arr_tc[:,2]

    Eh, up_e_h, dn_e_h, ef_eh = sumar_pdos(f_h)
    Ec, up_e_c, dn_e_c, ef_ec = sumar_pdos(f_c)

    ef_final_h = EF_MANUAL if (EF_MANUAL is not None) else (ef_h if ef_h is not None else (ef_th if ef_th is not None else ef_eh))
    ef_final_c = EF_MANUAL if (EF_MANUAL is not None) else (ef_c if ef_c is not None else (ef_tc if ef_tc is not None else ef_ec))

    Et_h, up_t_h, dn_t_h = preparar_series(Et_h, up_t_h, dn_t_h, ef_final_h)
    Eh,   up_e_h, dn_e_h = preparar_series(Eh,   up_e_h, dn_e_h, ef_final_h)
    Et_c, up_t_c, dn_t_c = preparar_series(Et_c, up_t_c, dn_t_c, ef_final_c)
    Ec,   up_e_c, dn_e_c = preparar_series(Ec,   up_e_c, dn_e_c, ef_final_c)

    fig, ax = crear_figura()
    ax.set_title(titulo_para(elemento, es_h=True, site=site, facet=facet) + " vs clean")
    ax.set_xlabel("DOS (States/eV)")
    ax.set_ylabel(r"$E - E_{\mathrm{Fermi}}$ (eV)" if ALINEAR_EF else "Energía (eV)")

    if (EMIN is not None) or (EMAX is not None):
        ax.set_ylim(EMIN, EMAX)
    if (XMIN is not None) or (XMAX is not None):
        if (XMIN is not None) and (XMAX is not None):
            ax.set_xlim(XMIN, XMAX)
        elif XMIN is not None:
            ax.set_xlim(left=XMIN)
        elif XMAX is not None:
            ax.set_xlim(right=XMAX)

    # +H Total (↓ negativo) con relleno
    ax.fill_betweenx(Et_h, 0,  up_t_h, alpha=ALPHA_FILL, color=COLOR_TOTAL_UP_FILL,   label=None)
    ax.fill_betweenx(Et_h, 0, -dn_t_h, alpha=ALPHA_FILL, color=COLOR_TOTAL_DOWN_FILL, label=None)
    ax.plot(up_t_h,  Et_h, color=COLOR_TOTAL_LINE, linestyle=TOTAL_UP_LINESTYLE,   linewidth=LINEWIDTH, alpha=ALPHA_LINE)
    ax.plot(-dn_t_h, Et_h, color=COLOR_TOTAL_LINE, linestyle=TOTAL_DOWN_LINESTYLE, linewidth=LINEWIDTH, alpha=ALPHA_LINE)

    # CLEAN Total (↓ negativo) solo líneas
    ax.plot(up_t_c,  Et_c, color=COLOR_TOTAL_LINE, linestyle="-.", linewidth=LINEWIDTH, alpha=0.9)
    ax.plot(-dn_t_c, Et_c, color=COLOR_TOTAL_LINE, linestyle=":",  linewidth=LINEWIDTH, alpha=0.9)

    # +H PDOS
    ax.plot(up_e_h,  Eh, color=COLOR_PDOS_UP_LINE,   linestyle=PDOS_UP_LINESTYLE, linewidth=LINEWIDTH, alpha=ALPHA_LINE)
    ax.plot(-dn_e_h, Eh, color=COLOR_PDOS_DOWN_LINE, linestyle=PDOS_DOWN_LINESTYLE, linewidth=LINEWIDTH, alpha=ALPHA_LINE)

    # CLEAN PDOS
    ax.plot(up_e_c,  Ec, color="gray", linestyle="--", linewidth=LINEWIDTH, alpha=0.9)
    ax.plot(-dn_e_c, Ec, color="gray", linestyle=":",  linewidth=LINEWIDTH, alpha=0.9)

    if GRID:
        ax.grid(True, linestyle=":")
    if MARCAR_EF:
        ax.axhline(0.0, lw=FERMI_LINE_WIDTH, ls=FERMI_LINE_STYLE, color=FERMI_LINE_COLOR, alpha=1.0)
    if LEGENDA:
        _aplicar_leyenda_h_vs_clean(ax, elemento)

    fig.tight_layout()
    out_dir = os.path.join(pdos_dir_h, "Plots")
    os.makedirs(out_dir, exist_ok=True)
    outname = f"{os.path.basename(pdos_dir_h)}_vs_clean_DOS_PDOS.{FORMATO}"
    outpath = os.path.join(out_dir, outname)
    fig.savefig(outpath, dpi=DPI)
    plt.close(fig)
    print(f"   [OK] Guardado: {outpath}")
    return outpath

def escanear_carpetas_pdos(raiz: str) -> List[str]:
    halladas = []
    for root, dirs, files in os.walk(raiz):
        if os.path.basename(root).lower() == "pdos":
            for d in sorted(dirs):
                hoja = os.path.join(root, d)
                if os.path.isdir(hoja):
                    if glob.glob(os.path.join(hoja, "*.pdos_atm#*")) or glob.glob(os.path.join(hoja, "*pdos_tot")):
                        halladas.append(hoja)
    return halladas

# =========================
# === MAIN ================
# =========================
def main():
    prefer = (FUENTE_PREFERIDA or "overleaf").strip().lower()
    if prefer not in ("overleaf", "times"):
        prefer = "overleaf"
    configurar_fuente("times" if prefer == "times" else "overleaf")
    aplicar_tamanos_fuente()

    raiz = os.path.normpath(RAIZ)
    print(f"[INFO] RAIZ = {raiz}")

    pdos_dirs = escanear_carpetas_pdos(raiz)
    if not pdos_dirs:
        print("[WARN] No se hallaron carpetas PDOS bajo la RAIZ.")
        subdirs = []
        for r, ds, fs in os.walk(raiz):
            for d in ds:
                subdirs.append(os.path.join(r, d))
            if len(subdirs) > 10:
                break
        if subdirs:
            print("[INFO] Subdirectorios encontrados (muestra 10):")
            for s in subdirs[:10]:
                print(f"    {s}")
        return

    print(f"[INFO] Carpetas con PDOS encontradas: {len(pdos_dirs)}")
    for p in pdos_dirs:
        print(f"    {p}")

    for pdos_dir in pdos_dirs:
        leaf = os.path.basename(pdos_dir)
        caso_dir = os.path.dirname(os.path.dirname(pdos_dir))
        caso = os.path.basename(caso_dir)
        elemento, site = parse_elemento_y_site(leaf)
        facet = facet_from_leaf(leaf)
        es_h = caso.endswith("-H")
        if es_h:
            clean_dir = construir_clean_counterpart(pdos_dir, raiz)
            if not clean_dir or not os.path.isdir(clean_dir):
                print(f"[SKIP] {pdos_dir}: no se encontró contraparte CLEAN.")
                continue
            plot_h_vs_clean(pdos_dir, clean_dir, elemento, site, facet)
        else:
            plot_solo_clean(pdos_dir, elemento, facet)

    print("[DONE] Procesamiento completo.")

if __name__ == "__main__":
    main()
