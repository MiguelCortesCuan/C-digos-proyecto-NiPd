# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:53:48 2026

"""

from pathlib import Path
import re
import numpy as np


# ==========================================================
# === CAMBIA ESTAS DOS RUTAS ===============================
# ==========================================================
CARPETA_ENTRADA = r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\Supercell\Hydrogen\111_2x2\SCF"
CARPETA_SALIDA  = r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\Supercell\Hydrogen\111_2x2\XSF"

# Patrón de archivos a convertir.
# Si quieres convertir todos los .in, cambia a "*.in".
PATRON_ARCHIVOS = "*-SCF.in"

# Si el archivo no dice unidad en CELL_PARAMETERS o ATOMIC_POSITIONS,
# se asumirá esta unidad.
UNIDAD_DEFAULT = "angstrom"

BOHR_TO_ANGSTROM = 0.529177210903


# ==========================================================
# === FUNCIONES DE PARSEO ==================================
# ==========================================================
def limpiar_linea(linea):
    """Quita comentarios tipo ! y espacios extremos."""
    return linea.split("!", 1)[0].strip()


def extraer_unidad(linea, unidad_default=UNIDAD_DEFAULT):
    """
    Lee la unidad de una línea tipo:
        CELL_PARAMETERS (angstrom)
        CELL_PARAMETERS {angstrom}
        ATOMIC_POSITIONS angstrom

    Devuelve: angstrom, bohr, alat, crystal, etc.
    """
    s = linea.strip().lower()

    m = re.search(r"[({]\s*([a-zA-Z_]+)\s*[)}]", s)
    if m:
        return m.group(1).lower()

    partes = s.split()
    if len(partes) >= 2:
        return partes[1].lower()

    return unidad_default.lower()


def es_inicio_de_bloque(linea):
    """Detecta si una línea inicia otro bloque de QE."""
    s = linea.strip().upper()
    if not s:
        return False

    bloques = (
        "&CONTROL", "&SYSTEM", "&ELECTRONS", "&IONS", "&CELL",
        "ATOMIC_SPECIES", "ATOMIC_POSITIONS", "CELL_PARAMETERS",
        "K_POINTS", "OCCUPATIONS", "CONSTRAINTS", "HUBBARD",
        "ATOMIC_FORCES", "SOLVENTS"
    )
    return any(s.startswith(b) for b in bloques) or s.startswith("/")


def leer_float_qe(texto):
    """Convierte números estilo QE: 1.0d-10 -> 1.0e-10."""
    return float(texto.replace("D", "E").replace("d", "e"))


def buscar_parametro_escala(texto):
    """
    Busca escala para unidades 'alat'.

    Prioridad:
    - A = valor      se toma en angstrom
    - celldm(1)      se toma en bohr y se convierte a angstrom

    Devuelve None si no encuentra escala.
    """
    m = re.search(r"\bA\s*=\s*([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)", texto, flags=re.IGNORECASE)
    if m:
        return leer_float_qe(m.group(1))

    m = re.search(r"celldm\s*\(\s*1\s*\)\s*=\s*([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)", texto, flags=re.IGNORECASE)
    if m:
        return leer_float_qe(m.group(1)) * BOHR_TO_ANGSTROM

    return None


def convertir_longitud_a_angstrom(valores, unidad, escala_alat=None):
    """Convierte un arreglo de longitudes a angstrom."""
    unidad = unidad.lower()
    valores = np.array(valores, dtype=float)

    if unidad in ("angstrom", "angstroms", "ang"):
        return valores

    if unidad == "bohr":
        return valores * BOHR_TO_ANGSTROM

    if unidad == "alat":
        if escala_alat is None:
            raise ValueError("Se encontró unidad 'alat', pero no se encontró A ni celldm(1) para convertir a angstrom.")
        return valores * escala_alat

    # Para casos sin unidad clara, se asume angstrom.
    return valores


def simbolo_quimico(etiqueta):
    """
    Convierte etiquetas tipo Ni, Ni1, Pd_surface a símbolo químico usable en XSF.
    """
    m = re.match(r"([A-Z][a-z]?)", etiqueta.strip())
    if m:
        return m.group(1)
    return etiqueta.strip()


def leer_cell_parameters(lineas, texto_completo):
    """Extrae matriz 3x3 de CELL_PARAMETERS en angstrom."""
    escala_alat = buscar_parametro_escala(texto_completo)

    for i, linea in enumerate(lineas):
        if limpiar_linea(linea).upper().startswith("CELL_PARAMETERS"):
            unidad = extraer_unidad(linea)
            vectores = []

            j = i + 1
            while j < len(lineas) and len(vectores) < 3:
                s = limpiar_linea(lineas[j])
                if s:
                    partes = s.split()
                    if len(partes) < 3:
                        raise ValueError("CELL_PARAMETERS tiene una línea con menos de 3 números.")
                    vectores.append([leer_float_qe(partes[0]), leer_float_qe(partes[1]), leer_float_qe(partes[2])])
                j += 1

            if len(vectores) != 3:
                raise ValueError("No se pudieron leer los 3 vectores de CELL_PARAMETERS.")

            return convertir_longitud_a_angstrom(vectores, unidad, escala_alat)

    raise ValueError("No se encontró el bloque CELL_PARAMETERS.")


def leer_atomic_positions(lineas, celda_angstrom, texto_completo):
    """
    Extrae ATOMIC_POSITIONS.

    Acepta:
    - ATOMIC_POSITIONS (angstrom)
    - ATOMIC_POSITIONS {angstrom}
    - ATOMIC_POSITIONS crystal
    - ATOMIC_POSITIONS bohr
    - ATOMIC_POSITIONS alat

    Ignora banderas finales 0 0 0 o 1 1 1.
    """
    escala_alat = buscar_parametro_escala(texto_completo)

    for i, linea in enumerate(lineas):
        if limpiar_linea(linea).upper().startswith("ATOMIC_POSITIONS"):
            unidad = extraer_unidad(linea)
            atomos = []

            j = i + 1
            while j < len(lineas):
                s = limpiar_linea(lineas[j])

                if not s:
                    j += 1
                    continue

                if es_inicio_de_bloque(s):
                    break

                partes = s.split()
                if len(partes) >= 4:
                    etiqueta = partes[0]
                    try:
                        x = leer_float_qe(partes[1])
                        y = leer_float_qe(partes[2])
                        z = leer_float_qe(partes[3])
                        atomos.append([simbolo_quimico(etiqueta), x, y, z])
                    except Exception:
                        pass

                j += 1

            if not atomos:
                raise ValueError("No se pudieron leer átomos en ATOMIC_POSITIONS.")

            etiquetas = [a[0] for a in atomos]
            coords = np.array([[a[1], a[2], a[3]] for a in atomos], dtype=float)

            if unidad in ("angstrom", "angstroms", "ang"):
                coords_angstrom = coords
            elif unidad == "bohr":
                coords_angstrom = coords * BOHR_TO_ANGSTROM
            elif unidad == "alat":
                if escala_alat is None:
                    raise ValueError("Se encontró ATOMIC_POSITIONS alat, pero no se encontró A ni celldm(1).")
                coords_angstrom = coords * escala_alat
            elif unidad in ("crystal", "crystal_sg"):
                # Coordenadas fraccionarias: r_cart = f1*a1 + f2*a2 + f3*a3
                coords_angstrom = coords @ celda_angstrom
            else:
                # Si no se reconoce la unidad, se asume angstrom para no detener casos simples.
                coords_angstrom = coords

            return etiquetas, coords_angstrom

    raise ValueError("No se encontró el bloque ATOMIC_POSITIONS.")


def leer_estructura_qe(path_in):
    """Lee un archivo .in de QE y devuelve celda + átomos en angstrom."""
    texto = Path(path_in).read_text(encoding="utf-8", errors="ignore")
    lineas = texto.splitlines()

    celda = leer_cell_parameters(lineas, texto)
    etiquetas, coords = leer_atomic_positions(lineas, celda, texto)

    return celda, etiquetas, coords


# ==========================================================
# === ESCRITURA XSF ========================================
# ==========================================================
def escribir_xsf(path_out, celda, etiquetas, coords):
    """Escribe archivo XSF para XCrySDen/VESTA."""
    with open(path_out, "w", encoding="utf-8") as f:
        f.write("CRYSTAL\n")
        f.write("PRIMVEC\n")
        for v in celda:
            f.write(f"  {v[0]:18.10f} {v[1]:18.10f} {v[2]:18.10f}\n")

        f.write("PRIMCOORD\n")
        f.write(f"  {len(etiquetas)} 1\n")
        for etiqueta, r in zip(etiquetas, coords):
            f.write(f"  {etiqueta:<3s} {r[0]:18.10f} {r[1]:18.10f} {r[2]:18.10f}\n")


def nombre_salida_xsf(path_in):
    """Genera nombre de salida limpio."""
    nombre = Path(path_in).stem
    if nombre.endswith("-SCF"):
        nombre = nombre[:-4]
    return nombre + ".xsf"


# ==========================================================
# === PROGRAMA PRINCIPAL ===================================
# ==========================================================
def convertir_todos():
    entrada = Path(CARPETA_ENTRADA)
    salida = Path(CARPETA_SALIDA)
    salida.mkdir(parents=True, exist_ok=True)

    archivos = sorted(entrada.glob(PATRON_ARCHIVOS))

    if not archivos:
        print(f"No se encontraron archivos con patrón {PATRON_ARCHIVOS} en:")
        print(entrada)
        return

    print(f"Carpeta de entrada: {entrada}")
    print(f"Carpeta de salida : {salida}")
    print(f"Archivos encontrados: {len(archivos)}\n")

    ok = 0
    errores = 0

    for path_in in archivos:
        try:
            celda, etiquetas, coords = leer_estructura_qe(path_in)
            path_out = salida / nombre_salida_xsf(path_in)
            escribir_xsf(path_out, celda, etiquetas, coords)

            ok += 1
            print(f"OK    {path_in.name}  ->  {path_out.name}")

        except Exception as exc:
            errores += 1
            print(f"ERROR {path_in.name}: {exc}")

    print("\nResumen")
    print(f"Convertidos correctamente: {ok}")
    print(f"Con error               : {errores}")


if __name__ == "__main__":
    convertir_todos()
