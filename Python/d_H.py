# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np

# Carpeta donde están los archivos *-SCF.in
CARPETA = r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\Supercell\Hydrogen\111_2x2-H\SCF"

PATRON = "*-SCF.in"

BOHR_TO_ANGSTROM = 0.529177210903


def limpiar_linea(linea):
    return linea.split("!", 1)[0].strip()


def leer_float(x):
    return float(x.replace("D", "E").replace("d", "e"))


def extraer_unidad(linea):
    linea = linea.lower()

    m = re.search(r"[({]\s*([a-zA-Z_]+)\s*[)}]", linea)
    if m:
        return m.group(1).lower()

    partes = linea.split()
    if len(partes) >= 2:
        return partes[1].lower()

    return "angstrom"


def buscar_alat(texto):
    """
    Busca A o celldm(1), por si las coordenadas vienen en alat.
    """
    m = re.search(r"\bA\s*=\s*([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)", texto)
    if m:
        return leer_float(m.group(1))

    m = re.search(r"celldm\s*\(\s*1\s*\)\s*=\s*([+-]?\d+(?:\.\d*)?(?:[dDeE][+-]?\d+)?)", texto)
    if m:
        return leer_float(m.group(1)) * BOHR_TO_ANGSTROM

    return None


def inicio_bloque_qe(linea):
    s = linea.strip().upper()

    bloques = [
        "&CONTROL", "&SYSTEM", "&ELECTRONS", "&IONS", "&CELL",
        "ATOMIC_SPECIES", "ATOMIC_POSITIONS", "CELL_PARAMETERS",
        "K_POINTS", "OCCUPATIONS", "CONSTRAINTS", "HUBBARD", "/"
    ]

    return any(s.startswith(b) for b in bloques)


def convertir_a_angstrom(valores, unidad, alat=None):
    valores = np.array(valores, dtype=float)
    unidad = unidad.lower()

    if unidad in ["angstrom", "angstroms", "ang"]:
        return valores

    if unidad == "bohr":
        return valores * BOHR_TO_ANGSTROM

    if unidad == "alat":
        if alat is None:
            raise ValueError("Hay coordenadas en alat, pero no encontré A ni celldm(1).")
        return valores * alat

    return valores


def leer_celda(lineas, texto):
    alat = buscar_alat(texto)

    for i, linea in enumerate(lineas):
        if limpiar_linea(linea).upper().startswith("CELL_PARAMETERS"):
            unidad = extraer_unidad(linea)
            celda = []

            for j in range(i + 1, i + 4):
                partes = limpiar_linea(lineas[j]).split()
                celda.append([
                    leer_float(partes[0]),
                    leer_float(partes[1]),
                    leer_float(partes[2])
                ])

            return convertir_a_angstrom(celda, unidad, alat)

    raise ValueError("No encontré CELL_PARAMETERS.")


def leer_atomos(lineas, texto, celda):
    alat = buscar_alat(texto)

    for i, linea in enumerate(lineas):
        if limpiar_linea(linea).upper().startswith("ATOMIC_POSITIONS"):
            unidad = extraer_unidad(linea)

            simbolos = []
            coords = []

            j = i + 1
            while j < len(lineas):
                s = limpiar_linea(lineas[j])

                if not s:
                    j += 1
                    continue

                if inicio_bloque_qe(s):
                    break

                partes = s.split()

                if len(partes) >= 4:
                    simbolos.append(partes[0])
                    coords.append([
                        leer_float(partes[1]),
                        leer_float(partes[2]),
                        leer_float(partes[3])
                    ])

                j += 1

            coords = np.array(coords, dtype=float)
            unidad = unidad.lower()

            if unidad in ["angstrom", "angstroms", "ang"]:
                coords_ang = coords

            elif unidad == "bohr":
                coords_ang = coords * BOHR_TO_ANGSTROM

            elif unidad == "alat":
                if alat is None:
                    raise ValueError("Hay ATOMIC_POSITIONS alat, pero no encontré A ni celldm(1).")
                coords_ang = coords * alat

            elif unidad in ["crystal", "crystal_sg"]:
                coords_ang = coords @ celda

            else:
                coords_ang = coords

            return simbolos, coords_ang

    raise ValueError("No encontré ATOMIC_POSITIONS.")


def ajustar_plano_superficie(simbolos, coords):
    """
    Ajusta z = ax + by + c usando los átomos metálicos de la capa superior.
    Excluye H.
    """
    metal = np.array([s.upper() != "H" for s in simbolos])
    coords_metal = coords[metal]

    if len(coords_metal) < 3:
        raise ValueError("No hay suficientes átomos metálicos para ajustar el plano.")

    zmax = np.max(coords_metal[:, 2])

    # Selecciona la capa metálica superior.
    # Toma átomos metálicos hasta 1.0 Å por debajo del metal más alto.
    capa_sup = coords_metal[coords_metal[:, 2] > zmax - 1.0]

    if len(capa_sup) < 3:
        raise ValueError("La capa superior tiene menos de 3 átomos.")

    x = capa_sup[:, 0]
    y = capa_sup[:, 1]
    z = capa_sup[:, 2]

    A = np.column_stack([x, y, np.ones(len(x))])
    a, b, c = np.linalg.lstsq(A, z, rcond=None)[0]

    return a, b, c


def calcular_dH(path):
    texto = open(path, "r", encoding="utf-8", errors="ignore").read()
    lineas = texto.splitlines()

    celda = leer_celda(lineas, texto)
    simbolos, coords = leer_atomos(lineas, texto, celda)

    indices_H = [i for i, s in enumerate(simbolos) if s.upper() == "H"]

    if len(indices_H) == 0:
        return None

    if len(indices_H) > 1:
        raise ValueError("Hay más de un H. Este código asume un solo H adsorbido.")

    iH = indices_H[0]

    xH = coords[iH, 0]
    yH = coords[iH, 1]
    zH = coords[iH, 2]

    a, b, c = ajustar_plano_superficie(simbolos, coords)

    zref = a * xH + b * yH + c
    dH = zH - zref

    return dH


def main():
    archivos = sorted(glob.glob(os.path.join(CARPETA, PATRON)))

    print("sistema\td_H_angstrom")

    for path in archivos:
        sistema = os.path.basename(path).replace("-SCF.in", "")

        try:
            dH = calcular_dH(path)

            if dH is None:
                continue

            print(f"{sistema}\t{dH:.6f}")

        except Exception as e:
            print(f"{sistema}\tERROR: {e}")


if __name__ == "__main__":
    main()