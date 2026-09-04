import re
from pathlib import Path
from decimal import Decimal
import pandas as pd

# ===== PON AQUÍ LA RUTA PRINCIPAL =====
ruta_base = Path(r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\Supercell\Hydrogen\111_2x2-H")

patron_energia = re.compile(
    r'^\s*!\s+total energy\s+=\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+Ry'
)

resultados = []

for archivo in ruta_base.rglob("*.out"):
    energia_final = None

    try:
        with open(archivo, "r", encoding="utf-8", errors="ignore") as f:
            for linea in f:
                m = patron_energia.match(linea)
                if m:
                    # Guarda el valor tal como aparece y luego lo formatea a 8 decimales
                    energia_final = f"{Decimal(m.group(1)):.8f}"
    except Exception as e:
        energia_final = m.group(1)

    resultados.append({
        "Archivo": archivo.name,
        "EnergiaFinal_Ry": energia_final if energia_final is not None else "No encontrada"
    })

df = pd.DataFrame(resultados).sort_values("Archivo").reset_index(drop=True)

print(df.to_string(index=False))

# Opcional: guardar para Excel
# df.to_csv(ruta_base / "energias_finales_QE.csv", sep="\t", index=False, encoding="utf-8")