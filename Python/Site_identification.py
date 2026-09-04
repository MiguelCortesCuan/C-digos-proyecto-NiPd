import os
import numpy as np
from ase.io import read

def get_total_energy(out_file):
    """Extrae la energía total del archivo .out de QE."""
    energy = None
    try:
        with open(out_file, 'r') as f:
            for line in f:
                if '!' in line:
                    energy = line.split()[4]
        return energy
    except:
        return "N/A"

def identify_site(atoms):
    # 1. Forzar el cálculo considerando las Condiciones Periódicas de Frontera
    atoms.pbc = [True, True, True]
    
    h_idx = [atom.index for atom in atoms if atom.symbol == 'H']
    metal_idx = [atom.index for atom in atoms if atom.symbol in ['Ni', 'Pd']]
    
    if not h_idx: return "No H found"
    h = h_idx[0]
    
    # 2. Separar las capas por altura (Z) con mayor tolerancia a las relajaciones
    z_coords = atoms.positions[metal_idx, 2]
    z_max = np.max(z_coords)
    top_layer = [i for i in metal_idx if abs(atoms.positions[i, 2] - z_max) < 1.0]
    
    # 3. Obtener distancias desde el H a la capa superior usando PBC
    dist_to_top = atoms.get_distances(h, top_layer, mic=True)
    dist_to_top.sort() # Ordenar de menor a mayor
    
    if len(dist_to_top) == 0: return "Unknown"
    
    # 4. Clasificación dinámica (Top, Bridge, Hollow)
    # Contamos cuántos átomos están "cerca" respecto a la primera distancia (tolerancia de 0.35 A)
    d0 = dist_to_top[0]
    neighbors = [d for d in dist_to_top if d - d0 < 0.35]
    num_neighbors = len(neighbors)
    
    if num_neighbors == 1:
        return "Top"
    elif num_neighbors == 2:
        return "Bridge"
    else:
        # 5. Diferenciar Hollow FCC vs HCP
        # Buscar los átomos de la segunda capa (ignorando la capa 1)
        z_remaining = [z for z in z_coords if z_max - z > 1.0]
        if not z_remaining: return "Hollow (?)"
        
        z_sub = max(z_remaining)
        sub_layer = [i for i in metal_idx if abs(atoms.positions[i, 2] - z_sub) < 1.0]
        
        # Obtener los vectores de distancia tridimensionales usando PBC (MIC)
        vecs = atoms.get_distances(h, sub_layer, mic=True, vector=True)
        
        # Calcular la norma solo en 2D (X e Y) ignorando la altura Z
        dist_2d = np.linalg.norm(vecs[:, :2], axis=1)
        
        # Si la distancia 2D al átomo de abajo más cercano es casi 0 (tolerancia 0.8 A)
        if min(dist_2d) < 0.8:
            return "Hollow HCP"
        else:
            return "Hollow FCC"

def main():
    ruta = r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\Supercell\Hydrogen\111_2x2-H\SCF"
    
    if not os.path.exists(ruta):
        print(f"Error: Ruta no encontrada:\n{ruta}")
        return

    files = [f for f in os.listdir(ruta) if f.endswith('.in')]
    
    print(f"{'Archivo':<30} | {'Sitio':<12} | {'Energía (Ry)':<15}")
    print("-" * 65)
    
    for in_file_name in files:
        in_file_path = os.path.join(ruta, in_file_name)
        out_file_path = in_file_path.replace('.in', '.out')
        
        try:
            atoms = read(in_file_path, format='espresso-in')
            site = identify_site(atoms)
        except Exception as e:
            site = "Error"
            
        energy = "N/A"
        if os.path.exists(out_file_path):
            energy = get_total_energy(out_file_path)
            
        print(f"{in_file_name:<30} | {site:<12} | {energy:<15}")

if __name__ == "__main__":
    main()