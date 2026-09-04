from pathlib import Path
from ase.io import read, write
from ase import Atoms
import numpy as np

# === USER CONFIGURATION ===

FOLDER = Path(r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\Celdaunitaria\New")
OUTPUT_DIR = Path(r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\Gaussian\spheres")

RADIUS = 12.0  # Radius of the nanoparticle in Angstroms
OUTPUT_FORMAT = "xyz"  # Output format (e.g., "xyz", "pdb", "cif")
DEFAULT_CHARGE = 0
DEFAULT_MULTIPLICITY = 1
DEFAULT_MEMORY = "160GB"
DEFAULT_CORES = 32
USE_QMMM = True  # QM/MM for outer/inner layers

# === END CONFIGURATION ===

def build_supercell(atoms: Atoms, radius: float) -> Atoms:
    """
    Expand the unit cell into a supercell and crop it to create a nanoparticle.
    """
    cell_lengths = atoms.get_cell().lengths()
    max_repeats = int(np.ceil(2 * radius / min(cell_lengths))) + 1
    supercell = atoms.repeat((max_repeats, max_repeats, max_repeats))  # Expand the unit cell into a supercell
    return supercell

def select_atoms_by_radius(supercell: Atoms, inner_radius: float, outer_radius: float) -> Atoms:
    """
    Select atoms within the radius range between inner and outer radii.
    """
    center = supercell.get_center_of_mass()
    distances = np.linalg.norm(supercell.get_positions() - center, axis=1)
    
    # Select atoms within the defined radius range
    selected_atoms = supercell[(distances >= inner_radius) & (distances <= outer_radius)]
    
    return selected_atoms

def write_gaussian_input(nanoparticle_atoms: Atoms, name: str, method: str, basis: str, output_dir: Path):
    """Write Gaussian input file (.gjf) for the nanoparticle."""
    num_atoms = len(nanoparticle_atoms)
    gjf_path = output_dir / f"{name}_nanoparticle.gjf"

    with open(gjf_path, "w") as f:
        f.write(f"%chk={name}_nanoparticle.chk\n")
        f.write(f"%nosave\n")
        f.write(f"%mem={DEFAULT_MEMORY}\n")
        f.write(f"%nprocshared={DEFAULT_CORES}\n")
        f.write(f"# opt {method}/{basis}\n\n")
        f.write(f"{name} nanoparticle\n\n")
        f.write(f"{DEFAULT_CHARGE} {DEFAULT_MULTIPLICITY}\n")

        symbols = nanoparticle_atoms.get_chemical_symbols()
        positions = nanoparticle_atoms.get_positions()

        for sym, pos in zip(symbols, positions):
            f.write(f"{sym:<2}   {pos[0]:.6f}   {pos[1]:.6f}   {pos[2]:.6f}\n")

        f.write("\n")

    print(f"[🧪] Gaussian input saved: {gjf_path.name} ({num_atoms} atoms)")

def main():
    if not FOLDER.exists():
        print(f"[✘] Folder not found: {FOLDER}")
        return

    cif_files = list(FOLDER.glob("*.cif"))
    if not cif_files:
        print(f"[ℹ] No CIF files found in {FOLDER}")
        return

    print(f"[ℹ] Found {len(cif_files)} CIF files in {FOLDER}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for cif_file in cif_files:
        try:
            atoms = read(cif_file)
            name = cif_file.stem

            # Step 1: Create supercell based on radius
            supercell = build_supercell(atoms, RADIUS)

            # Step 2: Select atoms for each layer (core, intermediate, surface)
            core_atoms = select_atoms_by_radius(supercell, 0, RADIUS * 0.4)  # Inner core (0 to 40% of the radius)
            intermediate_atoms = select_atoms_by_radius(supercell, RADIUS * 0.4, RADIUS * 0.7)  # Intermediate (40% to 70%)
            surface_atoms = select_atoms_by_radius(supercell, RADIUS * 0.7, RADIUS)  # Surface (70% to 100%)

            # Combine core, intermediate, and surface atoms to form the nanoparticle
            nanoparticle_atoms = core_atoms + intermediate_atoms + surface_atoms

            # Step 3: Save the nanoparticle structure
            out_file = OUTPUT_DIR / f"{name}_nanoparticle.{OUTPUT_FORMAT}"
            write(out_file, nanoparticle_atoms)
            print(f"[✔] Saved {out_file.name} ({len(nanoparticle_atoms)} atoms)")

            # Step 4: Generate Gaussian input file
            n_atoms = len(nanoparticle_atoms)
            method = "PBE" if n_atoms <= 150 else "PM6"
            basis = "LANL2DZ" if n_atoms <= 150 else ""  # PM6 doesn’t need a basis set

            write_gaussian_input(nanoparticle_atoms, name, method, basis, OUTPUT_DIR)

        except Exception as e:
            print(f"[⚠] Failed to process {cif_file.name}: {e}")

if __name__ == "__main__":
    main()
