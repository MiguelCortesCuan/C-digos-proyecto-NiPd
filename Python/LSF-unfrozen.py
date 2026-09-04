import re
from pathlib import Path

# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = Path(r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\Supercell\Hydrogen\111_2x2-H")
OUT_SUBDIR = "unfrozen"


# =========================================================
# FILE HELPERS
# =========================================================
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


# =========================================================
# LSF PARSING
# =========================================================
def find_relax_heredoc(text: str):
    pattern = re.compile(
        r'(cat\s*>\s*"\$RELDIR/\$\{NAME\}-relax\.in"\s*<<EOF\s*\n)(.*?)(\nEOF)',
        re.DOTALL,
    )
    return pattern.search(text)


def extract_relax_body(text: str) -> str | None:
    m = find_relax_heredoc(text)
    return m.group(2) if m else None


def hydrogen_frozen_in_relax(relax_body: str) -> bool:
    m = re.search(
        r"ATOMIC_POSITIONS\b[^\n]*\n(.*?)(?=\n\s*K_POINTS\s+automatic\b)",
        relax_body,
        flags=re.DOTALL,
    )
    if not m:
        return False

    for line in m.group(1).splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) >= 7 and parts[0] == "H":
            flags = parts[-3:]
            if any(f == "0" for f in flags):
                return True
    return False


def split_relax_body(relax_body: str):
    pattern = re.compile(
        r"(?P<head>.*?)(?P<geom>CELL_PARAMETERS\b[^\n]*\n.*?\nATOMIC_POSITIONS\b[^\n]*\n.*?)(?P<tail>\n\s*K_POINTS\s+automatic\b.*)",
        flags=re.DOTALL,
    )
    m = pattern.search(relax_body)
    if not m:
        raise ValueError("Could not split RELAX block into head/geometry/tail.")
    return m.group("head"), m.group("geom"), m.group("tail")


# =========================================================
# TEXT TRANSFORMS
# =========================================================
def replace_run_stage_function(text: str) -> str:
    pattern = re.compile(r"run_stage\s*\(\)\s*\{.*?\n\}", flags=re.DOTALL)
    replacement = """run_stage () {
  local stage="$1"
  local fout="$2"
  shift 2

  echo "[$stage] ejecutando..."
  "$@" > "$fout"
  check_done "$fout"
}"""
    if not pattern.search(text):
        raise ValueError("Could not find run_stage() function.")
    return pattern.sub(replacement, text, count=1)


def insert_unfrozen_helpers(text: str) -> str:
    marker = "# =========================\n# RELAX\n# =========================\n"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError("Could not find RELAX section marker.")

    block = r'''
make_unfrozen_geometry () {
  local source_geom="$1"
  local target_geom="$2"

  [[ -f "$source_geom" ]] || {
    echo "ERROR: source geometry does not exist: $source_geom"
    exit 1
  }

  awk '
    BEGIN {inpos=0}
    /^[[:space:]]*ATOMIC_POSITIONS/ {
      inpos=1
      print
      next
    }
    inpos && /^[[:space:]]*$/ {
      inpos=0
      print
      next
    }
    inpos && $1 ~ /^[A-Za-z]/ {
      if ($1 == "H") {
        printf "%-2s %20s %20s %20s    1   1   1\n", $1, $2, $3, $4
      } else {
        print
      }
      next
    }
    {
      print
    }
  ' "$source_geom" > "$target_geom"

  grep -q "ATOMIC_POSITIONS" "$target_geom" || {
    echo "ERROR: failed to build unfrozen geometry: $target_geom"
    exit 1
  }
}

prepare_start_geometry () {
  local start_geom="$RELDIR/${NAME}_start_geometry_unfrozen.in"
  local prev_final_geom="$RELDIR/${NAME}_final_geometry.in"

  if [[ ! -f "$prev_final_geom" ]]; then
    echo "[SKIP] previous final geometry does not exist: $prev_final_geom"
    exit 0
  fi

  echo "[GEOM] using previous final geometry: $prev_final_geom"
  make_unfrozen_geometry "$prev_final_geom" "$start_geom"

  [[ -f "$start_geom" ]] || {
    echo "ERROR: failed to create start geometry: $start_geom"
    exit 1
  }
}

clean_previous_results () {
  echo "[CLEAN] removing previous results for ${NAME}"

  rm -f "$RELDIR/${NAME}-relax.in" "$RELDIR/${NAME}-relax.out"
  rm -f "$RELDIR/${NAME}_final_cell.tmp" "$RELDIR/${NAME}_final_positions.tmp"
  rm -f "$RELDIR/${NAME}_final_geometry_unfrozen.in"
  rm -f "$RELDIR/${NAME}_start_geometry_unfrozen.in"

  rm -f "$SCFDIR/${NAME}-SCF.in" "$SCFDIR/${NAME}-SCF.out"
  rm -f "$NSCFDIR/${NAME}-NSCF.in" "$NSCFDIR/${NAME}-NSCF.out"

  rm -f "$DOSDIR/${NAME}-dos.in" "$DOSDIR/${NAME}-dos.out" "$DOSDIR/${NAME}.dos"

  rm -rf "$PDOSDIR"
  mkdir -p "$PDOSDIR"
  rm -f "$ROOT/PDOS/${NAME}/${NAME}"*
  rm -f "$ROOT/PDOS/${NAME}/"*.pdos*
  rm -f "$ROOT/PDOS/${NAME}/"*-pdos.in
  rm -f "$ROOT/PDOS/${NAME}/"*-pdos.out
  rm -f "$ROOT/PDOS/${NAME}/"*.pdos_tot

  rm -f "$MESPDIR/${NAME}_density.in" "$MESPDIR/${NAME}_density.out" "$MESPDIR/${NAME}_density.pp" "$MESPDIR/${NAME}_density.cube"
  rm -f "$MESPDIR/${NAME}_aecd.in" "$MESPDIR/${NAME}_aecd.out" "$MESPDIR/${NAME}_aecd.pp" "$MESPDIR/${NAME}_aecd.cube"
  rm -f "$MESPDIR/${NAME}_potential.in" "$MESPDIR/${NAME}_potential.out" "$MESPDIR/${NAME}_potential.pp" "$MESPDIR/${NAME}_potential.cube"
  rm -f "$MESPDIR/${NAME}_V_coulomb.in" "$MESPDIR/${NAME}_V_coulomb.out" "$MESPDIR/${NAME}_V_coulomb.pp" "$MESPDIR/${NAME}_V_coulomb.cube"

  rm -rf "$OUTDIR/${NAME}.save"
  rm -f "$OUTDIR/${NAME}.xml"
  rm -f "$OUTDIR/${NAME}.update"
  rm -f "$OUTDIR/${NAME}.bfgs"
  rm -f "$OUTDIR/${NAME}.mix"*
  rm -f "$OUTDIR/${NAME}.wfc"*
}

clean_previous_results
prepare_start_geometry

'''.lstrip()

    return text[:idx] + block + text[idx:]


def rewrite_relax_section(text: str) -> str:
    m = find_relax_heredoc(text)
    if not m:
        raise ValueError("Could not find RELAX here-doc.")

    relax_body = m.group(2)
    head, _geom, tail = split_relax_body(relax_body)

    new_relax = (
        'cat > "$RELDIR/${NAME}-relax.in" <<EOF\n'
        + head.rstrip()
        + "\n\nEOF\n\n"
        + 'cat "$RELDIR/${NAME}_start_geometry_unfrozen.in" >> "$RELDIR/${NAME}-relax.in"\n\n'
        + 'cat >> "$RELDIR/${NAME}-relax.in" <<EOF\n'
        + tail.lstrip("\n")
        + "\nEOF"
    )

    return text[:m.start()] + new_relax + text[m.end():]


def add_generate_unfrozen_after_relax(text: str) -> str:
    marker = 'extract_final_geometry "$RELDIR/${NAME}-relax.out" "$RELDIR/${NAME}_final_geometry.in"'
    if marker not in text:
        raise ValueError("Could not find extract_final_geometry call after RELAX.")

    replacement = marker + """

make_unfrozen_geometry "$RELDIR/${NAME}_final_geometry.in" "$RELDIR/${NAME}_final_geometry_unfrozen.in"
"""
    return text.replace(marker, replacement, 1)


def replace_scf_nscf_geometry_uses(text: str) -> str:
    text = text.replace(
        'cat "$RELDIR/${NAME}_final_geometry.in" >> "$SCFDIR/${NAME}-SCF.in"',
        'cat "$RELDIR/${NAME}_final_geometry_unfrozen.in" >> "$SCFDIR/${NAME}-SCF.in"',
    )
    text = text.replace(
        'cat "$RELDIR/${NAME}_final_geometry.in" >> "$NSCFDIR/${NAME}-NSCF.in"',
        'cat "$RELDIR/${NAME}_final_geometry_unfrozen.in" >> "$NSCFDIR/${NAME}-NSCF.in"',
    )
    return text


def fix_common_syntax_issues(text: str) -> str:
    text = re.sub(
        r"(restart_mode\s*=\s*'from_scratch')\s*\n",
        r"\1,\n",
        text,
    )
    return text


def transform_lsf(text: str) -> str:
    relax_body = extract_relax_body(text)
    if relax_body is None:
        raise ValueError("Could not find RELAX block.")

    if not hydrogen_frozen_in_relax(relax_body):
        raise ValueError("Hydrogen is not partially or fully frozen.")

    new_text = text
    new_text = replace_run_stage_function(new_text)
    new_text = rewrite_relax_section(new_text)
    new_text = insert_unfrozen_helpers(new_text)
    new_text = add_generate_unfrozen_after_relax(new_text)
    new_text = replace_scf_nscf_geometry_uses(new_text)
    new_text = fix_common_syntax_issues(new_text)
    return new_text


# =========================================================
# MAIN
# =========================================================
def main():
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Folder does not exist: {BASE_DIR}")

    outdir = BASE_DIR / OUT_SUBDIR
    outdir.mkdir(exist_ok=True)

    lsf_files = sorted(BASE_DIR.glob("*.lsf"))
    if not lsf_files:
        print("No .lsf files found.")
        return

    ok = 0
    skipped = 0
    failed = 0

    for lsf in lsf_files:
        try:
            text = read_text(lsf)
            relax_body = extract_relax_body(text)

            if relax_body is None:
                print(f"[ERROR] {lsf.name}: RELAX block not found")
                failed += 1
                continue

            if not hydrogen_frozen_in_relax(relax_body):
                print(f"[SKIP]  {lsf.name}: H is not partially or fully frozen")
                skipped += 1
                continue

            new_text = transform_lsf(text)
            outpath = outdir / lsf.name
            write_text(outpath, new_text)

            print(f"[OK]    {lsf.name} -> {outpath}")
            ok += 1

        except Exception as e:
            print(f"[ERROR] {lsf.name}: {e}")
            failed += 1

    print("\nSummary")
    print(f"Generated: {ok}")
    print(f"Skipped  : {skipped}")
    print(f"Errors   : {failed}")


if __name__ == "__main__":
    main()