#!/usr/bin/env python3
"""Convert hold-out LAMMPS data.in snapshots to/from ChIMES .xyzf frames."""

from __future__ import annotations

import json
from pathlib import Path

ELEMENT_BY_TYPE = {1: "C", 2: "N"}
TYPE_BY_ELEMENT = {"C": 1, "N": 2}


def read_data_in(path: Path) -> tuple[list[float], list[float], list[float], list[tuple[str, float, float, float]]]:
    """Parse a triclinic C/N data.in written by extract_md_holdout."""
    lines = path.read_text(encoding="utf-8").splitlines()
    xhi = yhi = zhi = xy = xz = yz = None
    in_atoms = False
    atoms: list[tuple[str, float, float, float]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "Atoms":
            in_atoms = True
            continue
        if not in_atoms:
            if stripped.endswith("xlo xhi"):
                xhi = float(stripped.split()[1])
            elif stripped.endswith("ylo yhi"):
                yhi = float(stripped.split()[1])
            elif stripped.endswith("zlo zhi"):
                zhi = float(stripped.split()[1])
            elif stripped.endswith("xy xz yz"):
                parts = stripped.split()
                xy, xz, yz = map(float, parts[:3])
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        atom_type = int(parts[1])
        element = ELEMENT_BY_TYPE.get(atom_type)
        if element is None:
            raise ValueError(f"Unsupported atom type {atom_type} in {path}")
        x, y, z = map(float, parts[2:5])
        atoms.append((element, x, y, z))
    if None in (xhi, yhi, zhi, xy, xz, yz) or not atoms:
        raise ValueError(f"Could not parse {path}")
    a = [xhi, 0.0, 0.0]
    b = [xy, yhi, 0.0]
    c = [xz, yz, zhi]
    return a, b, c, atoms


def frame_to_xyzf_lines(
    a: list[float], b: list[float], c: list[float], atoms: list[tuple[str, float, float, float]]
) -> tuple[int, str, list[str]]:
    n_atoms = len(atoms)
    box_line = (
        "NON_ORTHO "
        + " ".join(f"{v:.8g}" for v in (*a, *b, *c))
        + "\n"
    )
    atom_lines = [f"{element} {x:.8g} {y:.8g} {z:.8g}\n" for element, x, y, z in atoms]
    return n_atoms, box_line, atom_lines


def data_in_to_xyzf_frame(path: Path) -> tuple[int, str, list[str]]:
    return frame_to_xyzf_lines(*read_data_in(path))


def write_combined_xyzf(
    manifest: list[dict],
    snapshots_dir: Path,
    xyzf_path: Path,
    index_path: Path | None = None,
) -> int:
    """Append all hold-out frames to one .xyzf; optional JSON maps frame index -> metadata."""
    xyzf_path.parent.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict] = []
    with xyzf_path.open("w", encoding="utf-8") as handle:
        for frame_idx, entry in enumerate(manifest, start=1):
            snap_id = entry["snapshot_id"]
            data_in = snapshots_dir / snap_id / "data.in"
            if not data_in.is_file():
                raise FileNotFoundError(f"Missing {data_in}")
            n_atoms, box_line, atom_lines = data_in_to_xyzf_frame(data_in)
            handle.write(f"{n_atoms}\n")
            handle.write(box_line)
            handle.writelines(atom_lines)
            index_rows.append({"frame_index": frame_idx, **entry})
    if index_path is not None:
        index_path.write_text(json.dumps(index_rows, indent=2) + "\n", encoding="utf-8")
    return len(index_rows)


def iter_xyzf_frames(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            n_atoms = int(line.strip())
            box_line = handle.readline()
            atoms = [handle.readline() for _ in range(n_atoms)]
            yield n_atoms, box_line, atoms
