"""Parse LAMMPS custom dump trajectories (id type element xu yu zu)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrajFrame:
    step: int
    atoms: list[tuple[int, int, str, float, float, float]]  # id, type, element, x, y, z


def iter_frames(path: Path):
    """Yield TrajFrame objects from a lammpstrj file."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        if lines[i] != "ITEM: TIMESTEP":
            i += 1
            continue
        step = int(lines[i + 1])
        i += 2
        if i >= len(lines) or lines[i] != "ITEM: NUMBER OF ATOMS":
            break
        n_atoms = int(lines[i + 1])
        i += 2
        if i >= len(lines) or not lines[i].startswith("ITEM: BOX"):
            break
        i += 4  # skip box bounds (constant cell in our NVT runs)
        if i >= len(lines) or not lines[i].startswith("ITEM: ATOMS"):
            break
        i += 1
        atoms: list[tuple[int, int, str, float, float, float]] = []
        for _ in range(n_atoms):
            parts = lines[i].split()
            i += 1
            if len(parts) < 6:
                continue
            atom_id = int(parts[0])
            atom_type = int(parts[1])
            element = parts[2]
            x, y, z = map(float, parts[3:6])
            atoms.append((atom_id, atom_type, element, x, y, z))
        yield TrajFrame(step=step, atoms=atoms)


def frame_at_step(path: Path, target_step: int) -> TrajFrame | None:
    for frame in iter_frames(path):
        if frame.step == target_step:
            return frame
        if frame.step > target_step:
            return None
    return None


def available_steps(path: Path) -> list[int]:
    return [frame.step for frame in iter_frames(path)]
