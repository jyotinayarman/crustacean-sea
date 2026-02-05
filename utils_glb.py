import numpy as np
import trimesh
from io import BytesIO


def normalize_glb_to_unit_cube(glb_path_or_buffer, output_buffer: BytesIO) -> None:
    if isinstance(glb_path_or_buffer, (str, bytes)):
        mesh = trimesh.load(glb_path_or_buffer if isinstance(glb_path_or_buffer, str) else BytesIO(glb_path_or_buffer))
    else:
        mesh = trimesh.load(glb_path_or_buffer)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No trimesh in scene")
        mesh = trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = list(mesh.geometry.values())[0] if hasattr(mesh, "geometry") else mesh
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    min_b = verts.min(axis=0)
    max_b = verts.max(axis=0)
    center = (min_b + max_b) / 2
    extent = max_b - min_b
    scale = 1.0 / (np.max(extent) + 1e-8)
    verts = (verts - center) * scale
    margin = 0.999
    verts *= margin
    mesh.vertices = verts
    mesh.export(output_buffer, file_type="glb")
