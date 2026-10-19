"""Geometry and material toolkit for the procedural watch models (Blender 4.5, bpy).

Conventions: 1 unit = 10 mm. A watch is built lying flat with its dial facing +Z and
12 o'clock toward +Y, centred on the origin. Everything a watch is made of is parented to
one empty, so a shot can pose the whole watch by moving that empty.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import bmesh
import bpy
from mathutils import Matrix, Vector

TAU = math.tau


# ─────────────────────────────────────────────────────────────
# Scene housekeeping
# ─────────────────────────────────────────────────────────────

def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def link(obj: bpy.types.Object, parent: bpy.types.Object | None = None) -> bpy.types.Object:
    bpy.context.scene.collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    return obj


def empty(name: str, parent: bpy.types.Object | None = None) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, None)
    return link(obj, parent)


def mesh_object(name: str, bm: bmesh.types.BMesh, material, parent=None, smooth: bool = True) -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        for poly in me.polygons:
            poly.use_smooth = True
    obj = bpy.data.objects.new(name, me)
    if material is not None:
        obj.data.materials.append(material)
    return link(obj, parent)


def bevel(obj, width: float, segments: int = 3, angle_deg: float = 40.0) -> None:
    mod = obj.modifiers.new("bevel", "BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = "ANGLE"
    mod.angle_limit = math.radians(angle_deg)
    mod.harden_normals = False
    wn = obj.modifiers.new("normals", "WEIGHTED_NORMAL")
    wn.keep_sharp = True


# ─────────────────────────────────────────────────────────────
# 2D outlines
# ─────────────────────────────────────────────────────────────

def superellipse(a: float, b: float, n: float, count: int = 128) -> list[tuple[float, float]]:
    pts = []
    for i in range(count):
        t = TAU * i / count
        c, s = math.cos(t), math.sin(t)
        x = a * math.copysign(abs(c) ** (2 / n), c)
        y = b * math.copysign(abs(s) ** (2 / n), s)
        pts.append((x, y))
    return pts


def circle(r: float, count: int = 128) -> list[tuple[float, float]]:
    return [(r * math.cos(TAU * i / count), r * math.sin(TAU * i / count)) for i in range(count)]


def rounded_polygon(sides: int, r: float, fillet: float, rot: float = 0.0, per_corner: int = 8) -> list[tuple[float, float]]:
    """Regular polygon with rounded corners (fillet radius in units)."""
    pts = []
    for k in range(sides):
        a = rot + TAU * k / sides
        corner = Vector((r * math.cos(a), r * math.sin(a)))
        inward = -corner.normalized()
        half = math.pi / sides
        # distance from corner to fillet centre along the bisector
        d = fillet / math.sin(math.pi / 2 - half)
        centre = corner + inward * d
        # the fillet arc spans the exterior angle, centred on the corner direction
        for j in range(per_corner + 1):
            t = a - half + 2 * half * (j / per_corner)
            pts.append((centre.x + fillet * math.cos(t), centre.y + fillet * math.sin(t)))
    return pts


# ─────────────────────────────────────────────────────────────
# 3D primitives from outlines
# ─────────────────────────────────────────────────────────────

def prism(outline: Sequence[tuple[float, float]], z0: float, z1: float, taper: float = 1.0) -> bmesh.types.BMesh:
    """Extrude a closed outline from z0 to z1; the top is scaled by `taper`."""
    bm = bmesh.new()
    bottom = [bm.verts.new((x, y, z0)) for x, y in outline]
    top = [bm.verts.new((x * taper, y * taper, z1)) for x, y in outline]
    n = len(outline)
    bm.faces.new(list(reversed(bottom)))
    bm.faces.new(top)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((bottom[i], bottom[j], top[j], top[i]))
    return bm


def ring(outer: Sequence[tuple[float, float]], inner: Sequence[tuple[float, float]], z0: float, z1: float,
         top_taper: float = 1.0) -> bmesh.types.BMesh:
    """Annulus between two outlines with equal point counts (bezels, rehauts)."""
    assert len(outer) == len(inner)
    bm = bmesh.new()
    n = len(outer)
    ob = [bm.verts.new((x, y, z0)) for x, y in outer]
    ot = [bm.verts.new((x * top_taper, y * top_taper, z1)) for x, y in outer]
    ib = [bm.verts.new((x, y, z0)) for x, y in inner]
    it = [bm.verts.new((x, y, z1)) for x, y in inner]
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((ob[i], ob[j], ot[j], ot[i]))
        bm.faces.new((it[i], it[j], ib[j], ib[i]))
        bm.faces.new((ot[i], ot[j], it[j], it[i]))
        bm.faces.new((ib[i], ib[j], ob[j], ob[i]))
    return bm


def box(sx: float, sy: float, sz: float, center=(0.0, 0.0, 0.0)) -> bmesh.types.BMesh:
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
    bmesh.ops.translate(bm, vec=center, verts=bm.verts)
    return bm


def cylinder(r: float, z0: float, z1: float, segments: int = 48, center=(0.0, 0.0)) -> bmesh.types.BMesh:
    pts = [(center[0] + x, center[1] + y) for x, y in circle(r, segments)]
    return prism(pts, z0, z1)


def disc_with_uv(outline: Sequence[tuple[float, float]], z: float, extent: float) -> bmesh.types.BMesh:
    """Flat face with planar UVs in [0,1] over ±extent, for dial textures."""
    bm = bmesh.new()
    verts = [bm.verts.new((x, y, z)) for x, y in outline]
    face = bm.faces.new(verts)
    uv = bm.loops.layers.uv.new("UVMap")
    for loop in face.loops:
        co = loop.vert.co
        loop[uv].uv = ((co.x / extent + 1) / 2, (co.y / extent + 1) / 2)
    return bm


def transform(bm: bmesh.types.BMesh, matrix: Matrix) -> bmesh.types.BMesh:
    bmesh.ops.transform(bm, matrix=matrix, verts=bm.verts)
    return bm


def merge(parts: Iterable[bmesh.types.BMesh]) -> bmesh.types.BMesh:
    out = bmesh.new()
    for part in parts:
        me = bpy.data.meshes.new("_tmp")
        part.to_mesh(me)
        part.free()
        out.from_mesh(me)
        bpy.data.meshes.remove(me)
    return out


# ─────────────────────────────────────────────────────────────
# Watch parts
# ─────────────────────────────────────────────────────────────

def hand(length: float, width: float, tail: float, thickness: float, z: float, angle: float, tip: str = "pointed") -> bmesh.types.BMesh:
    """A hand pointing to `angle` (radians, clockwise from 12)."""
    w = width / 2
    if tip == "pointed":
        outline = [(-w * 0.7, -tail), (w * 0.7, -tail), (w, length * 0.82), (0.0, length), (-w, length * 0.82)]
    elif tip == "leaf":
        outline = [(-w * 0.3, -tail), (w * 0.3, -tail), (w, length * 0.55), (0.0, length), (-w, length * 0.55)]
    else:  # needle
        outline = [(-w, -tail), (w, -tail), (w * 0.5, length), (-w * 0.5, length)]
    bm = prism(outline, z, z + thickness)
    return transform(bm, Matrix.Rotation(-angle, 4, "Z"))


def batons(count: int, r_outer: float, length: float, width: float, height: float, z: float,
           skip: Iterable[int] = (), double: Iterable[int] = ()) -> bmesh.types.BMesh:
    parts = []
    skip, double = set(skip), set(double)
    for k in range(count):
        if k in skip:
            continue
        a = TAU * k / count
        offsets = (-width * 0.75, width * 0.75) if k in double else (0.0,)
        for off in offsets:
            b = box(width, length, height, center=(off, r_outer - length / 2, z + height / 2))
            parts.append(transform(b, Matrix.Rotation(-a, 4, "Z")))
    return merge(parts)


def minute_track(r: float, z: float, ticks: int = 60, length: float = 0.06, width: float = 0.012) -> bmesh.types.BMesh:
    parts = []
    for k in range(ticks):
        a = TAU * k / ticks
        b = box(width, length, 0.004, center=(0.0, r - length / 2, z + 0.002))
        parts.append(transform(b, Matrix.Rotation(-a, 4, "Z")))
    return merge(parts)


def arc_rows(rows: int, start_y: float, radius: float, step: float, row_builder, direction: int = 1,
             z_top: float = 0.0) -> bmesh.types.BMesh:
    """Repeat a bracelet row along an arc that wraps down and away from the case.

    `row_builder(i)` returns a bmesh centred on the origin, lying in XY with its length along Y.
    Rows follow a circle of `radius` around an axis parallel to X, so the bracelet curls
    under the case like it does around a wrist.
    """
    parts = []
    for i in range(rows):
        theta = (i + 0.5) * step / radius
        y = start_y + direction * radius * math.sin(theta)
        z = z_top - radius * (1 - math.cos(theta))
        m = Matrix.Translation((0.0, y, z)) @ Matrix.Rotation(direction * -theta, 4, "X")
        parts.append(transform(row_builder(i), m))
    return merge(parts)


# ─────────────────────────────────────────────────────────────
# Materials
# ─────────────────────────────────────────────────────────────

def _principled(name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    return mat, nodes, mat.node_tree.links, bsdf


def metal(name: str, color, roughness: float = 0.16, anisotropic: float = 0.0, brushed: bool = False):
    mat, nodes, links, bsdf = _principled(name)
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = roughness
    if anisotropic:
        bsdf.inputs["Anisotropic"].default_value = anisotropic
    if brushed:
        # fine linear brushing: stretched noise into roughness and a whisper of bump
        tex = nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = 900.0
        tex.inputs["Detail"].default_value = 2.0
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (1.0, 0.02, 1.0)
        coord = nodes.new("ShaderNodeTexCoord")
        links.new(coord.outputs["Object"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        ramp = nodes.new("ShaderNodeMapRange")
        ramp.inputs["To Min"].default_value = roughness * 0.7
        ramp.inputs["To Max"].default_value = roughness * 1.6
        links.new(tex.outputs["Fac"], ramp.inputs["Value"])
        links.new(ramp.outputs["Result"], bsdf.inputs["Roughness"])
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.05
        links.new(tex.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def glass(name: str = "sapphire"):
    mat, nodes, links, bsdf = _principled(name)
    bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
    bsdf.inputs["Roughness"].default_value = 0.0
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    # Real sapphire is ~1.77, but watch crystals carry an anti-reflective coating; a low
    # effective IOR keeps the dial readable instead of mirroring the studio lights.
    bsdf.inputs["IOR"].default_value = 1.06
    bsdf.inputs["Coat Weight"].default_value = 0.0
    return mat


def matte(name: str, color, roughness: float = 0.5, emission: float = 0.0):
    mat, nodes, links, bsdf = _principled(name)
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*color, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission
    return mat


def dial(name: str, inner, outer, pattern: str, scale: float = 40.0, roughness: float = 0.35, metallic: float = 0.6):
    """Dial plate: radial colour falloff (inner → outer) plus a relief pattern.

    pattern: "stripes" (horizontal embossing), "tapisserie" (raised grid), "sunburst"
    (radial brushing), "grain" (fine silvered grain), "plain".
    """
    mat, nodes, links, bsdf = _principled(name)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    coord = nodes.new("ShaderNodeTexCoord")

    # radial gradient for the colour falloff toward the rehaut
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coord.outputs["UV"], sep.inputs["Vector"])
    centred = nodes.new("ShaderNodeVectorMath")
    centred.operation = "SUBTRACT"
    centred.inputs[1].default_value = (0.5, 0.5, 0.0)
    links.new(coord.outputs["UV"], centred.inputs[0])
    dist = nodes.new("ShaderNodeVectorMath")
    dist.operation = "LENGTH"
    links.new(centred.outputs["Vector"], dist.inputs[0])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.15
    ramp.color_ramp.elements[0].color = (*inner, 1)
    ramp.color_ramp.elements[1].position = 0.5
    ramp.color_ramp.elements[1].color = (*outer, 1)
    links.new(dist.outputs["Value"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    height = None
    if pattern == "stripes":
        wave = nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = "Y"
        wave.inputs["Scale"].default_value = scale
        wave.inputs["Distortion"].default_value = 0.0
        links.new(coord.outputs["UV"], wave.inputs["Vector"])
        height = wave.outputs["Fac"]
    elif pattern == "tapisserie":
        brick = nodes.new("ShaderNodeTexChecker")
        brick.inputs["Scale"].default_value = scale
        grid = nodes.new("ShaderNodeTexBrick")
        grid.offset = 0.0
        grid.squash = 1.0
        grid.inputs["Scale"].default_value = scale
        grid.inputs["Mortar Size"].default_value = 0.06
        grid.inputs["Brick Width"].default_value = 1.0
        grid.inputs["Row Height"].default_value = 1.0
        links.new(coord.outputs["UV"], grid.inputs["Vector"])
        height = grid.outputs["Fac"]
    elif pattern == "sunburst":
        # angle around the centre drives very fine radial streaks
        atan = nodes.new("ShaderNodeMath")
        atan.operation = "ARCTAN2"
        sep2 = nodes.new("ShaderNodeSeparateXYZ")
        links.new(centred.outputs["Vector"], sep2.inputs["Vector"])
        links.new(sep2.outputs["Y"], atan.inputs[0])
        links.new(sep2.outputs["X"], atan.inputs[1])
        noise = nodes.new("ShaderNodeTexNoise")
        noise.noise_dimensions = "1D"
        noise.inputs["Scale"].default_value = 400.0
        links.new(atan.outputs["Value"], noise.inputs["W"])
        height = noise.outputs["Fac"]
        bsdf.inputs["Anisotropic"].default_value = 0.8
    elif pattern == "grain":
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 1500.0
        links.new(coord.outputs["UV"], noise.inputs["Vector"])
        height = noise.outputs["Fac"]

    if height is not None:
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.35 if pattern in {"stripes", "tapisserie"} else 0.08
        bump.inputs["Distance"].default_value = 0.002
        links.new(height, bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def leather(name: str, color):
    """Alligator-style strap: Voronoi scales as bump and slight colour variation."""
    mat, nodes, links, bsdf = _principled(name)
    bsdf.inputs["Roughness"].default_value = 0.42
    bsdf.inputs["Coat Weight"].default_value = 0.25
    coord = nodes.new("ShaderNodeTexCoord")
    vor = nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = 26.0
    links.new(coord.outputs["Object"], vor.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (color[0] * 0.35, color[1] * 0.35, color[2] * 0.35, 1)
    ramp.color_ramp.elements[1].position = 0.12
    ramp.color_ramp.elements[1].color = (*color, 1)
    links.new(vor.outputs["Distance"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    links.new(vor.outputs["Distance"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def text_mesh(body: str, size: float, z: float, center=(0.0, 0.0), material=None, parent=None, extrude: float = 0.004):
    curve = bpy.data.curves.new("txt", "FONT")
    curve.body = body
    curve.size = size
    curve.extrude = extrude
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    obj = bpy.data.objects.new("txt_" + body, curve)
    obj.location = (center[0], center[1], z)
    if material is not None:
        curve.materials.append(material)
    return link(obj, parent)
