"""Procedural watch models, built from scratch in Blender.

Each model follows the silhouette and finishing of a famous reference from one of the five
houses HypeVault covers, without logos or text: these are studio models "in the style of",
not replicas. The Richard Mille RM011 comes from the CC BY-NC model already in the frontend.

Every builder returns the watch's root empty (dial facing +Z, 12 o'clock toward +Y).
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Matrix

import kit
from kit import TAU

HAND_TIME = (10, 9, 35)  # the classic 10:09 display, seconds at 35


def _angles():
    h, m, s = HAND_TIME
    return (TAU * ((h % 12) + m / 60) / 12, TAU * m / 60, TAU * s / 60)


def _crown(root, x: float, z: float, r: float, length: float, mat):
    bm = kit.cylinder(r, 0.0, length, segments=40)
    kit.transform(bm, Matrix.Translation((x, 0.0, z)) @ Matrix.Rotation(math.pi / 2, 4, "Y"))
    crown = kit.mesh_object("crown", bm, mat, root)
    kit.bevel(crown, 0.01, 2)
    # knurling: fine ridges around the crown
    ridges = []
    for k in range(24):
        a = TAU * k / 24
        b = kit.box(length * 0.8, 0.012, 0.03, center=(length * 0.5, 0.0, r))
        ridges.append(kit.transform(b, Matrix.Rotation(a, 4, "X")))
    bm2 = kit.merge(ridges)
    kit.transform(bm2, Matrix.Translation((x, 0.0, z)))
    kit.mesh_object("crown_knurl", bm2, mat, root)


def _hands(root, z: float, mat, lume, center=(0.0, 0.0), scale: float = 1.0, style: str = "pointed", seconds_mat=None):
    ha, ma, sa = _angles()
    move = Matrix.Translation((center[0], center[1], 0.0))
    hour = kit.transform(kit.hand(0.78 * scale, 0.13 * scale, 0.12 * scale, 0.02, z, ha, style), move)
    minute = kit.transform(kit.hand(1.15 * scale, 0.10 * scale, 0.14 * scale, 0.02, z + 0.025, ma, style), move)
    kit.mesh_object("hand_hour", hour, mat, root)
    kit.mesh_object("hand_minute", minute, mat, root)
    if lume is not None:
        lh = kit.transform(kit.hand(0.6 * scale, 0.05 * scale, -0.15 * scale, 0.006, z + 0.02, ha, "needle"), move)
        lm = kit.transform(kit.hand(0.95 * scale, 0.04 * scale, -0.18 * scale, 0.006, z + 0.045, ma, "needle"), move)
        kit.mesh_object("lume_hour", lh, lume, root)
        kit.mesh_object("lume_minute", lm, lume, root)
    if seconds_mat is not None:
        sec = kit.transform(kit.hand(1.25 * scale, 0.022 * scale, 0.32 * scale, 0.01, z + 0.05, sa, "needle"), move)
        kit.mesh_object("hand_seconds", sec, seconds_mat, root)
    cap = kit.cylinder(0.05 * scale, z, z + 0.075, 24, center=center)
    kit.mesh_object("hand_cap", cap, mat, root)


def _date_window(root, x: float, y: float, z: float, frame_mat, digits: str = "19", w: float = 0.2, h: float = 0.16):
    frame = kit.box(w + 0.04, h + 0.04, 0.02, center=(x, y, z + 0.01))
    kit.mesh_object("date_frame", frame, frame_mat, root)
    white = kit.matte("date_disc", (0.94, 0.93, 0.9), 0.5)
    kit.mesh_object("date_disc", kit.box(w, h, 0.01, center=(x, y, z + 0.016)), white, root)
    ink = kit.matte("date_ink", (0.03, 0.03, 0.04), 0.4)
    kit.text_mesh(digits, h * 0.85, z + 0.021, center=(x, y), material=ink, parent=root, extrude=0.001)


def _crystal(root, outline, z0: float, z1: float):
    glass = kit.glass()
    bm = kit.prism(outline, z0, z1, taper=0.995)
    obj = kit.mesh_object("crystal", bm, glass, root)
    kit.bevel(obj, 0.012, 3, 30)


def _caseback(root, r: float, mat):
    parts = [kit.cylinder(r, -0.06, 0.02, 64)]
    for k in range(3):
        rr = r * (0.9 - 0.18 * k)
        parts.append(kit.cylinder(rr, -0.075 - 0.004 * k, -0.06, 64))
    obj = kit.mesh_object("caseback", kit.merge(parts), mat, root)
    kit.bevel(obj, 0.01, 2)


# ─────────────────────────────────────────────────────────────
# Patek Philippe — in the style of the Nautilus 5711
# ─────────────────────────────────────────────────────────────

def nautilus(name: str = "patek_nautilus") -> bpy.types.Object:
    root = kit.empty(name)
    steel = kit.metal("pp_polished", (0.80, 0.80, 0.81), 0.07)
    brushed = kit.metal("pp_brushed", (0.76, 0.77, 0.78), 0.26, anisotropic=0.6, brushed=True)
    white_gold = kit.metal("pp_white_gold", (0.92, 0.92, 0.93), 0.05)
    lume = kit.matte("pp_lume", (0.9, 0.95, 0.86), 0.4, emission=0.08)
    dial = kit.dial("pp_dial", inner=(0.035, 0.11, 0.26), outer=(0.004, 0.018, 0.06), pattern="stripes", scale=70)

    case = kit.mesh_object("case", kit.prism(kit.superellipse(1.86, 1.80, 3.4), 0.0, 0.5, taper=0.975), brushed, root)
    kit.bevel(case, 0.035, 3)
    # the "ears" — hinges at 3 and 9 that give the porthole case its silhouette
    for side in (-1, 1):
        ear = kit.mesh_object(f"ear_{side}", kit.box(0.28, 0.62, 0.4, center=(side * 1.93, 0.0, 0.3)), steel, root)
        kit.bevel(ear, 0.09, 4, 30)
    bezel_out = kit.superellipse(1.70, 1.64, 2.9)
    bezel_in = kit.superellipse(1.40, 1.36, 2.6)
    bezel = kit.mesh_object("bezel", kit.ring(bezel_out, bezel_in, 0.48, 0.72, top_taper=0.975), steel, root)
    kit.bevel(bezel, 0.02, 3, 25)

    dial_outline = kit.superellipse(1.42, 1.38, 2.6)
    kit.mesh_object("dial", kit.disc_with_uv(dial_outline, 0.52, 1.45), dial, root, smooth=False)
    kit.mesh_object("indices", kit.batons(12, 1.24, 0.30, 0.075, 0.035, 0.52, skip=[3]), white_gold, root)
    kit.mesh_object("index_lume", kit.batons(12, 1.22, 0.24, 0.03, 0.04, 0.52, skip=[3]), lume, root)
    kit.mesh_object("minute_track", kit.minute_track(1.34, 0.52, length=0.05), white_gold, root)
    _date_window(root, 1.0, 0.0, 0.52, white_gold)
    _hands(root, 0.57, white_gold, lume, style="pointed", seconds_mat=white_gold)
    _crystal(root, kit.superellipse(1.44, 1.40, 2.6), 0.68, 0.73)
    _crown(root, 1.98, 0.28, 0.17, 0.2, steel)
    _caseback(root, 1.5, brushed)

    def row(i: int):
        w = 2.25 - 0.5 * (i / 13)
        centre = kit.box(w * 0.36, 0.5, 0.26)
        outer_l = kit.box(w * 0.3, 0.5, 0.24, center=(-w * 0.345, 0, -0.01))
        outer_r = kit.box(w * 0.3, 0.5, 0.24, center=(w * 0.345, 0, -0.01))
        return kit.merge([centre, outer_l, outer_r])

    for direction, label in ((1, "bracelet_top"), (-1, "bracelet_bottom")):
        bm = kit.arc_rows(13, direction * 1.66, 2.6, 0.52, row, direction=direction, z_top=0.3)
        obj = kit.mesh_object(label, bm, brushed, root)
        kit.bevel(obj, 0.025, 2)
    return root


# ─────────────────────────────────────────────────────────────
# Audemars Piguet — in the style of the Royal Oak 15202
# ─────────────────────────────────────────────────────────────

def royal_oak(name: str = "ap_royal_oak") -> bpy.types.Object:
    root = kit.empty(name)
    steel = kit.metal("ap_polished", (0.80, 0.80, 0.82), 0.06)
    brushed = kit.metal("ap_brushed", (0.76, 0.77, 0.79), 0.3, anisotropic=0.7, brushed=True)
    white_gold = kit.metal("ap_white_gold", (0.92, 0.92, 0.94), 0.05)
    lume = kit.matte("ap_lume", (0.9, 0.95, 0.86), 0.4, emission=0.08)
    dial = kit.dial("ap_dial", inner=(0.03, 0.08, 0.2), outer=(0.006, 0.02, 0.07), pattern="tapisserie", scale=34)

    case = kit.mesh_object("case", kit.prism(kit.superellipse(1.98, 1.86, 3.0), 0.0, 0.46, taper=0.97), brushed, root)
    kit.bevel(case, 0.03, 3)
    octagon = kit.rounded_polygon(8, 1.78, 0.2, rot=math.pi / 8, per_corner=8)
    inner = kit.circle(1.36, len(octagon))
    bezel = kit.mesh_object("bezel", kit.ring(octagon, inner, 0.44, 0.78, top_taper=0.955), brushed, root)
    kit.bevel(bezel, 0.018, 3, 25)
    # eight hexagonal white-gold screws, one at each corner of the octagon
    screws = []
    for k in range(8):
        a = math.pi / 8 + TAU * k / 8
        cx, cy = 1.47 * math.cos(a), 1.47 * math.sin(a)
        hexagon = [(cx + 0.085 * math.cos(TAU * j / 6), cy + 0.085 * math.sin(TAU * j / 6)) for j in range(6)]
        screws.append(kit.prism(hexagon, 0.77, 0.805))
        screws.append(kit.box(0.13, 0.018, 0.012, center=(cx, cy, 0.81)))
    s = kit.mesh_object("screws", kit.merge(screws), white_gold, root)
    kit.bevel(s, 0.008, 2)

    kit.mesh_object("dial", kit.disc_with_uv(kit.circle(1.37), 0.48, 1.37), dial, root, smooth=False)
    kit.mesh_object("indices", kit.batons(12, 1.2, 0.3, 0.07, 0.035, 0.48, skip=[3]), white_gold, root)
    kit.mesh_object("index_lume", kit.batons(12, 1.18, 0.24, 0.028, 0.04, 0.48, skip=[3]), lume, root)
    kit.mesh_object("minute_track", kit.minute_track(1.31, 0.48, length=0.05), white_gold, root)
    _date_window(root, 0.98, 0.0, 0.48, white_gold)
    _hands(root, 0.53, white_gold, lume, style="pointed", seconds_mat=white_gold)
    _crystal(root, kit.circle(1.38), 0.7, 0.76)
    _crown(root, 1.98, 0.3, 0.16, 0.2, steel)
    _caseback(root, 1.5, brushed)

    def row(i: int):
        w = 2.3 - 0.55 * (i / 14)
        big = kit.box(w, 0.36, 0.22)
        link_l = kit.box(w * 0.18, 0.16, 0.2, center=(-w * 0.22, 0.26, 0))
        link_r = kit.box(w * 0.18, 0.16, 0.2, center=(w * 0.22, 0.26, 0))
        return kit.merge([big, link_l, link_r])

    for direction, label in ((1, "bracelet_top"), (-1, "bracelet_bottom")):
        bm = kit.arc_rows(14, direction * 1.7, 2.6, 0.5, row, direction=direction, z_top=0.28)
        obj = kit.mesh_object(label, bm, brushed, root)
        kit.bevel(obj, 0.03, 2)
    return root


# ─────────────────────────────────────────────────────────────
# A. Lange & Söhne — in the style of the Lange 1
# ─────────────────────────────────────────────────────────────

def lange_one(name: str = "lange_one") -> bpy.types.Object:
    root = kit.empty(name)
    gold = kit.metal("al_pink_gold", (0.98, 0.70, 0.58), 0.09)
    gold_satin = kit.metal("al_pink_gold_satin", (0.96, 0.68, 0.56), 0.22, brushed=True)
    dial = kit.dial("al_dial", inner=(0.86, 0.84, 0.8), outer=(0.74, 0.72, 0.68), pattern="grain", roughness=0.42, metallic=0.85)
    ink = kit.matte("al_ink", (0.05, 0.05, 0.06), 0.4)
    blued = kit.metal("al_blued", (0.1, 0.16, 0.45), 0.12)

    case = kit.mesh_object("case", kit.prism(kit.circle(1.92), 0.0, 0.38, taper=0.98), gold_satin, root)
    kit.bevel(case, 0.03, 3)
    bezel = kit.mesh_object("bezel", kit.ring(kit.circle(1.9), kit.circle(1.7), 0.36, 0.6, top_taper=0.97), gold, root)
    kit.bevel(bezel, 0.02, 3, 20)
    for sx in (-1, 1):
        for sy in (-1, 1):
            lug = kit.box(0.26, 0.7, 0.3, center=(sx * 1.16, sy * 1.95, 0.2))
            kit.transform(lug, Matrix.Translation((0, sy * 0.0, 0)))
            obj = kit.mesh_object(f"lug_{sx}_{sy}", lug, gold, root)
            kit.bevel(obj, 0.08, 4, 30)

    kit.mesh_object("dial", kit.disc_with_uv(kit.circle(1.7), 0.4, 1.7), dial, root, smooth=False)
    # off-centre hours and minutes: the Lange 1's signature asymmetric layout
    main = (-0.42, 0.04)
    chapter = kit.ring(kit.circle(0.98, 96), kit.circle(0.93, 96), 0.40, 0.412)
    kit.transform(chapter, Matrix.Translation((main[0], main[1], 0)))
    kit.mesh_object("main_ring", chapter, gold, root)
    idx = kit.batons(12, 0.9, 0.16, 0.045, 0.03, 0.40)
    kit.transform(idx, Matrix.Translation((main[0], main[1], 0)))
    kit.mesh_object("indices", idx, gold, root)
    # small seconds, lower left, overlapping the main chapter
    small = (-0.3, -1.02)
    sring = kit.ring(kit.circle(0.42, 64), kit.circle(0.39, 64), 0.40, 0.41)
    kit.transform(sring, Matrix.Translation((small[0], small[1], 0)))
    kit.mesh_object("seconds_ring", sring, gold, root)
    sticks = kit.minute_track(0.38, 0.40, ticks=60, length=0.04, width=0.008)
    kit.transform(sticks, Matrix.Translation((small[0], small[1], 0)))
    kit.mesh_object("seconds_track", sticks, ink, root)
    # outsize date at upper right: two windows in a gold frame
    for k, digit in enumerate("19"):
        _date_window(root, 0.55 + k * 0.29, 0.78, 0.40, gold, digits=digit, w=0.24, h=0.3)
    # power reserve arc on the right
    arc = []
    for k in range(9):
        a = math.radians(-60 + 15 * k)
        b = kit.box(0.012, 0.08, 0.004, center=(0.0, 0.5, 0.403))
        arc.append(kit.transform(b, Matrix.Translation((1.0, -0.25, 0)) @ Matrix.Rotation(-a + math.pi / 2, 4, "Z")))
    kit.mesh_object("power_reserve", kit.merge(arc), ink, root)
    _hands(root, 0.43, gold, None, center=main, scale=0.78, style="leaf")
    sec = kit.transform(kit.hand(0.36, 0.018, 0.08, 0.008, 0.44, TAU * 35 / 60, "needle"), Matrix.Translation((small[0], small[1], 0)))
    kit.mesh_object("hand_seconds", sec, blued, root)
    _crystal(root, kit.circle(1.72), 0.55, 0.61)
    _crown(root, 1.94, 0.26, 0.15, 0.2, gold)
    _caseback(root, 1.55, gold_satin)

    strap = kit.leather("al_strap", (0.16, 0.07, 0.035))

    def segment(i: int):
        w = 1.98 - 0.4 * (i / 34)
        return kit.box(w, 0.2, 0.12)  # overlapping segments read as one continuous strap

    for direction, label in ((1, "strap_top"), (-1, "strap_bottom")):
        bm = kit.arc_rows(34, direction * 2.2, 2.5, 0.145, segment, direction=direction, z_top=0.26)
        kit.mesh_object(label, bm, strap, root)
    return root


# ─────────────────────────────────────────────────────────────
# Vacheron Constantin — in the style of the Overseas
# ─────────────────────────────────────────────────────────────

def overseas(name: str = "vc_overseas") -> bpy.types.Object:
    root = kit.empty(name)
    steel = kit.metal("vc_polished", (0.80, 0.80, 0.82), 0.06)
    brushed = kit.metal("vc_brushed", (0.76, 0.77, 0.79), 0.28, anisotropic=0.7, brushed=True)
    white_gold = kit.metal("vc_white_gold", (0.92, 0.92, 0.94), 0.05)
    lume = kit.matte("vc_lume", (0.9, 0.95, 0.86), 0.4, emission=0.08)
    dial = kit.dial("vc_dial", inner=(0.05, 0.16, 0.42), outer=(0.01, 0.04, 0.14), pattern="sunburst", roughness=0.22, metallic=0.7)

    case = kit.mesh_object("case", kit.prism(kit.circle(2.02), 0.0, 0.44, taper=0.975), brushed, root)
    kit.bevel(case, 0.03, 3)
    # bezel with six notches — the half Maltese cross
    count = 180
    outer = []
    for i in range(count):
        t = TAU * i / count
        r = 1.88 - 0.075 * abs(math.cos(3 * t)) ** 10
        outer.append((r * math.cos(t), r * math.sin(t)))
    bezel = kit.mesh_object("bezel", kit.ring(outer, kit.circle(1.56, count), 0.42, 0.7, top_taper=0.965), steel, root)
    kit.bevel(bezel, 0.018, 3, 25)
    for side in (-1, 1):
        shoulder = kit.mesh_object(f"shoulder_{side}", kit.box(1.3, 0.5, 0.42, center=(0.0, side * 1.95, 0.28)), brushed, root)
        kit.bevel(shoulder, 0.08, 3, 30)

    kit.mesh_object("dial", kit.disc_with_uv(kit.circle(1.56), 0.46, 1.56), dial, root, smooth=False)
    kit.mesh_object("indices", kit.batons(12, 1.36, 0.3, 0.075, 0.035, 0.46, skip=[3], double=[0]), white_gold, root)
    kit.mesh_object("index_lume", kit.batons(12, 1.34, 0.24, 0.03, 0.04, 0.46, skip=[3]), lume, root)
    kit.mesh_object("minute_track", kit.minute_track(1.5, 0.46, length=0.05), white_gold, root)
    _date_window(root, 1.13, 0.0, 0.46, white_gold)
    _hands(root, 0.51, white_gold, lume, scale=1.1, style="pointed", seconds_mat=white_gold)
    _crystal(root, kit.circle(1.58), 0.64, 0.7)
    _crown(root, 2.05, 0.3, 0.18, 0.22, steel)
    _caseback(root, 1.6, brushed)

    def row(i: int):
        w = 2.1 - 0.45 * (i / 13)
        # half-Maltese-cross centre link between two outer links
        centre = kit.prism([(-w * 0.2, -0.24), (w * 0.2, -0.24), (w * 0.12, 0.24), (-w * 0.12, 0.24)], -0.12, 0.12)
        outer_l = kit.box(w * 0.3, 0.46, 0.22, center=(-w * 0.36, 0, 0))
        outer_r = kit.box(w * 0.3, 0.46, 0.22, center=(w * 0.36, 0, 0))
        return kit.merge([centre, outer_l, outer_r])

    for direction, label in ((1, "bracelet_top"), (-1, "bracelet_bottom")):
        bm = kit.arc_rows(13, direction * 2.15, 2.6, 0.5, row, direction=direction, z_top=0.28)
        obj = kit.mesh_object(label, bm, brushed, root)
        kit.bevel(obj, 0.025, 2)
    return root


# ─────────────────────────────────────────────────────────────
# Richard Mille — the RM011 model (Denis V, Sketchfab, CC BY-NC 4.0)
# ─────────────────────────────────────────────────────────────

RM011_GLB = Path(__file__).resolve().parents[1] / "frontend" / "public" / "richard_mille_rm011_low_poly.glb"


def rm011(name: str = "rm_rm011") -> bpy.types.Object:
    root = kit.empty(name)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(RM011_GLB))
    imported = [o for o in bpy.data.objects if o not in before]
    tops = [o for o in imported if o.parent is None]
    holder = kit.empty("rm_holder", root)
    for o in tops:
        o.parent = holder
    # normalise to ~4.4 units tall, centred, dial up
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ v.co for o in imported if o.type == "MESH" for v in o.data.vertices]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    size = max(hi[i] - lo[i] for i in range(3))
    s = 6.4 / size  # the longest side includes the strap
    holder.scale = (s, s, s)
    holder.location = tuple(-(lo[i] + hi[i]) / 2 * s for i in range(3))
    return root


BUILDERS = {
    "patek_nautilus": nautilus,
    "ap_royal_oak": royal_oak,
    "lange_one": lange_one,
    "vc_overseas": overseas,
    "rm_rm011": rm011,
}
