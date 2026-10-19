"""Render the HypeVault watch models with Cycles.

    blender -b -P render/scene.py -- --watch patek_nautilus --shot hero --out OUT [--device GPU]
    blender -b -P render/scene.py -- --watch ap_royal_oak --shot turntable --frames 72 --out OUT

Shots
  hero       one still, dial angled toward camera, transparent background
  turntable  `--frames` stills of the watch turning a full revolution (scroll-scrubbed on the site)
  flat       dial straight-on, for cards and thumbnails

The studio is lit like a Vice City dusk: peach key, pink rim, lavender sky in the
reflections. The background is transparent; the website supplies the backdrop.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kit  # noqa: E402
import watches  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", required=True, choices=sorted(watches.BUILDERS))
    ap.add_argument("--shot", default="hero", choices=["hero", "turntable", "flat"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=1280)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--blend", help="also save the scene as a .blend file")
    return ap.parse_args(argv)


def dusk_world() -> None:
    world = bpy.data.worlds.new("dusk")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    bg = nodes["Background"]
    coord = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coord.outputs["Generated"], sep.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (0.01, 0.012, 0.03, 1)        # navy ground
    els[1].position, els[1].color = 1.0, (0.30, 0.22, 0.55, 1)         # lavender sky
    mid = els.new(0.48)
    mid.color = (0.95, 0.48, 0.32, 1)                                   # peach horizon
    pink = els.new(0.6)
    pink.color = (0.85, 0.36, 0.55, 1)                                  # pink band above it
    links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.55


def area(name, location, target, size, power, color, shape="RECTANGLE", size_y=None):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = power
    data.color = color
    data.shape = shape
    data.size = size
    if size_y is not None:
        data.size_y = size_y
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    direction = Vector(target) - Vector(location)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(obj)
    return obj


def studio() -> None:
    dusk_world()
    area("key", (-11, -2, 6), (0, 0, 0), 6, 1700, (1.0, 0.74, 0.58))
    area("rim", (8, 5, 3), (0, 0, 0), 5, 1400, (1.0, 0.42, 0.66))
    area("strip", (0, 4, 10), (0, 0, 0), 12, 500, (0.95, 0.93, 1.0), size_y=0.8)
    area("kicker", (2, -9, -4), (0, 0, 0), 4, 300, (0.62, 0.5, 1.0))


def camera(distance: float, height: float, lens: float = 85, fstop: float = 5.6) -> None:
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = lens
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = fstop
    cam = bpy.data.objects.new("cam", cam_data)
    cam.location = (0, -distance, height)
    cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam_data.dof.focus_distance = cam.location.length
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam


def render_settings(args) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.max_bounces = 12
    scene.cycles.glossy_bounces = 8
    scene.cycles.transmission_bounces = 8
    scene.render.film_transparent = True
    scene.render.resolution_x = args.res
    scene.render.resolution_y = args.res
    scene.render.image_settings.file_format = "WEBP"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.quality = args.quality
    scene.view_settings.view_transform = "AgX"
    for look in ("AgX - Medium High Contrast", "AgX - Punchy", "None"):
        try:
            scene.view_settings.look = look
            break
        except TypeError:
            continue
    if args.device == "GPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type == backend]
                if gpus:
                    for d in prefs.devices:
                        d.use = d.type == backend
                    scene.cycles.device = "GPU"
                    print(f"[render] {backend}: {[d.name for d in gpus]}")
                    break
            except TypeError:
                continue
        else:
            print("[render] no GPU backend found, using CPU")
            scene.cycles.device = "CPU"
    else:
        scene.cycles.device = "CPU"


def main() -> None:
    args = parse_args()
    kit.reset_scene()
    studio()
    render_settings(args)
    root = watches.BUILDERS[args.watch]()

    # Pose: a pivot spins the watch about the vertical axis; the watch stands on it.
    pivot = kit.empty("pivot")
    root.parent = pivot
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.shot == "flat":
        root.rotation_euler = (math.radians(90), 0, 0)
        camera(17, 0, lens=85, fstop=8)
    elif args.shot == "hero":
        root.rotation_euler = (math.radians(68), 0, 0)
        pivot.rotation_euler = (0, 0, math.radians(-24))
        camera(17, 2.2, lens=85, fstop=4)
    else:
        root.rotation_euler = (math.radians(84), 0, 0)
        camera(17.5, 1.6, lens=85, fstop=6.3)

    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.blend).resolve()))

    scene = bpy.context.scene
    if args.shot == "turntable":
        for i in range(args.frames):
            pivot.rotation_euler = (0, 0, math.radians(-30) + math.tau * i / args.frames)
            scene.render.filepath = str(out / f"f_{i:03d}.webp")
            bpy.ops.render.render(write_still=True)
            print(f"[render] frame {i + 1}/{args.frames}", flush=True)
    else:
        scene.render.filepath = str(out / f"{args.shot}.webp")
        bpy.ops.render.render(write_still=True)
    print("[render] done", flush=True)


if __name__ == "__main__":
    main()
