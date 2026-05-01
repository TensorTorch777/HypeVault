"use client";

import { useGLTF } from "@react-three/drei";
import { useEffect, useMemo } from "react";
import * as THREE from "three";

const WATCH_MODEL_URL = "/richard_mille_rm011_low_poly.glb";

/* ─────────────────────────────────────────────────────────────
   WATCH — real Richard Mille RM011 GLB
   (credit: Denis V on Sketchfab, licensed CC BY-NC 4.0)
   ───────────────────────────────────────────────────────────── */

export function Watch3D({ scale = 1 }: { scale?: number }) {
  const { scene } = useGLTF(WATCH_MODEL_URL) as unknown as { scene: THREE.Group };

  // Clone so each instance owns its materials, then auto-center + normalise size.
  // We wrap the raw GLB in an inner group so we can orient the dial toward the
  // camera while the outer group's rotation is controlled by ScrollObjects.
  const prepared = useMemo(() => {
    const clone = scene.clone(true);
    const box = new THREE.Box3().setFromObject(clone);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    clone.position.sub(center);
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    clone.scale.setScalar(1.6 / maxDim); // fit within 1.6 world-unit bounding box
    return clone;
  }, [scene]);

  // Bump material realism on all meshes + enable shadows
  useEffect(() => {
    prepared.traverse((o: THREE.Object3D) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      m.castShadow = true;
      m.receiveShadow = true;
      const mat = m.material as THREE.MeshStandardMaterial;
      if (mat && "metalness" in mat) {
        mat.metalness = Math.min(1, (mat.metalness ?? 0.6) + 0.15);
        mat.roughness = Math.max(0.15, (mat.roughness ?? 0.5) - 0.1);
      }
    });
  }, [prepared]);

  return (
    <group scale={scale}>
      <primitive object={prepared} />
    </group>
  );
}

// Pre-load so first scroll doesn't stutter
useGLTF.preload(WATCH_MODEL_URL);
