"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { ContactShadows, Environment } from "@react-three/drei";
import { Suspense, useEffect, useRef, useState } from "react";
import * as THREE from "three";

import { Watch3D } from "./Models";

/** Cubic ease-in-out */
const ease = (k: number) =>
  k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;

/** Interpolate between keyframes. t ∈ [0, 1]. */
function keyframe(
  t: number,
  stops: {
    at: number;
    x: number;
    y: number;
    z?: number;
    s?: number;
    rotY?: number;
    rotX?: number;
  }[]
) {
  const clamped = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i];
    const b = stops[i + 1];
    if (clamped >= a.at && clamped <= b.at) {
      const k = (clamped - a.at) / Math.max(b.at - a.at, 1e-6);
      const e = ease(k);
      return {
        x: a.x + (b.x - a.x) * e,
        y: a.y + (b.y - a.y) * e,
        z: (a.z ?? 0) + ((b.z ?? 0) - (a.z ?? 0)) * e,
        s: (a.s ?? 1) + ((b.s ?? 1) - (a.s ?? 1)) * e,
        rotY: (a.rotY ?? 0) + ((b.rotY ?? 0) - (a.rotY ?? 0)) * e,
        rotX: (a.rotX ?? 0) + ((b.rotX ?? 0) - (a.rotX ?? 0)) * e,
      };
    }
  }
  const last = stops[stops.length - 1];
  return {
    x: last.x,
    y: last.y,
    z: last.z ?? 0,
    s: last.s ?? 1,
    rotY: last.rotY ?? 0,
    rotX: last.rotX ?? 0,
  };
}

function SceneObjects({ scrollRef }: { scrollRef: React.MutableRefObject<number> }) {
  const watchRef = useRef<THREE.Group>(null);
  const { viewport } = useThree();

  // Smoothed scroll progress (Apple-style catch-up)
  const smoothed = useRef(0);

  useFrame((_, delta) => {
    // 1. Ease toward the real scroll value — creates a subtle "coasting" feel
    const target = scrollRef.current;
    const k = 1 - Math.exp(-delta * 8); // time-correct lerp
    smoothed.current += (target - smoothed.current) * k;
    const t = smoothed.current;

    if (!watchRef.current) return;

    // Viewport half-dimensions
    const halfW = viewport.width / 2;
    const halfH = viewport.height / 2;

    /* ─── CHOREOGRAPHY ─────────────────────────────────────────────
       Pure scroll-driven, Apple-product-page style.
       - Product is always prominent, never in a corner
       - 2 full rotations across the whole page so every angle gets screen time
       - Scale breathes bigger at "reveal" moments
       ─────────────────────────────────────────────────────────── */
    const stage = keyframe(t, [
      // Hero — sits on the RIGHT, dial toward camera.
      { at: 0.00, x:  halfW * 0.52, y:  halfH * 0.08, s: 1.30, rotY:  0.00, rotX:  0.10 },
      // Brand marquee — quick swing to LEFT, slight tilt
      { at: 0.14, x: -halfW * 0.52, y:  halfH * 0.00, s: 1.15, rotY:  1.00, rotX: -0.05 },
      // How it works — back to RIGHT, ¼ rotation
      { at: 0.30, x:  halfW * 0.50, y: -halfH * 0.05, s: 1.25, rotY:  2.10, rotX:  0.15 },
      // Philosophy — LEFT, ½ rotation (showing side)
      { at: 0.48, x: -halfW * 0.48, y:  halfH * 0.08, s: 1.20, rotY:  3.20, rotX: -0.10 },
      // Categories — RIGHT, ¾ rotation
      { at: 0.66, x:  halfW * 0.52, y:  halfH * 0.00, s: 1.30, rotY:  4.40, rotX:  0.05 },
      // Recent listings — LEFT
      { at: 0.84, x: -halfW * 0.50, y: -halfH * 0.08, s: 1.30, rotY:  5.50, rotX:  0.10 },
      // Closing CTA — commanding centre, biggest, full rotation (dial back to camera)
      { at: 1.00, x:  0,             y: -halfH * 0.02, s: 1.85, rotY:  6.28, rotX:  0.00 },
    ]);

    watchRef.current.position.set(stage.x, stage.y, stage.z);
    watchRef.current.scale.setScalar(stage.s);
    watchRef.current.rotation.set(stage.rotX, stage.rotY, 0);
  });

  return (
    <>
      {/* Apple-style studio lighting */}
      <ambientLight intensity={0.55} />
      <directionalLight
        position={[6, 8, 6]}
        intensity={1.35}
        color="#ffffff"
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      <directionalLight position={[-6, 4, -3]} intensity={0.8} color="#9fc7ff" />
      <pointLight position={[0, -4, 4]} intensity={0.6} color="#FF3B00" />
      <spotLight
        position={[0, 6, 6]}
        angle={0.5}
        penumbra={1}
        intensity={1.0}
        color="#ffffff"
      />

      <Environment preset="studio" />

      <group ref={watchRef}>
        <Watch3D scale={1.0} />
        {/* Soft contact shadow grounds the product — Apple uses this constantly */}
        <ContactShadows
          position={[0, -1.4, 0]}
          opacity={0.55}
          scale={6}
          blur={2.2}
          far={2.5}
          color="#000"
        />
      </group>
    </>
  );
}

export function ScrollObjects() {
  const scrollRef = useRef(0);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);

    let raf = 0;
    const update = () => {
      const y =
        typeof window.__hvScroll === "number"
          ? window.__hvScroll
          : window.scrollY;
      const max =
        typeof window.__hvScrollMax === "number"
          ? window.__hvScrollMax
          : document.documentElement.scrollHeight - window.innerHeight;
      scrollRef.current = max > 0 ? y / max : 0;
      raf = requestAnimationFrame(update);
    };
    raf = requestAnimationFrame(update);
    return () => cancelAnimationFrame(raf);
  }, []);

  if (!mounted) return null;
  if (typeof window !== "undefined") {
    if (window.matchMedia("(max-width: 640px)").matches) return null;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return null;
  }

  return (
    <div
      className="pointer-events-none fixed inset-0 z-[30]"
      aria-hidden
    >
      <Canvas
        camera={{ position: [0, 0, 6], fov: 45 }}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: "high-performance",
        }}
        shadows
        dpr={[1, 2]}
        className="!bg-transparent"
      >
        <Suspense fallback={null}>
          <SceneObjects scrollRef={scrollRef} />
        </Suspense>
      </Canvas>
    </div>
  );
}
