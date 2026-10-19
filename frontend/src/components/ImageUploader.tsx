"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export type LocalImage = {
  id: string;
  file: File;
  url: string;
  width: number;
  height: number;
  error?: string;
};

function readDims(file: File) {
  return new Promise<{ w: number; h: number }>((resolve, reject) => {
    try {
      const url = URL.createObjectURL(file);
      const img = new window.Image();
      img.onload = () => {
        resolve({ w: img.naturalWidth, h: img.naturalHeight });
        URL.revokeObjectURL(url);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("load"));
      };
      img.src = url;
    } catch (e) {
      reject(e);
    }
  });
}

export function ImageUploader({
  minImages = 1,
  onChange,
}: {
  minImages?: number;
  onChange: (files: File[]) => void;
}) {
  const [items, setItems] = useState<LocalImage[]>([]);
  const [progress, setProgress] = useState(0);

  const validCount = useMemo(() => items.filter((i) => !i.error).length, [items]);

  useEffect(() => {
    onChange(items.filter((m) => !m.error).map((m) => m.file));
  }, [items, onChange]);

  const applyFiles = useCallback(
    async (fileList: FileList | null) => {
      if (!fileList) return;
      setProgress(10);
      const next: LocalImage[] = [];
      let p = 10;
      for (const file of Array.from(fileList)) {
        if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) continue;
        p = Math.min(95, p + 10);
        setProgress(p);
        try {
          const dims = await readDims(file);
          next.push({
            id: typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `${file.name}-${file.size}-${Date.now()}-${Math.random()}`,
            file,
            url: URL.createObjectURL(file),
            width: dims.w,
            height: dims.h,
          });
        } catch {
          next.push({
            id: typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `${file.name}-${file.size}-${Date.now()}-${Math.random()}`,
            file,
            url: URL.createObjectURL(file),
            width: 0,
            height: 0,
          });
        }
      }
      setItems((prev) => [...prev, ...next]);
      setProgress(100);
      setTimeout(() => setProgress(0), 450);
    },
    []
  );

  return (
    <div className="space-y-4">
      <div
        className={cn(
          "group relative overflow-hidden rounded-[20px] border border-white/10 px-8 py-14 text-center transition-colors duration-300 hover:border-accent/40"
        )}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          void applyFiles(e.dataTransfer.files);
        }}
      >
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-70"
          style={{
            background:
              "radial-gradient(90% 80% at 50% 110%, rgba(255,192,138,0.28) 0%, rgba(240,138,168,0.16) 35%, rgba(185,163,240,0.08) 60%, transparent 80%), #13152A",
          }}
        />
        <div
          aria-hidden
          className="pointer-events-none absolute inset-[10px] rounded-[14px] border border-dashed border-white/15 transition-colors duration-200 group-hover:border-white/30"
        />
        <p className="relative font-display text-[2.2rem] font-extrabold uppercase leading-none text-white">Drop your watch photo</p>
        <p className="relative mx-auto mt-3 max-w-sm text-[14px] leading-6 text-primary/65">
          JPEG, PNG or WebP, at least 400 px on the short side. Dial and case in frame, no filters or text overlays.
          {minImages > 1 ? ` At least ${minImages} photos.` : ""}
        </p>
        <div className="relative mt-6">
          <input
            type="file"
            multiple
            accept="image/*"
            className="hidden"
            id="hv-up"
            onChange={(e) => void applyFiles(e.target.files)}
          />
          <Button
            type="button"
            variant="primary"
            className="min-h-[44px]"
            onClick={() => document.getElementById("hv-up")?.click()}
          >
            Choose files
          </Button>
        </div>
      </div>

      {progress > 0 ? (
        <div className="h-2 w-full overflow-hidden rounded-full bg-primary/10">
          <div className="h-full bg-accent transition-all" style={{ width: `${progress}%` }} />
        </div>
      ) : null}

      {items.length > 0 ? (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
          {items.map((it) => (
            <div key={it.id} className="relative overflow-hidden rounded-[18px] border border-white/10 bg-card">
              <div className="relative aspect-square">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={it.url}
                  alt=""
                  className="h-full w-full object-cover transition duration-300 hover:scale-[1.03]"
                />
              </div>
              <div className="border-t border-white/10 p-3 text-xs text-primary/60">
                <p className="truncate font-semibold text-primary/80">{it.file.name}</p>
                <p className="mt-1">
                  {it.width}×{it.height}
                </p>
                {it.error ? <p className="mt-2 font-semibold text-danger">{it.error}</p> : null}
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
