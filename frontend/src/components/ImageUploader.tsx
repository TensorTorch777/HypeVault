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
          "group relative overflow-hidden rounded-[18px] border border-[rgba(255,31,164,0.28)] bg-[#120523]/85 p-8 text-center transition-all duration-300 hover:border-[rgba(0,225,255,0.55)] hover:shadow-[0_0_28px_rgba(255,31,164,0.28)]"
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
              "radial-gradient(120% 80% at 50% -20%, rgba(255,122,26,0.25) 0%, rgba(255,31,164,0.12) 35%, rgba(11,1,24,0.0) 68%)",
          }}
        />
        <div
          aria-hidden
          className="pointer-events-none absolute inset-[10px] rounded-[14px] border border-dashed border-[rgba(255,237,246,0.18)] transition-colors duration-300 group-hover:border-[rgba(0,225,255,0.45)]"
        />
        <p className="relative text-sm font-semibold uppercase tracking-[0.14em] text-[#FFEDF6]">Drag & drop images</p>
        <p className="relative mt-2 text-xs font-semibold uppercase tracking-[0.16em] text-[#FFEDF6]/55">
          JPEG/PNG/WebP only • any dimensions • need {minImages}+
        </p>
        <div className="mt-4">
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
            className="min-h-[44px] border border-[rgba(255,31,164,0.3)] shadow-[0_8px_22px_rgba(255,31,164,0.25)]"
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
            <div key={it.id} className="relative overflow-hidden rounded-card border border-primary/10 bg-card">
              <div className="relative aspect-square">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={it.url}
                  alt=""
                  className="h-full w-full object-cover transition duration-300 hover:scale-[1.03]"
                />
              </div>
              <div className="border-t border-primary/10 p-3 text-xs text-primary/60">
                <p className="font-semibold text-primary/75">{it.file.name}</p>
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
