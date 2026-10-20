"use client";

import { SUPPORTED_DECLARED_BRANDS } from "@/lib/supportedBrands";

export function DeclaredBrandSelect({
  id,
  value,
  onChange,
  disabled = false,
}: {
  id: string;
  value: string;
  onChange: (brand: string) => void;
  disabled?: boolean;
}) {
  return (
    <select
      id={id}
      className="mt-2 flex h-11 w-full rounded-xl border border-white/12 bg-[#0E0F1E] px-3 text-[0.9375rem] text-primary focus:border-accent focus:outline-none"
      value={value}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value)}
    >
      <option value="">
        Select a declared brand
      </option>
      {SUPPORTED_DECLARED_BRANDS.map((brand) => (
        <option key={brand} value={brand}>
          {brand}
        </option>
      ))}
    </select>
  );
}
