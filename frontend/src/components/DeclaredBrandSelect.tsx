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
      className="mt-2 flex h-11 w-full rounded-md border border-[#1D1D1F]/20 bg-white px-3 text-sm text-[#1D1D1F]"
      style={{ color: "#1D1D1F", backgroundColor: "#FFFFFF" }}
      value={value}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value)}
    >
      <option value="" style={{ color: "#1D1D1F", backgroundColor: "#FFFFFF" }}>
        Select a declared brand
      </option>
      {SUPPORTED_DECLARED_BRANDS.map((brand) => (
        <option key={brand} value={brand} style={{ color: "#1D1D1F", backgroundColor: "#FFFFFF" }}>
          {brand}
        </option>
      ))}
    </select>
  );
}
