/** Must stay identical to backend/inference/scope_gate.py SUPPORTED_BRANDS. */
export const SUPPORTED_DECLARED_BRANDS = [
  "A. Lange & Söhne",
  "Audemars Piguet",
  "Patek Philippe",
  "Richard Mille",
  "Vacheron Constantin",
] as const;

export type SupportedDeclaredBrand = (typeof SUPPORTED_DECLARED_BRANDS)[number];
