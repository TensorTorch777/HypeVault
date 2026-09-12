/** Curated luxury watch brands shown in Shop the vault (matches demo seed data). */

export const LUXURY_WATCH_BRANDS = [
  "A. Lange & Söhne",
  "Audemars Piguet",
  "Patek Philippe",
  "Richard Mille",
  "Vacheron Constantin",
  "Rolex",
  "Omega",
] as const;

export type LuxuryWatchBrand = (typeof LUXURY_WATCH_BRANDS)[number];
