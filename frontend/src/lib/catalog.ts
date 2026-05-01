/** Curated luxury brands shown in Shop the vault (matches demo seed data). */

export const LUXURY_SNEAKER_BRANDS = [
  "Alexander McQueen",
  "Balenciaga",
  "Dior",
  "Gucci",
  "Louis Vuitton",
] as const;

export const LUXURY_WATCH_BRANDS = [
  "A. Lange & Söhne",
  "Audemars Piguet",
  "Patek Philippe",
  "Richard Mille",
  "Vacheron Constantin",
] as const;

export type LuxurySneakerBrand = (typeof LUXURY_SNEAKER_BRANDS)[number];
export type LuxuryWatchBrand = (typeof LUXURY_WATCH_BRANDS)[number];
