/** The five studio models (render/watches.py). Each is "in the style of" a reference, without logos. */
export const STUDIO_WATCHES = [
  { id: "patek_nautilus", brand: "Patek Philippe", piece: "Nautilus-style", city: "Geneva", since: "1839", backdrop: "marina" },
  { id: "ap_royal_oak", brand: "Audemars Piguet", piece: "Royal Oak-style", city: "Le Brassus", since: "1875", backdrop: "ocean_drive" },
  { id: "lange_one", brand: "A. Lange & Söhne", piece: "Lange 1-style", city: "Glashütte", since: "1845", backdrop: "penthouse" },
  { id: "vc_overseas", brand: "Vacheron Constantin", piece: "Overseas-style", city: "Geneva", since: "1755", backdrop: "rooftop_pool" },
  { id: "rm_rm011", brand: "Richard Mille", piece: "RM011", city: "Les Breuleux", since: "1999", backdrop: "neon_rain" },
] as const;

export type StudioWatch = (typeof STUDIO_WATCHES)[number];

/** Painted dusk skies, one per house, used until generated backdrops exist. */
export const HOUSE_SKIES = [
  "linear-gradient(180deg,#2B2350 0%,#6B4A97 45%,#C77BB0 78%,#F6B3B0 100%)",
  "linear-gradient(180deg,#3A2552 0%,#9A4F86 40%,#F08AA8 70%,#FFC08A 100%)",
  "linear-gradient(180deg,#24204A 0%,#5A3F8C 35%,#E0789A 75%,#FFD9A0 100%)",
  "linear-gradient(180deg,#1C2550 0%,#3B4E9A 40%,#9C86C8 75%,#F6B3B0 100%)",
  "linear-gradient(180deg,#0F1230 0%,#2A2160 50%,#7A3C78 85%,#E0789A 100%)",
];
