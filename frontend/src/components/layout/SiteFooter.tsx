import Link from "next/link";

import { Lockup } from "@/components/brand/Lockup";

const COLUMNS = [
  {
    title: "Market",
    links: [
      { label: "Search listings", href: "/market" },
      { label: "Patek Philippe", href: "/market?brand=Patek%20Philippe" },
      { label: "Audemars Piguet", href: "/market?brand=Audemars%20Piguet" },
      { label: "Richard Mille", href: "/market?brand=Richard%20Mille" },
      { label: "A. Lange & Söhne", href: "/market?brand=A.%20Lange%20%26%20S%C3%B6hne" },
      { label: "Vacheron Constantin", href: "/market?brand=Vacheron%20Constantin" },
    ],
  },
  {
    title: "Sellers",
    links: [
      { label: "List a watch", href: "/seller/upload" },
      { label: "Seller desk", href: "/seller/dashboard" },
      { label: "Create an account", href: "/register" },
    ],
  },
  {
    title: "Research",
    links: [
      { label: "Model benchmark", href: "/benchmark" },
      { label: "How screening works", href: "/#method" },
      { label: "Research console", href: "/research" },
    ],
  },
];

export function SiteFooter() {
  return (
    <footer className="relative overflow-hidden border-t border-primary/10 bg-[#0A0B17] pb-28 pt-20 md:pb-14">
      <div className="mx-auto grid max-w-[77.5rem] gap-14 px-4 sm:px-6 md:grid-cols-[1.4fr_repeat(3,1fr)]">
        <div className="max-w-sm">
          <Lockup compact />
          <p className="mt-5 text-[0.8125rem] leading-6 text-primary/65">
            Market intelligence and AI-assisted listing screening for five houses of haute horlogerie. A university
            research project, not a live marketplace.
          </p>
        </div>
        {COLUMNS.map((col) => (
          <div key={col.title}>
            <p className="hv-eyebrow">{col.title}</p>
            <ul className="mt-5 space-y-3">
              {col.links.map((l) => (
                <li key={l.label}>
                  <Link href={l.href} className="hv-link-underline text-[0.8125rem] text-primary/70 transition-colors hover:text-primary">
                    {l.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <div className="mx-auto mt-16 flex max-w-[77.5rem] items-center gap-4 px-4 sm:px-6">
        <div className="grid h-16 w-12 shrink-0 place-items-center border-2 border-primary/80 font-display text-[1.9rem] font-black leading-none text-primary" aria-hidden="true">
          RP
        </div>
        <p className="max-w-sm text-[0.6875rem] leading-4 text-primary/65">
          <span className="font-bold text-primary/85">Research Prototype.</span> Screening evidence, not a certificate of authenticity.
          Studio watches are procedural models without logos; backdrops marked as generated are AI art.
        </p>
      </div>
      <div className="mx-auto mt-10 flex max-w-[77.5rem] flex-col gap-3 border-t border-primary/10 px-4 pt-8 text-[0.6875rem] leading-5 text-primary/50 sm:px-6 md:flex-row md:justify-between">
        <p>© {new Date().getFullYear()} HypeVault · K. J. Somaiya School of Engineering · Research prototype</p>
        <p>
          3D model: &ldquo;Richard Mille RM011&rdquo; by Denis V on Sketchfab,{" "}
          <a href="https://creativecommons.org/licenses/by-nc/4.0/" className="underline hover:text-primary" rel="noreferrer" target="_blank">CC BY-NC 4.0</a>
        </p>
        <p>Model classifications are screening evidence, never a certificate of authenticity.</p>
      </div>
    </footer>
  );
}
