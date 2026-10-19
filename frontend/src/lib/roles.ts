/** Role helpers. Roles come from the server; admins can do everything sellers can. */
export function canSell(role: string | null | undefined): boolean {
  return role === "seller" || role === "admin";
}

export function isAdmin(role: string | null | undefined): boolean {
  return role === "admin";
}
