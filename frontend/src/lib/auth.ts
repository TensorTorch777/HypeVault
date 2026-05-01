const LAST_EMAIL = "hypevault_last_email";

export function getAccessToken(): string | null {
  return null;
}

export function setTokens(_access: string, _refresh: string) {
  /* Cookie-based auth transport: tokens are stored in httpOnly cookies by backend. */
}

export function clearTokens() {
  /* Cookie-based auth transport: backend clears cookies via /auth/logout. */
}

export function getRefreshToken(): string | null {
  return null;
}

export function setLastSignedInEmail(email: string) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(LAST_EMAIL, email);
  } catch {
    /* no-op */
  }
}

export function getLastSignedInEmail(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(LAST_EMAIL);
  } catch {
    return null;
  }
}
