"use client";

import { useEffect, useMemo, useRef } from "react";

type Role = "buyer" | "seller";

type GoogleCredentialResponse = {
  credential?: string;
};

type GoogleAccountsId = {
  initialize: (options: {
    client_id: string;
    callback: (response: GoogleCredentialResponse) => void;
    hd?: string;
    ux_mode?: "popup" | "redirect";
  }) => void;
  renderButton: (
    parent: HTMLElement,
    options: {
      type: "standard";
      theme: "outline" | "filled_black";
      text: "continue_with" | "signin_with";
      size: "large";
      width: number;
      shape: "pill";
      logo_alignment?: "left" | "center";
    }
  ) => void;
};

type GoogleWindow = Window & {
  google?: {
    accounts?: {
      id?: GoogleAccountsId;
    };
  };
};

function loadGoogleScript(): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector('script[data-google-identity="true"]') as HTMLScriptElement | null;
    if (existing) {
      if ((window as GoogleWindow).google?.accounts?.id) resolve();
      else existing.addEventListener("load", () => resolve(), { once: true });
      return;
    }
    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.defer = true;
    script.dataset.googleIdentity = "true";
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Failed to load Google Identity script"));
    document.head.appendChild(script);
  });
}

export function GoogleSignInButton({
  role,
  disabled,
  hostedDomain,
  onCredential,
  onError,
}: {
  role: Role;
  disabled?: boolean;
  /** Restricts Google's account chooser to this Workspace domain. The server enforces it too. */
  hostedDomain?: string;
  onCredential: (idToken: string, role: Role) => void;
  onError: (message: string) => void;
}) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const clientId = useMemo(() => process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID?.trim() ?? "", []);

  useEffect(() => {
    if (!mountRef.current || disabled) return;
    if (!clientId) {
      onError("Google sign-in is not configured. Set NEXT_PUBLIC_GOOGLE_CLIENT_ID.");
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        await loadGoogleScript();
        if (cancelled || !mountRef.current) return;
        const googleId = (window as GoogleWindow).google?.accounts?.id;
        if (!googleId) {
          onError("Google sign-in failed to initialize.");
          return;
        }
        mountRef.current.innerHTML = "";
        googleId.initialize({
          client_id: clientId,
          ...(hostedDomain ? { hd: hostedDomain } : {}),
          callback: (response) => {
            const token = response.credential;
            if (!token) {
              onError("Google sign-in failed. No credential received.");
              return;
            }
            onCredential(token, role);
          },
        });
        googleId.renderButton(mountRef.current, {
          type: "standard",
          theme: "filled_black",
          text: "signin_with",
          size: "large",
          // Google accepts 200–400 px; fit the container so it never overflows a phone screen.
          width: Math.max(200, Math.min(340, Math.floor(mountRef.current.clientWidth || 340))),
          shape: "pill",
          logo_alignment: "left",
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Google sign-in is unavailable.";
        onError(msg);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [clientId, disabled, hostedDomain, onCredential, onError, role]);

  return <div ref={mountRef} className="flex min-h-[2.75rem] w-full justify-center" />;
}

