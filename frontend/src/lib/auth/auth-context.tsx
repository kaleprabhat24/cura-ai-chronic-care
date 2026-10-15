"use client";

import { useQueryClient } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api, ApiError, setAuthHeaderProvider, unwrap, type Schemas } from "@/lib/api/client";
import { config, configProblem } from "@/lib/config";
import {
  clearDevIdentity,
  devHeaders,
  loadDevIdentity,
  saveDevIdentity,
  supabaseHeaders,
  type DevIdentity,
} from "./session";
import { getSupabase } from "./supabase";

type Status = "loading" | "signed_out" | "signed_in";
type Me = Schemas["Me"];

type AuthState = {
  status: Status;
  me: Me | null;
  error: string | null;
  mode: typeof config.authMode;
  signInDev: (identity: DevIdentity) => Promise<void>;
  signInPassword: (email: string, password: string) => Promise<void>;
  signInMagicLink: (email: string) => Promise<void>;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<AuthState | null>(null);

function storage(): Storage | undefined {
  try {
    return typeof window === "undefined" ? undefined : window.localStorage;
  } catch {
    return undefined;
  }
}

// Registered at module load (before any component fetches) so no request leaves without credentials.
setAuthHeaderProvider(async () =>
  config.authMode === "supabase" ? supabaseHeaders(getSupabase()) : devHeaders(loadDevIdentity(storage())),
);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const queryClient = useQueryClient();
  const [status, setStatus] = useState<Status>("loading");
  const [me, setMe] = useState<Me | null>(null);
  const [error, setError] = useState<string | null>(configProblem());

  /** Ask the backend who we are; the role always comes from the server. */
  const resolveIdentity = useCallback(async () => {
    try {
      const who = await unwrap(api.GET("/api/v1/me"));
      setMe(who);
      setError(null);
      setStatus("signed_in");
    } catch (e) {
      setMe(null);
      setStatus("signed_out");
      if (e instanceof ApiError && e.status !== 401) setError(e.message);
      else if (!(e instanceof ApiError)) setError("Could not verify the session.");
    }
  }, []);

  // Session restoration on load, and Supabase session changes (sign-in, refresh, sign-out).
  useEffect(() => {
    if (configProblem()) {
      setStatus("signed_out");
      return;
    }
    if (config.authMode === "dev") {
      if (loadDevIdentity(storage())) void resolveIdentity();
      else setStatus("signed_out");
      return;
    }
    const supabase = getSupabase();
    supabase.auth.getSession().then(({ data }) => {
      if (data.session) void resolveIdentity();
      else setStatus("signed_out");
    });
    const { data: sub } = supabase.auth.onAuthStateChange((event, session) => {
      if (event === "SIGNED_OUT" || !session) {
        setMe(null);
        setStatus("signed_out");
        queryClient.clear();
      } else if (event === "SIGNED_IN" || event === "TOKEN_REFRESHED" || event === "USER_UPDATED") {
        void resolveIdentity();
      }
    });
    return () => sub.subscription.unsubscribe();
  }, [queryClient, resolveIdentity]);

  const signInDev = useCallback(
    async (identity: DevIdentity) => {
      const store = storage();
      if (!store) throw new Error("Browser storage is unavailable.");
      saveDevIdentity(store, identity);
      setStatus("loading");
      await resolveIdentity();
    },
    [resolveIdentity],
  );

  const signInPassword = useCallback(async (email: string, password: string) => {
    const { error: err } = await getSupabase().auth.signInWithPassword({ email, password });
    if (err) throw new Error(err.message);
  }, []);

  const signInMagicLink = useCallback(async (email: string) => {
    const { error: err } = await getSupabase().auth.signInWithOtp({
      email,
      options: { emailRedirectTo: `${window.location.origin}/login`, shouldCreateUser: false },
    });
    if (err) throw new Error(err.message);
  }, []);

  const signOut = useCallback(async () => {
    if (config.authMode === "supabase") await getSupabase().auth.signOut();
    clearDevIdentity(storage());
    queryClient.clear();
    setMe(null);
    setStatus("signed_out");
  }, [queryClient]);

  const value = useMemo<AuthState>(
    () => ({ status, me, error, mode: config.authMode, signInDev, signInPassword, signInMagicLink, signOut }),
    [status, me, error, signInDev, signInPassword, signInMagicLink, signOut],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}

export function hasRole(me: Me | null, ...roles: string[]): boolean {
  return !!me && roles.includes(me.role);
}
