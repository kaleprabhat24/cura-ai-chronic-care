/** Public runtime configuration. Only NEXT_PUBLIC_* values exist in the browser bundle; no secret may
 * ever be configured here (the Supabase anon/publishable key is designed to be public). */

export type AuthMode = "dev" | "supabase";

export const config = {
  apiBaseUrl: (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/+$/, ""),
  authMode: (process.env.NEXT_PUBLIC_AUTH_MODE === "supabase" ? "supabase" : "dev") as AuthMode,
  appEnv: process.env.NEXT_PUBLIC_APP_ENV || "development",
  supabaseUrl: process.env.NEXT_PUBLIC_SUPABASE_URL || "",
  supabaseAnonKey: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "",
};

/** A human-readable configuration problem, or null when the configuration is usable. */
export function configProblem(c = config): string | null {
  if (c.authMode === "dev" && c.appEnv === "production") {
    return "Development sign-in is disabled in production. Set NEXT_PUBLIC_AUTH_MODE=supabase.";
  }
  if (c.authMode === "supabase" && (!c.supabaseUrl || !c.supabaseAnonKey)) {
    return "Supabase sign-in needs NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY.";
  }
  return null;
}
