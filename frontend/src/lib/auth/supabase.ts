import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import { config } from "@/lib/config";

let client: SupabaseClient | null = null;

/** Browser Supabase client (public anon/publishable key only). The session is persisted in local
 * storage and restored on reload; access tokens auto-refresh. */
export function getSupabase(): SupabaseClient {
  if (!client) {
    client = createClient(config.supabaseUrl, config.supabaseAnonKey, {
      auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true, flowType: "pkce" },
    });
  }
  return client;
}
