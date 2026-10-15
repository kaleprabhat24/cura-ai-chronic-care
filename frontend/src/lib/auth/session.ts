/** Framework-independent session helpers (unit-tested in session.test.ts). */

import type { SupabaseClient, Session } from "@supabase/supabase-js";

export const ROLES = ["coordinator", "analyst", "admin"] as const;
export type Role = (typeof ROLES)[number];
export type DevIdentity = { userId: string; role: Role };

export const DEV_STORAGE_KEY = "cura.devIdentity";
const USER_ID = /^[A-Za-z0-9._@:-]{1,80}$/;

type KeyValueStore = Pick<Storage, "getItem" | "setItem" | "removeItem">;

export function validateDevIdentity(value: unknown): DevIdentity | null {
  if (!value || typeof value !== "object") return null;
  const { userId, role } = value as Record<string, unknown>;
  if (typeof userId !== "string" || !USER_ID.test(userId)) return null;
  if (typeof role !== "string" || !(ROLES as readonly string[]).includes(role)) return null;
  return { userId, role: role as Role };
}

export function loadDevIdentity(store: KeyValueStore | undefined): DevIdentity | null {
  if (!store) return null;
  try {
    const raw = store.getItem(DEV_STORAGE_KEY);
    return raw ? validateDevIdentity(JSON.parse(raw)) : null;
  } catch {
    return null;
  }
}

export function saveDevIdentity(store: KeyValueStore, identity: DevIdentity): void {
  const valid = validateDevIdentity(identity);
  if (!valid) throw new Error("Invalid development identity");
  store.setItem(DEV_STORAGE_KEY, JSON.stringify(valid));
}

export function clearDevIdentity(store: KeyValueStore | undefined): void {
  try {
    store?.removeItem(DEV_STORAGE_KEY);
  } catch {
    /* storage unavailable */
  }
}

export function devHeaders(identity: DevIdentity | null): Record<string, string> {
  return identity ? { "X-Dev-User": identity.userId, "X-Dev-Role": identity.role } : {};
}

/** Bearer header from the current Supabase session. supabase-js refreshes expired access tokens
 * inside getSession(), so the token sent to the API is always the latest one. */
export async function supabaseHeaders(client: Pick<SupabaseClient, "auth">): Promise<Record<string, string>> {
  const { data } = await client.auth.getSession();
  const token = (data.session as Session | null)?.access_token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}
