import { describe, expect, it } from "vitest";
import {
  clearDevIdentity,
  DEV_STORAGE_KEY,
  devHeaders,
  loadDevIdentity,
  saveDevIdentity,
  supabaseHeaders,
  validateDevIdentity,
} from "./session";
import { configProblem } from "@/lib/config";

function memoryStore() {
  const m = new Map<string, string>();
  return {
    getItem: (k: string) => m.get(k) ?? null,
    setItem: (k: string, v: string) => void m.set(k, v),
    removeItem: (k: string) => void m.delete(k),
    raw: m,
  };
}

describe("development identity", () => {
  it("validates user id and role", () => {
    expect(validateDevIdentity({ userId: "coord.1", role: "coordinator" })).toEqual({ userId: "coord.1", role: "coordinator" });
    expect(validateDevIdentity({ userId: "x", role: "superuser" })).toBeNull();
    expect(validateDevIdentity({ userId: "bad id with spaces", role: "admin" })).toBeNull();
    expect(validateDevIdentity(null)).toBeNull();
  });

  it("persists, restores and clears the session", () => {
    const store = memoryStore();
    expect(loadDevIdentity(store)).toBeNull();
    saveDevIdentity(store, { userId: "analyst.1", role: "analyst" });
    expect(loadDevIdentity(store)).toEqual({ userId: "analyst.1", role: "analyst" });
    clearDevIdentity(store);
    expect(loadDevIdentity(store)).toBeNull();
  });

  it("ignores tampered or corrupt stored sessions", () => {
    const store = memoryStore();
    store.setItem(DEV_STORAGE_KEY, JSON.stringify({ userId: "x", role: "root" }));
    expect(loadDevIdentity(store)).toBeNull();
    store.setItem(DEV_STORAGE_KEY, "{not json");
    expect(loadDevIdentity(store)).toBeNull();
    expect(() => saveDevIdentity(store, { userId: "x y", role: "admin" })).toThrow();
  });

  it("produces dev headers only for a valid identity", () => {
    expect(devHeaders({ userId: "a", role: "admin" })).toEqual({ "X-Dev-User": "a", "X-Dev-Role": "admin" });
    expect(devHeaders(null)).toEqual({});
  });
});

describe("Supabase session", () => {
  it("sends the current access token as a bearer token", async () => {
    const client = { auth: { getSession: async () => ({ data: { session: { access_token: "tok-123" } } }) } };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(await supabaseHeaders(client as any)).toEqual({ Authorization: "Bearer tok-123" });
  });

  it("sends nothing when signed out", async () => {
    const client = { auth: { getSession: async () => ({ data: { session: null } }) } };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(await supabaseHeaders(client as any)).toEqual({});
  });
});

describe("configuration safety", () => {
  const base = { apiBaseUrl: "http://x", supabaseUrl: "", supabaseAnonKey: "" };
  it("refuses dev sign-in in production", () => {
    expect(configProblem({ ...base, authMode: "dev", appEnv: "production" })).toMatch(/disabled in production/);
  });
  it("requires Supabase settings in Supabase mode", () => {
    expect(configProblem({ ...base, authMode: "supabase", appEnv: "production" })).toMatch(/NEXT_PUBLIC_SUPABASE_URL/);
    expect(configProblem({ ...base, authMode: "supabase", appEnv: "production", supabaseUrl: "https://p.supabase.co", supabaseAnonKey: "anon" })).toBeNull();
  });
  it("allows dev sign-in locally", () => {
    expect(configProblem({ ...base, authMode: "dev", appEnv: "development" })).toBeNull();
  });
});
