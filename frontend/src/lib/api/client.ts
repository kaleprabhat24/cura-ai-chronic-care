/** Central typed API client. Types come from the backend's OpenAPI document (`npm run gen:api`). */

import createClient, { type Client } from "openapi-fetch";
import { config } from "@/lib/config";
import type { components, paths } from "./schema";

export type Schemas = components["schemas"];

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }

  get isAuth(): boolean {
    return this.status === 401;
  }
}

type HeaderProvider = () => Promise<Record<string, string>>;
let headerProvider: HeaderProvider = async () => ({});

/** The auth layer registers how to obtain credentials (dev headers or a Supabase bearer token). */
export function setAuthHeaderProvider(provider: HeaderProvider): void {
  headerProvider = provider;
}

export function createApi(options: { baseUrl?: string; fetch?: typeof fetch } = {}): Client<paths> {
  const client = createClient<paths>({ baseUrl: options.baseUrl ?? config.apiBaseUrl, fetch: options.fetch });
  client.use({
    async onRequest({ request }) {
      const headers = await headerProvider();
      for (const [key, value] of Object.entries(headers)) request.headers.set(key, value);
      return request;
    },
  });
  return client;
}

export const api = createApi();

type Result<T> = { data?: T; error?: unknown; response: Response };

/** Resolve an openapi-fetch call to its data, or throw an ApiError with the backend's error body. */
export async function unwrap<T>(call: Promise<Result<T>>): Promise<T> {
  let result: Result<T>;
  try {
    result = await call;
  } catch {
    throw new ApiError(0, "network_error", "Cannot reach the Cura AI API. Check that the backend is running.");
  }
  const { data, error, response } = result;
  if (!response.ok || error !== undefined) {
    const body = (error ?? {}) as { error?: { code?: string; message?: string; details?: unknown } };
    throw new ApiError(
      response.status,
      body.error?.code ?? "http_error",
      body.error?.message ?? (response.statusText || `Request failed (${response.status})`),
      body.error?.details,
    );
  }
  return data as T;
}

/** Field-level messages from a 422 validation error, for forms. */
export function validationMessages(error: unknown): string[] {
  if (!(error instanceof ApiError) || error.status !== 422) return [];
  const details = Array.isArray(error.details) ? (error.details as { loc?: unknown[]; msg?: string }[]) : [];
  return details.map((d) => {
    const field = (d.loc ?? []).filter((x) => x !== "body").join(".");
    const msg = (d.msg ?? "invalid").replace(/^Value error, /, "");
    return field ? `${field}: ${msg}` : msg;
  });
}
