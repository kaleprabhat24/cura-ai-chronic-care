import { describe, expect, it } from "vitest";
import { ApiError, createApi, setAuthHeaderProvider, unwrap, validationMessages } from "./client";

function jsonResponse(status: number, body: unknown) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

describe("API client", () => {
  it("attaches credentials from the registered provider", async () => {
    let seen: Headers | null = null;
    setAuthHeaderProvider(async () => ({ Authorization: "Bearer abc" }));
    const api = createApi({
      baseUrl: "http://api.test",
      fetch: async (req: Request | URL | string) => {
        seen = (req as Request).headers;
        return jsonResponse(200, { user_id: "u", role: "admin", auth_mode: "supabase" });
      },
    });
    const me = await unwrap(api.GET("/api/v1/me"));
    expect(me.role).toBe("admin");
    expect(seen!.get("Authorization")).toBe("Bearer abc");
    setAuthHeaderProvider(async () => ({}));
  });

  it("maps backend error bodies to ApiError", async () => {
    const api = createApi({
      baseUrl: "http://api.test",
      fetch: async () => jsonResponse(403, { error: { code: "forbidden", message: "Role 'analyst' may not", details: null } }),
    });
    const err = await unwrap(api.GET("/api/v1/worklist")).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(403);
    expect(err.code).toBe("forbidden");
    expect(err.message).toContain("analyst");
  });

  it("reports expired sessions as 401", async () => {
    const api = createApi({
      baseUrl: "http://api.test",
      fetch: async () => jsonResponse(401, { error: { code: "invalid_token", message: "The access token is invalid or expired." } }),
    });
    const err = (await unwrap(api.GET("/api/v1/me")).catch((e) => e)) as ApiError;
    expect(err.isAuth).toBe(true);
    expect(err.code).toBe("invalid_token");
  });

  it("turns network failures into a network_error", async () => {
    const api = createApi({
      baseUrl: "http://api.test",
      fetch: async () => {
        throw new TypeError("Failed to fetch");
      },
    });
    const err = (await unwrap(api.GET("/api/v1/health")).catch((e) => e)) as ApiError;
    expect(err.code).toBe("network_error");
    expect(err.status).toBe(0);
  });

  it("extracts field messages from validation errors", () => {
    const err = new ApiError(422, "validation_error", "The request is invalid.", [
      { loc: ["body", "outcome"], msg: "Input should be 'no_answer', 'left_message'", type: "literal_error" },
      { loc: ["body"], msg: "Value error, next_action_due is required when a next action is set", type: "value_error" },
    ]);
    expect(validationMessages(err)).toEqual([
      "outcome: Input should be 'no_answer', 'left_message'",
      "next_action_due is required when a next action is set",
    ]);
    expect(validationMessages(new Error("x"))).toEqual([]);
  });
});
