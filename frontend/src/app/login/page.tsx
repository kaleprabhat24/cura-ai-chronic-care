"use client";

import { Activity } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { SyntheticBanner } from "@/components/app-shell";
import { InlineNotice, Spinner } from "@/components/states";
import { Button, Card, CardContent, Input, Label, Select } from "@/components/ui/primitives";
import { useAuth } from "@/lib/auth/auth-context";
import { ROLES, type Role } from "@/lib/auth/session";
import { configProblem } from "@/lib/config";

function LoginForm() {
  const { status, mode, error, signInDev, signInPassword, signInMagicLink } = useAuth();
  const router = useRouter();
  const params = useSearchParams();
  const next = params.get("next") || "/";
  const safeNext = next.startsWith("/") && !next.startsWith("//") ? next : "/";
  const [userId, setUserId] = useState("coordinator.demo");
  const [role, setRole] = useState<Role>("coordinator");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ tone: "ok" | "danger"; text: string } | null>(null);
  const problem = configProblem();

  useEffect(() => {
    if (status === "signed_in") router.replace(safeNext);
  }, [status, router, safeNext]);

  async function run(action: () => Promise<void>, success?: string) {
    setBusy(true);
    setMessage(null);
    try {
      await action();
      if (success) setMessage({ tone: "ok", text: success });
    } catch (e) {
      setMessage({ tone: "danger", text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className="w-full max-w-sm">
      <CardContent className="space-y-5 pt-6">
        <div className="flex items-center gap-2.5">
          <div className="flex size-9 items-center justify-center rounded-lg bg-brand text-white">
            <Activity className="size-4" />
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-ink">Sign in to Cura AI</h1>
            <p className="text-[12px] text-ink-subtle">Chronic-care follow-up prioritisation</p>
          </div>
        </div>
        {problem && <InlineNotice tone="danger">{problem}</InlineNotice>}
        {error && !problem && <InlineNotice tone="danger">{error}</InlineNotice>}
        {status === "loading" ? (
          <Spinner label="Restoring session…" />
        ) : mode === "dev" ? (
          <form
            className="space-y-4"
            onSubmit={(e) => {
              e.preventDefault();
              void run(() => signInDev({ userId: userId.trim(), role }));
            }}
          >
            <InlineNotice tone="warn">
              Local development sign-in. The API accepts these headers only with AUTH_MODE=dev and refuses to start that
              mode in production.
            </InlineNotice>
            <div className="space-y-1.5">
              <Label htmlFor="uid">User id</Label>
              <Input id="uid" value={userId} onChange={(e) => setUserId(e.target.value)} pattern="[A-Za-z0-9._@:\-]{1,80}" required />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="role">Role</Label>
              <Select id="role" value={role} onChange={(e) => setRole(e.target.value as Role)}>
                {ROLES.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </Select>
            </div>
            <Button type="submit" className="w-full" disabled={busy || !!problem}>
              {busy ? <Spinner label="Signing in…" /> : "Continue"}
            </Button>
          </form>
        ) : (
          <form
            className="space-y-4"
            onSubmit={(e) => {
              e.preventDefault();
              void run(() => signInPassword(email.trim(), password));
            }}
          >
            <div className="space-y-1.5">
              <Label htmlFor="email">Email</Label>
              <Input id="email" type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="password">Password</Label>
              <Input id="password" type="password" autoComplete="current-password" value={password}
                     onChange={(e) => setPassword(e.target.value)} />
            </div>
            <Button type="submit" className="w-full" disabled={busy || !!problem || !password}>
              {busy ? <Spinner label="Signing in…" /> : "Sign in"}
            </Button>
            <Button
              type="button"
              variant="outline"
              className="w-full"
              disabled={busy || !!problem || !email}
              onClick={() => void run(() => signInMagicLink(email.trim()), "Check your email for a sign-in link.")}
            >
              Email me a sign-in link
            </Button>
            <p className="text-[12px] text-ink-subtle">
              Access requires an administrator to provision your email with a role. Signing in alone grants nothing.
            </p>
          </form>
        )}
        {message && <InlineNotice tone={message.tone}>{message.text}</InlineNotice>}
      </CardContent>
    </Card>
  );
}

export default function LoginPage() {
  return (
    <div className="min-h-screen">
      <SyntheticBanner />
      <div className="flex min-h-[80vh] items-center justify-center px-4">
        <Suspense fallback={<Spinner />}>
          <LoginForm />
        </Suspense>
      </div>
    </div>
  );
}
