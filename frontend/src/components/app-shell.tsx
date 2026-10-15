"use client";

import {
  Activity,
  BarChart3,
  ClipboardList,
  FlaskConical,
  LayoutDashboard,
  LogOut,
  PhoneCall,
  ShieldCheck,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";
import { hasRole, useAuth } from "@/lib/auth/auth-context";
import { Badge, Button } from "@/components/ui/primitives";
import { Spinner } from "@/components/states";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard, roles: ["coordinator", "analyst", "admin"] },
  { href: "/worklist", label: "Worklist", icon: ClipboardList, roles: ["coordinator", "admin"] },
  { href: "/outreach", label: "Outreach", icon: PhoneCall, roles: ["coordinator", "admin"] },
  { href: "/outcomes", label: "Outcomes", icon: BarChart3, roles: ["coordinator", "analyst", "admin"] },
  { href: "/model", label: "Model & data", icon: FlaskConical, roles: ["coordinator", "analyst", "admin"] },
  { href: "/admin", label: "Admin & export", icon: ShieldCheck, roles: ["coordinator", "admin"] },
];

export function SyntheticBanner() {
  return (
    <div className="flex items-center justify-center gap-2 bg-ink px-4 py-1.5 text-center text-[12px] font-medium text-white/90">
      <Activity className="size-3.5 text-brand-ring" />
      Synthetic demonstration data — fictional patients and simulated predictions. Not for clinical use.
    </div>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const { status, me, signOut } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (status === "signed_out") router.replace(`/login?next=${encodeURIComponent(pathname)}`);
  }, [status, router, pathname]);

  if (status !== "signed_in" || !me) {
    return (
      <div className="min-h-screen">
        <SyntheticBanner />
        <div className="flex h-[70vh] items-center justify-center">
          <Spinner label={status === "loading" ? "Restoring session…" : "Redirecting to sign-in…"} />
        </div>
      </div>
    );
  }

  const items = NAV.filter((n) => hasRole(me, ...n.roles));
  return (
    <div className="min-h-screen">
      <SyntheticBanner />
      <div className="flex">
        <aside className="sticky top-0 hidden h-[calc(100vh-30px)] w-60 shrink-0 flex-col border-r border-border bg-surface md:flex">
          <div className="flex items-center gap-2.5 px-5 py-5">
            <div className="flex size-8 items-center justify-center rounded-lg bg-brand text-white">
              <Activity className="size-4" />
            </div>
            <div>
              <p className="text-[15px] font-semibold leading-tight tracking-tight text-ink">Cura AI</p>
              <p className="text-[11px] text-ink-subtle">Follow-up prioritisation</p>
            </div>
          </div>
          <nav className="flex-1 space-y-0.5 px-3" aria-label="Main">
            {items.map((n) => {
              const active = n.href === "/" ? pathname === "/" : pathname.startsWith(n.href);
              return (
                <Link
                  key={n.href}
                  href={n.href}
                  className={cn(
                    "flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13px] font-medium transition-colors",
                    active ? "bg-brand-soft text-brand-strong" : "text-ink-muted hover:bg-surface-muted hover:text-ink",
                  )}
                >
                  <n.icon className="size-4" />
                  {n.label}
                </Link>
              );
            })}
          </nav>
          <div className="border-t border-border p-4">
            <p className="truncate text-[13px] font-medium text-ink" title={me.user_id}>
              {me.display_name || me.user_id}
            </p>
            <div className="mt-1 flex items-center justify-between">
              <Badge tone="brand" className="capitalize">
                {me.role}
              </Badge>
              <span className="text-[11px] text-ink-subtle">{me.auth_mode === "dev" ? "dev sign-in" : "Supabase"}</span>
            </div>
            <Button variant="ghost" size="sm" className="mt-3 w-full justify-start" onClick={() => void signOut()}>
              <LogOut /> Sign out
            </Button>
          </div>
        </aside>
        <div className="min-w-0 flex-1">
          <header className="flex items-center justify-between border-b border-border bg-surface px-4 py-3 md:hidden">
            <span className="font-semibold text-ink">Cura AI</span>
            <Button variant="ghost" size="sm" onClick={() => void signOut()}>
              <LogOut /> Sign out
            </Button>
          </header>
          <nav className="flex gap-1 overflow-x-auto border-b border-border bg-surface px-3 py-2 md:hidden" aria-label="Main mobile">
            {items.map((n) => (
              <Link key={n.href} href={n.href} className="whitespace-nowrap rounded-md px-2.5 py-1 text-xs text-ink-muted hover:bg-surface-muted">
                {n.label}
              </Link>
            ))}
          </nav>
          <main className="mx-auto max-w-[1400px] px-4 py-6 md:px-8 md:py-8">{children}</main>
        </div>
      </div>
    </div>
  );
}

export function PageHeader({ title, description, actions }: { title: string; description?: string; actions?: React.ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-[22px] font-semibold tracking-tight text-ink">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-[13px] text-ink-muted">{description}</p>}
      </div>
      {actions}
    </div>
  );
}

export function RoleGate({ roles, children }: { roles: string[]; children: React.ReactNode }) {
  const { me } = useAuth();
  if (!hasRole(me, ...roles)) {
    return (
      <div className="rounded-lg border border-border bg-surface p-8 text-center text-sm text-ink-muted">
        Your role ({me?.role}) does not have access to this view. Access is also enforced by the API.
      </div>
    );
  }
  return <>{children}</>;
}
