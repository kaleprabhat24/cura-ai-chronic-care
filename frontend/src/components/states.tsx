"use client";

import { AlertTriangle, Inbox, Loader2, LockKeyhole, WifiOff } from "lucide-react";
import Link from "next/link";
import { ApiError } from "@/lib/api/client";
import { Button, Skeleton } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

export function LoadingBlock({ rows = 4, className }: { rows?: number; className?: string }) {
  return (
    <div className={cn("space-y-2.5", className)} aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-9 w-full" />
      ))}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-ink-subtle">
      <Loader2 className="size-4 animate-spin" /> {label}
    </span>
  );
}

export function EmptyState({ title, hint, className }: { title: string; hint?: string; className?: string }) {
  return (
    <div className={cn("flex flex-col items-center justify-center gap-2 py-12 text-center", className)}>
      <Inbox className="size-6 text-ink-subtle" />
      <p className="text-sm font-medium text-ink">{title}</p>
      {hint && <p className="max-w-md text-[13px] text-ink-subtle">{hint}</p>}
    </div>
  );
}

export function ErrorState({ error, onRetry, className }: { error: unknown; onRetry?: () => void; className?: string }) {
  const e = error instanceof ApiError ? error : null;
  const Icon = e?.code === "network_error" ? WifiOff : e && (e.status === 401 || e.status === 403) ? LockKeyhole : AlertTriangle;
  const title =
    e?.code === "network_error"
      ? "The Cura AI API is unreachable"
      : e?.status === 401
        ? "Your session has expired"
        : e?.status === 403
          ? "Your role does not have access to this view"
          : e?.status === 404
            ? "Not found"
            : e?.code === "dataset_not_loaded"
              ? "No cohort has been loaded yet"
              : "Something went wrong";
  return (
    <div role="alert" className={cn("flex flex-col items-center justify-center gap-2 py-10 text-center", className)}>
      <Icon className="size-6 text-danger" />
      <p className="text-sm font-medium text-ink">{title}</p>
      <p className="max-w-lg text-[13px] text-ink-subtle">{e?.message ?? String((error as Error)?.message ?? error)}</p>
      <div className="mt-2 flex gap-2">
        {onRetry && e?.status !== 403 && e?.status !== 401 && (
          <Button variant="outline" size="sm" onClick={onRetry}>
            Try again
          </Button>
        )}
        {e?.status === 401 && (
          <Button asChild size="sm">
            <Link href="/login">Sign in again</Link>
          </Button>
        )}
      </div>
    </div>
  );
}

export function InlineNotice({
  tone = "info",
  children,
  className,
}: {
  tone?: "info" | "warn" | "danger" | "ok";
  children: React.ReactNode;
  className?: string;
}) {
  const tones = {
    info: "border-info/20 bg-info-soft text-info",
    warn: "border-warn/25 bg-warn-soft text-warn",
    danger: "border-danger/20 bg-danger-soft text-danger",
    ok: "border-ok/20 bg-ok-soft text-ok",
  };
  return <div className={cn("rounded-lg border px-3 py-2 text-[13px]", tones[tone], className)}>{children}</div>;
}
