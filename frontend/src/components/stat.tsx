import { Card } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

export function Stat({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
  tone?: "default" | "brand" | "danger" | "warn";
}) {
  return (
    <Card className="px-5 py-4">
      <p className="text-[12px] font-medium uppercase tracking-wide text-ink-subtle">{label}</p>
      <p
        className={cn("tabular mt-1.5 text-[26px] font-semibold leading-none tracking-tight", {
          "text-ink": tone === "default",
          "text-brand-strong": tone === "brand",
          "text-danger": tone === "danger",
          "text-warn": tone === "warn",
        })}
      >
        {value}
      </p>
      {hint && <p className="mt-2 text-[12px] text-ink-subtle">{hint}</p>}
    </Card>
  );
}
