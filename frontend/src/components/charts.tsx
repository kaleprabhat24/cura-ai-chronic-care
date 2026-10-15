"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

export const COLORS = {
  brand: "#0f766e",
  brandSoft: "#5eb8ae",
  high: "#b42318",
  medium: "#d97706",
  low: "#94a3b8",
  ink: "#4a5b67",
  grid: "#e2e8ec",
  info: "#175cd3",
  ok: "#067647",
};

const tooltipStyle = {
  contentStyle: { borderRadius: 8, border: "1px solid #e2e8ec", fontSize: 12, boxShadow: "0 4px 12px rgb(15 29 38 / 0.08)" },
  labelStyle: { color: "#0f1d26", fontWeight: 600 },
};

type Datum = Record<string, string | number | null>;

/** Recharts tooltip formatter that tolerates undefined / non-numeric values. */
function tip(fmt?: (v: number) => string) {
  return (value: unknown): string => (typeof value === "number" ? (fmt ? fmt(value) : value.toLocaleString("en-GB")) : String(value ?? "—"));
}

export function SimpleBars({
  data,
  x,
  bars,
  height = 220,
  stacked,
  layout = "horizontal",
  colors,
  yFormatter,
}: {
  data: Datum[];
  x: string;
  bars: { key: string; label: string; color: string }[];
  height?: number;
  stacked?: boolean;
  layout?: "horizontal" | "vertical";
  colors?: string[];
  yFormatter?: (v: number) => string;
}) {
  const vertical = layout === "vertical";
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout={layout} margin={{ top: 8, right: 12, bottom: 0, left: vertical ? 8 : -12 }}>
        <CartesianGrid stroke={COLORS.grid} vertical={vertical} horizontal={!vertical} />
        {vertical ? (
          <>
            <XAxis type="number" tickLine={false} axisLine={false} tickFormatter={yFormatter} />
            <YAxis type="category" dataKey={x} tickLine={false} axisLine={false} width={150} />
          </>
        ) : (
          <>
            <XAxis dataKey={x} tickLine={false} axisLine={false} />
            <YAxis tickLine={false} axisLine={false} tickFormatter={yFormatter} />
          </>
        )}
        <Tooltip {...tooltipStyle} cursor={{ fill: "#f1f4f6" }} formatter={tip(yFormatter)} />
        {bars.length > 1 && <Legend iconType="circle" iconSize={8} />}
        {bars.map((b) => (
          <Bar key={b.key} dataKey={b.key} name={b.label} fill={b.color} stackId={stacked ? "s" : undefined}
               radius={stacked ? 0 : [4, 4, 0, 0]} maxBarSize={38}>
            {colors && bars.length === 1 && data.map((_, i) => <Cell key={i} fill={colors[i % colors.length]} />)}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function SimpleLines({
  data,
  x,
  lines,
  height = 220,
  yDomain,
  yFormatter,
  reference,
}: {
  data: Datum[];
  x: string;
  lines: { key: string; label: string; color: string }[];
  height?: number;
  yDomain?: [number | "auto", number | "auto"];
  yFormatter?: (v: number) => string;
  reference?: { y: number; label: string };
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        <XAxis dataKey={x} tickLine={false} axisLine={false} minTickGap={24} />
        <YAxis tickLine={false} axisLine={false} domain={yDomain ?? ["auto", "auto"]} tickFormatter={yFormatter} />
        <Tooltip {...tooltipStyle} formatter={tip(yFormatter)} />
        {lines.length > 1 && <Legend iconType="circle" iconSize={8} />}
        {reference && (
          <ReferenceLine y={reference.y} stroke={COLORS.low} strokeDasharray="4 4"
                         label={{ value: reference.label, position: "insideTopRight", fontSize: 11, fill: COLORS.ink }} />
        )}
        {lines.map((l) => (
          <Line key={l.key} type="monotone" dataKey={l.key} name={l.label} stroke={l.color} strokeWidth={2}
                dot={{ r: 3 }} connectNulls />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Reliability diagram: mean predicted vs observed rate per decile, with the y = x reference. */
export function CalibrationPlot({ points, height = 240 }: { points: { predicted: number; observed: number; n: number }[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: -8 }}>
        <CartesianGrid stroke={COLORS.grid} />
        <XAxis type="number" dataKey="predicted" name="Mean predicted" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
        <YAxis type="number" dataKey="observed" name="Observed" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
        <ZAxis type="number" dataKey="n" range={[40, 120]} name="Episodes" />
        <Tooltip {...tooltipStyle} formatter={(value: unknown, name: unknown) => (name === "Episodes" ? tip()(value) : tip((v) => `${(v * 100).toFixed(1)}%`)(value))} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={COLORS.low} strokeDasharray="4 4" />
        <Scatter data={points} fill={COLORS.brand} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
