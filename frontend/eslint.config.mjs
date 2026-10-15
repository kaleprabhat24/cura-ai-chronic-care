import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __dirname = dirname(fileURLToPath(import.meta.url));
const compat = new FlatCompat({ baseDirectory: __dirname });

const config = [
  { ignores: [".next/**", "node_modules/**", "next-env.d.ts", "src/lib/api/schema.d.ts"] },
  ...compat.extends("next/core-web-vitals", "next/typescript"),
];

export default config;
