# Supabase setup (PostgreSQL + Auth)

Cura AI runs fully locally without Supabase (SQLite + development sign-in). This guide connects it
to a Supabase project for a hosted demo. **None of these steps have been performed or verified by the
project author**: no Supabase project or credentials existed while this was built. The code paths are
covered by automated tests that use locally generated signing keys and offline PostgreSQL migration
rendering.

## 1. Create the project
1. Create a project at supabase.com. Choose a region near your Render region and set a strong
   database password (store it in a password manager).
2. From **Project Settings → API**, note:
   - **Project URL** (`https://<project-ref>.supabase.co`) → `SUPABASE_URL` (backend) and
     `NEXT_PUBLIC_SUPABASE_URL` (frontend).
   - the **anon / publishable key** → `NEXT_PUBLIC_SUPABASE_ANON_KEY` (frontend only; it is public by
     design).
   - **Never** copy the `service_role` / secret key anywhere in this project. Cura AI does not need it.

## 2. Database connection
From **Project Settings → Database → Connection string → URI**:

| Use | Connection | Settings |
|---|---|---|
| Render / local tools (recommended) | **Session pooler**, port 5432 (IPv4) | `DATABASE_URL=<uri>`, `DATABASE_SSLMODE=require` |
| Direct connection | `db.<ref>.supabase.co:5432` (IPv6 only on some plans) | same |
| Transaction pooler | port 6543 | also `DATABASE_DISABLE_PREPARED_STATEMENTS=true` |

`postgres://` and `postgresql://` URIs are accepted; the backend uses the psycopg 3 driver
(`pip install -e "backend[postgres]"`). Connections use a pre-pinged pool (`DATABASE_POOL_SIZE`,
`DATABASE_MAX_OVERFLOW`, `DATABASE_POOL_RECYCLE_SECONDS`).

## 3. Create the schema and load the data (from your machine)
In PowerShell from the project root, with the dataset and model bundle present locally:
```
$env:DATABASE_URL = "<session pooler URI>"
$env:DATABASE_SSLMODE = "require"
.venv\Scripts\cura-api db upgrade
.venv\Scripts\cura-api seed
.venv\Scripts\cura-api score
```
Migration `0003` enables **row-level security with no policies** on every table. This blocks Supabase's
public Data API (anon/authenticated roles) from reading the data. The backend connects as the table
owner and is unaffected. Check under **Database → Tables**: every table should show "RLS enabled".

## 4. Authentication
1. **Authentication → Providers → Email**: enable email sign-in. Under **Authentication → Sign In /
   Providers**, turn **off** "Allow new users to sign up". Access is by invitation only.
2. **Authentication → URL Configuration**:
   - Site URL: your Vercel URL, e.g. `https://cura-ai.vercel.app`
   - Redirect URLs: `http://localhost:3000/login` and `https://<your-vercel-domain>/login`
3. **Authentication → Users → Add user / Invite**: create each person (email + password, or invite).
4. **Grant a Cura AI role**. Supabase accounts get no access until provisioned:
   ```
   .venv\Scripts\cura-api users add you@example.org --role admin --name "Your Name"
   .venv\Scripts\cura-api users add coordinator@example.org --role coordinator
   .venv\Scripts\cura-api users list
   ```
   On first sign-in the verified Supabase identity (`sub`) is bound to that row. Use `set-role`,
   `deactivate` and `activate` to manage access. Every change is audited.
5. **Signing keys**: the API verifies tokens against `SUPABASE_URL/auth/v1/.well-known/jwks.json`
   (asymmetric RS256/ES256 keys, the default for new projects). If your project still uses the
   **legacy JWT secret** (HS256), also set `SUPABASE_JWT_SECRET` on the backend (a secret: only in
   Render's environment, never in the frontend). The API checks signature, issuer
   (`<SUPABASE_URL>/auth/v1`), audience (`authenticated`), expiry, `sub` and `role = authenticated`.

## 5. Configure the apps
Backend (`.env` locally, Render environment in production):
```
ENVIRONMENT=production            # or development when testing locally against Supabase
AUTH_MODE=supabase
SUPABASE_URL=https://<project-ref>.supabase.co
DATABASE_URL=<session pooler URI>
DATABASE_SSLMODE=require
CORS_ORIGINS=["https://<your-vercel-domain>","http://localhost:3000"]
```
Frontend (`frontend/.env.local` locally, Vercel environment in production):
```
NEXT_PUBLIC_API_BASE_URL=https://<your-render-service>.onrender.com
NEXT_PUBLIC_AUTH_MODE=supabase
NEXT_PUBLIC_SUPABASE_URL=https://<project-ref>.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=<anon/publishable key>
NEXT_PUBLIC_APP_ENV=production
```

## 6. Verify live (checklist)
1. `GET https://<api>/api/v1/health` returns `auth_mode: "supabase"` and `database: "ok"`.
2. Without signing in, `GET /api/v1/me` returns 401.
3. Sign in on the frontend with a provisioned user: the sidebar shows your role. An unprovisioned
   Supabase user sees "This account has no Cura AI access".
4. As a coordinator, open a patient, record an outreach outcome, refresh: it persists. As an
   admin, the audit log shows `outreach.create` with your email.
5. In the Supabase SQL editor, `select count(*) from public.outreach_attempts where origin = 'app';`
   matches what you recorded.
6. With only the anon key, a Data API query such as
   `GET https://<ref>.supabase.co/rest/v1/patients?select=id` returns no rows (RLS).
