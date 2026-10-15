# Deployment (Vercel + Render + Supabase)

These steps were used for the live demo: web https://cura-ai-chronic-care.vercel.app/, API
https://cura-ai-api.onrender.com. On 2026-10-16 the live API health check reported production mode,
Supabase auth and a reachable database, and `/api/v1/me` returned 401 without a token. Signed-in
flows were reported working by the project owner, but were not re-checked in that pass. Do the steps
in order.

## 1. Supabase
Follow [SUPABASE_SETUP.md](SUPABASE_SETUP.md):
1. Create the project.
2. Run `cura-api db upgrade`, `seed` and `score` against its database from your machine.
3. Configure Auth.
4. Provision users with `cura-api users add`.

## 2. Backend on Render
1. Push the repository to GitHub.
2. In Render, **New → Blueprint** and select the repository; Render reads [render.yaml](../render.yaml).
3. Fill in the secret variables it asks for: `SUPABASE_URL`, `DATABASE_URL` (session pooler URI),
   `CORS_ORIGINS` (JSON list containing your Vercel URL).
4. The build installs `ml` + `backend[postgres]` and regenerates the deterministic synthetic dataset
   (`cura-ml all`, used by `/data-quality`). The committed model bundle `ml/artifacts/models/cura-fu-1.0.0`
   is used as-is. The API never trains.
5. `preDeployCommand: cura-api db upgrade` needs a paid instance. On the free plan run it locally
   against the same `DATABASE_URL` before deploying.
6. Check `https://<service>.onrender.com/api/v1/health`.

The API refuses to start if `ENVIRONMENT=production` is combined with `AUTH_MODE=dev`, with SQLite,
or with `AUTH_MODE=supabase` but no `https://` `SUPABASE_URL`.

## 3. Frontend on Vercel
1. **Add New → Project**, import the repository, set **Root Directory** to `frontend`. The framework
   is detected as Next.js; build command `npm run build`.
2. Environment variables (Production):
   `NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_AUTH_MODE=supabase`, `NEXT_PUBLIC_SUPABASE_URL`,
   `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_APP_ENV=production`.
3. Add the Vercel URL to Supabase **Redirect URLs** (`https://<domain>/login`) and to the backend's
   `CORS_ORIGINS`, then redeploy the backend.

## 4. After deploying
Work through the live checklist in [SUPABASE_SETUP.md §6](SUPABASE_SETUP.md#6-verify-live-checklist).
If you regenerate the dataset or retrain the model, run `cura-api seed` and `cura-api score` again.
Scoring refuses to run if the database, dataset files and model bundle fingerprints disagree.
