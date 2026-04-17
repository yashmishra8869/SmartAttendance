SmartAttendance
===============

SmartAttendance is a FastAPI app for face registration, attendance marking, and a live monthly dashboard.

Persistence on Render
---------------------

The app supports two storage modes:

1. CSV/file mode for local development.
2. Database mode for deployment persistence.

If `SMARTATTENDANCE_DB_URL`, `DATABASE_URL`, `POSTGRES_URL`, or `POSTGRESQL_URL` is set, the app stores attendance records and face encodings in PostgreSQL instead of local files. This is the mode you want on Render free tier, because local disk is not persistent there.

Recommended setup
-----------------

1. Create a free PostgreSQL database with a provider such as Neon or Supabase.
2. Copy the connection string.
3. In Render, open your service settings and add this environment variable:

	`SMARTATTENDANCE_DB_URL=postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME`

   You can also use `DATABASE_URL` if your provider supplies that name.

4. Redeploy the service.

What this stores
----------------

- Face encodings used for registration and recognition.
- Attendance rows used by the dashboard and CSV download.

Local development
-----------------

For local use, you can run the app without a Postgres URL. In that case it falls back to file-based storage under the project data directory.

Notes
-----

- Render free instances can spin down after inactivity.
- If `SMARTATTENDANCE_DB_URL` is not configured, registrations and attendance can disappear after redeploys or restarts.
- The app binds to `0.0.0.0` and is designed to run on Render's web port.

