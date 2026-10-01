# Dashboard

`DispatchDashboard.jsx` is a standalone React component showing the 4-tier
architecture, routing chains, cost breakdown, and a route-preview tool.

It currently renders mock data. To wire it live, replace the MOCK constants
with fetches to:

- `GET  /dispatch/tiers`      — tier availability + budget
- `POST /dispatch/classify`   — route preview
- `GET  /memory/stats`        — memory engine stats

Drop it into any Vite/Next.js app, or use it as-is in a component sandbox.
