# Digital Simulator

Digital Simulator is a lightweight web app for **business operations modeling**. It lets you configure a real-world style service business (restaurant, retail, warehouse, healthcare, or general service), then runs a **discrete-event simulation** to estimate throughput, wait times, utilization, revenue, and operational bottlenecks—followed by prioritized recommendations and simple financial projections.

This project is intended for learning, experimentation, and “what-if” analysis (e.g., *What happens if I add one cashier? Extend hours? Increase demand?*).

---

## What it does

### Operations simulation
You configure your business and operating conditions (staffing, service speed, customer demand, hours, etc.). The simulator generates customers over time and models the flow through capacity constraints to estimate:

- Customers served vs. customers abandoned (lost due to waiting/capacity)
- Wait-time statistics (average, median, max, p95)
- Throughput per hour
- Estimated utilization and bottlenecks (where the system is overloaded)

### Recommendations (decision support)
Based on simulation results, the app generates actionable recommendations such as:

- Where the **critical bottleneck** is (and whether it’s near/over capacity)
- Whether wait times and abandonment are above target
- Whether staffing looks over/under-utilized
- Peak-hour scheduling tips

> Note: The UI labels this as “AI Recommendations,” but the current logic is a rules/heuristics-based recommendation engine driven by the simulation outputs.

### Financial projections
Using simulated demand and a basic cost model, the app estimates:

- Monthly/annual revenue projections
- Labor, rent, COGS, and “other” cost estimates
- Monthly profit and profit margin
- Break-even customer volume
- A simple scenario comparing current staffing vs. adding one additional staff member

### Optional business lookup (Google Places)
If a Google Places API key is available, the app can look up basic public business info (e.g., rating/reviews) to prefill or contextualize a simulation. If not, it returns realistic-looking sample data.

---

## How the simulation works (high level)

The core engine runs a time-based simulation across hours and days:

1. **Customer arrivals**  
   Customers are generated per hour using a Poisson-style arrival process, adjusted by:
   - hour-of-day patterns (lunch/dinner peaks, etc.)
   - day-of-week factors
   - weather and special events multipliers

2. **Service capacity**  
   Capacity is estimated from staffing and the selected business type (e.g., restaurant capacity depends on servers and kitchen throughput). Staff experience increases efficiency.

3. **Queue + abandonment**  
   Each customer has a “patience” value. If estimated wait exceeds patience, the customer abandons (lost revenue). Otherwise, they are served with a service-time distribution.

4. **KPIs + scoring**  
   The simulator aggregates results into a summary dashboard and computes performance scores (wait-time, throughput, efficiency) using industry benchmark targets.

---

## User experience (UI overview)

The main screen is designed as a “configure → run → analyze” workflow:

- Left panel: business profile, staffing, operations, financial assumptions, and external factors (day/weather/events)
- Run Simulation button: executes a full run with the chosen parameters
- Results area: performance scores, customer totals, wait times, utilization, and revenue
- Recommendations: prioritized action list based on bottlenecks and KPI gaps
- Financial projections: quick monthly/annual rollups + profit metrics

---

## Project structure (what’s in this repository)

- `app.py`  
  Flask server + simulation engine + recommendation and financial modeling logic.

- `templates/`  
  Frontend template served by Flask (single-page dashboard-style UI).

- `data/industry_data.json`  
  Built-in benchmark dataset (targets, hourly patterns, and reference values by business type/subtype).

- `build/`, `simulator.exe`, and other artifacts  
  Build outputs / experimental artifacts included in the repo (not required for the Flask web app).

---

## Notes, assumptions, and limitations

- This is a **model**, not a real POS/ERP system. Outputs are only as good as the assumptions you enter.
- Scraped/real-time data is not the goal here; the focus is controllable simulation with realistic patterns.
- Some configuration fields are foundation work for future expansion (not every field influences the simulation yet).
- Station-level utilization is represented as an estimated breakdown to help identify bottlenecks; it is not a physical sensor-backed measurement.

---

## Roadmap ideas (easy wins that add a lot of value)

- Add multiple simulation runs per config (Monte Carlo) and show confidence intervals
- Make competitor count/location factors influence demand (currently a placeholder)
- Add export-to-PDF/CSV for results and recommendations
- Expand business-type-specific models (e.g., fitting rooms for clothing retail, triage queues for urgent care)
- Add scenario comparison: “current vs. improved staffing vs. layout change” side-by-side

---

## License

No license file is currently included.
