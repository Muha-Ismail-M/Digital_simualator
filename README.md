Digital Simulator (Business Operations Digital Twin)
A lightweight “digital twin” simulator for modeling day-to-day business operations (customer arrivals, capacity, staffing, wait times, abandonment) and translating outcomes into KPIs, recommendations, and financial projections.

This repository currently ships as a Flask app that provides:

a simple UI template (templates/index.html)
JSON APIs to run simulations and retrieve benchmark data
What this project is
Digital Simulator helps you answer questions like:

If I add 1 server/cashier, how much does abandonment drop?
What are my peak hours and bottleneck stations?
What’s my estimated monthly profit given rent, labor, and COGS assumptions?
How do weather or special events change my throughput and revenue?
The simulator models demand using a Poisson arrival process and applies:

hourly patterns (lunch/dinner peaks, etc.)
day-of-week factors
weather multipliers
special-event multipliers
staff experience efficiency
It then produces:

operational KPIs (wait times, throughput, abandonment rate)
utilization metrics + bottleneck identification
recommendations prioritized by severity
monthly/annual financial projections + a simple ROI scenario
Tech stack
Python 3
Flask (API + template rendering)
Requests (optional Google Places lookups)
JSON benchmark dataset: data/industry_data.json
Project structure
text

.
├─ app.py                      # Flask app + simulation engine + financial model
├─ data/
│  └─ industry_data.json       # Benchmarks + hourly patterns + day-of-week factors
├─ templates/
│  └─ index.html               # UI template (currently minimal/static)
├─ .vscode/                    # Local VS Code tasks/debug configs (C++ oriented)
├─ simulator.exe               # Prebuilt Windows executable (legacy/experimental)
├─ digital_simulator.cpp       # Present, but currently appears to be empty/placeholder
└─ build/Debug/                # Build artifacts (legacy/experimental)
Note: The repository includes C++ build/debug configuration and a simulator.exe. The Flask simulator in app.py is the primary “source of truth” for the business modeling logic at the moment.

Quick start
1) Create a virtual environment (recommended)
Bash

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
2) Install dependencies
Bash

pip install flask requests
3) Run the server
Bash

python app.py
Open:

text

http://127.0.0.1:5000
Configuration (inputs)
The simulator accepts a JSON payload (POST /api/simulate) with many knobs. If you omit fields, sensible defaults are used.

Business setup
business_type: restaurant | retail | warehouse | healthcare | service
business_subtype: e.g. fast_food, casual_dining, fine_dining (varies by type)
business_name, location
Layout
square_footage
num_tables, seats_per_table (restaurant)
num_checkout_lanes (retail)
num_fitting_rooms (retail)
Staffing
num_hosts, num_servers, num_cooks, num_cashiers, num_managers, num_bussers
staff_experience: integer 1–5 (affects service efficiency)
Operating parameters
operating_hours (per day)
open_time (hour of day, e.g. 8 means 8:00)
simulation_days
Demand model
base_arrival_rate (customers/hour baseline before multipliers)
avg_party_size
customer_patience (minutes)
Financial assumptions
avg_ticket
hourly_wage
rent_per_sqft (annualized rent assumption; converted to monthly)
food_cost_percent
External factors
weather: clear | cloudy | rainy | snowy | hot | cold
special_event: none | holiday | local_event | sports_game | convention | competitor_promo
competitor_count (present in payload; currently not a major driver in logic)
API reference
GET /api/industry-data
Returns the full benchmark dataset used to drive targets, hourly patterns, and day factors.

Response

JSON object from data/industry_data.json
POST /api/search-business
Optionally enriches a simulation with real-world business metadata via Google Places (if enabled).

Request

JSON

{
  "name": "Example Restaurant",
  "location": "New York, NY"
}
Behavior

If GOOGLE_API_KEY is set, calls Google Places “Find Place From Text”
Otherwise returns simulated sample data
Response

JSON

{
  "success": true,
  "business": {
    "name": "Example Restaurant",
    "rating": 4.2,
    "reviews": 123,
    "price_level": 2,
    "type": "restaurant"
  }
}
POST /api/simulate
Runs the full simulation, generates recommendations, and computes financial projections.

Example request

Bash

curl -X POST http://127.0.0.1:5000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "business_type": "restaurant",
    "business_subtype": "casual_dining",
    "num_tables": 20,
    "seats_per_table": 4,
    "num_servers": 4,
    "num_cooks": 3,
    "operating_hours": 12,
    "simulation_days": 7,
    "base_arrival_rate": 30,
    "avg_ticket": 35,
    "hourly_wage": 15,
    "weather": "clear",
    "special_event": "none",
    "staff_experience": 3
  }'
Response (high level)

JSON

{
  "success": true,
  "config": { "...": "..." },
  "results": {
    "summary": { "...": "..." },
    "wait_times": { "...": "..." },
    "service": { "...": "..." },
    "financial": { "...": "..." },
    "utilization": { "...": "..." },
    "bottleneck": { "...": "..." },
    "peak_hours": ["..."],
    "scores": { "...": "..." },
    "hourly_data": ["..."]
  },
  "recommendations": [
    {
      "priority": "critical|high|medium|low",
      "category": "capacity|service|revenue|cost|risk|scheduling",
      "title": "...",
      "description": "...",
      "action": "...",
      "impact": "...",
      "cost": "...",
      "timeline": "..."
    }
  ],
  "financials": {
    "revenue": { "...": "..." },
    "costs": { "...": "..." },
    "profit": { "...": "..." },
    "breakeven": { "...": "..." },
    "scenarios": { "...": "..." },
    "lost_opportunity": { "...": "..." }
  }
}
How the simulation works (implementation notes)
This is a pragmatic discrete-event style simulation built for “what-if” planning:

Arrivals

Arrivals per hour are generated using a Poisson process.
Hourly arrivals are scaled by:
industry hourly patterns (e.g., lunch/dinner peaks)
day-of-week factors
weather factor
special event factor
Capacity Capacity is estimated from staffing/layout:

Restaurants: server capacity vs kitchen capacity (takes the minimum)
Retail: cashiers × ~15 customers/hour
Warehouse: pickers × ~8 orders/hour (if present)
Service + abandonment

Each customer has a patience window.
Wait time is approximated from queue length and service capacity.
Customers either get served or abandon (with a reason).
KPIs + scoring The simulator aggregates:

throughput/hour
abandonment rate
average/median/p95 wait time
revenue and lost revenue
utilization estimates
a composite “overall score” weighted across wait/throughput/efficiency
Recommendations Rules-based recommendations are generated and prioritized based on:

bottleneck utilization thresholds
wait time vs target
abandonment rate thresholds
utilization too low (overstaffing) or too high (burnout risk)
peak-hour scheduling hint
Financial model Generates projections:

daily/weekly/monthly/annual revenue
labor, rent, COGS, “other” costs
profit + margin
break-even customers/month
a simple ROI scenario (e.g., add 1 staff member to recover lost revenue)
Optional: Google Places API setup
To enable real business lookups for /api/search-business, set an environment variable:

Bash

# macOS/Linux
export GOOGLE_API_KEY="YOUR_KEY"

# Windows (PowerShell)
setx GOOGLE_API_KEY "YOUR_KEY"
Restart the Flask app after setting the variable.

