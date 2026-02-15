Digital Simulator (Business Operations Modeling)
A lightweight Flask app that simulates day-to-day business operations (customer arrivals, service capacity, wait times, abandonment) and turns the results into KPIs, actionable recommendations, and financial projections.

This is a “what-if” tool: tweak staffing, hours, demand, weather, and special events to quickly see how the system behaves under different scenarios.

Highlights
Multi-day simulation with hour-by-hour demand shaping
Demand multipliers:
Hourly traffic patterns
Day-of-week factors
Weather effects
Special-event effects
Staff experience efficiency
Outputs:
Served vs abandoned customers + abandonment rate
Throughput per hour
Wait-time distribution (avg/median/min/max/p95) + benchmark targets
Utilization + bottleneck identification (station-level “heatmap”)
Peak-hour detection
Rules-based recommendations (prioritized)
Monthly/annual financial projections + break-even and simple ROI scenario
Tech stack
Python 3
Flask
requests (optional: Google Places lookup)
Data model: data/industry_data.json
Repository layout
text

.
├── app.py
├── data/
│   └── industry_data.json
├── templates/
│   └── index.html
├── digital_simulator.cpp
├── simulator.exe
└── nano
Notes:

app.py contains the working Flask server, simulation engine, recommendation logic, and financial model.
industry_data.json is currently stored as a minified JSON file (single line) with benchmarks + patterns.
templates/index.html is a UI shell (mostly static text/structure).
digital_simulator.cpp and simulator.exe appear to be an experimental/alternate implementation.
nano is a process list dump (safe to remove if unneeded).
Quick start
1) Create a virtual environment (recommended)
text

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
2) Install dependencies
text

pip install flask requests
3) Run the server
text

python app.py
Open:

text

http://127.0.0.1:5000
Supported business types (as implemented)
The simulator is driven by data/industry_data.json. Current supported categories include:

restaurant: fast_food, casual_dining, fine_dining
retail: convenience_store, clothing_store, grocery_store, electronics
warehouse: small, medium, large
healthcare: urgent_care, dental_office, pharmacy
service: bank_branch, hair_salon, auto_service
The data file also includes:

hourly_patterns (currently defined for restaurant/retail/warehouse)
day_of_week_factors (currently defined for restaurant/retail)
labor_costs reference ranges
How it works (model overview)
This is a pragmatic simulation built for directional planning:

Demand generation
For each simulated hour, arrivals are generated using a Poisson process with an hour-specific rate:

Start with base_arrival_rate
Multiply by:
hourly pattern factor (from industry_data.json)
day-of-week factor (from industry_data.json)
weather factor (from a built-in map)
special event factor (from a built-in map)
Capacity model (simplified, but transparent)
Service capacity is estimated from staffing and business type:

Restaurant (as implemented):
server capacity = num_servers * 4 * seats_per_table
kitchen capacity = num_cooks * 10
effective capacity = min(server_capacity, kitchen_capacity)
Retail:
num_cashiers * 15 customers/hour
Warehouse:
num_pickers * 8 orders/hour (defaults if not provided)
Other:
fallback formula based on num_servers
Wait & abandonment logic
Each arriving customer gets:

a randomized party size
a randomized patience window
a randomized service time (adjusted by staff experience efficiency)
Customers are served if:

estimated wait time is within patience, and
capacity/load checks allow them to enter service
Otherwise they are marked abandoned with a reason like:

wait_too_long
capacity
Outputs & scoring
The app aggregates:

customer totals (served/abandoned)
wait-time stats (avg/median/min/max/p95)
revenue + lost revenue (based on ticket value for served vs abandoned)
utilization estimate
peak hour ranking
station utilization (simulated per station for bottleneck detection)
composite scores:
wait-time score
throughput score
efficiency score
overall score (weighted blend)
Recommendations
Recommendations are generated via rules such as:

bottleneck utilization thresholds
wait time vs benchmark target
abandonment rate thresholds
utilization too low (overstaffing/cost) or too high (burnout risk)
peak-hour scheduling hint
Financial projections
Financials are computed from simulation results and configuration:

revenue scaled to daily/weekly/monthly/annual
costs:
labor
rent (from rent_per_sqft)
COGS percentage (food_cost_percent)
other costs (estimated as 10% of revenue)
profit + margin
break-even customers/month
a simple ROI scenario (“add one staff member”)
Configuration (inputs)
Send a JSON payload to POST /api/simulate. Any missing fields fall back to defaults.

Business profile
business_type (default: restaurant)
business_subtype (default: casual_dining)
business_name
location
Physical layout
square_footage (default: 2000)
Restaurant:
num_tables (default: 20)
seats_per_table (default: 4)
Retail:
num_checkout_lanes (default: 3)
num_fitting_rooms (default: 4)
Staffing
num_hosts (default: 1)
num_servers (default: 4)
num_cooks (default: 3)
num_cashiers (default: 2)
num_managers (default: 1)
num_bussers (default: 2)
staff_experience (1–5, default: 3)
Operating parameters
operating_hours (default: 12)
open_time (default: 8)
simulation_days (default: 7)
Demand parameters
base_arrival_rate (default: 30 customers/hour)
peak_multiplier (default: 2.0)
avg_party_size (default: 2.5)
customer_patience (default: 15 minutes)
Service time parameters
service_time_mean (default: 45)
service_time_variance (default: 10)
Financial assumptions
avg_ticket (default: 35.00)
hourly_wage (default: 15.00)
rent_per_sqft (default: 25)
food_cost_percent (default: 30)
External factors
weather (default: clear)
supported values: clear, cloudy, rainy, snowy, hot, cold
special_event (default: none)
supported values: none, holiday, local_event, sports_game, convention, competitor_promo
competitor_count (default: 3)
API reference
GET /
Serves the UI shell (templates/index.html).

GET /api/industry-data
Returns the benchmark dataset as JSON (loaded from data/industry_data.json).

POST /api/search-business
Optional lookup using Google Places (if GOOGLE_API_KEY is set). If not set, returns simulated business info.

Request body:

text

{
  "name": "Example Restaurant",
  "location": "New York, NY"
}
POST /api/simulate
Runs the simulation and returns:

results: KPIs, wait-time stats, utilization, bottleneck, peak hours, hourly aggregates, scores
recommendations: prioritized actions
financials: projected revenue/cost/profit + break-even + ROI scenario
config: resolved inputs (payload merged with defaults)
Example curl:

text

curl -X POST "http://127.0.0.1:5000/api/simulate" \
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
Google Places setup (optional)
To enable real lookups for POST /api/search-business, set GOOGLE_API_KEY:

macOS/Linux:

text

export GOOGLE_API_KEY="YOUR_KEY"
Windows (PowerShell):

text

setx GOOGLE_API_KEY "YOUR_KEY"
Restart the app after setting the variable.

Practical cleanup / next improvements
Add a requirements.txt (or pyproject.toml) for reproducible installs.
Consider moving/removing simulator.exe, build artifacts, and the nano dump if they’re not part of the intended deliverable.
Expand the UI (currently a shell) to call the APIs and render charts/tables.
Add basic tests around:
arrival generation
capacity calculation
KPI aggregation
financial model math
License
No license file is included yet.
