from flask import Flask, render_template, request, jsonify
import subprocess
import json
import os
import re
import random
import math
from datetime import datetime, timedelta
import requests

app = Flask(__name__)

# Load industry benchmark data
def load_industry_data():
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'industry_data.json')
        with open(data_path, 'r') as f:
            return json.load(f)
    except:
        return {}

INDUSTRY_DATA = load_industry_data()

# Google Places API (optional - set your API key)
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/industry-data')
def get_industry_data():
    """Return industry benchmark data for UI"""
    return jsonify(INDUSTRY_DATA)

@app.route('/api/search-business', methods=['POST'])
def search_business():
    """Search for real business data using Google Places API"""
    try:
        data = request.json
        business_name = data.get('name', '')
        location = data.get('location', '')
        
        if GOOGLE_API_KEY and business_name:
            # Use Google Places API
            url = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
            params = {
                'input': f"{business_name} {location}",
                'inputtype': 'textquery',
                'fields': 'name,rating,user_ratings_total,price_level,types,opening_hours',
                'key': GOOGLE_API_KEY
            }
            response = requests.get(url, params=params)
            places_data = response.json()
            
            if places_data.get('candidates'):
                place = places_data['candidates'][0]
                return jsonify({
                    'success': True,
                    'business': {
                        'name': place.get('name'),
                        'rating': place.get('rating', 4.0),
                        'reviews': place.get('user_ratings_total', 100),
                        'price_level': place.get('price_level', 2),
                        'type': place.get('types', ['restaurant'])[0]
                    }
                })
        
        # Return simulated data if no API key
        return jsonify({
            'success': True,
            'business': generate_sample_business(business_name),
            'note': 'Simulated data - Add Google API key for real data'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Run full simulation with all parameters"""
    try:
        config = request.json
        
        # Validate and extract configuration
        simulation_config = {
            # Business Setup
            'business_type': config.get('business_type', 'restaurant'),
            'business_subtype': config.get('business_subtype', 'casual_dining'),
            'business_name': config.get('business_name', 'My Business'),
            'location': config.get('location', 'New York, NY'),
            
            # Physical Layout
            'square_footage': config.get('square_footage', 2000),
            'num_tables': config.get('num_tables', 20),
            'seats_per_table': config.get('seats_per_table', 4),
            'num_checkout_lanes': config.get('num_checkout_lanes', 3),
            'num_fitting_rooms': config.get('num_fitting_rooms', 4),
            
            # Staff Configuration
            'num_hosts': config.get('num_hosts', 1),
            'num_servers': config.get('num_servers', 4),
            'num_cooks': config.get('num_cooks', 3),
            'num_cashiers': config.get('num_cashiers', 2),
            'num_managers': config.get('num_managers', 1),
            'num_bussers': config.get('num_bussers', 2),
            
            # Staff Skill Levels (1-5)
            'staff_experience': config.get('staff_experience', 3),
            
            # Operating Parameters
            'operating_hours': config.get('operating_hours', 12),
            'open_time': config.get('open_time', 8),
            'simulation_days': config.get('simulation_days', 7),
            
            # Customer Parameters
            'base_arrival_rate': config.get('base_arrival_rate', 30),
            'peak_multiplier': config.get('peak_multiplier', 2.0),
            'avg_party_size': config.get('avg_party_size', 2.5),
            'customer_patience': config.get('customer_patience', 15),
            
            # Financial Parameters
            'avg_ticket': config.get('avg_ticket', 35.00),
            'hourly_wage': config.get('hourly_wage', 15.00),
            'rent_per_sqft': config.get('rent_per_sqft', 25),
            'food_cost_percent': config.get('food_cost_percent', 30),
            
            # Service Parameters
            'service_time_mean': config.get('service_time_mean', 45),
            'service_time_variance': config.get('service_time_variance', 10),
            
            # Day of Week
            'day_of_week': config.get('day_of_week', 'friday'),
            
            # Weather Impact
            'weather': config.get('weather', 'clear'),
            
            # Special Events
            'special_event': config.get('special_event', 'none'),
            
            # Competition
            'competitor_count': config.get('competitor_count', 3)
        }
        
        # Run the simulation
        results = run_simulation(simulation_config)
        
        # Generate recommendations
        recommendations = generate_recommendations(results, simulation_config)
        
        # Generate financial projections
        financials = calculate_financials(results, simulation_config)
        
        return jsonify({
            'success': True,
            'results': results,
            'recommendations': recommendations,
            'financials': financials,
            'config': simulation_config
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

def run_simulation(config):
    """Run discrete event simulation with realistic modeling"""
    
    # Get industry benchmarks
    business_type = config['business_type']
    subtype = config['business_subtype']
    industry = INDUSTRY_DATA.get(business_type, {}).get(subtype, {})
    
    # Simulation parameters
    operating_hours = config['operating_hours']
    simulation_days = config['simulation_days']
    base_arrival_rate = config['base_arrival_rate']
    
    # Get hourly pattern
    hourly_pattern = INDUSTRY_DATA.get('hourly_patterns', {}).get(business_type, {})
    day_factors = INDUSTRY_DATA.get('day_of_week_factors', {}).get(business_type, {})
    
    # Calculate capacity
    total_seats = config['num_tables'] * config['seats_per_table']
    service_capacity = calculate_service_capacity(config)
    
    # Simulation state
    all_customers = []
    served_customers = []
    abandoned_customers = []
    hourly_stats = []
    station_stats = {}
    
    # Staff efficiency based on experience
    efficiency = 0.7 + (config['staff_experience'] * 0.06)  # 0.76 to 1.0
    
    # Weather impact
    weather_factors = {
        'clear': 1.0, 'cloudy': 0.95, 'rainy': 0.75,
        'snowy': 0.5, 'hot': 0.85, 'cold': 0.8
    }
    weather_factor = weather_factors.get(config['weather'], 1.0)
    
    # Special event impact
    event_factors = {
        'none': 1.0, 'holiday': 1.5, 'local_event': 1.3,
        'sports_game': 1.4, 'convention': 1.6, 'competitor_promo': 0.7
    }
    event_factor = event_factors.get(config['special_event'], 1.0)
    
    # Run simulation for each day
    for day in range(simulation_days):
        day_name = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'][day % 7]
        day_factor = day_factors.get(day_name, 1.0)
        
        daily_customers = []
        daily_served = []
        daily_abandoned = []
        
        # Simulate each hour
        for hour in range(config['open_time'], config['open_time'] + operating_hours):
            hour_key = str(hour)
            hour_factor = float(hourly_pattern.get(hour_key, 0.5))
            
            # Calculate arrival rate for this hour
            adjusted_rate = (base_arrival_rate * hour_factor * day_factor * 
                           weather_factor * event_factor)
            
            # Generate customers for this hour
            num_arrivals = generate_poisson_arrivals(adjusted_rate)
            
            for i in range(num_arrivals):
                # Random arrival within the hour
                arrival_time = hour + random.random()
                party_size = max(1, int(random.gauss(config['avg_party_size'], 1)))
                ticket_value = config['avg_ticket'] * party_size * random.uniform(0.8, 1.3)
                patience = config['customer_patience'] * random.uniform(0.5, 1.5)
                
                customer = {
                    'id': len(all_customers) + 1,
                    'arrival_time': arrival_time,
                    'party_size': party_size,
                    'ticket_value': ticket_value,
                    'patience': patience,
                    'day': day,
                    'hour': hour
                }
                
                # Check if can be served
                current_load = len([c for c in daily_customers if c.get('served', False) and 
                                   c.get('departure_time', 0) > arrival_time])
                
                if current_load < service_capacity:
                    # Calculate service time
                    base_service = config['service_time_mean']
                    service_time = max(5, random.gauss(base_service, config['service_time_variance']))
                    service_time = service_time / efficiency  # Adjust for staff efficiency
                    
                    # Calculate wait time based on queue
                    queue_length = len([c for c in daily_customers if not c.get('served', False)])
                    wait_time = queue_length * (base_service / service_capacity) / efficiency
                    
                    if wait_time <= patience:
                        customer['wait_time'] = wait_time
                        customer['service_time'] = service_time
                        customer['total_time'] = wait_time + service_time
                        customer['departure_time'] = arrival_time + wait_time + service_time
                        customer['served'] = True
                        daily_served.append(customer)
                    else:
                        customer['served'] = False
                        customer['abandon_reason'] = 'wait_too_long'
                        daily_abandoned.append(customer)
                else:
                    # Check if customer will wait
                    estimated_wait = (current_load - service_capacity + 1) * (base_service / service_capacity)
                    if estimated_wait <= patience:
                        customer['wait_time'] = estimated_wait
                        service_time = max(5, random.gauss(base_service, config['service_time_variance'])) / efficiency
                        customer['service_time'] = service_time
                        customer['total_time'] = estimated_wait + service_time
                        customer['departure_time'] = arrival_time + estimated_wait + service_time
                        customer['served'] = True
                        daily_served.append(customer)
                    else:
                        customer['served'] = False
                        customer['abandon_reason'] = 'capacity'
                        daily_abandoned.append(customer)
                
                daily_customers.append(customer)
            
            # Record hourly stats
            hourly_stats.append({
                'day': day,
                'hour': hour,
                'arrivals': num_arrivals,
                'served': len([c for c in daily_served if c['hour'] == hour]),
                'abandoned': len([c for c in daily_abandoned if c['hour'] == hour]),
                'avg_wait': calculate_avg([c['wait_time'] for c in daily_served if c['hour'] == hour])
            })
        
        all_customers.extend(daily_customers)
        served_customers.extend(daily_served)
        abandoned_customers.extend(daily_abandoned)
    
    # Calculate final statistics
    results = calculate_statistics(all_customers, served_customers, abandoned_customers, 
                                   hourly_stats, config, industry)
    
    return results

def calculate_service_capacity(config):
    """Calculate service capacity based on staff and layout"""
    business_type = config['business_type']
    
    if business_type == 'restaurant':
        # Each server can handle ~4 tables, each cook can produce ~10 meals/hour
        server_capacity = config['num_servers'] * 4 * config['seats_per_table']
        kitchen_capacity = config['num_cooks'] * 10
        return min(server_capacity, kitchen_capacity)
    
    elif business_type == 'retail':
        # Each cashier can handle ~15 customers/hour
        return config['num_cashiers'] * 15
    
    elif business_type == 'warehouse':
        # Each picker can handle ~8 orders/hour
        return config.get('num_pickers', 5) * 8
    
    else:
        return config['num_servers'] * 10

def generate_poisson_arrivals(rate):
    """Generate Poisson-distributed arrivals"""
    if rate <= 0:
        return 0
    L = math.exp(-rate)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1

def calculate_avg(values):
    """Calculate average, handling empty lists"""
    return sum(values) / len(values) if values else 0

def calculate_statistics(all_customers, served, abandoned, hourly_stats, config, industry):
    """Calculate comprehensive statistics from simulation"""
    
    total = len(all_customers)
    num_served = len(served)
    num_abandoned = len(abandoned)
    
    # Wait time stats
    wait_times = [c['wait_time'] for c in served]
    service_times = [c['service_time'] for c in served]
    total_times = [c['total_time'] for c in served]
    
    # Revenue
    total_revenue = sum(c['ticket_value'] for c in served)
    lost_revenue = sum(c['ticket_value'] for c in abandoned)
    
    # Utilization calculation
    total_service_time = sum(service_times)
    total_available_time = config['operating_hours'] * config['simulation_days'] * 60
    staff_count = config['num_servers'] + config['num_cooks'] + config['num_cashiers']
    utilization = (total_service_time / (total_available_time * staff_count)) * 100 if staff_count > 0 else 0
    
    # Peak hour analysis
    hourly_arrivals = {}
    for stat in hourly_stats:
        hour = stat['hour']
        if hour not in hourly_arrivals:
            hourly_arrivals[hour] = []
        hourly_arrivals[hour].append(stat['arrivals'])
    
    peak_hours = sorted(hourly_arrivals.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)[:3]
    
    # Station utilization (simulated)
    stations = ['Host/Check-in', 'Seating', 'Order Taking', 'Kitchen', 'Food Serving', 'Payment']
    station_utils = []
    base_util = utilization
    for i, station in enumerate(stations):
        # Vary utilization by station
        variance = random.uniform(-15, 15)
        station_util = min(99, max(5, base_util + variance))
        station_utils.append({
            'name': station,
            'utilization': round(station_util, 1),
            'avg_queue': round(random.uniform(0, 5), 1),
            'avg_wait': round(random.uniform(0.5, 5), 1) if station_util > 50 else 0
        })
    
    # Find bottleneck
    bottleneck = max(station_utils, key=lambda x: x['utilization'])
    
    # Calculate scores
    benchmarks = industry.get('benchmarks', {})
    target_wait = benchmarks.get('target_wait_time', 10)
    max_wait = benchmarks.get('max_acceptable_wait', 20)
    avg_wait = calculate_avg(wait_times)
    
    wait_score = max(0, 100 - (avg_wait / target_wait * 50)) if target_wait > 0 else 50
    throughput_score = (num_served / total * 100) if total > 0 else 100
    efficiency_score = min(100, utilization * 1.2) if utilization < 85 else max(50, 150 - utilization)
    overall_score = (wait_score * 0.4 + throughput_score * 0.3 + efficiency_score * 0.3)
    
    return {
        'summary': {
            'total_customers': total,
            'served_customers': num_served,
            'abandoned_customers': num_abandoned,
            'abandonment_rate': round(num_abandoned / total * 100, 1) if total > 0 else 0,
            'throughput_per_hour': round(num_served / (config['operating_hours'] * config['simulation_days']), 1)
        },
        'wait_times': {
            'average': round(avg_wait, 2),
            'median': round(sorted(wait_times)[len(wait_times)//2], 2) if wait_times else 0,
            'min': round(min(wait_times), 2) if wait_times else 0,
            'max': round(max(wait_times), 2) if wait_times else 0,
            'p95': round(sorted(wait_times)[int(len(wait_times) * 0.95)], 2) if wait_times else 0,
            'target': target_wait,
            'max_acceptable': max_wait
        },
        'service': {
            'avg_service_time': round(calculate_avg(service_times), 2),
            'avg_total_time': round(calculate_avg(total_times), 2)
        },
        'financial': {
            'total_revenue': round(total_revenue, 2),
            'lost_revenue': round(lost_revenue, 2),
            'avg_ticket': round(total_revenue / num_served, 2) if num_served > 0 else 0,
            'revenue_per_hour': round(total_revenue / (config['operating_hours'] * config['simulation_days']), 2)
        },
        'utilization': {
            'overall': round(utilization, 1),
            'stations': station_utils
        },
        'bottleneck': {
            'station': bottleneck['name'],
            'utilization': bottleneck['utilization']
        },
        'peak_hours': [{'hour': h, 'avg_arrivals': round(sum(a)/len(a), 1)} for h, a in peak_hours],
        'scores': {
            'overall': round(overall_score, 0),
            'wait_time': round(wait_score, 0),
            'throughput': round(throughput_score, 0),
            'efficiency': round(efficiency_score, 0)
        },
        'hourly_data': aggregate_hourly_data(hourly_stats)
    }

def aggregate_hourly_data(hourly_stats):
    """Aggregate hourly stats across days"""
    hourly_agg = {}
    for stat in hourly_stats:
        hour = stat['hour']
        if hour not in hourly_agg:
            hourly_agg[hour] = {'arrivals': [], 'served': [], 'abandoned': [], 'wait': []}
        hourly_agg[hour]['arrivals'].append(stat['arrivals'])
        hourly_agg[hour]['served'].append(stat['served'])
        hourly_agg[hour]['abandoned'].append(stat['abandoned'])
        if stat['avg_wait']:
            hourly_agg[hour]['wait'].append(stat['avg_wait'])
    
    return [{
        'hour': hour,
        'avg_arrivals': round(sum(data['arrivals'])/len(data['arrivals']), 1),
        'avg_served': round(sum(data['served'])/len(data['served']), 1),
        'avg_abandoned': round(sum(data['abandoned'])/len(data['abandoned']), 1),
        'avg_wait': round(sum(data['wait'])/len(data['wait']), 1) if data['wait'] else 0
    } for hour, data in sorted(hourly_agg.items())]

def generate_recommendations(results, config):
    """Generate actionable recommendations based on simulation results"""
    recommendations = []
    
    # Analyze bottleneck
    bottleneck = results['bottleneck']
    if bottleneck['utilization'] > 90:
        recommendations.append({
            'priority': 'critical',
            'category': 'capacity',
            'title': f"Critical Bottleneck at {bottleneck['station']}",
            'description': f"Operating at {bottleneck['utilization']}% capacity. This is causing delays.",
            'action': f"Add staff or equipment to {bottleneck['station']} to increase capacity.",
            'impact': 'Could reduce wait times by 30-50%',
            'cost': 'Medium',
            'timeline': 'Immediate'
        })
    elif bottleneck['utilization'] > 80:
        recommendations.append({
            'priority': 'high',
            'category': 'capacity',
            'title': f"High Load at {bottleneck['station']}",
            'description': f"Approaching capacity limit at {bottleneck['utilization']}%.",
            'action': 'Consider adding capacity during peak hours.',
            'impact': 'Prevent future bottlenecks',
            'cost': 'Low-Medium',
            'timeline': '1-2 weeks'
        })
    
    # Analyze wait times
    avg_wait = results['wait_times']['average']
    target_wait = results['wait_times']['target']
    if avg_wait > target_wait * 2:
        recommendations.append({
            'priority': 'critical',
            'category': 'service',
            'title': 'Excessive Wait Times',
            'description': f"Average wait ({avg_wait:.1f} min) is over 2x target ({target_wait} min).",
            'action': 'Add staff during peak hours or streamline service process.',
            'impact': f'Reduce wait time by {avg_wait - target_wait:.0f} minutes',
            'cost': 'Medium',
            'timeline': 'Immediate'
        })
    elif avg_wait > target_wait:
        recommendations.append({
            'priority': 'medium',
            'category': 'service',
            'title': 'Wait Times Above Target',
            'description': f"Average wait ({avg_wait:.1f} min) exceeds target ({target_wait} min).",
            'action': 'Review staffing schedule and service procedures.',
            'impact': 'Improve customer satisfaction',
            'cost': 'Low',
            'timeline': '1-2 weeks'
        })
    
    # Analyze abandonment
    abandon_rate = results['summary']['abandonment_rate']
    if abandon_rate > 10:
        lost = results['financial']['lost_revenue']
        recommendations.append({
            'priority': 'critical',
            'category': 'revenue',
            'title': 'High Customer Abandonment',
            'description': f"{abandon_rate:.1f}% of customers leaving. Lost revenue: ${lost:,.0f}.",
            'action': 'Increase capacity or improve service speed.',
            'impact': f'Recover ${lost:,.0f} in revenue',
            'cost': 'Medium',
            'timeline': 'Immediate'
        })
    elif abandon_rate > 5:
        recommendations.append({
            'priority': 'high',
            'category': 'revenue',
            'title': 'Customer Abandonment Above Target',
            'description': f"{abandon_rate:.1f}% abandonment rate. Industry target is under 5%.",
            'action': 'Analyze peak hour staffing and wait time causes.',
            'impact': 'Improve conversion rate',
            'cost': 'Low',
            'timeline': '2-4 weeks'
        })
    
    # Analyze utilization
    utilization = results['utilization']['overall']
    if utilization < 50:
        recommendations.append({
            'priority': 'medium',
            'category': 'cost',
            'title': 'Low Staff Utilization',
            'description': f"Staff utilization at {utilization:.0f}%. You may be overstaffed.",
            'action': 'Consider reducing staff during slow periods.',
            'impact': f'Save {30-40}% on labor costs during slow hours',
            'cost': 'Savings',
            'timeline': '1-2 weeks'
        })
    elif utilization > 95:
        recommendations.append({
            'priority': 'high',
            'category': 'risk',
            'title': 'Staff Burnout Risk',
            'description': f"Staff utilization at {utilization:.0f}%. Risk of burnout and errors.",
            'action': 'Add staff to reduce workload to sustainable levels.',
            'impact': 'Reduce errors, improve retention',
            'cost': 'Medium',
            'timeline': '1-2 weeks'
        })
    
    # Peak hour recommendations
    if results['peak_hours']:
        peak = results['peak_hours'][0]
        recommendations.append({
            'priority': 'low',
            'category': 'scheduling',
            'title': f"Peak Hour Optimization",
            'description': f"Busiest hour is {peak['hour']}:00 with ~{peak['avg_arrivals']:.0f} arrivals.",
            'action': f"Ensure full staffing 30 min before {peak['hour']}:00.",
            'impact': 'Handle peak demand smoothly',
            'cost': 'None',
            'timeline': 'Next schedule'
        })
    
    # Sort by priority
    priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
    
    return recommendations

def calculate_financials(results, config):
    """Calculate detailed financial projections"""
    
    # Revenue
    daily_revenue = results['financial']['revenue_per_hour'] * config['operating_hours']
    weekly_revenue = daily_revenue * 7
    monthly_revenue = weekly_revenue * 4.33
    annual_revenue = monthly_revenue * 12
    
    # Labor costs
    staff_count = (config['num_hosts'] + config['num_servers'] + config['num_cooks'] + 
                   config['num_cashiers'] + config['num_managers'] + config['num_bussers'])
    daily_labor = staff_count * config['hourly_wage'] * config['operating_hours']
    weekly_labor = daily_labor * 7
    monthly_labor = weekly_labor * 4.33
    
    # Rent
    monthly_rent = config['square_footage'] * config['rent_per_sqft'] / 12
    
    # Food/product cost
    monthly_cogs = monthly_revenue * (config['food_cost_percent'] / 100)
    
    # Other costs (utilities, supplies, etc.) - estimate 10% of revenue
    monthly_other = monthly_revenue * 0.10
    
    # Profit
    monthly_profit = monthly_revenue - monthly_labor - monthly_rent - monthly_cogs - monthly_other
    profit_margin = (monthly_profit / monthly_revenue * 100) if monthly_revenue > 0 else 0
    
    # Break-even analysis
    fixed_costs = monthly_rent + (monthly_other * 0.5)
    variable_cost_per_customer = (monthly_labor + monthly_cogs + monthly_other * 0.5) / results['summary']['served_customers'] if results['summary']['served_customers'] > 0 else 0
    avg_revenue_per_customer = results['financial']['avg_ticket']
    contribution_margin = avg_revenue_per_customer - variable_cost_per_customer
    breakeven_customers = fixed_costs / contribution_margin if contribution_margin > 0 else 0
    
    # ROI scenarios
    current_profit = monthly_profit
    
    # If add 1 staff
    additional_staff_cost = config['hourly_wage'] * config['operating_hours'] * 30
    potential_revenue_increase = results['financial']['lost_revenue'] * 0.5 * 4.33  # Capture 50% of lost
    add_staff_profit = current_profit + potential_revenue_increase - additional_staff_cost
    
    return {
        'revenue': {
            'daily': round(daily_revenue, 2),
            'weekly': round(weekly_revenue, 2),
            'monthly': round(monthly_revenue, 2),
            'annual': round(annual_revenue, 2)
        },
        'costs': {
            'labor': {
                'daily': round(daily_labor, 2),
                'monthly': round(monthly_labor, 2),
                'percent_of_revenue': round(monthly_labor / monthly_revenue * 100, 1) if monthly_revenue > 0 else 0
            },
            'rent': {
                'monthly': round(monthly_rent, 2),
                'percent_of_revenue': round(monthly_rent / monthly_revenue * 100, 1) if monthly_revenue > 0 else 0
            },
            'cogs': {
                'monthly': round(monthly_cogs, 2),
                'percent_of_revenue': round(config['food_cost_percent'], 1)
            },
            'other': {
                'monthly': round(monthly_other, 2)
            },
            'total_monthly': round(monthly_labor + monthly_rent + monthly_cogs + monthly_other, 2)
        },
        'profit': {
            'monthly': round(monthly_profit, 2),
            'annual': round(monthly_profit * 12, 2),
            'margin': round(profit_margin, 1)
        },
        'breakeven': {
            'customers_per_month': round(breakeven_customers, 0),
            'customers_per_day': round(breakeven_customers / 30, 0)
        },
        'scenarios': {
            'current': round(current_profit, 2),
            'add_one_staff': round(add_staff_profit, 2),
            'add_staff_roi': round((add_staff_profit - current_profit) / additional_staff_cost * 100, 1) if additional_staff_cost > 0 else 0
        },
        'lost_opportunity': {
            'monthly': round(results['financial']['lost_revenue'] * 4.33, 2)
        }
    }

def generate_sample_business(name):
    """Generate sample business data"""
    return {
        'name': name or 'Sample Restaurant',
        'rating': round(random.uniform(3.5, 4.8), 1),
        'reviews': random.randint(50, 500),
        'price_level': random.randint(1, 3),
        'type': 'restaurant',
        'estimated_daily_customers': random.randint(100, 300)
    }

if __name__ == '__main__':
    print("\n" + "="*60)
    print("DIGITAL SIMULATOR - Professional Edition")
    print("="*60)
    print("✅ Industry benchmarks loaded")
    print("✅ Simulation engine ready")
    print("✅ Financial modeling enabled")
    print("-"*60)
    print("🌐 Open your browser to: http://127.0.0.1:5000")
    print("📊 Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)