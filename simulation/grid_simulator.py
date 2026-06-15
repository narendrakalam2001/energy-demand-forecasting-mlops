# ============================================================
# GRID SIMULATOR — Energy Demand Forecasting
# ============================================================
# Simulates realistic hourly energy demand requests to the API.
# 3 scenarios:
#   random  — mixed realistic hours (default)
#   peak    — evening peak hour profile (high load)
#   offpeak — overnight / weekend low-demand profile
#
# Real-world analog: POSOCO / NLDC India run similar load
# simulation drills before grid stress events.
# ============================================================

import requests
import random
import time
import os
import numpy as np

API_URL = os.getenv("ENERGY_API_URL", "http://localhost:8000") + "/predict"

# ── Spain grid reference statistics (2015-2018 ESIOS data) ───
SPAIN_DEMAND_MEAN = 30500   # MW
SPAIN_DEMAND_STD  =  5000   # MW
PEAK_HOURS        = list(range(18, 23))    # 6 PM – 10 PM
OFFPEAK_HOURS     = list(range(0,  7))     # midnight – 6 AM


# ============================================================
# GENERATE SYNTHETIC HOURLY INPUT
# ============================================================

def generate_hour_input(scenario: str = "random") -> dict:
    """
    Scenarios:
        random  — any realistic hour
        peak    — evening peak with high demand lags
        offpeak — overnight low-demand period
    """

    if scenario == "peak":
        hour        = random.choice(PEAK_HOURS)
        day_of_week = random.randint(0, 4)     # weekday peak is worst
        month       = random.choice([1, 2, 7, 8])   # winter/summer
        base_demand = random.uniform(38000, 45000)

    elif scenario == "offpeak":
        hour        = random.choice(OFFPEAK_HOURS)
        day_of_week = random.choice([5, 6])    # weekend overnight
        month       = random.choice([4, 5, 9, 10])  # spring/autumn
        base_demand = random.uniform(18000, 25000)

    else:  # random
        hour        = random.randint(0, 23)
        day_of_week = random.randint(0, 6)
        month       = random.randint(1, 12)

        # Realistic demand by hour
        if hour in PEAK_HOURS:
            base_demand = random.uniform(35000, 43000)
        elif hour in OFFPEAK_HOURS:
            base_demand = random.uniform(18000, 27000)
        else:
            base_demand = random.uniform(27000, 38000)

    is_weekend = 1 if day_of_week >= 5 else 0

    # Temperature by month (Spain, Celsius)
    monthly_temp = {
        1: 8, 2: 9, 3: 12, 4: 15, 5: 19, 6: 24,
        7: 28, 8: 28, 9: 23, 10: 17, 11: 12, 12: 9
    }
    temp_celsius = monthly_temp[month] + random.uniform(-5, 5)

    # Rolling stats consistent with base demand
    demand_lag_1h   = base_demand + random.uniform(-1000, 1000)
    demand_lag_24h  = base_demand + random.uniform(-2000, 2000)
    demand_lag_168h = base_demand + random.uniform(-3000, 3000)

    roll_mean = (demand_lag_1h + demand_lag_24h + demand_lag_168h) / 3
    roll_std  = abs(demand_lag_1h - demand_lag_24h) / 2 + 500

    return {
        "hour":                  hour,
        "day_of_week":           day_of_week,
        "month":                 month,
        "is_weekend":            is_weekend,
        "demand_lag_1h":         round(demand_lag_1h, 1),
        "demand_lag_24h":        round(demand_lag_24h, 1),
        "demand_lag_168h":       round(demand_lag_168h, 1),
        "demand_roll_mean_24h":  round(roll_mean, 1),
        "demand_roll_std_24h":   round(roll_std, 1),
        "temp_celsius":          round(temp_celsius, 1),
    }


# ============================================================
# SEND TO API + PRINT RESULT
# ============================================================

def send_hour(payload: dict, idx: int):
    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            action = result["dispatch_action"]
            print(
                f"[{idx+1:02d}]  "
                f"Hour={payload['hour']:02d}  "
                f"DoW={payload['day_of_week']}  "
                f"Mon={payload['month']:02d}  "
                f"Lag1h={payload['demand_lag_1h']:,.0f}MW  "
                f"→  Forecast={result['forecast_mw']:,.0f}MW  "
                f"Band={result['demand_band']:<7}  "
                f"Action={action}"
            )
        else:
            print(f"[{idx+1:02d}] API error: HTTP {response.status_code}")

    except Exception as e:
        print(f"[{idx+1:02d}] Connection error: {e}")


# ============================================================
# RUN SIMULATION
# ============================================================

def simulate_demand(n: int = 24, scenario: str = "random"):
    print(f"\nSimulating {n} hourly requests  |  scenario={scenario}")
    print("-" * 75)

    for i in range(n):
        payload = generate_hour_input(scenario)
        send_hour(payload, i)
        time.sleep(0.3)

    print("-" * 75)
    print("Simulation complete\n")


if __name__ == "__main__":
    simulate_demand(24, scenario="random")
