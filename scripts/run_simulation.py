# ============================================================
# RUN SIMULATION — runner script
# ============================================================

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from simulation.grid_simulator import simulate_demand

if __name__ == "__main__":
    simulate_demand(n=24, scenario="random")
