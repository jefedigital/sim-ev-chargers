import simpy
import random
from enum import Enum

class EVType(Enum):
    TESLA = 1
    OTHER_BEV = 2
    PHEV = 3

class ChargingLocationType(Enum):
    HOME = 1
    WORK = 2
    PUBLIC_L2 = 3
    PUBLIC_DC_FAST = 4

class SimulationParameters:
    def __init__(self):
        self.sim_duration = 365 * 24 * 60  # 1 year in minutes
        self.num_evs = 1000
        self.num_charging_stations = {
            ChargingLocationType.PUBLIC_L2: 100,
            ChargingLocationType.PUBLIC_DC_FAST: 20
        }
        self.workplace_charging_access_rate = 0.3  # 30% of drivers have access
        self.avg_annual_miles = 5300  # Based on Burlig et al. study

class EV:
    def __init__(self, env, ev_id, ev_type):
        self.env = env
        self.id = ev_id
        self.type = ev_type
        self.battery_capacity = self.get_battery_capacity()
        self.current_charge = self.battery_capacity  # Start fully charged
        self.miles_driven = 0
        self.has_workplace_charging = random.random() < SimulationParameters().workplace_charging_access_rate

    def get_battery_capacity(self):
        if self.type == EVType.TESLA:
            return random.uniform(75, 100)  # kWh
        elif self.type == EVType.OTHER_BEV:
            return random.uniform(40, 75)  # kWh
        else:  # PHEV
            return random.uniform(8, 20)  # kWh

class ChargingStation:
    def __init__(self, env, station_id, location_type, num_chargers):
        self.env = env
        self.id = station_id
        self.location_type = location_type
        self.chargers = simpy.Resource(env, capacity=num_chargers)

def drive_and_charge(env, ev, charging_stations):
    while True:
        # Simulate driving
        daily_miles = random.gauss(SimulationParameters().avg_annual_miles / 365, 10)
        energy_used = daily_miles / 3  # Assuming 3 miles per kWh on average
        ev.current_charge = max(0, ev.current_charge - energy_used)
        ev.miles_driven += daily_miles

        # Determine charging needs
        if ev.current_charge < 0.2 * ev.battery_capacity:
            # Choose charging location
            if random.random() < 0.75:  # 75% chance of home charging
                yield env.process(charge(env, ev, charging_stations[ChargingLocationType.HOME][ev.id]))
            elif ev.has_workplace_charging and 6 <= env.now % (24 * 60) < 18 * 60:  # Workday hours
                yield env.process(charge(env, ev, random.choice(charging_stations[ChargingLocationType.WORK])))
            else:
                # Choose between public L2 and DC fast charging
                station_type = random.choices(
                    [ChargingLocationType.PUBLIC_L2, ChargingLocationType.PUBLIC_DC_FAST],
                    weights=[0.7, 0.3]
                )[0]
                yield env.process(charge(env, ev, random.choice(charging_stations[station_type])))

        # Wait until next day
        yield env.timeout(24 * 60)  # 24 hours in minutes

def charge(env, ev, charging_station):
    with charging_station.chargers.request() as req:
        yield req

        charge_rate = get_charge_rate(ev, charging_station.location_type)
        charge_time = min((ev.battery_capacity - ev.current_charge) / charge_rate, 8 * 60)  # Max 8 hours
        
        yield env.timeout(charge_time)
        ev.current_charge = min(ev.current_charge + charge_rate * charge_time / 60, ev.battery_capacity)

def get_charge_rate(ev, location_type):
    if location_type == ChargingLocationType.HOME:
        return 7.2 if random.random() < 0.7 else 1.4  # 70% Level 2, 30% Level 1
    elif location_type == ChargingLocationType.WORK:
        return 7.2  # Level 2
    elif location_type == ChargingLocationType.PUBLIC_L2:
        return 7.2  # Level 2
    else:  # DC Fast
        return 50 if ev.type != EVType.PHEV else 25  # PHEVs typically charge slower

def run_simulation(params):
    env = simpy.Environment()
    
    # Create charging stations
    charging_stations = {
        ChargingLocationType.HOME: [ChargingStation(env, f"Home_{i}", ChargingLocationType.HOME, 1) for i in range(params.num_evs)],
        ChargingLocationType.WORK: [ChargingStation(env, f"Work_{i}", ChargingLocationType.WORK, 4) for i in range(int(params.num_evs * params.workplace_charging_access_rate / 4))],
        ChargingLocationType.PUBLIC_L2: [ChargingStation(env, f"PublicL2_{i}", ChargingLocationType.PUBLIC_L2, 2) for i in range(params.num_charging_stations[ChargingLocationType.PUBLIC_L2])],
        ChargingLocationType.PUBLIC_DC_FAST: [ChargingStation(env, f"PublicDC_{i}", ChargingLocationType.PUBLIC_DC_FAST, 1) for i in range(params.num_charging_stations[ChargingLocationType.PUBLIC_DC_FAST])]
    }
    
    # Create EVs and start their processes
    evs = []
    for i in range(params.num_evs):
        ev_type = random.choices([EVType.TESLA, EVType.OTHER_BEV, EVType.PHEV], weights=[0.2, 0.5, 0.3])[0]
        ev = EV(env, i, ev_type)
        evs.append(ev)
        env.process(drive_and_charge(env, ev, charging_stations))
    
    # Run the simulation
    env.run(until=params.sim_duration)

    # Collect and print results
    total_miles = sum(ev.miles_driven for ev in evs)
    avg_miles = total_miles / params.num_evs
    print(f"Average annual miles driven per EV: {avg_miles:.2f}")

if __name__ == "__main__":
    params = SimulationParameters()
    run_simulation(params)