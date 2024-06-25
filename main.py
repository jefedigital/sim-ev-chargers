import simpy
import random

class SimulationParameters:
    def __init__(self):
        self.sim_duration = 24 * 60  # 24 hours in minutes
        self.num_evs = 100
        self.num_charging_stations = 10
        # Add more parameters as needed

class EV:
    def __init__(self, env, ev_id, battery_capacity, charging_speed):
        self.env = env
        self.id = ev_id
        self.battery_capacity = battery_capacity
        self.charging_speed = charging_speed
        self.current_charge = battery_capacity  # Start fully charged

class ChargingStation:
    def __init__(self, env, station_id, num_chargers):
        self.env = env
        self.id = station_id
        self.chargers = simpy.Resource(env, capacity=num_chargers)

class Driver:
    def __init__(self, env, driver_id, ev):
        self.env = env
        self.id = driver_id
        self.ev = ev

def drive_and_charge(env, driver, charging_stations):
    while True:
        # Simulate driving
        trip_distance = random.randint(20, 100)  # km
        yield env.timeout(trip_distance)  # Assume 1 minute per km for simplicity
        
        # Check if charging is needed
        if driver.ev.current_charge < 0.2 * driver.ev.battery_capacity:  # 20% threshold
            # Find a charging station (simplified)
            station = random.choice(charging_stations)
            
            with station.chargers.request() as req:
                yield req
                
                # Start charging
                charge_time = (driver.ev.battery_capacity - driver.ev.current_charge) / driver.ev.charging_speed
                yield env.timeout(charge_time)
                
                driver.ev.current_charge = driver.ev.battery_capacity
                print(f"EV {driver.ev.id} charged at station {station.id} at time {env.now}")

def run_simulation(params):
    env = simpy.Environment()
    
    # Create charging stations
    charging_stations = [ChargingStation(env, i, 2) for i in range(params.num_charging_stations)]
    
    # Create EVs and drivers
    for i in range(params.num_evs):
        ev = EV(env, i, battery_capacity=75, charging_speed=50)  # 75 kWh battery, 50 kW charging
        driver = Driver(env, i, ev)
        env.process(drive_and_charge(env, driver, charging_stations))
    
    # Run the simulation
    env.run(until=params.sim_duration)

if __name__ == "__main__":
    params = SimulationParameters()
    run_simulation(params)