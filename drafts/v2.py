import simpy
import random
import pandas as pd

# this is a city-to-city trip model

# model variety of changer types and speeds .. basic proiles 
# model DC fast chargers and Level 2 chargers at charging stations.  DCs only go to 80% then slow to L2 rates.

# model varying departure times

# model major us EV models .. more basic profiles
# https://evmagazine.com/top10/top-10-best-selling-electric-vehicles-in-the-us

# then analysis of wait times, charge times, total trip times etc.

# parameterize and make a shiny app to run the simulation and display results


df_inventory = pd.read_csv('data/inventory.csv')


class SimulationParameters:
    def __init__(self):
        self.sim_duration = 24 * 60  # 24 hours in minutes
        self.num_evs = 1
        self.num_charging_stations = 1
        # Add more parameters as needed

class Driver:
    def __init__(self, env, id, ev, charging_threshold):
        self.env = env
        self.id = id
        self.ev = ev
        self.charging_threshold = charging_threshold # in pct

class EV:
    def __init__(self, id, model, capacity, efficiency, base_charge):
        self.model = model
        self.id = id
        self.capacity = capacity # in kWh
        self.efficiency = efficiency # in pct
        self.current_charge = base_charge  # in kWh

# class Charger:
#     def __init__(self, id, charging_speed, level):
#         self.id = id
#         self.charging_speed = charging_speed
#         self.level = level
#         self.in_use = False
        
class ChargingStation:
    def __init__(self, env, id, num_chargers):
        self.env = env
        self.id = id
        self.chargers = simpy.Resource(env, capacity=num_chargers)

def drive_and_charge(env, driver, charging_stations):
    trip_distance = 346  # km
    trip_distance_covered = 0
    leg_number = 0
    avg_speed = 65 * 1.609 # 65 mph converted to km/h

    # travel until driver's recharge threshold, then charge, then repeat until full trip distance is covered
    while trip_distance_covered < trip_distance:

        leg_number += 1
        max_leg_distance = (driver.ev.current_charge - (driver.ev.capacity * driver.charging_threshold)) / driver.ev.efficiency  # max distance willing to drive on current charge
        leg_distance = min(trip_distance - trip_distance_covered, max_leg_distance)
        # print('leg', leg_number, max_leg_distance, leg_distance, trip_distance_covered, trip_distance)
        
        yield env.timeout(leg_distance / avg_speed * 60) # minutes
        trip_distance_covered += leg_distance
        driver.ev.current_charge -= leg_distance * driver.ev.efficiency

        print(f"EV {driver.ev.id} completed leg {leg_number} of distance {leg_distance} at minute {env.now} (current charge: {driver.ev.current_charge})")

        # Check if charging is needed
        if round(driver.ev.current_charge,1) <= round(driver.charging_threshold * driver.ev.capacity,1):
            # Find a charging station (simplified)
            station = random.choice(charging_stations)

            with station.chargers.request() as req:
                yield req

                # Start charging
                charge_time = (driver.ev.capacity - driver.ev.current_charge) / 11 * 60  # minutes @ 10 kW charging speed (max L2 acceptance rate for most)
                yield env.timeout(charge_time)

                driver.ev.current_charge = driver.ev.capacity
                print(f"EV {driver.ev.id} completed {round(charge_time,0)}-minute charge at station {station.id} at minute {round(env.now,1)}")

    print(f"EV {driver.ev.id} completed trip at minute {round(env.now,1)}")
    

def run_simulation(params):
    env = simpy.Environment()

    # get random EV profile
    def ev_profiles():
        row = random.choices(df_inventory['Model'], df_inventory['Share']) # actual 2023 market share per model
        model= df_inventory.loc[df_inventory['Model'] == row[0]].values[0][0] # model name
        capacity = df_inventory.loc[df_inventory['Model'] == row[0]].values[0][2] # kWh
        efficiency = df_inventory.loc[df_inventory['Model'] == row[0]].values[0][3] # kWh/km
        base_charge = round(random.uniform(0.6, 1) * capacity,1) # random starting charge level 60-100%
        return(model, capacity, efficiency, base_charge)
    
    # get random driver profile
    def driver_profiles():
        charging_threshold=random.choices([.2,.25,.3], weights=[3,2,1])[0] # 20%, 25%, 30% thresholds
        return (charging_threshold)

    # create charging stations
    charging_stations = [ChargingStation(env, i+1, 2) for i in range(params.num_charging_stations)]
    print(f"Created {params.num_charging_stations} charging stations")

    # create EVs and drivers
    for i in range(params.num_evs):
        model, capacity, efficiency, base_charge = ev_profiles()
        charging_threshold = driver_profiles()
        ev = EV(i+1, model, capacity, efficiency, base_charge)
        driver = Driver(env, i+1, ev, charging_threshold)
        env.process(drive_and_charge(env, driver, charging_stations))

        # print a report of the EVs and drivers
        print(f"Created EV {ev.id} -  model: {ev.model}, capacity: {ev.capacity}, current charge: {ev.current_charge}, efficiency:{ev.efficiency}, range {round(ev.capacity / ev.efficiency,1)}, threshold: {driver.charging_threshold}")

    # run simulation
    env.run(until=params.sim_duration)

if __name__ == "__main__":
    params = SimulationParameters()
    run_simulation(params)

