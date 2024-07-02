import simpy
import random

# active factors:
# - EV.battery_capacity

# active assumptions:
# all chargers are 10kw 
# EV.efficiency is 0.25 kWh/hm (average)
# Driver.charging_threshold is 0.2 (20% charge remaining)


class SimulationParameters:
    def __init__(self):
        self.sim_duration = 24 * 60  # 24 hours in minutes
        self.num_evs = 5
        self.num_charging_stations = 1
        # Add more parameters as needed

class Driver:
    def __init__(self, env, id, ev, charging_threshold):
        self.env = env
        self.id = id
        self.ev = ev
        self.charging_threshold = charging_threshold # in pct

class EV:
    def __init__(self, id, battery_capacity, efficiency):
        self.id = id
        self.battery_capacity = battery_capacity # in kWh
        self.current_charge = battery_capacity  # in kWh
        self.efficiency = efficiency # in pct

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
        max_leg_distance = (driver.ev.current_charge - (driver.ev.battery_capacity * driver.charging_threshold)) / driver.ev.efficiency  # max distance willing to drive on current charge
        leg_distance = min(trip_distance - trip_distance_covered, max_leg_distance)
        
        yield env.timeout(leg_distance / avg_speed * 60) # minutes
        trip_distance_covered += leg_distance
        driver.ev.current_charge -= leg_distance * driver.ev.efficiency

        print(f"EV {driver.ev.id} completed leg {leg_number} of distance {round(leg_distance,1)} at minute {round(env.now,1)} (current charge: {round(driver.ev.current_charge,1)})")

        # Check if charging is needed
        if driver.ev.current_charge <= driver.charging_threshold * driver.ev.battery_capacity:
            # Find a charging station (simplified)
            station = random.choice(charging_stations)

            with station.chargers.request() as req:
                yield req

                # Start charging
                charge_time = (driver.ev.battery_capacity - driver.ev.current_charge) / 10 * 60  # minutes @ 10 kW charging speed
                yield env.timeout(charge_time)

                driver.ev.current_charge = driver.ev.battery_capacity
                print(f"EV {driver.ev.id} completed {round(charge_time,0)}-minute charge at station {station.id} at minute {round(env.now,1)}")

    print(f"EV {driver.ev.id} completed trip at minute {round(env.now,1)}")
    

def run_simulation(params):
    env = simpy.Environment()

    # create charging stations
    charging_stations = [ChargingStation(env, i+1, 2) for i in range(params.num_charging_stations)]
    print(f"Created {params.num_charging_stations} charging stations")

    # create EVs and drivers
    for i in range(params.num_evs):
        ev = EV(id=i+1, battery_capacity=round(random.uniform(60, 100),0), efficiency=0.25) # add distributions for battery capacity, efficiency, base charge level
        driver = Driver(env, id=i+1, ev=ev, charging_threshold=0.2) # add distribution for charging thresholds
        env.process(drive_and_charge(env, driver, charging_stations))

        # print a report of the EVs and drivers
        print(f"Created EV {ev.id} with battery capacity {ev.battery_capacity} kWh and efficiency {ev.efficiency}% (max range {round(ev.battery_capacity / ev.efficiency,1)} km)")

    # run simulation
    env.run(until=params.sim_duration)

if __name__ == "__main__":
    params = SimulationParameters()
    run_simulation(params)