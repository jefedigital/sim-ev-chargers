import simpy 
import random
import numpy as np
import pandas as pd
import logging

# discrete event simulation of a point-to-point electric vehicle charging network

# Issues

# simplify drive time 
# need to run optimization multiple times?
# response curve possible?
# add analysis
# 'wait' animation and 'no optimization result' message

# MVP Features

# app gets scrollable table of df 

# unit tests?

## Future Features

# real drive-time map 
# addl logic for find station - got to next if reachable, else go to closest (avoid backtracking)
# addl logic for charging to 80/90/100 pct
# parameterize avg speed
# "density" parameters .. stations per 100km, wait time (queue plus charge) per 100km etc.
# probability of trip start times (rush hour etc.)

## 


# get sample EV profiles
df_inventory = pd.read_csv('data/inventory.csv')


class SimulationParameters:
    def __init__(self):
        self.route = 'nyc_boston'
        self.sim_duration = 2 * 24 * 60  # 24 hours in minutes
        self.num_drivers = 5
        self.num_stations = 5
        self.num_chargers = 5
        self.loc_destination = 348  # km (NYC to Boston = 348)
        self.avg_speed = 104.585  # km/h (65 mph * 1.609)
        self.recharge_level_pct = 0.2 # pct


class Charger(simpy.Resource):
    def __init__(self, env, id, charging_speed, release_time):
        super().__init__(env, capacity=1)
        self.id = id
        self.charging_speed = charging_speed
        self.release_time = release_time

    def update_release_time(self, charging_time):
        self.release_time = max(self.release_time, self._env.now) + charging_time
    

class ChargingStation:
    def __init__(self, env, id, location, chargers):
        self.env = env
        self.id = id
        self.location = location
        self.chargers = chargers

    def get_earliest_charger(self):
        return sorted(self.chargers, key=lambda c: (-c.charging_speed, c.release_time))[0]
     
    def reserve_charger(self, charge_amount):
        charger = self.get_earliest_charger()
        charge_time = charge_amount / charger.charging_speed * 60 # minutes
        charger.update_release_time(charge_time) 
        return charger.request()


class ChargingNetwork:
    def __init__(self, stations):
        self.stations = stations

    def find_nearest_station(self, loc_current):
        nearest_station = None
        min_distance = float('inf')
        for station in self.stations:
            abs_distance = abs(station.location - loc_current)
            if abs_distance < min_distance:
                min_distance = abs_distance
                distance = station.location - loc_current
                nearest_station = station
        return nearest_station, distance # distance may be +/-
    
    def get_station(self, station_loc):
        for station in self.stations:
            if station.location == station_loc:
                return station
    
    def reserve_available_charger(self, station, charge_amount):
        return station.reserve_charger(charge_amount)
    

class Driver:
    instances = []

    def __init__(self, env, id, station_list, charging_network, avg_speed, loc_destination, model, battery_capacity, efficiency, battery_level, recharge_level_pct):
        self.env = env
        self.id = id
        self.station_list = station_list
        self.charging_network = charging_network
        self.avg_speed = avg_speed
        self.loc_destination = loc_destination
        self.model = model
        self.battery_capacity = battery_capacity # kWh
        self.efficiency = efficiency # kWh/km
        self.battery_level = battery_level # kWh
        self.recharge_level_pct = recharge_level_pct # pct
        self.avg_rate = self.avg_speed/60  # km/m
        self.loc_current = 0
        self.mileage = 0
        self.queue_time = 0
        self.charge_time = 0
        self.charges = 0
        self.trip_start_time = 0
        self.trip_end_time = 0
        self.trip_time = 0
        self.db = 0
        self.num_stations = 0
        Driver.instances.append(self)

    def check_battery(self):
        if self.battery_level <= 0:
            self.battery_level = 0
            raise DeadBatteryError(f"Driver {self.id} has a dead battery, towed to location {self.loc_current}")
       

    def drive_cycle(self):
        driver_log_msg(self, (f"Started trip in {self.model} with {self.battery_capacity} kWh battery"))

        # trip start metrics
        self.trip_start_time = self.env.now
        self.num_stations = len(self.charging_network.stations) # for logging

        # start driving
        while self.loc_current < self.loc_destination:

            # drive until 20% battery level or destination
            drive_distance_max = (self.battery_level - (self.battery_capacity * self.recharge_level_pct)) / self.efficiency # max distance to 20% on current battery_level
            drive_distance = min(self.loc_destination - self.loc_current, drive_distance_max)

            drive_time = abs(drive_distance) / self.avg_speed * 60 # minutes
            self.loc_current += drive_distance # adjust location
            self.mileage += abs(drive_distance) # add mileage
            self.battery_level -= abs(drive_distance) * self.efficiency # adjust battery level
            self.check_battery() # check whether enough battery to complete

            yield self.env.timeout(drive_time) # driving time

            driver_log_msg(self, "completed leg")
            
            # drive to nearest station
            if self.loc_current < self.loc_destination:

                station, drive_distance = self.charging_network.find_nearest_station(self.loc_current)

                drive_time = abs(drive_distance) / self.avg_speed * 60 # minutes
                self.loc_current += drive_distance # adjust location
                self.mileage += abs(drive_distance) # add mileage
                self.battery_level -= abs(drive_distance) * self.efficiency # adjust battery level
                self.check_battery() # check whether enough battery to complete

                yield self.env.timeout(drive_time) # driving time

                driver_log_msg(self, f"arrived at station {station.id}")

                # reserve available charger
                charge_amount = self.battery_capacity - self.battery_level
                charger_request = self.charging_network.reserve_available_charger(station, charge_amount)

                driver_log_msg(self, f"started queue for charger")
                queue_start = self.env.now

                with charger_request as req:

                    yield req 
                    
                    charger = req.resource

                    self.charges += 1
                    self.battery_level = self.battery_capacity # assuming 100% charge

                    charge_time = charge_amount / charger.charging_speed * 60 # minutes
                    self.charge_time += charge_time
            
                    self.queue_time += (self.env.now - queue_start) if (self.env.now - queue_start > charge_time) else 0
               
                    yield self.env.timeout(charge_time)
                    
                    driver_log_msg(self, f"used charger {charger.id} @ {charger.charging_speed} kwH at station {station.id}")
        
        driver_log_msg(self, "completed trip")

        # trip end metrics
        self.trip_end_time = self.env.now
        self.trip_time = self.trip_end_time - self.trip_start_time
        self.battery_level_end = self.battery_level
        
## Alternative drive cycle with station list
    def drive_cycle2(self):
        driver_log_msg(self, (f"Started trip in {self.model} with {self.battery_capacity} kWh battery"))

        # trip start vars
        self.trip_start_time = self.env.now
        stations_arr = np.array(self.station_list)
        self.num_stations = len(stations_arr)

        while self.loc_current < self.loc_destination:

            # max distance on current battery level
            drive_distance_max = self.battery_level / self.efficiency
            
            # find next best station or destination
            if self.loc_destination < self.loc_current + drive_distance_max:
                dest = self.loc_destination # if final destination reachable, go there
                station = None
            else:
                # if not, find next best station
                station_dist_arr = self.loc_current + drive_distance_max - np.array(stations_arr)
                station_dist_arr[station_dist_arr < 0] = 0
                dest = stations_arr[np.argmin(station_dist_arr[np.nonzero(station_dist_arr)])]
                station = self.charging_network.get_station(dest)

            # if no station reachable, go to max distance and trigger 'dead battery' error
            if dest < self.loc_current + drive_distance_max:
                dest = drive_distance_max
                station = None

            # drive  
            drive_distance = dest - self.loc_current
            drive_time = abs(drive_distance) / self.avg_speed * 60 # minutes
            self.loc_current += drive_distance # adjust location
            self.mileage += abs(drive_distance) # add mileage
            self.battery_level -= abs(drive_distance) * self.efficiency # adjust battery level
            yield self.env.timeout(drive_time) # driving time
            driver_log_msg(self, "completed leg")

            if station:
                # reserve available charger
                charge_amount = self.battery_capacity - self.battery_level
                charger_request = self.charging_network.reserve_available_charger(station, charge_amount)

                driver_log_msg(self, f"started queue for charger")
                queue_start = self.env.now

                with charger_request as req:

                    yield req 
                    
                    charger = req.resource

                    self.charges += 1
                    self.battery_level = self.battery_capacity

                    charge_time = charge_amount / charger.charging_speed * 60 # minutes
                    self.charge_time += charge_time
            
                    self.queue_time += (self.env.now - queue_start) if (self.env.now - queue_start > charge_time) else 0
               
                    yield self.env.timeout(charge_time)
                    
                    driver_log_msg(self, f"used charger {charger.id} @ {charger.charging_speed} kwH at station {station.id}")

            elif self.battery_level <= 0:
                self.battery_level = 0
                raise DeadBatteryError(f"Driver {self.id} has a dead battery at location {self.loc_current}")
            
        driver_log_msg(self, "completed trip")

        # trip end metrics
        self.trip_end_time = self.env.now
        self.trip_time = self.trip_end_time - self.trip_start_time
        self.battery_level_end = self.battery_level


    def run(self):
        try:
            yield self.env.process(self.drive_cycle2())
            
        except DeadBatteryError as e:
                log_msg(self.env, f"Error: {e}", level=logging.WARNING)
                self.db = 1


class DeadBatteryError(Exception):
    "Raised when the battery level <= 0"
    pass


def random_ev_profile():
    row = random.choices(df_inventory['Model'], df_inventory['Share']) # actual 2023 market share per model
    model= df_inventory.loc[df_inventory['Model'] == row[0]].values[0][0] # model name
    battery_capacity = df_inventory.loc[df_inventory['Model'] == row[0]].values[0][2] # kWh
    efficiency = df_inventory.loc[df_inventory['Model'] == row[0]].values[0][3] # kWh/km
    battery_level = round(random.uniform(0.6, 1) * battery_capacity,1) # random starting charge level 60-100%
    return(model, battery_capacity, efficiency, battery_level)



def setup_simulation(env, num_drivers, num_stations, num_chargers, loc_destination, avg_speed, recharge_level_pct):

    log_msg(env, f"\n\n=== \n\nBegan simulation with parameters - drivers: {num_drivers}, stations: {num_stations}, chargers: {num_chargers}, destination: {loc_destination}, speed: {avg_speed}, recharge level: {recharge_level_pct}")
    
    # initialize charging stations and network
    stations = [
        ChargingStation(
            env,
            id = n+1,
            location = np.round(np.linspace(0,loc_destination,num_stations),0)[n],
            chargers=[Charger(env, 
                              id = i+1, 
                              charging_speed=random.choice([20, 50, 150]),
                              release_time = i)
                      for i in range(num_chargers)]
        )
        for n in range(num_stations)
    ]

    station_list = [sorted([station.location for station in stations])]
    log_msg(env, "Stations at " + ', '.join([f"{s:.1f}" for s in sorted([station.location for station in stations])]))
     
    charging_network = ChargingNetwork(stations)
    
    # initialize drivers and start their trips
    for i in range(num_drivers):
        model, battery_capacity, efficiency, battery_level = random_ev_profile()
        
        driver = Driver(env, i+1, station_list, charging_network, avg_speed, loc_destination, model, battery_capacity, efficiency, battery_level, recharge_level_pct)
        
        env.process(driver.run())


def run_simulation(params, sweep = False):
    setup_logging()
    env = simpy.Environment()
    
    # setup sim
    setup_simulation(env, params.num_drivers, params.num_stations, params.num_chargers, params.loc_destination, params.avg_speed, params.recharge_level_pct)

    # run sim
    env.run(until=params.sim_duration)

    # process and return results
    if not sweep:
        results_df, results_dict = process_results()
        Driver.instances = []
        return results_df, results_dict


def parameter_sweep_chargers(params):

    for num_stations in range(1, 101):
        params.num_stations = num_stations
        run_simulation(params, sweep = True)
        results_df, results_dict = process_results()

    # constraints: no dead batteries, trip time under limit, min num_stations
    trip_time_limit = params.loc_destination / params.avg_speed * 60 * 1.25 # 25% over straight shot (200 mins for NYC to Boston)
    filter_results_df = results_df.groupby('num_stations').filter(
        lambda x: x['db'].sum() == 0 and x['trip_time'].mean() <= trip_time_limit)
    min_num_stations = filter_results_df['num_stations'].min()
    min_results_df = filter_results_df[filter_results_df['num_stations'] == min_num_stations]

    Driver.instances = []
    results_df.to_csv('qa/sweep_results.csv', index=False) # save for qa
    min_results_df.to_csv('qa/sweep_min_results.csv', index=False) # save for qa

    0 if np.isnan(min_num_stations) else min_num_stations
    return min_num_stations


def process_results(): # (env)

    # create driver dataframe
    results_df = create_results_dataframe()

    # calc final metrics
    avg_mileage = results_df["mileage"].mean()
    avg_trip_time = results_df["trip_time"].mean()
    avg_queue_time = results_df["queue_time"].mean()
    avg_charging_time = results_df["charge_time"].mean()
    dead_batteries =  results_df["db"].sum()

# solve for no env here

    # # log final metrics
    # log_msg(env, f"Average mileage: {avg_mileage:.1f}")
    # log_msg(env, f"Average trip time: {avg_trip_time.mean():.1f}")
    # log_msg(env, f"Average queue time: {avg_queue_time:.1f}")
    # log_msg(env, f"Average charging time: {avg_charging_time:.1f}")
    # log_msg(env, f"Dead batteries: {dead_batteries:.0f}")

    # dict to pass results to dash app
    results_dict = {
        'avg_mileage': avg_mileage,
        'avg_trip_time': avg_trip_time,
        'avg_queue_time': avg_queue_time,
        'avg_charging_time': avg_charging_time,
        'dead_batteries': dead_batteries
    }

    return results_df, results_dict


def create_results_dataframe():
    data = {
        'num_stations': [], 
        'driver_id': [],
        'model': [],
        'battery_capacity': [],
        'efficiency': [],
        'battery_level_end': [],
        'mileage': [],
        'trip_start_time': [],
        'trip_end_time': [],
        'trip_time': [],
        'queue_time': [],
        'charge_time': [],
        'charges': [],
        'db': []
    }
    for driver in Driver.instances:
        data['num_stations'].append(driver.num_stations)
        data['driver_id'].append(driver.id)
        data['model'].append(driver.model)
        data['battery_capacity'].append(driver.battery_capacity)
        data['efficiency'].append(driver.efficiency)
        data['battery_level_end'].append(driver.battery_level)
        data['trip_start_time'].append(driver.trip_start_time)
        data['trip_end_time'].append(driver.trip_end_time)
        data['trip_time'].append(driver.trip_time)
        data['mileage'].append(driver.mileage)
        data['queue_time'].append(driver.queue_time)
        data['charge_time'].append(driver.charge_time)
        data['charges'].append(driver.charges)
        data['db'].append(driver.db)

    df = pd.DataFrame(data)
    df.to_csv('qa/results.csv', index=False) # save for qa
    return df


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        filename = 'logs/ev_simulation.log',
        filemode='w', # overwrite log file
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def driver_log_msg(driver, message):
    status = "Driver %s [Loc: %.1f Mlg: %.1f Bat: %.1f QueT: %.1f ChgT: %.1f] - %s" % (driver.id, driver.loc_current, driver.mileage, driver.battery_level, driver.queue_time, driver.charge_time, message)
    log_msg(driver.env, status)


def log_msg(env, message, level=logging.INFO):
    timestamp = round(env.now, 1)
    logger = logging.getLogger('EVSimulation')
    logger.log(level, f"[{timestamp}] {message}")


if __name__ == "__main__":
    params = SimulationParameters()
    run_simulation(params)