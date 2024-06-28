import simpy 
import random

# may have to run double simluation, one for drivers and one for chargers


class SimulationParameters:
    def __init__(self):
        self.sim_duration = 24 * 60  # 24 hours in minutes
        self.num_drivers = 2
        self.num_stations = 10
        self.num_chargers = 3
        self.loc_destination = 350  # km (NYC to Boston = 346)
        self.avg_speed = 104.585  # km/h (65 mph * 1.609)

class Charger(simpy.Resource):
    def __init__(self, env, charging_speed, charger_type):
        super().__init__(env, capacity=1)
        self.charging_speed = charging_speed
        self.charger_type = charger_type

class ChargingStation:
    def __init__(self, env, location, chargers):
        self.env = env
        self.location = location
        self.chargers = chargers
        self.queue = simpy.Store(env)

    def get_available_charger(self):
        for charger in self.chargers:
            if charger.count == 0:
                return charger
        return None

    def add_to_queue(self, driver):
        return self.queue.put(driver)

class ChargingNetwork:
    def __init__(self, stations):
        self.stations = stations

    def find_nearest_available_station(self, loc_current):
        nearest_station = None
        min_distance = float('inf')
        for station in self.stations:
            abs_distance = abs(station.location - loc_current)
            if abs_distance < min_distance:
                min_distance = abs_distance
                distance = station.location - loc_current
                nearest_station = station
        return nearest_station, distance # distance may be +/-
    
    def request_charger(self, driver):
        station, distance = self.find_nearest_available_station(driver.loc_current)
        charger = station.get_available_charger()
        if charger is None:
            yield station.add_to_queue(driver)
            charger = yield station.queue.get()
        return charger, station, distance
    
class Driver:
    def __init__(self, env, id, charging_network, avg_speed, loc_destination, data_collector):
        self.env = env
        self.id = id
        self.charging_network = charging_network
        self.avg_speed = avg_speed
        self.loc_destination = loc_destination
        self.data_collector = data_collector

        self.avg_rate = self.avg_speed/60  # km/m
        self.battery_capacity = 60  # kWh
        self.battery_level = 60  # kWh
        self.efficiency = 0.3  # kWh/km
        self.loc_current = 0
        self.mileage = 0
        
    def drive(self):
        while self.loc_current < self.loc_destination:

            # drive until 20% battery level or destination
            drive_distance_max = (self.battery_level - (self.battery_capacity * 0.2)) / self.efficiency # max distance to 20% on current battery_level
            drive_distance = min(self.loc_destination - self.loc_current, drive_distance_max)
            drive_time = drive_distance / self.avg_speed * 60

            yield self.env.timeout(drive_time) # driving time
            self.loc_current += drive_distance # adjust location
            self.mileage += drive_distance # add mileage
            self.battery_level -= drive_distance * self.efficiency # adjust battery level

            print(f"Driver {self.id} drove for {round(drive_time,1)} minutes and is now at location {self.loc_current} with charge {round(self.battery_level,1)} kWh")

            if self.need_charge():
                charger, station, distance = yield self.env.process(self.charging_network.request_charger(self))
                with charger.request() as req:
                    yield req
                    yield self.env.process(self.charge(charger, distance))
                station.queue.put(charger)

    def need_charge(self):
        return self.battery_level <= (self.battery_capacity * 0.2)

    def charge(self, charger, distance):

        # drive to nearest station
        drive_time = abs(distance) / self.avg_speed * 60
        self.loc_current += distance  # adjust location
        self.mileage += abs(distance) # add mileage
        self.battery_level -= abs(distance) * self.efficiency # adjust battery level

        self.check_battery()
        yield self.env.timeout(drive_time) # driving time

        # charge battery
        charge_amount = self.battery_capacity - self.battery_level
        charge_time = charge_amount / charger.charging_speed * 60 # minutes 

        self.data_collector.record_charging_start(self.id, self.env.now)
        yield self.env.timeout(charge_time)
        self.data_collector.record_charging_end(self.id, self.env.now)
        self.battery_level = self.battery_capacity

        print(f"Driver {self.id} drove {round(abs(distance),1)} km in {round(drive_time,1)} mins to charger at {self.loc_current} km, charging took {round(charge_time,1)} mins at {charger.charging_speed} kWh")
    
    def check_battery(self):
        if self.battery_level <= 0:
            raise DeadBatteryError(f"Driver {self.id} has a dead battery, towed to location {self.loc_current}")

    def run(self):
        try:
            yield self.env.process(self.drive())
        except DeadBatteryError as e:
                print(f"Error: {e}")
                self.data_collector.record_dead_battery(self.id, self.env.now)

class DataCollector:
    def __init__(self):
        self.charging_events = []
        self.dead_battery_events = []

    def record_charging_start(self, driver_id, time):
        self.charging_events.append((driver_id, time, 'start'))

    def record_charging_end(self, driver_id, time):
        self.charging_events.append((driver_id, time, 'end'))

    def record_dead_battery(self, driver_id, time):
        self.dead_battery_events.append((driver_id, time))

    def get_average_charging_time(self):
        charging_times = []
        charging_starts = {}
        for event in self.charging_events:
            driver_id, time, event_type = event
            if event_type == 'start':
                charging_starts[driver_id] = time
            elif event_type == 'end':
                start_time = charging_starts.pop(driver_id)
                charging_times.append(time - start_time)
        return sum(charging_times) / len(charging_times) if charging_times else 0
    
    def get_dead_batteries(self):
        return len(self.dead_battery_events)

class DeadBatteryError(Exception):
    "Raised when the battery level <= 0"
    pass

def setup_simulation(env, num_drivers, num_stations, num_chargers, loc_destination, avg_speed):

    # initialize data collector
    data_collector = DataCollector()
    
    # initialize charging stations and network
    stations = [
        ChargingStation(
            env,
            location=random.randint(0, loc_destination),
            chargers=[Charger(env, charging_speed=random.choice([20, 50, 150]), charger_type=f"Type{i%3+1}")
                      for i in range(num_chargers)]
        )
        for _ in range(num_stations)
    ]

    # list comprehension of station.location
    print("Stations at ",sorted([station.location for station in stations]))

    # [[print(f"Station at km {station.location} has 1 charger of type {charger.charger_type} with speed {charger.charging_speed}") for charger in station.chargers] for station in stations]
    
    charging_network = ChargingNetwork(stations)
    
    # initialize drivers and start their trips
    for i in range(num_drivers):
        driver = Driver(env, i+1, charging_network, avg_speed, loc_destination, data_collector)
        env.process(driver.run())

        print(f"Driver {driver.id} started trip")
    
    # return data collector when simulation is complete
    return data_collector


def run_simulation(params):
    env = simpy.Environment()
    
    data_collector = setup_simulation(env, params.num_drivers, params.num_stations, params.num_chargers, params.loc_destination, params.avg_speed)

    env.run(until=params.sim_duration) # or until all trips complete
    
    print(f"Average charging time: {data_collector.get_average_charging_time():.2f}")
    print(f"Dead batteries: {data_collector.get_dead_batteries():.0f}")


if __name__ == "__main__":
    params = SimulationParameters()
    run_simulation(params)