import simpy
import random

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

    def find_nearest_available_station(self, driver_location):
        # Simplified: just returns a random station
        return random.choice(self.stations)

    def request_charger(self, driver):
        station = self.find_nearest_available_station(driver.current_location)
        charger = station.get_available_charger()
        if charger is None:
            yield station.add_to_queue(driver)
            charger = yield station.queue.get()
        return charger, station

class Driver:
    def __init__(self, env, id, charging_network, data_collector):
        self.env = env
        self.id = id
        self.charging_network = charging_network
        self.data_collector = data_collector
        self.battery_level = 100
        self.current_location = (0, 0)

    def drive(self):
        drive_time = random.randint(60, 180)
        self.battery_level -= drive_time / 3  # Simplified battery consumption
        yield self.env.timeout(drive_time)

    def need_charge(self):
        return self.battery_level < 20

    def charge(self, charger):
        charge_amount = 100 - self.battery_level
        charge_time = charge_amount / charger.charging_speed
        self.data_collector.record_charging_start(self.id, self.env.now)
        yield self.env.timeout(charge_time)
        self.battery_level = 100
        self.data_collector.record_charging_end(self.id, self.env.now)

    def run(self):
        while True:
            yield self.env.process(self.drive())
            if self.need_charge():
                charger, station = yield self.env.process(self.charging_network.request_charger(self))
                with charger.request() as req:
                    yield req
                    yield self.env.process(self.charge(charger))
                station.queue.put(charger)

class DataCollector:
    def __init__(self):
        self.charging_events = []

    def record_charging_start(self, driver_id, time):
        self.charging_events.append((driver_id, time, 'start'))

    def record_charging_end(self, driver_id, time):
        self.charging_events.append((driver_id, time, 'end'))

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

def setup_simulation(env, num_drivers, num_stations, chargers_per_station):
    data_collector = DataCollector()
    
    stations = [
        ChargingStation(
            env,
            location=(random.randint(0, 100), random.randint(0, 100)),
            chargers=[Charger(env, charging_speed=random.choice([7, 50, 150]), charger_type=f"Type{i%3+1}")
                      for i in range(chargers_per_station)]
        )
        for _ in range(num_stations)
    ]
    
    charging_network = ChargingNetwork(stations)
    
    for i in range(num_drivers):
        driver = Driver(env, i, charging_network, data_collector)
        env.process(driver.run())
    
    return data_collector

def run_simulation(sim_time=1000, num_drivers=50, num_stations=5, chargers_per_station=3):
    env = simpy.Environment()
    data_collector = setup_simulation(env, num_drivers, num_stations, chargers_per_station)
    env.run(until=sim_time)
    
    print(f"Average charging time: {data_collector.get_average_charging_time():.2f}")

if __name__ == "__main__":
    run_simulation()