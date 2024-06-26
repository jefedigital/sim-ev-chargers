import simpy
import random

class Charger:
    def __init__(self, id, charging_speed, level):
        self.id = id
        self.charging_speed = charging_speed
        self.level = level
        self.in_use = False

class ChargingStation(simpy.Resource):
    def __init__(self, env, num_chargers):
        super().__init__(env, capacity=num_chargers)
        self.chargers = [
            Charger(1, 50, 3),    # DC Fast charger (Level 3)
            Charger(2, 7.7, 2),   # Level 2 charger
            Charger(3, 7.7, 2)    # Level 2 charger
        ]
        self.env = env
        self.charger_queue = simpy.Store(env)

    def request(self):
        return self.charger_queue.get()

    def release(self, charger):
        charger.in_use = False
        return self.charger_queue.put(charger)

    def charge_ev(self, ev):
        """Simulate charging an EV"""
        with self.request() as req:
            charger = yield req
            charger.in_use = True
            print(f"EV {ev.id} starting to charge on Level {charger.level} charger {charger.id} at time {self.env.now}")

            # Simulate charging
            charging_time = ev.battery_capacity / charger.charging_speed
            yield self.env.timeout(charging_time)

            yield self.release(charger)
            print(f"EV {ev.id} finished charging on Level {charger.level} charger {charger.id} at time {self.env.now}")

    def run(self):
        while True:
            available_chargers = [c for c in self.chargers if not c.in_use]
            if available_chargers:
                # Sort chargers by level in descending order
                best_charger = max(available_chargers, key=lambda c: c.level)
                yield self.charger_queue.put(best_charger)
            yield self.env.timeout(1)  # Check every time unit

class EV:
    def __init__(self, id, battery_capacity):
        self.id = id
        self.battery_capacity = battery_capacity

def ev_arrival(env, ev, charging_station):
    print(f"EV {ev.id} arrived at the charging station at time {env.now}")
    yield env.process(charging_station.charge_ev(ev))
    print(f"EV {ev.id} left the charging station at time {env.now}")

# Setup and run the simulation
env = simpy.Environment()
charging_station = ChargingStation(env, num_chargers=3)
env.process(charging_station.run())

# Create some EVs
evs = [EV(i, random.uniform(60, 100)) for i in range(5)]

# Schedule EV arrivals
for ev in evs:
    env.process(ev_arrival(env, ev, charging_station))

# Run the simulation
env.run(until=200)