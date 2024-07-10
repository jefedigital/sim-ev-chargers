import requests
from dotenv import dotenv_values

secrets = dotenv_values('.env')

def get_route_nycbos():
    api_key = secrets['api_google_directions']
    origin = 'New York, NY'
    destination = 'Boston, MA'
    url = f'https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&key={api_key}'
    response = requests.get(url)
    return response.json()

# Corrected polyline decoding function
def decode_polyline(polyline_str):
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {'latitude': 0, 'longitude': 0}

    while index < len(polyline_str):
        for key in changes.keys():
            shift, result = 0, 0

            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if byte < 0x20:
                    break

            if (result & 1):
                changes[key] = ~(result >> 1)
            else:
                changes[key] = (result >> 1)

            if key == 'latitude':
                lat += changes['latitude']
            else:
                lng += changes['longitude']

        coordinates.append((lat / 1e5, lng / 1e5))

    return coordinates

# Get the route and decode the polyline
route = get_route_nycbos()
points = decode_polyline(route['routes'][0]['overview_polyline']['points'])

