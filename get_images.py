import requests
from tqdm import tqdm
import os
import json
from random import randint
import argparse
from csv import writer
# Consider using https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.utils_geo.sample_points for street gps coords
# Stack Overflow on how to get the coordinates: https://stackoverflow.com/questions/68367074/how-to-generate-random-lat-long-points-within-geographical-boundaries

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", help="The folder full of addresses per city to read and extract GPS coordinates from", required=True, type=str)
    parser.add_argument("--output", help="The output folder where the images will be stored, (defaults to: images/)", default='images/', type=str)
    parser.add_argument("--icount", help="The amount of images to pull (defaults to 25,000)", default=25000, type=int)
    parser.add_argument("--key", help="Your Google Street View API Key", type=str, required=True)
    return parser.parse_args()

args = get_args()
url = 'https://maps.googleapis.com/maps/api/streetview'
cities = []

def load_cities():
    for city in os.listdir(args.cities):
        with open(os.path.join(args.cities, city)) as f:
            coordinates = []
            print(f'Loading {city} addresses...')
            for line in tqdm(f):
                data = json.loads(line)
                coordinates.append(data['geometry']['coordinates'])
            cities.append(coordinates)

def main():
    # Open and create all the necessary files & folders
    os.makedirs(args.output, exist_ok=True)
    
    load_cities()
    
    coord_output_file = open(os.path.join(args.output, 'picture_coords.csv'), 'w', newline='')
    csv_writer = writer(coord_output_file)
    
    for i in tqdm(range(args.icount)):
        cities_count = []
        cities_count = [0] * len(cities)
        city_index = randint(0, len(cities) - 1)
        city = cities[city_index]
        cities_count[city_index] += 1
        addressLoc = city[randint(0, len(city) - 1)]
        city.remove(addressLoc) # Remove the address from the list so we don't get the same one twice
        # Set the parameters for the API call to Google Street View
        params = {
            'key': args.key,
            'size': '640x640',
            'location': str(addressLoc[1]) + ',' + str(addressLoc[0]),
            'heading': str((randint(0, 3) * 90) + randint(-15, 15)),
            'pitch': '20',
            'fov': '90'
            }
        
        response = requests.get(url, params)
        
        # Save the image to the output folder
        with open(os.path.join(args.output, f'street_view_{i}.jpg'), "wb") as file:
            file.write(response.content)
        
        # Save the coordinates to the output file
        csv_writer.writerow([addressLoc[1], addressLoc[0]])

    coord_output_file.close()
    
    for i in range(len(cities_count)):
        city_count = cities_count[i]
        city_name = os.listdir(args.cities)[i]
        print(f'{city_count} images pulled from {city_name}')

if __name__ == '__main__':
    main()
