#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

total_parcels = 100
min_weight, max_weight = 1, 60
min_volume, max_volume = 1, 150

address_ids = np.arange(1, total_parcels + 1)
np.random.shuffle(address_ids)


total_heavy_parcels = total_parcels // 100 * 30
total_light_parcels = total_parcels - total_heavy_parcels
heavy_parcels = np.random.randint(min_weight + 9, max_weight, size=total_heavy_parcels)
light_parcels = np.random.randint(min_weight, 10, size=total_light_parcels)
weight = np.concatenate((heavy_parcels, light_parcels))


total_large_parcels = total_parcels // 100 * 30
total_small_parcels = total_parcels - total_large_parcels
large_parcels = np.random.randint(min_volume + 4, max_volume, size=total_large_parcels)
small_parcels = np.random.randint(min_volume, 5, size=total_small_parcels)
volume = np.concatenate((large_parcels, small_parcels))


parcels = pd.DataFrame(
    {
        'parcel_id': np.arange(total_parcels),
        'address_id': address_ids,
        'weight': weight,
        'volume': volume,
    }
)


total_addresses = total_parcels + 1

upper_triangle = np.triu(np.random.randint(1, 30, size=(total_addresses, total_addresses)), 1)

lower_triangle = upper_triangle.transpose()

distance_matrix = upper_triangle + lower_triangle


distances = pd.DataFrame(distance_matrix)


total_vehicles = 8
total_pedestrian_couriers = 4
max_pedestrian_weight = 20
max_vehicle_weight = 200
max_pedestrian_volume = 30
max_vehicle_volume = 400


couriers = pd.DataFrame(
    {
        'courier_id': np.arange(total_vehicles + total_pedestrian_couriers),
        'max_weight': np.concatenate(
            (
                np.full(total_vehicles, max_vehicle_weight),
                np.full(total_pedestrian_couriers, max_pedestrian_weight),
            )
        ),
    }
)


def assign_max_volume(row):
    if row['max_weight'] == max_pedestrian_weight:
        return max_pedestrian_volume
    else:
        return max_vehicle_volume


def assign_type(row):
    if row['max_weight'] == max_pedestrian_weight:
        return 'pedestrian'
    else:
        return 'vehicle'


couriers['max_volume'] = couriers.apply(assign_max_volume, axis=1)


couriers['type'] = couriers.apply(assign_type, axis=1)


hub_coords = np.array([[0, 0]])


parcel_coords = 30 * (np.random.rand(total_parcels, 2) - 0.5)
all_coords = np.concatenate([hub_coords, parcel_coords])


distance = pd.DataFrame(cdist(all_coords, all_coords, 'euclidean'))


parcels['address_id'] = np.arange(1, total_parcels + 1)


parcels['x'] = parcel_coords[:, 0]
parcels['y'] = parcel_coords[:, 1]


parcels['distance_to_hub'] = distance.loc[0, 1:].reset_index(drop=True)


parcels.to_csv('out/delivery/parcels.csv')
distance.to_csv('out/delivery/distance.csv')
couriers.to_csv('out/delivery/couriers.csv')
