import json
import warnings

import numpy as np
import pandas as pd

from evocomp.algorithms.swarm.aco import AntColonyOptimization

max_delivery_radius = 2

hourly_salary = 70
walk_speed = 10
drive_speed = 50
fuel_cost_per_km = 0.4

total_cost = 0


def get_assigned_distance_matrix(assigned_parcels, parcels, distance):
    assigned_address_ids = parcels[parcels['parcel_id'].isin(assigned_parcels)]['address_id']
    assigned_address_ids = [distance.index[0]] + list(assigned_address_ids)
    assigned_distance_matrix = distance.iloc[assigned_address_ids, assigned_address_ids]
    return assigned_distance_matrix.to_numpy()


def heuristic_pedestrian(distance, walk_speed, hourly_salary):
    cost = (distance / walk_speed) * hourly_salary
    warnings.filterwarnings('ignore')
    cost = 1 / cost
    cost[cost == np.inf] = 0
    return cost


def heuristic_vehicle(distance, hourly_salary, fuel_cost_per_km):
    fuel_cost = distance * fuel_cost_per_km
    salary_cost = (distance / drive_speed) * hourly_salary
    total_cost = fuel_cost + salary_cost
    warnings.filterwarnings('ignore')
    total_cost = 1 / total_cost
    total_cost[total_cost == np.inf] = 0
    return total_cost


def vehicle_cost(dist, drive_speed, hourly_salary, fuel_cost_per_km):
    fuel_cost = dist * fuel_cost_per_km
    salary_cost = (dist / drive_speed) * hourly_salary
    return fuel_cost + salary_cost


def pedestrian_cost(dist, walk_speed, hourly_salary):
    return (dist / walk_speed) * hourly_salary


parcels = pd.read_csv('out/delivery/parcels.csv', index_col=0)
distance = pd.read_csv('out/delivery/distance.csv', index_col=0)
couriers = pd.read_csv('out/delivery/couriers.csv', index_col=0)

x_mid = (parcels['x'].max() + parcels['x'].min()) / 2
y_mid = (parcels['y'].max() + parcels['y'].min()) / 2

sectors = {
    'upper_right': parcels[(parcels['x'] > x_mid) & (parcels['y'] > y_mid)],
    'lower_right': parcels[(parcels['x'] > x_mid) & (parcels['y'] <= y_mid)],
    'lower_left': parcels[(parcels['x'] <= x_mid) & (parcels['y'] <= y_mid)],
    'upper_left': parcels[(parcels['x'] <= x_mid) & (parcels['y'] > y_mid)],
}

couriers_sectors = ['upper_right', 'lower_right', 'lower_left', 'upper_left'] * (len(couriers) // 4)
couriers_sectors += ['upper_right', 'lower_right', 'lower_left', 'upper_left'][: len(couriers) % 4]
couriers['sector'] = couriers_sectors[: len(couriers)]

couriers['assigned_parcels'] = [json.dumps([]) for _ in range(len(couriers))]

couriers = couriers.sort_values(by='type', ascending=True)


print('Total parcels scheduled for delivery:', len(parcels))
print('Total couriers available:', len(couriers))

for idx, courier in couriers.iterrows():
    courier_sector = courier['sector']
    parcels_in_sector = sectors[courier_sector].sort_values('weight', ascending=False).copy()

    if courier['type'] == 'pedestrian':
        parcels_in_sector = parcels_in_sector[
            parcels_in_sector['distance_to_hub'] <= max_delivery_radius
        ]

    courier_volume = 0
    courier_weight = 0
    assigned_parcels = json.loads(courier['assigned_parcels'])

    for parcel_idx, parcel in parcels_in_sector.iterrows():
        new_volume = courier_volume + parcel['volume']
        new_weight = courier_weight + parcel['weight']

        if new_weight <= courier['max_weight'] and new_volume <= courier['max_volume']:
            assigned_parcels.append(parcel['parcel_id'].item())

            courier_volume = new_volume
            courier_weight = new_weight

            sectors[courier_sector] = sectors[courier_sector].drop(parcel_idx)

    couriers.loc[idx, 'assigned_parcels'] = json.dumps(assigned_parcels)

    if len(assigned_parcels) > 0:
        print(
            f"Courier {courier['courier_id']} ({courier['type']}) assigned parcels: {list(map(int, assigned_parcels))}"
        )


couriers['assigned_parcels'] = couriers['assigned_parcels'].apply(json.loads)

couriers_active = couriers[couriers['assigned_parcels'].apply(len) > 0]
couriers_single_parcel = couriers_active[couriers_active['assigned_parcels'].apply(len) == 1]
couriers_multi_parcels = couriers_active[couriers_active['assigned_parcels'].apply(len) > 1]


for idx, courier in couriers_multi_parcels.iterrows():
    assigned_parcels = courier['assigned_parcels']
    courier_dist = get_assigned_distance_matrix(assigned_parcels, parcels, distance)

    if courier['type'] == 'vehicle':
        heuristic = heuristic_vehicle(courier_dist, hourly_salary, fuel_cost_per_km)
    else:
        heuristic = heuristic_pedestrian(courier_dist, walk_speed, hourly_salary)

    aco = AntColonyOptimization(n_ants=100, rho=0.03, alpha=1.6, beta=1.25)
    route, dist = aco.solve(courier_dist, heuristic)
    if courier['type'] == 'vehicle':
        cost = vehicle_cost(dist, drive_speed, hourly_salary, fuel_cost_per_km)
    else:
        cost = pedestrian_cost(dist, walk_speed, hourly_salary)

    print(
        f"Delivery for courier {courier['courier_id']} ({courier['type']}) costs: {round(cost, 2)}"
    )
    total_cost += cost


for idx, courier in couriers_single_parcel.iterrows():
    assigned_parcels = courier['assigned_parcels']
    courier_dist = get_assigned_distance_matrix(assigned_parcels, parcels, distance)

    dist = courier_dist[0, 1]

    if courier['type'] == 'vehicle':
        cost = vehicle_cost(dist, drive_speed, hourly_salary, fuel_cost_per_km)
    else:
        cost = pedestrian_cost(dist, walk_speed, hourly_salary)

    print(
        f"Delivery for courier {courier['courier_id']} ({courier['type']}) costs: {round(cost, 2)}"
    )
    total_cost += cost


print('Total Delivery cost:', round(total_cost, 2))
