from telnetlib import STATUS
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from holidays import Switzerland
import json
import urllib.request
import osmnx as ox
from math import atan, cos, radians, sin, tan, asin, sqrt

def get_litter_columns(df):
    litter_columns = []
    for item in df.columns.to_list():
        if item.isdigit(): litter_columns.append(item) 
    return litter_columns

def clean_df(df):
    not_needed_columns = ['Unnamed: 0', 'value.Vehicle_Mode', 'speed', '_id',]
    df.drop(not_needed_columns, axis=1, inplace=True, errors='ignore')
    df.rename(columns = {'suitcase.id':'suitcase_id', 'date.utc':'date_utc', 
                        'edge.id':'edge_id', 'edge.osmid':'edge_osmid', 
                        'place.id':'place_id', 'osm.highway':'osm_highway'}, inplace=True, errors='ignore')
    litter_columns = get_litter_columns(df)
    df = df.dropna(subset=['edge_id']).copy()
    df.drop('place_id', axis=1, inplace=True)
    df['date_utc'] = pd.to_datetime(df['date_utc']).dt.date
    df[litter_columns] = df[litter_columns].fillna(0)
    df[litter_columns] = df[litter_columns].astype(np.int64) 
    df['total_litter'] = df[litter_columns].sum(axis=1)
    df['edge_osmid'] = df['edge_osmid'].astype(int)
    return df

def aggregate_df(df, method='sum'):
    aggregation_type = method
    litter_columns = get_litter_columns(df)
    to_agg = {'edge_osmid' : 'first', 'osm_highway' : 'first', 'total_litter' : aggregation_type}
    for litter in litter_columns:
        to_agg[litter] = aggregation_type
    df = df.groupby(['date_utc', 'edge_id'], as_index=False).agg(to_agg)
    return df

def make_date_features(df):
    df['Year'] = pd.DatetimeIndex(df['date_utc']).year.astype(object)
    df['month'] = pd.DatetimeIndex(df['date_utc']).month.astype(object)
    df['day'] = pd.DatetimeIndex(df['date_utc']).day.astype(object)
    weekdays = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
    df['weekday'] = pd.to_datetime(df['date_utc']).dt.date.apply(lambda x: x.weekday())
    df['weekday'] = df['weekday'].apply(lambda x: weekdays[x])
    holiday = [holiday for holiday in Switzerland(years=[2021, 2022]).items()]
    for day in set(holiday):
        holiday.append(((day[0] + timedelta(days=1)), day[1]))
        holiday.append(((day[0] + timedelta(days=2)), day[1]))

    holidays_df = pd.DataFrame(holiday, columns=["date", "holiday"])
    holidays_df['holiday'] = holidays_df['holiday'].astype(str)
    df['holiday'] = df['date_utc'].apply(lambda x: 1 if x in holidays_df['date'].values else 0)
    return df

def make_coordinates_features(df):
    with urllib.request.urlopen('https://raw.githubusercontent.com/dominik117/data-science-toolkit/main/data/edges.geojson') as url:
        data = json.loads(url.read().decode())
    df_edges = pd.DataFrame(data['features'])  # <-- The only column needed
    df_edges = pd.concat([df_edges.drop(['properties'], axis=1), df_edges['properties'].apply(pd.Series)], axis=1)  # <--Explode properties dictionary inside cells
    df_edges.rename(columns = {'id':'edge_id'}, inplace = True)
    # Rearrange BBOX (lat_north, lat_south, lon_east, lon_west)
    def sort_bbox(x):
        lat = sorted([x[1], x[3]], key=float, reverse=True)
        lon = sorted([x[0], x[2]], key=float, reverse=True)
        return lat + lon
    df_edges['bbox'] = df_edges['bbox'].apply(sort_bbox)
    df_coordinates = df_edges[['edge_id', 'bbox']].copy()
    bbox_exploded = pd.DataFrame(df_coordinates["bbox"].to_list(), columns=['lat_north', 'lat_south', 'lon_east', 'lon_west'])
    df_coordinates = pd.concat([df_coordinates, bbox_exploded], axis=1)
    df_coordinates.drop(['bbox'], axis=1, inplace=True)
    df = pd.merge(df, df_coordinates, how="left", on="edge_id")
    return df


def make_edge_length_feature(df):
    def haversine_distance(lat1, lon1, lat2, lon2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        EQUATORIAL_RADIUS = 6378 # Radius of earth in kilometers
        return c * EQUATORIAL_RADIUS
    def lamberts_ellipsoidal_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        AXIS_A = 6378137.0
        AXIS_B = 6356752.314245
        EQUATORIAL_RADIUS = 6378137
        flattening = (AXIS_A - AXIS_B) / AXIS_A
        b_lat1 = atan((1 - flattening) * tan(radians(lat1)))
        b_lat2 = atan((1 - flattening) * tan(radians(lat2)))
        sigma = haversine_distance(lat1, lon1, lat2, lon2) / EQUATORIAL_RADIUS
        P_value = (b_lat1 + b_lat2) / 2
        Q_value = (b_lat2 - b_lat1) / 2
        X_numerator = (sin(P_value) ** 2) * (cos(Q_value) ** 2)
        X_demonimator = cos(sigma / 2) ** 2
        X_value = (sigma - sin(sigma)) * (X_numerator / X_demonimator)
        Y_numerator = (cos(P_value) ** 2) * (sin(Q_value) ** 2)
        Y_denominator = sin(sigma / 2) ** 2
        Y_value = (sigma + sin(sigma)) * (Y_numerator / Y_denominator)
        distance = abs(EQUATORIAL_RADIUS * (sigma - ((flattening / 2) * (X_value + Y_value))))
        return int(distance)
    df['edge_length'] = df.apply(lambda x: lamberts_ellipsoidal_distance(x.lat_north, x.lon_east, x.lat_south, x.lon_west), axis=1)
    return df

def make_weather_features(df):
    df_weather = pd.read_csv("https://raw.githubusercontent.com/dominik117/data-science-toolkit/main/data/weather_basel_2021-2022.csv")
    # Data obtained from https://www.meteoblue.com/en/weather/archive/export/basel_switzerland_2661604
    df['date_utc'] = pd.to_datetime(df['date_utc']).dt.date
    weather_columns = {'location':'date_utc', 'Basel':'temperature_max', 'Basel.1':'temperature_min', 'Basel.2':'temperature_mean', 'Basel.3':'precipitation',
                    'Basel.4':'snowfall', 'Basel.5':'humidity_max', 'Basel.6':'humidity_min', 'Basel.7':'humidity_mean', 'Basel.8':'cloud_coverage',
                    'Basel.9':'wind_speed_max', 'Basel.10':'wind_speed_min', 'Basel.11':'wind_speed_mean'}
    df_weather.rename(columns = weather_columns, inplace = True)
    df_weather.drop(['Basel.12'], axis=1, inplace=True)
    df_weather = df_weather.iloc[9:].copy()  # <-- Rows with metadata
    df_weather['date_utc'] = pd.to_datetime(df_weather['date_utc']).dt.date  # <-- Match main df type and name
    weather_date = df_weather.pop('date_utc')  # <-- Pop date so it doesn't get converted
    df_weather = df_weather.apply(pd.to_numeric)
    df_weather = df_weather.round(decimals = 1)
    df_weather.insert(0, 'date_utc', weather_date) # <-- Reassign unmodified date to df
    df = pd.merge(df, df_weather, how="left", on="date_utc")
    return df

def make_osm_features(df):
    # Extraction of the features on the dictionary from OSM
    basel = 'Basel, Basel, Switzerland'
    # Features dictionary at https://wiki.openstreetmap.org/wiki/Map_features

    tags = {'amenity': ['vending_machine', 'bench', 'bar', 'fast_food', 'ice_cream', 'kindergarten', 'school', 'hospital', 'cinema', 
                        'fountain', 'dog_toilet', 'recycling', 'waste_basket', 'waste_disposal', 'childcare', 'marketplace',
                        'bus_station', 'fuel', 'taxi', 'parking', 'atm', 'clinic', 'nightclub', 'toilets'
                        ]}
    amenity = ox.geometries_from_place(basel, tags=tags)
    # tags = {'leisure': 'park'}
    # leisure = ox.geometries_from_place(basel, tags=tags)
    df_osm = pd.DataFrame(amenity)  # <-- Convert to DF
    df_osm = df_osm[['amenity', 'geometry']].copy()  # <-- Select the only needed columns
    df_osm['osm_id'] = df_osm.index.to_numpy()  # <-- Detach the index and assign it to a normal column
    df_osm.reset_index(drop=True, inplace=True)  # <-- Drop index
    osm_id_exploded = pd.DataFrame(df_osm["osm_id"].to_list(), columns=['type', 'osm'])  # <-- Explode index since it contains two indices
    df_osm = pd.concat([df_osm, osm_id_exploded], axis=1) 
    df_osm.drop(['osm_id', 'osm'], axis=1, inplace=True)
    df_osm = df_osm[df_osm['type'] == 'node']  # <-- Drop Multipoligon points
    #### Clean the coordinates from the GeoPandas geometry format to latitude and longitude columns
    df_osm['lon'] = df_osm[df_osm['type'] == "node"]['geometry'].apply(lambda p: p.x)
    df_osm['lat'] = df_osm[df_osm['type'] == "node"]['geometry'].apply(lambda p: p.y)
    df_osm.drop(['geometry', 'type'], axis=1, inplace=True)
    df_edges_coordinates = df[['edge_id', 'lat_north', 'lat_south', 'lon_east', 'lon_west']].copy()
    df_edges_coordinates = df_edges_coordinates.drop_duplicates(subset='edge_id', keep='first')

    # Make a list of the edges that have an amenity to them based on lat,lon conditional
    def is_between(a, x, b):
        return min(a, b) < x < max(a, b)
    edges_dict = []
    for edges_row in df_edges_coordinates.itertuples():
        for osm_row in df_osm.itertuples():
            if is_between(edges_row.lat_south, osm_row.lat, edges_row.lat_north) and is_between(edges_row.lon_west, osm_row.lon, edges_row.lon_east):
                edges_dict.append([edges_row.edge_id, osm_row.amenity])

    # Group by edge_id and get the value counts per amenity
    df_edges_dict = pd.DataFrame(edges_dict, columns = ['edge_id', 'amenity'])
    df_edges_dict = df_edges_dict.groupby('edge_id')['amenity'].value_counts().unstack(fill_value=0).reset_index()
    df = pd.merge(df, df_edges_dict, how="left", on="edge_id")
    osm_columns = list(df_edges_dict.columns[1:])
    df[osm_columns] = df[osm_columns].fillna(value=0)  # <-- Fill missing values, since not all edges have amenities
    df[osm_columns] = df[osm_columns].astype(int)
    return df, osm_columns
