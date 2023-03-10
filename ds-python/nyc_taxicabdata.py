import pandas as pd 
import datetime
from geopy.distance import geodesic
import tqdm
import pprint
import pymongo

'''
https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data?select=train.zip

The competition dataset is based on the 2016 NYC Yellow Cab trip record data made 
available in Big Query on Google Cloud Platform. The data was originally published 
by the NYC Taxi and Limousine Commission (TLC). The data was sampled and cleaned 
for the purposes of this playground competition. Based on individual trip attributes, 
participants should predict the duration of each trip in the test set.

attributes
'id', 'vendor_id', 'pickup_datetime', 'dropoff_datetime',
       'passenger_count', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',
       'trip_duration'
'''


attribute_index = {
    'VendorID' : 0,
    'tpep_pickup_datetime' : 1,
    'tpep_dropoff_datetime' : 2,
    'passenger_count' : 3,
    'trip_distance' : 4,
    'RatecodeID' : 5,
    'store_and_fwd_flag' : 6,
    'PULocationID' : 7,
    'DOLocationID' : 8,
    'payment_type' : 9,
    'fare_amount' : 10,
    'extra' : 11,
    'mta_tax' : 12,
    'tip_amount' : 13,
    'tolls_amount' : 14,
    'improvement_surcharge' : 15,
    'total_amount' : 16,
    'congestion_surcharge' : 17,
    'airport_fee' : 18,
    'PULong' : 19,
    'PULat' : 20,
    'DOLong' : 21,
    'DOLat' : 22,
    'ID' : 23
}

def load_data():
    # file_loc = '/localdisk3/nyc_2018.csv'
    file_loc = '/localdisk3/nyc_yellowtaxidata_2021-09.csv'
    # df = pd.read_csv(file_loc, nrows=1000)
    # print(df.head())
    # print(df.keys())
    f = open(file_loc, 'r')
    lines = f.readlines()
    data = [line.strip() for line in lines]
    f.close()
    attributes = data[0]
    del data[0]
    data = [d.split(',') for d in data]
    print(data[1])
    # print(attributes)
    lat_long_file = open('/localdisk3/nyc_tlc_latlong.csv', 'r')
    lat_long_lines = lat_long_file.readlines()
    lat_long_data = [line.strip() for line in lat_long_lines]
    lat_long_file.close()
    LONG_INDEX = 0
    LAT_INDEX = 1
    LOC_ID = 3
    del lat_long_data[0]
    lat_long_data = [d.split(',') for d in lat_long_data]
    lat_long_tuples = {}
    for ll_d in lat_long_data:
        long_lat = (ll_d[LONG_INDEX], ll_d[LAT_INDEX])
        lat_long_tuples[ll_d[LOC_ID]] = long_lat

    # print(lat_long_tuples[data[1][7]])
    updated_data = []
    for i,d in enumerate(data):
        PU_ID = d[7]
        DO_ID = d[8]
        if PU_ID in lat_long_tuples.keys() and DO_ID in lat_long_tuples.keys():
            PU_long_lat = lat_long_tuples[PU_ID]
            DO_long_lat = lat_long_tuples[DO_ID]
            lat_long = [PU_long_lat[0], PU_long_lat[1], DO_long_lat[0], DO_long_lat[1]]
            d = d + lat_long + [i]
            updated_data.append(d)
    
    # print(updated_data[1])
    print(len(data))
    print(len(updated_data))
    with open('/localdisk3/nyc_2021-09_updated.csv', 'w') as out:
        for d in updated_data:
            out.write(','.join(str(i) for i in d))
            out.write('\n')
    out.close()


def load_small_data():
    file_loc = '/localdisk3/nyc_2018_small.csv'

    f = open(file_loc, 'r')
    lines = f.readlines()
    data = [line.strip() for line in lines]
    f.close()
    del data[0]
    data = [d.split(',') for d in data]
    return data



def load_data_from_disk():
    # file_loc = '/localdisk3/nyc_2018_updated.csv'
    file_loc = '/localdisk3/nyc_2021-09_updated.csv'
    f = open(file_loc, 'r')
    lines = f.readlines()
    data = [line.strip() for line in lines]
    f.close()
    data = [d.split(',') for d in data]
    return data



def datetime_format():
    data = load_data_from_disk()
    # format = "%m/%d/%Y %I:%M:%S %p"
    format = "%Y-%m-%d %H:%M:%S"
    # print(data[0][1:3])
    # pickupdatetime = data[0][1]
    # dropoffdatetime = data[0][2]
    # pickupmydate = datetime.datetime.strptime(pickupdatetime, format)
    # dropoffmydate = datetime.datetime.strptime(dropoffdatetime, format)
    # delta = dropoffmydate - pickupmydate
    # print(pickupmydate)
    # print(dropoffmydate)
    # print(delta.total_seconds())
    datetime_data = [] 
    # each element is a list of form: [ID, pickup_datetime, dropoff_datetime, timediff(seconds)]

    for d in data:
        ID = d[23]
        pickupdatetime = d[1]
        dropoffdatetime = d[2]
        pickupmydate = datetime.datetime.strptime(pickupdatetime, format)
        dropoffmydate = datetime.datetime.strptime(dropoffdatetime, format)
        delta = dropoffmydate - pickupmydate
        time_diff = delta.total_seconds()
        datetime_data.append([ID, pickupmydate, dropoffmydate, time_diff])
    return datetime_data


def datetime_index(data, PUtime_diff, DOtime_diff):
    result = {}
    intervals_progress = tqdm.tqdm(total=len(data), position=0)
    for d1 in data:
        d1_ID =  d1[0]
        d1_pickup_datetime = d1[1]
        d1_dropoff_datetime = d1[2]
        if d1_ID not in result.keys():
            result[d1_ID] = set()
        
        result[d1_ID].add(d1_ID)
        for d2 in data:
            d2_ID = d2[0]
            if d2_ID not in result[d1_ID]:
                d2_pickup_datetime = d2[1]
                d2_dropoff_datetime = d2[2]
                pickup_diff = d1_pickup_datetime - d2_pickup_datetime
                dropoff_diff = d1_dropoff_datetime - d2_dropoff_datetime
                pickup_delta = abs(pickup_diff.total_seconds())
                dropoff_delta = abs(dropoff_diff.total_seconds())
                if pickup_delta <= PUtime_diff and dropoff_delta <= DOtime_diff:
                    result[d1_ID].add(d2_ID)
                    if d2_ID not in result.keys():
                        result[d2_ID] = set()
                    result[d2_ID].add(d1_ID)
        intervals_progress.update(1)

    # with open('/localdisk3/nyc_2018_time_sim_PU_{0}_DO_{1}.txt'.format(PUtime_diff, DOtime_diff), 'w') as f:
    with open('/localdisk3/nyc_2021-09_time_sim_PU_{0}_DO_{1}.txt'.format(PUtime_diff, DOtime_diff), 'w') as f:
        for key, value in result.items():
            f.write(str(key) + ' : ' + str(value) + '\n')
    
    f.close()

def location_index(PU_dist_threshold, DO_dist_threshold):
    data = load_data_from_disk()
    # data = load_small_data()
    result = {}
    intervals_progress = tqdm.tqdm(total=len(data), position=0)
    for d1 in data:
        d1_PU_lat = float(d1[20])
        d1_PU_long = float(d1[19])
        d1_DO_lat = float(d1[22])
        d1_DO_long = float(d1[21])
        d1_PU = (d1_PU_lat, d1_PU_long)
        d1_DO = (d1_DO_lat, d1_DO_long)
        d1_ID = d1[23]
        if d1_ID not in result.keys():
            result[d1_ID] = set()
        
        result[d1_ID].add(d1_ID)
        for d2 in data:
            d2_ID = d2[23]
            if d2_ID not in result[d1_ID]:
                d2_PU_lat = float(d2[20])
                d2_PU_long = float(d2[19])
                d2_DO_lat = float(d2[22])
                d2_DO_long = float(d2[21])

                d2_PU = (d2_PU_lat, d2_PU_long)
                d2_DO = (d2_DO_lat, d2_DO_long)

                PU_distance = geodesic(d1_PU, d2_PU).miles
                DO_distance = geodesic(d1_DO, d2_DO).miles

                if PU_distance <= PU_dist_threshold and DO_distance <= DO_dist_threshold:
                    result[d1_ID].add(d2_ID)
                    if d2_ID not in result.keys():
                        result[d2_ID] = set()
                    
                    result[d2_ID].add(d1_ID)
        intervals_progress.update(1)

    with open('/localdisk3/nyc_2021-09_dist_sim_PU_{0}_DO_{1}.txt'.format(PU_dist_threshold, DO_dist_threshold), 'w') as f:
    # with open('/localdisk3/nyc_2018_dist_sim_PU_{0}_DO_{1}_small.txt'.format(PU_dist_threshold, DO_dist_threshold), 'w') as f:
        for key, value in result.items():
            f.write(str(key) + ' : ' + str(value) + '\n')
    
    f.close()

def convert_to_geojson(pickup=True):
    data = load_data_from_disk()
    geo_data = []
    for d in data:
        d_ID = d[23]
        d_PU_lat = float(d[20])
        d_PU_long = float(d[19])
        d_DO_lat = float(d[22])
        d_DO_long = float(d[21])
        if pickup:
            temp = {"location" : {'type' : 'Point', 'coordinates' : [d_PU_long, d_PU_lat]}, "name" : d_ID}
        else:
            temp = {"location" : {'type' : 'Point', 'coordinates' : [d_DO_long, d_DO_lat]}, "name" : d_ID}
        geo_data.append(temp)
    
    return geo_data

def mongo_geoindex(PU_dist_threshold, DO_dist_threshold):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["geodatabase"]
    print(myclient.list_database_names())
    
    places_pickup = mydb["places_pickup"]
    places_dropoff = mydb["places_dropoff"]
    
    mydb.places_dropoff.drop()
    mydb.places_pickup.drop()
    
    mydb.places_pickup.create_index([( "location", pymongo.GEOSPHERE )])
    data_pickup = convert_to_geojson()
    mydb.places_pickup.insert_many(data_pickup)

    mydb.places_dropoff.create_index([( "location", pymongo.GEOSPHERE )])
    data_dropoff = convert_to_geojson(pickup=False)
    mydb.places_dropoff.insert_many(data_dropoff)

    result = {}
    progress_bar = tqdm.tqdm(total=len(data_pickup), position=0)
    for d_PU, d_DO in zip(data_pickup, data_dropoff):
        pickup_coordinates = d_PU['location']['coordinates']
        dropoff_coordinates = d_DO['location']['coordinates']
        data_id = d_PU['name']
        check_id = d_DO['name']
        assert data_id == check_id
        temp_pu_results = set()
        temp_do_results = set()
        for doc in mydb.places_pickup.find({"location" : {
            "$nearSphere" : {
                "$geometry" : {
                    "type" : "Point",
                    "coordinates" : pickup_coordinates
                },
                "$maxDistance" : PU_dist_threshold}}}):
            # pprint.pprint(doc)
            temp_pu_results.add(doc['name'])
        
        for doc in mydb.places_dropoff.find({"location" : {
            "$nearSphere" : {
                "$geometry" : {
                    "type" : "Point",
                    "coordinates" : dropoff_coordinates
                },
                "$maxDistance" : DO_dist_threshold}}}):
            # pprint.pprint(doc)
            temp_do_results.add(doc['name'])

        result[data_id] = temp_pu_results.intersection(temp_do_results)
        progress_bar.update(1)

    with open('/localdisk3/nyc_2021-09_dist_sim_PU_{0}_DO_{1}.txt'.format(PU_dist_threshold, DO_dist_threshold), 'w') as f:
        for key, value in result.items():
            f.write(str(key) + ' : ' + str(value) + '\n')
    
    f.close()

def parse_geojson():
    geo_data = convert_to_geojson()
    for d in geo_data:
        print(d['location']['coordinates'])
        break

def load_parquet_data():
    file_loc = '/localdisk3/yellow_tripdata_2021-09.parquet'
    df = pd.read_parquet(file_loc, engine='pyarrow')
    # print(df.head())
    # print(df.size)
    df_sampled = df.sample(70000)
    print(df_sampled.size)
    print(df_sampled.keys())
    df_sampled.to_csv('/localdisk3/nyc_yellowtaxidata_2021-09.csv', encoding='utf-8', index=False)

if __name__ == '__main__':
    # load_parquet_data()
    # load_data()
    # data = datetime_format()
    # datetime_index(data, 300, 420)
    # location_index(1, 1)
    mongo_geoindex(1,1)
    # parse_geojson()
    