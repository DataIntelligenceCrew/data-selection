import numpy as np 
import psycopg2
import faiss 
# from algorithms import *
from paths import *
from utils_algo import * 
import tqdm
import time 
import multiprocessing
import argparse

def posting_list_to_db():
    conn = None
    posting_list_file = POSTING_LIST_LOC.format('cifar10', 0.9, 'resnet-18')
    print(posting_list_file)
    f2 = open(posting_list_file, 'r')
    lines = f2.readlines()
    data = [line.strip().replace('{', '').replace('}', '') for line in lines]
    f2.close()
    result = {}
    p_bar = tqdm.tqdm(total=len(data), position=0)
    for d in data:
        pl = d.split(':')
        key = int(pl[0])
        value = pl[1].split(',')
        value = [int(v.replace("{", "").replace("}", "").replace("'", '').strip()) for v in value]
        result[key] = list(value)
        p_bar.update(1)


    try:
        conn = psycopg2.connect(database="pmundra")
        cur = conn.cursor()
        print('database connection established')
        progress_bar = tqdm.tqdm(total=len(result), position=0)
        for key, value in result.items():
            cur.execute("INSERT INTO cifar10_postinglist(id, pl) VALUES(%s, %s);", (key, value, ))
            conn.commit()
            progress_bar.update(1)
        
        
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def gfkc_db(cf, dr, dataset_name, dataset_size, num_classes):
    '''
        set cover using SQL queries 
        TABLES:
            cifar10_postinglist(id INT, pl INT[])
            cifar10_coverage_tracker(id INT, cf INT) -- need to be populated on the fly & drop all values after run
            cifar10_fairness_tracker(group_id INT, dr INT) -- need to be populated on the fly & drop all values after run
    '''
    start_time = time.time()
    coreset = set()
    # create the labels_dict = {}
    label_file = open(LABELS_FILE_LOC.format(dataset_name), 'r')
    labels = label_file.readlines()
    labels_dict = dict()    
    for l in labels:
        txt = l.split(':')
        arr = np.zeros(num_classes)
        arr[int(txt[1].strip())] = 1
        labels_dict[int(txt[0].strip())] = arr
    label_file.close()

    conn = psycopg2.connect(database="pmundra")
    print('Database connection established')
    cur = conn.cursor()
    cur.execute("select * from cifar10_postinglist where ARRAY_LENGTH(pl, 1) < %s;", (cf, ))
    records = cur.fetchall()
    for rows in records:
        coreset.add(rows[0])
        coreset = coreset.union(set(rows[1]))
    
    # # populate the coverage tracker table
    # for i in range(dataset_size):
    #     if i not in coreset:
    #         cur.execute("INSERT INTO cifar10_coverage_tracker(id, cf) VALUES(%s, %s);", (i, cf, ))
    #         conn.commit()
    
    # # populate the fairness tracker table
    # for i in range(num_classes):
    #     cur.execute("INSERT INTO cifar10_fairness_tracker(group_id, dr) VALUES(%s, %s);", (i, dr[i], ))
    #     conn.commit()

    
    # cur.execute("select count(*) from cifar10_coverage_tracker where cf > 0;")

    delta = set([i for i in range(dataset_size)])
    CC = np.zeros((1, dataset_size))
    CC[0][list(delta)] = cf 
    CC[0][list(coreset)] = 0
    GC = np.zeros((1, num_classes))
    GC[0] = np.array(dr)
    while ((not np.all(CC[0] <= 0)) or (not np.all(GC[0]))) and len(coreset) < dataset_size:
        best_point, max_score = -1, float("-inf")
        possible_ids = tuple(delta.difference(coreset))

        posting_list = np.zeros((len(possible_ids), dataset_size))
        index_to_p_id = {}
        cur.execute('select id, pl from cifar10_postinglist where id in %s', (possible_ids, ))
        records = cur.fetchall()
        # print('db query time: {0}'.format(time.time() - psql_start_time))
        for index, rows in enumerate(records):
            posting_list[index][list(rows[1])] = 1
            index_to_p_id[index] = rows[0]

        p_scores = np.matmul(CC, posting_list.T)
        p_scores[0] = np.array([j + np.dot(GC[0].T, labels_dict[index_to_p_id[i]]) for i, j in enumerate(p_scores[0])])
        max_score_index = p_scores[0].argmax()
        p_score = p_scores[0][max_score_index] 
        if p_score > max_score:
            max_score = p_score
            best_point = index_to_p_id[max_score_index]

        # # parallel
        # batch_size = int(0.1 * len(possible_ids))
        # partition_points = [tuple(possible_ids[i : i+batch_size]) for i in range(0, len(possible_ids), batch_size)]
        # parallel_solutions = []
        # q = multiprocessing.Queue()
        # processes = []
        # for i in range(len(partition_points)):
        #     p = multiprocessing.Process(
        #         target=best_point_finder,
        #         args=(partition_points[i], CC, GC, labels_dict, dataset_size, q)
        #     )
        #     processes.append(p)
        #     p.start()
        
        # for p in processes:
        #         sol_data = q.get()
        #         parallel_solutions.append(sol_data)
            
        # for p in processes:
        #     p.join()
        
        # for s in parallel_solutions:
        #     point_id = s[0]
        #     p_score = s[1] 
        #     if p_score > max_score:
        #         max_score = p_score
        #         best_point = point_id


        if best_point == -1:
            print("cannot find a point")
            break
        
        # cur.execute('select pl from cifar10_postinglist where id=%s', (best_point, ))
        # row = cur.fetchnone()[0]
        # posting_list_best_point = np.zeros(dataset_size)
        # posting_list_best_point[list(row)] = 1

        coreset.add(best_point)
        
        CC[0] = np.clip(np.subtract(CC[0], posting_list[best_point]), 0, None)
        GC[0] = np.clip(np.subtract(GC[0], labels_dict[best_point]), 0, None)


    end_time = time.time()
    print(len(coreset))
    print('Time Taken : {0}'.format(end_time - start_time))



# def best_point_finder(points, CC, GC, labels_dict, delta_size, q):
#     conn = psycopg2.connect(database="pmundra")
#     cur = conn.cursor()
#     posting_list = np.zeros((len(points), delta_size))
#     index_to_p_id = {}
#     # psql_start_time = time.time()
#     cur.execute('select id, pl from cifar10_postinglist where id in %s', (points, ))
#     records = cur.fetchall()
#     # print('db query time: {0}'.format(time.time() - psql_start_time))

#     for index, rows in enumerate(records):
#         posting_list[index][list(rows[1])] = 1
#         index_to_p_id[index] = rows[0]
    
#     p_scores = np.matmul(CC, posting_list.T)
#     max_score_index = p_scores[0].argmax()
#     cov_score = p_scores[0][max_score_index]
#     point_id = index_to_p_id[max_score_index]
#     q.put((point_id, cov_score))

#     if conn is not None:
#         conn.close()

if __name__ == '__main__':
    # posting_list_to_db()
    parser = argparse.ArgumentParser()

    # data selection parameters
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
    parser.add_argument('--coverage_threshold', type=float, default=1, help='coverage threshold to generate metadata')
    parser.add_argument('--partitions', type=int, default=10, help="number of partitions")
    parser.add_argument('--algo_type', type=str, default='greedyNC', help='which algorithm to use')
    parser.add_argument('--distribution_req', type=int, default=50, help='number of samples ')
    parser.add_argument('--coverage_factor', type=int, default=30, help='defining the coverage factor')
    parser.add_argument('--model_type', type=str, default='resnet-18', help='model used to produce the feature_vector')
    # parser.add_argument('--k', type=int, default=0, help='number of centers for k_centersNC')
    params = parser.parse_args()


    if params.dataset == 'cifar10':
        params.num_classes = 10 
        params.dataset_size = 50000 
    
    dr = [params.distribution_req] * params.num_classes
    gfkc_db(params.coverage_factor, dr, params.dataset, params.dataset_size, params.num_classes)