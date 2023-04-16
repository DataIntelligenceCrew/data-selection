import math
import os
import numpy as np
import multiprocessing
import argparse
import faiss
from algorithms import *
from paths import *
from utils_algo import *
import pandas as pd
import tqdm 
import psycopg2
import multiprocessing
import random 



def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def get_labels_nyc():
    df = open('/localdisk3/nyc_1mil_2021-09_updated_labels.csv', 'r')
    lines = df.readlines()
    data = [line.strip() for line in lines]
    df.close()
    del data[0]
    # print(len(data))
    labels_dicts = {}
    for row in data:
        # print(row)

        row_s = row.split(',')
        key = int(row_s[0])
        try:
            label = int(row_s[1])
            if label not in labels_dicts.keys():
                labels_dicts[label] = list()
            labels_dicts[label].append(key)
        except ValueError:
            continue
    # print(labels_dicts)
    counts = {k:len(v) for k,v in labels_dicts.items()}
    # print(len(labels_dicts.keys()))
    return labels_dicts, counts



def run_algo(params, dr, labels_dict):
    print('Running Algorithm: {0}\nDataset:{1}\nDistribution Requirement:{2}\nCoverage Factor:{3}\nCoverage Threshold:{4}\n'.format(
            params.algo_type,
            params.dataset,
            params.distribution_req,
            params.coverage_factor,
            params.coverage_threshold
        ))
    solution_data = []
    if params.algo_type == 'greedyNC':
        # posting_list = get_posting_list_nyc_dist()
        s, res_time = greedyNC_nyc(params.coverage_factor, dr, params.dataset_size, params.num_classes, labels_dict)
        solution_data.append((s, res_time))
    
    elif params.algo_type == 'greedyC_random':
        points = [i for i in range(params.dataset_size)]
        p = 10
        paritions_ids = partition(points, p)
        q = multiprocessing.Queue()
        processes = []
        dr = [dr[i] // 10 for i in range(len(dr))]
        for i in range(p):
            p = multiprocessing.Process(
                target=greedyC_random_nyc,
                args=(params.coverage_factor, dr, params.dataset_size, params.num_classes, labels_dict, paritions_ids[i], q)
            )
            processes.append(p)
            p.start()
        

        for p in processes:
            sol_data = q.get()
            solution_data.append(sol_data)
        
        for p in processes:
            p.join()


    coreset = set()
    response_time = 0
    for t in solution_data:
        coreset = coreset.union(t[0])
        response_time = max(response_time, t[1])
    
    print('Time Taken:{0}'.format(response_time))
    print('Solution Size: {0}'.format(len(coreset)))
    # report metrics
    metric_file_name = METRIC_FILE2.format(params.dataset, params.coverage_factor, params.distribution_req, params.algo_type, params.coverage_threshold, params.model_type)
    with open(metric_file_name, 'w') as f:
        f.write(
            'Dataset Size: {0}\nCoreset Size: {1}\nTime Taken: {2}\n'.format(
                int(params.dataset_size),
                len(coreset),
                response_time
            )
        )
    f.close()

    # log coreset points
    solution_file_name = SOLUTION_FILENAME2.format(params.dataset, params.coverage_factor, params.distribution_req, params.algo_type, params.coverage_threshold, params.model_type)
    with open(solution_file_name, 'w') as f:
        for s in coreset:
            f.write("{0}\n".format(s))
    f.close()



def greedyNC_nyc(coverage_factor, distribution_req, dataset_size, num_classes, labels_dict):
    '''
    Computes the greedy fair set cover for the entire dataset
    @params
        coverage_factor : number of points needed for a point to be covered
        distribution_req : number of points from each group
        sp : sampling weight to get the correct posting_list directory
    @returns
        solution of the greedy fair set cover that satisfies the coverage and 
        distribution requirements
        coverage score for this solution
        time taken 
    '''
    start_time = time.time()

    delta_size = dataset_size
    solution = set() # solution set 
    delta = set([i for i in range(dataset_size)])
    sparse_points = list()
    conn = psycopg2.connect(database="pmundra")
    print('database connection to find sparse points')
    cur = conn.cursor()
    cur.execute("select * from nyc_postinglist where ARRAY_LENGTH(pl, 1)<%s;", (coverage_factor, ))
    records = cur.fetchall()
    for rows in records:
        p_id = rows[0]
        id_pl = set(rows[1])
        sparse_points.append(p_id)
        solution.add(p_id)
        solution = solution.union(id_pl)
    
    labels_dict_np = dict()
    for key, values in labels_dict.items():
        for p in values:
            arr = np.zeros(num_classes)
            arr[key] = 1
            labels_dict_np[p] = arr

    CC = np.zeros((1, delta_size)) # coverage tracker
    CC[0][list(delta)] = coverage_factor
    CC[0][sparse_points] = 0
    print(CC.shape)
    # with open('nyc_sparsepoints_cf_{0}.txt'.format(coverage_factor), 'w') as f:
    #     for p in sparse_points:
    #         f.write('{0}\n'.format(str(p)))
    # f.close()
    print('Number of sparse points for CF:{0} is {1}'.format(coverage_factor, len(sparse_points)))
    GC = np.zeros((1, num_classes))
    GC[0] = np.array(distribution_req)  # group count tracker
    # main loop
    iters = 0
    while ((not np.all(CC[0] <= 0)) or (not np.all(GC[0] <= 0))) and len(solution) < len(delta):
    # while (not check_for_zeros(CC)) and len(solution) < len(delta):
        best_point, max_score = -1, float('-inf')
        # toDo: optimize this loop using scipy
        possible_ids = list(delta.difference(solution))
        batch_size = 200000
        progress_bar = tqdm.tqdm(total=len(possible_ids), position=0)
        for i in range(0, len(possible_ids), batch_size):
            batch_ids = tuple(possible_ids[i:i+batch_size])
            
            # parallel search for a large batch 
            parallel_solutions = []
            partition_size = batch_size // 20 
            part_points = [tuple(batch_ids[j:j+partition_size]) for j in range(0, len(batch_ids), partition_size)]
            q = multiprocessing.Queue()
            processes = []
            for i in range(len(part_points)):
                p = multiprocessing.Process(
                    target=best_point_finder,
                    args=(part_points[i], CC, delta_size, q)
                )
                processes.append(p)
                p.start()
            
            for p in processes:
                sol_data = q.get()
                parallel_solutions.append(sol_data)
            
            for p in processes:
                p.join()
            
            for s in parallel_solutions:
                point_id = s[0]
                p_score = s[1] + distritbution_score(GC, labels_dict_np[point_id])
                if p_score > max_score:
                    max_score = p_score
                    best_point = point_id
            ## sequential search for a batch
            # posting_list = np.zeros((batch_size, delta_size))
            # index_to_p_id = {}
            # psql_start_time = time.time()
            # cur.execute('select id, pl from nyc_postinglist where id in %s', (batch_ids, ))
            # records = cur.fetchall()
            # print('db query time: {0}'.format(time.time() - psql_start_time))
            # for index, rows in enumerate(records):
            #     posting_list[index][list(rows[1])] = 1
            #     index_to_p_id[index] = rows[0]

            # p_scores = np.matmul(CC, posting_list.T)
            # max_score_index = p_scores[0].argmax()
            # p_score = p_scores[0][max_score_index] + distritbution_score(GC, labels_dict_np[p])
            # if p_score > max_score:
            #     max_score = p_score
            #     best_point = index_to_p_id[max_score_index]

            progress_bar.update(batch_size)
        if best_point == -1:
            print("cannot find a point")
            break
        
        cur.execute('select pl from nyc_postinglist where id=%s', (best_point, ))
        row = cur.fetchall()
        for r in row:
            pl = list(r[0])
            posting_list_best_point = np.zeros(delta_size)
            posting_list_best_point[pl] = 1

        solution.add(best_point)
        
        CC[0] = np.clip(np.subtract(CC[0], posting_list_best_point), 0, None)
        GC[0] = np.clip(np.subtract(GC, labels_dict_np[best_point]), 0, None)
        
        iters += 1
    end_time = time.time()
    print(len(solution)) 

    res_time = end_time - start_time
    if conn is not None:
        conn.close()
    return solution, res_time



def greedyC_random_nyc(coverage_factor, distribution_req, dataset_size, num_classes, labels_dict, operatable_points, q):
    '''
    Computes the greedy fair set cover for the entire dataset
    @params
        coverage_factor : number of points needed for a point to be covered
        distribution_req : number of points from each group
        sp : sampling weight to get the correct posting_list directory
    @returns
        solution of the greedy fair set cover that satisfies the coverage and 
        distribution requirements
        coverage score for this solution
        time taken 
    '''
    start_time = time.time()

    delta_size = dataset_size
    solution = set() # solution set 
    # delta = [i for i in range(dataset_size)]
    sparse_points = list()
    conn = psycopg2.connect(database="pmundra")
    print('database connection to find sparse points')
    cur = conn.cursor()
    operatable_points = tuple(operatable_points)
    cur.execute("select * from nyc_postinglist where ARRAY_LENGTH(pl, 1)<%s and id in %s;", (coverage_factor, operatable_points, ))
    records = cur.fetchall()
    for rows in records:
        p_id = rows[0]
        id_pl = set(rows[1])
        sparse_points.append(p_id)
        solution.add(p_id)
        solution = solution.union(id_pl)
    
    labels_dict_np = dict()
    for key, values in labels_dict.items():
        for p in values:
            if p in operatable_points:
                arr = np.zeros(num_classes)
                arr[key] = 1
                labels_dict_np[p] = arr

    CC = np.zeros((1, delta_size)) # coverage tracker
    CC[0][list(operatable_points)] = coverage_factor
    CC[0][sparse_points] = 0
    print(CC.shape)
    # with open('nyc_sparsepoints_cf_{0}.txt'.format(coverage_factor), 'w') as f:
    #     for p in sparse_points:
    #         f.write('{0}\n'.format(str(p)))
    # f.close()
    print('Number of sparse points for CF:{0} is {1}'.format(coverage_factor, len(sparse_points)))
    GC = np.zeros((1, num_classes))
    GC[0] = np.array(distribution_req)  # group count tracker
    operatable_points = set(operatable_points)
    # main loop
    iters = 0
    while ((not np.all(CC[0] <= 0)) or (not np.all(GC[0] <= 0))) and len(solution) < len(operatable_points):
    # while (not check_for_zeros(CC)) and len(solution) < len(delta):
        best_point, max_score = -1, float('-inf')
        # toDo: optimize this loop using scipy
        possible_ids = list(operatable_points.difference(solution))
        batch_size = 20000
        progress_bar = tqdm.tqdm(total=len(possible_ids), position=0)
        for i in range(0, len(possible_ids), batch_size):
            batch_ids = tuple(possible_ids[i:i+batch_size])
            
            # parallel search for a large batch 
            parallel_solutions = []
            partition_size = batch_size // 20 
            part_points = [tuple(batch_ids[j:j+partition_size]) for j in range(0, len(batch_ids), partition_size)]
            q = multiprocessing.Queue()
            processes = []
            for i in range(len(part_points)):
                p = multiprocessing.Process(
                    target=best_point_finder,
                    args=(part_points[i], CC, GC, labels_dict_np, delta_size, q)
                )
                processes.append(p)
                p.start()
            
            for p in processes:
                sol_data = q.get()
                parallel_solutions.append(sol_data)
            
            for p in processes:
                p.join()
            
            for s in parallel_solutions:
                point_id = s[0]
                p_score = s[1] 
                if p_score > max_score:
                    max_score = p_score
                    best_point = point_id
            ## sequential search for a batch
            # posting_list = np.zeros((batch_size, delta_size))
            # index_to_p_id = {}
            # psql_start_time = time.time()
            # cur.execute('select id, pl from nyc_postinglist where id in %s', (batch_ids, ))
            # records = cur.fetchall()
            # print('db query time: {0}'.format(time.time() - psql_start_time))
            # for index, rows in enumerate(records):
            #     posting_list[index][list(rows[1])] = 1
            #     index_to_p_id[index] = rows[0]

            # p_scores = np.matmul(CC, posting_list.T)
            # max_score_index = p_scores[0].argmax()
            # p_score = p_scores[0][max_score_index] + distritbution_score(GC, labels_dict_np[p])
            # if p_score > max_score:
            #     max_score = p_score
            #     best_point = index_to_p_id[max_score_index]

            progress_bar.update(batch_size)
        if best_point == -1:
            print("cannot find a point")
            break
        
        cur.execute('select pl from nyc_postinglist where id=%s', (best_point, ))
        row = cur.fetchall()
        for r in row:
            pl = list(r[0])
            posting_list_best_point = np.zeros(delta_size)
            posting_list_best_point[pl] = 1

        solution.add(best_point)
        
        CC[0] = np.clip(np.subtract(CC[0], posting_list_best_point), 0, None)
        GC[0] = np.clip(np.subtract(GC[0], labels_dict_np[best_point]), 0, None)
        
        iters += 1
    end_time = time.time()
    print(len(solution)) 
    # cscore = calculate_cscore(solution, posting_list, delta_size)
    res_time = end_time - start_time
    if conn is not None:
        conn.close()
    q.put((solution, res_time))

def best_point_finder(points, CC, GC, labels_dict, delta_size, q):
    conn = psycopg2.connect(database="pmundra")
    cur = conn.cursor()
    posting_list = np.zeros((len(points), delta_size))
    index_to_p_id = {}
    # psql_start_time = time.time()
    cur.execute('select id, pl from nyc_postinglist where id in %s', (points, ))
    records = cur.fetchall()
    # print('db query time: {0}'.format(time.time() - psql_start_time))

    for index, rows in enumerate(records):
        posting_list[index][list(rows[1])] = 1
        index_to_p_id[index] = rows[0]
    
    p_scores = np.matmul(CC, posting_list.T)
    p_scores[0] = np.array([j + np.dot(GC[0].T, labels_dict[index_to_p_id[i]]) for i, j in enumerate(p_scores[0])])
    max_score_index = p_scores[0].argmax()
    cov_score = p_scores[0][max_score_index]
    point_id = index_to_p_id[max_score_index]
    q.put((point_id, cov_score))

    if conn is not None:
        conn.close()

def combine_posting_lists():
    N = 10 
    dir_loc = "/localdisk3/data-selection/data/metadata/nyc_taxicab/10/loc_pl/{0}"
    posting_list = {}
    progress_bar = tqdm.tqdm(total=N, position=0)
    for i in range(N):
        curr_dir = dir_loc.format(i)
        files = [os.path.join(curr_dir, x) for x in os.listdir(curr_dir)]
        # print(files)
        progress_bar_internal = tqdm.tqdm(total=len(files), position=1)
        for f in files:
            f1 = open(f)
            dist_lines = f1.readlines()
            dist_data = [line.strip().replace('{', '').replace('}', '') for line in dist_lines]
            f1.close()
            dist_dict = convert_to_dict(dist_data)
            for key in dist_dict.keys():
                if key not in posting_list:
                    posting_list[key] = set()
                
                posting_list[key] = posting_list[key].union(dist_dict[key])
        progress_bar_internal.update(1)
    progress_bar.update(1)

    with open('/localdisk3/data-selection/data/metadata/nyc_taxicab/sim_PU_1_DO_1.txt', 'w') as f:
        for key, value in posting_list.items():
            f.write(str(key) + ' : ' + str(value) + '\n')
    
    f.close()



def convert_to_dict(data):
    result = {}
    for d in data:
        pl = d.split(':')
        key = int(pl[0])
        value = pl[1].split(',')
        if value != [' set()']:
            value = [int(v.replace("{", "").replace("}", "").replace("'", '').strip()) for v in value]
        else:
            value = []
        result[key] = set(value)
        # print(values[0])
    return result


def posting_list_postgress():
    posting_list = get_posting_list_nyc_dist()
    conn = None 
    try:
        conn = psycopg2.connect(database="pmundra")
        cur = conn.cursor()
        print('Database connection established')
        progress_bar = tqdm.tqdm(total=len(posting_list), position=0)
        for key, value in posting_list.items():
            v = [int(vl) for vl in value]
            cur.execute("INSERT INTO nyc_postinglist(id, pl) VALUES(%s, %s);", (key, v, ))
            conn.commit()
            progress_bar.update(1)

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('database connection closed')


def test_fetch_db():
    conn = None 
    try:
        conn = psycopg2.connect(database="pmundra")
        cur = conn.cursor()
        # cur.execute("select pl from nyc_postinglist where id=%s;", (6, ))
        cur.execute("select * from nyc_postinglist where ARRAY_LENGTH(pl, 1)<%s LIMIT 5;", (5, ))
        recs = cur.fetchall()
        for row in recs:
            print(row)

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Databse closed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nyc_taxicab', help='dataset to use')
    parser.add_argument('--coverage_threshold', type=float, default=1, help='coverage threshold to generate metadata')
    parser.add_argument('--partitions', type=int, default=10, help="number of partitions")
    parser.add_argument('--algo_type', type=str, default='greedyC_random', help='which algorithm to use')
    parser.add_argument('--distribution_req', type=int, default=50, help='number of samples ')
    parser.add_argument('--coverage_factor', type=int, default=5, help='defining the coverage factor')
    parser.add_argument('--model_type', type=str, default='resnet-18', help='model used to produce the feature_vector')
    parser.add_argument('--k', type=int, default=0, help='number of centers for k_centersNC')
    params = parser.parse_args()
    
    params.dataset_size = 1000000
    params.num_classes = 7

    labels_dict, counts = get_labels_nyc()
    dr = [params.distribution_req] * params.num_classes
    dr_updated = [counts[i] if counts[i] < dr[i] else dr[i] for i in range(params.num_classes)]
    # print(dr_updated)

    # print(counts)
    run_algo(params, dr_updated, labels_dict)
    # combine_posting_lists()
    # posting_list_postgress()
    # test_fetch_db()