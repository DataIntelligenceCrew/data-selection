import faiss
import numpy as np 
import os 
from paths import *
import pickle 
import time 
import tqdm
import random 
import math 
import statistics
import sqlite3 
from json import dumps 
import psycopg2





'''

    Number of Partitions : 50 
        - faiss_index size : 221204
        - query_time: 6000 queries ~ 7 seconds
        - memory footprint:  8 GB

    Number of Partitions : 100
        - faiss_inde size : 110602
        - query_time: 6000 queries ~ 5 seconds 
        - memory footprint: 4 Gb 
'''

# def composable_test(partition_size):
#     N = feature_vectors.shape[0]
#     d = feature_vectors.shape[1]
#     all_ids = [i for i in range(N)]
#     sampled_ids = list(random.sample(all_ids, int(N/partition_size)))
#     fv_partitions = np.take(feature_vectors, sampled_ids, 0)

#     print(fv_partitions.shape[0])

#     xb = fv_partitions.astype('float32')
#     N = fv_partitions.shape[0]
#     xb[:, 0] += np.arange(N) / 1000
    
#     faiss_index = faiss.IndexFlatL2(d)
#     faiss.normalize_L2(x=xb)
#     faiss_index.add(xb)

#     print('successfully built faiss index (size : {0})'.format(faiss_index.ntotal))

#     batch_size = 1000
#     progress_bar = tqdm.tqdm(total=N, position=0)
#     for i in range(0, xb.shape[0], batch_size):
#         limits, D, I = faiss_index.range_search(xb[i:i+batch_size], 0.9)
#         progress_bar.update(batch_size)
#         if i > 4000:
#             break 
    

'''
    GPU 
        - only supports nearest neighour search with K <= 2048 
        - query time: 1000 queries ~ 0.15 seconds 
        - memory footprint : 65% memory utilization on all 4 GPUs
        - problem find the nearest neighbors not necessarily 
          the ones with sim(q,n) > threshold, most of them are least similar to the query
        - 809/2048 above threshold average over 1000 queries
    
    CPU
'''

def quantized_index():
    feature_vectors = pickle.load(open(FEATURE_VECTOR_LOC.format('imagenet', 'resnet-18'), 'rb'))
    d = feature_vectors.shape[1]
    print(d)
    N = feature_vectors.shape[0]
    M = 64 
    nlist = int(math.sqrt(N))
    nsegment = 32 
    nbit = 8
    xb = feature_vectors.astype('float32')
    xb[:, 0] += np.arange(N) / 1000

    coarse_quantizer = faiss.IndexHNSWFlat(d, M)
    # coarse_quantizer = faiss.IndexFlatL2(d)
    cpu_faiss_index = faiss.IndexIVFPQ(coarse_quantizer, d, nlist, nsegment, nbit)
    ngpus = faiss.get_num_gpus()
    faiss_index = faiss.index_cpu_to_all_gpus(cpu_faiss_index)
    faiss_index = faiss.IndexIVFPQ(coarse_quantizer, d, nlist, nsegment, nbit)
    faiss.normalize_L2(x=xb)
    faiss_index.train(xb)
    print('Faiss Trained')
    faiss_index.add(xb)
    print('successfully built faiss index (size : {0})'.format(faiss_index.ntotal))
    faiss_index.nprobe = int(nlist/2)
    start_time = time.time()
    D, I  = faiss_index.search(xb[:10001], 2048)
    duration = time.time() - start_time
    # print(type(D))
    # D = D.tolist()
    # D_above_cf = D[0][D[0] > 0.9]
    # print(D_above_cf)
    # print(len(D_above_cf))
    D_above_cf = []
    for d in D:
        D_above_cf.append(len(d[d>0.9]))
    # print(I)
    print(duration)
    print(statistics.mean(D_above_cf))

'''

Try it with postgress, as it sa an array datatype. 
'''

def init_db():
    con = sqlite3.connect("/localdisk3/data-selection/data/metadata/imagenet/0.9/posting_list.db")
    cursor = con.cursor()
    print('DB init')
    query = "CREATE TABLE posting_list(int ID, )"


def quantized_index_range_search_sql():
    feature_vectors = pickle.load(open(FEATURE_VECTOR_LOC.format('imagenet', 'resnet-18'), 'rb'))
    d = feature_vectors.shape[1]
    N = feature_vectors.shape[0]
    nlist = int(math.sqrt(N))
    nbit = faiss.ScalarQuantizer.QT_8bit
    xb = feature_vectors.astype('float32')
    xb[:, 0] += np.arange(N) / 1000

    coarse_quantizer = faiss.IndexFlatL2(d)
    faiss_index = faiss.IndexIVFScalarQuantizer(coarse_quantizer, d, nlist, nbit, faiss.METRIC_L2)
    faiss.normalize_L2(x=xb)
    faiss_index.train(xb)
    print('Faiss Trained')
    faiss_index.add(xb)
    faiss_index.nprobe = 16
    print('successfully built faiss index (size : {0})'.format(faiss_index.ntotal))
    batch_size = 1000
    progress_bar = tqdm.tqdm(total=N, position=0)
    posting_list = {}

    for i in range(0, xb.shape[0], batch_size):
        # start_time = time.time()
        limits, D, I = faiss_index.range_search(xb[i:i+batch_size], 0.9)
        # end_time = time.time()
        try:
            for j in range(batch_size):
                # print(j)
                pl = set(I[limits[j] : limits[j+1]])
                # posting_list[i+j] = pl
                key = i+j
                value = pl

        except IndexError:
            break
        progress_bar.update(batch_size)
        # break 


def quantized_index_range_search():
    feature_vectors = pickle.load(open(FEATURE_VECTOR_LOC.format('imagenet', 'resnet-18'), 'rb'))
    d = feature_vectors.shape[1]
    N = feature_vectors.shape[0]
    nlist = int(math.sqrt(N))
    nbit = faiss.ScalarQuantizer.QT_8bit
    xb = feature_vectors.astype('float32')
    xb[:, 0] += np.arange(N) / 1000
    conn = psycopg2.connect(database="pmundra")
    cur = conn.cursor()
    print('Database connection established')
    coarse_quantizer = faiss.IndexFlatL2(d)
    faiss_index = faiss.IndexIVFScalarQuantizer(coarse_quantizer, d, nlist, nbit, faiss.METRIC_L2)
    faiss.normalize_L2(x=xb)
    faiss_index.train(xb)
    print('Faiss Trained')
    faiss_index.add(xb)
    faiss_index.nprobe = 4
    print('successfully built faiss index (size : {0})'.format(faiss_index.ntotal))
    batch_size = 1000
    progress_bar = tqdm.tqdm(total=N - 6979000, position=0)
    posting_list = {}
    # with open('/localdisk2/faiss_sq_pl_resnet-18_imagenet.txt', 'w') as f:
    for i in range(6988000, xb.shape[0], batch_size):
        # start_time = time.time()
        limits, D, I = faiss_index.range_search(xb[i:i+batch_size], 0.9)
        # end_time = time.time()
        try:
            for j in range(batch_size):
                # print(j)
                pl = set(I[limits[j] : limits[j+1]])
                # posting_list[i+j] = pl
                key = i+j
                value = [int(p) for p in pl]
                # f.write(str(key) + ' : ' + str(value) + '\n')
                cur.execute("INSERT INTO postinglist9(id , pl) VALUES(%s, %s);",(key, value, ))
            conn.commit()
        except IndexError:
            break
        progress_bar.update(batch_size)
    cur.close()
    if conn is not None:
        conn.close()
        print('Database connection closed')
        # break 
    # f.close()
    # with open('/localdisk3/data-selection/data/metadata/imagenet/0.9/faiss_sq_pl_resnet-18.txt', 'w') as f:
    #     for key, value in posting_list.items():
    #         f.write(str(key) + ' : ' + str(value) + '\n')
    # f.close()
    # print(end_time-start_time)


def sq_paramter():
    feature_vectors = pickle.load(open(FEATURE_VECTOR_LOC.format('imagenet', 'resnet-18'), 'rb'))
    d = feature_vectors.shape[1]
    N = feature_vectors.shape[0]
    nlist = int(math.sqrt(N))
    nbit = faiss.ScalarQuantizer.QT_8bit
    xb = feature_vectors.astype('float32')
    xb[:, 0] += np.arange(N) / 1000

    coarse_quantizer = faiss.IndexFlatL2(d)
    faiss_index = faiss.IndexIVFScalarQuantizer(coarse_quantizer, d, nlist, nbit, faiss.METRIC_L2)
    faiss.normalize_L2(x=xb)
    faiss_index.train(xb)
    print('Faiss Trained')
    faiss_index.add(xb)
    # print(int(nlist/128))
    faiss_index.nprobe = 3
    batch_size = 10
    posting_list= {}
    for i in range(0, xb.shape[0], batch_size):
        start_time = time.time()
        limits, D, I = faiss_index.range_search(xb[i:i+batch_size], 0.9)
        end_time = time.time()
        for j in range(batch_size):
            pl = set(I[limits[j] : limits[j+1]])
            posting_list[i+j] = pl
        break 
        # try:
        #     for j in range(batch_size):
        #         # print(j)
        #         pl = set(I[limits[j] : limits[j+1]])
        #         # posting_list[i+j] = pl
        #         key = i+j
        #         value = pl
        #         f.write(str(key) + ' : ' + str(value) + '\n')
        # except IndexError:
        #     break
        # progress_bar.update(batch_size)
    print(end_time-start_time)
    print(len(posting_list[0]))
    print(posting_list[0])
    # return posting_list
# def pca_index():
#     pass


'''
    - using Scalar Quantizer 
    - to encode the vectors into codes using 8 bits 
    - avg_realtive_error (decoding, encoding): 3.845615e-06
    - built and index using this : IVFScalarQuantizer
    - moderate memory usage: 60GB
    - fast query time : 1000 queries ~ 0.2 seconds 
    - approxiamte index so compared to the exact search index
    - we generate smaller posting lists : on average 800-1000 less neighbours
'''
# def scalar_quantizer():
#     d = feature_vectors.shape[1]
#     N = feature_vectors.shape[0]
#     xb = feature_vectors.astype('float32')
#     xb[:, 0] += np.arange(N) / 1000

#     # QT_8bit allocates 8 bits per dimension (QT_4bit also works)
#     sq = faiss.ScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit)
#     sq.train(xb)

#     # encode 
#     codes = sq.compute_codes(xb)

#     # decode
#     x2 = sq.decode(codes)

#     # compute reconstruction error
#     avg_relative_error = ((xb - x2)**2).sum() / (xb ** 2).sum()
#     print(avg_relative_error)

def psql_db(posting_list):
    conn = None 
    try:
        conn = psycopg2.connect(
            database="pmundra"
        )
        cur = conn.cursor()
        print('PostgreSQL database version:')
        # cur.execute('SELECT version()')
        # cur.execute('CREATE TABLE postingList9(id INTEGER, pl INTEGER[]);')
        # for key, value in posting_list.items():
        #     v = [int(vl) for vl in value]
        #     cur.execute("INSERT INTO postinglist9(id , pl) VALUES(%s, %s);",(key, v, ))
        # conn.commit()
        cur.execute("select * from postinglist9;")
        records = cur.fetchall()
        for row in records:
            print('ID:{0}\nPL{1}'.format(row[0], row[1]))
        # db_version = cur.fetchone()
        # print(db_version)
        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    
    finally:
        if conn is not None:
            conn.close()
            print('database connection closed')





if __name__ == '__main__':
    # composable_test(50)
    # composable_test(100)
    # quantized_index()
    quantized_index_range_search()
    # init_db()
    # posting_list = sq_paramter()
    # posting_list= None
    # psql_db(posting_list)
    # sq_paramter()
    # scalar_quantizer()
    # p = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    # print(len(p))