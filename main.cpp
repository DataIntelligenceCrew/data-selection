#include <iostream>
// #include <omp.h>
// #include <faiss/IndexFlat.h>
// #include <faiss/gpu/GpuIndexFlat.h>
// #include <faiss/gpu/StandardGpuResources.h>
// #include <faiss/utils/distances.h>
// #include <faiss/index_io.h>
#include <stdlib.h>
#include <map>
#include <set>
#include <iterator>
#include <chrono>
#include <iomanip>
// #include <sqlite3.h>
#include <fstream>
#include <vector>
#include <sstream>
#include <random>
#include <mutex>
/*
    we have a SQL database with the following for an image : name/id, label, feature_vector
    create nameToID set
    create vec_array indexed by ID 
    create label_array indexed by ID
    after faiss search create posting_list and inverted_index
*/
using namespace std;
// decreasing comparator
struct cmp_decreasing {
    bool operator() (pair<int, int> lhs, pair<int, int> rhs) const {
        return get<0>(lhs) >= get<0>(rhs);
    }
};
// using idx_t = faiss::Index::idx_t;
// faiss::gpu::StandardGpuResources res; // global GPU resource


// /**********************
//  * Sqlite3 Instance
// ***********************/
// class Database {
//     private:
//         sqlite3 *db; // f_vector database instance
//         sqlite3 *dest; // database location
//         std::unordered_map<int, string> int2word; // integer_ID --> image name
//         std::unordered_map<string, int> word2int; // image name --> integer_ID
//         std::unordered_map<int, vector<float>> int2vector; // integer_ID --> feature vector
//         std::unordered_map<int, int> int2label; // integer_ID --> image label


//         // convert mysql entry from bytes to vector
//         vector<float> bytes_to_vec(const unsigned char *data, size_t dimension) {
//             vector<float> result;
//             for (size_t i = 0; i < dimension * 4; i = i + 4) {
//                 float f;
//                 unsigned char b[] = {data[i], data[i+1], data[i+2], data[i+3]};
//                 memcpy(&f, &b, sizeof(f));
//                 result.push_back(f);
//             }
//             return result;
//         }
    
//     public:
//         Database(string path) {
//             cout << path.c_str() << endl;
//             int rc = sqlite3_open(path.c_str(), &dest);
            
//             if (rc) {
//                 cout << "Cannot open database: " << sqlite3_errmsg(dest) << endl;
//                 exit(0);
//             } else {
//                 cout << "Successfully opened sqlite3 database" << endl;
//             }

//             // open the database in-memory for better performance
//             rc = sqlite3_open(":memory:", &db);
// 			if (rc) {
// 				cout << "Cannot open in-memory database:  " << sqlite3_errmsg(db) << endl; 
// 				exit(0);
// 			} else {
// 				cout << "Successfully opened in-memory database" << endl;
// 			}

// 			sqlite3_backup *pBackup;
// 			pBackup = sqlite3_backup_init(db, "main", dest, "main");
// 			if(pBackup){
// 				sqlite3_backup_step(pBackup, -1);
// 				sqlite3_backup_finish(pBackup);
// 			}
// 			rc = sqlite3_errcode(db);
// 			if (rc) {
// 				cout << "Cannot copy database:  " << sqlite3_errmsg(db) << endl; 
// 				exit(0);
// 			} else {
// 				cout << "Successfully copied to memory" << endl;
// 			}
// 			sqlite3_close(dest);
//         }

//         void terminate() {
//             sqlite3_close(db);
//             cout << "Successfully terminated sqlite3 database" << endl;
//         }

//         vector<vector<float>> get_all_vectors() {
//             vector<vector<float>> result;
//             int rc;
//             stringstream ss;
            
//             ss << "SELECT name, label, f_vector FROM fv;";
//             string query = ss.str();
//             sqlite3_stmt *stmt = NULL;
            
//             rc = sqlite3_prepare(db, query.c_str(), -1, &stmt, NULL);
            
//             if (rc != SQLITE_OK) {
//                 cout << "SELECT failed: " << sqlite3_errmsg(db) << endl;
//             }
//             int nid = 0; 
//             while((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
//                 const char *word = (char*)sqlite3_column_text(stmt, 0);
//                 int label_id = sqlite3_column_int(stmt, 1); // need to figure this out
// 				unsigned char *bytes = (unsigned char*)sqlite3_column_blob(stmt, 2);
//                 string str_t(word);
//                 vector<float> vector_temp = bytes_to_vec(bytes, 300);
//                 int2word[nid] = str_t;
//                 word2int[str_t] = nid;
//                 int2vector[nid] = vector_temp;
//                 // int2label[nid] = label_id; // need to figure this out
//                 result.push_back(vector_temp);
//                 nid += 1;
//             }
//             cout << "Dataset Size: " << nid << endl;
//             return result;
//         }
// };

// /***************************
//  * FAISS Index (GPU Wrapper)
// ****************************/
// class FaissIndexGPU {
//     private:
//         faiss::gpu::GpuIndexFlatIP *index;
//         vector<int> dictionary;
    
//     public:
//         FaissIndexGPU(string path, Database *db) {
            
//             string indexpath = path + "faiss.index";
//             int dimension = 300;
            
//             vector<vector<float>> vectors = db->get_all_vectors();
//             index = new faiss::gpu::GpuIndexFlatIP(&res, dimension);
            
//             int nb = vectors.size();
//             float *xb = new float[dimension * nb];
//             for (int i = 0; i < nb; i++) {
//                 for (int j = 0; j < dimension; j++) {
//                     xb[dimension * i + j] = vectors[i][j];
//                 }
//             }

//             // faiss::fvec_renorm_L2(dimension, nb, xb); // do we need this?
//             index->add(nb, xb);
//             cout << "Successfully Initialized FAISS Index" << endl;
//         }

//         // toDo : implement this
//         // tuple<vector<idx_t>, vector<float>> range_search(int nq, vector<float> vxq, double threshold) {}

//         void destroy() {
//             index = NULL;
//             cout << "Destroyed FAISS Index" << endl;
//         }
// };

/************************************
 * Main Algorithm + Helper Functions
*************************************/
bool k_coverage_satisfied(vector<int> *coverage) {
    for (auto it = coverage->begin(); it != coverage->end(); it++) {
        if (*it > 0) {
            return false;
        }
    }
    return true;
}


void update_costs(map<int, int> *id2cost, vector<int> posting_list, map<int, vector<int>> *inverted_index) {
    for (auto it : posting_list) {
        vector<int> inv_idx = inverted_index->at(it);
        for (auto it2 : inv_idx) {
            id2cost->at(it2)--;
        }
    }
}

void update_k_coverage_counter(vector<int> *k_coverage, vector<int> posting_list) {
    for (auto it :  posting_list) {
        k_coverage->at(it)--;
    }
}

set<pair<int, int>, cmp_decreasing> update_pq(map<int, int> *id2cost, set<int> *result_set) {
    std::set<pair<int, int>, cmp_decreasing> token_stream;
    for (auto it = id2cost->begin(); it != id2cost->end(); it++) {
        // it->first == set_id; it->second == cost 
        if ((result_set->find(it->first) == result_set->end())) {
            token_stream.insert(make_pair(it->second, it->first));
        }
    }
    return token_stream;
}

void algorithm(int k_coverage, map<int, vector<int>> posting_lists, 
                        map<int, vector<int>> inverted_index, std::set<int> *global_solution, std::mutex *gmtx) {
    
    // data structures
    vector<int> k_coverage_counter(posting_lists.size(), k_coverage);
    std::map<int, int> id2cost;
    std::set<pair<int, int>, cmp_decreasing> token_stream; // (cost, set_id)
    std::set<int> set_cover_solution;

    // initialization
    std::map<int, vector<int>>::iterator it;
    for (it = posting_lists.begin(); it != posting_lists.end(); it++) {
        int set_id = it->first;
        int posting_list_size = static_cast<int>(it->second.size());
        token_stream.insert(make_pair(posting_list_size * k_coverage, set_id));
        id2cost[set_id] = posting_list_size * k_coverage;
        // if (global_solution.find(set_id) == global_solution.end()) {
        //     token_stream.insert(make_pair(posting_list_size * k_coverage, set_id));
        //     id2cost[set_id] = posting_list_size * k_coverage;
        // } else {
        //     id2cost[set_id] = 0;
        // }
    }

    // master loop
    while (!k_coverage_satisfied(&k_coverage_counter)) {
        if (token_stream.empty()) {
            cout << "Not enough points" << endl;
            break;
        } 
        auto it = token_stream.begin();
        pair<int, int> token = *it;
        token_stream.erase(it);
        int set_id = token.second;
        set_cover_solution.insert(set_id);
        update_k_coverage_counter(&k_coverage_counter, posting_lists.at(set_id)); // can be avoided
        update_costs(&id2cost, posting_lists.at(set_id), &inverted_index); // extra bit
        token_stream = update_pq(&id2cost, &set_cover_solution); // instead of rebuilding, just keep track of the token_id and if its -1 then break
    }
    // ofstream output;
    // output.open("set_cover_solution_alexNet.txt");
    // if (!output) {
    //     for (auto i : set_cover_solution) {
    //         cout << i << endl;
    //     }
    //     cerr << "Error : file couldn't be opened" << endl;
    //     exit(1);
    // }
    // for (auto i : set_cover_solution) {
    //     output << i  << endl;
    // }
    // output.close();
    // cout << "Set Cover Size: " << set_cover_solution.size() << " for k = " << k_coverage << endl;
    // float percent_of_total = (set_cover_solution.size() / posting_lists.size()) * 100;
    // cout << "Percent of Total Data: " << percent_of_total <<  endl;
    // return set_cover_solution;
    gmtx->lock();
    for (auto ls : set_cover_solution) {
        global_solution->insert(ls);
    }
    gmtx->unlock();

}


// void algorithm_2(int k, map<int, vector<int>> posting_lists, map<int, vector<int>> inverted_index) {
//     // datastructures
//     std::set<int> global_set_cover;


//     for (int i = 0; i < k; i++) {
//         std::set<int> local_set_cover = algorithm(1, posting_lists, inverted_index, global_set_cover);
//         global_set_cover.insert(local_set_cover.begin(), local_set_cover.end());
//     }


//     ofstream output;
//     output.open("global_set_cover_alexNET.txt");
//     if (!output) {
//         for (auto i : global_set_cover) {
//             cout << i << endl;
//         }
//         cerr << "Error : file couldn't be opened" << endl;
//         exit(1);
//     }
//     for (auto i : global_set_cover) {
//         output << i  << endl;
//     }
//     output.close();
//     cout << "Set Cover Size: " << global_set_cover.size() << " for k = " << k << endl;

// }


/***************************
 * Generate Metadata from Files
****************************/
vector<int> get_values(const string &s, char delim, std::set<int> partition) {
    vector<int> tokens;
    string token;
    stringstream tokenStream(s);
    while (getline(tokenStream, token, delim)) {
        // check if token in the subset and then insert
        if (partition.find(stoi(token)) != partition.end()) {
            tokens.push_back(stoi(token));
        }
        // tokens.push_back(stoi(token));
    }
    return tokens;
}

map<int, vector<int>> get_metadata(string filepath, std::set<int> partition) {
    map<int, vector<int>> mymap;
    ifstream file_stream;
    file_stream.open(filepath);
    string line;
    string delim = ":";
    while (file_stream) {
        getline(file_stream, line);
        vector<int> value_ids;
        if (line.size() > 3) {
            string key = line.substr(0, line.find(delim));
            string value = line.substr(line.find(delim) + 2, line.size());
            value = value.substr(1, value.size() - 2);
            value_ids = get_values(value, ',', partition);
            if (partition.find(stoi(key)) != partition.end()) {
                mymap.insert(pair<int, vector<int>>(stoi(key), value_ids));
            }
            // mymap.insert(pair<int, vector<int>>(stoi(key), value_ids));
        }
    }
    return mymap;
}


int main(int argc, char const *argv[]) {
    // double vicinity_threshold = 0.0;
    // int coverage_threshold = 0;
    // string count_requirement_filepath = "";

    // if (argc < 3) {
    //     cerr << "Requires 3 arguments. Usage: ./main vicinity_threshold(double) coverage_threshold(int) filepath_for_group_count_requirements(string)" << endl;
    //     return 1;
    // }
    // std::cout << "FAISS test" << std::endl;
    // int d = 32;
    // int nb = 1000;
    // int nq = 100;
    // float *xb = new float[d * nb];
    // float *xq = new float[d * nq];
    // for(int i = 0; i < nb; i++) {
    //     for(int j = 0; j < d; j++) xb[d * i + j] = drand48();
    //     xb[d * i] += i / 1000.;
    // }
    // for(int i = 0; i < nq; i++) {
    //     for(int j = 0; j < d; j++) xq[d * i + j] = drand48();
    //     xq[d * i] += i / 1000.;
    // }

    // faiss::IndexFlatL2 index(d);
    // printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    // index.add(nb, xb);                     // add vectors to the index
    // printf("ntotal = %ld\n", index.ntotal);

    // int k = 10;
    // // sanity check: search 5 first vectors of xb
    // idx_t *I = new idx_t[k * 5];
    // float *D = new float[k * 5];
    // index.search(5, xb, k, D, I);
    // printf("I=\n");
    // for(int i = 0; i < 5; i++) {
    //     for(int j = 0; j < k; j++) printf("%5ld ", I[i * k + j]);
    //     printf("\n");
    // }
    // delete [] I;
    // delete [] D;

    // // testing sqlite3
    // Database *ftdb = new Database("/localdisk1/sematic-overlap-cpp/ft.sqlite3");

    // random partitioning for compasable algorithm
    cout << "Starting partitioning" << endl;
    int number_of_partitions = 5;
    std::map<int, set<int>> partitions;
    int DELTA_SIZE = 50000;
    for (int i = 0; i < DELTA_SIZE; i++) {
        int part_id = rand() % number_of_partitions;
        if (partitions.find(part_id) == partitions.end()) {
            std::set<int> s;
            partitions.insert(make_pair(part_id, s));
        }
        partitions[part_id].insert(i);
    }
    cout << "Paritioning Done" << endl;


    std::set<int> global_solution;
    std::mutex gmtx;
    string posting_file_location = "/localdisk3/data-selection/posting_list_alexnet.txt";
    string inverted_index_location = "/localdisk3/data-selection/inverted_index_alexnet.txt";

    for (int j = 0; j < number_of_partitions; j++) {
        chrono::time_point<chrono::high_resolution_clock> tstart, tmiddle, tend;
        chrono::duration<double> elapsed, total_elapsed;
        
        tstart = chrono::high_resolution_clock::now();
        
        map<int, vector<int>> posting_lists = get_metadata(posting_file_location, partitions[j]);
        map<int, vector<int>> inverted_index = get_metadata(inverted_index_location, partitions[j]);
        
        tmiddle = chrono::high_resolution_clock::now();
        cout << posting_lists.size() << endl;
        cout << inverted_index.size() << endl;
        elapsed = tmiddle - tstart;
        cout << "Time taken to load metadata: " << elapsed.count() << " for parition_number: " << j << endl;
        
        int k = 2;
        cout << "Starting for parition_number" << j << endl;
        algorithm(k, posting_lists, inverted_index, &global_solution, &gmtx);
        tend = chrono::high_resolution_clock::now();
        total_elapsed = tend - tmiddle;
        cout << "Set Cover Time: " << total_elapsed.count() << endl;
    }
    
    ofstream output;
    output.open("global_set_cover_alexNET_composable.txt");
    if (!output) {
        for (auto i : global_solution) {
            cout << i << endl;
        }
        cerr << "Error : file couldn't be opened" << endl;
        exit(1);
    }
    for (auto i : global_solution) {
        output << i  << endl;
    }
    output.close();
    cout << "Set Cover Size: " << global_solution.size() << " for k = 2" << endl;

    // string posting_file_location = "/localdisk3/data-selection/posting_list_alexnet.txt";
    // string inverted_index_location = "/localdisk3/data-selection/inverted_index_alexnet.txt";
    
    // chrono::time_point<chrono::high_resolution_clock> tstart, tmiddle, tend;
    // chrono::duration<double> elapsed, total_elapsed;
    
    // tstart = chrono::high_resolution_clock::now();
    
    // map<int, vector<int>> posting_lists = get_metadata(posting_file_location);
    // map<int, vector<int>> inverted_index = get_metadata(inverted_index_location);
    
    // tmiddle = chrono::high_resolution_clock::now();
    // cout << posting_lists.size() << endl;
    // cout << inverted_index.size() << endl;
    // elapsed = tmiddle - tstart;
    // cout << "Time taken to load metadata: " << elapsed.count() << endl;
    // ifstream posting_file_stream;
    // posting_file_stream.open(posting_file_location);
    // string line;
    // string delim = ":";
    // while (posting_file_stream) {
    //     getline(posting_file_stream, line);
    //     string key = line.substr(0, line.find(delim));
    //     string value = line.substr(line.find(delim) + 2, line.size());
    //     value = value.substr(1, value.size() - 2);
    //     vector<int> value_ids = get_values(value, ',');
    //     posting_lists.insert(pair<int, vector<int>>(stoi(key), value_ids));
    //     // cout << key << endl;
    //     // cout << value_ids[0] << endl;
    //     // cout << value_ids[value_ids.size() - 1] << endl;
    //     break;
    // }
    // int k = 2;
    // // set<int> final_solution = algorithm(k, posting_lists, inverted_index);
    // algorithm_2(k, posting_lists, inverted_index);
    // tend = chrono::high_resolution_clock::now();
    // total_elapsed = tend - tmiddle;
    // cout << "Set Cover Time: " << total_elapsed.count() << endl;
    return 0;
}


/*

toDo : main algorithm
     : FAISS index, SQL DB for vector storage 
     : balancing stage algo

**** Greedy Weighted Set Cover Algorithm (Pseudocode) ****

Assumptions : - we have an index that returns us top-k vectors similar to the query vector
              - 

Data-structures : - point_stream - priority_queue of tuples - (point, cost) ordered by cost in decreasing order
                  - posting_list - query_point --> [list of points it covers / it's top-m neighbours] (?? should the posting list contain the point itself ??)
                  - inverted_index - query_point --> [list of points that cover this point, i.e, all points such that q_p belongs to the points top-m neighours]
                  - k-coverage-counter - keeps tracks whether each point satisfies the k-coverage criteria 
                  - 
Approach:
D : all points in the data_set
S : solution of the set_cover



// assigns cost of choosing a query_point (?? don't need this as we initialize and then subtract ??)
cost_func(query_point):
    p_list = get_posting_list(query_point)
    cost = 0.0
    for e in p_list:
        cost += (k - k_coverage_counter.of(e)) 

    return cost

// cost update based on posting_list
update_cost(p_list):
    for each point e in p_list:
        list_points_that_contain_e = inverted_index(e)
        for each e1 in list_points_that_contain_e:
            e1.cost--


// initialization:
    for each point d of D:
        update posting list --> get it's neighbours using threshold search with vicinity threshold
        update inverted index
        set cost(d) = k * posting_list(d).size()    
        insert into point_stream
k = 2
D = {1,2,3,4}
1 = {2,3} c = 2
2 = {1,3} c = 2
3 = {3, 1, 2} c = 3
4 = {4}  c = 2

S = {3,}
3 = {1, 2, 3}

1 = {2,3} , c = 0
2 = {1, 3} , c = 0
4 = {4} , c = 2

S = {3, 1, }

4 = 
2 = {1, 3} , c = 0



// main-loop:
    while (point_stream.size() != 0 && k_coverage_criteria(not satisfied)):
        p --> point_stream.pop()
        S.insert(p)
        update k_coverage_counter(posting_list(p))
        update_cost(posting_list(p))

?? Datastructure for point_stream ?? 

- some sort of trees, where on cost update delete from tree and then reinsert 
OR 
- maintain an array(costs) such that costs[i] = cost of selecting point i
- top - 5 elements x1 > x2 > x3 > x4 > x5 > ....>
select x1 - {x2}
- find the max of the array using binary search (O(logn))
- so worst case the while loop would run for O(n*logn) 
?? think as of it overlap ??
ordered_sets 
|D|*|D - 1|*|D |
tradeoff between number of updates v/s number of iterations for set cover

// cost update based on posting_list
// ?? approximate updates or more efficient utility function ?? 
update_cost(p_list):
    for each point e in p_list:
        list_points_that_contain_e = inverted_index(e)
        for each e1 in list_points_that_contain_e:
            e1.cost--


Balancing Stage:


*/
