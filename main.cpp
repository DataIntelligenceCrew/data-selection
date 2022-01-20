#include <iostream>
#include <omp.h>
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/utils/distances.h>
#include <faiss/index_io.h>
#include <stdlib.h>
#include <sqlite3.h>


/*
    we have a SQL database with the following for an image : name/id, label, feature_vector
    create nameToID set
    create vec_array indexed by ID 
    create label_array indexed by ID
    after faiss search create posting_list and inverted_index
*/
using namespace std;
using idx_t = faiss::Index::idx_t;
faiss::gpu::StandardGpuResources res; // global GPU resource


/**********************
 * Sqlite3 Instance
***********************/
class Database {
    private:
        sqlite3 *db; // f_vector database instance
        sqlite3 *dest; // database location
        std::unordered_map<int, string> int2word; // integer_ID --> image name
        std::unordered_map<string, int> word2int; // image name --> integer_ID
        std::unordered_map<int, vector<float>> int2vector; // integer_ID --> feature vector
        std::unordered_map<int, int> int2label; // integer_ID --> image label


        // convert mysql entry from bytes to vector
        vector<float> bytes_to_vec(const unsigned char *data, size_t dimension) {
            vector<float> result;
            for (size_t i = 0; i < dimension * 4; i = i + 4) {
                float f;
                unsigned char b[] = {data[i], data[i+1], data[i+2], data[i+3]};
                memcpy(&f, &b, sizeof(f));
                result.push_back(f);
            }
            return result;
        }
    
    public:
        Database(string path) {
            cout << path.c_str() << endl;
            int rc = sqlite3_open(path.c_str(), &dest);
            
            if (rc) {
                cout << "Cannot open database: " << sqlite3_errmsg(dest) << endl;
                exit(0);
            } else {
                cout << "Successfully opened sqlite3 database" << endl;
            }

            // open the database in-memory for better performance
            rc = sqlite3_open(":memory:", &db);
			if (rc) {
				cout << "Cannot open in-memory database:  " << sqlite3_errmsg(db) << endl; 
				exit(0);
			} else {
				cout << "Successfully opened in-memory database" << endl;
			}

			sqlite3_backup *pBackup;
			pBackup = sqlite3_backup_init(db, "main", dest, "main");
			if(pBackup){
				sqlite3_backup_step(pBackup, -1);
				sqlite3_backup_finish(pBackup);
			}
			rc = sqlite3_errcode(db);
			if (rc) {
				cout << "Cannot copy database:  " << sqlite3_errmsg(db) << endl; 
				exit(0);
			} else {
				cout << "Successfully copied to memory" << endl;
			}
			sqlite3_close(dest);
        }

        void terminate() {
            sqlite3_close(db);
            cout << "Successfully terminated sqlite3 database" << endl;
        }

        vector<vector<float>> get_all_vectors() {
            vector<vector<float>> result;
            int rc;
            stringstream ss;
            
            ss << "SELECT name, label, f_vector FROM fv;";
            string query = ss.str();
            sqlite3_stmt *stmt = NULL;
            
            rc = sqlite3_prepare(db, query.c_str(), -1, &stmt, NULL);
            
            if (rc != SQLITE_OK) {
                cout << "SELECT failed: " << sqlite3_errmsg(db) << endl;
            }
            int nid = 0; 
            while((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
                const char *word = (char*)sqlite3_column_text(stmt, 0);
                int label_id = sqlite3_column_int(stmt, 1); // need to figure this out
				unsigned char *bytes = (unsigned char*)sqlite3_column_blob(stmt, 2);
                string str_t(word);
                vector<float> vector_temp = bytes_to_vec(bytes, 300);
                int2word[nid] = str_t;
                word2int[str_t] = nid;
                int2vector[nid] = vector_temp;
                // int2label[nid] = label_id; // need to figure this out
                result.push_back(vector_temp);
                nid += 1;
            }
            cout << "Dataset Size: " << nid << endl;
            return result;
        }
};

/***************************
 * FAISS Index (GPU Wrapper)
****************************/
class FaissIndexGPU {
    private:
        faiss::gpu::GpuIndexFlatIP *index;
        vector<int> dictionary;
    
    public:
        FaissIndexGPU(string path, Database *db) {
            
            string indexpath = path + "faiss.index";
            int dimension = 300;
            
            vector<vector<float>> vectors = db->get_all_vectors();
            index = new faiss::gpu::GpuIndexFlatIP(&res, dimension);
            
            int nb = vectors.size();
            float *xb = new float[dimension * nb];
            for (int i = 0; i < nb; i++) {
                for (int j = 0; j < dimension; j++) {
                    xb[d * i + j] = vectors[i][j];
                }
            }

            // faiss::fvec_renorm_L2(dimension, nb, xb); // do we need this?
            index->add(nb, xb);
            cout << "Successfully Initialized FAISS Index" << endl;
        }

        // toDo : implement this
        // tuple<vector<idx_t>, vector<float>> range_search(int nq, vector<float> vxq, double threshold) {}

        void destroy() {
            index = NULL;
            cout << "Destroyed FAISS Index" << endl;
        }
};







int main(int argc, char const *argv[]) {
    // double vicinity_threshold = 0.0;
    // int coverage_threshold = 0;
    // string count_requirement_filepath = "";

    // if (argc < 3) {
    //     cerr << "Requires 3 arguments. Usage: ./main vicinity_threshold(double) coverage_threshold(int) filepath_for_group_count_requirements(string)" << endl;
    //     return 1;
    // }
    std::cout << "FAISS test" << std::endl;
    int d = 32;
    int nb = 1000;
    int nq = 100;
    float *xb = new float[d * nb];
    float *xq = new float[d * nq];
    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < d; j++) xb[d * i + j] = drand48();
        xb[d * i] += i / 1000.;
    }
    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++) xq[d * i + j] = drand48();
        xq[d * i] += i / 1000.;
    }

    faiss::IndexFlatL2 index(d);
    printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    index.add(nb, xb);                     // add vectors to the index
    printf("ntotal = %ld\n", index.ntotal);

    int k = 10;
    // sanity check: search 5 first vectors of xb
    idx_t *I = new idx_t[k * 5];
    float *D = new float[k * 5];
    index.search(5, xb, k, D, I);
    printf("I=\n");
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < k; j++) printf("%5ld ", I[i * k + j]);
        printf("\n");
    }
    delete [] I;
    delete [] D;

    // testing sqlite3
    Database *ftdb = new Database("/localdisk1/sematic-overlap-cpp/ft.sqlite3");
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

?? Datastructure for point_stream ?? some sort of trees, where on cost udpate delete from tree and then reinsert 
- maintain an array(costs) such that costs[i] = cost of point i
- find the max of the array using binary search (O(logn))
- so worst case the while loop would run for O(n*logn) 
?? think as of it overlap ??
ordered_sets 
|D|*|D - 1|*|D |

// cost update based on posting_list
// ?? approximate updates or more efficient utility function ?? 
update_cost(p_list):
    for each point e in p_list:
        list_points_that_contain_e = inverted_index(e)
        for each e1 in list_points_that_contain_e:
            e1.cost--


Balancing Stage:


*/