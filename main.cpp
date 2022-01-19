#include <iostream>
#include <omp.h>
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/utils/distances.h>
#include <faiss/index_io.h>
#include <stdlib.h>
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

using idx_t = faiss::Index::idx_t;

int main() {
    std::cout << "Hello, World!" << std::endl;
    // problem linking library
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


    return 0;
}