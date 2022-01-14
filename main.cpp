#include<iostream>

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
        get it's top-m neighbours 
        update posting list
        update inverted index
        set cost(d) = k * posting_list(d).size()    
        insert into point_stream

// main-loop:
    while (point_stream.size() != 0 && k_coverage_criteria(not satisfied)):
        p --> point_stream.pop()
        S.insert(p)
        update k_coverage_counter(posting_list(p))
        update_cost(posting_list(p))



Balancing Stage:


*/



int main() {
    std::cout << "Hello, World!" << std::endl;

    return 0;
}