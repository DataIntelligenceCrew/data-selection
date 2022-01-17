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



int main() {
    std::cout << "Hello, World!" << std::endl;

    return 0;
}