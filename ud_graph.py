# Course: CS 261
# Author: Jesse Narkmanee
# Assignment: A6 Graph Implplementation
# Description: Creating an undirected graph using an adjacency list
# with different helper/pathfinding methods. 


class UndirectedGraph:
    """
    Class to implement undirected graph
    - duplicate edges not allowed
    - loops not allowed
    - no edge weights
    - vertex names are strings
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency list
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.adj_list = dict()

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            for u, v in start_edges:
                self.add_edge(u, v)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        out = [f'{v}: {self.adj_list[v]}' for v in self.adj_list]
        out = '\n  '.join(out)
        if len(out) < 70:
            out = out.replace('\n  ', ', ')
            return f'GRAPH: {{{out}}}'
        return f'GRAPH: {{\n  {out}}}'

    # ------------------------------------------------------------------ #

    def add_vertex(self, v: str) -> None:
        """
        We add a list of list for every new vertex added.
        """
        if v not in self.adj_list.keys():
            self.adj_list[v] = list()


    def add_edge(self, u: str, v: str) -> None:
        """
        First we make sure the adjacent vertices exist and 
        that they are not each other. Then we add both vertices
        to each other's adjacency list.
        """
        if u == v:
            return
        if u not in self.adj_list.keys():
            self.add_vertex(u)
        if v not in self.adj_list.keys():
            self.add_vertex(v)
        if u == v or (v in self.adj_list[u]) or (u in self.adj_list[v]):
            return
        # if u in self.adj_list.keys() and (v in self.adj_list[u]):
        #     return
        # if v in self.adj_list.keys() and (u in self.adj_list[v]):
        #     return

        self.adj_list[v].append(u)
        self.adj_list[u].append(v)


    def remove_edge(self, v: str, u: str) -> None:
        """
        If the vertices exist, we remove each vertex from
        each other's adjacency list.
        """
        vert_lst = self.get_vertices()
        if v not in vert_lst or u not in vert_lst:
            return

        v_len = len(self.adj_list[v])
        u_len = len(self.adj_list[u])

        for i in range(v_len):
            if self.adj_list[v][i] == u:
                self.adj_list[v].pop(i)
                break

        for i in range(u_len):
            if self.adj_list[u][i] == v:
                self.adj_list[u].pop(i)
                break

    def remove_vertex(self, v: str) -> None:
        """
        We remove the vertex then all remnant instances in
        every other vertex's adjacency through iteration.
        """
        self.adj_list.pop(v, None)

        for i, val_lst in self.adj_list.items():
            if v in val_lst:
                self.adj_list[i].remove(v)


    def get_vertices(self) -> []:
        """
        Simply return the keys of the adjacency list as a list.
        """
        return list(self.adj_list.keys())

    def get_edges(self) -> []:
        """
        We first iterate through every vertex's adjacency list
        and add every src, dest vertex as a tuple. To elimiate 
        duplicates, we sort the edges so that we can and delete the
        exact duplicates. Then we return what is left in the form of a list.
        """
        edge_lst = []

        for key, val_lst in self.adj_list.items():
            for i in range(len(val_lst)):
                edge_lst.append((key, val_lst[i]))

        return list(set([tuple(sorted(i)) for i in edge_lst]))


    def is_valid_path(self, path: []) -> bool:
        """
        We iterate through the path using a looking glass of two elments
        each. We then see if the second of two elements is in the first
        elements' adjacency list until we reach the end of the path traversal.
        """
        if not path:
            return True
        elif len(path) == 1 and (path[0] not in self.get_vertices()):
            return False
        else:
            for i in range(len(path) - 1):
                src = i
                dest = i + 1
                if path[dest] not in self.adj_list[path[src]]:
                    return False
                    
            return True 
            # print("dfs", self.dfs(path[0], path[-1]))
            # print("dfs", self.dfs(path[0]))
            # print("path", path)

            



    def dfs(self, v_start, v_end=None) -> []:
        """
        Using a stack data structure, we add reachable nodes
        to our reachable list and push successor vertices 
        to the stack as a way to search through depth and pop off
        to add to our reachable list when we have gone the deepest possible.
        """
        if v_start not in self.get_vertices():
            return []
        elif v_end not in self.get_vertices():
            v_end = None

        reachable = []
        stack = [v_start]

        while stack:
            popped_val = stack.pop()
            if popped_val not in reachable:
                reachable.append(popped_val)
            if popped_val == v_end:
                return reachable

            for i in sorted(self.adj_list[popped_val], reverse=True):
                if i not in reachable:
                    stack.append(i)

        return reachable



    def bfs(self, v_start, v_end=None) -> []:
        """
        Using a queue data structure, we add reachable nodes
        to our reachable list and push successor vertices 
        to the queue as a way to search first added adjacent nodes first and pop off
        to add to our reachable list when we have gone through our closest neighboring vertices.
        """
        if v_start not in self.get_vertices():
            return []
        elif v_end not in self.get_vertices():
            v_end = None

        reachable = []
        queue = [v_start]

        while queue:
            popped_val = queue.pop(0)
            if popped_val not in reachable:
                reachable.append(popped_val)
            if popped_val == v_end:
                return reachable

            for i in sorted(self.adj_list[popped_val], reverse=False):
                if i not in reachable:
                    queue.append(i)

        return reachable        

    def count_connected_components(self):
        """
        We run depth-first-search on every vertex and add a reachable list
        to a connected list. We then sort each reachable list so that using
        sets, we can eliminate duplicates. What we have left are all the
        connected graphs possible in a list, therefore we just return the length
        of the list.
        """
        comp_lst = []

        for i in self.get_vertices():
            comp_lst.append(sorted(self.dfs(i)))

        uniq_lsts = [list(i) for i in set(tuple(i) for i in comp_lst)]
        return len(uniq_lsts)




    def has_cycle(self):
        """
        Return True if graph contains a cycle, False otherwise
        """
        # for i in self.get_vertices():
        #     if len(self.dfs(i)) == len(self.get_vertices()):
        #         return True
        
        # return False

        """
        We use DFS to search through the list. When a vertex is encountered twice
        we know for sure that a cycle is in the graph so we return True.
        """
        if len(self.get_vertices()) == 0 or len(self.get_vertices()) == 1:
            return False


        for index in range(len(self.get_vertices())):
            last_vert = self.get_vertices()[index]
            reachable = []
            stack = [(self.get_vertices()[index], last_vert)]    


            while stack:
                popped_val = stack.pop()
                cur_vert = popped_val[0]
                last_vert = popped_val[1]
                if cur_vert not in reachable:
                    reachable.append(cur_vert)

                for i in sorted(self.adj_list[cur_vert], reverse=True):
                    if i not in reachable:
                        stack.append((i, cur_vert))
                    if i in reachable and i != last_vert:
                        return True

        return False

   


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = UndirectedGraph()
    print(g)

    for v in 'ABCDE':
        g.add_vertex(v)
    print(g)

    g.add_vertex('A')
    print(g)

    for u, v in ['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE', ('B', 'C')]:
        g.add_edge(u, v)
    print(g)


    print("\nPDF - method remove_edge() / remove_vertex example 1")
    print("----------------------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    g.remove_vertex('DOES NOT EXIST')
    g.remove_edge('A', 'B')
    g.remove_edge('X', 'B')
    print(g)
    g.remove_vertex('D')
    print(g)


    print("\nPDF - method get_vertices() / get_edges() example 1")
    print("---------------------------------------------------")
    g = UndirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE'])
    print(g.get_edges(), g.get_vertices(), sep='\n')


    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    test_cases = ['ABC', 'ADE', 'ECABDCBE', 'ACDECB', '', 'D', 'Z']
    for path in test_cases:
        print(list(path), g.is_valid_path(list(path)))


    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = 'ABCDEGH'
    for case in test_cases:
        print(f'{case} DFS:{g.dfs(case)} BFS:{g.bfs(case)}')
    print('-----')
    for i in range(1, len(test_cases)):
        v1, v2 = test_cases[i], test_cases[-1 - i]
        print(f'{v1}-{v2} DFS:{g.dfs(v1, v2)} BFS:{g.bfs(v1, v2)}')


    print("\nPDF - method count_connected_components() example 1")
    print("---------------------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print(g.count_connected_components(), end=' ')
    print()


    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG',
        'add FG', 'remove GE')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print('{:<10}'.format(case), g.has_cycle())
