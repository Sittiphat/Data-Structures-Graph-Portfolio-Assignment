import heapq
from collections import deque

# Course: CS 261
# Author: Jesse Narkmanee
# Assignment: A6 Graph Implplementation
# Description: Creating an directed graph using an adjacency matrix
# with different helper/pathfinding methods. 

class DirectedGraph:
    """
    Class to implement directed weighted graph
    - duplicate edges not allowed
    - loops not allowed
    - only positive edge weights
    - vertex names are integers
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency matrix
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.v_count = 0
        self.adj_matrix = []

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            v_count = 0
            for u, v, _ in start_edges:
                v_count = max(v_count, u, v)
            for _ in range(v_count + 1):
                self.add_vertex()
            for u, v, weight in start_edges:
                self.add_edge(u, v, weight)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        if self.v_count == 0:
            return 'EMPTY GRAPH\n'
        out = '   |'
        out += ' '.join(['{:2}'.format(i) for i in range(self.v_count)]) + '\n'
        out += '-' * (self.v_count * 3 + 3) + '\n'
        for i in range(self.v_count):
            row = self.adj_matrix[i]
            out += '{:2} |'.format(i)
            out += ' '.join(['{:2}'.format(w) for w in row]) + '\n'
        out = f"GRAPH ({self.v_count} vertices):\n{out}"
        return out

    # ------------------------------------------------------------------ #

    def add_vertex(self) -> int:
        """
        We first make our own 2d array based on the new size of the 
        vertex count after adding a vertex. Once we do that, we copy the original 
        adj_matrix to our new and bigger adj_matrix which leaves the newly
        added vertex at zero weight.
        """
        self.v_count += 1
        rows, cols = (self.v_count, self.v_count)
        arr = [[0 for i in range(cols)] for j in range(rows)]

        if not self.adj_matrix:
            self.adj_matrix = arr
        else:
            for i in range(rows - 1):
                for j in range(cols - 1):
                    arr[i][j] = self.adj_matrix[i][j]

            self.adj_matrix = arr

        return self.v_count

    def add_edge(self, src: int, dst: int, weight=1) -> None:
        """
        Our source and destinations are the coordinates on the
        adjacency matrix. So just add its weight.
        """
        if src == dst or src < 0 or dst < 0 or src >= self.v_count or dst >= self.v_count:
            return 

        self.adj_matrix[src][dst] = weight

    def remove_edge(self, src: int, dst: int) -> None:
        """
        Our source and destinations are the coordinates on the
        adjacency matrix. So just add zero.
        """
        if src == dst or src < 0 or dst < 0 or src >= self.v_count or dst >= self.v_count:
            return 

        self.adj_matrix[src][dst] = 0

    def get_vertices(self) -> []:
        """
        We just return the range of the number of vertices as a list.
        """
        return list(range(self.v_count))

    def get_edges(self) -> []:
        """
        We have to traverse through the entire matrix
        and add the elements to our return list
        when the weight is greater than 0.
        """
        edge_lst = []
        for i in range(self.v_count):
            for j in range(self.v_count):
                weight = self.adj_matrix[i][j]
                if weight > 0:
                    edge_lst.append((i, j, weight))

        return edge_lst


    def is_valid_path(self, path: []) -> bool:
        """
        We iterate through the path using a looking glass of two elments
        each. We look at where the src,dest index is pointing towards
        and return false if at least one of them is 0 or less than 0.
        If all of them are greater than 0 we eventually return true.
        """
        if not path:
            return True
        elif len(path) == 1 and (path[0] not in self.get_vertices()):
            return False
        else:
            for i in range(len(path) - 1):
                src = i
                dest = i + 1
                if self.adj_matrix[path[src]][path[dest]] <= 0:
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

            i = popped_val
            for j in reversed(range(self.v_count)):
                if j not in reachable and self.adj_matrix[i][j] > 0:
                    stack.append(j)

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

            i = popped_val
            for j in range(self.v_count):
                if j not in reachable and self.adj_matrix[i][j] > 0:
                    queue.append(j)

        return reachable

    def has_cycle(self):
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

                i = cur_vert
                # print("lv", last_vert)
                for j in reversed(range(self.v_count)):
                    if j not in reachable and self.adj_matrix[i][j] > 0:
                        stack.append((j, cur_vert))
                    if j in reachable and self.adj_matrix[i][j] > 0:
                        if i in self.bfs(j):
                            # print("i", i)
                            # print("j", j)
                            # print("reachable", reachable)
                            # print("last_vert", last_vert)
                            return True
        return False

    def dijkstra(self, src: int) -> []:
        """
        TODO: Write this implementation
        """
        if src not in self.get_vertices():
            return []

        # reachable = []
        # stack = [v_start]

        # visited = {i: float('inf') for i in self.get_vertices()}
        visited = {}
        heap = []
        distances = {vertex: float('inf') for vertex in self.get_vertices()}
        heapq.heappush(heap, (0, src))

        while heap:
            popped_val = heapq.heappop(heap)
            dist = popped_val[0]
            pop_vert = popped_val[1]
            
            if pop_vert not in visited.keys():
                visited[pop_vert] = dist


            i = pop_vert
            for j in reversed(range(self.v_count)):
                d_i = self.adj_matrix[i][j]
                total_dist = dist + d_i
                if self.adj_matrix[i][j] > 0 and j not in visited.keys() and total_dist < distances[j]:
                    distances[j] = total_dist
                    heapq.heappush(heap, (total_dist, j))
                    # print(heap)
            # print("For Loop end")

        # print(visited)
        # return [(i, visited[i]) for i in sorted(visited.keys())]
        # return [visited[i] for i in sorted(visited.keys())]
        # return [(i, visited[i]) if i in visited.keys() else (i, float('inf')) for i in self.get_vertices()]
        return [visited[i] if i in visited.keys() else float('inf') for i in self.get_vertices()]


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = DirectedGraph()
    print(g)
    for _ in range(5):
        g.add_vertex()
    print(g)

    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    for src, dst, weight in edges:
        g.add_edge(src, dst, weight)
    print(g)


    print("\nPDF - method get_edges() example 1")
    print("----------------------------------")
    g = DirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    print(g.get_edges(), g.get_vertices(), sep='\n')


    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    test_cases = [[0, 1, 4, 3], [1, 3, 2, 1], [0, 4], [4, 0], [], [2]]
    for path in test_cases:
        print(path, g.is_valid_path(path))


    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for start in range(5):
        print(f'{start} DFS:{g.dfs(start)} BFS:{g.bfs(start)}')


    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)

    edges_to_remove = [(3, 1), (4, 0), (3, 2)]
    for src, dst in edges_to_remove:
        g.remove_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')

    edges_to_add = [(4, 3), (2, 3), (1, 3), (4, 0)]
    for src, dst in edges_to_add:
        g.add_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')
    print('\n', g)


    print("\nPDF - dijkstra() example 1")
    print("--------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
    g.remove_edge(4, 3)
    print('\n', g)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
