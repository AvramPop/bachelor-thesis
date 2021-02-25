from collections import defaultdict


class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)

    def add_edge(self, u, v, w):
        self.graph[u].append((v, w))

    def topological_sort_util(self, v, visited, stack):
        visited[v] = True
        if v in self.graph.keys():
            for node, weight in self.graph[v]:
                if not visited[node]:
                    self.topological_sort_util(node, visited, stack)
        stack.append(v)

    def shortest_path(self, start):

        visited = [False] * self.V
        stack = []

        for i in range(self.V):
            if not visited[i]:
                self.topological_sort_util(start, visited, stack)

        dist = [float("Inf")] * self.V
        dist[start] = 0

        while stack:
            i = stack.pop()
            for node, weight in self.graph[i]:
                if dist[node] > dist[i] + weight:
                    dist[node] = dist[i] + weight

        # Print the calculated shortest distances
        for i in range(self.V):
            print("%d" % dist[i]) if dist[i] != float("Inf") else "Inf",


print("Following are shortest distances from source %d " % s)
g.shortest_path(s)

def max_weight_path_in_dag(similarity_matrix):
    g = Graph(len(similarity_matrix))
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            if similarity_matrix[i][j] != 0:
                g.add_edge(i, j, similarity_matrix * -1)
    return g.shortest_path(0) * -1
