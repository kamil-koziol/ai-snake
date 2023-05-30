from __future__ import annotations
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class AStarNode:
    f_cost: int
    g_cost: int
    value: int
    is_start: bool = False
    is_target: bool = False
    is_wall: bool = False
    parent: Optional[AStarNode] = None
    position: Tuple[int, int] = (0, 0)

class AStar:
    open_nodes: List[AStarNode]
    closed_nodes: List[AStarNode]
    nodes: List[List[AStarNode]]
    shape: Tuple[int, int]

    def __init__(self, shape: Tuple[int, int]) -> None:
        self.open_nodes = []
        self.closed_nodes = []
        self.nodes = []
        self.shape = shape

        # Initialize the nodes grid
        for i in range(shape[0]):
            row = []
            for j in range(shape[1]):
                node = AStarNode(f_cost=0, g_cost=0, value=0, position=(i, j))
                row.append(node)
            self.nodes.append(row)

    def set_wall(self, position: Tuple[int, int]) -> None:
        x, y = position
        if self.is_valid_position(x, y):
            node = self.nodes[x][y]
            node.is_wall = True

    def calculate_heuristic(self, current_node: AStarNode, target_node: AStarNode) -> int:
        # Calculate the Manhattan distance as the heuristic
        return abs(target_node.position[0] - current_node.position[0]) + abs(target_node.position[1] - current_node.position[1])

    def generate_neighbors(self, node: AStarNode) -> List[AStarNode]:
        neighbors = []
        x, y = node.position

        deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            if self.is_valid_position(nx, ny):
                neighbor = self.nodes[nx][ny]
                neighbors.append(neighbor)

        return neighbors

    def is_valid_position(self, x: int, y: int) -> bool:
        return 0 <= x < self.shape[0] and 0 <= y < self.shape[1]

    def calculate_distance(self, node1: AStarNode, node2: AStarNode) -> int:
        return 1

    def reconstruct_path(self, target_node: AStarNode) -> List[Tuple[int, int]]:
        path = []
        current_node = target_node

        while current_node is not None:
            path.append(current_node.position)
            current_node = current_node.parent

        path.reverse()
        return path

    def find_path(self, start_position: Tuple[int, int], target_position: Tuple[int, int], longest: bool = False) -> Optional[List[Tuple[int, int]]]:
        start_node = self.nodes[start_position[0]][start_position[1]]
        target_node = self.nodes[target_position[0]][target_position[1]]

        self.open_nodes.append(start_node)

        while self.open_nodes:
            if longest:
                current_node = max(self.open_nodes, key=lambda node: node.f_cost)
            else:
                current_node = min(self.open_nodes, key=lambda node: node.f_cost)

            self.open_nodes.remove(current_node)
            self.closed_nodes.append(current_node)

            if current_node is target_node:
                path = self.reconstruct_path(current_node)
                return path

            neighbors = self.generate_neighbors(current_node)

            for neighbor in neighbors:
                if neighbor in self.closed_nodes or neighbor.is_wall:
                    continue

                g_cost = current_node.g_cost + self.calculate_distance(current_node, neighbor)
                if neighbor not in self.open_nodes:
                    self.open_nodes.append(neighbor)
                elif g_cost >= neighbor.g_cost:
                    continue

                neighbor.g_cost = g_cost
                neighbor.f_cost = g_cost + self.calculate_heuristic(neighbor, target_node)
                neighbor.parent = current_node

        return None

