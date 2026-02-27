
import math
import numpy as np
import networkx as nx


class Calculator:
    def __init__(self, config):
        self.config = config

    def euclidean_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        if len(p1) == 2 and len(p2) == 2:
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def calculate_path_loss_access(self, user_pos, drone_pos):
        """Calculate path loss for access link"""
        x_u, y_u = user_pos
        x_d, y_d, h_d = drone_pos
        delta = math.sqrt((x_d - x_u) ** 2 + (y_d - y_u) ** 2)
        if delta == 0:
            delta = 1e-6
        theta = math.degrees(math.atan(h_d / delta))
        distance = math.sqrt(delta ** 2 + h_d ** 2)
        phi_L = 20 * math.log10(4 * math.pi * self.config.FC * distance / self.config.C) + self.config.ETA_L
        phi_N = 20 * math.log10(4 * math.pi * self.config.FC * distance / self.config.C) + self.config.ETA_N
        p_L = 1 / (1 + self.config.A * math.exp(-self.config.B * (theta - self.config.A)))
        path_loss = p_L * phi_L + (1 - p_L) * phi_N
        return path_loss

    def calculate_path_loss_backhaul(self, drone1, drone2):
        """Calculate path loss for backhaul link"""
        distance = self.euclidean_distance(drone1, drone2)
        if distance == 0:
            distance = 1e-6
        path_loss = 20 * math.log10(4 * math.pi * self.config.FC * distance / self.config.C) + self.config.ETA_L
        return path_loss

    def calculate_snr(self, path_loss_db):
        """Calculate SNR from path loss"""
        path_loss_linear = 10 ** (path_loss_db / 10)
        received_power = self.config.TX_POWER / path_loss_linear
        snr = received_power / self.config.NOISE_POWER
        return 10 * math.log10(snr)

    def calculate_network_efficiency(self, drone_energy, backhaul_links, inter_cluster_links, no_drones):
        """Calculate network efficiency metric"""
        V = [i for i in range(no_drones) if drone_energy[i] > 0]
        N = len(V)
        if N < 2:
            return 0.0
        G = nx.Graph()
        G.add_nodes_from(V)
        edges = backhaul_links + [(min(i, j), max(i, j)) for i, j in inter_cluster_links]
        G.add_edges_from(edges)
        total = 0.0
        for i_idx in range(N):
            for j_idx in range(i_idx + 1, N):
                i_node = V[i_idx]
                j_node = V[j_idx]
                try:
                    d = nx.shortest_path_length(G, i_node, j_node)
                    total += 2 / d
                except nx.NetworkXNoPath:
                    pass
        denominator = N * (N - 1)
        return total / denominator if denominator > 0 else 0.0

    def calculate_cluster_radius(self, cluster_id, cluster_members, drone_positions):
        """Calculate cluster radius"""
        anchor_pos = np.array(drone_positions[cluster_id][:2])
        members = cluster_members[cluster_id]
        max_dist = 0
        for member in members:
            member_pos = np.array(drone_positions[member][:2])
            dist = np.linalg.norm(anchor_pos - member_pos)
            if dist > max_dist:
                max_dist = dist
        return max_dist