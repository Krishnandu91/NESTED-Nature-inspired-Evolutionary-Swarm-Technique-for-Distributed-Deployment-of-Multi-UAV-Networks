
import numpy as np
class LinkBuilder:
    def __init__(self, config, calculator):
        self.config = config
        self.calculator = calculator
    # Access link (UAV-UAV)
    def build_access_links(self, users, drones, drone_positions, drone_energy):
        """Build access links between users and drones"""
        access_links = []
        drone_xy = np.array([(x, y) for x, y, _ in drone_positions])

        for u_idx, user in enumerate(users):
            user.connected_drone = None
            for d_idx, drone in enumerate(drones):
                if drone_energy[d_idx] == 0:
                    continue
                if self.calculator.euclidean_distance(user.position, drone_xy[d_idx]) > self.config.ACCESS_LINK_RANGE:
                    continue
                pl = self.calculator.calculate_path_loss_access(user.position, drone.position)
                if self.calculator.calculate_snr(pl) >= self.config.ACCESS_SNR_THRESHOLD:
                    access_links.append((u_idx, d_idx))
                    user.connected_drone = d_idx
                    break

        return access_links
    # Backhaul link (UAV-USER)
    def build_backhaul_links(self, drones, drone_positions, drone_energy, no_drones):
        """Build backhaul links between drones"""
        backhaul_links = []
        neighbors = {i: set() for i in range(no_drones)}
        drone_xy = np.array([(x, y) for x, y, _ in drone_positions])

        for i in range(no_drones):
            if drone_energy[i] == 0:
                continue
            for j in range(i + 1, no_drones):
                if drone_energy[j] == 0:
                    continue
                if self.calculator.euclidean_distance(drone_xy[i], drone_xy[j]) > self.config.BACKHAUL_LINK_RANGE:
                    continue
                pl = self.calculator.calculate_path_loss_backhaul(drone_positions[i], drone_positions[j])
                if self.calculator.calculate_snr(pl) >= self.config.BACKHAUL_SNR_THRESHOLD:
                    backhaul_links.append((i, j))
                    neighbors[i].add(j)
                    neighbors[j].add(i)
                    drones[i].neighbors.add(j)
                    drones[j].neighbors.add(i)

        return backhaul_links, neighbors