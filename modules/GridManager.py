import numpy as np
import random
import networkx as nx

class GridManager:
    def __init__(self, config):
        self.config = config
        self.grid_size = config.calculate_grid_parameters()
        self.grid_rows = config.GRID_ROWS
        self.grid_cols = config.GRID_COLS
        self.num_grids = config.NUM_GRIDS

        # Grid information - store boundaries for each grid
        self.grid_centers = self._calculate_grid_info()

        # Grid occupancy - CRITICAL: one UAV per grid
        self.grid_occupancy = {grid_id: None for grid_id in range(self.num_grids)}

        print(f"\nGrid Manager initialized:")
        print(f"  Grid layout: {self.grid_rows}x{self.grid_cols} = {self.num_grids} grids")
        print(f"  Grid size: {self.grid_size:.1f}m")
        print(f"  Field size: {config.FIELD_SIZE / 1000:.1f}km x {config.FIELD_SIZE / 1000:.1f}km")
        print(f"  Each UAV will be in a DIFFERENT grid")

    def _calculate_grid_info(self):
        """Calculate grid boundaries and centers"""
        grid_info = {}
        grid_id = 0

        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                # Grid boundaries
                min_x = j * self.grid_size
                max_x = (j + 1) * self.grid_size
                min_y = i * self.grid_size
                max_y = (i + 1) * self.grid_size

                # Center point
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2

                grid_info[grid_id] = {
                    'min_x': min_x, 'max_x': max_x,
                    'min_y': min_y, 'max_y': max_y,
                    'center_x': center_x, 'center_y': center_y,
                    'row': i, 'col': j
                }
                grid_id += 1

        return grid_info

    def place_uavs_random_one_per_grid(self, num_uavs, calculator):
        """
        Distributed placement: Place all UAVs randomly in different grids first.
        Then, each isolated UAV (no connections) independently jumps smart-random to new grids until connected to at least one.
        """
        print(f"\n=== Distributed Placement of {num_uavs} UAVs (Random Initial, Then Self-Adjust if Isolated) ===")

        if num_uavs > self.num_grids:
            print(f"❌ ERROR: Need {num_uavs} grids but only {self.num_grids} available!")
            return None, None, None

        # Reset occupancy
        self.grid_occupancy = {grid_id: None for grid_id in range(self.num_grids)}

        # Select random grids for initial placement
        available_grids = list(range(self.num_grids))
        random.shuffle(available_grids)
        selected_grids = available_grids[:num_uavs]

        positions = []
        grid_ids = [None] * num_uavs
        margin = self.grid_size * 0.1

        # Initial random placement
        for uav_id in range(num_uavs):
            grid_id = selected_grids[uav_id]
            grid_info = self.grid_centers[grid_id]
            x = random.uniform(grid_info['min_x'] + margin, grid_info['max_x'] - margin)
            y = random.uniform(grid_info['min_y'] + margin, grid_info['max_y'] - margin)
            position = (x, y, self.config.UAV_HEIGHT)
            positions.append(position)
            grid_ids[uav_id] = grid_id
            self.grid_occupancy[grid_id] = uav_id
            print(f"  Initial placement UAV {uav_id} in grid {grid_id} at ({x:.0f}, {y:.0f})")

        # Build initial backhaul links
        backhaul_links = []
        neighbors = {i: set() for i in range(num_uavs)}
        for i in range(num_uavs):
            for j in range(i + 1, num_uavs):
                distance = calculator.euclidean_distance(positions[i][:2], positions[j][:2])
                if distance <= self.config.BACKHAUL_LINK_RANGE:
                    pl = calculator.calculate_path_loss_backhaul(positions[i], positions[j])
                    if calculator.calculate_snr(pl) >= self.config.BACKHAUL_SNR_THRESHOLD:
                        backhaul_links.append((i, j))
                        neighbors[i].add(j)
                        neighbors[j].add(i)

        # Identify isolated UAVs (no neighbors)
        isolated = [i for i in range(num_uavs) if not neighbors[i]]

        print(f"  Initial links: {len(backhaul_links)}, Isolated UAVs: {len(isolated)}/{num_uavs}")

        # For each isolated UAV, perform smart-random jumps until connected
        for uav_id in isolated:
            print(f"    Adjusting isolated UAV {uav_id}...")
            current_grid = grid_ids[uav_id]
            visited = set([current_grid])  # Avoid current if failed
            placed = False
            max_attempts = self.num_grids * 2
            attempt = 0

            while not placed and attempt < max_attempts:
                attempt += 1

                # Get available empty grids not visited
                available = [g for g in range(self.num_grids) if self.grid_occupancy[g] is None and g not in visited]

                if not available:
                    break

                candidate_grid = random.choice(available)
                grid_info = self.grid_centers[candidate_grid]
                x = random.uniform(grid_info['min_x'] + margin, grid_info['max_x'] - margin)
                y = random.uniform(grid_info['min_y'] + margin, grid_info['max_y'] - margin)
                candidate_pos = (x, y, self.config.UAV_HEIGHT)

                # Hypothetical connections: check if can connect to any existing (non-moving) UAV
                can_connect = False
                for other_id in range(num_uavs):
                    if other_id == uav_id:
                        continue
                    dist = calculator.euclidean_distance(candidate_pos, positions[other_id])
                    if dist <= self.config.BACKHAUL_LINK_RANGE:
                        pl = calculator.calculate_path_loss_backhaul(candidate_pos, positions[other_id])
                        if calculator.calculate_snr(pl) >= self.config.BACKHAUL_SNR_THRESHOLD:
                            can_connect = True
                            break

                if can_connect:
                    # Move: vacate current, occupy new, update position and grid
                    self.vacate_grid(current_grid)
                    self.occupy_grid(candidate_grid, uav_id)
                    positions[uav_id] = candidate_pos
                    grid_ids[uav_id] = candidate_grid
                    placed = True
                    print(f"      Moved UAV {uav_id} to grid {candidate_grid} at ({x:.0f}, {y:.0f}) after {attempt} attempts")
                else:
                    visited.add(candidate_grid)

            if not placed:
                print(f"      WARNING: UAV {uav_id} could not connect after {attempt} attempts; keeping initial position")

        # Rebuild all backhaul links after adjustments
        backhaul_links = []
        neighbors = {i: set() for i in range(num_uavs)}
        for i in range(num_uavs):
            for j in range(i + 1, num_uavs):
                distance = calculator.euclidean_distance(positions[i][:2], positions[j][:2])
                if distance <= self.config.BACKHAUL_LINK_RANGE:
                    pl = calculator.calculate_path_loss_backhaul(positions[i], positions[j])
                    if calculator.calculate_snr(pl) >= self.config.BACKHAUL_SNR_THRESHOLD:
                        backhaul_links.append((i, j))
                        neighbors[i].add(j)
                        neighbors[j].add(i)

        # Verify connectivity
        G = nx.Graph()
        G.add_nodes_from(range(num_uavs))
        G.add_edges_from(backhaul_links)
        if nx.is_connected(G):
            print("✓ All UAVs connected after adjustments!")
        else:
            print(f"⚠ {nx.number_connected_components(G)} components remain")

        print(f"✓ Final backhaul links: {len(backhaul_links)}")

        return positions, grid_ids, backhaul_links

    def position_to_grid_id(self, position):
        """Convert (x,y) position to grid ID"""
        x, y = position[:2]
        col = min(int(x // self.grid_size), self.grid_cols - 1)
        row = min(int(y // self.grid_size), self.grid_rows - 1)

        grid_id = row * self.grid_cols + col
        return grid_id

    def grid_id_to_position(self, grid_id, height=None, random_offset=True):
        """Convert grid ID to position WITHIN that grid"""
        if grid_id is None or grid_id < 0 or grid_id >= self.num_grids:
            x = random.uniform(0, self.config.FIELD_SIZE)
            y = random.uniform(0, self.config.FIELD_SIZE)
        else:
            grid_info = self.grid_centers[grid_id]

            if random_offset:
                margin = self.grid_size * 0.1
                x = random.uniform(grid_info['min_x'] + margin, grid_info['max_x'] - margin)
                y = random.uniform(grid_info['min_y'] + margin, grid_info['max_y'] - margin)
            else:
                x = grid_info['center_x']
                y = grid_info['center_y']

        if height is None:
            height = self.config.UAV_HEIGHT

        return (x, y, height)

    def is_grid_occupied(self, grid_id):
        if grid_id not in self.grid_occupancy:
            return False
        return self.grid_occupancy[grid_id] is not None

    def occupy_grid(self, grid_id, uav_id):
        if 0 <= grid_id < self.num_grids:
            if not self.is_grid_occupied(grid_id):
                self.grid_occupancy[grid_id] = uav_id
                return True
        return False

    def vacate_grid(self, grid_id):
        if grid_id in self.grid_occupancy:
            self.grid_occupancy[grid_id] = None
            return True
        return False

    def get_available_grids(self):
        available = []
        for grid_id in range(self.num_grids):
            if not self.is_grid_occupied(grid_id):
                available.append(grid_id)
        return available

    def get_neighboring_grids(self, grid_id, radius=1):
        if grid_id < 0 or grid_id >= self.num_grids:
            return []

        row = grid_id // self.grid_cols
        col = grid_id % self.grid_cols

        neighbors = []
        for i in range(max(0, row - radius), min(self.grid_rows, row + radius + 1)):
            for j in range(max(0, col - radius), min(self.grid_cols, col + radius + 1)):
                neighbor_id = i * self.grid_cols + j
                if neighbor_id != grid_id:
                    neighbors.append(neighbor_id)

        return neighbors

    def get_grid_distance(self, grid1, grid2):
        row1 = grid1 // self.grid_cols
        col1 = grid1 % self.grid_cols
        row2 = grid2 // self.grid_cols
        col2 = grid2 % self.grid_cols

        return max(abs(row1 - row2), abs(col1 - col2))

    # NEW: Method for a single UAV to jump smart-random until connected
    def adjust_uav_position(self, uav_id, positions, calculator, backhaul_links, neighbors):
        current_grid = [g for g, uid in self.grid_occupancy.items() if uid == uav_id][0]
        visited = set([current_grid])
        placed = False
        max_attempts = self.num_grids * 2
        attempt = 0

        while not placed and attempt < max_attempts:
            attempt += 1
            available = self.get_available_grids()
            candidates = [g for g in available if g not in visited]
            if not candidates:
                break

            candidate_grid = random.choice(candidates)
            margin = self.grid_size * 0.1
            grid_info = self.grid_centers[candidate_grid]
            x = random.uniform(grid_info['min_x'] + margin, grid_info['max_x'] - margin)
            y = random.uniform(grid_info['min_y'] + margin, grid_info['max_y'] - margin)
            candidate_pos = (x, y, self.config.UAV_HEIGHT)

            # Check hypothetical connection
            can_connect = False
            for other_id in range(len(positions)):
                if other_id == uav_id:
                    continue
                dist = calculator.euclidean_distance(candidate_pos, positions[other_id])
                if dist <= self.config.BACKHAUL_LINK_RANGE:
                    pl = calculator.calculate_path_loss_backhaul(candidate_pos, positions[other_id])
                    if calculator.calculate_snr(pl) >= self.config.BACKHAUL_SNR_THRESHOLD:
                        can_connect = True
                        break

            if can_connect:
                self.vacate_grid(current_grid)
                self.occupy_grid(candidate_grid, uav_id)
                positions[uav_id] = candidate_pos
                placed = True
            else:
                visited.add(candidate_grid)

        return placed