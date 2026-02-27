import numpy as np
from collections import defaultdict


class CoverageCalculator:
    def __init__(self, config, grid_manager):
        self.config = config
        self.grid_manager = grid_manager
        self.coverage_decay = config.COVERAGE_DECAY
        self.global_coverage = defaultdict(float)
        self.coverage_history = []

    def calculate_coverage_gain(self, grid_id, users, drone_positions, access_links):
        """Calculate coverage gain for a specific grid"""
        grid_center = self.grid_manager.grid_centers[grid_id]

        covered_users = 0
        for user in users:
            if user.connected_drone is not None:
                continue  

            # Calculate distance to grid center (assuming UAV at grid center)
            distance = np.sqrt(
                (user.position[0] - grid_center[0]) ** 2 +
                (user.position[1] - grid_center[1]) ** 2
            )

            if distance <= self.config.ACCESS_LINK_RANGE:
                # coverage decreases with distance
                coverage_value = max(0, 1 - distance / self.config.ACCESS_LINK_RANGE)
                covered_users += coverage_value

        return covered_users / max(1, len(users))

    def update_global_coverage(self, time_step, users, drone_positions, access_links):
        """Update global coverage metrics"""
        total_coverage = 0
        covered_users = 0

        for user in users:
            if user.connected_drone is not None:
                covered_users += 1
                # Get distance to connected drone
                drone_pos = drone_positions[user.connected_drone]
                distance = np.sqrt(
                    (user.position[0] - drone_pos[0]) ** 2 +
                    (user.position[1] - drone_pos[1]) ** 2
                )
                # Quality of coverage decreases with distance
                coverage_quality = max(0.1, 1 - distance / self.config.ACCESS_LINK_RANGE)
                total_coverage += coverage_quality

        coverage_percentage = (covered_users / max(1, len(users))) * 100
        avg_coverage_quality = total_coverage / max(1, covered_users)

        # Apply decay to historical coverage
        for grid_id in list(self.global_coverage.keys()):
            self.global_coverage[grid_id] *= self.coverage_decay

        # Record coverage
        self.coverage_history.append({
            'time': time_step,
            'coverage_percentage': coverage_percentage,
            'avg_quality': avg_coverage_quality,
            'covered_users': covered_users
        })

        return coverage_percentage, avg_coverage_quality

    def get_grid_coverage(self, grid_id):
        """Get current coverage value for a grid"""
        return self.global_coverage.get(grid_id, 0)