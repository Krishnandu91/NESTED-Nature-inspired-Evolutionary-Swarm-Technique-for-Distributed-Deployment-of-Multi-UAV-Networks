import numpy as np
from collections import defaultdict
import random
import csv
import os
from datetime import datetime
from modules.energy import EnergyModel


class DistributedAlgorithm:
    
    
    def __init__(self, config, grid_manager, coverage_calc, energy_logger=None):
        self.config = config
        self.grid_manager = grid_manager
        self.coverage_calc = coverage_calc
        self.energy_logger = energy_logger

        # Message passing tracking
        self.messages_sent = 0
        self.messages_received = 0

        # Movement tracking
        self.last_movement_time = -10
        self.total_movements = 0
        
        # Death tracking
        self.death_warnings = {}  # {uav_id: [neighbors_notified]}
        self.dead_uavs = set()

        self.uavs_initialized = set()  # Track UAVs that have exchanged components

        print(f"Distributed Algorithm initialized:")
        print(f"  Coverage weight: {config.WE_COVERAGE}")
        print(f"  Exploration rate: {config.EXPLORATION_RATE}")
        print(f"  Energy logging: {'Enabled' if energy_logger else 'Disabled'}")
        
    def consume_movement_energy(self, uav, old_position, new_position, time_step):
        """
        Calculate and consume energy for UAV movement
        Updated to use realistic processing power from paper
        """
        if uav.failed:
            return 0

        # Update velocity
        velocity = uav.update_velocity(new_position, time_step)

        # Calculate delta velocity (acceleration)
        delta_velocity = velocity

        # Calculate transmit power (using current TX_POWER from config)
        tx_power = self.config.TX_POWER

        # Calculate total energy consumption (UPDATED)
        # Now uses realistic processing power (2.0 W from Table 6)
        energy_consumed = EnergyModel.total_energy(
            velocity, delta_velocity, tx_power, message_type='data'
        )

        # Consume the energy
        died = uav.consume_energy(energy_consumed)

        # Log the consumption
        if self.energy_logger:
            # Get detailed energy breakdown
            energy_breakdown = EnergyModel.get_energy_breakdown(
                velocity, delta_velocity, tx_power, message_type='data'
            )

            self.energy_logger.log_energy_consumption(time_step, uav, energy_breakdown)

        return energy_consumed

    def consume_hovering_energy(self, uav, time_step):
        """
        Calculate and consume energy for hovering (stationary)
        Updated to use realistic processing power from paper
        """
        if uav.failed:
            return 0

        # Hovering: zero velocity
        velocity = np.array([0.0, 0.0, 0.0])
        delta_velocity = np.array([0.0, 0.0, 0.0])
        tx_power = self.config.TX_POWER

        # Calculate energy consumption (UPDATED)
        # Now uses realistic processing power (2.0 W from Table 6)
        energy_consumed = EnergyModel.total_energy(
            velocity, delta_velocity, tx_power, message_type='control'
        )

        died = uav.consume_energy(energy_consumed)

        # Log the consumption
        if self.energy_logger:
            energy_breakdown = {
                'hovering_power': EnergyModel.hovering_power(),
                'kinetic_power': 0,
                'flying_power': EnergyModel.hovering_power(),
                'tx_power': tx_power,
                'processing_power': EnergyModel.processing_power('control'),  # Now 2.0 W
                'message_cost': 0,
                'total_consumption': energy_consumed
            }
            self.energy_logger.log_energy_consumption(time_step, uav, energy_breakdown)

        return energy_consumed
    
    
    
    def consume_message_energy(self, uav, time_step, message_type='control'):
        """
        Consume energy for message passing with realistic packet-size-based calculation
        Uses parameters from paper (Table 6) for realistic energy modeling
        """
        if uav.failed:
            return 0

        message_cost = self.config.calculate_message_energy_cost(message_type)

        died = uav.consume_energy(message_cost)

        if died:
            print(f" [Energy] UAV {uav.id} died while sending {message_type} message")
    
        return message_cost
    
    def handle_death_warnings(self, uavs, time_step):
        """
        Handle UAVs sending death warnings to neighbors
        Updated to use realistic message energy costs
        """
        death_warnings_sent = []
        
        for uav in uavs:
            if uav.send_death_warning():
                # UAV is dying and needs to send warning
                neighbors_to_notify = list(uav.neighbors)
                
                for neighbor_id in neighbors_to_notify:
                    if neighbor_id < len(uavs) and not uavs[neighbor_id].failed:
                        # Log the death warning message
                        if self.energy_logger:
                            self.energy_logger.log_message(
                                time_step, uav.id, neighbor_id, 'DEATH_WARNING',
                                uav.energy, uavs[neighbor_id].energy,
                                f"UAV {uav.id} dying, energy: {uav.energy:.2f}J"
                            )
                        
                        # Consume energy for sending message (UPDATED)
                        # Now uses realistic energy based on packet size
                        self.consume_message_energy(uav, time_step, message_type='DEATH_WARNING')
                
                self.death_warnings[uav.id] = neighbors_to_notify
                death_warnings_sent.append((uav.id, neighbors_to_notify))
                
                print(f"  [Death Warning] UAV {uav.id} notified {len(neighbors_to_notify)} neighbors")
        
        return death_warnings_sent
    
    def detect_deaths(self, uavs, time_step):
        """Detect and handle UAV deaths"""
        newly_dead = []
        
        for uav in uavs:
            if uav.failed and uav.id not in self.dead_uavs:
                # This is a newly dead UAV
                self.dead_uavs.add(uav.id)
                newly_dead.append(uav.id)
                
                # Log the death event
                if self.energy_logger:
                    neighbors_notified = self.death_warnings.get(uav.id, [])
                    self.energy_logger.log_death_event(
                        time_step, uav.id, 'ENERGY_DEPLETION',
                        uav.energy, neighbors_notified, replacement_needed=True
                    )
                
                print(f"  [Death Detected] UAV {uav.id} confirmed dead at step {time_step}")
        
        return newly_dead
    
    

    def log_uav_decisions(self, time_step, uavs, drone_positions, candidate_scores, aggregated_scores, movements):
        """Log detailed UAV decisions, locations, scores, and movements"""
        log_dir = "uav_logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(log_dir, f"uav_decisions_{timestamp}.csv")

        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'time_step', 'uav_id', 'current_grid', 'current_pos_x', 'current_pos_y', 'current_pos_z',
                    'candidate_grid', 'score', 'coverage_gain', 'grid_type',
                    'chosen_grid', 'movement_made', 'exploration_triggered'
                ])

            for uav_id in candidate_scores:
                uav = uavs[uav_id]
                current_grid = uav.current_grid
                current_pos = drone_positions[uav_id]
                chosen_grid = max(aggregated_scores[uav_id], key=aggregated_scores[uav_id].get) if uav_id in aggregated_scores else None
                movement_made = any(m[0] == uav_id for m in movements)
                exploration = 'Yes' if random.random() < self.config.EXPLORATION_RATE else 'No'

                for grid_id, score_info in candidate_scores[uav_id].items():
                    writer.writerow([
                        time_step,
                        uav_id,
                        current_grid,
                        current_pos[0],
                        current_pos[1],
                        current_pos[2],
                        grid_id,
                        score_info.get('score', 0),
                        score_info.get('coverage_gain', 0),
                        score_info.get('type', 'unknown'),
                        chosen_grid if grid_id == chosen_grid else '',
                        'Yes' if movement_made else 'No',
                        exploration
                    ])

        print(f"UAV decisions logged to {filename}")


    def run_iteration(self, time_step, uavs, users, drone_positions,
                      backhaul_links, neighbors):
        print(f"\n  [Algorithm] Starting iteration at time_step {time_step}")
        
        # 1. Handle death warnings first
        death_warnings = self.handle_death_warnings(uavs, time_step)
        
        # 2. Detect newly dead UAVs
        newly_dead = self.detect_deaths(uavs, time_step)

        # Check for newly added UAVs (those not yet initialized)
        new_uavs = [uav.id for uav in uavs if uav.id not in self.uavs_initialized and not uav.failed]
        if new_uavs:
            self._exchange_component_information(uavs, time_step, new_uavs)
            self.uavs_initialized.update(new_uavs)

        # 3. Calculate coverage
        uncovered_users = [u for u in users if u.connected_drone is None]
        uncovered_count = len(uncovered_users)

        if uncovered_count > 0:
            avg_x = np.mean([u.position[0] for u in uncovered_users])
            avg_y = np.mean([u.position[1] for u in uncovered_users])
            user_hotspot = (avg_x, avg_y)
            print(f"  [Users] Uncovered: {uncovered_count}, Hotspot at ({avg_x:.0f}, {avg_y:.0f})")
        else:
            user_hotspot = None
            print(f"  [Users] All users covered!")


        # 5. Exchange coverage maps (with energy cost)
        self._exchange_coverage_maps(uavs, time_step)

        # 6. Update each UAV's covered users count
        for uav in uavs:
            if not uav.failed:
                uav.covered_access_links = sum(1 for user in users if user.connected_drone == uav.id)

        # 7. Compute candidate scores
        candidate_scores = self._compute_candidate_scores_improved(
            uavs, users, drone_positions, backhaul_links, neighbors,
            time_step, user_hotspot
        )

        # 8. Aggregate scores
        aggregated_scores = self._aggregate_scores(candidate_scores, uavs, neighbors, time_step)

        # 9. Update movements (with energy consumption)
        movements = self._update_movements_with_energy(
            aggregated_scores, uavs, drone_positions, time_step,
            uncovered_count
        )
        
        # 10. Consume hovering energy for UAVs that didn't move
        moved_uav_ids = set(m[0] for m in movements)
        for uav in uavs:
            if uav.id not in moved_uav_ids and not uav.failed:
                self.consume_hovering_energy(uav, time_step)


        print(f"  [Algorithm] Completed - {len(movements)} movements, "
              f"{uncovered_count} uncovered users, {len(newly_dead)} deaths")

        return movements
    
    def _exchange_coverage_maps(self, uavs, time_step):
        """
        Exchange coverage maps between neighbors (with realistic energy cost)
        FIXED: Now logs COVERAGE_MAP messages properly
        """
        for uav in uavs:
            if not uav.connected or len(uav.neighbors) == 0 or uav.failed:
                continue
            
            for grid_id, coverage_value in uav.coverage_map.items():
                timestamp = uav.timestamps.get(grid_id, -1)

                for neighbor_id in uav.neighbors:
                    if neighbor_id < len(uavs) and not uavs[neighbor_id].failed:
                        neighbor_uav = uavs[neighbor_id]

                        # Update neighbor's coverage map
                        neighbor_uav.update_coverage_map(
                            grid_id, coverage_value, timestamp
                        )

                        # Share distributed knowledge AND energy status
                        neighbor_uav.update_knowledge(
                            uav.id, 
                            uav.known_connected_ids,
                            uav.covered_access_links,
                            uav.energy
                        )

                        # ===== LOG COVERAGE_MAP MESSAGE =====
                        if self.energy_logger:
                            self.energy_logger.log_message(
                                time_step, uav.id, neighbor_id, 'COVERAGE_MAP',
                                uav.energy, neighbor_uav.energy,
                                {'coverage_grids': len(uav.coverage_map)}
                            )

                        # Consume energy for message exchange
                        self.consume_message_energy(uav, time_step, message_type='COVERAGE_MAP')
                        self.consume_message_energy(neighbor_uav, time_step, message_type='COVERAGE_MAP')

                        self.messages_sent += 1
                        self.messages_received += 1

    def _exchange_component_information(self, uavs, time_step, new_uavs):
        """
        Exchange component/system information when NEW UAVs join the system
        CORRECTED: Only runs when new UAVs are added, not every timestep
        """
        for uav_id in new_uavs:
            uav = uavs[uav_id]
            
            if not uav.connected or len(uav.neighbors) == 0 or uav.failed:
                continue
            
            # New UAV queries neighbors about system state
            for neighbor_id in uav.neighbors:
                if neighbor_id < len(uavs) and not uavs[neighbor_id].failed:
                    neighbor_uav = uavs[neighbor_id]
                    
                    # ===== LOG COMPONENT_QUERY MESSAGE (only on UAV addition) =====
                    if self.energy_logger:
                        self.energy_logger.log_message(
                            time_step, uav.id, neighbor_id, 'COMPONENT_QUERY',
                            uav.energy, neighbor_uav.energy,
                            {'query_type': 'system_state', 'component': 'initialization'}
                        )
                    
                    # Consume energy for component query
                    self.consume_message_energy(uav, time_step, message_type='COMPONENT_QUERY')
                    
                    self.messages_sent += 1
                    
                    # ===== LOG COMPONENT_RESPONSE MESSAGE (only on UAV addition) =====
                    if self.energy_logger:
                        self.energy_logger.log_message(
                            time_step, neighbor_id, uav.id, 'COMPONENT_RESPONSE',
                            neighbor_uav.energy, uav.energy,
                            {'response_type': 'system_state', 'component': 'initialization'}
                        )
                    
                    # Consume energy for component response
                    self.consume_message_energy(neighbor_uav, time_step, message_type='COMPONENT_RESPONSE')
                    
                    self.messages_received += 1
            
            print(f"  [Component Exchange] New UAV {uav_id} exchanged info with {len(uav.neighbors)} neighbors")

    def perform_checkpoint_flood(self, uavs, neighbors):
        """Simulate distributed flood to check if all UAVs are reachable"""
        reachable_sets = {}
        for start_uav in uavs:
            if not start_uav.connected:
                continue
            visited = set()
            queue = [start_uav.id]
            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                queue.extend(n for n in neighbors.get(curr, set()) if n not in visited)
            reachable_sets[start_uav.id] = visited

        # Assume flood propagates the largest known set
        max_reachable = max(reachable_sets.values(), key=len, default=set())
        for uav in uavs:
            uav.known_connected_ids = max_reachable.copy()

        all_known = all(len(uav.known_connected_ids) == len(uav.known_uav_ids) for uav in uavs if uav.connected)
        return all_known

    def _update_movements_with_energy(self, aggregated_scores, uavs,
                                      drone_positions, time_step,
                                      uncovered_user_count):
        """Update movements with energy consumption tracking"""
        movements = []

        if uncovered_user_count > len(uavs) * 0.3:
            exploration_chance = 0.85
            force_exploration = (time_step % 3 == 0)
            print(f"  [Exploration] AGGRESSIVE MODE - {uncovered_user_count} uncovered users")
        else:
            exploration_chance = self.config.EXPLORATION_RATE
            force_exploration = (time_step % 5 == 0)

        uav_order = list(range(len(uavs)))
        random.shuffle(uav_order)

        for uav_id in uav_order:
            uav = uavs[uav_id]
            
            # Skip failed or dying UAVs with insufficient energy
            if uav.failed or uav.energy < self.config.ENERGY_THRESHOLD_DEATH * 2:
                continue
            
            if uav_id not in aggregated_scores or not aggregated_scores[uav_id]:
                if random.random() < exploration_chance or force_exploration:
                    self._force_random_move(uav_id, uavs, drone_positions, movements, time_step)
                continue

            scores = aggregated_scores[uav_id]
            current_grid = uav.current_grid

            if not scores:
                continue

            best_grid = max(scores.items(), key=lambda x: x[1])[0]
            best_score = scores[best_grid]
            current_score = scores.get(current_grid, 0)

            should_move = False
            move_reason = ""

            if best_score > current_score + 0.05:
                should_move = True
                move_reason = f"score improvement ({best_score:.3f} > {current_score:.3f})"
            elif random.random() < exploration_chance:
                should_move = True
                move_reason = "random exploration"
            elif force_exploration and uav_id % 3 == time_step % 3:
                should_move = True
                move_reason = "forced exploration"

            if should_move and best_grid != current_grid:
                old_position = drone_positions[uav_id]
                if self._execute_move(uav_id, best_grid, uavs, drone_positions):
                    new_position = drone_positions[uav_id]
                    
                    # Consume energy for movement
                    energy_consumed = self.consume_movement_energy(
                        uav, old_position, new_position, self.config.TIME_SLOT
                    )
                    
                    movements.append((uav_id, current_grid, best_grid))
                    self.total_movements += 1
                    self.last_movement_time = time_step
                    print(f"    UAV {uav_id}: {current_grid} → {best_grid} "
                          f"(score: {best_score:.3f}, energy: {uav.energy:.0f}J, "
                          f"consumed: {energy_consumed:.2f}J, reason: {move_reason})")

        if not movements and time_step - self.last_movement_time > 5:
            print(f"    [Force] Initiating random movement (stagnant for {time_step - self.last_movement_time} steps)")
            uav_id = random.choice([i for i in range(len(uavs)) if not uavs[i].failed])
            self._force_random_move(uav_id, uavs, drone_positions, movements, time_step)

        return movements
    
    

    def _compute_candidate_scores_improved(self, uavs, users, drone_positions,
                                           backhaul_links, neighbors, time_step, user_hotspot):
        candidate_scores = defaultdict(dict)

        for uav_id, uav in enumerate(uavs):
            if not uav.connected:
                continue

            current_grid = uav.current_grid
            candidate_grids = self.grid_manager.get_neighboring_grids(current_grid, radius=2)

            all_candidate_grids = []
            for grid_id in candidate_grids:
                if not self.grid_manager.is_grid_occupied(grid_id):
                    all_candidate_grids.append((grid_id, 'empty'))
                else:
                    occupant_id = self.grid_manager.grid_occupancy[grid_id]
                    if occupant_id is not None and occupant_id != uav_id:
                        all_candidate_grids.append((grid_id, 'occupied'))

            if not all_candidate_grids:
                for grid_id in range(self.grid_manager.num_grids):
                    if grid_id != current_grid:
                        if not self.grid_manager.is_grid_occupied(grid_id):
                            all_candidate_grids.append((grid_id, 'empty'))
                        else:
                            occupant_id = self.grid_manager.grid_occupancy[grid_id]
                            if occupant_id is not None and occupant_id != uav_id:
                                all_candidate_grids.append((grid_id, 'occupied'))

            current_coverage = self._calculate_local_coverage_improved(uav_id, drone_positions, users)
            

            hotspot_bias = 0
            if user_hotspot is not None:
                current_pos = drone_positions[uav_id]
                dist_to_hotspot = np.sqrt(
                    (current_pos[0] - user_hotspot[0]) ** 2 +
                    (current_pos[1] - user_hotspot[1]) ** 2
                )
                if dist_to_hotspot < self.config.ACCESS_LINK_RANGE * 3:
                    hotspot_bias = 0.3

            for grid_id, grid_type in all_candidate_grids[:15]:
                coverage_gain = self._calculate_coverage_gain_for_grid_improved(
                    uav_id, grid_id, drone_positions, users
                )

                

                candidate_hotspot_bonus = 0
                if user_hotspot is not None:
                    candidate_pos = self.grid_manager.grid_id_to_position(
                        grid_id, self.config.UAV_HEIGHT, random_offset=True
                    )
                    dist_to_hotspot = np.sqrt(
                        (candidate_pos[0] - user_hotspot[0]) ** 2 +
                        (candidate_pos[1] - user_hotspot[1]) ** 2
                    )
                    if dist_to_hotspot < self.config.ACCESS_LINK_RANGE * 5:
                        candidate_hotspot_bonus = 0.5 * (1 - dist_to_hotspot / (self.config.ACCESS_LINK_RANGE * 5))

                swap_penalty = -0.05 if grid_type == 'occupied' else 0

                max_possible_users = len(users)
                normalized_coverage = min(1.0, coverage_gain / max(1, max_possible_users * 0.1))
                normalized_coverage = max(-1, min(1, normalized_coverage))

                
                score = (
                    self.config.WE_COVERAGE * normalized_coverage +
                    self.config.WE_HOTSPOT  * candidate_hotspot_bonus +swap_penalty
                )

                noise = random.uniform(-0.05, 0.05)
                final_score = score + noise

                candidate_scores[uav_id][grid_id] = {
                    'score': final_score,
                    'coverage_gain': coverage_gain,
                    'hotspot_bonus': candidate_hotspot_bonus,
                    'type': grid_type
                }

        return candidate_scores

    def _calculate_coverage_gain_for_grid_improved(self, uav_id, grid_id, drone_positions, users):
        candidate_pos = self.grid_manager.grid_id_to_position(
            grid_id, self.config.UAV_HEIGHT, random_offset=True
        )

        covered_uncovered = 0
        for user in users:
            if user.connected_drone is None:
                distance = np.sqrt(
                    (user.position[0] - candidate_pos[0]) ** 2 +
                    (user.position[1] - candidate_pos[1]) ** 2
                )
                if distance <= self.config.ACCESS_LINK_RANGE:
                    quality = 1.0 - (distance / self.config.ACCESS_LINK_RANGE)
                    covered_uncovered += quality

        return covered_uncovered

    def _calculate_local_coverage_improved(self, uav_id, drone_positions, users):
        if uav_id >= len(drone_positions):
            return 0

        drone_pos = drone_positions[uav_id]
        covered = 0

        for user in users:
            if user.connected_drone is None or user.connected_drone == uav_id:
                distance = np.sqrt(
                    (user.position[0] - drone_pos[0]) ** 2 +
                    (user.position[1] - drone_pos[1]) ** 2
                )
                if distance <= self.config.ACCESS_LINK_RANGE:
                    coverage_quality = 1.0 - (distance / self.config.ACCESS_LINK_RANGE)
                    covered += coverage_quality

        return covered

  
    
    def _aggregate_scores(self, candidate_scores, uavs, neighbors, time_step):
        final_scores = defaultdict(dict)

        for uav_id in candidate_scores:
            uav = uavs[uav_id]
            if uav.failed:
                continue

            # Get all valid neighbors
            valid_neighbors = [
                nid for nid in uav.neighbors
                if nid in candidate_scores and not uavs[nid].failed
            ]

            for grid_id, score_info in candidate_scores[uav_id].items():
                own_score = score_info['score']  # S^self_i(g)

                # Collect neighbor scores for this grid → Σ S^self_j(g)
                neighbor_scores = []
                for neighbor_id in valid_neighbors:
                    if grid_id in candidate_scores[neighbor_id]:
                        neighbor_scores.append(
                            candidate_scores[neighbor_id][grid_id]['score']
                        )

                if neighbor_scores:
                    avg_neighbor = sum(neighbor_scores) / len(neighbor_scores)
                    final_scores[uav_id][grid_id] = (
                        (1 - self.config.BETA) * own_score +
                        self.config.BETA * avg_neighbor
                    )
                else:
                    final_scores[uav_id][grid_id] = own_score

            # Log and consume energy for score exchange
            for neighbor_id in valid_neighbors:
                if self.energy_logger:
                    self.energy_logger.log_message(
                        time_step, uav_id, neighbor_id, 'SCORE_EXCHANGE',
                        uav.energy, uavs[neighbor_id].energy,
                        f"Exchanged {len(candidate_scores[neighbor_id])} scores"
                    )
                self.consume_message_energy(uav, time_step, message_type='SCORE_EXCHANGE')
                self.consume_message_energy(uavs[neighbor_id], time_step, message_type='SCORE_EXCHANGE')
                self.messages_sent += 1
                self.messages_received += 1

        return final_scores

    def get_energy_stats(self, uavs):
        active_uavs = [u for u in uavs if not u.failed]

        stats = {
            'active_count': len(active_uavs),
            'dead_count': len(self.dead_uavs),
            'dying_count': 0,
            'total_energy': 0.0,
            'avg_energy': 0.0,
            'min_energy': 0.0,
            'max_energy': 0.0
        }

        if active_uavs:
            energies = [u.energy for u in active_uavs]
            stats.update({
                'total_energy': sum(energies),
                'avg_energy': np.mean(energies),
                'min_energy': min(energies),
                'max_energy': max(energies),
                'dying_count': sum(1 for u in active_uavs if u.is_dying)
            })

        return stats
    
    def _force_random_move(self, uav_id, uavs, drone_positions, movements, time_step):
        """Force random movement with energy tracking"""
        if uav_id >= len(uavs) or uavs[uav_id].failed:
            return
            
        current_grid = uavs[uav_id].current_grid
        all_grids = list(range(self.grid_manager.num_grids))
        all_grids.remove(current_grid)

        if not all_grids:
            return

        random.shuffle(all_grids)
        for new_grid in all_grids[:5]:
            old_position = drone_positions[uav_id]
            if self._execute_move(uav_id, new_grid, uavs, drone_positions):
                new_position = drone_positions[uav_id]
                
                # Consume energy for movement
                energy_consumed = self.consume_movement_energy(
                    uavs[uav_id], old_position, new_position, self.config.TIME_SLOT
                )
                
                movements.append((uav_id, current_grid, new_grid))
                self.total_movements += 1
                print(f"    UAV {uav_id} random move: {current_grid} → {new_grid} "
                      f"(energy: {uavs[uav_id].energy:.0f}J, consumed: {energy_consumed:.2f}J)")
                break
            
    def _execute_move(self, uav_id, new_grid, uavs, drone_positions):
        current_grid = uavs[uav_id].current_grid

        if new_grid == current_grid:
            return False

        if self.grid_manager.is_grid_occupied(new_grid):
            occupant_id = self.grid_manager.grid_occupancy[new_grid]
            if occupant_id is not None and occupant_id < len(uavs):
                if random.random() < 0.95:
                    self.grid_manager.vacate_grid(current_grid)
                    self.grid_manager.vacate_grid(new_grid)

                    self.grid_manager.occupy_grid(new_grid, uav_id)
                    self.grid_manager.occupy_grid(current_grid, occupant_id)

                    new_position = self.grid_manager.grid_id_to_position(
                        new_grid, self.config.UAV_HEIGHT, random_offset=True
                    )
                    old_position = self.grid_manager.grid_id_to_position(
                        current_grid, self.config.UAV_HEIGHT, random_offset=True
                    )

                    uavs[uav_id].update_position(new_position)
                    uavs[uav_id].current_grid = new_grid
                    drone_positions[uav_id] = new_position

                    uavs[occupant_id].update_position(old_position)
                    uavs[occupant_id].current_grid = current_grid
                    drone_positions[occupant_id] = old_position

                    return True
                else:
                    return False
            else:
                return False

        candidate_pos = self.grid_manager.grid_id_to_position(
            new_grid, self.config.UAV_HEIGHT, random_offset=True
        )

        grid_distance = self._grid_distance(current_grid, new_grid)
        if grid_distance > 20:
            return False

        self.grid_manager.vacate_grid(current_grid)

        if self.grid_manager.occupy_grid(new_grid, uav_id):
            new_position = self.grid_manager.grid_id_to_position(
                new_grid, self.config.UAV_HEIGHT, random_offset=True
            )
            uavs[uav_id].update_position(new_position)
            uavs[uav_id].current_grid = new_grid
            drone_positions[uav_id] = new_position
            return True

        return False

    def _grid_distance(self, grid1, grid2):
        row1 = grid1 // self.grid_manager.grid_cols
        col1 = grid1 % self.grid_manager.grid_cols
        row2 = grid2 // self.grid_manager.grid_cols
        col2 = grid2 % self.grid_manager.grid_cols

        return max(abs(row1 - row2), abs(col1 - col2))

    def get_message_stats(self):
        return {
            'sent': self.messages_sent,
            'received': self.messages_received,
            'total': self.messages_sent + self.messages_received
        }
