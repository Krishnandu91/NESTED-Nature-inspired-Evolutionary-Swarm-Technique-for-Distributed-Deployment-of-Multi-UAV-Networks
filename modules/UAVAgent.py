import numpy as np
from collections import defaultdict

class UAVAgent:
    def __init__(self, agent_id, initial_position, config):
        self.id = agent_id
        self.position = initial_position
        self.neighbors = set()
        self.connected = True

        # Energy management
        self.energy = config.MAX_BATTERY_ENERGY  # Start with full battery
        self.max_energy = config.MAX_BATTERY_ENERGY
        self.energy_threshold = config.ENERGY_THRESHOLD_DEATH
        self.warning_threshold = config.ENERGY_WARNING_THRESHOLD
        self.is_dying = False  # Flag for imminent death
        self.death_message_sent = False  # Track if death message was sent
        
        # Velocity tracking for energy calculation
        self.velocity = np.array([0.0, 0.0, 0.0])  # Current velocity (m/s)
        self.previous_position = initial_position
        
        # Energy history
        self.energy_history = [self.energy]
        
        # Coverage map for this UAV
        self.coverage_map = defaultdict(float)
        self.timestamps = defaultdict(float)


        # Algorithm parameters
        self.alpha = config.ALPHA
        self.we_coverage = config.WE_COVERAGE

        # Current grid
        self.current_grid = None

        # Store position for coverage calculation
        self.x = initial_position[0]
        self.y = initial_position[1]
       
        self.failed = False
        
        # ======================================================
        # STOCHASTIC FAILURE TRACKING (NEW)
        # ======================================================
        self.stochastic_failure_check_time = config.STOCHASTIC_FAILURE_START_TIME
        self.stochastic_failure_happened = False  # True if failed due to stochastic random
        self.failure_time = None  # Time step when failure occurred
        self.failure_type = None  # "energy_depletion", "stochastic_random", "scheduled", "targeted"

        # Distributed knowledge
        self.total_users = config.NO_PEOPLE
        self.known_uav_ids = set(range(config.NO_UAV))
        self.known_connected_ids = {self.id}
        self.covered_access_links = 0
        
        # Neighbor energy knowledge
        self.neighbor_energies = {}  # {neighbor_id: energy_level}
        
        # Component information for gravity-based decisions
        self.component_size = 1  # Start with just self
        self.component_members = {self.id}  # Set of UAV IDs in my component

    def update_velocity(self, new_position, time_step):
        """Calculate velocity based on position change"""
        old_pos = np.array(self.previous_position)
        new_pos = np.array(new_position)
        
        displacement = new_pos - old_pos
        velocity = displacement / time_step if time_step > 0 else np.array([0.0, 0.0, 0.0])
        
        self.velocity = velocity
        self.previous_position = new_position
        
        return velocity

    def consume_energy(self, energy_amount):
        """Consume energy and check if UAV should die"""
        self.energy = max(0, self.energy - energy_amount)
        self.energy_history.append(self.energy)
        
        # Check if UAV is dying
        if self.energy <= self.energy_threshold and not self.is_dying:
            self.is_dying = True
            print(f"    [Energy] UAV {self.id} entering dying state (energy: {self.energy:.2f}J)")
        
        # Check if UAV is dead
        if self.energy <= 0:
            self.fail(failure_type="energy_depletion")
            return True
        
        return False

    def send_death_warning(self):
        """Send warning to neighbors about imminent death"""
        if self.is_dying and not self.death_message_sent:
            self.death_message_sent = True
            return True
        return False

    def get_energy_percentage(self):
        """Get remaining energy as percentage"""
        return (self.energy / self.max_energy) * 100

    def fail(self, failure_type="unknown"):
        """Fail the UAV due to energy depletion or other causes"""
        self.failed = True
        self.connected = False
        self.energy = 0
        self.neighbors = set()
        self.failure_type = failure_type
        print(f"    [Death] UAV {self.id} has DIED (reason: {failure_type})")
    
    def check_stochastic_failure(self, time_step, config):
        """
        Check for stochastic failure - randomly fail based on threshold
        
        Returns:
            bool: True if stochastic failure occurred, False otherwise
        """
        if self.failed or not config.STOCHASTIC_FAILURE_ENABLED:
            return False
        
        # Only check after start time and at intervals
        if (time_step >= config.STOCHASTIC_FAILURE_START_TIME and 
            (time_step - config.STOCHASTIC_FAILURE_START_TIME) % config.STOCHASTIC_FAILURE_CHECK_INTERVAL == 0):
            
            # Generate random number between 0 and 1
            random_value = np.random.uniform(0, 1)
            
            # Check if it's below threshold
            if random_value < config.STOCHASTIC_FAILURE_THRESHOLD:
                self.stochastic_failure_happened = True
                self.failure_time = time_step
                self.fail(failure_type="stochastic_random")
                return True
        
        return False
       
    def update_position(self, new_position):
        """Update UAV position"""
        self.position = new_position
        self.x = new_position[0]
        self.y = new_position[1]

    def update_coverage_map(self, grid_id, coverage_value, timestamp):
        """Update coverage map with neighbor information"""
        if timestamp > self.timestamps.get(grid_id, -1):
            old_value = self.coverage_map.get(grid_id, 0)
            self.coverage_map[grid_id] = (
                    self.alpha * coverage_value + (1 - self.alpha) * old_value
            )
            self.timestamps[grid_id] = timestamp
            return True
        return False

    def get_current_coverage(self):
        """Get coverage value for current grid"""
        return self.coverage_map.get(self.current_grid, 0)

    def calculate_score(self, candidate_grid, coverage_gain):
        return self.we_coverage * coverage_gain

    def reset_neighbors(self):
        """Reset neighbor list"""
        self.neighbors = set()

    def update_knowledge(self, sender_id, sender_connected, sender_covered, sender_energy=None):
        """Update local knowledge from sender's message"""
        self.known_connected_ids.add(sender_id)
        self.known_connected_ids.update(sender_connected)
        self.known_uav_ids.add(sender_id)
        self.known_uav_ids.update(sender_connected)
        
        # Update neighbor energy knowledge
        if sender_energy is not None:
            self.neighbor_energies[sender_id] = sender_energy
        
        # Update component information
        self.update_component_info()

    def update_component_info(self):
        """Update component size and members based on known connected IDs"""
        self.component_members = self.known_connected_ids.copy()
        self.component_size = len(self.component_members)

    def query_component_info(self):
        """
        Return component information for message passing
        Returns: dict with component_size and component_members
        """
        return {
            'component_size': self.component_size,
            'component_members': list(self.component_members),
            'uav_id': self.id,
            'position': self.position,
            'energy': self.energy
        }

    def receive_component_query_response(self, responder_info):
        """
        Process response from another UAV about their component
        Used by new UAVs to discover network structure
        
        Args:
            responder_info: dict with component information from another UAV
        """
        # Learn about the responder's component
        responder_id = responder_info['uav_id']
        component_members = set(responder_info['component_members'])
        
        # Update knowledge
        self.known_uav_ids.add(responder_id)
        self.known_uav_ids.update(component_members)
        self.known_connected_ids.add(responder_id)
        self.known_connected_ids.update(component_members)
        
        # Store energy info if available
        if 'energy' in responder_info:
            self.neighbor_energies[responder_id] = responder_info['energy']
        
        self.update_component_info()

    def can_communicate_with(self, other_position, config):
        """
        Check if this UAV can establish communication with another position
        Used for temporary connections during discovery
        """
        distance = np.sqrt(
            (self.position[0] - other_position[0]) ** 2 +
            (self.position[1] - other_position[1]) ** 2 +
            (self.position[2] - other_position[2]) ** 2
        )
        return distance <= config.BACKHAUL_LINK_RANGE

    def get_status_dict(self):
        """Get current status as dictionary for logging"""
        return {
            'id': self.id,
            'energy': self.energy,
            'energy_pct': self.get_energy_percentage(),
            'position': self.position,
            'velocity': self.velocity,
            'connected': self.connected,
            'failed': self.failed,
            'is_dying': self.is_dying,
            'neighbors': list(self.neighbors),
            'covered_users': self.covered_access_links,
            'component_size': self.component_size,
            'component_members': list(self.component_members),
            'stochastic_failure': self.stochastic_failure_happened,
            'failure_type': self.failure_type,
            'failure_time': self.failure_time
        }