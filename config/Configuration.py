# Synthetic Dataset Names are: 
# 1. 100 ground users in 30 X 30 sq. km. field : Coverage_recover_and_degradation_analysis_100.csv
# 2. Default field size and ground users: Coverage_recover_and_degradation_analysis_200.csv
# 3. 300 ground users in 30 X 30 sq. km. field : Coverage_recover_and_degradation_analysis_300.csv
# 4. 20 X 20 sq. km. field: Coverage_recover_and_degradation_analysis_20km.csv
# 5. 40 X 40 sq. km. filed: Coverage_recover_and_degradation_analysis_40km.csv
# 6. Original Dataset: Original Dataset

import numpy as np

class Config:
    
    FIELD_SIZE = 30000  # Field Size 
    NO_UAV = 5 # Inital UAVs
    UAV_HEIGHT = 1500 # UAVs flight altitude
    NO_PEOPLE = 200 # default no of users
    # User mobility dataset
    
    #USER_MOBILITY_FILE = r"/Users/parthapratimsarmah/#1_ALL_EXPERIMENT_IN_HERE_RELATED_TO_PHD/#1_NESTED/Synthetic_Dataset/Dataset/Coverage_recover_and_degradation_analysis_20km.csv"
    #USER_MOBILITY_FILE = r"/Users/parthapratimsarmah/#1_ALL_EXPERIMENT_IN_HERE_RELATED_TO_PHD/#1_NESTED/Synthetic_Dataset/Dataset/Coverage_recover_and_degradation_analysis_40km.csv"
    #USER_MOBILITY_FILE = r"/Users/parthapratimsarmah/#1_ALL_EXPERIMENT_IN_HERE_RELATED_TO_PHD/#1_NESTED/Synthetic_Dataset/Dataset/Coverage_recover_and_degradation_analysis_100.csv"
    USER_MOBILITY_FILE = r"/Users/parthapratimsarmah/#1_ALL_EXPERIMENT_IN_HERE_RELATED_TO_PHD/#1_NESTED/Synthetic_Dataset/Dataset/Coverage_recover_and_degradation_analysis_200.csv" # It also work for 30 X 30 sq. km
    #USER_MOBILITY_FILE = r"/Users/parthapratimsarmah/#1_ALL_EXPERIMENT_IN_HERE_RELATED_TO_PHD/#1_NESTED/Synthetic_Dataset/Dataset/Coverage_recover_and_degradation_analysis_300.csv"
    #USER_MOBILITY_FILE = r"/Users/parthapratimsarmah/#1_ALL_EXPERIMENT_IN_HERE_RELATED_TO_PHD/#1_NESTED/Dataset/Original_Dataset.csv"
    # Link parameters
    ACCESS_LINK_RANGE = 3300
    BACKHAUL_LINK_RANGE = 8200
    # Path loss parameters
    A = 4.88
    B = 0.429
    ETA_L = 0.1
    ETA_N = 21
    FC = 2e9
    C = 3e8
    # Power and SNR parameters
    TX_POWER = 0.5
    ACCESS_SNR_THRESHOLD = 17
    BACKHAUL_SNR_THRESHOLD = 10
    NOISE_POWER = 10 ** (-170 / 10)
    # Algorithm parameters
    ALPHA = 0.7          # For coverage map update
    WE_COVERAGE = 0.70   # Weight for coverage score
    WE_HOTSPOT = 0.30    # Weight for hotspot score
    BETA = 0.3           # Weight determine the influence of neighbor score
    COVERAGE_DECAY = 0.95
    EXPLORATION_RATE = 0.7
    # Simulation parameters
    SIM_DURATION = 75 #45
    NUM_SIMULATION_RUNS = 50
    TIME_STEP = 1  # seconds
    ITERATIONS_PER_STAGE = 1
    MAX_UAVS_PER_GRID = 1
    RANDOM_SEED = 42
    GRID_SIZE = 4300               
    GRAVITY = 9.81                 # m/s^2
    AIR_DENSITY = 1.225            # kg/m^3
    # UAV parameters
    UAV_MASS = 3.49                 
    NUM_PROPELLERS = 6              
    PROPELLER_RADIUS = 0.21       
    # Aerodynamic coefficients
    LIFT_COEFF = 0.0435  # C_L (CHANGED from 0.8)
    DRAG_COEFF = 0.002  # C_D (CHANGED from 0.6)
    PITCH_COEFF = 0.00016  # C_M (CHANGED from 0.04)
    PROPELLER_PITCH = 0.5  # J (CHANGED from 0.1)
    # REALISTIC COMMUNICATION PARAMETERS (FROM PAPER - Table 6)    
    # Packet sizes (in Bytes) - DIRECTLY FROM TABLE 6
    DATA_PACKET_SIZE = 100  # Bytes (Td) - from Table 6
    CONTROL_PACKET_SIZE = 5  # Bytes (Tc) - from Table 6
    # Data rate - DIRECTLY FROM TABLE 6
    DATA_RATE = 250_000  # 250 kbps = 250,000 bits per second - from Table 6
    # Radio power states from paper (Table 6) - convert mW to W
    RADIO_TX_POWER = 62.64 / 1000  # 62.64 mW = 0.06264 W - from Table 6
    RADIO_RX_POWER = 57.4 / 1000   # 57.4 mW = 0.0574 W - from Table 6
    RADIO_IDLE_POWER = 62.64 / 1000  # 62.64 mW = 0.06264 W - from Table 6
    RADIO_SLEEP_POWER = 72e-6      # 72 µW = 0.000072 W - from Table 6
    # Processing power - DIRECTLY FROM TABLE 6
    PROCESSING_POWER = 2.0         # W - from Table 6
    # Buffer check parameters - DIRECTLY FROM TABLE 6
    BUFFER_CHECK_TIME_FACTOR = 0.1  # Te = 0.1 × Td - from Table 6
    PACKET_PROCESSING_OVERHEAD = 0.001  # 1ms per packet processing
    COVERAGE_MAP_SIZE_MULTIPLIER = 3  # Coverage maps are 3x control packet
    SCORE_EXCHANGE_SIZE_MULTIPLIER = 3  # Score exchange is 3x control packet
    MAX_TX_POWER = 0.1             
    TIME_SLOT = 0.5                
    MAX_BATTERY_ENERGY = 125_000 # Joules (125,000 J = 34.7 Wh)
    ENERGY_THRESHOLD_DEATH = 100  # Joules - UAV dies below this
    ENERGY_WARNING_THRESHOLD = 5000  # Joules - UAV sends warning below this    
    MOVEMENT_BASE_VELOCITY = 10.0  
    # Distributed parameters
    COVERAGE_THRESHOLD = 80  # % below which add UAV
    CHECKPOINT_INTERVAL = 5   # Checking time for coverage and connectivity
    MAX_CHECKPOINT_STEP = 72   
    # System malfunction parameters
    STOCHASTIC_FAILURE_ENABLED = True  # Enable/disable stochastic failures
    STOCHASTIC_FAILURE_START_TIME = 10  
    STOCHASTIC_FAILURE_THRESHOLD = 0.001  # System Malfunction thresold
    STOCHASTIC_FAILURE_CHECK_INTERVAL = 1  
    
    @classmethod
    def calculate_packet_transmission_time(cls, packet_size_bytes):
        """
        Calculate transmission time for a packet
        Formula from paper: Time = (packet_size_bytes × 8 bits/byte) / data_rate
        """
        bits = packet_size_bytes * 8
        return bits / cls.DATA_RATE
    
    @classmethod
    def get_packet_size(cls, message_type='control'):

        # DIRECTLY FROM PAPER
        if message_type == 'data':
            return cls.DATA_PACKET_SIZE  # 100 bytes        
        # DIRECTLY FROM PAPER
        if message_type == 'control' or message_type in [
            'DEATH_WARNING', 
            'COMPONENT_QUERY', 
            'COMPONENT_RESPONSE'
        ]:
            return cls.CONTROL_PACKET_SIZE  # 5 bytes
        
        if message_type in ['COVERAGE_MAP', 'SCORE_EXCHANGE']:
            return cls.CONTROL_PACKET_SIZE * cls.COVERAGE_MAP_SIZE_MULTIPLIER  # 15 bytes
        
        # Default to control packet
        return cls.CONTROL_PACKET_SIZE  # 5 bytes
    
    @classmethod
    def calculate_processing_power(cls, message_type='data'):
        
        return cls.PROCESSING_POWER  # 2.0 W
    
    @classmethod
    def calculate_message_energy_cost(cls, message_type='control'):
        # Get packet size
        packet_size = cls.get_packet_size(message_type)
        # Calculate transmission time (from paper's formula)
        tx_time = cls.calculate_packet_transmission_time(packet_size)
        processing_time = cls.PACKET_PROCESSING_OVERHEAD  
        # Total time
        total_time = tx_time + processing_time
        # Energy calculation (from paper)
        # During TX: use TX power (from Table 6)
        tx_energy = cls.RADIO_TX_POWER * tx_time
        # During processing: use processing power (from Table 6)
        processing_energy = cls.PROCESSING_POWER * processing_time
        total_energy = tx_energy + processing_energy
        return total_energy
    
    @classmethod
    def calculate_buffer_check_time(cls):
        """
        Calculate buffer check time
        DIRECTLY FROM TABLE 6: Te = 0.1 × Td
        Returns:
            Buffer check time in seconds
        """
        data_tx_time = cls.calculate_packet_transmission_time(cls.DATA_PACKET_SIZE)
        return cls.BUFFER_CHECK_TIME_FACTOR * data_tx_time
    
    @classmethod
    def calculate_grid_parameters(cls):
        """Calculate grid parameters based on GRID_SIZE"""
        cls.GRID_COLS = int(np.ceil(cls.FIELD_SIZE / cls.GRID_SIZE))
        cls.GRID_ROWS = int(np.ceil(cls.FIELD_SIZE / cls.GRID_SIZE))
        cls.NUM_GRIDS = cls.GRID_ROWS * cls.GRID_COLS
        print(f"Grid configuration: {cls.GRID_ROWS}x{cls.GRID_COLS} = {cls.NUM_GRIDS} grids")
        print(f"Grid size: {cls.GRID_SIZE:.1f}m")
        return cls.GRID_SIZE
    
    @classmethod
    def print_communication_parameters(cls):
        """Print all communication parameters for verification"""
        print("\n" + "="*70)
        print("COMMUNICATION PARAMETERS (From Paper - Table 6)")
        print("="*70)
        print(f"Data Packet Size: {cls.DATA_PACKET_SIZE} Bytes")
        print(f"Control Packet Size: {cls.CONTROL_PACKET_SIZE} Bytes")
        print(f"Data Rate: {cls.DATA_RATE / 1000} kbps")
        print(f"Radio TX Power: {cls.RADIO_TX_POWER * 1000} mW")
        print(f"Radio RX Power: {cls.RADIO_RX_POWER * 1000} mW")
        print(f"Radio Idle Power: {cls.RADIO_IDLE_POWER * 1000} mW")
        print(f"Radio Sleep Power: {cls.RADIO_SLEEP_POWER * 1e6} µW")
        print(f"Processing Power: {cls.PROCESSING_POWER} W")
        print(f"Buffer Check Time Factor: {cls.BUFFER_CHECK_TIME_FACTOR}")
        print("\nCALCULATED VALUES:")
        print(f"Data Packet TX Time: {cls.calculate_packet_transmission_time(cls.DATA_PACKET_SIZE)*1000:.3f} ms")
        print(f"Control Packet TX Time: {cls.calculate_packet_transmission_time(cls.CONTROL_PACKET_SIZE)*1000:.3f} ms")
        print(f"Control Message Energy: {cls.calculate_message_energy_cost('control'):.6f} J")
        print(f"Data Message Energy: {cls.calculate_message_energy_cost('data'):.6f} J")
        print(f"Coverage Map Energy: {cls.calculate_message_energy_cost('COVERAGE_MAP'):.6f} J")
        print("="*70 + "\n")