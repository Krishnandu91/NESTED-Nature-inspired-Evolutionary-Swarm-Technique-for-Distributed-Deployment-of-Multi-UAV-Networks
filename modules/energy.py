import numpy as np
from config.Configuration import Config


class EnergyModel:

    MAX_MOTOR_POWER = 5000.0
    MAX_ACCELERATION = 5.0

    @staticmethod
    def hovering_power():
        """
        Calculate hovering power for UAV
        """
        m = Config.UAV_MASS
        g = Config.GRAVITY
        rho = Config.AIR_DENSITY
        xi = Config.NUM_PROPELLERS
        r = Config.PROPELLER_RADIUS
        CL = Config.LIFT_COEFF
        CD = Config.DRAG_COEFF

        induced = np.sqrt(
            (2 * m * g) /
            (xi * rho * np.pi * r ** 2 * CL)
        )

        return (m * g * CD / CL) * induced

    @staticmethod
    def limit_acceleration(v, dv):
        max_dv = EnergyModel.MAX_ACCELERATION * Config.TIME_SLOT
        norm = np.linalg.norm(dv)
        if norm > max_dv:
            dv = dv / norm * max_dv
        return dv

    @staticmethod
    def kinetic_power(velocity, delta_velocity):
        delta_velocity = EnergyModel.limit_acceleration(
            velocity,
            delta_velocity
        )
        m = Config.UAV_MASS
        CM = Config.PITCH_COEFF
        J = Config.PROPELLER_PITCH
        CL = Config.LIFT_COEFF
        delta = Config.TIME_SLOT

        v_norm = np.linalg.norm(velocity)
        dv_norm = np.linalg.norm(delta_velocity)

        coeff = (2 * np.pi * m * CM) / (delta * J * CL)
        return coeff * v_norm * dv_norm

    @staticmethod
    def flying_power(velocity, delta_velocity):
        p_hover = EnergyModel.hovering_power()
        p_kinetic = EnergyModel.kinetic_power(
            velocity,
            delta_velocity
        )

        power = p_hover + p_kinetic
        return min(power, EnergyModel.MAX_MOTOR_POWER)

    @staticmethod
    def processing_power(message_type='data'):
        return Config.calculate_processing_power(message_type)

    @staticmethod
    def radio_power(mode='tx'):
        if mode == 'tx':
            return Config.RADIO_TX_POWER  # 62.64 mW
        elif mode == 'rx':
            return Config.RADIO_RX_POWER  # 57.4 mW
        elif mode == 'idle':
            return Config.RADIO_IDLE_POWER  # 62.64 mW
        elif mode == 'sleep':
            return Config.RADIO_SLEEP_POWER  # 72 µW
        else:
            return Config.RADIO_IDLE_POWER  # Default to idle

    @staticmethod
    def total_energy(velocity, delta_velocity, tx_power, message_type='data'):
        # Limit transmission power to maximum
        tx_power = min(tx_power, Config.MAX_TX_POWER)

        # Calculate flying power (hovering + kinetic)
        p_f = EnergyModel.flying_power(
            velocity,
            delta_velocity
        )
        
        # Use realistic processing power based on message type
        # FROM TABLE 6: Processing Power = 2.0 W
        p_processing = EnergyModel.processing_power(message_type)

        # Total power consumption
        total_power = (
            p_f +              # Flying power (hovering + kinetic)
            p_processing +     # Processing power (2.0 W from Table 6)
            tx_power          # Transmission power
        )

        # Energy = Power × Time
        energy = Config.TIME_SLOT * total_power
        return max(0.0, energy)
    
    @staticmethod
    def communication_energy(message_type='control', mode='send'):
        # Get packet size
        packet_size = Config.get_packet_size(message_type)
        
        # Calculate transmission time
        tx_time = Config.calculate_packet_transmission_time(packet_size)
        
        # Calculate processing time (ASSUMPTION)
        processing_time = Config.PACKET_PROCESSING_OVERHEAD
        
        # Energy based on mode
        if mode == 'send':
            # Radio TX energy + Processing energy
            radio_energy = Config.RADIO_TX_POWER * tx_time
            processing_energy = Config.PROCESSING_POWER * processing_time
        else:  # receive
            # Radio RX energy + Processing energy
            radio_energy = Config.RADIO_RX_POWER * tx_time
            processing_energy = Config.PROCESSING_POWER * processing_time
        
        total_energy = radio_energy + processing_energy
        return total_energy
    
    @staticmethod
    def get_energy_breakdown(velocity, delta_velocity, tx_power, message_type='data'):
        hovering = EnergyModel.hovering_power()
        kinetic = EnergyModel.kinetic_power(velocity, delta_velocity)
        flying = EnergyModel.flying_power(velocity, delta_velocity)
        processing = EnergyModel.processing_power(message_type)
        tx = min(tx_power, Config.MAX_TX_POWER)
        
        total = Config.TIME_SLOT * (flying + processing + tx)
        
        return {
            'hovering_power': hovering,
            'kinetic_power': kinetic,
            'flying_power': flying,
            'processing_power': processing,
            'tx_power': tx,
            'total_power': flying + processing + tx,
            'total_energy': total,
            'time_slot': Config.TIME_SLOT
        }