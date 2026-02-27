
import csv
import os
from datetime import datetime
import json
import numpy as np
import pandas as pd


class EnergyLogger:
    def __init__(self, config, output_dir="results", iteration=0):
        self.config = config
        self.output_dir = output_dir
        self.iteration = iteration
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Energy consumption log
        if not hasattr(config, 'shared_energy_file') or config.shared_energy_file is None:
            self.energy_file = os.path.join(output_dir, f"energy_consumption_{timestamp}.csv")
            config.shared_energy_file = self.energy_file
            self._initialize_energy_csv()
            print(f"  Created shared energy CSV: {self.energy_file}")
        else:
            self.energy_file = config.shared_energy_file
            print(f"  Using shared energy CSV: {self.energy_file}")
        
        # Message passing log
        if not hasattr(config, 'shared_message_file') or config.shared_message_file is None:
            self.message_file = os.path.join(output_dir, f"message_passing_{timestamp}.csv")
            config.shared_message_file = self.message_file
            self._initialize_message_csv()
            print(f"  Created shared message CSV: {self.message_file}")
        else:
            self.message_file = config.shared_message_file
            print(f"  Using shared message CSV: {self.message_file}")
        
        # NEW: Message statistics per time step
        if not hasattr(config, 'shared_message_stats_file') or config.shared_message_stats_file is None:
            self.message_stats_file = os.path.join(output_dir, f"message_statistics_{timestamp}.csv")
            config.shared_message_stats_file = self.message_stats_file
            self._initialize_message_stats_csv()
            print(f"  Created shared message stats CSV: {self.message_stats_file}")
        else:
            self.message_stats_file = config.shared_message_stats_file
            print(f"  Using shared message stats CSV: {self.message_stats_file}")
        
        # Death events log
        if not hasattr(config, 'shared_death_file') or config.shared_death_file is None:
            self.death_file = os.path.join(output_dir, f"death_events_{timestamp}.csv")
            config.shared_death_file = self.death_file
            self._initialize_death_csv()
            print(f"  Created shared death CSV: {self.death_file}")
        else:
            self.death_file = config.shared_death_file
            print(f"  Using shared death CSV: {self.death_file}")
        
        # Stochastic failures log
        if not hasattr(config, 'shared_stochastic_file') or config.shared_stochastic_file is None:
            self.stochastic_file = os.path.join(output_dir, f"stochastic_failures_{timestamp}.csv")
            config.shared_stochastic_file = self.stochastic_file
            self._initialize_stochastic_csv()
            print(f"  Created shared stochastic CSV: {self.stochastic_file}")
        else:
            self.stochastic_file = config.shared_stochastic_file
            print(f"  Using shared stochastic CSV: {self.stochastic_file}")
        
        # Energy summary per UAV
        self.energy_summary = {}
        
        # NEW: In-memory message counter for current iteration
        self.message_counts = {}  # {time_step: {message_type: count}}
    
    def _initialize_energy_csv(self):
        with open(self.energy_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'time_step', 'uav_id', 'energy_joules', 'energy_percent',
                'hovering_power_w', 'kinetic_power_w', 'flying_power_w',
                'tx_power_w', 'processing_power_w', 'message_cost_j',
                'total_consumption_j', 'velocity_mps', 'status', 'failure_type'
            ])
    
    def _initialize_message_csv(self):
        """Initialize detailed message log"""
        with open(self.message_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'time_step', 'sender_id', 'receiver_id', 'message_type',
                'sender_energy', 'receiver_energy', 'content_summary'
            ])
    
    def _initialize_message_stats_csv(self):
        """NEW: Initialize aggregated message statistics per time step"""
        with open(self.message_stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'time_step', 
                'coverage_map', 'score_exchange', 
                'component_query', 'component_response', 
                'death_warning', 'total_messages'
            ])
    
    def _initialize_death_csv(self):
        with open(self.death_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'time_step', 'uav_id', 'death_type', 'final_energy',
                'neighbors_notified', 'replacement_needed'
            ])
    
    def _initialize_stochastic_csv(self):
        with open(self.stochastic_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'time_step', 'uav_id', 'random_value', 'threshold',
                'failure_occurred', 'energy_at_failure', 'failure_reason'
            ])
    
    def log_energy_consumption(self, time_step, uav, energy_breakdown):
        try:
            with open(self.energy_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                status = 'DEAD' if uav.failed else ('DYING' if uav.is_dying else 'ACTIVE')
                failure_type = uav.failure_type if uav.failed else 'None'
                
                writer.writerow([
                    self.iteration,
                    time_step,
                    uav.id,
                    round(uav.energy, 2),
                    round(uav.get_energy_percentage(), 2),
                    round(energy_breakdown.get('hovering_power', 0), 4),
                    round(energy_breakdown.get('kinetic_power', 0), 4),
                    round(energy_breakdown.get('flying_power', 0), 4),
                    round(energy_breakdown.get('tx_power', 0), 4),
                    round(energy_breakdown.get('processing_power', 0), 4),
                    round(energy_breakdown.get('message_cost', 0), 4),
                    round(energy_breakdown.get('total_consumption', 0), 2),
                    round(np.linalg.norm(uav.velocity), 2) if hasattr(uav, 'velocity') else 0,
                    status,
                    failure_type
                ])
        except Exception as e:
            print(f"Error logging energy for UAV {uav.id}: {e}")
    
    def log_message(self, time_step, sender_id, receiver_id, message_type, 
                   sender_energy, receiver_energy, content=None):
        """Log individual message AND update statistics counter"""
        try:
            # Log detailed message
            with open(self.message_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                content_summary = ""
                if content:
                    if isinstance(content, dict):
                        content_summary = f"Keys: {list(content.keys())[:3]}"
                    else:
                        content_summary = str(content)[:50]
                
                writer.writerow([
                    self.iteration,
                    time_step,
                    sender_id,
                    receiver_id,
                    message_type,
                    round(sender_energy, 2),
                    round(receiver_energy, 2),
                    content_summary
                ])
            
            # NEW: Update in-memory counter
            if time_step not in self.message_counts:
                self.message_counts[time_step] = {
                    'COVERAGE_MAP': 0,
                    'SCORE_EXCHANGE': 0,
                    'COMPONENT_QUERY': 0,
                    'COMPONENT_RESPONSE': 0,
                    'DEATH_WARNING': 0
                }
            
            if message_type in self.message_counts[time_step]:
                self.message_counts[time_step][message_type] += 1
            
        except Exception as e:
            print(f"Error logging message {sender_id}->{receiver_id}: {e}")
    
    def save_message_statistics_for_timestep(self, time_step):
        """NEW: Save aggregated message statistics for a time step"""
        try:
            counts = self.message_counts.get(time_step, {
                'COVERAGE_MAP': 0,
                'SCORE_EXCHANGE': 0,
                'COMPONENT_QUERY': 0,
                'COMPONENT_RESPONSE': 0,
                'DEATH_WARNING': 0
            })
            
            total = sum(counts.values())
            
            with open(self.message_stats_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.iteration,
                    time_step,
                    counts.get('COVERAGE_MAP', 0),
                    counts.get('SCORE_EXCHANGE', 0),
                    counts.get('COMPONENT_QUERY', 0),
                    counts.get('COMPONENT_RESPONSE', 0),
                    counts.get('DEATH_WARNING', 0),
                    total
                ])
        except Exception as e:
            print(f"Error saving message statistics for timestep {time_step}: {e}")
    
    def log_death_event(self, time_step, uav_id, death_type, final_energy, 
                       neighbors_notified, replacement_needed=True):
        try:
            with open(self.death_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.iteration,
                    time_step,
                    uav_id,
                    death_type,
                    round(final_energy, 2),
                    len(neighbors_notified) if neighbors_notified else 0,
                    'Yes' if replacement_needed else 'No'
                ])
            
            print(f"  [Death Log] UAV {uav_id} death recorded - Type: {death_type}, Energy: {final_energy:.2f}J")
        except Exception as e:
            print(f"Error logging death for UAV {uav_id}: {e}")
    
    def log_stochastic_failure(self, time_step, uav_id, random_value, threshold, 
                              failure_occurred, energy_at_failure, failure_reason="None"):
        try:
            with open(self.stochastic_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.iteration,
                    time_step,
                    uav_id,
                    round(random_value, 6),
                    round(threshold, 6),
                    'Yes' if failure_occurred else 'No',
                    round(energy_at_failure, 2),
                    failure_reason
                ])
            
            if failure_occurred:
                print(f"  [Stochastic Failure Log] UAV {uav_id} failed due to random draw")
        except Exception as e:
            print(f"Error logging stochastic failure for UAV {uav_id}: {e}")
    
    def update_summary(self, uav_id, time_step, energy):
        if uav_id not in self.energy_summary:
            self.energy_summary[uav_id] = {
                'initial_energy': energy,
                'final_energy': energy,
                'min_energy': energy,
                'total_consumption': 0,
                'lifetime': 0
            }
        
        summary = self.energy_summary[uav_id]
        summary['final_energy'] = energy
        summary['min_energy'] = min(summary['min_energy'], energy)
        summary['lifetime'] = time_step
    
    def save_summary(self):
        summary_file = os.path.join(self.output_dir, f"energy_summary_iter{self.iteration}.txt")
        
        try:
            with open(summary_file, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write(f"ENERGY CONSUMPTION SUMMARY - ITERATION {self.iteration}\n")
                f.write("=" * 60 + "\n\n")
                
                for uav_id, summary in sorted(self.energy_summary.items()):
                    f.write(f"UAV {uav_id}:\n")
                    f.write(f"  Initial Energy: {summary['initial_energy']:.2f} J\n")
                    f.write(f"  Final Energy: {summary['final_energy']:.2f} J\n")
                    f.write(f"  Min Energy: {summary['min_energy']:.2f} J\n")
                    f.write(f"  Total Consumed: {summary['initial_energy'] - summary['final_energy']:.2f} J\n")
                    f.write(f"  Lifetime: {summary['lifetime']} steps\n")
                    f.write(f"  Energy/Step: {(summary['initial_energy'] - summary['final_energy'])/max(1, summary['lifetime']):.2f} J\n")
                    f.write("\n")
                
                total_initial = sum(s['initial_energy'] for s in self.energy_summary.values())
                total_final = sum(s['final_energy'] for s in self.energy_summary.values())
                total_consumed = total_initial - total_final
                
                f.write("=" * 60 + "\n")
                f.write("OVERALL STATISTICS\n")
                f.write("=" * 60 + "\n")
                f.write(f"Total UAVs: {len(self.energy_summary)}\n")
                f.write(f"Total Initial Energy: {total_initial:.2f} J\n")
                f.write(f"Total Final Energy: {total_final:.2f} J\n")
                f.write(f"Total Energy Consumed: {total_consumed:.2f} J\n")
                f.write(f"Average Energy/UAV: {total_consumed/len(self.energy_summary):.2f} J\n")
            
            print(f"\nEnergy summary saved to: {summary_file}")
        
        except Exception as e:
            print(f"Error saving energy summary: {e}")
            
    def analyze_messages_by_type(self):
        try:
            df = pd.read_csv(self.message_file)
            df_iter = df[df['iteration'] == self.iteration]
            type_counts = df_iter.groupby('message_type').size()
            type_energy_cost = df_iter.groupby('message_type').agg({
                'sender_energy': 'mean',
                'receiver_energy': 'mean'
            })
            analysis = {
                'total_messages': len(df_iter),
                'by_type': type_counts.to_dict(),
                'avg_energy_by_type': type_energy_cost.to_dict()
            }
            return analysis
        except Exception as e:
            print(f"Error analyzing messages: {e}")
            return {}

    def analyze_messages_over_time(self):
        try:
            df = pd.read_csv(self.message_file)
            df_iter = df[df['iteration'] == self.iteration]
            time_analysis = df_iter.groupby('time_step').agg({
                'message_type': 'count',
                'sender_energy': 'mean',
                'receiver_energy': 'mean'
            }).rename(columns={'message_type': 'message_count'})
            type_time_analysis = df_iter.groupby(['time_step', 'message_type']).size().unstack(fill_value=0)
            return {
                'time_series': time_analysis,
                'type_time_matrix': type_time_analysis
            }
        except Exception as e:
            print(f"Error analyzing time series: {e}")
            return {}

    def generate_message_analysis_csv(self):
        analysis_file = os.path.join(self.output_dir, f"message_analysis_iter{self.iteration}.csv")
        try:
            df = pd.read_csv(self.message_file)
            df_iter = df[df['iteration'] == self.iteration]
            summary = df_iter.groupby(['time_step', 'message_type']).agg({
                'sender_id': 'count',
                'sender_energy': 'mean',
                'receiver_energy': 'mean'
            }).rename(columns={'sender_id': 'count'})
            summary.to_csv(analysis_file)
            print(f"\nMessage analysis saved to: {analysis_file}")
            return analysis_file
        except Exception as e:
            print(f"Error generating analysis CSV: {e}")
            return None
        
        