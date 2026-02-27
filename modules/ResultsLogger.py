import csv
import os
from datetime import datetime


class ResultsLogger:
    """
    Results Logger with shared CSV file across iterations
    """

    def __init__(self, config, output_dir="results", iteration=0):
        self.config = config
        self.output_dir = output_dir
        self.iteration = iteration
        self.results = []

        os.makedirs(output_dir, exist_ok=True)

        if not hasattr(config, 'shared_results_file'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.filename = os.path.join(output_dir, f"simulation_results_{timestamp}.csv")
            config.shared_results_file = self.filename
            self._initialize_csv()
            print(f" Created shared results CSV: {self.filename}")
        else:
            self.filename = config.shared_results_file
            print(f" Using shared results CSV: {self.filename}")

    def _initialize_csv(self):
        """Initialize CSV file with headers (only called for first iteration)"""
        headers = [
            'iteration', 'time', 'no_uav', 'coverage_obtained', 'message_passing',
            'no_backhaul_links', 'uavs_connected'
        ]
        try:
            with open(self.filename, 'w', newline='') as f:
                csv.writer(f).writerow(headers)
        except Exception as e:
            print(f" Error creating CSV: {e}")
            self.filename = "simulation_results.csv"
            with open(self.filename, 'w', newline='') as f:
                csv.writer(f).writerow(headers)

    def log_iteration(self, time_step, active_uavs, coverage_percentage,
                      message_stats, backhaul_links, connected_uavs):
        # BUG FIX 1: removed extra argument that Main.py was passing
        # Old signature had a mismatched number of args after motif removal
        result = {
            'iteration': self.iteration,
            'time': time_step,
            'no_uav': active_uavs,
            'coverage_obtained': round(coverage_percentage, 2),
            'message_passing': message_stats['total'],
            'no_backhaul_links': len(backhaul_links),
            'uavs_connected': connected_uavs
        }

        self.results.append(result)

        try:
            with open(self.filename, 'a', newline='') as f:
                csv.writer(f).writerow([
                    result['iteration'],
                    result['time'],
                    result['no_uav'],
                    result['coverage_obtained'],
                    result['message_passing'],
                    result['no_backhaul_links'],
                    result['uavs_connected']
                ])
        except Exception as e:
            print(f" Error writing to CSV: {e}")

        return result

    def save_summary(self):
        """Save summary statistics (per iteration)"""
        if not self.results:
            print("No results to save in summary")
            return

        summary_file = os.path.join(
            self.output_dir, f"simulation_summary_iter{self.iteration}.txt")

        try:
            with open(summary_file, 'w') as f:
                f.write(f"=== Simulation Summary - Iteration {self.iteration} ===\n")
                f.write(f"Field Size: {self.config.FIELD_SIZE / 1000} km x "
                        f"{self.config.FIELD_SIZE / 1000} km\n")
                f.write(f"Number of UAVs: {self.config.NO_UAV}\n")
                f.write(f"Number of Users: {self.config.NO_PEOPLE}\n")
                f.write(f"Simulation Duration: {self.config.SIM_DURATION} seconds\n")
                f.write(f"UAV Height: {self.config.UAV_HEIGHT} m\n")
                f.write(f"Access Link Range: {self.config.ACCESS_LINK_RANGE} m\n")
                f.write(f"Backhaul Link Range: {self.config.BACKHAUL_LINK_RANGE} m\n")

                f.write("\n=== Algorithm Weights ===\n")
                f.write(f"Coverage Weight (w_C): {self.config.WE_COVERAGE}\n")
                f.write(f"Hotspot Weight (w_H):  {self.config.WE_HOTSPOT}\n")
                f.write(f"Neighbor Beta (β):     {self.config.BETA}\n")

                f.write("\n=== Final Results ===\n")
                last_result = self.results[-1]
                for key, value in last_result.items():
                    f.write(f"{key}: {value}\n")

                # BUG FIX 2: removed avg_motifs line that referenced deleted key
                if len(self.results) > 1:
                    avg_coverage = sum(r['coverage_obtained'] for r in self.results) / len(self.results)
                    avg_messages = sum(r['message_passing'] for r in self.results) / len(self.results)
                    avg_links = sum(r['no_backhaul_links'] for r in self.results) / len(self.results)

                    f.write("\n=== Average Values ===\n")
                    f.write(f"Average Coverage:        {avg_coverage:.2f}%\n")
                    f.write(f"Average Messages:        {avg_messages:.2f}\n")
                    f.write(f"Average Backhaul Links:  {avg_links:.2f}\n")

            print(f"\n Summary saved to: {summary_file}")

        except Exception as e:
            print(f" Error saving summary: {e}")