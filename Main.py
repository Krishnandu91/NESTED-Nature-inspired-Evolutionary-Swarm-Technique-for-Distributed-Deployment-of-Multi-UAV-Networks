import simpy
import networkx as nx
import numpy as np
import random
import os
import csv
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from config.Configuration import Config
from modules.GridManager import GridManager
from modules.UAVAgent import UAVAgent
from modules.CoverageCalculator import CoverageCalculator
from modules.DistributedAlgorithm import DistributedAlgorithm
from modules.Building_link import LinkBuilder
from modules.Calculation import Calculator
from modules.ResultsLogger import ResultsLogger
from modules.EnergyLogger import EnergyLogger
from modules.Visualization import Visualizer



# User Mobility
class User:
    def __init__(self, user_id, position, step=0):
        self.id            = user_id
        self.position      = position
        self.connected_drone = None
        self.current_step  = step

# Simulation
class UnifiedUAVSimulation:
    def __init__(self, config, exp_name, iteration=0, visualize_iterations=None):
        self.config     = config
        self.exp_name   = exp_name
        self.iteration  = iteration
        self.visualize_iterations = (set(visualize_iterations)
                                     if visualize_iterations else set())

        self.env = simpy.Environment()
        print("\n" + "=" * 60)
        print("INITIALIZING PURE SIMPY DES SIMULATION")
        print(f"  Each UAV = independent SimPy agent process")
        print(f"  Synchronisation via SimPy Events (AllOf barrier)")
        print(f"  Inter-agent comms via SimPy Stores (mailboxes)")
        print("=" * 60)
        self.grid_manager  = GridManager(config)
        self.calculator    = Calculator(config)
        self.users         = self._load_users_from_csv()

        # UAVs state 
        self.uavs            = []
        self.drone_positions = []
        self.drone_energy    = []
        self.grid_ids        = []
        self.backhaul_links  = []
        self.access_links    = []
        self.neighbors       = {}

        self.link_builder  = LinkBuilder(config, self.calculator)
        self.coverage_calc = CoverageCalculator(config, self.grid_manager)

        # Output directory 
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"Cannot write to {output_dir}")
        print(f"\n Output directory: {output_dir}")

        # Loggers 
        self.energy_logger = EnergyLogger(config, output_dir, iteration=iteration)
        self.logger_files  = {
            'energy':        self.energy_logger.energy_file,
            'message':       self.energy_logger.message_file,
            'message_stats': self.energy_logger.message_stats_file,
            'death':         self.energy_logger.death_file,
            'stochastic':    self.energy_logger.stochastic_file,
        }
        print(f"\n Logger Files (Iteration {iteration}):")
        for name, filepath in self.logger_files.items():
            exists = os.path.exists(filepath)
            size   = os.path.getsize(filepath) if exists else 0
            status = f"{size} bytes" if exists else "NOT CREATED"
            print(f"  {name:15} -> {os.path.basename(filepath):55} {status}")

        self.logger    = ResultsLogger(config, output_dir, iteration=iteration)
        self.algorithm = DistributedAlgorithm(
            config, self.grid_manager, self.coverage_calc, self.energy_logger)

        # Visualiser 
        self.visualizer    = Visualizer(config, self.grid_manager)
        self._do_visualize = (iteration in self.visualize_iterations)

        self.uav_energy_done = {}   
        self.step_complete   = {}   
        self.uav_mailboxes   = {}  

        self._setup_csv_logging()
        print(f"\n Simulation ready — Iteration {iteration+1}"
              f"/{config.NUM_SIMULATION_RUNS}")


    # Load the users

    def _load_users_from_csv(self):
        print(f"\nLoading users from: {self.config.USER_MOBILITY_FILE}")
        try:
            df = pd.read_csv(self.config.USER_MOBILITY_FILE)
            print(f" Loaded {len(df)} rows")
            required = ['uid', 'step', 'x', 'y', 'sim_time_min']
            missing  = [c for c in required if c not in df.columns]
            if missing:
                print(f" Missing columns: {missing} -> using random users")
                return self._create_random_users()
            user_ids = df['uid'].unique()
            print(f"  Found {len(user_ids)} unique users")
            users, user_data_map = [], {}
            for uid in user_ids:
                udf  = df[df['uid'] == uid].sort_values('step')
                init = udf[udf['step'] == 0]
                if len(init) > 0:
                    row  = init.iloc[0]
                    user = User(uid, (float(row['x']), float(row['y'])), step=0)
                    users.append(user)
                    user_data_map[uid] = {
                        'positions': list(zip(udf['x'].values, udf['y'].values)),
                        'steps':     udf['step'].values.tolist(),
                        'times':     udf['sim_time_min'].values.tolist(),
                    }
            print(f"  Initialized {len(users)} users")
            self.config.user_mobility_data = user_data_map
            self.config.user_ids           = [u.id for u in users]
            return users
        except Exception as e:
            print(f" CSV load error: {e} -> using random users")
            return self._create_random_users()

    def _create_random_users(self):
        users = [User(i,
                      (random.uniform(0, self.config.FIELD_SIZE),
                       random.uniform(0, self.config.FIELD_SIZE)))
                 for i in range(self.config.NO_PEOPLE)]
        self.config.user_mobility_data = {}
        self.config.user_ids           = [u.id for u in users]
        return users

    # CSV setup

    def _setup_csv_logging(self):
        self.csv_path = getattr(self.config, 'shared_csv_path', None)
        if self.csv_path is None:
            ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_path = os.path.join("results", f"{self.exp_name}_{ts}.csv")
            self.config.shared_csv_path = self.csv_path
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'iteration', 'timestamp', 'no_of_UAV', 'no_of_Fail_UAV',
                    'Coverage', 'network_efficiency', 'num_components'
                ])
            print(f"\n Created shared results CSV: {self.csv_path}")
        else:
            print(f"\n Appending to shared results CSV: {self.csv_path}")

    # UAV initialisation

    def initialize_uavs(self):
        drone_positions, grid_ids, init_backhaul = \
            self.grid_manager.place_uavs_random_one_per_grid(
                self.config.NO_UAV, self.calculator)
        self.drone_positions = drone_positions
        self.grid_ids        = grid_ids

        for i in range(self.config.NO_UAV):
            uav              = UAVAgent(i, drone_positions[i], self.config)
            uav.current_grid = grid_ids[i]
            uav.connected    = True
            print(f" UAV {i} | energy={uav.energy:.0f}J")
            self.uavs.append(uav)
            self.drone_energy.append(1)

        print(f"\n  Placed {len(self.uavs)} UAVs | "
              f"Total energy: {sum(u.energy for u in self.uavs):.0f}J")

        self.neighbors = {i: set() for i in range(self.config.NO_UAV)}
        for i, j in init_backhaul:
            self.neighbors[i].add(j); self.neighbors[j].add(i)
            self.uavs[i].neighbors.add(j); self.uavs[j].neighbors.add(i)

        self.backhaul_links = init_backhaul
        self.access_links   = self.link_builder.build_access_links(
            self.users, self.uavs, self.drone_positions, self.drone_energy)

        for uav in self.uavs:
            uav.known_connected_ids.update(self.neighbors[uav.id])
            uav.known_connected_ids.add(uav.id)
            uav.update_component_info()

        covered = sum(1 for u in self.users if u.connected_drone is not None)
        print(f"  Initial coverage: {covered/len(self.users)*100:.1f}%")


    def _init_des_events(self):
        """
        Pre-allocate SimPy Events for every (time-step, uav) pair and
        create a per-UAV mailbox Store for inter-agent communication.
        Called once after all initial UAVs are created.
        """
        n_uavs = len(self.uavs)
        for t in range(1, self.config.SIM_DURATION + 1):
            self.uav_energy_done[t] = {
                uid: self.env.event() for uid in range(n_uavs)
            }
            self.step_complete[t] = self.env.event()

        for uid in range(n_uavs):
            self.uav_mailboxes[uid] = simpy.Store(self.env)

    def _register_new_uav_events(self, new_id, start_t):
        """
        When a UAV is added dynamically, create its DES events for all
        remaining time steps and give it a mailbox.
        """
        for t in range(start_t, self.config.SIM_DURATION + 1):
            if t in self.uav_energy_done:
                self.uav_energy_done[t][new_id] = self.env.event()
        self.uav_mailboxes[new_id] = simpy.Store(self.env)

    def _trigger_all_future_events(self, uav_id):
        """
        When a UAV fails, immediately succeed all its outstanding
        uav_energy_done events so the AllOf barrier never stalls.
        """
        current_t = int(self.env.now)
        for t in range(current_t, self.config.SIM_DURATION + 1):
            if t in self.uav_energy_done:
                evt = self.uav_energy_done[t].get(uav_id)
                if evt is not None and not evt.triggered:
                    evt.succeed()


    def _uav_agent_process(self, uav_id, start_t=1):        
        uav = self.uavs[uav_id]

        for t in range(start_t, self.config.SIM_DURATION + 1):

            yield self.env.timeout(self.config.TIME_STEP)

            if uav.failed:
                self._release_barrier(uav_id, t)
                yield self.step_complete[t]
                continue

            
            self.algorithm.consume_hovering_energy(uav, t)

           
            if self.config.STOCHASTIC_FAILURE_ENABLED:
                rv    = np.random.uniform(0, 1)
                th    = self.config.STOCHASTIC_FAILURE_THRESHOLD
                start = self.config.STOCHASTIC_FAILURE_START_TIME
                ivl   = self.config.STOCHASTIC_FAILURE_CHECK_INTERVAL
                if t >= start and (t - start) % ivl == 0 and rv < th:
                    self._fire_failure_event(
                        uav_id, t, 'STOCHASTIC_FAILURE', rv=rv, th=th)

                    self._trigger_all_future_events(uav_id)
                    return          

            # Energy-depletion failure event 
            if uav.energy <= 0 and not uav.failed:
                self._fire_failure_event(uav_id, t, 'ENERGY_DEPLETION')
                self._trigger_all_future_events(uav_id)
                return              # process ends — UAV is dead

            # Signal barrier
            self._release_barrier(uav_id, t)

            yield self.step_complete[t]

    def _release_barrier(self, uav_id, t):
        """Trigger this UAV's energy-done event for step t."""
        evt = self.uav_energy_done.get(t, {}).get(uav_id)
        if evt is not None and not evt.triggered:
            evt.succeed()

    def _fire_failure_event(self, uav_id, t, failure_type, rv=None, th=None):
        """Handle a UAV failure as a discrete event."""
        uav = self.uavs[uav_id]
        if failure_type == 'STOCHASTIC_FAILURE':
            if self.energy_logger:
                self.energy_logger.log_stochastic_failure(
                    t, uav_id, rv, th,
                    failure_occurred=True,
                    energy_at_failure=uav.energy,
                    failure_reason=f"draw {rv:.6f} < threshold {th:.6f}")
                self.energy_logger.log_death_event(
                    t, uav_id, 'STOCHASTIC_FAILURE',
                    uav.energy, list(uav.neighbors), replacement_needed=True)
            uav.stochastic_failure_happened = True
            uav.failure_time = t
            uav.fail(failure_type="stochastic_random")
            print(f"\n[STOCHASTIC EVENT] UAV {uav_id} failed at t={t}"
                  f"  (draw={rv:.4f} < th={th:.4f})")
        else:
            if self.energy_logger:
                self.energy_logger.log_death_event(
                    t, uav_id, 'ENERGY_DEPLETION',
                    uav.energy, list(uav.neighbors), replacement_needed=True)
            uav.fail(failure_type="energy_depletion")
            print(f"\n[ENERGY DEPLETION EVENT] UAV {uav_id} died at t={t}")
        self.drone_energy[uav_id] = 0

    # User mobility process

    def _user_mobility_process(self):
        for t in range(1, self.config.SIM_DURATION + 1):
            yield self.env.timeout(self.config.TIME_STEP)
            if not getattr(self.config, 'user_mobility_data', None):
                continue
            for user in self.users:
                if user.id in self.config.user_mobility_data:
                    data = self.config.user_mobility_data[user.id]
                    idx  = min(t, len(data['steps']) - 1)
                    if 0 <= idx < len(data['positions']):
                        x, y           = data['positions'][idx]
                        user.position  = (float(x), float(y))
                        user.current_step = data['steps'][idx]

    
    def _checkpoint_process(self):
        t = self.config.CHECKPOINT_INTERVAL
        while t <= self.config.MAX_CHECKPOINT_STEP:
            yield self.env.timeout(self.config.CHECKPOINT_INTERVAL)
            t = int(self.env.now)

            # Coordinator has already finished this step — safe to read state
            print(f"\n  [CHECKPOINT EVENT -> t={t}]")
            fully_connected = self.algorithm.perform_checkpoint_flood(
                self.uavs, self.neighbors)
            _, cov = self._coverage()

            if not fully_connected or cov < self.config.COVERAGE_THRESHOLD:
                reason = 'fragmented' if not fully_connected else 'low coverage'
                print(f"   Adding new UAV ({reason})")
                self._add_new_uav(t)
            else:
                print(f"   Network healthy")

            t += self.config.CHECKPOINT_INTERVAL


    def _network_coordinator_process(self):
        _, cov0 = self._coverage()
        nc0     = self._compute_graph_components()
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                self.iteration, 0, len(self.uavs), 0,
                round(cov0, 2), 1.0, nc0
            ])

        print(f"\n{'='*70}")
        print(f"SIMULATION: {self.exp_name}  |  "
              f"Iter {self.iteration+1}/{self.config.NUM_SIMULATION_RUNS}")
        print(f"Duration={self.config.SIM_DURATION}  "
              f"Failure={self.config.FAILURE_MODE}")
        print(f"Visualization={'ON' if self._do_visualize else 'OFF'}")
        print(f"{'='*70}\n")

        for t in range(1, self.config.SIM_DURATION + 1):

            # 1. Wait for all UAV agents (AllOf DES barrier)
            #    Agents signal via uav_energy_done[t][uav_id].succeed()
            all_energy_events = list(self.uav_energy_done[t].values())
            if all_energy_events:
                yield simpy.AllOf(self.env, all_energy_events)

            print(f"\n{'='*60}")
            print(f"Step {t}/{self.config.SIM_DURATION}  "
                  f"(Iter {self.iteration+1})")
            print(f"{'='*60}")

            # 2. Sync drone_energy flags
            self.drone_energy = [0 if u.failed else 1 for u in self.uavs]

            # 3. Energy summary
            energy_stats = self.algorithm.get_energy_stats(self.uavs)
            no_fail      = sum(1 for u in self.uavs if u.failed)
            active_n     = energy_stats.get('active_count', len(self.uavs) - no_fail)
            print(f"  Active={active_n}  Failed={no_fail}"
                  f"  AvgE={energy_stats.get('avg_energy', 0):.0f}J"
                  f"  MinE={energy_stats.get('min_energy', 0):.0f}J")

            # 4. Run distributed algorithm (decision-making per agent)
            movements = self.algorithm.run_iteration(
                t, self.uavs, self.users,
                self.drone_positions, self.backhaul_links, self.neighbors)
            print(f"  Movements: {len(movements)}")

            self.energy_logger.save_message_statistics_for_timestep(t)

            # 5. Rebuild links
            self._rebuild_links()

            # 6. Compute metrics
            _, cov  = self._coverage()
            net_eff = self.calculator.calculate_network_efficiency(
                self.drone_energy, self.backhaul_links, [], len(self.uavs))
            nc      = self._compute_graph_components()
            no_fail = sum(1 for u in self.uavs if u.failed)

            print(f"  Coverage={cov:.2f}%  NetEff={net_eff:.4f}  Components={nc}")

            # 7. Log
            for uav in self.uavs:
                self.energy_logger.update_summary(uav.id, t, uav.energy)

            msg_stats = self.algorithm.get_message_stats()
            print(f"  Messages: {msg_stats['total']}")

            self.logger.log_iteration(
                t, len(self.uavs) - no_fail, cov,
                msg_stats, self.backhaul_links, len(self.uavs) - no_fail)

            with open(self.csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self.iteration, t,
                    len(self.uavs) - no_fail, no_fail,
                    round(cov, 2), round(net_eff, 4), nc
                ])

            # 8. Visualise
            if self._do_visualize:
                self.visualizer.plot_and_save_distribution_frame(
                    t, self.uavs, self.users,
                    self.drone_positions, self.access_links)

            # 9. File-size check
            if t % 10 == 0:
                print(f"\n  [File check -> t={t}]")
                for name, fp in self.logger_files.items():
                    sz = os.path.getsize(fp) if os.path.exists(fp) else 0
                    print(f"    {name:15} -> {sz} bytes")

            # 10. Release all UAV agents to advance to t+1
            if not self.step_complete[t].triggered:
                self.step_complete[t].succeed()

        print(f"\n{'='*60}")
        print(f"Coordinator finished (Iter {self.iteration+1})")
        print(f"{'='*60}\n")

    def _coverage(self):
        covered = sum(1 for u in self.users if u.connected_drone is not None)
        pct     = covered / len(self.users) * 100 if self.users else 0
        return covered, pct

    def _rebuild_links(self):
        self.backhaul_links, self.neighbors = \
            self.link_builder.build_backhaul_links(
                self.uavs, self.drone_positions,
                self.drone_energy, len(self.uavs))
        self.access_links = self.link_builder.build_access_links(
            self.users, self.uavs, self.drone_positions, self.drone_energy)
        for uav in self.uavs:
            uav.neighbors = self.neighbors.get(uav.id, set())

    def _compute_graph_components(self):
        active = [i for i in range(len(self.uavs)) if self.drone_energy[i] > 0]
        G = nx.Graph()
        G.add_nodes_from(active)
        for i, j in self.backhaul_links:
            if i in active and j in active:
                G.add_edge(i, j)
        return nx.number_connected_components(G) if active else 0

    def _add_new_uav(self, time_step):
        new_id = len(self.uavs)
        print(f"\n  [UAV ADDITION EVENT] Adding UAV {new_id} at t={time_step}")

        available = self.grid_manager.get_available_grids()
        if not available:
            print(" No available grids!")
            return False

        exp_grid = random.choice(available)
        exp_pos  = self.grid_manager.grid_id_to_position(
            exp_grid, self.config.UAV_HEIGHT, random_offset=True)

        discovered, _ = self._discover_components(exp_pos, time_step)
        best_grid     = self._gravity_grid_search(new_id, exp_pos, discovered)

        if best_grid is None:
            print(" No suitable grid found")
            return False

        margin = self.grid_manager.grid_size * 0.1
        gi     = self.grid_manager.grid_centers[best_grid]
        x      = random.uniform(gi['min_x'] + margin, gi['max_x'] - margin)
        y      = random.uniform(gi['min_y'] + margin, gi['max_y'] - margin)
        pos    = (x, y, self.config.UAV_HEIGHT)

        new_uav              = UAVAgent(new_id, pos, self.config)
        new_uav.current_grid = best_grid
        new_uav.connected    = True
        new_uav.known_uav_ids.add(new_id)
        new_uav.known_connected_ids.add(new_id)

        for comp in discovered:
            for mid in comp['members']:
                if mid < len(self.uavs):
                    info = self.uavs[mid].query_component_info()
                    new_uav.receive_component_query_response(info)

        self.uavs.append(new_uav)
        self.drone_positions.append(pos)
        self.grid_manager.occupy_grid(best_grid, new_id)
        self.drone_energy.append(1)

        # Register DES events for this new UAV
        start_t = time_step + 1
        self._register_new_uav_events(new_id, start_t)

        self._rebuild_links()

        for neigh_id in self.neighbors.get(new_id, set()):
            n = self.uavs[neigh_id]
            new_uav.update_knowledge(
                neigh_id, n.known_connected_ids,
                n.covered_access_links, n.energy)
            n.update_knowledge(
                new_id, new_uav.known_connected_ids,
                new_uav.covered_access_links, new_uav.energy)

        self.config.NO_UAV += 1

        # Start new UAV's independent agent process
        self.env.process(self._uav_agent_process(new_id, start_t=start_t))
        print(f" UAV {new_id} added — agent process started at t={start_t}")
        return True

    def _discover_components(self, new_pos, time_step):
        discovered = {}
        msg_count  = 0
        for uid, uav in enumerate(self.uavs):
            if uav.failed:
                continue
            d = self.calculator.euclidean_distance(new_pos, uav.position)
            if d > self.config.BACKHAUL_LINK_RANGE:
                continue
            pl = self.calculator.calculate_path_loss_backhaul(new_pos, uav.position)
            if self.calculator.calculate_snr(pl) < self.config.BACKHAUL_SNR_THRESHOLD:
                continue
            info      = uav.query_component_info()
            msg_count += 2
            if self.energy_logger:
                self.energy_logger.log_message(
                    time_step, -1, uid, 'COMPONENT_QUERY',
                    0, uav.energy, f"New UAV querying UAV {uid}")
            key = frozenset(info['component_members'])
            if key not in discovered:
                cxs    = [self.drone_positions[m][0] for m in info['component_members']]
                cys    = [self.drone_positions[m][1] for m in info['component_members']]
                center = (np.mean(cxs), np.mean(cys))
                dc     = np.sqrt((new_pos[0]-center[0])**2+(new_pos[1]-center[1])**2)
                discovered[key] = {
                    'size':     info['component_size'],
                    'members':  set(info['component_members']),
                    'center':   center,
                    'distance': dc,
                }
        return list(discovered.values()), msg_count

    def _gravity_grid_search(self, new_uav_id, new_pos, discovered):
        if not discovered:
            avail = self.grid_manager.get_available_grids()
            return random.choice(avail) if avail else None
        candidates = {}
        explored   = set()
        for hop in range(1, 4):
            if hop == 1:
                grids = []
                for comp in discovered:
                    cg = self.grid_manager.position_to_grid_id(comp['center'])
                    grids.extend(self.grid_manager.get_neighboring_grids(cg, radius=1))
            else:
                if not explored:
                    for comp in discovered:
                        explored.add(
                            self.grid_manager.position_to_grid_id(comp['center']))
                nxt   = set()
                for g in explored:
                    nxt.update(self.grid_manager.get_neighboring_grids(g, radius=1))
                grids = list(nxt - explored)
            avail = [g for g in grids
                     if not self.grid_manager.is_grid_occupied(g)
                     and g not in explored]
            if not avail:
                continue
            for g in avail:
                explored.add(g)
                gpos    = self.grid_manager.grid_id_to_position(
                    g, self.config.UAV_HEIGHT, random_offset=True)
                gravity = 0.0
                for comp in discovered:
                    d = max(1.0, np.sqrt(
                        (gpos[0]-comp['center'][0])**2 +
                        (gpos[1]-comp['center'][1])**2))
                    gravity += comp['size'] / (d ** 2)
                can_connect = any(
                    self.calculator.calculate_snr(
                        self.calculator.calculate_path_loss_backhaul(
                            gpos, self.drone_positions[i])
                    ) >= self.config.BACKHAUL_SNR_THRESHOLD
                    for i in range(len(self.drone_positions))
                    if not self.uavs[i].failed
                    and self.calculator.euclidean_distance(
                        gpos, self.drone_positions[i]
                    ) <= self.config.BACKHAUL_LINK_RANGE
                )
                candidates[g] = gravity + (100.0 if can_connect else 0.0)
        if not candidates:
            avail = self.grid_manager.get_available_grids()
            return random.choice(avail) if avail else None
        return max(candidates, key=candidates.get)


    def run(self):
        print(f"\n{'='*70}")
        print(f"STARTING — Iter {self.iteration+1}/{self.config.NUM_SIMULATION_RUNS}")
        print(f"{'='*70}\n")

        self.initialize_uavs()
        self._init_des_events()

        #Register all SimPy processes 
        for uid in range(len(self.uavs)):
            self.env.process(self._uav_agent_process(uid, start_t=1))
        # User mobility process
        self.env.process(self._user_mobility_process())
        # Checkpoint process
        self.env.process(self._checkpoint_process())
        # Network coordinator process
        self.env.process(self._network_coordinator_process())

        self.env.run(until=self.config.SIM_DURATION + 1)

        print(f"\n{'='*70}")
        print(f"FINALIZING — Iter {self.iteration+1}")
        print(f"{'='*70}")

        self.logger.save_summary()
        self.energy_logger.save_summary()

        if self._do_visualize:
            print("\nBuilding animation...")
            self.visualizer.create_animation(interval=500)
            print("Animation done.")
        else:
            print("\n (Animation skipped for this iteration)")

        print(f"\n{'='*70}")
        print(f"ITERATION {self.iteration+1} — OUTPUT FILES")
        print(f"{'='*70}")
        for name, fp in self.logger_files.items():
            exists = os.path.exists(fp)
            sz     = os.path.getsize(fp) if exists else 0
            lines  = 0
            if exists:
                try:
                    with open(fp) as fh:
                        lines = len(fh.readlines())
                except Exception:
                    lines = -1
            print(f"{'Exist' if exists else 'Not exist'} {name:15} "
                  f" {os.path.basename(fp):55} {lines} lines, {sz} bytes")

        print(f"\n  Main results: {self.csv_path}")
        print(f"  Msg stats:    {self.energy_logger.message_stats_file}")
        return self.csv_path, self.energy_logger.message_stats_file



# Visualization 
def ask_visualization_iterations(total_iterations: int) -> set:
    print("\n" + "=" * 60)
    print("VISUALIZATION CONFIGURATION")
    print("=" * 60)
    print(f" Total iterations to run: {total_iterations}")
    print(" Generating an animation costs extra time (rendering frames).")
    print(" You can choose to animate a subset of iterations.\n")
    print(" Options:")
    print(" Enter 0 : No visualization for any iteration")
    print(" Enter 1 : Only the FIRST iteration")
    print(f" Enter {total_iterations} : All iterations")
    raw = input("Your choice: ").strip()
    if raw == '0':
        print(" No visualization will be generated.")
        return set()
    if ',' in raw:
        try:
            chosen = {int(x.strip()) - 1 for x in raw.split(',') if x.strip()}
            chosen = {i for i in chosen if 0 <= i < total_iterations}
            print(f" Visualizing iterations: {sorted(i+1 for i in chosen)}")
            return chosen
        except ValueError:
            pass
    try:
        n      = int(raw)
        n      = max(0, min(n, total_iterations))
        chosen = set(range(n))
        if chosen:
            print(f" Visualizing first {n} iteration(s).")
        else:
            print(" No visualization will be generated.")
        return chosen
    except ValueError:
        print("  Unrecognised input -> no visualization.")
        return set()



# Experiment runner
def run__simulation(exp_name: str, failure_mode, config):
    print(f"\n{'='*80}")
    print(f"UNIFIED SIMULATION: {exp_name.upper()}")
    print(f"Iterations: {config.NUM_SIMULATION_RUNS}")
    print(f"Failure mode: {failure_mode if failure_mode else 'None'}")
    print(f"{'='*80}\n")

    visualize_set = ask_visualization_iterations(config.NUM_SIMULATION_RUNS)

    for attr in ('shared_csv_path', 'shared_energy_file', 'shared_message_file',
                 'shared_message_stats_file', 'shared_death_file',
                 'shared_stochastic_file', 'shared_results_file'):
        setattr(config, attr, None)

    csv_path = msg_stats_path = None

    for iteration in range(config.NUM_SIMULATION_RUNS):
        print(f"\n{'#'*80}")
        print(f"#  ITERATION {iteration + 1} / {config.NUM_SIMULATION_RUNS}")
        print(f"{'#'*80}\n")

        if iteration == 0:
            config.INITIAL_NO_UAV = config.NO_UAV
        else:
            config.NO_UAV = config.INITIAL_NO_UAV

        seed = config.RANDOM_SEED + iteration
        random.seed(seed)
        np.random.seed(seed)

        config.FAILURE_MODE = failure_mode

        sim = UnifiedUAVSimulation(
            config, exp_name, iteration,
            visualize_iterations=visualize_set)

        csv_path, msg_stats_path = sim.run()

    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE: {exp_name.upper()}")
    print(f"{'='*80}")
    print(f"\n  MAIN RESULTS CSV :  {csv_path}")
    print(f"  MESSAGE STATS CSV:  {msg_stats_path}")
    print(f"  ALL OUTPUTS      :  ./results/")

    if visualize_set:
        print(f"  ANIMATION : visualizations/simulation_animation.gif")
        print(f"  (iterations: {sorted(i+1 for i in visualize_set)})")
    else:
        print(f"  ANIMATION : not generated")

    if csv_path and os.path.exists(csv_path):
        print(f"\nMAIN CSV PREVIEW:")
        try:
            df = pd.read_csv(csv_path)
            print(f"  Rows: {len(df)}  |  Iterations: {df['iteration'].max()+1}")
            print(f"\n  First 5 rows:")
            print(df.head(5).to_string(index=False))
            print(f"\n  Last 5 rows:")
            print(df.tail(5).to_string(index=False))
        except Exception as e:
            print(f"  Could not read CSV: {e}")

    print(f"\n{'='*80}\n")



if __name__ == "__main__":
    config       = Config()
    failure_mode = None
    exp_name     = "NoFailure_SimPy_"
    run__simulation(exp_name, failure_mode, config)