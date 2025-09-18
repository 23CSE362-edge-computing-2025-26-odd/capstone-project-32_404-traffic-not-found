import traci
import csv

# Start SUMO with GUI
sumoBinary = "sumo-gui"
sumoCmd = [sumoBinary, "-c", "osm.sumocfg"]
traci.start(sumoCmd)

# Pick first traffic light
tls_id = traci.trafficlight.getIDList()[0]
phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases
cycle = list(range(len(phases)))   # match number of phases

# Simulation setup
phase_index = 0
step = 0
MAX_STEPS = 800

# Metrics
waiting_times = []
time_losses = []
throughputs = []

while step < MAX_STEPS:
    traci.simulationStep()

    # set phase
    traci.trafficlight.setPhase(tls_id, cycle[phase_index])

    if step % 20 == 0:  # change every 20 steps
        phase_index = (phase_index + 1) % len(cycle)

    veh_ids = traci.vehicle.getIDList()

    # Average waiting time
    if len(veh_ids) > 0:
        total_wait = sum(traci.vehicle.getWaitingTime(vid) for vid in veh_ids)
        avg_wait = total_wait / len(veh_ids)
    else:
        avg_wait = 0.0
    waiting_times.append(avg_wait)

    # Average time loss
    if len(veh_ids) > 0:
        total_loss = sum(traci.vehicle.getTimeLoss(vid) for vid in veh_ids)
        avg_loss = total_loss / len(veh_ids)
    else:
        avg_loss = 0.0
    time_losses.append(avg_loss)

    # Throughput = vehicles that already arrived
    throughput = traci.simulation.getArrivedNumber()
    throughputs.append(throughput)

    step += 1

traci.close()

# Save results to CSV
csv_file = "ftc_history.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Step", "AvgWaitingTime", "AvgTimeLoss", "Throughput"])
    for i in range(len(waiting_times)):
        writer.writerow([i, waiting_times[i], time_losses[i], throughputs[i]])

print(f"[FTC] finished. Results saved to {csv_file}")

