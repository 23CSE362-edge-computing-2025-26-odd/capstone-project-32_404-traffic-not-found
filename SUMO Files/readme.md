#  Traffic Signal Controller with SUMO, ANN & PPO

This repository contains implementations of **traffic light control strategies** using SUMO simulation, **Artificial Neural Networks (ANN)** for risk prediction, and **Reinforcement Learning (PPO)** for adaptive decision-making.  
The project compares **fixed-time control**, **hybrid ANN + PPO control**.

---

# Project Files

- **control_ftc.py** → Implements a **Fixed-Time Control (FTC)** traffic signal baseline.  
- **control_hybrid_with_ann.py** → Hybrid controller using **ANN risk detection** + **PPO reinforcement agent**.  
- **control_osm_train_ppo.py** → Script to train a PPO agent on the SUMO environment.  
- **ftc_history.csv** → Simulation logs for fixed-time baseline.  
- **traffic_history.csv** → Simulation logs for hybrid ANN+PPO control.  
- **osm.sumocfg** → SUMO configuration file for the network.  
- **myroutes.rou.xml** → Vehicle route definitions for the SUMO simulation.    
- **readme.md** → Project documentation.

---

# Running the project

-Fixed-Time Control (baseline):
   python control_ftc.py 


-Hybrid ANN + PPO Control:
   python control_hybrid_with_ann.py --cfg osm.sumocfg --gui --ann traffic_ann_best.pt --yellow 30 --threshold 0.8 --steps 100


-Train PPO Agent:
   python control_osm_train_ppo.py

