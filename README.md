
# NS2 Network Performance Dashboard

This project focuses on analyzing the performance of different TCP congestion control algorithms—**TCP Reno**, **TCP Tahoe**, and **TCP Vegas**—using **Network Simulator 2 (NS2)**. The goal is to evaluate and compare throughput, packet loss, latency, and congestion window behavior under various simulated network conditions.
---

## Project Structure

- **TCL Scripts (`.tcl`)**: Define the network topology and traffic for each TCP variant.
  - `tcp_reno.tcl`, `tcp_tahoe.tcl`, `tcp_vegas.tcl`, `tcp_100nodes.tcl`, etc.
- **NAM/Trace Files (`.nam`, `.tr`)**: Output files from NS2 simulations.
  - Example: `100nodes-reno.nam`, `out-reno.tr`
- **Python Scripts**:
  - `calculate_metrics_reno.py`, `calculate_metrics_tahoe.py`, `calculate_metrics_vegas.py`: Parse trace files to extract performance metrics.
  - `dashboard.py`: Provides comparative plots for all TCP variants.
- **XGraph Files (`.xg`)**: Contains throughput, latency, packet drop, and cwnd data for visualization.
  - Example: `throughput-reno.xg`, `cwnd-tahoe.xg`, `latency-vegas.xg`, `packet_drop_data.xg`

## 🚀 Features
- Simulates complex topologies (including 100-node configurations)
- 📈 Real-time dashboard showing:
  - **Latency**
  - **Throughput**
  - **Packet Drop**
  - **Congestion Window**
- Automatically generates `.xg` files for plotting with Xgraph
- Easy metric comparison using `dashboard.py`
- 🎥 NS2 + NAM animation for a 100-node network topology
- 📊 Separate metric calculations for:
  - **TCP Tahoe**
  - **TCP Reno**
  - **TCP Vegas**
- 🧮 Custom `calculate_metric` module for data extraction and analysis

## How to Use
1. Run simulation scripts with NS2:
   ns tcp_reno.tcl
   ns tcp_tahoe.tcl
   ns tcp_vegas.tcl
   
3. Extract metrics: 
   python calculate_metrics_reno.py
   python calculate_metrics_tahoe.py
   python calculate_metrics_vegas.py

3.Visualize using Xgraph:
   python dashboard.py

## Requirements
1)NS2
2)Xgraph (4.38 included for Windows & Linux)
3)Python 3.x
4)matplotlib library (for dashboard.py)
5)Xlaunch (for stimulations,graph and nam files)
