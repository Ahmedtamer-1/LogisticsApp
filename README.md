# Smart Logistics Optimization Platform 🚚

An interactive, Python-based desktop application built with Tkinter and Matplotlib that visualizes and compares essential logistics and graph algorithms in real-time.

## Features

This platform provides an intuitive interface to demonstrate and analyze four key algorithm categories applied to logistics and supply chain optimization:

- **Shortest Paths (Dijkstra's Algorithm):** Calculates and visualizes the most efficient delivery routes from a selected source node to all other locations.
- **Resource Allocation (0/1 Knapsack - Dynamic Programming):** Optimizes the selection of delivery orders based on weight constraints and value maximization.
- **Network Optimization (Minimum Spanning Tree):** 
  - **Prim's Algorithm:** Uses a greedy, priority-queue approach to construct an MST.
  - **Kruskal's Algorithm:** Uses edge sorting and a Union-Find data structure to build the MST.
- **Vehicle Loading (Greedy Bin Packing - First-Fit Decreasing):** Approximates the optimal distribution of orders across multiple delivery vehicles with defined capacities.

The application also features a comprehensive **Algorithm Comparison** tab that generates performance metrics (Execution Time, Approximation Ratios, Time Complexity) to analyze the trade-offs between exact and exact approximate computational methods.

## Requirements

The project uses Python 3. Make sure you have the following dependencies installed:

```bash
pip install matplotlib networkx numpy
```

*(Note: Tkinter is typically included with standard Python installations, but depending on your OS, you may need to install it separately via your system's package manager).*

## How to Run

Clone the repository and run the main Python script:

```bash
git clone https://github.com/Ahmedtamer-1/LogisticsApp.git
cd LogisticsApp
python LogisticsApp.py
```

## Interface Overview

- **Toolbar:** Adjust the number of nodes, edges, and vehicle capacity. Use the "Randomize" button to generate a new graph, and "Run All" to execute all algorithms.
- **Step-by-Step Trace:** Use the trace controls to step through the execution of individual algorithms and observe their decision-making processes.
- **Visualizations:** The graph tab displays a network diagram of suppliers, warehouses, and customers. Other tabs provide visual breakdowns of algorithm execution, including dynamic programming tables and truck loading bins.
