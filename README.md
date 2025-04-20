# Virtual Memory Simulator

A simulation tool for visualizing and comparing different page replacement algorithms used in virtual memory management.

## Overview

This project provides:
1. A visual simulation of page replacement algorithms (LRU and Optimal)
2. The ability to select real running processes and generate page access sequences based on their memory usage
3. Visualization of memory state transitions and page fault statistics

## Prerequisites

- Python 3.6+
- Required packages listed in `requirements.txt`

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/VMO.git
cd VMO
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Running the Application

### Option 1: Web Application (with Streamlit)

```
streamlit run original_web_app.py
```

This version includes interactive web-based visualizations using Streamlit and Plotly.

### Option 2: Desktop GUI (with Tkinter)

```
python vm_simulator.py
```

This version includes interactive graphs and visualizations using Matplotlib in a desktop application.

### Option 3: Simple Application

```
python simple_app.py
```

The simple version provides all the same features but uses a text-based output instead of matplotlib graphs.

## Features

- **Process Selection**: Choose from currently running processes on your system
- **Memory Frame Configuration**: Set the number of frames for simulation
- **Algorithm Selection**: Choose between LRU (Least Recently Used) and Optimal page replacement algorithms
- **Custom Page Sequences**: Manually enter page sequences or generate based on real process memory usage
- **Memory State Visualization**: See how memory frames change with each page access
- **Statistics**: Compare page fault rates and efficiency between algorithms
- **Interactive Interface**: Visualize page tables and memory allocation in real-time
- **Educational Tool**: Understand how virtual memory management works through visualization

## Usage Tips

1. Use the "Refresh" button to update the list of running processes
2. Select a process to generate a page sequence based on its memory usage
3. Adjust the number of memory frames as needed
4. Choose an algorithm (LRU or Optimal)
5. Click "Run Simulation" to see the results
6. Optionally, enter your own page sequence (comma-separated numbers) in the input field

## Troubleshooting

If you encounter issues with the matplotlib version:
- Try running the web-based version (`streamlit run original_web_app.py`)
- Check that all dependencies are installed correctly
- Ensure you have appropriate permissions to monitor system processes

