# Virtual Memory Management Simulator

A comprehensive tool for visualizing and understanding virtual memory management concepts including paging, page replacement algorithms, and memory allocation.

## Features

- Interactive visualization of memory states and page tables
- Multiple page replacement algorithms:
  - LRU (Least Recently Used)
  - Optimal Replacement
- Real-time visualization of:
  - Memory allocation
  - Page faults
  - Page fault rates
  - Access patterns
- Configurable parameters:
  - Memory frame size
  - Page size
  - Custom page reference sequences

## Project Structure

```
virtual_memory_sim/
├── src/
│   ├── algorithms/
│   │   └── page_replacement.py
│   ├── visualization/
│   │   └── memory_viz.py
│   └── main.py
├── tests/
│   └── test_page_replacement.py
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

From the project root directory:
```bash
streamlit run src/main.py
```

## Running Tests

From the project root directory:
```bash
python -m pytest tests/
```

## Usage

1. Configure Memory Parameters:
   - Set the number of memory frames (4-20)
   - Adjust page size (1-16 KB)

2. Input Page Reference Sequence:
   - Enter comma-separated page numbers
   - Example: 1,2,3,4,1,2,5,1,2,3,4,5

3. Select Algorithm:
   - Choose between LRU and Optimal replacement

4. Run Simulation:
   - Click "Run Simulation" to see the results
   - View memory state visualization
   - Analyze page fault patterns
   - Review simulation history

## Contributing

Feel free to submit issues and enhancement requests!
