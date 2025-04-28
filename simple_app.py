
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import subprocess
import re
import random
from collections import OrderedDict

class PageReplacementAlgorithm:
    def __init__(self, total_frames):
        self.total_frames = total_frames
        self.memory = [-1] * total_frames
        self.page_faults = 0
        self.history = []

    def lru_replace(self, page_sequence):
        self.memory = [-1] * self.total_frames
        self.page_faults = 0
        access_history = OrderedDict()
        self.history = []

        for page in page_sequence:
            current_state = self.memory.copy()
            if page not in self.memory:
                self.page_faults += 1
                if -1 in self.memory:
                    # Empty frame available
                    idx = self.memory.index(-1)
                else:
                    # Find least recently used
                    lru_page = next(iter(access_history))
                    idx = self.memory.index(lru_page)
                    del access_history[lru_page]
                self.memory[idx] = page
            
            # Update access history
            if page in access_history:
                del access_history[page]
            access_history[page] = True
            
            self.history.append({
                'page': page,
                'state': current_state.copy(),
                'fault': page not in current_state
            })

        return self.page_faults, self.memory, self.history

    def optimal_replace(self, page_sequence):
        self.memory = [-1] * self.total_frames
        self.page_faults = 0
        self.history = []

        for i, page in enumerate(page_sequence):
            current_state = self.memory.copy()
            if page not in self.memory:
                self.page_faults += 1
                if -1 in self.memory:
                    # Empty frame available
                    idx = self.memory.index(-1)
                else:
                    # Find optimal victim
                    future_use = {}
                    for p in self.memory:
                        try:
                            future_use[p] = page_sequence[i:].index(p)
                        except ValueError:
                            future_use[p] = float('inf')
                    victim = max(future_use.items(), key=lambda x: x[1])[0]
                    idx = self.memory.index(victim)
                self.memory[idx] = page

            self.history.append({
                'page': page,
                'state': current_state.copy(),
                'fault': page not in current_state
            })

        return self.page_faults, self.memory, self.history

class SimpleMemorySimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Virtual Memory Simulator")
        self.root.geometry("900x700")
        
        # Neo-inspired color scheme
        self.BACKGROUND_COLOR = '#0c1021'  # Dark blue background
        self.ACCENT_COLOR = '#00ffff'      # Cyan accent
        self.SECONDARY_COLOR = '#007acc'   # Lighter blue
        self.TEXT_COLOR = '#ffffff'        # White text
        
        self.root.configure(bg=self.BACKGROUND_COLOR)
        
        # Process tracking variables
        self.process_list = []
        self.last_selected_pid = None
        
        # Configure styles
        style = ttk.Style()
        if hasattr(style, 'theme_use'):
            try:
                style.theme_use('clam')  # Most basic theme available on all platforms
            except:
                pass  # Fallback to default if theme not available
                
        # Set colors        
        style.configure('TFrame', background=self.BACKGROUND_COLOR)
        style.configure('TLabel', background=self.BACKGROUND_COLOR, foreground=self.TEXT_COLOR)
        style.configure('TButton', background=self.SECONDARY_COLOR)
        style.configure('TRadiobutton', background=self.BACKGROUND_COLOR, foreground=self.TEXT_COLOR)
        
        # Create main container
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(header_frame, 
                              text="VIRTUAL MEMORY SIMULATOR", 
                              font=('Courier', 20, 'bold'),
                              bg=self.BACKGROUND_COLOR,
                              fg=self.ACCENT_COLOR)
        title_label.pack(pady=10)
        
        # Create controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Process selection
        process_frame = ttk.Frame(controls_frame)
        process_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(process_frame, text="Select Process:").pack(side=tk.LEFT, padx=5)
        self.selected_process = tk.StringVar()
        self.process_dropdown = ttk.Combobox(process_frame, 
                                           textvariable=self.selected_process,
                                           width=50)
        self.process_dropdown.pack(side=tk.LEFT, padx=5, expand=True)
        
        # Refresh button
        ttk.Button(process_frame, 
                 text="Refresh", 
                 command=self.update_process_list).pack(side=tk.LEFT, padx=5)
        
        # Simulation parameters frame
        param_frame = ttk.Frame(controls_frame)
        param_frame.pack(fill=tk.X, pady=10)
        
        # Frames
        ttk.Label(param_frame, text="Memory Frames:").pack(side=tk.LEFT, padx=5)
        self.frames_var = tk.StringVar(value="4")
        ttk.Entry(param_frame, textvariable=self.frames_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Page sequence
        ttk.Label(param_frame, text="Page Sequence (optional):").pack(side=tk.LEFT, padx=5)
        self.sequence_var = tk.StringVar(value="")
        ttk.Entry(param_frame, textvariable=self.sequence_var, width=30).pack(side=tk.LEFT, padx=5)
        
        # Algorithm selection
        ttk.Label(param_frame, text="Algorithm:").pack(side=tk.LEFT, padx=5)
        self.algorithm_var = tk.StringVar(value="LRU")
        ttk.Radiobutton(param_frame, text="LRU", variable=self.algorithm_var, value="LRU").pack(side=tk.LEFT)
        ttk.Radiobutton(param_frame, text="Optimal", variable=self.algorithm_var, value="Optimal").pack(side=tk.LEFT)
        
        # Run button
        run_frame = ttk.Frame(controls_frame)
        run_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(run_frame, 
                 text="Run Simulation", 
                 command=self.run_simulation).pack(anchor=tk.CENTER, pady=10)
        
       
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Results text
        self.results_text = tk.Text(results_frame, 
                                   width=80, 
                                   height=20, 
                                   bg=self.SECONDARY_COLOR,
                                   fg=self.TEXT_COLOR,
                                   font=('Courier', 12))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, 
                               textvariable=self.status_var)
        status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        # Initialize process list
        self.update_process_list()
        
        # Bind selection event
        self.process_dropdown.bind("<<ComboboxSelected>>", self.on_process_selected)
        
    def update_process_list(self):
        """Update the list of running processes"""
        try:
            # Run ps command to get process information
            result = subprocess.run(
                ['ps', '-eo', 'pid,pcpu,pmem,comm'], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                messagebox.showerror("Error", "Failed to get process list")
                return
                
            self.process_list = []
            # Parse the output, skipping the header line
            for line in result.stdout.strip().split('\n')[1:]:
                parts = re.split(r'\s+', line.strip(), maxsplit=3)
                if len(parts) >= 4:
                    pid, cpu, mem, name = parts
                    # Extract the base name from path
                    base_name = name.split('/')[-1]
                    self.process_list.append({
                        'pid': pid,
                        'cpu': cpu,
                        'mem': mem,
                        'name': base_name
                    })
            
            # Sort by memory usage (highest first)
            self.process_list = sorted(self.process_list, key=lambda x: float(x['mem']), reverse=True)
            
            # Update dropdown values
            process_values = [f"{p['pid']} - {p['name']} (CPU: {p['cpu']}%, MEM: {p['mem']}%)" 
                            for p in self.process_list]
            
            # Update combobox
            self.process_dropdown['values'] = process_values
            if process_values and not self.selected_process.get():
                self.process_dropdown.current(0)
                
            self.status_var.set(f"Found {len(self.process_list)} processes")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to get process list: {str(e)}")
            
    def on_process_selected(self, event):
        
        selection = self.selected_process.get()
        if selection:
            try:
                pid = selection.split(' - ')[0].strip()
                self.last_selected_pid = pid
                # Clear the sequence field to use process-based sequence
                self.sequence_var.set("")
                self.status_var.set(f"Selected process: {selection}")
            except Exception as e:
                self.status_var.set(f"Error in selection: {str(e)}")
                
    def get_process_memory_pages(self, pid):
        """Generate simulated memory page accesses based on actual process memory info"""
        try:
     
            result = subprocess.run(
                ['ps', '-o', 'pid,rss,vsz,command', '-p', pid], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                raise ValueError(f"Failed to get info for PID {pid}")
                
           
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                raise ValueError(f"No process found with PID {pid}")
                
            # Split the line by whitespace
            parts = re.split(r'\s+', lines[1].strip(), maxsplit=3)
            if len(parts) < 3:
                raise ValueError(f"Unexpected output format from ps command")
                
            # Extract RSS (Resident Set Size) and VSZ (Virtual Memory Size)
            rss = int(parts[1])  # In KB
            vsz = int(parts[2])  # In KB
            
            # Calculate the number of pages based on memory size
            page_size = 4096  # 4KB page size (standard on most systems)
            num_pages = min(10, max(4, int(rss / (1024 * 10))))
            
            # Generate a sequence with locality of reference based on real memory usage
            sequence = []
            hot_pages = list(range(1, min(5, num_pages) + 1))
            cold_pages = list(range(min(5, num_pages) + 1, num_pages + 1))
            
            # Ratio of RSS to VSZ gives an idea of memory usage pattern
            locality_factor = min(0.8, max(0.2, rss / vsz))
            
            # Generate sequence with locality of reference
            for i in range(30):  # Generate 30 page accesses
                if random.random() < locality_factor:
                    # Access a hot page (representing frequent access)
                    if hot_pages:
                        sequence.append(random.choice(hot_pages))
                    else:
                        sequence.append(1)  # Fallback if no hot pages
                else:
                    # Access a cold page or introduce a new page
                    if i > 20 and random.random() < 0.3:
                        # Occasionally introduce a page fault by accessing new page
                        sequence.append(num_pages + 1 + len(set(sequence)))
                    elif cold_pages:
                        sequence.append(random.choice(cold_pages))
                    else:
                        sequence.append(random.choice(range(1, num_pages + 1)))
                        
            self.status_var.set(f"Generated sequence based on PID {pid} (RSS: {rss}KB, VSZ: {vsz}KB)")
            return sequence
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showwarning("Warning", f"Couldn't get memory info for PID {pid}: {str(e)}")
            # Fallback to a default sequence
            return [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
            
    def run_simulation(self):
        """Run the page replacement simulation"""
        try:
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Running simulation...\n\n")
            self.root.update()
            
            # Get parameters
            frames = int(self.frames_var.get())
            
            # Get sequence - either from process or manual entry
            if self.selected_process.get() and not self.sequence_var.get():
                selected_pid = self.selected_process.get().split(' - ')[0].strip()
                sequence = self.get_process_memory_pages(selected_pid)
                # Update sequence field for visibility
                self.sequence_var.set(','.join(map(str, sequence)))
            else:
                # Use manually entered sequence
                sequence = [int(x.strip()) for x in self.sequence_var.get().split(',') if x.strip()]
                
            if not sequence:
                self.results_text.insert(tk.END, "Error: No page sequence provided\n")
                return
                
            # Run algorithm
            algorithm = PageReplacementAlgorithm(frames)
            
            if self.algorithm_var.get() == "LRU":
                faults, final_state, history = algorithm.lru_replace(sequence)
            else:
                faults, final_state, history = algorithm.optimal_replace(sequence)
                
            # Display results
            self.show_text_results(sequence, history, faults, final_state)
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.results_text.insert(tk.END, f"Error: {str(e)}\n")
            
    def show_text_results(self, sequence, history, faults, final_state):
        """Display text-based results"""
        self.results_text.delete(1.0, tk.END)
        
        # Get process info if available
        if self.selected_process.get():
            self.results_text.insert(tk.END, f"Process: {self.selected_process.get()}\n\n")
        
        # Basic statistics
        total_faults = sum(1 for h in history if h['fault'])
        fault_rate = (total_faults / len(history)) * 100
        
        self.results_text.insert(tk.END, f"Algorithm: {self.algorithm_var.get()}\n")
        self.results_text.insert(tk.END, f"Total Frames: {len(final_state)}\n")
        self.results_text.insert(tk.END, f"Page Faults: {total_faults}\n")
        self.results_text.insert(tk.END, f"Fault Rate: {fault_rate:.2f}%\n\n")
        
        # Show page access sequence
        self.results_text.insert(tk.END, "Access Sequence:\n")
        self.results_text.insert(tk.END, ' → '.join(map(str, sequence)) + "\n\n")
        
        # Show memory state transitions
        self.results_text.insert(tk.END, "Memory State Transitions:\n")
        self.results_text.insert(tk.END, "Step | Access | Memory State | Fault\n")
        self.results_text.insert(tk.END, "-----+--------+--------------+------\n")
        
        for i, h in enumerate(history):
            fault_mark = "X" if h['fault'] else " "
            state_str = "[" + ", ".join(str(x) if x != -1 else "_" for x in h['state']) + "]"
            self.results_text.insert(tk.END, f"{i:4d} | {h['page']:6d} | {state_str:12s} | {fault_mark}\n")
            
        # Final state
        self.results_text.insert(tk.END, f"\nFinal Memory State: {final_state}\n")
        
        # Update status
        self.status_var.set(f"Simulation completed - {total_faults} page faults detected")


def main():

    try:
        root = tk.Tk()
        app = SimpleMemorySimulator(root)
        root.mainloop()
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":

    main()
