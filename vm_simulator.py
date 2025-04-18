import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
from collections import OrderedDict
import numpy as np
import subprocess
import re
import random

# Configure matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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

class VirtualMemorySimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Memory Management Simulator")
        self.root.geometry("1200x900")
        # Neo-inspired color scheme
        self.BACKGROUND_COLOR = '#0c1021'  # Dark blue background
        self.ACCENT_COLOR = '#00ffff'      # Cyan accent
        self.SECONDARY_COLOR = '#007acc'   # Lighter blue
        self.TEXT_COLOR = '#ffffff'        # White text
        self.HIGHLIGHT_COLOR = '#1e90ff'   # Bright blue highlight
        
        self.root.configure(bg=self.BACKGROUND_COLOR)
        
        # Configure styles
        style = ttk.Style()
        style.configure('TFrame', background=self.BACKGROUND_COLOR)
        style.configure('TLabel', background=self.BACKGROUND_COLOR, foreground=self.ACCENT_COLOR)
        style.configure('TButton', background=self.SECONDARY_COLOR, foreground=self.TEXT_COLOR)
        style.configure('TRadiobutton', background=self.BACKGROUND_COLOR, foreground=self.ACCENT_COLOR)
        style.configure('TEntry', fieldbackground=self.SECONDARY_COLOR, foreground=self.TEXT_COLOR)
        
        # Configure matplotlib style
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': self.BACKGROUND_COLOR,
            'axes.facecolor': self.BACKGROUND_COLOR,
            'grid.color': self.SECONDARY_COLOR,
            'text.color': self.ACCENT_COLOR,
            'axes.labelcolor': self.ACCENT_COLOR,
            'axes.edgecolor': self.ACCENT_COLOR,
            'xtick.color': self.ACCENT_COLOR,
            'ytick.color': self.ACCENT_COLOR
        })
        
        # Create main frames
        self.control_frame = ttk.Frame(root, padding="10")
        self.control_frame.pack(fill=tk.X)
        
        self.viz_frame = ttk.Frame(root, padding="10")
        self.viz_frame.pack(fill=tk.BOTH, expand=True)
        
        self.setup_controls()
        self.setup_visualization()
        
    def setup_controls(self):
        # Create a tech-styled frame
        control_inner_frame = ttk.Frame(self.control_frame, style='TFrame')
        control_inner_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Add a title label with tech font
        title_label = ttk.Label(control_inner_frame, 
                               text="VIRTUAL MEMORY SIMULATOR v1.0",
                               font=('Courier', 16, 'bold'),
                               foreground=self.ACCENT_COLOR)
        subtitle_label = ttk.Label(control_inner_frame,
                                  text="[ MATRIX SYSTEMS INITIALIZED ]",
                                  font=('Courier', 10),
                                  foreground=self.SECONDARY_COLOR)
        subtitle_label.pack(pady=(0, 10))
        title_label.pack(pady=(0, 20))
        
        # Frames input
        ttk.Label(self.control_frame, text="Number of Frames:").pack(side=tk.LEFT, padx=5)
        self.frames_var = tk.StringVar(value="4")
        ttk.Entry(self.control_frame, textvariable=self.frames_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Page sequence input
        ttk.Label(self.control_frame, text="Page Sequence (comma-separated):").pack(side=tk.LEFT, padx=5)
        self.sequence_var = tk.StringVar(value="1,2,3,4,1,2,5,1,2,3,4,5")
        ttk.Entry(self.control_frame, textvariable=self.sequence_var, width=30).pack(side=tk.LEFT, padx=5)
        
        # Algorithm selection
        self.algorithm_var = tk.StringVar(value="LRU")
        ttk.Radiobutton(self.control_frame, text="LRU", variable=self.algorithm_var, value="LRU").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(self.control_frame, text="Optimal", variable=self.algorithm_var, value="Optimal").pack(side=tk.LEFT, padx=5)
        
        # Run button
        ttk.Button(self.control_frame, text="Run Simulation", command=self.run_simulation).pack(side=tk.LEFT, padx=5)
        
    def setup_visualization(self):
        # Create matplotlib figure with dark theme
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor='#1e1e1e')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Results text with tech styling
        self.results_text = tk.Text(self.viz_frame, height=6, width=70,
                                  bg=self.SECONDARY_COLOR, fg=self.TEXT_COLOR,
                                  font=('Courier', 10),
                                  relief=tk.RIDGE,
                                  insertbackground=self.ACCENT_COLOR)
        self.results_text.pack(fill=tk.X, padx=20, pady=10)
        
        # Add a border effect
        border_frame = ttk.Frame(self.viz_frame, style='TFrame')
        border_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="System Ready")
        status_label = ttk.Label(border_frame, 
                                textvariable=self.status_var,
                                font=('Courier', 10),
                                foreground=self.ACCENT_COLOR)
        status_label.pack(side=tk.RIGHT)
        
    def run_simulation(self):
        try:
            print("DEBUG: Starting simulation")
            # Set flag to prevent refresh during simulation
            self.simulation_running = True
            
            # Clear previous results first
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Running simulation...\n")
            self.root.update()
            
            frames = int(self.frames_var.get())
            print(f"DEBUG: Using {frames} frames")
            
            # Use manually entered sequence
            sequence = [int(x.strip()) for x in self.sequence_var.get().split(',') if x.strip()]
            
            if not sequence:
                print("DEBUG: No sequence provided")
                messagebox.showerror("Error", "No page sequence provided")
                self.simulation_running = False
                return
                
            print(f"DEBUG: Final sequence length: {len(sequence)}")    
            algorithm = PageReplacementAlgorithm(frames)
            
            print(f"DEBUG: Running {self.algorithm_var.get()} algorithm")
            if self.algorithm_var.get() == "LRU":
                faults, final_state, history = algorithm.lru_replace(sequence)
            else:
                faults, final_state, history = algorithm.optimal_replace(sequence)
            
            print("DEBUG: Simulation completed, visualizing results")
            self.visualize_results(sequence, history, faults, final_state)
            
            # Simulation is done, allow refreshes again
            self.simulation_running = False
            
        except Exception as e:
            print(f"DEBUG: Error in run_simulation: {str(e)}")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Simulation error: {str(e)}\n")
            messagebox.showerror("Error", f"Simulation error: {str(e)}")
            self.simulation_running = False
            return
        
        algorithm = PageReplacementAlgorithm(frames)
        
        if self.algorithm_var.get() == "LRU":
            faults, final_state, history = algorithm.lru_replace(sequence)
        else:
            faults, final_state, history = algorithm.optimal_replace(sequence)
            
        self.visualize_results(sequence, history, faults, final_state)
        
    def visualize_results(self, sequence, history, faults, final_state):
        print("DEBUG: Starting visualization")
        self.status_var.set("Processing simulation...")
        self.root.update()
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot memory state changes with custom colormap
        states = np.array([h['state'] for h in history])
        im = self.ax1.imshow(states.T, aspect='auto', cmap='Blues_r')
        self.ax1.set_title('Memory State Over Time')
        self.ax1.set_xlabel('Access Sequence')
        self.ax1.set_ylabel('Frame Number')
        
        # Plot page faults
        self.ax2.plot(faults, '-o', color=self.ACCENT_COLOR, label='Page Fault',
                     markerfacecolor=self.HIGHLIGHT_COLOR)
        self.ax2.set_title('Page Faults')
        self.ax2.set_xlabel('Access Sequence')
        self.ax2.set_ylabel('Fault (1) / Hit (0)')
        self.ax2.grid(True)
        
        self.fig.tight_layout()
        self.canvas.draw()
        self.status_var.set(f"Simulation completed - {sum(faults)} page faults detected")
        
        # Update results text
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Total Page Faults: {sum(faults)}\n")
        self.results_text.insert(tk.END, f"Page Fault Rate: {(sum(faults)/len(faults))*100:.2f}%\n")
        self.results_text.insert(tk.END, f"Final Memory State: {final_state}\n")
        self.results_text.insert(tk.END, f"Access Sequence: {' → '.join(map(str, sequence))}\n")

# Main function to run the GUI app
def main():
    root = ThemedTk(theme="equilux")
    app = VirtualMemorySimulator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
