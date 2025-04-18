import tkinter as tk
from tkinter import ttk, messagebox
from collections import OrderedDict
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

class MemorySimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Memory Simulator")
        self.root.geometry("1000x700")
        
        # Configure the style for a futuristic green on black theme
        self.configure_style()
        
        # Create main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Input frame
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=10)
        
        # Parameter input
        ttk.Label(input_frame, text="Number of Frames:").grid(row=0, column=0, padx=5, pady=5)
        self.frames_var = tk.StringVar(value="4")
        ttk.Entry(input_frame, textvariable=self.frames_var, width=5).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Page Sequence:").grid(row=0, column=2, padx=5, pady=5)
        self.sequence_var = tk.StringVar(value="1,2,3,4,1,2,5,1,2,3,4,5")
        ttk.Entry(input_frame, textvariable=self.sequence_var, width=30).grid(row=0, column=3, padx=5, pady=5)
        
        # Algorithm selection
        algo_frame = ttk.Frame(main_frame)
        algo_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(algo_frame, text="Algorithm:").pack(side=tk.LEFT, padx=5)
        self.algorithm_var = tk.StringVar(value="LRU")
        ttk.Radiobutton(algo_frame, text="LRU", variable=self.algorithm_var, value="LRU").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(algo_frame, text="Optimal", variable=self.algorithm_var, value="Optimal").pack(side=tk.LEFT, padx=5)
        
        # Run button
        ttk.Button(algo_frame, text="Run Simulation", command=self.run_simulation).pack(side=tk.LEFT, padx=20)
        
        # Create visualization area
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={'height_ratios': [2, 1]})
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Results text
        self.results_text = tk.Text(main_frame, height=8, width=80, bg="black", fg="#33ff33")
        self.results_text.pack(fill=tk.X, pady=10)

    def configure_style(self):
        # Set up the green-on-black theme
        style = ttk.Style()
        
        # Define colors
        bg_color = "black"
        fg_color = "#33ff33"  # Bright green
        
        # Configure ttk styles
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color)
        style.configure("TButton", background=bg_color, foreground=fg_color)
        style.configure("TRadiobutton", background=bg_color, foreground=fg_color)
        style.configure("TEntry", fieldbackground=bg_color, foreground=fg_color)
        
        # Configure root window
        self.root.configure(bg=bg_color)
        
        # Configure matplotlib style
        plt.style.use('dark_background')

    def run_simulation(self):
        try:
            frames = int(self.frames_var.get())
            page_sequence = [int(x.strip()) for x in self.sequence_var.get().split(',') if x.strip()]
            
            if not page_sequence:
                messagebox.showerror("Error", "Please enter a valid page sequence")
                return
                
            algorithm = PageReplacementAlgorithm(frames)
            
            if self.algorithm_var.get() == "LRU":
                faults, final_state, history = algorithm.lru_replace(page_sequence)
            else:
                faults, final_state, history = algorithm.optimal_replace(page_sequence)
                
            self.visualize_results(page_sequence, history, faults, final_state)
            
        except Exception as e:
            messagebox.showerror("Error", f"Simulation error: {str(e)}")
    
    def visualize_results(self, sequence, history, faults, final_state):
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot memory states
        states = np.array([h['state'] for h in history])
        im = self.ax1.imshow(states.T, aspect='auto', cmap='viridis')
        
        # Add colorbar
        plt.colorbar(im, ax=self.ax1)
        
        # Add labels
        self.ax1.set_title('Memory State Over Time', color='#33ff33')
        self.ax1.set_xlabel('Access Sequence')
        self.ax1.set_ylabel('Frame Number')
        
        # Plot page faults
        faults_array = [1 if h['fault'] else 0 for h in history]
        self.ax2.plot(faults_array, '-o', color='#33ff33')
        self.ax2.set_title('Page Faults', color='#33ff33')
        self.ax2.set_xlabel('Access Sequence')
        self.ax2.set_ylabel('Fault (1) / Hit (0)')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Update text results
        self.results_text.delete(1.0, tk.END)
        
        total_faults = sum(faults_array)
        fault_rate = (total_faults / len(sequence)) * 100
        
        self.results_text.insert(tk.END, f"Algorithm: {self.algorithm_var.get()}\n")
        self.results_text.insert(tk.END, f"Page Sequence: {', '.join(map(str, sequence))}\n")
        self.results_text.insert(tk.END, f"Total Frames: {len(final_state)}\n")
        self.results_text.insert(tk.END, f"Page Faults: {total_faults}\n")
        self.results_text.insert(tk.END, f"Page Fault Rate: {fault_rate:.2f}%\n")
        self.results_text.insert(tk.END, f"Final Memory State: {final_state}\n")

def main():
    root = tk.Tk()
    app = MemorySimulator(root)
    root.mainloop()

if __name__ == "__main__":
    main() 