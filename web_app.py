import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import OrderedDict
import psutil
import random
import time


st.set_page_config(
    page_title="Virtual Memory Simulator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark grey and orange theme
st.markdown("""
<style>
    .main {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    .stApp {
        background-color: #1E1E1E;
    }
    .stTextInput > div > div > input {
        background-color: #2D2D2D;
        color: #FF8C00;
    }
    .stNumberInput > div > div > input {
        background-color: #2D2D2D;
        color: #FF8C00;
    }
    .stSelectbox > div > div > div {
        background-color: #2D2D2D;
        color: #FF8C00;
    }
    h1, h2, h3 {
        color: #FF8C00;
    }
    .stButton > button {
        background-color: #FF8C00;
        color: #1E1E1E;
    }
    .stButton > button:hover {
        background-color: #FFA500;
        color: #1E1E1E;
    }
    .stSidebar {
        background-color: #2D2D2D;
    }
    .stRadio > div {
        background-color: #2D2D2D;
        padding: 15px;
        border-radius: 5px;
    }
    .stProgress > div > div > div {
        background-color: #FF8C00;
    }
    .css-145kmo2 {
        color: #E0E0E0;
    }
    .metric-container {
        background-color: #2D2D2D;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .metric-label {
        color: #FF8C00;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .metric-value {
        color: #E0E0E0;
        font-size: 1.8rem;
        margin-top: 10px;
    }
    .process-item {
        background-color: #2D2D2D;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 5px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .process-item:hover {
        background-color: #3D3D3D;
        transform: translateX(2px);
    }
    .process-name {
        color: #FF8C00;
        font-weight: bold;
    }
    .process-details {
        color: #E0E0E0;
        font-size: 0.9rem;
    }
    .selected-process {
        border-left: 4px solid #FF8C00;
    }
    /* Process list container styling */
    .process-list-container {
        max-height: 500px;
        overflow-y: auto;
        padding-right: 10px;
        margin-bottom: 20px;
        scrollbar-width: thin;
    }
    /* Manual frame input styling */
    .frame-input-container {
        background-color: #2D2D2D;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    /* Refresh button pulse animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .refresh-button:hover {
        animation: pulse 1s infinite;
    }
    /* Process selection highlighting */
    .process-select-button {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# Page title
st.title(" Virtual Memory Simulator")
st.markdown("### Visualize page replacement algorithms in virtual memory management")

# Function to get running processes
@st.cache_data(ttl=5)  # Cache for 5 seconds
def get_running_processes():
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
        try:
            # Get process info
            proc_info = proc.info
            # Check if memory_info is available
            if proc_info['memory_info'] is not None:
                # Calculate memory usage in MB
                memory_mb = proc_info['memory_info'].rss / (1024 * 1024)
                
                # Only include processes with some memory usage for relevance
                if memory_mb > 1:
                    processes.append({
                        'pid': proc_info['pid'],
                        'name': proc_info['name'],
                        'memory_mb': memory_mb,
                        'cpu_percent': proc_info['cpu_percent']
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    # Sort by memory usage (highest first)
    return sorted(processes, key=lambda x: x['memory_mb'], reverse=True)

# Function to generate page sequence from process
def generate_process_page_sequence(process_info):
    # Extract memory information
    memory_mb = process_info['memory_mb']
    
    # Calculate number of pages based on memory size
    # Scale down for visualization - minimum 4, maximum 10 pages
    num_pages = min(10, max(4, int(memory_mb / 50)))
    
    # Create "hot" pages (frequently accessed)
    hot_pages = list(range(1, min(5, num_pages) + 1))
    
    # Create "cold" pages (less frequently accessed)
    cold_pages = list(range(min(5, num_pages) + 1, num_pages + 1))
    
    # Locality factor based on memory size (larger processes tend to have more locality)
    locality_factor = min(0.8, max(0.2, 0.4 + (memory_mb / 1000)))
    
    # Generate sequence with locality of reference
    sequence = []
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
    
    return sequence

# Function to run LRU algorithm
def lru_replacement(page_sequence, frame_count):
    memory = [-1] * frame_count
    page_faults = 0
    access_history = OrderedDict()
    history = []
    
    for step, page in enumerate(page_sequence):
        current_state = memory.copy()
        is_fault = False
        
        if page not in memory:
            page_faults += 1
            is_fault = True
            if -1 in memory:
                # Empty frame available
                idx = memory.index(-1)
            else:
                # Find least recently used
                lru_page = next(iter(access_history))
                idx = memory.index(lru_page)
                del access_history[lru_page]
            memory[idx] = page
        
        # Update access history
        if page in access_history:
            del access_history[page]
        access_history[page] = True
        
        history.append({
            'step': step,
            'page': page,
            'state': memory.copy(),
            'fault': is_fault
        })
    
    return history, page_faults

# Function to run Optimal algorithm
def optimal_replacement(page_sequence, frame_count):
    memory = [-1] * frame_count
    page_faults = 0
    history = []
    
    for step, page in enumerate(page_sequence):
        current_state = memory.copy()
        is_fault = False
        
        if page not in memory:
            page_faults += 1
            is_fault = True
            if -1 in memory:
                # Empty frame available
                idx = memory.index(-1)
            else:
                # Find page that won't be used for the longest time
                future_use = {}
                for p in memory:
                    try:
                        future_use[p] = page_sequence[step+1:].index(p) + 1
                    except ValueError:
                        future_use[p] = float('inf')
                
                victim = max(future_use.items(), key=lambda x: x[1])[0]
                idx = memory.index(victim)
            
            memory[idx] = page
        
        history.append({
            'step': step,
            'page': page,
            'state': memory.copy(),
            'fault': is_fault
        })
    
    return history, page_faults

# Function to create memory state visualization
def create_memory_state_fig(history, frame_count):
   
    states = np.array([list(h['state']) for h in history])
    
    # Create a dataframe for visualization
    data = []
    for step in range(len(history)):
        for frame in range(frame_count):
            page = states[step, frame]
            is_fault = history[step]['fault']
            data.append({
                'Step': step,
                'Frame': f"Frame {frame}",
                'Page': page if page != -1 else "Empty",
                'Page_Int': page,
                'Reference': history[step]['page'],
                'Is_Fault': is_fault
            })
    
    df = pd.DataFrame(data)
    
    # Create heatmap using plotly
    fig = px.density_heatmap(
        df, 
        x='Step', 
        y='Frame', 
        z='Page_Int', 
        color_continuous_scale=[(0, "#2D2D2D"), (0.5, "#805940"), (1, "#FF8C00")],
        labels={'Step': 'Memory Access Step', 'Frame': 'Memory Frame', 'Page_Int': 'Page Number'},
    )
    
    # Add text annotations for page numbers
    for i, row in df.iterrows():
        text = 'Empty' if row['Page'] == 'Empty' else str(int(row['Page']))
        fig.add_annotation(
            x=row['Step'],
            y=row['Frame'],
            text=text,
            showarrow=False,
            font=dict(color="white" if row['Page'] == 'Empty' else "black")
        )
    
    # Customize layout
    fig.update_layout(
        title='Memory State Visualization',
        paper_bgcolor='#1E1E1E',
        plot_bgcolor='#1E1E1E',
        font=dict(color='#E0E0E0'),
        height=400,
        coloraxis_showscale=False
    )
    
    return fig


def create_page_fault_fig(history):
    df = pd.DataFrame(history)
    df['step'] = df.index
    
    # Create figure
    fig = go.Figure()
    
    # Add page references
    fig.add_trace(go.Scatter(
        x=df['step'],
        y=df['page'],
        mode='lines+markers',
        name='Page Reference',
        line=dict(color='#FF8C00', width=2),
        marker=dict(size=8, color='#FF8C00')
    ))
    
    # Add page faults
    fault_df = df[df['fault']].copy()
    fig.add_trace(go.Scatter(
        x=fault_df['step'],
        y=fault_df['page'],
        mode='markers',
        name='Page Fault',
        marker=dict(
            symbol='x',
            size=12,
            color='#FF5555',
            line=dict(width=2, color='#FF5555')
        )
    ))
    
    # Customize layout
    fig.update_layout(
        title='Page References and Faults',
        xaxis_title='Access Step',
        yaxis_title='Page Number',
        paper_bgcolor='#1E1E1E',
        plot_bgcolor='#1E1E1E',
        font=dict(color='#E0E0E0'),
        legend=dict(
            x=0,
            y=1.1,
            orientation='h',
            bgcolor='#2D2D2D',
            font=dict(color='#E0E0E0')
        ),
        height=300
    )
    
    # Grid lines
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#2D2D2D'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#2D2D2D'
    )
    
    return fig

# Function to display memory stack
def display_memory_stack(history, frame_count):
    if not history:
        return
    
    # Get the final state of memory
    final_state = history[-1]['state']
    
    # Create a figure
    fig = go.Figure()
    
    # Add memory frames
    y_pos = list(range(frame_count))
    colors = ['#FF8C00' if page != -1 else '#2D2D2D' for page in final_state]
    
    fig.add_trace(go.Bar(
        x=[1] * frame_count,
        y=y_pos,
        orientation='h',
        marker_color=colors,
        text=[f'Page {page}' if page != -1 else 'Empty' for page in final_state],
        textposition='inside',
        width=0.8
    ))
    
    # Customize layout
    fig.update_layout(
        title='Final Memory Stack',
        xaxis_title='',
        yaxis_title='Frame Number',
        paper_bgcolor='#1E1E1E',
        plot_bgcolor='#1E1E1E',
        font=dict(color='#E0E0E0'),
        height=400,
        showlegend=False,
        xaxis=dict(
            showticklabels=False,
            range=[0, 1.5]
        ),
        yaxis={'autorange': 'reversed'}
    )
    
    return fig

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []
    st.session_state.page_faults = 0
    st.session_state.has_run = False
    st.session_state.selected_process = None
    st.session_state.process_sequence = []
    st.session_state.last_refresh = time.time()

# Create a sidebar for inputs
with st.sidebar:
    st.header("Simulation Parameters")
    
    # Algorithm selection
    algorithm = st.radio(
        "Select Page Replacement Algorithm:",
        ["LRU (Least Recently Used)", "Optimal"]
    )
    
    # Number of frames - Allow both slider and manual input
    st.subheader("Memory Frames")
    frame_input_method = st.radio("Frame Input Method:", ["Slider", "Manual Entry"], horizontal=True)
    
    if frame_input_method == "Slider":
        frames = st.slider(
            "Number of Memory Frames:",
            min_value=2,
            max_value=20,
            value=4
        )
    else:
        frames = st.number_input(
            "Enter Number of Memory Frames:",
            min_value=2,
            max_value=50,
            value=4,
            step=1,
            help="Enter a value between 2 and 50"
        )
    
    # Page sequence input
    st.subheader("Page Reference Sequence")
    
    input_method = st.radio(
        "Input Method:",
        ["System Process", "Example Sequence", "Custom Input"]
    )
    
    if input_method == "System Process":
        # Process selection section
        st.subheader("Select Running Process")
        
        # Add refresh button with better styling and responsiveness
        refresh_col1, refresh_col2 = st.columns([3, 1])
        with refresh_col2:
            refresh_button = st.button(
                "🔄 Refresh",
                help="Refresh the list of running processes",
                use_container_width=True,
                key="refresh_processes"
            )
            
            # Add custom HTML for refresh button animation
            st.markdown(
                """
                <style>
                [data-testid="baseButton-secondary"]:has(div:contains("🔄 Refresh")) {
                    background-color: #FF8C00;
                    color: #1E1E1E;
                    font-weight: bold;
                    transition: all 0.3s ease;
                }
                [data-testid="baseButton-secondary"]:has(div:contains("🔄 Refresh")):hover {
                    background-color: #FFA500;
                    transform: scale(1.05);
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            if refresh_button:
                # Force cache invalidation by updating the timestamp
                st.session_state.last_refresh = time.time()
                st.session_state.selected_process = None
                st.rerun()  # Force a rerun to immediately show changes
        
        # Get running processes with the current timestamp to ensure refreshes work
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
            
        # Get running processes
        processes = get_running_processes()
        
        # Add a search box
        search_term = st.text_input("Search Process:", placeholder="Search by name...")
        
        # Filter processes by search term
        if search_term:
            filtered_processes = [p for p in processes if search_term.lower() in p['name'].lower()]
        else:
            filtered_processes = processes
        
        # Process display options
        st.markdown("### Available Processes")
        
        # Let user choose how many processes to display
        if 'max_processes' not in st.session_state:
            st.session_state.max_processes = 10
            
        max_display_options = [10, 20, 50, 100, "All"]
        max_display_col1, max_display_col2 = st.columns([3, 1])
        
        with max_display_col2:
            max_display = st.selectbox(
                "Show:",
                max_display_options,
                index=0,
                help="Number of processes to display"
            )
            if max_display == "All":
                st.session_state.max_processes = len(filtered_processes)
            else:
                st.session_state.max_processes = int(max_display)
        
        # Calculate display count
        display_count = min(st.session_state.max_processes, len(filtered_processes))
        
        # Show process count
        st.caption(f"Showing {display_count} of {len(filtered_processes)} processes")
        
        # Create a container with scrolling for the process list
        st.markdown('<div class="process-list-container">', unsafe_allow_html=True)
        process_container = st.container()
        
        # Display processes
        with process_container:
            for i, proc in enumerate(filtered_processes[:display_count]):
                # Create a clickable process item
                selected_class = " selected-process" if st.session_state.selected_process == proc['pid'] else ""
                
                # Use columns for better layout
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    # Display process info
                    st.markdown(
                        f"""
                        <div class="process-item{selected_class}" id="proc-{proc['pid']}">
                            <div class="process-name">{proc['name']} (PID: {proc['pid']})</div>
                            <div class="process-details">
                                Memory: {proc['memory_mb']:.2f} MB | CPU: {proc['cpu_percent']:.1f}%
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Make it clickable
                with col2:
                    # Centered select button
                    if st.button(f"Select", key=f"btn-{proc['pid']}", use_container_width=True):
                        st.session_state.selected_process = proc['pid']
                        # Generate sequence based on the process
                        st.session_state.process_sequence = generate_process_page_sequence(proc)
                        # Force refresh to show selected process details immediately
                        st.rerun()
        
        # Close the scrollable container
        st.markdown('</div>', unsafe_allow_html=True)
        
        # If a process is selected, show its details and generated sequence
        if st.session_state.selected_process:
            # Find the selected process
            selected_proc = next((p for p in processes if p['pid'] == st.session_state.selected_process), None)
            
            if selected_proc:
                st.markdown("### Selected Process")
                st.markdown(f"""
                <div class="process-item selected-process">
                    <div class="process-name">{selected_proc['name']} (PID: {selected_proc['pid']})</div>
                    <div class="process-details">
                        Memory: {selected_proc['memory_mb']:.2f} MB | CPU: {selected_proc['cpu_percent']:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show generated sequence
                st.markdown("### Generated Page Sequence")
                st.text_area(
                    "Page Sequence:",
                    ", ".join(map(str, st.session_state.process_sequence)),
                    disabled=True,
                    key="process_sequence_display",
                    height=100
                )
                
                # Option to regenerate sequence
                if st.button("Regenerate Sequence", key="regenerate_btn"):
                    st.session_state.process_sequence = generate_process_page_sequence(selected_proc)
                    st.rerun()
                
                # Use this sequence
                page_sequence = st.session_state.process_sequence

    elif input_method == "Example Sequence":
        example = st.selectbox(
            "Select an example:",
            ["Basic Sequence", "With Locality of Reference", "Random Sequence"]
        )
        
        if example == "Basic Sequence":
            page_sequence = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
        elif example == "With Locality of Reference":
            page_sequence = [1, 2, 3, 4, 1, 2, 1, 2, 1, 3, 4, 5, 4, 5, 4, 5]
        else:
            page_sequence = list(np.random.randint(1, 10, size=20))
        
        st.text_area("Page Sequence:", ", ".join(map(str, page_sequence)), disabled=True)
    else:
        sequence_input = st.text_input(
            "Enter comma-separated page numbers:",
            "1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5"
        )
        try:
            page_sequence = [int(x.strip()) for x in sequence_input.split(',') if x.strip()]
        except ValueError:
            st.error("Please enter valid integers separated by commas.")
            page_sequence = []
    
    # Run button
    simulate = st.button("▶️ Run Simulation")

# Main content area
if simulate or st.session_state.has_run:
    if simulate:  # Only recalculate if the button was just pressed
        if algorithm == "LRU (Least Recently Used)":
            st.session_state.history, st.session_state.page_faults = lru_replacement(page_sequence, frames)
        else:  # Optimal
            st.session_state.history, st.session_state.page_faults = optimal_replacement(page_sequence, frames)
        st.session_state.has_run = True
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-container"><p class="metric-label">Algorithm</p><p class="metric-value">' + 
                    algorithm + '</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container"><p class="metric-label">Page Faults</p><p class="metric-value">' + 
                    str(st.session_state.page_faults) + '</p></div>', unsafe_allow_html=True)
    
    with col3:
        fault_rate = (st.session_state.page_faults / len(page_sequence)) * 100
        st.markdown('<div class="metric-container"><p class="metric-label">Fault Rate</p><p class="metric-value">' + 
                    f"{fault_rate:.2f}%" + '</p></div>', unsafe_allow_html=True)
    
    # Memory state visualization
    st.plotly_chart(create_memory_state_fig(st.session_state.history, frames), use_container_width=True)
    
    # Page fault visualization
    st.plotly_chart(create_page_fault_fig(st.session_state.history), use_container_width=True)
    
    # Two-column layout for memory stack and step table
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Memory stack visualization
        st.plotly_chart(display_memory_stack(st.session_state.history, frames), use_container_width=True)
    
    with col2:
        # Create table of steps
        df = pd.DataFrame(st.session_state.history)
        df['state'] = df['state'].apply(lambda x: ', '.join([str(i) if i != -1 else '_' for i in x]))
        df['fault'] = df['fault'].apply(lambda x: '✓' if x else '')
        
        # Rename columns for display
        df = df.rename(columns={
            'step': 'Step',
            'page': 'Page Referenced',
            'state': 'Memory State',
            'fault': 'Page Fault'
        })
        
        # Style the dataframe
        st.dataframe(
            df,
            column_config={
                "Page Fault": st.column_config.Column(
                    "Page Fault",
                    help="Indicates if a page fault occurred",
                    width="small",
                )
            },
            use_container_width=True
        )

# Add explanation section
with st.expander("About Virtual Memory Management"):
    st.markdown("""
    ## How Virtual Memory Works
    
    Virtual memory is a memory management technique that provides an "idealized abstraction of the storage resources that are actually available on a given machine" which "creates the illusion to users of a very large (main) memory."
    
    ### Page Replacement Algorithms
    
    - **LRU (Least Recently Used)**: Replaces the page that hasn't been used for the longest period of time. This works on the principle that pages that have been heavily used in the last few instructions will probably be heavily used again in the next few.
    
    - **Optimal**: Replaces the page that will not be used for the longest period of time in future. This is theoretically optimal but impossible to implement in practice (requires future knowledge), so it's used as a benchmark.
    
    ### Page Faults
    
    A page fault occurs when a program accesses a page that is mapped in the virtual address space, but not loaded in physical memory. The OS must then load the required page from secondary storage.
    
    ### Interpreting the Visualization
    
    - **Memory State**: Shows which pages are in which frames at each step
    - **Page Fault**: Marked with ✓ when a referenced page is not in memory
    - **Memory Stack**: Shows the final state of physical memory frames
    """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; color: #777777;">
    Virtual Memory Simulator | Educational Tool for Understanding Memory Management
</div>
""", unsafe_allow_html=True) 
