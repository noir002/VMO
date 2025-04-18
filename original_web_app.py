import streamlit as st
import numpy as np
import pandas as pd
from collections import OrderedDict
import random
import psutil
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Virtual Memory Simulator",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --background: #0E1117;
        --secondary-background: #262730;
        --text: #FAFAFA;
        --accent-cyan: #00CCFF;
        --accent-orange: #FF6B00;
        --accent-blue: #FF8C00;
        --accent-green: #00CC96;
        --accent-red: #EF553B;
    }
    
    /* Base styling */
    .main {
        background-color: var(--background);
        color: var(--text);
    }
    
    h1, h2, h3 {
        color: var(--accent-orange);
        font-weight: 600;
    }
    
    /* Custom containers */
    .highlight-container {
        background-color: rgba(255, 107, 0, 0.1);
        border-left: 3px solid var(--accent-orange);
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    .metric-container {
        background-color: var(--secondary-background);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
    }
    
    .metric-label {
        color: var(--accent-orange);
        font-size: 1rem;
        font-weight: bold;
    }
    
    .metric-value {
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 5px;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--accent-orange);
        color: white;
        border: none;
        border-radius: 5px;
        transition: all 0.3s;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #FF8C00;
        transform: scale(1.05);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--secondary-background);
    }
    
    /* Slider and widget colors */
    .stSlider > div > div > div {
        background-color: var(--accent-orange);
    }
    
    .stSlider > div > div > div > div {
        background-color: #FF8C00;
    }
    
    /* Process item styling */
    .process-item {
        background-color: var(--secondary-background);
        border-radius: 5px;
        padding: 12px;
        margin-bottom: 8px;
        transition: all 0.3s;
        border-left: 3px solid transparent;
    }
    
    .process-item:hover {
        background-color: rgba(255, 107, 0, 0.2);
        border-left: 3px solid var(--accent-orange);
        transform: translateX(3px);
    }
    
    .process-name {
        color: var(--accent-orange);
        font-weight: bold;
    }
    
    .process-details {
        color: var(--text);
        opacity: 0.8;
        font-size: 0.9rem;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
    }
    
    /* Dataframe styling */
    .dataframe-container {
        background-color: var(--secondary-background);
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--secondary-background);
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-orange) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Page title
st.title("Virtual Memory Simulator")
st.markdown("### Interactive visualization of page replacement algorithms")

# Function to get running processes
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
    
    # Locality factor based on memory size
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
def create_memory_state_vis(history, frame_count):
    # Extract memory states and create a 2D array
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
                'Page': page if page != -1 else "Free",
                'Page_Int': page,
                'Reference': history[step]['page'],
                'Is_Fault': is_fault
            })
    
    df = pd.DataFrame(data)
    
    # Better color scale based on the algorithm
    if 'Optimal' in st.session_state.get('current_algorithm', 'LRU'):
        color_scale = [(0, "#1E1E1E"), (0.5, "#A05C00"), (1, "#FF8C00")]
    else:
        color_scale = [(0, "#1E1E1E"), (0.5, "#A64B00"), (1, "#FF6B00")]
    
    # Create heatmap using plotly
    fig = px.density_heatmap(
        df, 
        x='Step', 
        y='Frame', 
        z='Page_Int', 
        color_continuous_scale=color_scale,
        labels={'Step': 'Memory Access Step', 'Frame': 'Memory Frame', 'Page_Int': 'Page Number'},
    )
    
    # Add text annotations for page numbers
    for i, row in df.iterrows():
        # Use "□" (empty square) for empty frames instead of "Empty"
        text = "□" if row['Page'] == "Free" else str(int(row['Page']))
        fig.add_annotation(
            x=row['Step'],
            y=row['Frame'],
            text=text,
            showarrow=False,
            font=dict(color="white" if row['Page'] == "Free" else "black", size=14)
        )
    
    # Customize layout
    fig.update_layout(
        title='Memory State Visualization',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='#FAFAFA'),
        height=400,
        coloraxis_showscale=False,
        hoverlabel=dict(
            bgcolor="rgba(255, 107, 0, 0.8)",
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig

# Function to create page fault visualization
def create_page_fault_vis(history):
    df = pd.DataFrame(history)
    df['step'] = df.index
    
    # Create figure
    fig = go.Figure()
    
    # Choose colors based on algorithm - now all using orange variants
    if 'Optimal' in st.session_state.get('current_algorithm', 'LRU'):
        line_color = '#FF8C00'
        fault_color = '#EF553B'
    else:
        line_color = '#FF6B00'
        fault_color = '#EF553B'
    
    # Add page references
    fig.add_trace(go.Scatter(
        x=df['step'],
        y=df['page'],
        mode='lines+markers',
        name='Page Reference',
        line=dict(color=line_color, width=2),
        marker=dict(size=8, color=line_color)
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
            color=fault_color,
            line=dict(width=2, color=fault_color)
        )
    ))
    
    # Customize layout
    fig.update_layout(
        title='Page References and Faults',
        xaxis_title='Access Step',
        yaxis_title='Page Number',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='#FAFAFA'),
        legend=dict(
            x=0,
            y=1.1,
            orientation='h',
            bgcolor='rgba(38, 39, 48, 0.8)',
            font=dict(color='#FAFAFA')
        ),
        height=300,
        hoverlabel=dict(
            bgcolor="rgba(255, 107, 0, 0.8)",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Grid lines
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255, 255, 255, 0.1)'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255, 255, 255, 0.1)'
    )
    
    return fig

# Function to create a memory stack visualization
def create_memory_stack(history, frame_count):
    if not history:
        return None
    
    # Get the final state of memory
    final_state = history[-1]['state']
    
    # Create a figure
    fig = go.Figure()
    
    # Add memory frames
    y_pos = list(range(frame_count))
    
    # Different colors based on algorithm - now all orange variants
    if 'Optimal' in st.session_state.get('current_algorithm', 'LRU'):
        colors = ['#FF8C00' if page != -1 else '#262730' for page in final_state]
    else:
        colors = ['#FF6B00' if page != -1 else '#262730' for page in final_state]
    
    # Create text for each slot
    text_labels = []
    for page in final_state:
        if page != -1:
            text_labels.append(f'Page {page}')
        else:
            text_labels.append('□')  # Use square symbol for empty frames
    
    fig.add_trace(go.Bar(
        x=[1] * frame_count,
        y=y_pos,
        orientation='h',
        marker_color=colors,
        text=text_labels,
        textposition='inside',
        width=0.8
    ))
    
    # Customize layout
    fig.update_layout(
        title='Final Memory Stack',
        xaxis_title='',
        yaxis_title='Frame Number',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='#FAFAFA'),
        height=400,
        showlegend=False,
        xaxis=dict(
            showticklabels=False,
            range=[0, 1.5]
        ),
        yaxis={'autorange': 'reversed'}
    )
    
    return fig

# Initialize session state variables if needed
if 'selected_process' not in st.session_state:
    st.session_state.selected_process = None
    st.session_state.process_sequence = []
    st.session_state.has_run = False
    st.session_state.current_algorithm = "LRU (Least Recently Used)"

# Main app layout
with st.sidebar:
    st.header("Simulation Parameters")
    
    # Algorithm selection
    algorithm = st.radio(
        "Select Algorithm:",
        ["LRU (Least Recently Used)", "Optimal"],
        help="LRU replaces the least recently used page. Optimal uses future knowledge to make the best decision."
    )
    st.session_state.current_algorithm = algorithm
    
    # Number of frames with enhanced slider and manual input option
    st.subheader("Memory Configuration")
    
    frame_input_method = st.radio(
        "Frame Input Method:",
        ["Slider", "Manual Entry"],
        horizontal=True,
        help="Choose how you want to specify the number of frames"
    )
    
    if frame_input_method == "Slider":
        frames = st.slider(
            "Number of Memory Frames:",
            min_value=2,
            max_value=15,
            value=4,
            help="More frames means less page faults, but more memory usage."
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
        
        # Add a note about higher values
        if frames > 15:
            st.caption("⚠️ Large frame counts may affect visualization layout")
    
    # Page sequence input method
    st.subheader("Page Reference Sequence")
    input_method = st.radio(
        "Input Method:",
        ["System Process", "Manual Input"],
        help="Choose to generate a sequence from a running process or enter your own."
    )
    
    # Default page sequence
    page_sequence = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    
    if input_method == "Manual Input":
        sequence_input = st.text_area(
            "Enter comma-separated page numbers:",
            "1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5",
            height=100,
            help="Enter the sequence of page references to simulate."
        )
        # Parse input
        try:
            page_sequence = [int(x.strip()) for x in sequence_input.split(',') if x.strip()]
        except ValueError:
            st.sidebar.error("⚠️ Please enter valid integers separated by commas.")
    
    else:  # System Process
        st.subheader("Select Running Process")
        
        # Add refresh button with better styling
        if st.button("🔄 Refresh Process List", use_container_width=True):
            st.session_state.selected_process = None
        
        # Get running processes
        processes = get_running_processes()
        
        # Add a search box
        search_term = st.text_input("Search Process:", placeholder="Filter by name...")
        
        # Filter processes by search term
        if search_term:
            filtered_processes = [p for p in processes if search_term.lower() in p['name'].lower()]
        else:
            filtered_processes = processes
        
        # Process display options
        st.markdown("### Available Processes")
        
        # Let user choose how many processes to display
        max_display = st.select_slider(
            "Show:",
            options=[10, 20, 50, "All"],
            value=20,
            help="Number of processes to display"
        )
        
        # Calculate display count
        if max_display == "All":
            display_count = len(filtered_processes)
        else:
            display_count = min(int(max_display), len(filtered_processes))
        
        # Show process count
        st.caption(f"Showing {display_count} of {len(filtered_processes)} processes")
        
        # Display processes in the sidebar with better styling
        for i, proc in enumerate(filtered_processes[:display_count]):
            # Create a unique key for each process button
            button_key = f"proc_{proc['pid']}"
            
            # Show process info and selection button
            process_container = st.container()
            with process_container:
                col1, col2 = st.columns([3, 1])
                
                # Process details with custom styling
                with col1:
                    st.markdown(
                        f"""
                        <div class="process-item">
                            <div class="process-name">{proc['name']}</div>
                            <div class="process-details">
                                PID: {proc['pid']} | Memory: {proc['memory_mb']:.1f} MB | CPU: {proc['cpu_percent']:.1f}%
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                # Selection button
                with col2:
                    if st.button("Select", key=button_key):
                        st.session_state.selected_process = proc['pid']
                        st.session_state.process_sequence = generate_process_page_sequence(proc)
                        st.toast(f"Generated sequence from process: {proc['name']}", icon="✅")
        
        # If a process is selected, use its sequence
        if st.session_state.selected_process:
            selected_proc = next((p for p in processes if p['pid'] == st.session_state.selected_process), None)
            if selected_proc:
                st.markdown("### Selected Process")
                
                # Display selected process with better styling
                st.markdown(
                    f"""
                    <div class="process-item" style="border-left: 3px solid #FF6B00;">
                        <div class="process-name">{selected_proc['name']}</div>
                        <div class="process-details">
                            PID: {selected_proc['pid']} | Memory: {selected_proc['memory_mb']:.1f} MB | CPU: {selected_proc['cpu_percent']:.1f}%
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Show generated sequence
                st.text_area(
                    "Generated Page Sequence:",
                    ", ".join(map(str, st.session_state.process_sequence)),
                    disabled=True,
                    height=100
                )
                
                # Use the generated sequence
                page_sequence = st.session_state.process_sequence
    
    # Run button with better styling
    run_col1, run_col2 = st.columns([1, 3])
    with run_col2:
        simulate = st.button("▶️ Run Simulation", type="primary", use_container_width=True)

# Main content
if simulate or st.session_state.has_run:
    if simulate:  # Only recalculate if the button was just pressed
        # Show loading message
        with st.spinner('Running simulation...'):
            if algorithm == "LRU (Least Recently Used)":
                history, page_faults = lru_replacement(page_sequence, frames)
            else:  # Optimal
                history, page_faults = optimal_replacement(page_sequence, frames)
            st.session_state.history = history
            st.session_state.page_faults = page_faults
            st.session_state.has_run = True
            st.session_state.page_sequence = page_sequence
            st.session_state.frames = frames
    else:
        # Use cached results
        history = st.session_state.history
        page_faults = st.session_state.page_faults
        page_sequence = st.session_state.page_sequence
        frames = st.session_state.frames
    
    # Create tabs for different views
    tabs = st.tabs(["Dashboard", "Detailed View", "Data Table"])
    
    with tabs[0]:  # Dashboard
        # Show metrics with enhanced styling
        st.markdown('<div class="highlight-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Algorithm</div>
                    <div class="metric-value">{algorithm}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Page Faults</div>
                    <div class="metric-value">{page_faults}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            fault_rate = (page_faults / len(page_sequence)) * 100
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Fault Rate</div>
                    <div class="metric-value">{fault_rate:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations
        st.plotly_chart(create_memory_state_vis(history, frames), use_container_width=True, config={'displayModeBar': False})
        st.plotly_chart(create_page_fault_vis(history), use_container_width=True, config={'displayModeBar': False})
    
    with tabs[1]:  # Detailed View
        # Two-column layout for memory stack and sequence details
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Memory stack visualization
            st.plotly_chart(create_memory_stack(history, frames), use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            # Sequence information
            st.subheader("Page Sequence Details")
            st.markdown(f"**Total Pages:** {len(page_sequence)}")
            st.markdown(f"**Unique Pages:** {len(set(page_sequence))}")
            
            # Show sequence with better formatting
            st.markdown("**Full Sequence:**")
            sequence_html = ""
            for i, page in enumerate(page_sequence):
                is_fault = history[i]['fault']
                
                if is_fault:
                    sequence_html += f'<span style="background-color: rgba(239, 85, 59, 0.2); padding: 3px 6px; border-radius: 4px; margin: 2px; display: inline-block;">{page}</span>'
                else:
                    sequence_html += f'<span style="background-color: rgba(0, 204, 150, 0.2); padding: 3px 6px; border-radius: 4px; margin: 2px; display: inline-block;">{page}</span>'
            
            st.markdown(f'<div style="line-height: 2.5;">{sequence_html}</div>', unsafe_allow_html=True)
            
            # Legend
            st.markdown("""
            <div style="display: flex; gap: 20px; margin-top: 10px;">
                <div>
                    <span style="background-color: rgba(239, 85, 59, 0.2); padding: 3px 6px; border-radius: 4px;">X</span>
                    <span style="margin-left: 5px;">Page Fault</span>
                </div>
                <div>
                    <span style="background-color: rgba(0, 204, 150, 0.2); padding: 3px 6px; border-radius: 4px;">X</span>
                    <span style="margin-left: 5px;">Page Hit</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tabs[2]:  # Data Table
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        # Show table of steps
        st.subheader("Step-by-Step Simulation Data")
        df = pd.DataFrame(history)
        df['state'] = df['state'].apply(lambda x: ', '.join([str(i) if i != -1 else '_' for i in x]))
        df['fault'] = df['fault'].apply(lambda x: '✓' if x else '')
        
        # Rename columns for display
        df = df.rename(columns={
            'step': 'Step',
            'page': 'Page Referenced',
            'state': 'Memory State',
            'fault': 'Page Fault'
        })
        
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Page Fault": st.column_config.Column(
                    "Page Fault",
                    help="Indicates if a page fault occurred",
                    width="small",
                )
            },
            hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a download button for CSV export
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Simulation Data",
        csv,
        "virtual_memory_simulation.csv",
        "text/csv",
        key='download-csv'
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
    """)

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; color: #777777;">
    Virtual Memory Simulator | Educational Tool for Understanding Memory Management
</div>
""", unsafe_allow_html=True) 