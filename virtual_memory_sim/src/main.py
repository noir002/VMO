import os
import sys

# Add the src directory to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(src_dir))
sys.path.insert(0, project_root)

import streamlit as st
from src.algorithms.page_replacement import LRUAlgorithm, OptimalAlgorithm
from src.visualization.memory_viz import create_memory_visualization, create_page_fault_graph

def parse_page_sequence(input_str: str) -> list:
    """Parse the input string into a list of page numbers."""
    try:
        return [int(x.strip()) for x in input_str.split(',')]
    except ValueError:
        st.error('Please enter valid numbers separated by commas')
        return []

def main():
    st.set_page_config(
        page_title="Virtual Memory Simulator",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title('Virtual Memory Management Visualization')
    st.sidebar.header('Memory Configuration')
    
    # Memory configuration
    total_frames = st.sidebar.slider('Total Memory Frames', 4, 20, 8)
    page_size = st.sidebar.slider('Page Size (KB)', 1, 16, 4)
    
    # Page replacement configuration
    st.sidebar.header('Page Replacement')
    page_sequence = st.sidebar.text_input(
        'Enter page sequence (comma-separated)',
        '1,2,3,4,1,2,5,1,2,3,4,5'
    ).strip()
    
    algorithm_name = st.sidebar.selectbox(
        'Select Page Replacement Algorithm',
        ['LRU', 'Optimal']
    )
    
    # Initialize algorithm
    algorithm = LRUAlgorithm(total_frames) if algorithm_name == 'LRU' else OptimalAlgorithm(total_frames)
    
    if st.sidebar.button('Run Simulation'):
        sequence = parse_page_sequence(page_sequence)
        if sequence:
            with st.spinner('Running simulation...'):
                page_faults, final_memory, history = algorithm.simulate(sequence)
                
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric('Page Faults', page_faults)
                    st.plotly_chart(create_memory_visualization(final_memory, page_size))
                
                with col2:
                    st.metric('Page Fault Rate', f"{(page_faults/len(sequence))*100:.2f}%")
                    st.plotly_chart(create_page_fault_graph(history))
                
                # Display page access sequence
                st.subheader('Page Access Sequence')
                st.write(' → '.join(map(str, sequence)))
                
                # Display simulation history
                st.subheader('Simulation History')
                for i, state in enumerate(history):
                    st.text(
                        f"Step {i+1}: Accessing Page {state['page_accessed']} "
                        f"{'(Page Fault)' if state['page_fault'] else '(Hit)'}"
                    )

if __name__ == '__main__':
    main()
