import plotly.graph_objects as go
from typing import List, Dict
import numpy as np

def create_memory_visualization(memory_state: List[int], page_size: int) -> go.Figure:
    """Create a visualization of the current memory state."""
    fig = go.Figure()
    
    y_positions = np.arange(len(memory_state))
    colors = ['#2ecc71' if x != -1 else '#e74c3c' for x in memory_state]
    
    fig.add_trace(go.Bar(
        x=[1] * len(memory_state),
        y=y_positions,
        orientation='h',
        marker_color=colors,
        text=[f'Page {x}' if x != -1 else 'Empty' for x in memory_state],
        textposition='inside',
        width=0.8
    ))
    
    fig.update_layout(
        title='Memory State Visualization',
        xaxis_title='Memory Pages',
        yaxis_title='Frame Number',
        showlegend=False,
        height=400,
        width=600,
        yaxis={'autorange': 'reversed'}
    )
    
    return fig

def create_page_fault_graph(history: List[Dict]) -> go.Figure:
    """Create a visualization of page faults over time."""
    fig = go.Figure()
    
    x = list(range(len(history)))
    y = [1 if state['page_fault'] else 0 for state in history]
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        name='Page Faults',
        marker=dict(
            color=['red' if fault else 'green' for fault in y],
            size=10
        )
    ))
    
    fig.update_layout(
        title='Page Faults Over Time',
        xaxis_title='Access Sequence',
        yaxis_title='Page Fault',
        showlegend=True,
        height=300,
        width=600,
        yaxis=dict(
            tickmode='array',
            ticktext=['No Fault', 'Fault'],
            tickvals=[0, 1]
        )
    )
    
    return fig
