from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

class PageReplacementAlgorithm:
    def __init__(self, total_frames: int):
        self.total_frames = total_frames
        self.page_faults = 0
        self.memory_state: List[int] = [-1] * total_frames
        
    def reset(self):
        self.page_faults = 0
        self.memory_state = [-1] * self.total_frames

class LRUAlgorithm(PageReplacementAlgorithm):
    def __init__(self, total_frames: int):
        super().__init__(total_frames)
        self.access_history = OrderedDict()
        
    def simulate(self, page_sequence: List[int]) -> Tuple[int, List[int], List[Dict]]:
        self.reset()
        self.access_history.clear()
        history = []
        
        for page in page_sequence:
            current_state = {
                'page_accessed': page,
                'memory_state': self.memory_state.copy(),
                'page_fault': False
            }
            
            if page not in self.memory_state:
                self.page_faults += 1
                current_state['page_fault'] = True
                
                if -1 in self.memory_state:
                    # There's an empty frame
                    index = self.memory_state.index(-1)
                else:
                    # Find least recently used page
                    lru_page = next(iter(self.access_history))
                    index = self.memory_state.index(lru_page)
                    del self.access_history[lru_page]
                
                self.memory_state[index] = page
            
            # Update access history
            if page in self.access_history:
                del self.access_history[page]
            self.access_history[page] = True
            
            history.append(current_state)
            
        return self.page_faults, self.memory_state, history

class OptimalAlgorithm(PageReplacementAlgorithm):
    def __init__(self, total_frames: int):
        super().__init__(total_frames)
        
    def find_optimal_victim(self, current_pages: List[int], future_sequence: List[int]) -> int:
        future_use = {page: float('inf') for page in current_pages if page != -1}
        
        for page in future_use:
            try:
                future_use[page] = future_sequence.index(page)
            except ValueError:
                pass
                
        return max(future_use.items(), key=lambda x: x[1])[0]
    
    def simulate(self, page_sequence: List[int]) -> Tuple[int, List[int], List[Dict]]:
        self.reset()
        history = []
        
        for i, page in enumerate(page_sequence):
            current_state = {
                'page_accessed': page,
                'memory_state': self.memory_state.copy(),
                'page_fault': False
            }
            
            if page not in self.memory_state:
                self.page_faults += 1
                current_state['page_fault'] = True
                
                if -1 in self.memory_state:
                    # There's an empty frame
                    index = self.memory_state.index(-1)
                else:
                    # Find optimal victim
                    victim = self.find_optimal_victim(
                        self.memory_state,
                        page_sequence[i+1:]
                    )
                    index = self.memory_state.index(victim)
                
                self.memory_state[index] = page
            
            history.append(current_state)
            
        return self.page_faults, self.memory_state, history
