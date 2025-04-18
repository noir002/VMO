import unittest
from src.algorithms.page_replacement import LRUAlgorithm, OptimalAlgorithm

class TestPageReplacement(unittest.TestCase):
    def setUp(self):
        self.total_frames = 3
        self.lru = LRUAlgorithm(self.total_frames)
        self.optimal = OptimalAlgorithm(self.total_frames)
        
    def test_lru_algorithm(self):
        sequence = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
        page_faults, final_memory, history = self.lru.simulate(sequence)
        
        self.assertGreater(page_faults, 0)
        self.assertEqual(len(final_memory), self.total_frames)
        self.assertEqual(len(history), len(sequence))
        
    def test_optimal_algorithm(self):
        sequence = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
        page_faults, final_memory, history = self.optimal.simulate(sequence)
        
        self.assertGreater(page_faults, 0)
        self.assertEqual(len(final_memory), self.total_frames)
        self.assertEqual(len(history), len(sequence))
        
    def test_empty_frames(self):
        sequence = [1, 2]
        
        # Test LRU
        page_faults, final_memory, _ = self.lru.simulate(sequence)
        self.assertEqual(page_faults, 2)
        self.assertEqual(final_memory.count(-1), 1)
        
        # Test Optimal
        page_faults, final_memory, _ = self.optimal.simulate(sequence)
        self.assertEqual(page_faults, 2)
        self.assertEqual(final_memory.count(-1), 1)

if __name__ == '__main__':
    unittest.main()
