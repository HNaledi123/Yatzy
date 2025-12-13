# test_overflow.py
import numpy as np
import unittest

# This value is the maximum for a 64-bit unsigned integer.
# The overflow happens when we try to add to a number larger than this.
UINT64_MAX = np.iinfo(np.uint64).max

def mock_worker_result(batch_size=100_000):
    """
    Creates a fake result tuple, just like _worker_sim_batch would.
    The important part is score_bins, which determines the return type of sum().
    """
    score_bins = np.zeros(10, dtype=np.uint32)
    score_bins[0] = batch_size 
    # The sum of score_bins will be a numpy.uint64 scalar with the value of batch_size
    return (None, score_bins, None, None, None, None, None, None, None)

class TestIntegerOverflow(unittest.TestCase):

    def test_original_code_fails(self):
        """
        This test demonstrates the overflow with the original code.
        We expect a FloatingPointError because np.seterr(all='raise') is active.
        """
        # Start with a number just below the overflow threshold
        sims_completed = UINT64_MAX - 50_000
        
        # Get the mock result from a "worker"
        _, score_bins, _, _, _, _, _, _, _ = mock_worker_result(100_000)
        
        # This is the problematic operation.
        # We are adding a numpy.uint64 to a Python int that is larger
        # than what a numpy.uint64 can hold.
        sum_of_batch = sum(score_bins)

        # In a unittest context, this operation raises a standard Python OverflowError
        # before NumPy's FloatingPointError is triggered. We check for either to be robust.
        with self.assertRaises((OverflowError, FloatingPointError)):
            # This line should fail
            sims_completed += sum_of_batch

        print("Successfully confirmed that the original code fails as expected.")

    def test_fixed_code_succeeds(self):
        """
        This test shows that casting the sum to a Python int prevents the overflow.
        """
        # Start with the same large number
        sims_completed = UINT64_MAX - 50_000
        
        # Get the mock result
        _, score_bins, _, _, _, _, _, _, _ = mock_worker_result(100_000)
        
        # This is the corrected operation.
        # By casting to int(), we ensure the addition uses Python's
        # arbitrary-precision integers.
        sum_of_batch = int(sum(score_bins))
        
        try:
            # This line should now succeed
            sims_completed += sum_of_batch
            
            # The final value should be correct
            expected_result = UINT64_MAX + 50_000
            self.assertEqual(sims_completed, expected_result)
            
            print("Successfully confirmed that the fixed code works correctly.")
        except FloatingPointError:
            self.fail("The fixed code unexpectedly raised a FloatingPointError.")

if __name__ == '__main__':
    # Set numpy to raise on errors, just like in YatzySuite.py
    np.seterr(all='raise')
    unittest.main()