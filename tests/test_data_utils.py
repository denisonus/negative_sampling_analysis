import unittest

import numpy as np
import torch

from utils.data_utils import TrainLoader


class TrainLoaderTests(unittest.TestCase):
    def test_batches_have_expected_length_and_tensors(self):
        interactions = [(0, 10), (1, 11), (2, 12), (3, 13), (4, 14)]
        loader = TrainLoader(interactions, batch_size=2, shuffle=False)

        batches = list(loader)

        self.assertEqual(len(loader), 3)
        self.assertEqual(len(batches), 3)
        self.assertTrue(torch.equal(batches[0][0], torch.tensor([0, 1])))
        self.assertTrue(torch.equal(batches[0][1], torch.tensor([10, 11])))
        self.assertEqual(tuple(batches[-1][0].shape), (1,))
        self.assertEqual(tuple(batches[-1][1].shape), (1,))

    def test_accepts_numpy_interactions(self):
        interactions = np.array([[5, 15], [6, 16], [7, 17]], dtype=np.int64)
        loader = TrainLoader(interactions, batch_size=3, shuffle=False)

        users, items = next(iter(loader))

        self.assertTrue(torch.equal(users, torch.tensor([5, 6, 7])))
        self.assertTrue(torch.equal(items, torch.tensor([15, 16, 17])))


if __name__ == "__main__":
    unittest.main()
