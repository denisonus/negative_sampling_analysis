import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from utils.data_utils import TrainLoader, load_dataset


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


class DatasetLoaderTests(unittest.TestCase):
    def test_ml100k_loader_reads_official_files_and_temporal_split(self):
        with TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "ml-100k"
            dataset_dir.mkdir()
            (dataset_dir / "u.data").write_text(
                "\n".join(
                    [
                        "1\t10\t5\t100",
                        "1\t11\t4\t200",
                        "1\t12\t3\t300",
                        "1\t13\t5\t400",
                        "2\t10\t4\t100",
                        "2\t14\t5\t200",
                        "3\t16\t5\t150",
                    ]
                )
                + "\n",
                encoding="latin-1",
            )

            dataset, train, valid, test = load_dataset(
                "ml-100k",
                data_path=tmpdir,
                min_rating=4,
            )

        self.assertEqual(dataset.num(dataset.uid_field), 3)
        self.assertEqual(dataset.num(dataset.iid_field), 5)
        self.assertEqual(train, [(0, 0), (1, 0), (2, 4)])
        self.assertEqual(valid.uid_list.tolist(), [0])
        self.assertEqual(valid.uid2positive_item[0].tolist(), [1])
        self.assertEqual(valid.uid2history_item[0].tolist(), [0])
        self.assertEqual(test.uid_list.tolist(), [0, 1])
        self.assertEqual(test.uid2positive_item[0].tolist(), [2])
        self.assertEqual(test.uid2history_item[0].tolist(), [0, 1])
        self.assertEqual(test.uid2positive_item[1].tolist(), [3])
        self.assertEqual(test.uid2history_item[1].tolist(), [0])

    def test_ml1m_loader_reads_double_colon_format(self):
        with TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "ml-1m"
            dataset_dir.mkdir()
            (dataset_dir / "ratings.dat").write_text(
                "\n".join(
                    [
                        "10::100::5::100",
                        "10::101::4::200",
                        "10::102::5::300",
                        "11::100::2::100",
                        "11::103::4::200",
                    ]
                )
                + "\n",
                encoding="latin-1",
            )

            dataset, train, valid, test = load_dataset(
                "ml-1m",
                data_path=tmpdir,
                min_rating=4,
            )

        self.assertEqual(dataset.num(dataset.uid_field), 2)
        self.assertEqual(dataset.num(dataset.iid_field), 4)
        self.assertEqual(train, [(0, 0), (1, 3)])
        self.assertEqual(valid.uid_list.tolist(), [0])
        self.assertEqual(test.uid_list.tolist(), [0])

    def test_gowalla_loader_reads_raw_lightgcn_files(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "gowalla-1m" / "raw-lightgcn"
            raw_dir.mkdir(parents=True)
            (raw_dir / "train.txt").write_text(
                "0 1 2 2 3\n1 5\n2 8 9\n",
                encoding="utf-8",
            )
            (raw_dir / "test.txt").write_text(
                "0 3 4 5\n1 6\n2 10\n",
                encoding="utf-8",
            )

            dataset, train, valid, test = load_dataset(
                "gowalla-1m",
                data_path=tmpdir,
                min_rating=None,
                seed=42,
            )

        self.assertEqual(dataset.num(dataset.uid_field), 3)
        self.assertEqual(dataset.num(dataset.iid_field), 9)
        self.assertEqual(set(train) & set(valid.uid2positive_item[0].tolist()), set())
        self.assertEqual(valid.uid_list.tolist(), [0, 2])
        self.assertEqual(test.uid_list.tolist(), [0, 2])


if __name__ == "__main__":
    unittest.main()
