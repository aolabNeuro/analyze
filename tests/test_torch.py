import unittest
from aopy.torch import *

class DatasetTests(unittest.TestCase):

    def test_tensor_dataset(self):
        # create tensors + dataset
        n_batch = 10
        tensor_1 = torch.randn(n_batch,10,5)
        tensor_2 = torch.randn(n_batch,1,15)

        device = 'cpu'

        tensor_dataset = TensorDataset(
            tensor_1, tensor_2,
            device=device
        )

        # test len
        tensor_dataset_len = len(tensor_dataset)
        self.assertTrue(tensor_dataset_len == n_batch)

        # test sampling
        sample_idx = 0
        sample = next(iter(tensor_dataset))
        self.assertTrue(
            (sample[0] == tensor_1[sample_idx,:,:]).all().item() and (sample[1] == tensor_2[sample_idx,:,:]).all().item()
        )

if __name__ == "__main__":
    unittest.main()


