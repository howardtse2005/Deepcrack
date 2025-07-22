import preprocess_pipeline as pp
from torch.utils.data import Dataset
from config import Config as cfg



class loadedDataset(Dataset):
    """
    Create a torch Dataset from data
    """

    def __init__(self, dataset, preprocess=None):
        super(loadedDataset, self).__init__()
        self.samples = []
        if preprocess is None:
            preprocess = lambda x: x
        
        # Flatten all crops from all images
        print("Preprocessing dataset...")
        for item in dataset:
            crops = preprocess(item)
            if isinstance(crops, list):
                self.samples.extend(crops)
            else:
                self.samples.append(crops)
        print(f"Total samples after processing: {len(self.samples)}")

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

