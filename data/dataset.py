from torch.utils.data import Dataset
from .preprocess import *

class LipreadingDataset(Dataset):
    """BBC Lip Reading dataset."""


    def build_file_list(dir):
        labels = os.listdir(dir)
        print(labels)


    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """

        self.file_list = build_file_list("/udisk/pszts-ssd/AV-ASR-data/BBC_Oxford/lipread_mp4")

        #self.landmarks_frame = pd.read_csv(csv_file)
        #self.root_dir = root_dir

    def __len__(self):
        return 25000

    def __getitem__(self, idx):
        #load video into a tensor
        filename = 'AFTERNOON.mp4'
        vidframes = load_video(filename)
        temporalvolume = bbc(vidframes)

        sample = {'temporalvolume': temporalvolume, 'label': torch.LongTensor([42])}

        return sample
