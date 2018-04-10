from torch.utils.data import Dataset
from .preprocess import *
import os

class LipreadingDataset(Dataset):
    """BBC Lip Reading dataset."""

    def build_file_list(self, dir, set):
        labels = os.listdir(dir)

        completeList = []

        for i, label in enumerate(labels):

            dirpath = dir + "/{}/{}".format(label, set)
            print(i, label, dirpath)

            files = os.listdir(dirpath)

            for file in files:
                if file.endswith("mp4"):
                    filepath = dirpath + "/{}".format(file)
                    entry = (i, filepath)
                    completeList.append(entry)


        return labels, completeList


    def __init__(self, directory, set):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """

        self.label_list, self.file_list = self.build_file_list(directory, set)

        #self.landmarks_frame = pd.read_csv(csv_file)
        #self.root_dir = root_dir

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        #load video into a tensor
        label, filename = self.file_list[idx]
        vidframes = load_video(filename)
        temporalvolume = bbc(vidframes)

        sample = {'temporalvolume': temporalvolume, 'label': torch.LongTensor([label])}

        return sample
