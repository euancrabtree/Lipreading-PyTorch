import imageio
import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
import torch

def load_video(filename):
    """Loads the specified video using ffmpeg.

    Args:
        filename (str): The path to the file to load.
            Should be a format that ffmpeg can handle.

    Returns:
        List[FloatTensor]: the frames of the video as a list of 3D tensors
            (channels, width, height)"""

    #Download ffmpeg using imageio, just in case it isn't installed
    imageio.plugins.ffmpeg.download()

    vid = imageio.get_reader(filename,  'ffmpeg')
    frames = []
    for i in range(0, 29):
        image = vid.get_data(i)
        image = functional.to_tensor(image)
        frames.append(image)
    return frames

def bbc(vidframes):
    """Preprocesses the specified list of frames by center cropping.
    This will only work correctly on videos that are already centered on the
    mouth region, such as LRITW.

    Args:
        vidframes (List[FloatTensor]):  The frames of the video as a list of
            3D tensors (channels, width, height)

    Returns:
        FloatTensor: The video as a temporal volume, represented as a 5D tensor
            (batch, channel, time, width, height)"""

    temporalvolume = torch.FloatTensor(1,29,112,112)

    for i in range(0, 29):
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((112, 112)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.4161,],[0.1688,]),
        ])(vidframes[i])

        temporalvolume[0][i] = result

    return temporalvolume
