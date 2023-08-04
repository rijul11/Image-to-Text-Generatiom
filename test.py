from data import get_loader
from config import config
import torchvision.transforms as transforms
from vocabulary import serialize 

mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]

transform = transforms.Compose(
    [transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)]
)

train_loader, dataset = get_loader(
    root_folder = config['images_path'],
    annotation_file = config['caption_file_path'],
    transform = transform, 
    batch_size = 1,
    num_workers = 4,
    test = False
)

serialize()

# print([dataset])