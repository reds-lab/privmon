#! python3
import logging
import sys
import torchvision
import torch
from torchvision import transforms

# Kafka consumer config
kafka_consumer_conf = {'bootstrap.servers': "localhost:9092",
        'group.id': "queries",
        'auto.offset.reset': 'smallest',
        'max.poll.interval.ms': 600000
        }


# Default logging setting
# __name__ should be passed
def get_logger(name, fname='default', level=logging.DEBUG, is_save=True):
    fmt='%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    datefmt='%Y-%m-%d %H:%M:%S'
    if name == "__main__":
        name = "main"
    else:
        name = "main" + "." + name
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if name != "main":
        return logger
    s_handler = logging.StreamHandler(sys.stdout)
    s_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    s_handler.setLevel(level)
    logger.addHandler(s_handler)
    if is_save:
        filename = fname +'.log'
        f_handler = logging.FileHandler(filename=filename, mode='w')
        f_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        f_handler.setLevel(level)
        logger.addHandler(f_handler)
    return logger


# load samples from datasets
def load_samples(dataset="CIFAR10", train=True):
    # The image sizes of GTSRB range from 15x15 to 280x280
    # The image size of Face is 150x150
    sizeMap = {
            "CIFAR10": 32,
            "CIFAR100": 32,
            "Face": 250,
            "Face64": 64,
            "GTSRB": 64,
            "FS": 64,
            "CelebA": 64,
            "MNIST": 28
            }
    image_size = sizeMap[dataset]
    transform_list = []
    if dataset in ["GTSRB", "Face64","FS", "CelebA"]:
        transform_list.append(transforms.Resize((image_size, image_size)))
    transform_list.append(transforms.ToTensor())
    # transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform_list)
    # The default image size of CIFA10 is (3, 32, 32)
    if dataset=="CIFAR10":
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train,
                download=True, transform=transform)
    elif dataset=="CIFAR100":
        dataset = torchvision.datasets.CIFAR100(root='./data', train=train,
                download=True, transform=transform)
    elif dataset=="Face64" or dataset=="Face":
        dataset = torchvision.datasets.ImageFolder(root="./data/lfw", transform=transform)
    elif dataset=="GTSRB":
        dataset = torchvision.datasets.ImageFolder(root="./data/GTSRB/Train", transform=transform)
    elif dataset=="FS":
        dataset = torchvision.datasets.ImageFolder(root="./data/facescrub", transform=transform)
    elif dataset=="CelebA":
        dataset = torchvision.datasets.ImageFolder(root="./data/celeba", transform=transform)
    else:
        raise ValueError(f"No such dataset -> {dataset}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1,
            shuffle=True, num_workers=1)
    return iter(loader)


if __name__ == "__main__":
    # load_samples('CIFAR10', True)
    # load_samples('GTSRB', True)
    image_iter = load_samples('CelebA', True)
    image, label = image_iter.next()
    print(image.shape, label)
