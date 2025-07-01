import torch
from torch.utils.data import Dataset
from utils import *


class ASLDataset(Dataset):
    def __init__(self, data_dir, extractor, transform=None,  limit_label=None, limit_count=None):
        self.data_dir = data_dir
        # if indices is not None:
        #     self.data = [self.data[i] for i in indices]
        self.transform = transform
        self.extractor = extractor

        self.image_paths=[]
        self.labels=[]
        self.class_to_idx = {}

        #get list of all classes (A-Z)
        classes = sorted(os.listdir(data_dir))

        for idx, class_name in enumerate(classes):
            if limit_label is not None and class_name!=limit_label:
                continue

            class_path = os.path.join(data_dir, class_name)  #extract each letter's dir
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = idx  #assign an index to each class

            count=0
            for fname in os.listdir(class_path):
                if limit_count is not None and count>=limit_count:
                    break
                self.image_paths.append(os.path.join(class_path, fname))  #extract each image path from class dir
                self.labels.append(idx)
                count+=1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)

        hand_landmarks = self.extractor.feature_extraction(img_path) # hand_landmarks = 21 coord points

        if hand_landmarks is None:
            return None
        hand_landmarks_tensor = torch.tensor(hand_landmarks, dtype=torch.float32).view(-1) #flatten to view as a single (63,) tensor
        return hand_landmarks_tensor, label

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    inputs,labels = zip(*batch)
    return torch.stack(inputs), torch.stack(labels)  #returns (batch_size, 63) , (labels_list)


