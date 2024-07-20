import os
import cv2
import numpy as np
import pandas as pd
import random
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
import ssl
import certifi

ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = ssl._create_unverified_context

# Specify paths for data directories
DATA_DIR = r'/Users/harshkumarsaha/Downloads/Linear_Feature_Extraction'
TEST_DIR = r'/Users/harshkumarsaha/Downloads/Linear_Feature_Extraction/test'
VALID_DIR = r'/Users/harshkumarsaha/Downloads/Linear_Feature_Extraction/validation'

# Load metadata and preprocess data
metadata_df = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))
metadata_df = metadata_df[metadata_df['split']=='train']
metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
metadata_df['mask_path'] = metadata_df['mask_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))

if metadata_df is not None:
    print("Metadata loaded and preprocessed successfully!")
    print("Metadata shape:", metadata_df.shape)
    print("Metadata columns:", metadata_df.columns)
    print("First 5 rows of metadata:")
    print(metadata_df.head())

    # Check if file paths are correct
    for index, row in metadata_df.iterrows():
        sat_image_path = row['sat_image_path']
        mask_path = row['mask_path']
        if os.path.exists(sat_image_path) and os.path.exists(mask_path):
            print(f"File paths for image {row['image_id']} are correct!")
        else:
            print(f"Error: File paths for image {row['image_id']} are incorrect!")

else:
    print("Error loading metadata!")

# Perform 90/10 split for train / val
valid_df = metadata_df.sample(frac=0.1, random_state=42)
train_df = metadata_df.drop(valid_df.index)

print("Original metadata shape:", metadata_df.shape)
print("Splitting into train and validation sets...")

print("Train set shape:", train_df.shape)
print("Validation set shape:", valid_df.shape)

print("Length of train set:", len(train_df))
print("Length of validation set:", len(valid_df))

# Define class names and RGB values
class_dict = pd.read_csv(os.path.join(DATA_DIR, 'class_dict.csv'))
class_names = class_dict['name'].tolist()
class_rgb_values = class_dict[['r','g','b']].values.tolist()

print("Class names:")
print(class_names)

print("\nClass RGB values:")
for i, rgb in enumerate(class_rgb_values):
    print(f"Class {class_names[i]}: RGB({rgb[0]}, {rgb[1]}, {rgb[2]})")

# Useful to shortlist specific classes in datasets with large number of classes
select_classes = ['background', 'road']

# Get RGB values of required classes
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

# Define helper functions
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]);
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map

def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    """
    x = np.argmax(image, axis=-1)
    return x

def colour_code_segmentation(image, label_values):
    """
     Given a 1-channel array of class keys, colour code the segmentation results.
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x

# Define dataset class
class RoadsDataset(torch.utils.data.Dataset):
    def __init__(self, df, class_rgb_values=None, augmentation=None, preprocessing=None, data_dir=None):
        self.image_paths = df['sat_image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.data_dir = data_dir

    def __getitem__(self, i):
        image_path = os.path.join(self.data_dir, self.image_paths[i])
        mask_path = os.path.join(self.data_dir, self.mask_paths[i])
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask

    def __len__(self):
        return len(self.image_paths)

# Define preprocessing and augmentation functions
def get_training_augmentation():
    train_transform = [
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
    ]
    return album.Compose(train_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(_transform)

# Create dataset instances
train_dataset = RoadsDataset(
    train_df,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(),
    class_rgb_values=select_class_rgb_values,
    data_dir=DATA_DIR
)

valid_dataset = RoadsDataset(
    valid_df,
    preprocessing=get_preprocessing(),
    class_rgb_values=select_class_rgb_values,
    data_dir=VALID_DIR
)

test_dataset = RoadsDataset(
    valid_df,
    preprocessing=get_preprocessing(),
    class_rgb_values=select_class_rgb_values,
    data_dir=TEST_DIR
)

# Get train and val data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)

# Define model parameters
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = select_classes
ACTIVATION = 'sigmoid'

# Create segmentation model
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# Train the model
if __name__ == '__main__':
    # Set flag to train the model or not
    TRAINING = True
    EPOCHS = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5)]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.00008), ])

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    if TRAINING:
        best_iou_score = 0.0
        train_logs_list, valid_logs_list = [], []

        for i in range(0, EPOCHS):
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)
            if best_iou_score < valid_logs['iou_score']:
                best_iou_score = valid_logs['iou_score']
                torch.save(model, './best_model.pth')
                print('Model saved!')

        if not TRAINING:
            model = torch.load('./best_model.pth')

        # Evaluate the model on the test set
        test_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
            verbose=True,
        )

        test_logs = test_epoch.run(test_loader)
        print('Test IoU score: {:.4f}'.format(test_logs['iou_score']))

        # Visualize some predictions
        for i in range(5):
            image, gt_mask = test_dataset[i]
            image = image.to(DEVICE)
            gt_mask = gt_mask.to(DEVICE)
            with torch.no_grad():
                pr_mask = model(image.unsqueeze(0))
            pr_mask = pr_mask.squeeze(0).cpu().numpy()
            gt_mask = gt_mask.cpu().numpy()
            visualize(
                image=image,
                ground_truth_mask=gt_mask,
                predicted_mask=pr_mask
            )
