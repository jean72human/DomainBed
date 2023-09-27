import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# 1. Load a pretrained model (e.g., ResNet-18)
model_A = models.resnet18(pretrained=True)
model_A.eval()  # Set the model to evaluation mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_A.to(device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_dataset = datasets.ImageFolder("./data/spawrious224", transform=transform)
dataloader = DataLoader(image_dataset, batch_size=32, shuffle=False)

# 2. Perform classification over the dataset & 3. Store results
predictions = []
true_labels = []
with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_A(inputs)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Convert to torch tensors for easier comparison
predictions = torch.tensor(predictions)
true_labels = torch.tensor(true_labels)

# Create a binary tensor (1 for wrong predictions, 0 for correct ones)
wrong_predictions = (predictions != true_labels).int()


# 4. Create new PyTorch dataset object
class CustomDataset(Dataset):
    def __init__(self, original_dataset, wrong_predictions):
        self.original_dataset = original_dataset
        self.wrong_predictions = wrong_predictions

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, y = self.original_dataset[idx]
        w = self.wrong_predictions[idx]
        return x, y, w


triplet_dataset = CustomDataset(image_dataset, wrong_predictions)
triplet_loader = DataLoader(triplet_dataset, batch_size=32, shuffle=False)

# To iterate over the new dataset:
for x, y, w in triplet_loader:
    pass  # Replace with whatever you want to do with x, y, w