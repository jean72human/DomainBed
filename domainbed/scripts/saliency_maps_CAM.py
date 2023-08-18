
# %%
import sys
sys.path.append("/home/aengusl/Desktop/Projects/OOD_workshop/DomainBed-SP")
# Define your model
from torchvision.models import resnet18
model = resnet18(pretrained=True).eval()

# Set your CAM extractor
from torchcam.methods import SmoothGradCAMpp
cam_extractor = SmoothGradCAMpp(model)
import torchvision.transforms as transforms
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18, resnet50
from torchcam.methods import SmoothGradCAMpp
from PIL import Image
import torch
from domainbed import algorithms
from domainbed import hparams_registry

import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask


# model = resnet18(pretrained=True).eval()
pretrained_model_path = "/home/aengusl/Desktop/Projects/OOD_workshop/DomainBed-SP/erm_output/resnet50_SpawriousM2M_hard_ERM_model.pkl"

class_list = ["bulldog", "corgi", "dachshund", "labrador"]

# model_dict = torch.load(pretrained_model_path)["model_dict"]
# hparams = torch.load(pretrained_model_path)["model_hparams"]
# erm_model = algorithms.ERM(
#        input_shape=torch.Size([3, 224, 224]),  # input shape
#         num_classes=4,  # number of classes
#         hparams=hparams,  # hparams
#         num_domains=1,  # number of training domains
# )

# erm_model.load_state_dict(model_dict)
# # erm_model.network.eval()
# for param in erm_model.parameters():
#     param.requires_grad = True
# # Get your input
# img = Image.open("/home/aengusl/Desktop/Projects/OOD_workshop/DomainBed-SP/data/spawrious224/0/dirt/bulldog/dirt_bulldog_0.png")
# img = img.convert("RGB")
# # Preprocess it for your chosen model
# # input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# # Define the preprocessing steps
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Preprocess the input image
# input_tensor = preprocess(img)

# # with SmoothGradCAMpp(model) as cam_extractor:
# cam_extractor = SmoothGradCAMpp(erm_model.network)
#   # Preprocess your data and feed it to the model
# out = erm_model.network(input_tensor.unsqueeze(0))
# # Retrieve the CAM by passing the class index and the model output
# activation_map = cam_extractor(class_list.index('bulldog'), out)


# import matplotlib.pyplot as plt
# # Visualize the raw CAM
# plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()




# # Resize the CAM and overlay it
# result = overlay_mask(img, to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
# # Display it
# plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()





# ------------------------------------------




# Get misclassifications, then for the images that got misclassified, get the CAMs for predicted class
import os
import tqdm


data_dir = '/home/aengusl/Desktop/Projects/OOD_workshop/DomainBed-SP/data/spawrious224/0/snow/bulldog'
# data_dir = '/home/aengusl/Desktop/Projects/OOD_workshop/DomainBed-SP/data/spawrious224/0/dirt/bulldog'
image_paths = []
for image_file in os.listdir(data_dir):
    image_paths.append(os.path.join(data_dir, image_file))

input_shape = (3, 224, 224)
test_transforms = transforms.Compose([
    transforms.Resize((input_shape[1], input_shape[2])),
    transforms.transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
label = torch.tensor(class_list.index('bulldog'), dtype=torch.long)
misclassifications = [0] * len(image_paths)

model_dict = torch.load(pretrained_model_path)["model_dict"]
hparams = torch.load(pretrained_model_path)["model_hparams"]

erm_model = algorithms.ERM(
    input_shape=torch.Size([3, 224, 224]),  # input shape
        num_classes=4,  # number of classes
        hparams=hparams,  # hparams
        num_domains=1,  # number of training domains
)
erm_model.load_state_dict(model_dict)
for param in erm_model.parameters():
    param.requires_grad = True
erm_model.to('cuda')

misclassification_count = 0
correct_count = 0

save_dir = "/home/aengusl/Desktop/Projects/OOD_workshop/DomainBed-SP/data/misclassified_m2m_hard_labrador_18_aug"
os.makedirs(save_dir, exist_ok=True)


print("\t\tComputing misclassifications...")
for idx, image_path in enumerate(image_paths):
    pil_image = Image.open(image_path).convert('RGB')
    image = test_transforms(pil_image).to('cuda').unsqueeze(0)
    output = erm_model.network(image)
    _, predicted = output.max(1)

    if predicted != label:
        misclassifications[idx] = 1
        
        if predicted == class_list.index('labrador'):
            misclassification_count += 1
            print("\n\nLabrador misclassification found!")
            # Retrieve the CAM by passing the class index and the model output
            erm_model = algorithms.ERM(
                input_shape=torch.Size([3, 224, 224]),  # input shape
                    num_classes=4,  # number of classes
                    hparams=hparams,  # hparams
                    num_domains=1,  # number of training domains
            )
            erm_model.load_state_dict(model_dict)
            for param in erm_model.parameters():
                param.requires_grad = True
            erm_model.to('cuda')
            cam_extractor = SmoothGradCAMpp(erm_model.network)
            output = erm_model.network(image)
            activation_map = cam_extractor(class_list.index('labrador'), output)
            result = overlay_mask(pil_image, to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
            plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
            # save image
            pil_image.save(os.path.join(save_dir,f"original_labrador_as_bulldog{misclassification_count}.png"))
            result.save(os.path.join(save_dir,f"saliency_bulldog_as_labrador{misclassification_count}.png"))


# ------------------------------------------
# %%

# For o2o hard, need to loop over both 0 and 1 for bulldogs in mountains, which got mistakenly classified as dachshunds

dachshund_dir1 = '/home/aengusl/Desktop/Projects/OOD_workshop/DomainBed-SP/data/spawrious224/0/mountain/bulldog'
dachshund_dir2 = '/home/aengusl/Desktop/Projects/OOD_workshop/DomainBed-SP/data/spawrious224/1/mountain/bulldog'

o2o_hard_model_path = '/home/aengusl/Desktop/Projects/OOD_workshop/DomainBed-SP/erm_output/resnet50_SpawriousO2O_hard_ERM_model.pkl'


dachshund_image_paths = []
for image_file in os.listdir(dachshund_dir1):
    dachshund_image_paths.append(os.path.join(dachshund_dir1, image_file))
for image_file in os.listdir(dachshund_dir2):
    dachshund_image_paths.append(os.path.join(dachshund_dir2, image_file))

input_shape = (3, 224, 224)
test_transforms = transforms.Compose([
    transforms.Resize((input_shape[1], input_shape[2])),
    transforms.transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
label = torch.tensor(class_list.index('bulldog'), dtype=torch.long)
misclassifications = [0] * len(image_paths)

model_dict = torch.load(o2o_hard_model_path)["model_dict"]
hparams = torch.load(o2o_hard_model_path)["model_hparams"]

erm_model = algorithms.ERM(
    input_shape=torch.Size([3, 224, 224]),  # input shape
        num_classes=4,  # number of classes
        hparams=hparams,  # hparams
        num_domains=2,  # number of training domains
)

erm_model.load_state_dict(model_dict)

for param in erm_model.parameters():
    param.requires_grad = True
erm_model.to('cuda')

misclassification_count = 0
correct_count = 0

dachshund_save_dir = '/home/aengusl/Desktop/Projects/OOD_workshop/DomainBed-SP/data/misclassified_o2o_hard_dachshund_18_aug'
os.makedirs(dachshund_save_dir, exist_ok=True)

for idx, image_path in enumerate(dachshund_image_paths):
    pil_image = Image.open(image_path).convert()

    image = test_transforms(pil_image).to('cuda').unsqueeze(0)
    output = erm_model.network(image)
    _, predicted = output.max(1)

    if predicted != label:
        misclassifications[idx] = 1
        
        if predicted == class_list.index('dachshund'):
            misclassification_count += 1
            print("\n\nDachshund misclassification found!")
            erm_model = algorithms.ERM(
                input_shape=torch.Size([3, 224, 224]),  # input shape
                    num_classes=4,  # number of classes
                    hparams=hparams,  # hparams
                    num_domains=2,  # number of training domains
            )
            erm_model.load_state_dict(model_dict)
            for param in erm_model.parameters():
                param.requires_grad = True
            erm_model.to('cuda')
            cam_extractor = SmoothGradCAMpp(erm_model.network)
            output = erm_model.network(image)
            activation_map = cam_extractor(class_list.index('dachshund'), output)
            result = overlay_mask(pil_image, to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
            plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
            # save image
            pil_image.save(os.path.join(dachshund_save_dir,f"original_dachshund_as_bulldog{misclassification_count}.png"))
            result.save(os.path.join(dachshund_save_dir,f"saliency_bulldog_as_dachshund{misclassification_count}.png"))

# %%