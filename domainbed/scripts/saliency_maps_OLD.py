%%
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from domainbed import algorithms
import torchvision
import domainbed.networks
from domainbed import hparams_registry


# Load a pre-trained model
pretrained_model_path = "/home/aengusl/Desktop/Projects/OOD_workshop/DomainBed-SP/erm_output/resnet50_SpawriousM2M_easy_ERM_model.pkl"
model_dict = torch.load(pretrained_model_path)["model_dict"]
model = torchvision.models.resnet50(weights=model_dict)
model.eval()
import os
for image_file in os.listdir("/home/aengusl/Desktop/Projects/OOD_workshop/DomainBed-SP/data/misclassifications/"):
# Load an input image
    print (image_file)
    image_path = os.path.join("/home/aengusl/Desktop/Projects/OOD_workshop/DomainBed-SP/data/misclassifications/", image_file)
    number_label = image_file.split(".")[0]
    input_image = Image.open(image_path).convert('RGB')
    # Preprocess the input image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    # Calculate gradients
    input_batch.requires_grad_()
    output = model(input_batch)
    output_class = output.argmax(dim=1).item()
    output[0, output_class].backward()
    # Visualize the saliency map
    saliency_map = input_batch.grad.data.squeeze().numpy()
    saliency_map = np.max(np.abs(saliency_map), axis=0)




    plt.imshow(saliency_map, cmap='hot')
    plt.axis('off')
    plt.savefig(f'saliency_map_{number_label}.png')  # Save the saliency map to a file
    plt.show()



# import os
# import matplotlib.pyplot as plt
# from PIL import Image

# # Define the directory where the images are stored
# # image_dir = "/path/to/images"

# # Define the size of the table
# num_rows = 16
# num_cols = 2

# # Create a new figure
# fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 6))

# # Loop over the images and add them to the table
# for i, image_file in enumerate(os.listdir("/home/aengusl/Desktop/Projects/OOD_workshop/DomainBed-SP/domainbed/saliency_maps")):
#     # Load the image
#     image_number = int(image_file.split(".")[0].split("_")[-1])
#     image_path = os.path.join("/home/aengusl/Desktop/Projects/OOD_workshop/DomainBed-SP/domainbed/saliency_maps", image_file)
#     saliency_image = Image.open(image_path)

#     original_image = Image.open(os.path.join("/home/aengusl/Desktop/Projects/OOD_workshop/DomainBed-SP/data/misclassifications/", f"{image_number}.png"))

#     # Calculate the row and column indices for the current image
#     # row_idx = i // num_cols
#     # col_idx = i % num_cols
#     row_idx = image_number-1

#     # Display the image in the corresponding cell of the table
#     axs[row_idx, 0].imshow(original_image)
#     axs[row_idx, 0].axis('off')
#     axs[row_idx, 0].set_title(image_number)

#     axs[row_idx, 1].imshow(saliency_image)
#     axs[row_idx, 1].axis('off')
#     axs[row_idx, 1].set_title(image_number)

# fig.subplots_adjust(wspace=0.0000001, hspace=0.000000001)
# # Save the table to a file
# plt.savefig("image_table.png")

# # Show the table
# plt.show()