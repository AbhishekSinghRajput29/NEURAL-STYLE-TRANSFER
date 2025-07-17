import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loader and preprocessor
def load_image(path, max_size=400, shape=None):
    image = Image.open(path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape:
        size = shape
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    image = in_transform(image).unsqueeze(0)
    return image.to(device)

# Gram matrix for style representation
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(b * c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram.div(b * c * h * w)

# Load images
content = load_image('content.jpg')
style = load_image('style.jpg', shape=content.shape[-2:])

# Load VGG19 pretrained model
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

# Freeze parameters
for param in vgg.parameters():
    param.requires_grad = False

# Layers to use for style/content
content_layers = ['21']  # 'conv4_2'
style_layers = ['0', '5', '10', '19', '28']  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1

# Extract features function
def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

# Get features for content and style
content_features = get_features(content, vgg, content_layers)
style_features = get_features(style, vgg, style_layers)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}

# Initialize target as content image clone
target = content.clone().requires_grad_(True).to(device)

# Hyperparameters
style_weight = 1e6
content_weight = 1

# Optimizer
optimizer = optim.Adam([target], lr=0.003)

print("🚀 Starting Neural Style Transfer...")

# Training loop
steps = 300
for step in range(steps):
    optimizer.zero_grad()
    target_features = get_features(target, vgg, content_layers + style_layers)
    
    content_loss = torch.mean((target_features[content_layers[0]] - content_features[content_layers[0]]) ** 2)
    
    style_loss = 0
    for layer in style_layers:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        style_loss += torch.mean((target_gram - style_gram) ** 2)
        
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Total loss: {total_loss.item():.4f}")

# Postprocessing and saving output image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    image = image.clip(0, 1)
    image = (image * 255).astype('uint8')
    return Image.fromarray(image)

output_image = im_convert(target)
output_image.save('output.jpg')
print("✅ Style transfer complete. Output saved as output.jpg")
