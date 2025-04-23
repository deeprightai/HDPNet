import torch
import HDPNet_model
import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import functional as F

def create_side_by_side(original, prediction):
    # Resize original image to match prediction size
    original = cv2.resize(original, (384, 384))
    # Convert prediction to RGB
    prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2RGB)
    # Create side by side image
    h, w = original.shape[:2]
    combined = np.zeros((h, w*2, 3), dtype=np.uint8)
    combined[:, :w] = original
    combined[:, w:] = prediction
    return combined

def main():
    # Create output directories
    os.makedirs('results/predictions', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HDPNet_model.Model(None, img_size=384).to(device)
    
    # Load weights
    weights_path = 'weights/model_108_loss_0.02126.pth'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process test images
    test_images = [f for f in os.listdir('testdata') if f.endswith(('.jpg', '.png'))]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('results/visualization.mp4', fourcc, 2.0, (384*2, 384))
    
    print("Processing test images...")
    for img_name in tqdm(test_images):
        # Load and preprocess image
        img_path = os.path.join('testdata', img_name)
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Prepare input
        img = Image.fromarray(original_img)
        img = transform(img).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(img)[2]
        
        # Process output
        output = F.interpolate(output, (384, 384), mode='bilinear', align_corners=True)
        output = output.squeeze().cpu().numpy()
        output = ((output - output.min()) / (output.max() - output.min() + 1e-8) * 255).astype(np.uint8)
        
        # Save prediction
        cv2.imwrite(f'results/predictions/{img_name}', output)
        
        # Create and save visualization
        visualization = create_side_by_side(original_img, output)
        cv2.imwrite(f'results/visualizations/{img_name}', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        
        # Add frame to video
        video_writer.write(cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    
    # Release video writer
    video_writer.release()
    print("Inference completed! Results saved in 'results' directory.")

if __name__ == '__main__':
    main() 