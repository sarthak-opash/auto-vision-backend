# auto_label_then_correct.py

import os
import json
import torch
import shutil
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import torchvision.models as models
import torchvision.transforms as transforms

class AutoLabeler:
    def __init__(self):
        # Load pretrained ResNet50
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        
        # ImageNet classes (we'll map to car types)
        self.load_imagenet_classes()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Car-related ImageNet classes (approximate mapping)
        self.car_type_mapping = {
            'sports_car': 'sports',
            'convertible': 'sports',
            'racer': 'sports',
            'beach_wagon': 'van',
            'minivan': 'van',
            'pickup': 'truck',
            'tow_truck': 'truck',
            'limousine': 'sedan',
            'Model_T': 'sedan',
            'cab': 'sedan',
        }
    
    def load_imagenet_classes(self):
        """Load ImageNet class labels"""
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        
        try:
            with urllib.request.urlopen(url) as f:
                self.imagenet_classes = [line.decode('utf-8').strip() for line in f.readlines()]
        except:
            print("Warning: Could not load ImageNet classes from URL")
            self.imagenet_classes = []
    
    def predict_car_type(self, image_path):
        """Predict car type from image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top5_prob, top5_idx = torch.topk(probabilities, 5)
            
            # Check if any top-5 prediction is car-related
            for idx, prob in zip(top5_idx[0], top5_prob[0]):
                class_name = self.imagenet_classes[idx]
                if class_name in self.car_type_mapping:
                    return self.car_type_mapping[class_name], prob.item()
            
            # Default to 'unknown' if no car class detected
            return 'unknown', 0.0
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return 'unknown', 0.0
    
    def auto_label_dataset(self, input_folder, output_folder):
        """Auto-label all images"""
        
        categories = ['sedan', 'suv', 'truck', 'coupe', 'hatchback', 'van', 'sports', 'unknown']
        
        # Create output directories
        for category in categories:
            os.makedirs(os.path.join(output_folder, category), exist_ok=True)
        
        # Get all images
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        predictions = {}
        
        print("Auto-labeling images...")
        for img_file in tqdm(image_files):
            img_path = os.path.join(input_folder, img_file)
            
            predicted_type, confidence = self.predict_car_type(img_path)
            
            # Copy to predicted category
            dest_path = os.path.join(output_folder, predicted_type, img_file)
            shutil.copy2(img_path, dest_path)
            
            predictions[img_file] = {
                'predicted_type': predicted_type,
                'confidence': confidence
            }
        
        # Save predictions
        with open(os.path.join(output_folder, 'auto_labels.json'), 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("AUTO-LABELING SUMMARY")
        print("=" * 60)
        for category in categories:
            count = sum(1 for p in predictions.values() if p['predicted_type'] == category)
            print(f"{category}: {count} images")
        print("=" * 60)
        print("\n⚠️  IMPORTANT: Review and correct labels manually!")
        print("Many images may be in 'unknown' - you need to label these manually.")
        
        return predictions


if __name__ == '__main__':
    labeler = AutoLabeler()
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    INPUT_FOLDER = BASE_DIR / 'datasets' / 'damage' / 'train' / 'images'
    OUTPUT_FOLDER = BASE_DIR / 'auto_labeled_cars'

    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER}")
    
    predictions = labeler.auto_label_dataset(str(INPUT_FOLDER), str(OUTPUT_FOLDER))
    
    print("\nNext step: Use the simple_labeler.py tool to correct the labels!")