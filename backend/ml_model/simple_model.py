import os
import numpy as np
import cv2
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class SimpleEyeDiseaseModel:
    def __init__(self):
        self.model = None
        self.class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
        
    def extract_features(self, image):
        """Extract simple features from image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Basic features
        features = []
        
        # Mean intensity
        features.append(np.mean(gray))
        
        # Standard deviation
        features.append(np.std(gray))
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features.extend(hist.flatten()[:10])  # First 10 histogram bins
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))
        
        # Color features (RGB means)
        features.append(np.mean(image[:, :, 0]))  # Red
        features.append(np.mean(image[:, :, 1]))  # Green
        features.append(np.mean(image[:, :, 2]))  # Blue
        
        return np.array(features)
    
    def load_and_preprocess_data(self, data_dir):
        """Load and preprocess the dataset"""
        images = []
        labels = []
        
        # Define class mapping
        class_mapping = {
            'cataract': 0,
            'diabetic_retinopathy': 1,
            'glaucoma': 2,
            'normal': 3
        }
        
        print("Loading dataset...")
        
        # Load images from each class directory
        for class_name in self.class_names:
            class_dir = Path(data_dir) / class_name
            if not class_dir.exists():
                print(f"Warning: Directory {class_dir} not found. Creating synthetic data...")
                self.create_synthetic_data(images, labels, class_mapping[class_name], 50)
                continue
                
            # Get all image files
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
            
            if not image_files:
                print(f"Warning: No images found in {class_dir}. Creating synthetic data...")
                self.create_synthetic_data(images, labels, class_mapping[class_name], 50)
                continue
            
            print(f"Loading {len(image_files)} images from {class_name}...")
            
            for img_path in image_files:
                try:
                    # Load and preprocess image
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224))
                        
                        images.append(img)
                        labels.append(class_mapping[class_name])
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        # Extract features
        print("Extracting features...")
        features = []
        for img in images:
            features.append(self.extract_features(img))
        
        X = np.array(features)
        y = np.array(labels)
        
        print(f"Dataset loaded: {len(X)} samples, {len(y)} labels")
        print(f"Feature dimension: {X.shape[1]}")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def create_synthetic_data(self, images, labels, class_label, num_samples):
        """Create synthetic data for demonstration purposes"""
        print(f"Creating {num_samples} synthetic samples for class {class_label}")
        
        for i in range(num_samples):
            # Create a random image with some patterns
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Add some patterns based on class
            if class_label == 0:  # cataract - add cloudy patterns
                img = cv2.add(img, np.random.randint(0, 50, img.shape, dtype=np.uint8))
            elif class_label == 1:  # diabetic retinopathy - add red patterns
                img[:, :, 0] = np.clip(img[:, :, 0] + np.random.randint(0, 100, (224, 224), dtype=np.uint8), 0, 255)
            elif class_label == 2:  # glaucoma - add dark patterns
                img = cv2.multiply(img, 0.7)
            else:  # normal - keep mostly clean
                pass
            
            images.append(img)
            labels.append(class_label)
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model"""
        print("Training Random Forest model...")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_score = self.model.score(X_val, y_val)
        print(f"Validation accuracy: {val_score:.4f}")
        
        return val_score
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        print("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        return y_pred
    
    def save_model(self, model_path='eye_disease_model.joblib'):
        """Save the trained model"""
        if self.model:
            joblib.dump(self.model, model_path)
            print(f"Model saved to {model_path}")
            
            # Save class names
            with open('class_names.json', 'w') as f:
                json.dump(self.class_names, f)
    
    def predict(self, image_path):
        """Predict disease from image"""
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not load image")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            
            # Extract features
            features = self.extract_features(img)
            
            # Make prediction
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            
            predicted_disease = self.class_names[prediction]
            confidence = float(probabilities[prediction])
            
            return {
                'disease': predicted_disease,
                'confidence': confidence,
                'all_predictions': {
                    self.class_names[i]: float(probabilities[i]) 
                    for i in range(len(self.class_names))
                }
            }
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize model
    model_trainer = SimpleEyeDiseaseModel()
    
    # Load data
    data_dir = "dataset"
    if not os.path.exists(data_dir):
        print(f"Dataset directory {data_dir} not found. Creating synthetic dataset...")
        os.makedirs(data_dir, exist_ok=True)
        
        # Create synthetic dataset structure
        for class_name in model_trainer.class_names:
            os.makedirs(os.path.join(data_dir, class_name), exist_ok=True)
    
    # Load and preprocess data
    X, y = model_trainer.load_and_preprocess_data(data_dir)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    val_score = model_trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    y_pred = model_trainer.evaluate(X_test, y_test)
    
    # Save model
    model_trainer.save_model()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 