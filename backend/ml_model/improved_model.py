import os
import numpy as np
import cv2
from PIL import Image
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.filters import sobel
import warnings
warnings.filterwarnings('ignore')

class ImprovedEyeDiseaseModel:
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def extract_features(self, image):
        """Extract comprehensive features from eye image"""
        features = []
        
        # Ensure image is in uint8 format for OpenCV
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert to grayscale for texture analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # 1. Color features (RGB channels)
        if len(image.shape) == 3:
            for channel in range(3):
                features.extend([
                    np.mean(image[:, :, channel]),
                    np.std(image[:, :, channel]),
                    np.median(image[:, :, channel]),
                    np.percentile(image[:, :, channel], 25),
                    np.percentile(image[:, :, channel], 75)
                ])
        
        # 2. Grayscale features
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.median(gray),
            np.percentile(gray, 25),
            np.percentile(gray, 75),
            np.max(gray),
            np.min(gray)
        ])
        
        # 3. Texture features using Local Binary Pattern
        try:
            lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
            features.extend(lbp_hist)
        except:
            features.extend([0] * 10)
        
        # 4. Edge features
        try:
            edges = sobel(gray)
            features.extend([
                np.mean(edges),
                np.std(edges),
                np.sum(edges > np.mean(edges))
            ])
        except:
            features.extend([0, 0, 0])
        
        # 5. GLCM (Gray Level Co-occurrence Matrix) features
        try:
            glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')
            dissimilarity = graycoprops(glcm, 'dissimilarity')
            homogeneity = graycoprops(glcm, 'homogeneity')
            energy = graycoprops(glcm, 'energy')
            correlation = graycoprops(glcm, 'correlation')
            
            features.extend([
                np.mean(contrast), np.std(contrast),
                np.mean(dissimilarity), np.std(dissimilarity),
                np.mean(homogeneity), np.std(homogeneity),
                np.mean(energy), np.std(energy),
                np.mean(correlation), np.std(correlation)
            ])
        except:
            features.extend([0] * 10)
        
        # 6. Shape and morphological features
        try:
            # Threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = w / h if h > 0 else 0
                
                features.extend([area, perimeter, circularity, aspect_ratio])
            else:
                features.extend([0, 0, 0, 0])
        except:
            features.extend([0, 0, 0, 0])
        
        # 7. Histogram features
        try:
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            features.extend([
                np.mean(hist),
                np.std(hist),
                np.argmax(hist),  # Peak
                np.sum(hist > np.mean(hist))  # Above mean count
            ])
        except:
            features.extend([0, 0, 0, 0])
        
        return np.array(features)
    
    def load_and_preprocess_data(self, data_dir):
        """Load and preprocess the dataset"""
        images = []
        labels = []
        features_list = []
        
        class_mapping = {
            'cataract': 0,
            'diabetic_retinopathy': 1,
            'glaucoma': 2,
            'normal': 3
        }
        
        print("Loading dataset and extracting features...")
        
        for class_name in self.class_names:
            class_dir = Path(data_dir) / class_name
            if not class_dir.exists():
                print(f"Warning: Directory {class_dir} not found. Creating synthetic data...")
                self.create_synthetic_data(images, labels, features_list, class_mapping[class_name], 200)
                continue
                
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
            
            if not image_files:
                print(f"Warning: No images found in {class_dir}. Creating synthetic data...")
                self.create_synthetic_data(images, labels, features_list, class_mapping[class_name], 200)
                continue
            
            print(f"Processing {len(image_files)} images from {class_name}...")
            
            for img_path in image_files:
                try:
                    # Load and preprocess image
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        
                        # Extract features
                        features = self.extract_features(img)
                        
                        images.append(img)
                        labels.append(class_mapping[class_name])
                        features_list.append(features)
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels)
        
        print(f"Dataset loaded: {len(X)} samples with {X.shape[1]} features")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def create_synthetic_data(self, images, labels, features_list, class_label, num_samples):
        """Create more realistic synthetic data"""
        print(f"Creating {num_samples} synthetic samples for class {class_label}")
        
        for i in range(num_samples):
            # Create base image in uint8 format
            img = (np.random.rand(self.img_size, self.img_size, 3) * 255).astype(np.uint8)
            
            # Add realistic patterns based on disease type
            if class_label == 0:  # cataract - cloudy, hazy appearance
                # Add white/cloudy patches
                for _ in range(np.random.randint(3, 8)):
                    x, y = np.random.randint(0, self.img_size, 2)
                    radius = np.random.randint(10, 30)
                    cv2.circle(img, (x, y), radius, (204, 204, 204), -1)  # Light gray
                img = np.clip(img + np.random.normal(0, 25, img.shape), 0, 255).astype(np.uint8)
                
            elif class_label == 1:  # diabetic retinopathy - red spots and vessels
                # Add red spots
                for _ in range(np.random.randint(5, 15)):
                    x, y = np.random.randint(0, self.img_size, 2)
                    radius = np.random.randint(2, 8)
                    cv2.circle(img, (x, y), radius, (204, 51, 51), -1)  # Red
                # Add vessel-like structures
                for _ in range(np.random.randint(3, 8)):
                    x1, y1 = np.random.randint(0, self.img_size, 2)
                    x2, y2 = np.random.randint(0, self.img_size, 2)
                    cv2.line(img, (x1, y1), (x2, y2), (153, 26, 26), 2)  # Dark red
                
            elif class_label == 2:  # glaucoma - dark areas and pressure signs
                # Add dark areas
                for _ in range(np.random.randint(2, 6)):
                    x, y = np.random.randint(0, self.img_size, 2)
                    radius = np.random.randint(15, 40)
                    cv2.circle(img, (x, y), radius, (26, 26, 26), -1)  # Dark gray
                img = (img * 0.6).astype(np.uint8)  # Darken overall
                
            else:  # normal - clean, clear appearance
                # Add subtle variations
                img = np.clip(img + np.random.normal(0, 13, img.shape), 0, 255).astype(np.uint8)
            
            # Extract features
            features = self.extract_features(img)
            
            images.append(img)
            labels.append(class_label)
            features_list.append(features)
    
    def train(self, X, y):
        """Train the improved model"""
        print("Training improved model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try multiple models and select the best one
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        best_score = 0
        best_model = None
        best_name = None
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            print(f"{name} accuracy: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        
        self.model = best_model
        print(f"\nBest model: {best_name} with accuracy: {best_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Detailed evaluation
        y_pred = best_model.predict(X_test_scaled)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        return best_score
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix - Improved Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('improved_confusion_matrix.png')
        plt.show()
    
    def predict(self, image):
        """Predict disease from image"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Preprocess image
        if len(image.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img = image
            
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Extract features
        features = self.extract_features(img)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        return {
            'disease': self.class_names[prediction],
            'confidence': float(np.max(probabilities)),
            'all_predictions': dict(zip(self.class_names, probabilities.tolist())),
            'processing_time': 0.1  # Simulated processing time
        }
    
    def save_model(self, model_path='improved_eye_disease_model.pkl'):
        """Save the trained model"""
        if self.model:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'class_names': self.class_names,
                'img_size': self.img_size
            }
            joblib.dump(model_data, model_path)
            print(f"Model saved to {model_path}")
            
            # Save class names separately
            with open('class_names.json', 'w') as f:
                json.dump(self.class_names, f)

def main():
    # Initialize model
    model = ImprovedEyeDiseaseModel()
    
    # Load and preprocess data
    data_dir = "dataset"
    X, y = model.load_and_preprocess_data(data_dir)
    
    # Train model
    accuracy = model.train(X, y)
    
    # Save model
    model.save_model()
    
    print(f"\nTraining completed! Model accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 