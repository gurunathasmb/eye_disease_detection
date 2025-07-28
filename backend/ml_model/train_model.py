import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import cv2
from PIL import Image
import json
from pathlib import Path

class EyeDiseaseModel:
    def __init__(self, img_size=224, num_classes=4):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
        
    def create_model(self):
        """Create a CNN model for eye disease classification"""
        # Use MobileNetV2 as base model for transfer learning
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create the model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
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
                # Create synthetic data for demonstration
                self.create_synthetic_data(images, labels, class_mapping[class_name], 100)
                continue
                
            # Get all image files
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
            
            if not image_files:
                print(f"Warning: No images found in {class_dir}. Creating synthetic data...")
                self.create_synthetic_data(images, labels, class_mapping[class_name], 100)
                continue
            
            print(f"Loading {len(image_files)} images from {class_name}...")
            
            for img_path in image_files:
                try:
                    # Load and preprocess image
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        img = img / 255.0  # Normalize
                        
                        images.append(img)
                        labels.append(class_mapping[class_name])
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Convert labels to categorical
        y = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
        
        print(f"Dataset loaded: {len(X)} images, {len(y)} labels")
        print(f"Class distribution: {np.sum(y, axis=0)}")
        
        return X, y
    
    def create_synthetic_data(self, images, labels, class_label, num_samples):
        """Create synthetic data for demonstration purposes"""
        print(f"Creating {num_samples} synthetic samples for class {class_label}")
        
        for i in range(num_samples):
            # Create a random image with some patterns
            img = np.random.rand(self.img_size, self.img_size, 3)
            
            # Add some patterns based on class
            if class_label == 0:  # cataract - add cloudy patterns
                img += np.random.normal(0, 0.1, img.shape)
                img = np.clip(img, 0, 1)
            elif class_label == 1:  # diabetic retinopathy - add red patterns
                img[:, :, 0] += np.random.normal(0.2, 0.1, (self.img_size, self.img_size))
                img = np.clip(img, 0, 1)
            elif class_label == 2:  # glaucoma - add dark patterns
                img *= 0.7
                img = np.clip(img, 0, 1)
            else:  # normal - keep mostly clean
                img = np.clip(img, 0, 1)
            
            images.append(img)
            labels.append(class_label)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        print("Training model...")
        
        # Data augmentation for training
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Train the model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            steps_per_epoch=len(X_train) // batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        print("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=self.class_names))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
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
    
    def save_model(self, model_path='eye_disease_model.h5'):
        """Save the trained model"""
        if self.model:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
            
            # Save class names
            with open('class_names.json', 'w') as f:
                json.dump(self.class_names, f)
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Initialize model
    model_trainer = EyeDiseaseModel()
    
    # Create model
    model = model_trainer.create_model()
    print("Model created successfully")
    
    # Load data (replace with your actual dataset path)
    data_dir = "dataset"  # Create this directory and add your dataset
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
    history = model_trainer.train(X_train, y_train, X_val, y_val, epochs=30)
    
    # Plot training history
    model_trainer.plot_training_history(history)
    
    # Evaluate model
    y_pred = model_trainer.evaluate(X_test, y_test)
    
    # Save model
    model_trainer.save_model()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 