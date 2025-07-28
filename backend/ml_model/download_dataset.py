import os
import shutil
import zipfile
import kagglehub
from pathlib import Path

def download_and_prepare_dataset():
    """Download and prepare the eye diseases classification dataset"""
    print("🚀 Downloading Eye Diseases Classification Dataset...")
    
    try:
        # Download the dataset using kagglehub
        print("Downloading dataset from Kaggle...")
        dataset_path = kagglehub.dataset_download("gunavenkatdoddi/eye-diseases-classification")
        print(f"Dataset downloaded to: {dataset_path}")
        
        # Create dataset directory structure
        dataset_dir = Path("dataset")
        dataset_dir.mkdir(exist_ok=True)
        
        # Expected class directories
        class_dirs = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
        
        # Create class directories
        for class_name in class_dirs:
            (dataset_dir / class_name).mkdir(exist_ok=True)
        
        # Find and extract the dataset files
        print("Extracting and organizing dataset...")
        
        # Look for zip files in the downloaded path
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.zip'):
                    zip_path = os.path.join(root, file)
                    print(f"Found zip file: {zip_path}")
                    
                    # Extract the zip file
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall("temp_extract")
                    
                    # Organize the extracted files
                    organize_extracted_files("temp_extract", dataset_dir, class_dirs)
                    
                    # Clean up temp directory
                    if os.path.exists("temp_extract"):
                        shutil.rmtree("temp_extract")
                    
                    break  # Process only the first zip file
        
        print("✅ Dataset preparation completed!")
        print(f"Dataset organized in: {dataset_dir.absolute()}")
        
        # Print dataset statistics
        print_dataset_stats(dataset_dir, class_dirs)
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("Creating synthetic dataset as fallback...")
        create_synthetic_dataset()

def organize_extracted_files(extract_path, dataset_dir, class_dirs):
    """Organize extracted files into the correct class directories"""
    print("Organizing files into class directories...")
    
    # Map common variations of class names
    class_mapping = {
        'cataract': ['cataract', 'Cataract', 'CATARACT'],
        'diabetic_retinopathy': ['diabetic_retinopathy', 'diabetic', 'Diabetic', 'retinopathy', 'Retinopathy'],
        'glaucoma': ['glaucoma', 'Glaucoma', 'GLAUCOMA'],
        'normal': ['normal', 'Normal', 'NORMAL', 'healthy', 'Healthy']
    }
    
    # Walk through extracted files
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                
                # Determine class based on directory name or file path
                target_class = None
                path_parts = root.lower().split(os.sep)
                
                for class_name, variations in class_mapping.items():
                    if any(var in path_parts for var in variations):
                        target_class = class_name
                        break
                
                # If no class found, try to infer from filename
                if target_class is None:
                    file_lower = file.lower()
                    for class_name, variations in class_mapping.items():
                        if any(var in file_lower for var in variations):
                            target_class = class_name
                            break
                
                # If still no class found, default to normal
                if target_class is None:
                    target_class = 'normal'
                
                # Copy file to appropriate class directory
                target_path = dataset_dir / target_class / file
                shutil.copy2(file_path, target_path)
                print(f"Copied {file} to {target_class}/")

def create_synthetic_dataset():
    """Create synthetic dataset as fallback"""
    print("Creating synthetic dataset...")
    
    dataset_dir = Path("dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    class_dirs = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
    
    for class_name in class_dirs:
        class_dir = dataset_dir / class_name
        class_dir.mkdir(exist_ok=True)
        print(f"Created directory: {class_dir}")
    
    print("✅ Synthetic dataset structure created")
    print("Please add your own eye disease images to the respective class directories.")

def print_dataset_stats(dataset_dir, class_dirs):
    """Print statistics about the prepared dataset"""
    print("\n📊 Dataset Statistics:")
    print("=" * 40)
    
    total_images = 0
    for class_name in class_dirs:
        class_dir = dataset_dir / class_name
        if class_dir.exists():
            image_count = len([f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            total_images += image_count
            print(f"{class_name:20}: {image_count:4} images")
    
    print("=" * 40)
    print(f"Total images: {total_images}")
    
    if total_images == 0:
        print("\n⚠️  No images found in dataset directories.")
        print("Please ensure the dataset was downloaded and extracted correctly.")
        print("You can manually add images to the class directories:")
        for class_name in class_dirs:
            print(f"  - dataset/{class_name}/")

if __name__ == "__main__":
    download_and_prepare_dataset() 