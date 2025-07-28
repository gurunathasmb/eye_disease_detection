import sys
import os
import json
import random

def mock_predict_disease(image_path):
    """Mock disease prediction for demonstration"""
    
    # Simulate some processing time
    import time
    time.sleep(0.5)
    
    # Mock disease classes
    diseases = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
    
    # Random prediction with some bias towards normal
    if random.random() < 0.4:
        predicted_disease = 'normal'
        confidence = random.uniform(0.8, 0.95)
    else:
        predicted_disease = random.choice(diseases)
        confidence = random.uniform(0.7, 0.9)
    
    # Create mock probabilities
    probabilities = [random.uniform(0.1, 0.3) for _ in diseases]
    disease_index = diseases.index(predicted_disease)
    probabilities[disease_index] = confidence
    
    # Normalize probabilities
    total = sum(probabilities)
    probabilities = [p/total for p in probabilities]
    
    return {
        'disease': predicted_disease,
        'confidence': confidence,
        'all_predictions': {
            diseases[i]: float(probabilities[i]) 
            for i in range(len(diseases))
        }
    }

def main():
    """Main function called by Node.js"""
    if len(sys.argv) != 2:
        print("Usage: python mock_predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        sys.exit(1)
    
    # Make mock prediction
    result = mock_predict_disease(image_path)
    
    # Print result as JSON (this will be captured by Node.js)
    print(json.dumps(result))

if __name__ == "__main__":
    main() 