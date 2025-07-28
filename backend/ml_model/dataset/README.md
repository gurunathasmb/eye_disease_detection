# Eye Disease Dataset Structure

This directory should contain your eye disease images organized by class. The model expects the following structure:

```
dataset/
├── cataract/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── diabetic_retinopathy/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── glaucoma/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── normal/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Image Requirements

- **Format**: JPG, JPEG, or PNG
- **Size**: Any size (will be resized to 224x224 during preprocessing)
- **Channels**: RGB (3 channels)
- **Quality**: Clear, well-lit images of the eye

## Recommended Dataset Sources

1. **Kaggle**: Search for "eye disease" or "retinal images"
2. **NIH Chest X-ray Dataset**: Contains some eye-related images
3. **Medical Image Datasets**: Various medical imaging repositories
4. **Research Papers**: Datasets mentioned in ophthalmology research

## Minimum Dataset Size

For a basic working model:
- **Per class**: At least 50-100 images
- **Total**: 200-400 images minimum
- **Recommended**: 500+ images per class for better accuracy

## Data Augmentation

The training script automatically applies data augmentation:
- Rotation (±20 degrees)
- Width/height shifts (±20%)
- Horizontal flipping
- Zoom (±20%)
- Brightness/contrast adjustments

## Synthetic Data

If you don't have real data, the training script will create synthetic data for demonstration purposes. However, for production use, you should use real medical images.

## Important Notes

⚠️ **Medical Disclaimer**: This is for educational purposes only. Real medical diagnosis should be performed by qualified healthcare professionals.

⚠️ **Data Privacy**: Ensure you have proper permissions to use any medical images.

⚠️ **Model Accuracy**: The model's accuracy depends heavily on the quality and quantity of training data. 