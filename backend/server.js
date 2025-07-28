const express = require('express');
const cors = require('cors');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const multer = require('multer');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// In-memory user storage (replace with database in production)
const users = [];
const detectionHistory = []; // Store detection history

// JWT Secret (use environment variable in production)
const JWT_SECRET = 'your-secret-key-change-in-production';

// Disease information and cure suggestions
const diseaseInfo = {
  'cataract': {
    name: 'Cataract',
    description: 'A clouding of the eye\'s natural lens that affects vision.',
    symptoms: ['Blurred vision', 'Difficulty seeing at night', 'Sensitivity to light', 'Fading colors', 'Double vision'],
    treatment: 'Surgery to remove the cloudy lens and replace it with an artificial lens (intraocular lens).',
    severity: 'Moderate',
    recommendations: [
      'Schedule an appointment with an ophthalmologist',
      'Consider cataract surgery when vision significantly affects daily activities',
      'Use proper lighting and anti-glare sunglasses',
      'Regular eye checkups every 6-12 months'
    ]
  },
  'diabetic_retinopathy': {
    name: 'Diabetic Retinopathy',
    description: 'Diabetes-related damage to the blood vessels in the retina.',
    symptoms: ['Blurred vision', 'Dark spots or floaters', 'Difficulty seeing colors', 'Vision loss', 'Fluctuating vision'],
    treatment: 'Laser treatment, injections, or surgery depending on severity. Blood sugar control is crucial.',
    severity: 'High',
    recommendations: [
      'Immediate consultation with a retina specialist',
      'Strict blood sugar control and monitoring',
      'Regular eye examinations every 3-6 months',
      'Blood pressure management',
      'Quit smoking if applicable'
    ]
  },
  'glaucoma': {
    name: 'Glaucoma',
    description: 'A group of eye conditions that damage the optic nerve, often due to high eye pressure.',
    symptoms: ['Gradual loss of peripheral vision', 'Eye pain', 'Nausea and vomiting', 'Blurred vision', 'Halos around lights'],
    treatment: 'Eye drops, laser treatment, or surgery to reduce eye pressure and prevent further damage.',
    severity: 'High',
    recommendations: [
      'Immediate consultation with a glaucoma specialist',
      'Regular eye pressure monitoring',
      'Use prescribed eye drops as directed',
      'Avoid activities that increase eye pressure',
      'Regular follow-up appointments'
    ]
  },
  'normal': {
    name: 'Normal Eye',
    description: 'No significant eye disease detected. Your eyes appear healthy.',
    symptoms: ['None'],
    treatment: 'Regular eye checkups recommended for preventive care.',
    severity: 'None',
    recommendations: [
      'Continue regular eye examinations every 1-2 years',
      'Maintain good eye hygiene',
      'Protect eyes from UV radiation with sunglasses',
      'Take breaks during screen time',
      'Maintain a healthy diet rich in eye-friendly nutrients'
    ]
  }
};

// Multer configuration for image upload
const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'), false);
    }
  },
});

// Authentication middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ message: 'Access token required' });
  }

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) {
      return res.status(403).json({ message: 'Invalid token' });
    }
    req.user = user;
    next();
  });
};

// Mock disease prediction function
function mockDiseasePrediction(imageBuffer) {
  // Simulate processing time
  const processingTime = Math.random() * 1000 + 500; // 500-1500ms
  
  // Disease classes
  const diseases = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal'];
  
  // Random prediction with bias towards normal (60% chance)
  let predictedDisease;
  const random = Math.random();
  
  if (random < 0.6) {
    predictedDisease = 'normal';
  } else if (random < 0.75) {
    predictedDisease = 'cataract';
  } else if (random < 0.9) {
    predictedDisease = 'glaucoma';
  } else {
    predictedDisease = 'diabetic_retinopathy';
  }
  
  // Generate confidence score
  const confidence = predictedDisease === 'normal' 
    ? Math.random() * 0.2 + 0.8  // 80-100% for normal
    : Math.random() * 0.3 + 0.7; // 70-100% for diseases
  
  // Generate probabilities for all classes
  const probabilities = {};
  diseases.forEach(disease => {
    if (disease === predictedDisease) {
      probabilities[disease] = confidence;
    } else {
      probabilities[disease] = (1 - confidence) / (diseases.length - 1);
    }
  });
  
  return {
    disease: predictedDisease,
    confidence: confidence,
    all_predictions: probabilities,
    processing_time: processingTime
  };
}

// Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', message: 'Eye Disease Detection API is running' });
});

// User registration
app.post('/api/auth/signup', async (req, res) => {
  try {
    const { name, email, password } = req.body;

    // Check if user already exists
    const existingUser = users.find(user => user.email === email);
    if (existingUser) {
      return res.status(400).json({ message: 'User already exists' });
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Create new user
    const newUser = {
      id: users.length + 1,
      name,
      email,
      password: hashedPassword,
      createdAt: new Date()
    };

    users.push(newUser);

    // Generate JWT token
    const token = jwt.sign(
      { userId: newUser.id, email: newUser.email },
      JWT_SECRET,
      { expiresIn: '24h' }
    );

    // Return user data (without password) and token
    const { password: _, ...userWithoutPassword } = newUser;
    res.status(201).json({
      message: 'User created successfully',
      token,
      user: userWithoutPassword
    });
  } catch (error) {
    console.error('Signup error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// User login
app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    // Find user
    const user = users.find(user => user.email === email);
    if (!user) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    // Check password
    const isValidPassword = await bcrypt.compare(password, user.password);
    if (!isValidPassword) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    // Generate JWT token
    const token = jwt.sign(
      { userId: user.id, email: user.email },
      JWT_SECRET,
      { expiresIn: '24h' }
    );

    // Return user data (without password) and token
    const { password: _, ...userWithoutPassword } = user;
    res.json({
      message: 'Login successful',
      token,
      user: userWithoutPassword
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

// Disease detection endpoint
app.post('/api/detect', authenticateToken, upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No image file provided' });
    }

    console.log('Processing image for user:', req.user.userId);

    // Process image with Sharp
    const processedImageBuffer = await sharp(req.file.buffer)
      .resize(224, 224) // Resize to model input size
      .jpeg({ quality: 90 })
      .toBuffer();

    // Mock disease prediction
    const prediction = mockDiseasePrediction(processedImageBuffer);
    
    // Get disease information
    const diseaseData = diseaseInfo[prediction.disease];
    
    // Create detection record
    const detectionRecord = {
      id: detectionHistory.length + 1,
      userId: req.user.userId,
      imagePath: req.file.originalname || 'uploaded_image.jpg',
      disease: prediction.disease,
      confidence: prediction.confidence,
      timestamp: new Date(),
      diseaseInfo: diseaseData
    };

    // Save to history
    detectionHistory.push(detectionRecord);

    console.log('Detection completed:', prediction.disease, 'with confidence:', prediction.confidence);

    // Return comprehensive result
    res.json({
      disease: prediction.disease,
      confidence: prediction.confidence,
      all_predictions: prediction.all_predictions,
      diseaseInfo: diseaseData,
      message: 'Detection completed successfully',
      timestamp: detectionRecord.timestamp
    });

  } catch (error) {
    console.error('Detection error:', error);
    res.status(500).json({ message: 'Error processing image' });
  }
});

// Get user detection history
app.get('/api/history', authenticateToken, (req, res) => {
  try {
    // Get user's detection history
    const userHistory = detectionHistory.filter(record => record.userId === req.user.userId);
    
    // If no history, return some mock data
    if (userHistory.length === 0) {
      const mockHistory = [
        {
          id: 1,
          userId: req.user.userId,
          imagePath: 'eye_scan_1.jpg',
          disease: 'normal',
          confidence: 0.95,
          timestamp: new Date(Date.now() - 86400000), // 1 day ago
          diseaseInfo: diseaseInfo.normal
        },
        {
          id: 2,
          userId: req.user.userId,
          imagePath: 'eye_scan_2.jpg',
          disease: 'cataract',
          confidence: 0.87,
          timestamp: new Date(Date.now() - 172800000), // 2 days ago
          diseaseInfo: diseaseInfo.cataract
        }
      ];
      return res.json(mockHistory);
    }

    res.json(userHistory);
  } catch (error) {
    console.error('History error:', error);
    res.status(500).json({ message: 'Error fetching history' });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Error:', error);
  res.status(500).json({ message: 'Internal server error' });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/api/health`);
});
