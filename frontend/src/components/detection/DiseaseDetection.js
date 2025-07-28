import React, { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import Webcam from 'react-webcam';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { 
  Camera, 
  Upload, 
  ArrowLeft, 
  RotateCcw, 
  Download,
  Eye,
  AlertTriangle,
  CheckCircle,
  Loader2,
  Clock,
  Heart,
  Shield,
  FileText
} from 'lucide-react';

const DiseaseDetection = () => {
  const [activeTab, setActiveTab] = useState('camera'); // 'camera' or 'upload'
  const [capturedImage, setCapturedImage] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const webcamRef = useRef(null);
  const navigate = useNavigate();

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    setCapturedImage(imageSrc);
  }, [webcamRef]);

  const retake = () => {
    setCapturedImage(null);
    setAnalysisResult(null);
  };

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setUploadedImage(reader.result);
        setAnalysisResult(null);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: false
  });

  const analyzeImage = async (imageData) => {
    setIsAnalyzing(true);
    try {
      const formData = new FormData();
      
      // Convert base64 to blob
      const response = await fetch(imageData);
      const blob = await response.blob();
      formData.append('image', blob, 'eye_image.jpg');

      const result = await axios.post('http://localhost:5000/api/detect', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setAnalysisResult(result.data);
      toast.success('Analysis completed successfully!');
    } catch (error) {
      console.error('Analysis error:', error);
      toast.error('Failed to analyze image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAnalyze = () => {
    const imageToAnalyze = activeTab === 'camera' ? capturedImage : uploadedImage;
    if (imageToAnalyze) {
      analyzeImage(imageToAnalyze);
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <button
                onClick={() => navigate('/dashboard')}
                className="flex items-center space-x-2 text-gray-600 hover:text-gray-900"
              >
                <ArrowLeft className="h-5 w-5" />
                <span>Back to Dashboard</span>
              </button>
            </div>
            <h1 className="text-2xl font-bold text-gray-900">Eye Disease Detection</h1>
            <div></div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Image Capture/Upload */}
          <div className="space-y-6">
            {/* Tab Navigation */}
            <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
              <button
                onClick={() => setActiveTab('camera')}
                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'camera'
                    ? 'bg-white text-blue-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Camera className="h-4 w-4 inline mr-2" />
                Camera
              </button>
              <button
                onClick={() => setActiveTab('upload')}
                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'upload'
                    ? 'bg-white text-blue-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Upload className="h-4 w-4 inline mr-2" />
                Upload
              </button>
            </div>

            {/* Camera Tab */}
            {activeTab === 'camera' && (
              <div className="space-y-4">
                {!capturedImage ? (
                  <div className="bg-white rounded-lg shadow-sm border p-6">
                    <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
                      <Webcam
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className="mt-4 flex justify-center">
                      <button
                        onClick={capture}
                        className="flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                      >
                        <Camera className="h-5 w-5" />
                        <span>Capture Photo</span>
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="bg-white rounded-lg shadow-sm border p-6">
                    <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
                      <img
                        src={capturedImage}
                        alt="Captured"
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className="mt-4 flex justify-center space-x-4">
                      <button
                        onClick={retake}
                        className="flex items-center space-x-2 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                      >
                        <RotateCcw className="h-4 w-4" />
                        <span>Retake</span>
                      </button>
                      <button
                        onClick={handleAnalyze}
                        disabled={isAnalyzing}
                        className="flex items-center space-x-2 px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
                      >
                        {isAnalyzing ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Eye className="h-4 w-4" />
                        )}
                        <span>{isAnalyzing ? 'Analyzing...' : 'Analyze'}</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Upload Tab */}
            {activeTab === 'upload' && (
              <div className="space-y-4">
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                    isDragActive
                      ? 'border-blue-400 bg-blue-50'
                      : 'border-gray-300 hover:border-gray-400'
                  }`}
                >
                  <input {...getInputProps()} />
                  <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  {uploadedImage ? (
                    <div>
                      <img
                        src={uploadedImage}
                        alt="Uploaded"
                        className="max-h-64 mx-auto rounded-lg mb-4"
                      />
                      <p className="text-sm text-gray-600 mb-4">Image uploaded successfully</p>
                      <button
                        onClick={handleAnalyze}
                        disabled={isAnalyzing}
                        className="flex items-center space-x-2 px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 mx-auto"
                      >
                        {isAnalyzing ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Eye className="h-4 w-4" />
                        )}
                        <span>{isAnalyzing ? 'Analyzing...' : 'Analyze'}</span>
                      </button>
                    </div>
                  ) : (
                    <div>
                      <p className="text-lg font-medium text-gray-900 mb-2">
                        {isDragActive ? 'Drop the image here' : 'Upload an eye image'}
                      </p>
                      <p className="text-sm text-gray-600">
                        Drag and drop an image here, or click to select
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Analysis Results */}
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-sm border p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Analysis Results</h2>
              
              {!analysisResult ? (
                <div className="text-center py-12">
                  <Eye className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600">
                    {isAnalyzing 
                      ? 'Analyzing your image...' 
                      : 'Upload or capture an image to get started'
                    }
                  </p>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Disease Detection Result */}
                  <div className="border rounded-lg p-4">
                    <div className="flex items-center space-x-3 mb-3">
                      {analysisResult.disease === 'normal' ? (
                        <CheckCircle className="h-6 w-6 text-green-600" />
                      ) : (
                        <AlertTriangle className="h-6 w-6 text-red-600" />
                      )}
                      <h3 className="text-lg font-medium text-gray-900">
                        {analysisResult.diseaseInfo?.name || analysisResult.disease}
                      </h3>
                    </div>
                    <p className="text-sm text-gray-600 mb-3">
                      {analysisResult.diseaseInfo?.description}
                    </p>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-medium text-gray-500">Confidence:</span>
                      <span className="text-sm font-medium text-gray-900">
                        {(analysisResult.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    {analysisResult.timestamp && (
                      <div className="flex items-center space-x-2 mt-2">
                        <Clock className="h-3 w-3 text-gray-400" />
                        <span className="text-xs text-gray-500">
                          {formatTimestamp(analysisResult.timestamp)}
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Disease Details */}
                  {analysisResult.diseaseInfo && (
                    <div className="space-y-4">
                      {/* Symptoms */}
                      <div>
                        <h4 className="font-medium text-gray-900 mb-2 flex items-center">
                          <AlertTriangle className="h-4 w-4 mr-2" />
                          Symptoms
                        </h4>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {analysisResult.diseaseInfo.symptoms.map((symptom, index) => (
                            <li key={index} className="flex items-center space-x-2">
                              <div className="h-1.5 w-1.5 bg-gray-400 rounded-full"></div>
                              <span>{symptom}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Treatment */}
                      <div>
                        <h4 className="font-medium text-gray-900 mb-2 flex items-center">
                          <Heart className="h-4 w-4 mr-2" />
                          Recommended Treatment
                        </h4>
                        <p className="text-sm text-gray-600">
                          {analysisResult.diseaseInfo.treatment}
                        </p>
                      </div>

                      {/* Recommendations */}
                      <div>
                        <h4 className="font-medium text-gray-900 mb-2 flex items-center">
                          <Shield className="h-4 w-4 mr-2" />
                          Recommendations
                        </h4>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {analysisResult.diseaseInfo.recommendations.map((rec, index) => (
                            <li key={index} className="flex items-center space-x-2">
                              <div className="h-1.5 w-1.5 bg-blue-400 rounded-full"></div>
                              <span>{rec}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Severity Level */}
                      <div>
                        <h4 className="font-medium text-gray-900 mb-2">Severity Level</h4>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          analysisResult.diseaseInfo.severity === 'High'
                            ? 'bg-red-100 text-red-800'
                            : analysisResult.diseaseInfo.severity === 'Moderate'
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-green-100 text-green-800'
                        }`}>
                          {analysisResult.diseaseInfo.severity}
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Action Buttons */}
                  <div className="flex space-x-3 pt-4 border-t">
                    <button className="flex-1 flex items-center justify-center space-x-2 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors">
                      <Download className="h-4 w-4" />
                      <span>Download Report</span>
                    </button>
                    <button 
                      onClick={() => navigate('/dashboard')}
                      className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                    >
                      Back to Dashboard
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DiseaseDetection; 