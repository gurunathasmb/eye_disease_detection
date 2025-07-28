import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { toast } from 'react-toastify';
import axios from 'axios';
import { 
  Camera, 
  Upload, 
  LogOut, 
  Eye, 
  Clock, 
  TrendingUp,
  Activity,
  Calendar,
  AlertTriangle,
  CheckCircle,
  Heart,
  Shield
} from 'lucide-react';

const Dashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalScans: 0,
    normalResults: 0,
    diseaseDetected: 0,
    averageConfidence: 0
  });

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('http://localhost:5000/api/history', {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      
      setHistory(response.data);
      
      // Calculate stats
      const totalScans = response.data.length;
      const normalResults = response.data.filter(item => item.disease === 'normal').length;
      const diseaseDetected = totalScans - normalResults;
      const averageConfidence = totalScans > 0 
        ? response.data.reduce((sum, item) => sum + item.confidence, 0) / totalScans 
        : 0;

      setStats({
        totalScans,
        normalResults,
        diseaseDetected,
        averageConfidence: (averageConfidence * 100).toFixed(1)
      });
      
    } catch (error) {
      console.error('Error fetching history:', error);
      toast.error('Failed to load analysis history');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const formatDate = (timestamp) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getDiseaseIcon = (disease) => {
    switch (disease) {
      case 'normal':
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      case 'cataract':
      case 'glaucoma':
      case 'diabetic_retinopathy':
        return <AlertTriangle className="h-5 w-5 text-red-600" />;
      default:
        return <Eye className="h-5 w-5 text-blue-600" />;
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'High':
        return 'bg-red-100 text-red-800';
      case 'Moderate':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-green-100 text-green-800';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
              <p className="text-sm text-gray-600">Welcome back, {user?.name}</p>
            </div>
            <button
              onClick={handleLogout}
              className="flex items-center space-x-2 px-4 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <LogOut className="h-4 w-4" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Quick Actions */}
        <div className="mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <button
              onClick={() => navigate('/detection')}
              className="flex items-center space-x-3 p-4 bg-white rounded-lg shadow-sm border hover:shadow-md transition-shadow"
            >
              <div className="p-2 bg-blue-100 rounded-lg">
                <Camera className="h-6 w-6 text-blue-600" />
              </div>
              <div className="text-left">
                <h3 className="font-medium text-gray-900">Capture Photo</h3>
                <p className="text-sm text-gray-600">Use camera to scan your eye</p>
              </div>
            </button>
            <button
              onClick={() => navigate('/detection')}
              className="flex items-center space-x-3 p-4 bg-white rounded-lg shadow-sm border hover:shadow-md transition-shadow"
            >
              <div className="p-2 bg-green-100 rounded-lg">
                <Upload className="h-6 w-6 text-green-600" />
              </div>
              <div className="text-left">
                <h3 className="font-medium text-gray-900">Upload Image</h3>
                <p className="text-sm text-gray-600">Upload an eye image for analysis</p>
              </div>
            </button>
          </div>
        </div>

        {/* Statistics */}
        <div className="mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Your Statistics</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-white p-4 rounded-lg shadow-sm border">
              <div className="flex items-center">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <Activity className="h-5 w-5 text-blue-600" />
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-600">Total Scans</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.totalScans}</p>
                </div>
              </div>
            </div>
            <div className="bg-white p-4 rounded-lg shadow-sm border">
              <div className="flex items-center">
                <div className="p-2 bg-green-100 rounded-lg">
                  <CheckCircle className="h-5 w-5 text-green-600" />
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-600">Normal Results</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.normalResults}</p>
                </div>
              </div>
            </div>
            <div className="bg-white p-4 rounded-lg shadow-sm border">
              <div className="flex items-center">
                <div className="p-2 bg-red-100 rounded-lg">
                  <AlertTriangle className="h-5 w-5 text-red-600" />
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-600">Disease Detected</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.diseaseDetected}</p>
                </div>
              </div>
            </div>
            <div className="bg-white p-4 rounded-lg shadow-sm border">
              <div className="flex items-center">
                <div className="p-2 bg-purple-100 rounded-lg">
                  <TrendingUp className="h-5 w-5 text-purple-600" />
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-600">Avg Confidence</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.averageConfidence}%</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Analysis History */}
        <div>
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Recent Analysis History</h2>
          {loading ? (
            <div className="bg-white rounded-lg shadow-sm border p-8 text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
              <p className="mt-2 text-gray-600">Loading history...</p>
            </div>
          ) : history.length === 0 ? (
            <div className="bg-white rounded-lg shadow-sm border p-8 text-center">
              <Eye className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No analysis history yet</h3>
              <p className="text-gray-600 mb-4">Start by capturing or uploading an eye image for analysis</p>
              <button
                onClick={() => navigate('/detection')}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Start Analysis
              </button>
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Date
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Disease
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Confidence
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Severity
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Details
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {history.map((item) => (
                      <tr key={item.id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <Clock className="h-4 w-4 text-gray-400 mr-2" />
                            <span className="text-sm text-gray-900">
                              {formatDate(item.timestamp)}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            {getDiseaseIcon(item.disease)}
                            <span className="ml-2 text-sm font-medium text-gray-900">
                              {item.diseaseInfo?.name || item.disease}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="text-sm text-gray-900">
                            {(item.confidence * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getSeverityColor(item.diseaseInfo?.severity)}`}>
                            {item.diseaseInfo?.severity || 'Unknown'}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <div className="flex items-center space-x-2">
                            {item.diseaseInfo?.symptoms && (
                              <span className="flex items-center">
                                <AlertTriangle className="h-3 w-3 mr-1" />
                                {item.diseaseInfo.symptoms.length} symptoms
                              </span>
                            )}
                            {item.diseaseInfo?.recommendations && (
                              <span className="flex items-center">
                                <Shield className="h-3 w-3 mr-1" />
                                {item.diseaseInfo.recommendations.length} recommendations
                              </span>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 