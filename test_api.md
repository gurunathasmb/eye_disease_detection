# API Testing Guide

## Test the Backend API

### 1. Health Check
```bash
curl http://localhost:5000/api/health
```
Expected: `{"status":"OK","message":"Eye Disease Detection API is running"}`

### 2. User Registration
```bash
curl -X POST http://localhost:5000/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"name":"Test User","email":"test@example.com","password":"password123"}'
```

### 3. User Login
```bash
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'
```

### 4. Get History (with token from login)
```bash
curl http://localhost:5000/api/history \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## Frontend Testing

1. Open http://localhost:3000
2. Register/Login
3. Go to Detection page
4. Upload an image or use camera
5. Check results display
6. Go back to Dashboard to see history

## Expected Results

### Analysis Results Should Include:
- Disease name (cataract, glaucoma, diabetic_retinopathy, or normal)
- Confidence percentage
- Disease description
- Symptoms list
- Treatment recommendations
- Severity level
- Timestamp

### Dashboard Should Show:
- Total scans count
- Normal vs disease results
- Average confidence
- Analysis history table
- Real-time statistics

## Troubleshooting

If analysis doesn't work:
1. Check browser console for errors
2. Check backend console for errors
3. Verify both servers are running
4. Check network tab for API calls 