# 🚀 Deployment Guide - Plant Disease Detection System

This guide explains how to deploy the Plant Disease Detection system with **frontend on Vercel** and **backend on Render** to optimize resource usage and avoid platform limits.

---

## 📋 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT ARCHITECTURE                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐              ┌─────────────────────────┐  │
│  │   Vercel     │   HTTP/S     │      Render.com         │  │
│  │  (Frontend)  │◄────────────►│   (Backend API)         │  │
│  │              │   API Calls   │                         │  │
│  │  React +     │              │  FastAPI + TensorFlow   │  │
│  │  Vite Build  │              │  + Model (19MB)         │  │
│  └──────────────┘              └─────────────────────────┘  │
│        │                                    │                │
│        │ Static Assets                     │ Dynamic API     │
│        │ (HTML/CSS/JS)                     │ /predict        │
│        │                                    │ /retrain        │
│        └────────────────────────────────────┘                │
│                     User Browser                             │
└─────────────────────────────────────────────────────────────┘
```

**Why This Approach?**
- ✅ Vercel: Optimized for static frontends (React/Vite), free tier, global CDN
- ✅ Render: Better suited for Python backends with ML models
- ✅ Separation prevents exceeding Render's free tier size limits (512MB)
- ✅ Frontend deploys independently from backend (faster iterations)

---

## 🎯 Part 1: Deploy Backend to Render

### Step 1: Prepare Render Deployment

The backend is already configured for Render deployment. The current setup includes:
- `Dockerfile` for containerization
- `main.py` with FastAPI backend
- Model files in `models/` directory (19MB)

### Step 2: Create Render Web Service

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository: `Elvis-Kayonga/PlantDisease-Analysis`
4. Configure the service:
   ```
   Name: plant-disease-api
   Region: Choose closest to your users
   Branch: main
   Root Directory: (leave empty - uses repo root)
   Environment: Docker
   Plan: Free (or paid for better performance)
   ```

5. **Important Environment Variables** (optional but recommended):
   ```
   TF_CPP_MIN_LOG_LEVEL=2
   PYTHON_VERSION=3.10
   ```

6. Click **"Create Web Service"**

### Step 3: Wait for Build & Get URL

- Render will build your Docker container (may take 5-10 minutes first time)
- Once deployed, you'll get a URL like: `https://plant-disease-api-xxxx.onrender.com`
- **Copy this URL** - you'll need it for frontend configuration

### Step 4: Test Backend API

Test your deployed API:
```bash
curl https://your-render-url.onrender.com/health
# Should return: {"status":"healthy","model_loaded":true,"api_version":"1.0.0"}

curl https://your-render-url.onrender.com/model-status
# Should return model information
```

---

## 🌐 Part 2: Deploy Frontend to Vercel

### Step 1: Update Frontend Configuration

1. **Update the production environment file** (already created):
   ```bash
   # frontend/.env.production
   VITE_API_BASE_URL=https://your-actual-render-url.onrender.com
   ```

   ⚠️ **IMPORTANT**: Replace with your actual Render URL from Part 1, Step 3

### Step 2: Create Vercel Project

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click **"Add New..."** → **"Project"**
3. Import your GitHub repository: `Elvis-Kayonga/PlantDisease-Analysis`
4. Configure the project:
   ```
   Framework Preset: Other
   Root Directory: ./frontend
   Build Command: npm run build
   Output Directory: dist
   Install Command: npm install
   ```

5. **Environment Variables**: Add this in Vercel:
   ```
   VITE_API_BASE_URL = https://your-actual-render-url.onrender.com
   ```
   (Same URL from Render deployment)

6. Click **"Deploy"**

### Step 3: Verify Deployment

- Vercel will build and deploy (usually takes 1-2 minutes)
- You'll get a URL like: `https://plant-disease-analysis.vercel.app`
- Visit the URL and test:
  - ✅ Page loads (React UI visible)
  - ✅ Check sidebar shows "Model Loaded: Yes" (confirms API connection)
  - ✅ Try uploading an image for prediction

### Step 4: Fix CORS if Needed

If you see CORS errors in browser console:

1. **Verify your backend** allows the Vercel domain:
   ```python
   # main.py (already configured but verify)
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # This allows all origins
       # Or specify: ["https://your-app.vercel.app"]
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. If you want to restrict to specific domain, update and redeploy:
   ```python
   allow_origins=[
       "https://your-app.vercel.app",
       "http://localhost:5173"  # for local dev
   ]
   ```

---

## 🔧 Part 3: Local Development Setup

### Frontend Only (connects to production backend)
```bash
cd frontend
npm install
npm run dev
# Opens on http://localhost:5173
# Uses VITE_API_BASE_URL from .env.example or .env.local
```

### Full Stack Local (backend + frontend)
```bash
# Terminal 1: Backend
python main.py
# Runs on http://localhost:8000

# Terminal 2: Frontend
cd frontend
npm install
npm run dev
# Runs on http://localhost:5173
```

**For local development**, create `frontend/.env.local`:
```bash
VITE_API_BASE_URL=http://localhost:8000
```

---

## 🐛 Troubleshooting

### Issue 1: "Model not loaded" in Frontend
**Cause**: Backend hasn't loaded model yet or model files missing
**Solution**:
- Check Render logs for backend errors
- Ensure `models/` directory contains `plant_disease_model.h5` (19MB)
- Wait 30 seconds after deployment for model to load

### Issue 2: CORS Errors
**Cause**: Backend rejecting frontend domain
**Solution**:
- Verify `VITE_API_BASE_URL` is correct in Vercel environment variables
- Check backend CORS settings in `main.py` line 52-58
- Use `allow_origins=["*"]` for testing, then restrict for production

### Issue 3: "Network Error" or "Failed to Fetch"
**Cause**: Wrong API URL or backend not responding
**Solution**:
- Test backend directly: `curl https://your-render-url.onrender.com/health`
- Verify `VITE_API_BASE_URL` matches your Render URL exactly
- Check Render service status (may be sleeping on free tier)

### Issue 4: Render Build Failed
**Cause**: Docker build issues or missing dependencies
**Solution**:
- Check Render build logs
- Ensure `requirements.txt` has all dependencies
- Try redeploying from Render dashboard

### Issue 5: Frontend Build Failed on Vercel
**Cause**: Missing dependencies or build errors
**Solution**:
- Check Vercel build logs
- Ensure `frontend/package.json` is correct
- Try building locally: `cd frontend && npm run build`

### Issue 6: Render Free Tier Sleeping
**Cause**: Render free tier sleeps after 15 min inactivity
**Solution**:
- First request after sleep takes 30-60 seconds (cold start)
- Consider upgrading to paid tier for always-on service
- Or implement a keep-alive ping service

---

## 📊 Deployment Checklist

### ✅ Backend (Render) Ready When:
- [ ] Docker build succeeds on Render
- [ ] Health check returns `{"status":"healthy"}`
- [ ] Model status shows `"loaded":true`
- [ ] `/predict` endpoint works (test with curl)

### ✅ Frontend (Vercel) Ready When:
- [ ] Build completes successfully
- [ ] Page loads without errors
- [ ] Sidebar shows "System ONLINE"
- [ ] Sidebar shows "Model Loaded: Yes"
- [ ] Can upload image and get prediction

---

## 🎨 Frontend Configuration Files

```
frontend/
├── .env.example          # Template (for local dev)
├── .env.local           # Local development (git ignored)
├── .env.production      # Production (Vercel uses this)
├── package.json
├── vite.config.js
└── src/
    └── api.js          # Uses VITE_API_BASE_URL
```

**Priority Order** (Vite):
1. `.env.production` (when running `npm run build`)
2. `.env.local` (local dev)
3. `.env.example` (fallback)

---

## 🔄 Updating Deployments

### Update Frontend Only
```bash
# Make changes to frontend code
git add frontend/
git commit -m "Update frontend UI"
git push
# Vercel auto-deploys on push to main branch
```

### Update Backend Only
```bash
# Make changes to backend code
git add main.py src/
git commit -m "Update API endpoints"
git push
# Render auto-deploys on push to main branch
```

### Update Both
```bash
git add .
git commit -m "Update both frontend and backend"
git push
# Both Vercel and Render will deploy independently
```

---

## 💰 Cost Considerations

### Current Setup (Free Tiers):
- **Vercel Free Tier**:
  - ✅ 100GB bandwidth/month
  - ✅ Unlimited deployments
  - ✅ Global CDN
  - ✅ Perfect for React frontends

- **Render Free Tier**:
  - ✅ 512MB RAM
  - ✅ Sleeps after 15 min inactivity
  - ⚠️ 19MB model + dependencies ≈ 400MB (within limit)
  - ⚠️ Cold starts (30-60s first request)

### If Hitting Limits:
- **Render**: Upgrade to $7/month for always-on
- **Vercel**: Pro at $20/month (rarely needed for this project)

---

## 🔐 Security Best Practices

1. **Environment Variables**:
   - Never commit `.env.production` with real secrets
   - Use Vercel dashboard for sensitive env vars

2. **CORS Configuration**:
   - Production: Restrict to your Vercel domain
   - Development: Use wildcard `*` or localhost

3. **API Keys** (if you add them later):
   - Store in Render environment variables
   - Never hardcode in source

---

## 📞 Support & Resources

- **Vercel Docs**: https://vercel.com/docs
- **Render Docs**: https://render.com/docs
- **Vite Env Docs**: https://vitejs.dev/guide/env-and-mode.html
- **FastAPI Deployment**: https://fastapi.tiangolo.com/deployment/

---

## 🎯 Quick Deploy Commands

```bash
# Initial setup (one-time)
git clone <repo>
cd PlantDisease-Analysis

# Update production URL in frontend
echo "VITE_API_BASE_URL=https://your-render-url.onrender.com" > frontend/.env.production

# Commit and push
git add .
git commit -m "Configure production deployment"
git push

# Deploy happens automatically via:
# - Render: watches main branch
# - Vercel: watches main branch
```

---

**Status**: ✅ Ready for deployment!

Your frontend and backend are now properly separated for optimal deployment on Vercel and Render respectively.
