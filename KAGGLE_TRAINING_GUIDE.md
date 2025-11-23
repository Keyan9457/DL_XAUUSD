# Kaggle Training Guide for XAUUSD Model

## 🎯 Why Kaggle?

**Advantages over Colab:**
- ✅ **30GB RAM** (vs Colab's 12GB) - Can handle full dataset!
- ✅ **Free GPU** (P100 or T4)
- ✅ **9-hour sessions** (vs Colab's 12 hours)
- ✅ **Persistent datasets** - Upload once, use forever
- ✅ **Better for large datasets**

---

## 📋 Step-by-Step Guide

### Step 1: Create Kaggle Account
1. Go to https://www.kaggle.com
2. Sign up (free)
3. Verify your phone number (required for GPU access)

### Step 2: Upload Your Historical Data

**Option A: Upload as Dataset (Recommended)**
1. Go to https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Click **"Upload"**
4. Select your `XAUUSD_HISTORICAL_DATA` folder
5. Settings:
   - **Title:** `XAUUSD Historical Data 2004-2025`
   - **Subtitle:** `Gold price data across multiple timeframes`
   - **Visibility:** Private (or Public if you want to share)
6. Click **"Create"**
7. **Wait for upload to complete** (10-30 minutes for 1GB+ data)

**Option B: Upload Directly to Notebook**
1. Create notebook first (Step 3)
2. Click **"Add Data"** → **"Upload"**
3. Upload files directly

### Step 3: Create New Notebook

1. Go to https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Settings (right sidebar):
   - **Accelerator:** GPU T4 x2 (or P100)
   - **Internet:** ON
   - **Language:** Python
4. Click **"File"** → **"Upload Notebook"**
5. Upload `train_model_kaggle.ipynb` from your project

### Step 4: Link Your Dataset

1. In the notebook, click **"Add Data"** (right sidebar)
2. Search for your dataset: `xauusd-historical-data`
3. Click **"Add"**
4. The data will be available at `/kaggle/input/xauusd-historical-data/`

### Step 5: Run the Notebook

**Run cells in order:**

1. **Cell 1:** Check GPU
   - Should show: `GPU Available: [PhysicalDevice(name='/physical_device:GPU:0'...)]`

2. **Cell 2:** Install packages
   - Wait for "✓ Packages installed"

3. **Cell 3:** Verify data
   - Update `DATA_DIR` if your dataset name is different
   - Should list your CSV files

4. **Cell 4:** Load processing functions
   - Should show "✓ Data processing functions loaded"

5. **Cell 5:** Process data
   - Takes 5-10 minutes
   - Should process M5, M15, H1, H4

6. **Cell 6:** Load training functions
   - Should show "✓ Training functions loaded"

7. **Cell 7:** Train model
   - **This takes 2-4 hours with GPU**
   - Watch for:
     - Epoch progress
     - Validation loss decreasing
     - Model checkpoints saving

8. **Cell 8:** Evaluate model
   - Shows final accuracy metrics

9. **Cell 9:** Download files
   - Click "Output" tab on right
   - Download all 3 files

### Step 6: Download Trained Model

1. Click **"Output"** tab (right sidebar)
2. Download these files:
   - `best_xauusd_model.keras`
   - `scaler.pkl`
   - `target_scaler.pkl`
3. Copy to `C:\Users\Admin\Documents\DL_XAUUSD\`

---

## 🔧 Troubleshooting

### "Out of Memory" Error
- Reduce sampling percentage in Cell 7:
  ```python
  # Change from 0.2 to 0.1 (10% historical data)
  sample_size = int(len(df_old) * 0.1)
  ```

### "Dataset not found"
- Check dataset name in Cell 3
- Update `DATA_DIR` to match your dataset name

### "GPU not available"
- Go to Settings (right sidebar)
- Change Accelerator to "GPU T4 x2"
- Click "Save"
- Restart notebook

### Session Timeout
- Kaggle gives 9-hour sessions
- Training should complete in 2-4 hours
- If timeout, model checkpoints are saved
- Can resume from last checkpoint

---

## 💡 Tips for Kaggle

1. **Enable GPU:** Always check GPU is enabled before training
2. **Save Often:** Kaggle auto-saves, but manually save important cells
3. **Monitor Progress:** Watch epoch progress to estimate completion time
4. **Use Comments:** Add notes to cells for future reference
5. **Version Control:** Kaggle saves notebook versions automatically

---

## 📊 Expected Results

**With Kaggle GPU:**
- **Training Time:** 2-4 hours (vs 6-10 hours on CPU)
- **Memory Usage:** ~20-25 GB (Kaggle has 30 GB)
- **Final Accuracy:** 60-65% directional
- **MAE:** ~0.0095 (similar to local training)

---

## 🚀 Next Steps After Training

1. Download model files
2. Copy to local project:
   ```
   C:\Users\Admin\Documents\DL_XAUUSD\
   ├── best_xauusd_model.keras
   ├── scaler.pkl
   └── target_scaler.pkl
   ```
3. Run evaluation locally:
   ```bash
   python evaluate_model.py
   ```
4. Start live trading:
   ```bash
   python main.py
   ```

---

## 📁 Files to Upload to Kaggle

**Required:**
1. `train_model_kaggle.ipynb` - The notebook
2. `XAUUSD_HISTORICAL_DATA/` - Your data folder (as dataset)

**Not needed (already in notebook):**
- model_training.py
- process_historical_data.py
- evaluate_model.py

---

## ⚡ Quick Start Checklist

- [ ] Create Kaggle account
- [ ] Verify phone number
- [ ] Upload historical data as dataset
- [ ] Create new notebook
- [ ] Upload `train_model_kaggle.ipynb`
- [ ] Link dataset to notebook
- [ ] Enable GPU in settings
- [ ] Run all cells
- [ ] Wait for training (2-4 hours)
- [ ] Download model files
- [ ] Copy to local project

---

**Ready to start? Upload `train_model_kaggle.ipynb` to Kaggle now!** 🚀
