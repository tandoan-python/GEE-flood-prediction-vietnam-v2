from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import os
from typing import List

# --- Cấu hình Đường dẫn ---
MODEL_SAVE_PATH = os.path.join('..', 'models', 'flood_model.keras')
PREPROCESSOR_SAVE_PATH = os.path.join('..', 'models', 'preprocessor.joblib')

# *** SỬA LỖI TẠI ĐÂY ***
# Thứ tự này PHẢI KHỚP 100% với thứ tự trong train_model.py
STATIC_COL_ORDER = ['elevation', 'slope', 'twi', 'stream_proximity', 'lulc']

# --- Khởi tạo FastAPI & Tải Mô hình ---
app = FastAPI(title="Flood Prediction API", version="1.0")

model = None
preprocessor = None
assets_loaded = False

@app.on_event("startup")
def load_model_assets():
    """Tải mô hình và preprocessor khi API khởi động."""
    global model, preprocessor, assets_loaded
    
    try:
        # Tải mô hình TensorFlow/Keras
        # Chúng ta cần import keras ở đây để nó hoạt động trong môi trường server
        from tensorflow.keras.models import load_model
        
        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"Lỗi: Không tìm thấy tệp mô hình tại {MODEL_SAVE_PATH}")
            return
        if not os.path.exists(PREPROCESSOR_SAVE_PATH):
            print(f"Lỗi: Không tìm thấy tệp preprocessor tại {PREPROCESSOR_SAVE_PATH}")
            return
            
        model = load_model(MODEL_SAVE_PATH)
        preprocessor = joblib.load(PREPROCESSOR_SAVE_PATH)
        assets_loaded = True
        print("--- Mô hình và Preprocessor đã được tải thành công ---")
        model.summary() # In cấu trúc mô hình để xác nhận
        
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi tải mô hình: {e}")
        # assets_loaded vẫn là False

# --- Định nghĩa Dữ liệu Đầu vào (Pydantic) ---
class FloodInput(BaseModel):
    elevation: float
    slope: float
    twi: float
    lulc: int # LULC là mã (int)
    stream_proximity: float
    rainfall_sequence: List[float] = Field(..., min_items=7, max_items=7)

# --- Endpoint Dự báo ---
@app.post("/predict")
async def predict(data: FloodInput):
    """
    Nhận 6 đặc trưng đầu vào và trả về xác suất ngập lụt.
    """
    global assets_loaded, model, preprocessor
    
    if not assets_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Mô hình hoặc preprocessor chưa sẵn sàng. Vui lòng kiểm tra logs của API."
        )

    try:
        # 1. Chuẩn bị Dữ liệu Tĩnh (Static)
        static_data = {
            'elevation': data.elevation,
            'slope': data.slope,
            'twi': data.twi,
            'stream_proximity': data.stream_proximity, # Sửa thứ tự
            'lulc': data.lulc,                         # Sửa thứ tự
        }
        
        # Tạo DataFrame với thứ tự cột CHÍNH XÁC
        static_df = pd.DataFrame([static_data], columns=STATIC_COL_ORDER)
        
        # Áp dụng preprocessor (scale + OHE)
        static_processed = preprocessor.transform(static_df)
        
        # 2. Chuẩn bị Dữ liệu Chuỗi thời gian (Time-Series)
        # Đảm bảo shape là (1, 7, 1)
        ts_processed = np.array(data.rainfall_sequence).reshape(1, 7, 1)
        
        # 3. Thực hiện Dự báo
        # Mô hình của chúng ta nhận dict làm đầu vào
        prediction_input = {
            'ts_input': ts_processed,
            'static_input': static_processed
        }
        
        probability = model.predict(prediction_input)
        
        # Kết quả trả về là một array, ví dụ [[0.123]]
        result_prob = float(probability[0][0])

        return {"flood_probability": result_prob}

    except Exception as e:
        print(f"Lỗi trong quá trình dự báo: {e}")
        # In chi tiết lỗi ra console của API
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự báo: {e}")