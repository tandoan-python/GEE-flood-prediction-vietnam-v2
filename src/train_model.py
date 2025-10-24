import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
import os
import joblib

# --- CÁC THAM SỐ ---
# Cập nhật tên tệp CSV mới
DATA_PATH = os.path.join('..', 'data', 'flood_data_vn_v3.csv')
MODEL_SAVE_PATH = os.path.join('..', 'models', 'flood_model.keras')
PREPROCESSOR_SAVE_PATH = os.path.join('..', 'models', 'preprocessor.joblib')

# DATA_PATH = os.path.join("data", "flood_data_vn_v3.csv")
# MODEL_SAVE_PATH = os.path.join("models", "flood_model.keras")
# # Sẽ lưu 2 tệp: một cho scaler (liên tục) và một cho one-hot (danh nghĩa)
# PREPROCESSOR_SAVE_PATH = os.path.join("models", "preprocessor.joblib")

RAINFALL_LOOKBACK_DAYS = 7
# Cập nhật danh sách đặc trưng tĩnh
STATIC_CONTINUOUS_FEATURES = ["elevation", "slope", "twi", "stream_proximity"]
STATIC_CATEGORICAL_FEATURES = ["lulc"]
TARGET_VARIABLE = "flood_label"

rain_cols = [f"rain_{i}" for i in range(RAINFALL_LOOKBACK_DAYS)]

# --- 1. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU ---
print("Đang tải dữ liệu...")
if not os.path.exists(DATA_PATH):
    print(f"LỖI: Không tìm thấy tệp dữ liệu tại {DATA_PATH}")
    print(
        "Vui lòng chạy Giai đoạn 3 (prepare_data.py) và di chuyển tệp CSV vào thư mục 'data'."
    )
    exit()

df = pd.read_csv(DATA_PATH)
df.replace(-9999, np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Kích thước dữ liệu sau khi làm sạch: {df.shape}")

if df.shape[0] < 50:
    print(f"LỖI: Dữ liệu quá nhỏ ({df.shape[0]} hàng) để huấn luyện.")
    exit()

# Tách X và y
X_static_df = df[STATIC_CONTINUOUS_FEATURES + STATIC_CATEGORICAL_FEATURES]
X_ts_df = df[rain_cols]
y_series = df[TARGET_VARIABLE]

class_counts = y_series.value_counts()
print(f"Phân phối lớp:\n{class_counts}")
if class_counts.min() < 2:
    print(
        "LỖI: Lớp ít phổ biến nhất có ít hơn 2 mẫu. Không thể thực hiện 'phân tầng (stratify)'."
    )
    exit()

# Chia Train/Test TRƯỚC KHI chuẩn hóa
X_static_train, X_static_test, X_ts_train, X_ts_test, y_train, y_test = (
    train_test_split(
        X_static_df,
        X_ts_df,
        y_series,
        test_size=0.2,
        random_state=42,
        stratify=y_series,
    )
)

# --- 2. Xử lý Đặc trưng (Scaling & One-Hot Encoding) ---
print("Tạo pipeline tiền xử lý (Scaling và One-Hot Encoding)...")

# Pipeline cho các cột liên tục
continuous_transformer = Pipeline(steps=[("scaler", StandardScaler())])

# Pipeline cho các cột danh nghĩa
categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        )  # Bỏ qua nếu gặp giá trị lạ trong test set
    ]
)

# Kết hợp 2 pipeline bằng ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("cont", continuous_transformer, STATIC_CONTINUOUS_FEATURES),
        ("cat", categorical_transformer, STATIC_CATEGORICAL_FEATURES),
    ],
    remainder="passthrough",  # Giữ lại các cột không được chỉ định (nếu có)
)

# Fit preprocessor CHỈ trên X_static_train
print("Fit preprocessor trên tập huấn luyện...")
X_static_train_processed = preprocessor.fit_transform(X_static_train)
# Transform X_static_test
X_static_test_processed = preprocessor.transform(X_static_test)

# Chuyển đổi sparse matrix (từ OneHot) thành dense array
if hasattr(X_static_train_processed, "toarray"):
    X_static_train_processed = X_static_train_processed.toarray()
    X_static_test_processed = X_static_test_processed.toarray()

# Lấy số lượng đặc trưng tĩnh sau khi xử lý (để xây dựng model)
STATIC_FEATURES_COUNT_PROCESSED = X_static_train_processed.shape[1]
print(f"Số đặc trưng tĩnh sau khi xử lý: {STATIC_FEATURES_COUNT_PROCESSED}")

# Reshape dữ liệu chuỗi thời gian cho LSTM
X_ts_train_reshaped = X_ts_train.values.reshape(-1, RAINFALL_LOOKBACK_DAYS, 1)
X_ts_test_reshaped = X_ts_test.values.reshape(-1, RAINFALL_LOOKBACK_DAYS, 1)

# --- 3. XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU ---
print("Tính toán trọng số lớp...")
try:
    weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights_dict = dict(enumerate(weights))
    print(f"Trọng số lớp: {class_weights_dict}")
except ValueError as e:
    print(f"Lỗi khi tính trọng số lớp: {e}. Sử dụng trọng số mặc định.")
    class_weights_dict = None

# --- 4. XÂY DỰNG MÔ HÌNH ---
print("Xây dựng kiến trúc mô hình...")

ts_input = Input(shape=(RAINFALL_LOOKBACK_DAYS, 1), name="ts_input")
lstm_out = LSTM(32, activation="relu")(ts_input)

# Cập nhật shape cho đầu vào tĩnh
static_input = Input(shape=(STATIC_FEATURES_COUNT_PROCESSED,), name="static_input")
dense_out = Dense(16, activation="relu")(static_input)

concatenated = Concatenate()([lstm_out, dense_out])
x = Dense(16, activation="relu")(concatenated)
output = Dense(1, activation="sigmoid", name="output")(x)

model = Model(inputs=[ts_input, static_input], outputs=output)
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Recall(name="recall")],
)
model.summary()

# --- 5. HUẤN LUYỆN MÔ HÌNH ---
print("Bắt đầu huấn luyện...")
history = model.fit(
    [X_ts_train_reshaped, X_static_train_processed],
    y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weights_dict,
    verbose=1,
)

# --- 6. ĐÁNH GIÁ MÔ HÌNH ---
print("\nĐánh giá mô hình trên tập kiểm tra...")
loss, accuracy, recall = model.evaluate(
    [X_ts_test_reshaped, X_static_test_processed], y_test, verbose=0
)
print(
    f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}, Test Recall: {recall:.4f}"
)

# --- 7. LƯU MÔ HÌNH VÀ PREPROCESSOR ---
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
print(f"Lưu mô hình vào: {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)

print(f"Lưu preprocessor (scaler & one-hot) vào: {PREPROCESSOR_SAVE_PATH}")
joblib.dump(preprocessor, PREPROCESSOR_SAVE_PATH)

print("Hoàn thành Giai đoạn 4!")
