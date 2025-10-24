# Dự án Dự báo Ngập lụt End-to-End (GEE & LSTM)

Sử dụng Google Earth Engine (GEE) và mô hình Deep Learning (LSTM)

## Bước 1: Cài đặt (Windows & VS Code)

### 1.1. Cài đặt Python

* Python 3.10 hoặc 3.11
* **Quan trọng**: Khi cài đặt, hãy đánh dấu vào ô "Add Python to PATH".*

### 1.2. Cài đặt Google Cloud SDK (Công cụ gcloud)

*Đây là bước bắt buộc để xác thực GEE từ local.*

* Truy cập Google Cloud SDK installer và tải về bản cài đặt cho Windows.

* Chạy trình cài đặt. Sau khi hoàn tất, mở một cửa sổ Command Prompt (cmd) hoặc PowerShell.

* Chạy lệnh sau để khởi tạo SDK:

    `gcloud init`


* Làm theo hướng dẫn trên màn hình:

* Đăng nhập vào tài khoản Google của bạn (tài khoản đã đăng ký GEE).

* Chọn dự án Google Cloud của bạn (ví dụ: `gee-project-tandoan`) khi được hỏi.

### 1.3. Xác thực Earth Engine

Sau khi `gcloud` đã được cài đặt và đăng nhập, hãy chạy lệnh sau trong **terminal** để xác thực thư viện Python ee:

`earthengine authenticate`

Lệnh này sẽ mở trình duyệt, yêu cầu bạn cấp quyền và cung cấp cho bạn một mã xác thực để dán trở lại vào terminal.

### 1.4. Cài đặt Thư viện Python

Tạo tệp `requirements.txt` trong thư mục gốc dự án của bạn với nội dung bên dưới và chạy lệnh pip install -r `requirements.txt`.

Lưu ý: Lỗi `google-cloud-cli` trước đó là do tên gói không chính xác. Công cụ dòng lệnh được cài đặt qua SDK (bước 1.2), không phải qua pip.

```
earthengine-api
google-api-python-client
google-auth-httplib2
pandas
numpy
scikit-learn
tensorflow
fastapi
uvicorn[standard]
streamlit
requests
joblib
```

Lệnh cài đặt:

`pip install -r requirements.txt`


## Bước 2: Tạo Cấu trúc Thư mục

Mở VS Code trong thư mục dự án của bạn (ví dụ: Flood_Prediction_VN) và tạo cấu trúc sau:
```
Flood_Prediction_VN/
├── app/
│   └── dashboard.py      # Giao diện Streamlit
├── data/
│   └── (Trống - sẽ chứa file CSV sau Giai đoạn 3)
├── models/
│   └── (Trống - sẽ chứa model và scaler sau Giai đoạn 4)
├── src/
│   ├── __init__.py       # Tệp trống
│   ├── api.py            # API Backend
│   ├── prepare_data.py   # Script xử lý GEE
│   └── train_model.py    # Script huấn luyện model
└── requirements.txt        # Các thư viện cần thiết
```

## Bước 3: Thu thập và Tiền xử lý Dữ liệu (GEE)

Cách thực hiện:

* Chạy script này từ terminal trong VS Code:

    `python src/prepare_data.py`

* Script sẽ khởi tạo xác thực GEE.

* Một tác vụ (Task) có tên `export_flood_data_vn` sẽ được tạo trong tab Tasks của GEE Code Editor (trên web).

* Truy cập GEE Code Editor, qua tab Tasks và nhấn Run để bắt đầu xuất tệp.

* Tệp `flood_data_vn.csv` sẽ được lưu vào Google Drive của bạn.

* **Quan trọng**: Tải tệp `flood_data_vn.csv` từ Google Drive về và đặt nó vào thư mục `data/` của dự án.

(Xem mã trong tệp src/prepare_data.py)

## Bước 4: Huấn luyện Mô hình

Cách thực hiện:

* Đảm bảo bạn đã có tệp `data/flood_data_vn.csv` từ giai đoạn 3.

* Chạy script huấn luyện:

* python `src/train_model.py`

Script thực hiện:

* Tải dữ liệu CSV.
* Tiền xử lý và chia dữ liệu.
* Xây dựng mô hình LSTM kết hợp.
* Huấn luyện mô hình.
* Lưu tệp `flood_model.keras` và `scaler.joblib` vào thư mục `models/`.

## Bước 5: Xây dựng API Backend (FastAPI)

*Mở Terminal 1: (backend) trong môi trường ảo*
Cách thực hiện:
* Kích hoạt môi trường ảo: `.venv\Scripts\Activate`
* Chuyển đến thư mục `app/`
* Chạy: `powershell uvicorn main:app --reload`

## Bước 6: Xây dựng Giao diện (Streamlit)

*Mở Terminal 1: (backend) trong môi trường ảo*
Cách thực hiện:
* Kích hoạt môi trường ảo: `.venv\Scripts\Activate`
* Chuyển đến thư mục `app/`
* Chạy: `powershell streamlit run dashboard.py`

