import streamlit as st
import requests
import ee
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import json
import math

# Tắt cảnh báo SSL
import requests.packages.urllib3
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# --- Cấu hình API ---
API_URL = "http://127.0.0.1:8000/predict"

# --- Khởi tạo GEE & Geocoder (Sử dụng Cache) ---
@st.cache_resource
def get_gee_authenticator():
    try:
        ee.Initialize(project='gee-project-tandoan')
        print("GEE đã khởi tạo thành công.")
        return True
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo Google Earth Engine: {e}")
        st.warning("V..." , icon="⚠️") # Giữ ngắn gọn
        return None

@st.cache_resource
def get_geocoder():
    try:
        return Nominatim(user_agent="flood_risk_app")
    except Exception as e:
        st.error(f"Không thể khởi tạo dịch vụ Geocoding. Lỗi: {e}")
        return None

# --- Các hàm gọi API ---

def get_location_data(geolocator, location_name):
    try:
        location = geolocator.geocode(location_name, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        st.error(f"Lỗi dịch vụ Geocoding: {e}. Vui lòng thử lại.")
        return None, None

def get_terrain_data(lat, lon):
    """
    Tính toán tất cả 5 đặc trưng địa hình. 
    TWI sẽ được tính toán thủ công từ SRTM và MERIT.
    """
    try:
        point = ee.Geometry.Point(lon, lat)
        
        # --- 1. Lấy DEM và Slope (từ SRTM 30m) ---
        dem_image = ee.Image("USGS/SRTMGL1_003")
        
        # 1a. Lấy Độ cao (Elevation)
        elevation = dem_image.select('elevation').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=30
        ).get('elevation')

        # 1b. Lấy Độ dốc (Slope)
        slope_deg = ee.Terrain.slope(dem_image) # Độ dốc (degrees)
        
        slope_value = slope_deg.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=30
        ).get('slope')

        # --- 2. Lấy UPA (Vùng tích lũy) (từ MERIT 90m) ---
        # MERIT/Hydro/v1_0_1 là công cộng và có band 'upa'
        upa_image = ee.Image("MERIT/Hydro/v1_0_1").select('upa')

        # --- 3. Tính toán TWI (Kết hợp cả hai) ---
        
        # Resample UPA (90m) về 30m để khớp với Slope
        upa_resampled = upa_image.resample('bilinear').reproject(
            crs=dem_image.projection(), 
            scale=30
        )
        
        # Chuyển Slope từ độ (degrees) sang radians
        slope_rad = slope_deg.multiply(math.pi / 180.0)
        
        # Tính TWI: log( (UPA + 1) / (tan(slope_rad) + 0.001) )
        # Thêm hằng số nhỏ để tránh lỗi chia cho 0 hoặc log(0)
        twi_image = ee.Image().expression(
            'log((upa + 1) / (tan(slope_rad) + 0.001))', {
              'upa': upa_resampled,
              'slope_rad': slope_rad
            }
        ).rename('twi')
        
        # Lấy giá trị TWI tại điểm
        twi = twi_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=30
        ).get('twi')

        # --- 4. LULC (Sử dụng đất) (từ ESA 10m) ---
        lulc = ee.ImageCollection("ESA/WorldCover/v100").first().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10
        ).get('Map')

        # --- 5. Stream Proximity (Khoảng cách đến sông) (từ JRC) ---
        water_occurrence = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')
        water_mask = water_occurrence.gt(50)
        distance_to_water = water_mask.fastDistanceTransform().sqrt()
        
        stream_proximity = distance_to_water.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=30
        ).get('distance')

        # --- Tập hợp kết quả ---
        terrain_data = {
            'elevation': elevation.getInfo(),
            'slope': slope_value.getInfo(),
            'twi': twi.getInfo(),
            'lulc': lulc.getInfo(),
            'stream_proximity': stream_proximity.getInfo()
        }
        
        # Làm sạch dữ liệu (GEE có thể trả về None)
        for key, value in terrain_data.items():
            if value is None:
                st.error(f"Không thể lấy dữ liệu cho '{key}'. Sử dụng giá trị 0.")
                terrain_data[key] = 0
            
        return terrain_data

    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu GEE: {e}")
        return None

def get_rainfall_data(lat, lon):
    # (Hàm này không đổi, vẫn giữ nguyên)
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "past_days": 7,
            "hourly": "precipitation",
            "timezone": "auto"
        }
        response = requests.get(url, params=params, verify=False, timeout=15)
        response.raise_for_status()
        data = response.json()
        hourly_rain = data['hourly']['precipitation']
        daily_rain = []
        for i in range(7):
            day_total = sum(hourly_rain[i*24 : (i+1)*24])
            daily_rain.append(day_total)
        return daily_rain
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi khi gọi API thời tiết: {e}")
        return None
    except KeyError:
        st.error("API thời tiết trả về dữ liệu không mong muốn.")
        return None

# --- Giao diện Streamlit (Không đổi) ---
st.set_page_config(page_title="Hệ thống Cảnh báo Lũ lụt", layout="wide")
st.title("Hệ thống Cảnh báo Lũ lụt Việt Nam")
st.markdown("Nhập một địa điểm tại Việt Nam để dự báo nguy cơ ngập lụt dựa trên dữ liệu địa hình và lượng mưa 7 ngày qua.")

gee_ready = get_gee_authenticator()
geolocator = get_geocoder()

if not gee_ready: st.stop()
if not geolocator: st.stop()

location_name = st.text_input(
    "Nhập tên địa điểm (ví dụ: 'Huế', 'Hội An', 'Quận 1, TPHCM'):", 
    "Thành phố Huế"
)

if st.button("Lấy Dữ liệu & Dự báo"):
    if not location_name:
        st.warning("Vui lòng nhập tên địa điểm.")
    else:
        with st.spinner(f"Đang tìm kiếm tọa độ cho '{location_name}'..."):
            lat, lon = get_location_data(geolocator, location_name)
        
        if lat is None or lon is None:
            st.error(f"Không tìm thấy địa điểm '{location_name}'. Vui lòng thử lại.")
        else:
            st.success(f"Đã tìm thấy '{location_name}': (Vĩ độ: {lat:.4f}, Kinh độ: {lon:.4f})")
            
            col1, col2 = st.columns(2)
            
            with col1:
                with st.spinner("Đang lấy dữ liệu địa hình từ Google Earth Engine... (Việc này có thể mất một chút thời gian do phải tính TWI)"):
                    terrain_data = get_terrain_data(lat, lon)
            
            with col2:
                with st.spinner("Đang lấy dữ liệu lượng mưa 7 ngày qua..."):
                    rainfall_sequence = get_rainfall_data(lat, lon)
            
            if terrain_data and rainfall_sequence:
                with col1:
                    st.subheader("Dữ liệu Địa hình (Tĩnh)")
                    st.json(terrain_data)
                
                with col2:
                    st.subheader("Dữ liệu Lượng mưa 7 ngày (mm)")
                    st.json({"rainfall_sequence_daily": rainfall_sequence})

                with st.spinner("Đang gửi dữ liệu đến mô hình AI để dự báo..."):
                    payload = {
                        "elevation": terrain_data['elevation'],
                        "slope": terrain_data['slope'],
                        "twi": terrain_data['twi'],
        "lulc": terrain_data['lulc'],
                        "stream_proximity": terrain_data['stream_proximity'],
                        "rainfall_sequence": rainfall_sequence
                    }
                    
                    try:
                        api_response = requests.post(API_URL, json=payload, timeout=30)
                        api_response.raise_for_status()
                        result = api_response.json()
                        probability = result.get('flood_probability', 0)
                        
                        st.subheader("Kết quả Dự báo Nguy cơ Ngập lụt")
                        prob_percent = probability * 100
                        
                        if prob_percent > 75: st.error(f"Rất Cao: {prob_percent:.2f}%")
                        elif prob_percent > 50: st.warning(f"Cao: {prob_percent:.2f}%")
                        elif prob_percent > 25: st.info(f"Trung bình: {prob_percent:.2f}%")
                        else: st.success(f"Thấp: {prob_percent:.2f}%")
                        st.progress(probability)

                    except requests.exceptions.HTTPError as http_err:
                        st.error(f"Lỗi HTTP từ API: {http_err.response.status_code}")
                        try: st.json(http_err.response.json())
                        except json.JSONDecodeError: st.error(http_err.response.text)
                    except requests.exceptions.RequestException as req_err:
                        st.error(f"Không thể kết nối đến API dự báo. Vui lòng đảm bảo API đang chạy. Lỗi: {req_err}")