import ee
import os
import google.auth

# --- CẤU HÌNH ---
try:
    credentials, project_id = google.auth.default()
    ee.Initialize(credentials=credentials, project='gee-project-tandoan')
    print("Đã xác thực GEE thành công với dự án: gee-project-tandoan")
except Exception as e:
    print(f"Lỗi khi xác thực GEE: {e}")
    
    # # Nếu chạy trên colab
    # ee.Authenticate()
    # ee.Initialize(project='gee-project-tandoan')

# --- THAM SỐ TẠO DỮ LIỆU TỰ ĐỘNG (Phương án 1) ---

# 1. Định nghĩa Vùng nghiên cứu (Đã chọn Thừa Thiên Huế)
region_hue = ee.FeatureCollection("FAO/GAUL/2015/level1") \
    .filter(ee.Filter.And(
        ee.Filter.eq('ADM0_NAME', 'Viet Nam'),
        ee.Filter.eq('ADM1_NAME', 'Thua Thien - Hue')
    )).geometry()

# 2. Định nghĩa Sự kiện Lũ (Lũ lịch sử T10/2020)
FLOOD_EVENT_DATE = '2020-10-12' # Ngày T-0 (đỉnh lũ)
BEFORE_START_DATE = '2020-08-01'
BEFORE_END_DATE = '2020-09-30'
FLOOD_START_DATE = '2020-10-08'
FLOOD_END_DATE = '2020-10-16'

# 3. Tham số Kỹ thuật
SAMPLE_SIZE_PER_CLASS = 1500 # 1500 điểm ngập, 1500 điểm không ngập
SAR_THRESHOLD_DB = -16      # Ngưỡng (dB) cho mặt nước trong ảnh SAR (VV)
SPECKLE_FILTER_RADIUS = 50  # Bán kính (mét) để lọc nhiễu ảnh SAR
RAINFALL_LOOKBACK_DAYS = 7  # 7 ngày (T-6, T-5, ..., T-0)
EXPORT_FILE_NAME = 'flood_data_vn_v3'

# --- NGUỒN DỮ LIỆU GEE ---

# 1. Dữ liệu địa hình (Static)
dem = ee.Image("USGS/SRTMGL1_003").clip(region_hue)
slope = ee.Terrain.slope(dem).rename('slope')

# 2. Đặc trưng thủy văn Topographic Wetness Index (TWI) hay Chỉ số Độ Ẩm Địa Hình
flow_accumulation = ee.Image("WWF/HydroSHEDS/15ACC").clip(region_hue)
twi = dem.expression(
    'log( (flow_acc + 1) * pixel_area / tan(slope_rad) )', {
        'flow_acc': flow_accumulation,
        'slope_rad': slope.multiply(ee.Number(3.14159).divide(180)),
        'pixel_area': ee.Image.pixelArea()
    }
).rename('twi')

# 3. Đặc trưng thủy văn (Stream Proximity) hay Khoảng cách đến Sông/Suối
rivers = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(region_hue)
stream_proximity = rivers.distance(searchRadius=50000).clip(region_hue).rename('stream_proximity')

# 4. Dữ liệu sử dụng đất (Static)
lulc = ee.ImageCollection("ESA/WorldCover/v100").first().clip(region_hue).select('Map').rename('lulc')

# 5. Dữ liệu mưa (Dynamic)
rainfall = ee.ImageCollection("NASA/GPM_L3/IMERG_V06").filterDate('2010-01-01', '2024-01-01') \
    .select('precipitationCal')

# Gộp các đặc trưng tĩnh
static_features_image = dem.rename('elevation') \
    .addBands(slope) \
    .addBands(twi) \
    .addBands(stream_proximity) \
    .addBands(lulc)

# --- GIAI ĐOẠN 3.A: TẠO DỮ LIỆU NHÃN TỪ SENTINEL-1 ---
print("Bắt đầu Giai đoạn 3.A: Tạo Dữ liệu Nhãn từ Sentinel-1...")

# Tải bộ sưu tập Sentinel-1
s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
    .filter(ee.Filter.eq('instrumentMode', 'IW')) \
    .filterBounds(region_hue) \
    .select('VV')

# Lọc nhiễu ảnh SAR (Speckle Filtering)
def filter_speckle(img):
    return img.focal_mean(SPECKLE_FILTER_RADIUS, 'circle', 'meters', 1)

# Tạo ảnh "Trước Lũ" và "Trong Lũ"
before_img = s1.filterDate(BEFORE_START_DATE, BEFORE_END_DATE).mean().clip(region_hue)
flood_img = s1.filterDate(FLOOD_START_DATE, FLOOD_END_DATE).mean().clip(region_hue)

before_img_smooth = filter_speckle(before_img)
flood_img_smooth = filter_speckle(flood_img)

# Xác định vùng nước
water_before = before_img_smooth.lt(SAR_THRESHOLD_DB)
water_during = flood_img_smooth.lt(SAR_THRESHOLD_DB)

# Vùng ngập mới (Trước là đất, sau là nước)
new_flood = water_during.And(water_before.Not())
new_flood_map = new_flood.updateMask(new_flood.eq(1)) # Chỉ giữ lại pixel ngập

# Vùng nước vĩnh viền (Trước là nước, sau là nước)
permanent_water = water_before.And(water_during)

# Vùng đất khô (Không bao gồm ngập và nước vĩnh viễn)
# Cũng loại bỏ sườn dốc > 20 độ (dễ bị nhiễu radar và ít bị ngập)
dry_land = new_flood.Not().And(permanent_water.Not()).And(slope.lt(20))
dry_land_map = dry_land.updateMask(dry_land.eq(1))

print(f"Đã xác định bản đồ ngập lụt cho sự kiện {FLOOD_EVENT_DATE}.")

# Lấy mẫu
print(f"Bắt đầu lấy mẫu {SAMPLE_SIZE_PER_CLASS} điểm ngập...")
flood_points = new_flood_map.sample(
    region=region_hue,
    scale=30,
    numPixels=SAMPLE_SIZE_PER_CLASS,
    seed=123,
    geometries=True
).map(lambda f: f.set('flood_label', 1, 'event_date', FLOOD_EVENT_DATE))

print(f"Bắt đầu lấy mẫu {SAMPLE_SIZE_PER_CLASS} điểm không ngập...")
non_flood_points = dry_land_map.sample(
    region=region_hue,
    scale=30,
    numPixels=SAMPLE_SIZE_PER_CLASS,
    seed=456,
    geometries=True
).map(lambda f: f.set('flood_label', 0, 'event_date', FLOOD_EVENT_DATE))

# Gộp hai bộ dữ liệu
points = flood_points.merge(non_flood_points)
print(f"Đã tạo thành công {SAMPLE_SIZE_PER_CLASS * 2} điểm dữ liệu có nhãn.")

# --- GIAI ĐOẠN 3.B: TRÍCH XUẤT ĐẶC TRƯNG (NHƯ TRƯỚC) ---

# Gợi ý 1: Cập nhật hàm Rainfall Windowing
def extract_rainfall(feature):
    event_date = ee.Date(feature.get('event_date'))
    start_date = event_date.advance(-RAINFALL_LOOKBACK_DAYS + 1, 'day') # T-(N-1)

    rainfall_series = rainfall.filterDate(start_date, event_date.advance(1, 'day'))

    def get_daily_rain(day_offset_from_start):
        n = ee.Number(day_offset_from_start)
        current_date = start_date.advance(n, 'day')
        daily_total = rainfall_series.filterDate(current_date, current_date.advance(1, 'day')).sum()

        prop_name = ee.String('rain_').cat(n.int().format()) # rain_0, rain_1, ..., rain_6

        value = daily_total.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=feature.geometry(),
            scale=10000
        ).get('precipitationCal')

        # Return a function that takes a feature and sets the property
        return lambda f: f.set(prop_name, ee.Algorithms.If(value, value, -9999))

    day_indices = ee.List.sequence(0, RAINFALL_LOOKBACK_DAYS - 1)

    def accumulate(day_index, intermediate_feature):
      rain_setter_fn = get_daily_rain(ee.Number(day_index))
      return rain_setter_fn(ee.Feature(intermediate_feature))

    feature_with_rain = ee.Feature(day_indices.iterate(
        accumulate,
        feature
    ))
    return feature_with_rain


# Trích xuất đặc trưng tĩnh tại điểm
def extract_static(feature):
    static_values = static_features_image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=feature.geometry(),
        scale=30
    )
    static_values = static_values.map(lambda k, v: ee.Algorithms.If(v, v, -9999))
    return feature.set(static_values)

# --- CHẠY QUY TRÌNH ---
print("Bắt đầu trích xuất đặc trưng tĩnh...")
data_with_static = points.map(extract_static)

print("Bắt đầu trích xuất chuỗi thời gian mưa (Rainfall Windowing)...")
training_data = data_with_static.map(extract_rainfall)

# --- GIAI ĐOẠN 3.C: XUẤT DỮ LIỆU ---
# Các thuộc tính cần xuất
rain_prop_names = ee.List([ee.String('rain_').cat(ee.Number(i).int().format()) for i in range(RAINFALL_LOOKBACK_DAYS)])
static_prop_names = ee.List(['elevation', 'slope', 'twi', 'stream_proximity', 'lulc'])
label_prop_name = ee.List(['flood_label'])
all_properties = static_prop_names.cat(rain_prop_names).cat(label_prop_name).cat(ee.List(['event_date']))

print(f"Bắt đầu tác vụ xuất '{EXPORT_FILE_NAME}'...")
print("Vui lòng kiểm tra tab 'Tasks' trong GEE Code Editor để 'Run'.")

task = ee.batch.Export.table.toDrive(
    collection=training_data,
    description=EXPORT_FILE_NAME,
    folder='GEE_Exports',
    fileNamePrefix=EXPORT_FILE_NAME,
    fileFormat='CSV',
    selectors=all_properties.getInfo() # Convert ee.List to Python list
)
task.start()