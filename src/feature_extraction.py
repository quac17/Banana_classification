import cv2
import numpy as np

def extract_features(image, mask):
    """
    Trích xuất đặc trưng từ ảnh quả chuối đã crop.
    
    Args:
        image: Ảnh quả chuối (BGR).
        mask: Mask nhị phân tương ứng (255 là chuối, 0 là nền).
        
    Returns:
        features: Dictionary chứa các đặc trưng:
            - mean_hue: Giá trị Hue trung bình.
            - mean_saturation: Giá trị Saturation trung bình.
            - mean_value: Giá trị Value trung bình.
            - brown_spot_ratio: Tỉ lệ diện tích đốm nâu/đen.
            - edge_density: Mật độ cạnh (Canny).
    """
    if image is None or mask is None:
        return None
    
    feature_data = {}
    
    # --- 1. Đặc trưng màu sắc (Color Features) ---
    # Chuyển sang HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Tính mean color CHỈ trong vùng mask (vùng quả chuối)
    # cv2.mean trả về (mean_ch1, mean_ch2, mean_ch3, 0)
    mean_val = cv2.mean(hsv, mask=mask)
    feature_data['mean_hue'] = mean_val[0]
    feature_data['mean_saturation'] = mean_val[1]
    feature_data['mean_value'] = mean_val[2]
    
    # --- 2. Đặc trưng kết cấu (Texture/Spot Detection) ---
    
    # a. Tỉ lệ đốm nâu/đen
    # Định nghĩa màu nâu/đen trong HSV
    # Chuối chín/xanh thường có Value cao (sáng). Chuối hỏng có Value thấp (tối).
    # Hoặc Hue màu nâu (Orange-ish) và Value thấp.
    
    # Ở đây dùng ngưỡng đơn giản: Value < 80 được coi là tối/đốm đen (trong dải 0-255)
    # Lưu ý: Cần kết hợp với mask để không tính nền đen bên ngoài mask
    
    v_channel = hsv[:, :, 2]
    # Pixel thuộc mask VÀ có Value < 80
    dark_spots = cv2.bitwise_and(mask, cv2.inRange(v_channel, 0, 80))
    
    banana_area = cv2.countNonZero(mask)
    if banana_area > 0:
        spot_area = cv2.countNonZero(dark_spots)
        feature_data['brown_spot_ratio'] = spot_area / banana_area
    else:
        feature_data['brown_spot_ratio'] = 0.0

    # b. Canny Edge Detection
    # Phát hiện cạnh trong vùng chuối
    # Cần làm mờ nhẹ trước khi Canny để tránh nhiễu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Canny threshold thấp/cao tùy chỉnh
    edges = cv2.Canny(blurred_gray, 50, 150)
    
    # Chỉ tính cạnh nằm TRONG mask
    edges_in_banana = cv2.bitwise_and(edges, edges, mask=mask)
    
    if banana_area > 0:
        edge_pixels = cv2.countNonZero(edges_in_banana)
        feature_data['edge_density'] = edge_pixels / banana_area
    else:
        feature_data['edge_density'] = 0.0
        
    return feature_data
