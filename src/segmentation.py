import cv2
import numpy as np

def segment_image(image):
    """
    Phân đoạn ảnh dùng Mean Shift và Color Thresholding (Step 2 & 3 part 1):
    1. Mean Shift Filtering để làm phẳng màu (posterization).
    2. Chuyển sang HSV để tách màu Chuối (Vàng/Xanh).
    3. Tạo mask nhị phân.
    4. Xử lý hình thái học.
    
    Args:
        image: Ảnh đầu vào đã qua Gaussian Blur (BGR).
               
    Returns:
        mask: Ảnh nhị phân (0 và 255), 255 là vùng chuối.
    """
    if image is None:
        return None

    # --- Bước 2: Phân đoạn ảnh bằng Mean Shift ---
    # sp: Bán kính không gian (Spatial Window Radius)
    # sr: Bán kính màu (Color Window Radius)
    # Mean Shift giúp làm mịn ảnh, gộp các regions cùng màu, xóa chi tiết nhiễu.
    shifted = cv2.pyrMeanShiftFiltering(image, sp=15, sr=30)
    
    # --- Bước 3 (Phần 1): Tạo mặt nạ (Binary Mask) dựa trên màu sắc ---
    # Chuyển sang không gian màu HSV
    hsv_shifted = cv2.cvtColor(shifted, cv2.COLOR_BGR2HSV)
    
    # Định nghĩa khoảng màu cho chuối (Vàng đến Xanh lá)
    # Lưu ý: Hue trong OpenCV là 0-179.
    # Chuối chín (Vàng): Hue khoảng 20-30 (tương ứng 40-60 độ)
    # Chuối xanh (Xanh lá): Hue khoảng 30-70 (tương ứng 60-140 độ)
    # Ta lấy khoảng rộng từ 10 (Cam vàng) đến 90 (Xanh cyan) để bao quát.
    
    lower_banana = np.array([15, 40, 40])  # Hue min, Sat min, Val min
    upper_banana = np.array([90, 255, 255]) # Hue max, Sat max, Val max
    
    # Tạo mask
    mask = cv2.inRange(hsv_shifted, lower_banana, upper_banana)
    
    # --- Bước 3 (Phần 2): Xử lý hình thái học ---
    # Dùng kernel để khử nhiễu
    kernel = np.ones((5, 5), np.uint8)
    
    # Morphology Open: Xóa nhiễu trắng nhỏ trên nền đen
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Morphology Close: Lấp lỗ đen nhỏ trong vùng trắng (chuối)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return mask
