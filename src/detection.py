import cv2
import numpy as np

def detect_and_crop_objects(original_image, mask):
    """
    Tách từng quả chuối (Object Detection & Cropping):
    1. Tìm contours trên mask.
    2. Lọc bỏ các contour không hợp lệ (nhiễu, đá, lá cây).
    3. Crop ảnh gốc theo bounding box.
    
    Args:
        original_image: Ảnh gốc (BGR) để crop.
        mask: Ảnh nhị phân (đen trắng) từ bước segmentation.
        
    Returns:
        cropped_images: List các tuple (roi, roi_mask) đã được crop chứa từng quả chuối.
    """
    if mask is None or original_image is None:
        return []

    # 1. Tìm contours
    # cv2.RETR_EXTERNAL: Chỉ lấy đường bao ngoài cùng
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cropped_images = []
    
    for contour in contours:
        # --- Lọc nhiễu và vật thể lạ ---
        
        # 1. Diện tích (Area)
        area = cv2.contourArea(contour)
        if area < 2000: # Tăng ngưỡng để loại bỏ lá vụn/nhiễu
            continue
            
        # 2. Bounding Box & Tỉ lệ khung hình (Aspect Ratio)
        x, y, w, h = cv2.boundingRect(contour)
        
        aspect_ratio = float(w) / h if w > h else float(h) / w
        if aspect_ratio < 1.2: 
            # Loại bỏ vật thể quá vuông/tròn (như cục đá, đốm tròn)
            # Chuối thường có hình dáng thuôn dài
            continue
        
        # 3. Crop để kiểm tra màu sắc
        roi = original_image[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]
        
        if roi.size == 0:
            continue

        # 4. Kiểm tra độ bão hòa màu (Saturation) - Loại bỏ đá (màu xám)
        # Chuyển sang HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Tính mean saturation CHỈ trên vùng mask (bỏ qua nền đen nếu có)
        # Ở đây roi là hình chữ nhật, roi_mask xác định chính xác pixel
        mean_val = cv2.mean(hsv_roi, mask=roi_mask)
        mean_saturation = mean_val[1] # H, S, V -> Index 1 là Saturation

        if mean_saturation < 20:
            # Saturation quá thấp nghĩa là màu xám/trắng/đen -> Cục đá
            continue

        # 5. Kiểm tra "Lá cây khô/Nâu xám" (Dry Leaf Filter)
        # Case specific: ripe_ripe5_crop_1.jpg có Saturation ~66 (rất thấp so với chuối > 100)
        # Chuối thường có màu tươi (Saturation cao). Lá khô màu nâu xám xỉn.
        if mean_saturation < 90:
            # Ngưỡng 90 an toàn vì chuối unripe thấp nhất cũng khoảng 120
            continue

        # Nếu vượt qua tất cả các bài test
        cropped_images.append((roi, roi_mask))
            
    return cropped_images
