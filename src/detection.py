import cv2
import numpy as np

def detect_and_crop_objects(original_image, mask):
    """
    Tách từng quả chuối (Object Detection & Cropping):
    1. Tìm contours trên mask.
    2. Lấy bounding box cho mỗi contour.
    3. Crop ảnh gốc theo bounding box.
    
    Args:
        original_image: Ảnh gốc (BGR) để crop (không phải ảnh HSV hay blurred).
        mask: Ảnh nhị phân (đen trắng) từ bước segmentation.
        
    Returns:
        cropped_images: List các ảnh (numpy array) đã được crop chứa từng quả chuối.
    """
    if mask is None or original_image is None:
        return []

    # 1. Tìm contours
    # cv2.RETR_EXTERNAL: Chỉ lấy đường bao ngoài cùng (bỏ qua lỗ bên trong)
    # cv2.CHAIN_APPROX_SIMPLE: Nén đường bao (chỉ lưu các điểm gấp khúc)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cropped_images = []
    
    for contour in contours:
        # Lọc nhiễu: Bỏ qua các contour quá nhỏ
        area = cv2.contourArea(contour)
        if area < 1000: # Ngưỡng diện tích tùy chỉnh, ví dụ 1000 pixel
            continue
            
        # 2. Bounding Box
        x, y, w, h = cv2.boundingRect(contour)
        
        # 3. Crop
        # Cắt vùng ảnh từ ảnh gốc
        roi = original_image[y:y+h, x:x+w]
        # Cắt vùng mask tương ứng
        roi_mask = mask[y:y+h, x:x+w]
        
        if roi.size > 0:
            cropped_images.append((roi, roi_mask))
            
    return cropped_images
