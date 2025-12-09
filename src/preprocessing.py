import cv2

def preprocess_image(image):
    """
    Tiền xử lý ảnh:
    1. Chuyển đổi sang không gian màu HSV.
    2. Áp dụng Gaussian Blur để lọc nhiễu.
    """
    if image is None:
        return None
    
    # 1. Chuyển sang không gian màu HSV
    # HSV (Hue, Saturation, Value) tốt hơn RGB trong việc tách nền dựa trên màu sắc
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 2. Lọc nhiễu (Gaussian Blur)
    # Kích thước kernel (5, 5) để làm mịn ảnh mà không mất quá nhiều chi tiết
    blurred_image = cv2.GaussianBlur(hsv_image, (5, 5), 0)
    
    return blurred_image
