import cv2

def preprocess_image(image):
    """
    Tiền xử lý ảnh:
    1. Làm mờ Gaussian (Gaussian Blur) để lọc nhiễu.
    
    Lưu ý: Không chuyển sang HSV ở đây vì bước Mean Shift (Segmentation) 
           thường hoạt động trên không gian màu RGB/BGR hoặc LAB.
           Việc chuyển đổi màu sẽ thực hiện khi cần thiết trong bước Segmentation.
    """
    if image is None:
        return None
    
    # 1. Lọc nhiễu (Gaussian Blur)
    # Kích thước kernel (5, 5) để làm mịn ảnh, loại bỏ nhiễu liti
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    return blurred_image
