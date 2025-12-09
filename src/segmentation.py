import cv2
import numpy as np

def segment_image(image):
    """
    Phân đoạn ảnh và tách nền dùng K-Means Clustering:
    1. Flatten ảnh.
    2. K-Means (K=2) để phân cụm Nền vs Chuối.
    3. Tạo mask nhị phân.
    4. Xử lý hình thái học (Erosion, Dilation).
    
    Args:
        image: Ảnh đầu vào (nên là ảnh đã được tiền xử lý hoặc ảnh gốc, 
               nhưng thường K-Means chạy trên RGB hoặc LAB/HSV đều được. 
               Tuy nhiên để hiển thị mask đúng, ta thường dùng logic màu).
               Ở đây ta nhận vào ảnh 3 kênh (HSV hoặc RGB/BGR).
               
    Returns:
        mask: Ảnh nhị phân (0 và 255), 255 là vùng chuối.
    """
    if image is None:
        return None

    # Reshape ảnh thành mảng các điểm ảnh (pixel) 2D: (height * width, 3)
    pixel_values = image.reshape((-1, 3))
    # Chuyển sang float32 cho K-Means
    pixel_values = np.float32(pixel_values)

    # Tiêu chí dừng (criteria) cho K-Means
    # cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER: Dừng khi đạt max iter hoặc độ thay đổi nhỏ hơn epsilon
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Số cụm K = 2 (Nền và Chuối)
    k = 2
    
    # Áp dụng K-Means
    # labels: nhãn của từng pixel (0 hoặc 1)
    # centers: tâm của các cụm
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Chuyển centers về uint8
    centers = np.uint8(centers)
    
    # Reshape labels về kích thước ảnh gốc để tạo mask
    labels = labels.flatten()
    # Tạo mask tạm thời
    # Lưu ý: Ta chưa biết cụm nào là chuối, cụm nào là nền.
    # Giả định chuối thường có màu sáng hơn (hoặc vàng/xanh), nền tối/trắng/khác biệt.
    # Tuy nhiên, cách đơn giản là dựa vào vị trí hoặc giả định nền chiếm diện tích lớn hơn?
    # Ở đây, ta sẽ thử logic: Chuối thường nằm ở giữa, hoặc ta coi class ít pixel hơn là chuối nến background lớn?
    # Hoặc đơn giản trả về label map rồi xử lý tiếp.
    
    # Nhưng theo yêu cầu, ta cần tạo mask binary.
    # Ta sẽ tạo ảnh phân đoạn trước
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    
    # Để xác định đâu là nền, ta có thể giả sử 4 góc ảnh là nền.
    # Lấy nhãn của góc (0,0)
    corner_label = labels[0] # Giả sử góc trái trên là nền
    
    # Tạo mask: Vùng KHÔNG PHẢI corner_label sẽ là 255 (Chuối), vùng corner_label là 0 (Nền)
    mask = np.where(labels.reshape(image.shape[:2]) == corner_label, 0, 255).astype('uint8')
    
    # 4. Xử lý hình thái học (Morphological Operations)
    # Erosion để loại bỏ nhiễu nhỏ
    # Dilation để lấp đầy các lỗ nhỏ trong quả chuối
    kernel = np.ones((5, 5), np.uint8)
    
    # Open: Erosion -> Dilation (khử nhiễu bên ngoài)
    # Close: Dilation -> Erosion (lấp lỗ bên trong)
    # Theo note.txt: "Erosion... và Dilation... để xóa các lỗ nhỏ bên trong hoặc viền răng cưa"
    
    # Dùng MorphologyEx Open để khử nhiễu liti nền
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Dùng MorphologyEx Close để lấp lỗ trong chuối
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask
