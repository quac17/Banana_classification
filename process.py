import os
import cv2
import csv
import sys
from src.preprocessing import preprocess_image
from src.segmentation import segment_image
from src.detection import detect_and_crop_objects
from src.feature_extraction import extract_features

def main():
    # Đường dẫn đến folder data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    # Các folder nhãn
    labels = ['overripe', 'ripe', 'rotten', 'unripe']
    
    # Tạo folder output
    output_dir = os.path.join(base_dir, 'output')
    feature_output_dir = os.path.join(output_dir, 'feature_output')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(feature_output_dir):
        os.makedirs(feature_output_dir)
        
    csv_path = os.path.join(feature_output_dir, 'features.csv')
    
    print(f"Outputting features to: {csv_path}")
    
    # Mở file CSV để ghi
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # Ghi header
        # Cột: Tên file (crop), Nhãn, Mean Hue, Mean Saturation, Mean Value, Tỉ lệ đốm nâu, Mật độ cạnh
        header = ['Filename', 'Label', 'Mean_Hue', 'Mean_Saturation', 'Mean_Value', 'Brown_Spot_Ratio', 'Edge_Density']
        writer.writerow(header)
    
        for label in labels:
            folder_path = os.path.join(data_dir, label)
            if not os.path.exists(folder_path):
                print(f"Warning: Folder {folder_path} does not exist.")
                continue
                
            print(f"Processing folder: {label}")
            
            # Duyệt qua các file ảnh trong folder
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    file_path = os.path.join(folder_path, filename)
                    print(f"  Processing file: {filename}")
                    
                    # 1. Đọc ảnh
                    original_image = cv2.imread(file_path)
                    if original_image is None:
                        print(f"    Error reading file: {filename}")
                        continue
                    
                    # 2. Tiền xử lý
                    preprocessed = preprocess_image(original_image)
                    
                    # 3. Phân đoạn
                    mask = segment_image(preprocessed)
                    
                    # 4. Tách đối tượng
                    cropped_objects = detect_and_crop_objects(original_image, mask)
                    
                    print(f"    Found {len(cropped_objects)} object(s).")
                    
                    # Loop xử lý từng quả chuối
                    for i, (banana_img, banana_mask) in enumerate(cropped_objects):
                        
                        # 5. Trích xuất đặc trưng
                        features = extract_features(banana_img, banana_mask)
                        
                        # Tạo tên file cho ảnh crop (đơn giản hơn)
                        save_name = f"{label}_{os.path.splitext(filename)[0]}_crop_{i}.jpg"
                        save_path = os.path.join(output_dir, save_name)
                        
                        # Lưu ảnh crop
                        cv2.imwrite(save_path, banana_img)
                        
                        # Ghi vào CSV
                        row = [
                            save_name,
                            label,
                            f"{features['mean_hue']:.2f}",
                            f"{features['mean_saturation']:.2f}",
                            f"{features['mean_value']:.2f}",
                            f"{features['brown_spot_ratio']:.4f}",
                            f"{features['edge_density']:.4f}"
                        ]
                        writer.writerow(row)

if __name__ == "__main__":
    main()
