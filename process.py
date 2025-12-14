import os
import cv2
import csv
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import preprocessing
from src.preprocessing import preprocess_image
from src.segmentation import segment_image
from src.detection import detect_and_crop_objects
from src.feature_extraction import extract_features

def load_data(csv_path):
    features = []
    labels = []
    filenames = []
    if not os.path.exists(csv_path):
        return np.array(features), np.array(labels), np.array(filenames)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Based on header: ['Filename', 'Label', 'Mean_Hue', 'Mean_Saturation', 'Mean_Value', 'Brown_Spot_Ratio', 'Edge_Density']
                feat = [
                    float(row['Mean_Hue']), 
                    float(row['Mean_Saturation']), 
                    float(row['Mean_Value']), 
                    float(row['Brown_Spot_Ratio']), 
                    float(row['Edge_Density'])
                ]
                features.append(feat)
                labels.append(row['Label'])
                filenames.append(row['Filename'])
            except ValueError:
                continue # Skip bad rows
    return np.array(features), np.array(labels), np.array(filenames)

def generate_report(clf, X, y, filenames, output_dir, prefix, le):
    """
    Generate classification report and accuracy csv for a given dataset (train or test).
    """
    if len(X) == 0:
        print(f"No data for {prefix} set.")
        return

    print(f"\n--- Evaluating {prefix.capitalize()} Set ---")
    
    # Predict
    y_pred = clf.predict(X)
    
    # Calculate Metrics
    y_encoded = le.transform(y) # y is raw labels, need encoding for accuracy_score if y_pred is encoded? 
    # DecisionTree predict returns encoded labels if trained on encoded.
    # We need to make sure y matches format.
    # In main, we'll encode y before passing or handle it here.
    # Let's assume standard sklearn pattern: fit(X, y_enc) -> predict returns y_enc.
    
    # Actually, let's look at how I trained. I fit (X_train, y_encoded).
    # So predict returns encoded.
    
    # Decode for CSV
    y_pred_decoded = le.inverse_transform(y_pred)
    
    # Calculate Accuracy
    # y is string labels. y_pred is integer. Must encode y or decode y_pred.
    # Better to compare strings to avoid confusion, or encode y.
    # Let's encode y locally.
    y_true_encoded = le.transform(y)
    acc = accuracy_score(y_true_encoded, y_pred)
    
    print(f"Accuracy: {acc:.2f}")
    print(classification_report(y_true_encoded, y_pred, target_names=le.classes_))
    
    # Paths
    results_csv_path = os.path.join(output_dir, f'{prefix}_classification_results.csv')
    accuracy_csv_path = os.path.join(output_dir, f'{prefix}_accuracy_report.csv')
    
    # Detailed Result CSV
    # Helper to clean filename
    def extract_original_name(crop_name, label):
        prefix_str = f"{label}_"
        if crop_name.startswith(prefix_str):
            name_without_prefix = crop_name[len(prefix_str):]
        else:
            name_without_prefix = crop_name
        idx = name_without_prefix.rfind('_crop_')
        if idx != -1:
            return name_without_prefix[:idx]
        return name_without_prefix

    total_correct = 0
    total_incorrect = 0
    unique_files_all = set()

    with open(results_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename_Crop', 'Original_Filename', 'True_Label', 'Predicted_Label', 'Status'])
        
        for i in range(len(filenames)):
            crop_name = filenames[i]
            true_lbl = y[i]
            pred_lbl = y_pred_decoded[i]
            
            is_correct = (true_lbl == pred_lbl)
            status = "Correct" if is_correct else "Incorrect"
            
            if is_correct:
                total_correct += 1
            else:
                total_incorrect += 1
            
            original_name = extract_original_name(crop_name, true_lbl)
            unique_files_all.add(original_name)
            
            writer.writerow([crop_name, original_name, true_lbl, pred_lbl, status])
            
    # Accuracy Report CSV
    print(f"Saving {prefix} accuracy report to: {accuracy_csv_path}")
    with open(accuracy_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total_Crops', len(X)])
        writer.writerow(['Total_Unique_Files', len(unique_files_all)])
        writer.writerow(['Total_Correct_Predictions', total_correct])
        writer.writerow(['Total_Incorrect_Predictions', total_incorrect])
        writer.writerow(['Accuracy', f"{acc:.4f}"])

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def run_knn_pipeline(X_train, y_train, files_train, X_test, y_test, files_test, base_output_dir, le):
    print("\n------------------------------------------------")
    print("RUNNING KNN CLASSIFICATION")
    print("------------------------------------------------")
    
    # Create KNN output directory
    knn_output_dir = os.path.join(base_output_dir, 'KNN_output')
    if not os.path.exists(knn_output_dir):
        os.makedirs(knn_output_dir)
        
    print(f"KNN Results will be saved to: {knn_output_dir}")

    # Scale Data (Important for KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Use the same scaler for test data
    if len(X_test) > 0:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = []

    # Encode labels
    # le passed from main pipeline logic or create new
    # reusing le ensures consistency
    y_train_encoded = le.transform(y_train)

    # Initialize and Train KNN
    # k=5 is standard starter, user note suggests k=10
    k = 10
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train_scaled, y_train_encoded)
    
    # Generate Reports
    # We reuse generate_report but point to knn_output_dir
    # Note: X passed to generate_report must be scaled!
    generate_report(clf, X_train_scaled, y_train, files_train, knn_output_dir, 'train_knn', le)
    
    if len(X_test) > 0:
        generate_report(clf, X_test_scaled, y_test, files_test, knn_output_dir, 'test_knn', le)

from sklearn.svm import SVC
from plot import plot_svm_boundary

def run_svm_pipeline(X_train, y_train, files_train, X_test, y_test, files_test, base_output_dir, le):
    print("\n------------------------------------------------")
    print("RUNNING SVM CLASSIFICATION")
    print("------------------------------------------------")
    
    # Create SVM output directory
    svm_output_dir = os.path.join(base_output_dir, 'SVM_output')
    if not os.path.exists(svm_output_dir):
        os.makedirs(svm_output_dir)
        
    print(f"SVM Results will be saved to: {svm_output_dir}")

    # Scale Data (Important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Use the same scaler for test data
    if len(X_test) > 0:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = []

    # Encode labels
    y_train_encoded = le.transform(y_train)

    # Initialize and Train SVM (RBF Kernel)
    clf = SVC(kernel='rbf', random_state=42)
    clf.fit(X_train_scaled, y_train_encoded)
    
    # Generate Reports
    generate_report(clf, X_train_scaled, y_train, files_train, svm_output_dir, 'train_svm', le)
    
    if len(X_test) > 0:
        generate_report(clf, X_test_scaled, y_test, files_test, svm_output_dir, 'test_svm', le)
        
    # --- Generate Decision Boundary Plot (PCA 2D) ---
    # We pass the class constructor (SVC) so the plotter can re-train on 2D data
    # We pass the scaled training data
    plot_path = os.path.join(svm_output_dir, 'svm_decision_boundary.png')
    try:
        plot_svm_boundary(SVC(kernel='rbf', random_state=42), X_train_scaled, y_train_encoded, plot_path, le)
    except Exception as e:
        print(f"Failed toplot SVM Decision Boundary: {e}")

def run_training_pipeline(X_train, y_train, files_train, X_test, y_test, files_test, output_dir):
    print("\n--- Training Decision Tree ---")
    if len(X_train) == 0:
        print("No training data available.")
        return

    # Encode labels based on ALL possible labels in train (and hopefully test)
    # Ideally should fit encoder on all labels or known fixed list.
    # Let's perform a fit on y_train. If y_test has new labels, it will error (which is expected/correct behavior).
    le = preprocessing.LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # --- Decision Tree ---
    clf_dt = DecisionTreeClassifier(random_state=42)
    clf_dt.fit(X_train, y_train_encoded)
    
    # Generate Train Report
    generate_report(clf_dt, X_train, y_train, files_train, output_dir, 'train', le)
    
    # Generate Test Report
    if len(X_test) > 0:
        generate_report(clf_dt, X_test, y_test, files_test, output_dir, 'test', le)
    else:
        print("No test data found. Skipping test evaluation.")
        
    # --- KNN ---
    # Pass le to ensure same label encoding
    run_knn_pipeline(X_train, y_train, files_train, X_test, y_test, files_test, os.path.dirname(output_dir), le)

    # --- SVM ---
    run_svm_pipeline(X_train, y_train, files_train, X_test, y_test, files_test, os.path.dirname(output_dir), le)

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
        
    train_csv_path = os.path.join(feature_output_dir, 'train_features.csv')
    test_csv_path = os.path.join(feature_output_dir, 'test_features.csv')
    
    print(f"Features will be saved to: \n - {train_csv_path} \n - {test_csv_path}")
    
    # Initialize separate writers
    # We will open files in append mode inside loop or keep them open?
    # Better to open both contexts.
    
    with open(train_csv_path, mode='w', newline='', encoding='utf-8') as f_train, \
         open(test_csv_path, mode='w', newline='', encoding='utf-8') as f_test:
        
        train_writer = csv.writer(f_train)
        test_writer = csv.writer(f_test)
        
        header = ['Filename', 'Label', 'Mean_Hue', 'Mean_Saturation', 'Mean_Value', 'Brown_Spot_Ratio', 'Edge_Density', 'Dataset_Type']
        train_writer.writerow(header)
        test_writer.writerow(header)

        # Scan data folder to find train_data and test_data
        data_subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        for sub_dir in data_subdirs:
            input_sub_path = os.path.join(data_dir, sub_dir)
            
            # Determine processing mode and writer
            is_train = False
            if 'train' in sub_dir.lower():
                output_sub_name = 'train_result'
                current_writer = train_writer
                is_train = True
            elif 'test' in sub_dir.lower():
                output_sub_name = 'test_result'
                current_writer = test_writer
            else:
                # Default fallback, maybe treat as test or ignore?
                # User said "update data data trong data/test_data", implies structure is clear.
                output_sub_name = sub_dir + '_result'
                current_writer = test_writer # Default to test if unsure? Or skip? Let's default to test writer for safety or create a new one.
                # Given the user request, let's assume strict naming or just map to test if not train.
                if 'train' not in sub_dir.lower():
                     # Assume anything not train is test-like
                     pass
            
            output_sub_path = os.path.join(output_dir, output_sub_name)
            if not os.path.exists(output_sub_path):
                os.makedirs(output_sub_path)
            
            print(f"Scanning Dataset Group: {sub_dir} -> Output: {output_sub_name}")

            for label in labels:
                folder_path = os.path.join(input_sub_path, label)
                if not os.path.exists(folder_path):
                    continue
                    
                print(f"  Processing folder: {label}")
                
                if os.path.isdir(folder_path):
                    for filename in os.listdir(folder_path):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            file_path = os.path.join(folder_path, filename)
                            # print(f"    Processing file: {filename}")
                            
                            original_image = cv2.imread(file_path)
                            if original_image is None:
                                continue
                            
                            preprocessed = preprocess_image(original_image)
                            mask = segment_image(preprocessed)
                            cropped_objects = detect_and_crop_objects(original_image, mask)
                            
                            for i, (banana_img, banana_mask) in enumerate(cropped_objects):
                                features = extract_features(banana_img, banana_mask)
                                save_name = f"{label}_{os.path.splitext(filename)[0]}_crop_{i}.jpg"
                                save_path = os.path.join(output_sub_path, save_name)
                                cv2.imwrite(save_path, banana_img)
                                
                                row = [
                                    save_name,
                                    label,
                                    f"{features['mean_hue']:.2f}",
                                    f"{features['mean_saturation']:.2f}",
                                    f"{features['mean_value']:.2f}",
                                    f"{features['brown_spot_ratio']:.4f}",
                                    f"{features['edge_density']:.4f}",
                                    sub_dir
                                ]
                                current_writer.writerow(row)

    # --- Step 5: Classification ---
    print("\n------------------------------------------------")
    print("STEP 5: CLASSIFICATION")
    print("------------------------------------------------")
    
    # Load data
    print("Loading TRAIN data...")
    X_train, y_train, filenames_train = load_data(train_csv_path)
    print(f" - Loaded {len(X_train)} training samples.")
    
    print("Loading TEST data...")
    X_test, y_test, filenames_test = load_data(test_csv_path)
    print(f" - Loaded {len(X_test)} testing samples.")
    
    if len(X_train) > 0:
        run_training_pipeline(X_train, y_train, filenames_train, X_test, y_test, filenames_test, output_dir)
    else:
        print("No training data extracted. Cannot perform classification.")

if __name__ == "__main__":
    main()
