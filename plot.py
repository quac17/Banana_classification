import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix_from_csv(csv_path, output_image_path, title_suffix=""):
    """
    Reads classification results from CSV and plots a normalized confusion matrix
    showing the percentage of predictions for each true label.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return

    true_labels = []
    predicted_labels = []

    print(f"Reading data from: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_labels.append(row['True_Label'])
            predicted_labels.append(row['Predicted_Label'])

    if not true_labels:
        print("No data found in CSV to plot.")
        return

    # Get unique labels (sorted for consistency)
    labels = sorted(list(set(true_labels) | set(predicted_labels)))
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    
    # Normalize by row (True Label count) to get percentages
    # Add epsilon or handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    cm_normalized = np.nan_to_num(cm_normalized) # Replace NaNs with 0

    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm_normalized, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Set axis labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix {title_suffix} (Prediction %)', pad=20, fontsize=14)

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            # Show percentage and raw count
            percent = cm_normalized[i, j] * 100
            count = cm[i, j]
            
            # Choose text color based on background intensity
            text_color = "white" if cm_normalized[i, j] > 0.5 else "black"
            
            label_text = f"{percent:.1f}%\n({count})"
            ax.text(j, i, label_text, ha="center", va="center", color=text_color, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_image_path)
    print(f"Success! Confusion Matrix saved to: {output_image_path}")
    plt.close()

def plot_svm_boundary(model_class, X, y, output_path, le, title="SVM Decision Boundary"):
    """
    Trains a 2D visualization model (using PCA) and plots decision boundaries.
    Note: This trains a NEW model on 2D projected data for visualization purposes.
    """
    from sklearn.decomposition import PCA
    from sklearn.inspection import DecisionBoundaryDisplay
    
    # Reduce to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    # Train a 2D model for plotting
    clf_2d = model_class
    clf_2d.fit(X_2d, y)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot decision boundary
    DecisionBoundaryDisplay.from_estimator(
        clf_2d,
        X_2d,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        plot_method="pcolormesh",
        shading="auto",
        alpha=0.6,
        ax=ax
    )
    
    # Plot data points
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k", s=50)
    
    # Legend
    handles, _ = scatter.legend_elements()
    class_names = le.classes_
    ax.legend(handles, class_names, title="Classes")
    
    plt.title(title)
    plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.2f} var)")
    plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.2f} var)")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Success! SVM Boundary Chart saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Decision Tree Report
    dt_input_csv = os.path.join(base_dir, 'output', 'test_classification_results.csv')
    dt_output_png = os.path.join(base_dir, 'output', 'test_confusion_matrix.png')
    if os.path.exists(dt_input_csv):
        plot_confusion_matrix_from_csv(dt_input_csv, dt_output_png, "(Decision Tree)")
        
    # 2. KNN Report
    knn_input_csv = os.path.join(base_dir, 'KNN_output', 'test_knn_classification_results.csv')
    knn_output_png = os.path.join(base_dir, 'KNN_output', 'test_knn_confusion_matrix.png')
    if os.path.exists(knn_input_csv):
        plot_confusion_matrix_from_csv(knn_input_csv, knn_output_png, "(KNN)")
        
    # 3. SVM Report (Confusion Matrix Only - Boundary plot is called from process.py)
    svm_input_csv = os.path.join(base_dir, 'SVM_output', 'test_svm_classification_results.csv')
    svm_output_png = os.path.join(base_dir, 'SVM_output', 'test_svm_confusion_matrix.png')
    if os.path.exists(svm_input_csv):
        plot_confusion_matrix_from_csv(svm_input_csv, svm_output_png, "(SVM)")
