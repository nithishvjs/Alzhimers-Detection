import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import joblib
from PIL import Image
import matplotlib.pyplot as plt

# Constants
TRAIN_WIDTH = 176
TRAIN_HEIGHT = 208
CHANNELS = 1  # Grayscale
NUM_CLASSES = 4  # MID, MOD, VMD, ND

# Configure your test image path here
TEST_IMAGE_PATH = r"MRI_-_alzheimer-disease-FLAIR.jpg"  # Test image for prediction

class AlzheimerPredictor:
    def __init__(self, model_dir='models'):
        self.model = None
        self.feature_extractor = None
        self.le = LabelEncoder()
        self.classes = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
        self.model_dir = model_dir
        self.load_models()

    def load_models(self):
        """Load pre-trained models"""
        self.model = load_model(os.path.join(self.model_dir, 'cnn_model.h5'))
        self.feature_extractor = Sequential(self.model.layers[:-1])
        self.feature_extractor.compile(optimizer='adam', loss='categorical_crossentropy')

        self.trained_classifiers = {
            'Random Forest': joblib.load(os.path.join(self.model_dir, 'random_forest.pkl')),
            'SVM': joblib.load(os.path.join(self.model_dir, 'svm.pkl')),
            'KNN': joblib.load(os.path.join(self.model_dir, 'knn.pkl')),
            'Decision Tree': joblib.load(os.path.join(self.model_dir, 'decision_tree.pkl')),
            'XGBoost': joblib.load(os.path.join(self.model_dir, 'xgboost.pkl'))
        }
        self.le = joblib.load(os.path.join(self.model_dir, 'label_encoder.pkl'))
        print(f"\nModels loaded from {self.model_dir}")

    def preprocess_image(self, img_path):
        """Preprocess any image to model's expected format"""
        try:
            img = Image.open(img_path).convert('L')
            original_size = img.size

            # Preserve aspect ratio
            ratio = min(TRAIN_WIDTH/img.width, TRAIN_HEIGHT/img.height)
            new_size = (int(img.width*ratio), int(img.height*ratio))
            img = img.resize(new_size, Image.LANCZOS)

            # Center padding
            new_img = Image.new('L', (TRAIN_WIDTH, TRAIN_HEIGHT), 0)
            offset = ((TRAIN_WIDTH-new_size[0])//2, (TRAIN_HEIGHT-new_size[1])//2)
            new_img.paste(img, offset)

            img_array = np.array(new_img)/255.0
            img_array = np.expand_dims(img_array, axis=(0,-1))

            return img_array, original_size, new_size, offset
        except Exception as e:
            print(f"Image processing error: {e}")
            return None, None, None, None

    def predict(self, img_path):
        """Get predictions from all models for a single image"""
        img_array, orig_size, proc_size, offset = self.preprocess_image(img_path)
        if img_array is None:
            return None

        # CNN Prediction
        cnn_pred = self.model.predict(img_array)
        cnn_class = self.le.inverse_transform([np.argmax(cnn_pred)])[0]
        cnn_probs = cnn_pred[0]

        # Feature extraction
        features = self.feature_extractor.predict(img_array)

        # Get classifier predictions
        predictions = {}
        for name, clf in self.trained_classifiers.items():
            pred = clf.predict(features)
            proba = clf.predict_proba(features)[0] if hasattr(clf, 'predict_proba') else None
            predictions[name] = {
                'class': self.le.inverse_transform(pred)[0],
                'confidence': np.max(proba) if proba is not None else "N/A"
            }

        # Visualization
        self.visualize_results(img_path, orig_size, proc_size, offset,
                             cnn_class, cnn_probs, predictions)

        return {
            'original_dimensions': orig_size,
            'processed_dimensions': proc_size,
            'cnn_prediction': cnn_class,
            'cnn_confidence': np.max(cnn_probs),
            'classifier_predictions': predictions
        }

    def visualize_results(self, img_path, orig_size, proc_size, offset,
                        cnn_class, cnn_probs, classifier_preds):
        """Visualize input image and prediction results"""
        plt.figure(figsize=(18, 6))

        # Original Image
        plt.subplot(1, 3, 1)
        orig_img = Image.open(img_path).convert('L')
        plt.imshow(orig_img, cmap='gray')
        plt.title(f"Original\n{orig_size[0]}x{orig_size[1]}")
        plt.axis('off')

        # Processed Image
        plt.subplot(1, 3, 2)
        proc_img = Image.open(img_path).convert('L').resize(proc_size, Image.LANCZOS)
        final_img = Image.new('L', (TRAIN_WIDTH, TRAIN_HEIGHT), 0)
        final_img.paste(proc_img, offset)
        plt.imshow(final_img, cmap='gray')
        plt.title(f"Preprocessed\n{TRAIN_WIDTH}x{TRAIN_HEIGHT}")
        plt.axis('off')

        # Predictions
        plt.subplot(1, 3, 3)
        x = np.arange(len(self.classes))
        width = 0.15
        offset = width * 2
        plt.bar(x - offset, cnn_probs, width, label='CNN')

        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, (name, pred) in enumerate(classifier_preds.items()):
            if isinstance(pred['confidence'], float):
                plt.bar(x - offset + width*(i+1),
                       [1 if self.le.transform([pred['class']])[0] == j else 0
                        for j in range(len(self.classes))],
                       width, label=name, alpha=0.6, color=colors[i])

        plt.xticks(x, self.classes)
        plt.ylim(0, 1.2)
        plt.title('Predictions')
        plt.legend(bbox_to_anchor=(1, 1))
        plt.suptitle(f"Final Prediction: {cnn_class}", y=1.05)
        plt.tight_layout()
        plt.savefig('prediction_result.png')

# Run the code to test the trained model
if __name__ == "__main__":
    # Initialize the predictor with saved models
    predictor = AlzheimerPredictor(model_dir='models')  # Update this path if needed

    # Make prediction on a new test image
    if os.path.exists(TEST_IMAGE_PATH):
        print("\n===== Making Prediction =====")
        result = predictor.predict(TEST_IMAGE_PATH)

        if result:
            print("\nPrediction Summary:")
            print(f"Original Dimensions: {result['original_dimensions']}")
            print(f"CNN Prediction: {result['cnn_prediction']} (Confidence: {result['cnn_confidence']:.2%})")

            print("\nClassifier Predictions:")
            for name, pred in result['classifier_predictions'].items():
                print(f"{name}: {pred['class']} (Confidence: {pred['confidence']})")
    else:
        print(f"Error: Test image not found at {TEST_IMAGE_PATH}")
