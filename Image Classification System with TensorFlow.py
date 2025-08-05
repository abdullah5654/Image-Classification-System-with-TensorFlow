import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import cv2

# ----------- Step 1: Load Dataset -------------
data_dir = "dataset"  # Make sure this has "train" and "validation" subfolders
img_size = (160, 160)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, "train"),
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, "validation"),
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

class_names = train_ds.class_names
print("Classes:", class_names)

# ----------- Step 2: Preprocess & Optimize ----------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ----------- Step 3: Load Pretrained Model -------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Use as feature extractor

# Add custom top layers
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(2, activation='softmax')  # 2 classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ----------- Step 4: Train the Model ----------------
history = model.fit(train_ds, validation_data=val_ds, epochs=5)

# ----------- Step 5: Evaluate & Visualize -------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()

plt.figure()
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()

# ----------- Step 6: Save the Model -------------
model.save("cats_vs_dogs_model.h5")

# ----------- Step 7: Predict on New Image ----------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)
    
    # Display the image with prediction
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({confidence}%)")
    plt.axis('off')
    plt.show()

    print(f"Prediction: {predicted_class} with {confidence}% confidence")

# Example usage
predict_image("Enter your image path here.")



# ----------- Step 8: Webcam Prediction (Optional) -------------
# Uncomment the following code to use webcam for real-time predictions
# def webcam_prediction():
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         img = cv2.resize(frame, img_size)
#         img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
#         img_array = np.expand_dims(img, axis=0)

#         prediction = model.predict(img_array)[0]
#         predicted_class = class_names[np.argmax(prediction)]
#         confidence = round(100 * np.max(prediction), 2)

#         label = f"{predicted_class} ({confidence}%)"
#         cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                     (0, 255, 0), 2)
#         cv2.imshow('Webcam Prediction', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# Run this function to use webcam
# webcam_prediction()
