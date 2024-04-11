import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the paths for your dataset
train_data_dir = '/Users/shivyanshusaini/Downloads/archive (3) 2/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)'
validation_data_dir = '/Users/shivyanshusaini/Downloads/archive (3) 2/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)'
test_data_dir = '/Users/shivyanshusaini/Downloads/archive (3) 2'

# Define parameters
img_width, img_height = 224, 224
batch_size = 32  # Adjusted batch size
epochs = 20  # Adjusted number of epochs
num_classes = 2  # Adjusted number of classes
# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,  # Additional augmentation
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2  # Splitting data for validation
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Subset for training
)

validation_generator = train_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Subset for validation
)

# Load pre-trained EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze base model layers
base_model.trainable = False

# Build model on top of base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),  # Additional Dense layer
    layers.Dropout(0.5),  # Additional Dropout layer
    layers.Dense(64, activation='relu'),  # Additional Dense layer
    layers.Dropout(0.5),  # Additional Dropout layer
    layers.Dense(32, activation='relu'),  # Additional Dense layer
    layers.Dropout(0.5),  # Additional Dropout layer
    layers.Dense(num_classes, activation='softmax')
])


model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('modified_best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model with callbacks
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint]
)

model.save("trained_model.h5")

# Evaluate the model on the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

eval_result = model.evaluate(test_generator)
print(f'Test loss: {eval_result[0]}, Test accuracy: {eval_result[1]}')

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

# Load the trained model
model = models.load_model('modified_best_model.h5')  # Load the saved model

# Define the path to the image you want to classify
image_path = '/Users/shivyanshusaini/Desktop/pdp/0b444634-b557-45f4-a68a-8e9e38cd6683___RS_HL 2184_90deg.JPG'

# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))  # Assuming input size of EfficientNetB0 is (224, 224)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Get model predictions
predictions = model.predict(img_array)

# Custom decoding function
def decode_predictions_custom(preds, top=3):
    class_indices = {'class1': 0, 'class2': 1}  # Replace 'class1', 'class2', ... with your actual class names
    decoded_preds = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        decoded_preds.append([(class_name, pred[class_indices[class_name]]) for class_name, index in class_indices.items() if index in top_indices])
    return decoded_preds

# Decode predictions
decoded_predictions = decode_predictions_custom(predictions)
for pred in decoded_predictions:
    for class_name, prob in pred:
        print(f'{class_name}: {prob*100:.2f}%')