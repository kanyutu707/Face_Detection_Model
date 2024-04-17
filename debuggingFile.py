#%%
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Input, Flatten, Lambda
import cv2
import os
import numpy as np
import uuid
from tensorflow.keras.metrics import Recall, Precision
import matplotlib.pyplot as plt

#%%

# Defining the path for images
verified_images_path = os.path.join('dataset', 'verified')
unverified_images_path = os.path.join('dataset', 'Faces')
stored_images_path = os.path.join('dataset', 'stored')


#%%

# Capturing images from the video camera
recording_camera = cv2.VideoCapture(0)

while recording_camera.isOpened():
    ret, frame = recording_camera.read()
    # Resizing and repositioning the captured image
    resize_dimensions = frame[150:250+150, 200:200+250, :]
    # Collecting images to store
    if cv2.waitKey(1) & 0xFF == ord('s'):
        captured_image = os.path.join(stored_images_path, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(captured_image, resize_dimensions)
    
    if cv2.waitKey(1) & 0xFF == ord('p'):
        captured_image = os.path.join(verified_images_path, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(captured_image, resize_dimensions)
    cv2.imshow('Image Collection', resize_dimensions)
    # Setting the connection breaker by pressing the letter q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
recording_camera.release()
cv2.destroyAllWindows()

#%%

# Getting images from directories
verified = tf.data.Dataset.list_files(verified_images_path + '/*.jpg')
unverified = tf.data.Dataset.list_files(unverified_images_path + '/*.jpg')
stored = tf.data.Dataset.list_files(stored_images_path + '/*.jpg').take(400)


#%%

# Image preprocess
def preprocess(image_path):
    spec_img = tf.io.read_file(image_path)
    decoded_image = tf.io.decode_jpeg(spec_img)
    resized_image = tf.image.resize(decoded_image, (100, 100))
    scaled_image = resized_image / 255.0
    return scaled_image

#%%
verified_images = tf.boolean_mask(stored, tf.strings.regex_full_match(stored, ".*verified.*"))

# Convert the tensor to a list for further processing
verified_images_list = verified_images.numpy().tolist()

# Filter the stored tensor to include only images from the 'unverified' folder
unverified_images = tf.boolean_mask(stored, tf.strings.regex_full_match(stored, ".*unverified.*"))

# Convert the tensor to a list for further processing
unverified_images_list = unverified_images.numpy().tolist()

def preprocess_twin(input_img_path, validation_img_path):
    label = input_img_path  # or validation_img_path, depending on your preference
    return (preprocess(input_img_path), preprocess(validation_img_path), label)

def create_combined_dataset(verified_images, unverified_images):
    # Create datasets for verified and unverified images
    verified_dataset = tf.data.Dataset.from_tensor_slices(verified_images)
    unverified_dataset = tf.data.Dataset.from_tensor_slices(unverified_images)

    # Pairing input and validation images
    verified_dataset = verified_dataset.map(lambda x: (x, x))
    unverified_dataset = unverified_dataset.map(lambda x: (x, x))

    # Combine datasets with their labels
    verified_labeled_dataset = tf.data.Dataset.zip((verified_dataset, tf.data.Dataset.from_tensor_slices(tf.ones(len(verified_images)))))
    unverified_labeled_dataset = tf.data.Dataset.zip((unverified_dataset, tf.data.Dataset.from_tensor_slices(tf.zeros(len(unverified_images)))))

    # Concatenate the datasets
    combined_dataset = verified_labeled_dataset.concatenate(unverified_labeled_dataset)

    return combined_dataset

# Assuming 'verified_images_list' and 'unverified_images_list' are lists of file paths for verified and unverified images
combined_dataset = create_combined_dataset(verified_images_list, unverified_images_list)

# Splitting the combined dataset into training and testing
train_size = round(len(combined_dataset) * 0.7)
test_size = round(len(combined_dataset) * 0.3)

train_dataset = combined_dataset.take(train_size)
test_dataset = combined_dataset.skip(train_size)

# Batch and prefetch the datasets
train_dataset = train_dataset.batch(20).prefetch(10)
test_dataset = test_dataset.batch(20).prefetch(10)


#%%

# Creating the neural network model
def create_embedding_layer():
    InputImage = Input(shape=(100, 100, 3), name='input_image')
    First_Layer_Conv2D = Conv2D(64, (10, 10), activation='relu')(InputImage)
    First_Layer_MaxPooling = MaxPooling2D(64, (2, 2), padding='same')(First_Layer_Conv2D)
    Second_Layer_Conv2D = Conv2D(128, (7, 7), activation='relu')(First_Layer_MaxPooling)
    Second_Layer_MaxPooling = MaxPooling2D(64, (2, 2), padding='same')(Second_Layer_Conv2D)
    Third_Layer_Conv2D = Conv2D(128, (4, 4), activation='relu')(Second_Layer_MaxPooling)
    Third_Layer_MaxPooling = MaxPooling2D(64, (2, 2), padding='same')(Third_Layer_Conv2D)
    Fourth_Layer_Conv2D = Conv2D(256, (4, 4), activation='relu')(Third_Layer_MaxPooling)
    Flattening_Layer = Flatten()(Fourth_Layer_Conv2D)
    Dense_Layer = Dense(4096, activation='sigmoid')(Flattening_Layer)
    return Model(inputs=[InputImage], outputs=[Dense_Layer], name='EmbeddingLayer')

embedding_summary = create_embedding_layer()

#%%

# Building the model without custom distance layer
def machine_learning_function():
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_img', shape=(100, 100, 3))
    
    # Calculate absolute difference between embeddings
    distance = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([embedding_summary(input_image), embedding_summary(validation_image)])
    
    classifier = Dense(1, activation='sigmoid')(distance)
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='ConvolutionalModel')

prediction_model = machine_learning_function()

# Setting up loss function and optimizer
loss_function = tf.losses.BinaryCrossentropy()
optimizer_function = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer_function=optimizer_function, prediction_model=prediction_model)

#%%

# Training the model
@tf.function
def training_function(batch):
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]
        y_pred = prediction_model(X, training=True)
        loss = loss_function(y, y_pred)
        gradient_achieved = tape.gradient(loss, prediction_model.trainable_variables)
        optimizer_function.apply_gradients(zip(gradient_achieved, prediction_model.trainable_variables))
        return loss
    
#%%
    
def loop_training_function(dataset, EPOCHS):
    for epoch in range(1, EPOCHS):
        print('\nEpoch {}/{}'.format(epoch, EPOCHS))
        progressBar = tf.keras.utils.Progbar(len(train_dataset))
        for idx, batch in enumerate(dataset):
            training_function(batch)
            progressBar.update(idx + 1)
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            

#%%
EPOCHS = 10
loop_training_function(train_dataset, EPOCHS)

#%%

# Testing the model
input_test, test_value, y_value = test_dataset.as_numpy_iterator().next()
y_prediction = prediction_model.predict([input_test, test_value])

#%%

# Classification metrics
recall_metric = Recall()
recall_metric.update_state(y_value, y_prediction)
print("Recall:", recall_metric.result().numpy())

#%%

precision_metric = Precision()
precision_metric.update_state(y_value, y_prediction)
print("Precision:", precision_metric.result().numpy())

#%%

# Visualizing the results
plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
plt.imshow(input_test[2])
plt.subplot(1, 2,2)

plt.imshow(test_value[2])
plt.show()

#%%
# Save the model
prediction_model.save('face_detection_model2.h5')

#%%

# Load the model
model = tf.keras.models.load_model('face_detection_model2.h5')
#%%
# Verifying the model
def model_verification_function(model, detection_measure, verification_measure):
    closest_image = None
    closest_similarity = 0.0
    
    stored_images_path = os.path.join('image_storage', 'stored_images')
    
    for file in os.listdir(stored_images_path):
        comparison_img = preprocess(os.path.join('image_storage', 'comparison_images', 'comparison_image.jpg'))
        stored_img = preprocess(os.path.join(stored_images_path, file))
        similarity = model.predict(list(np.expand_dims([comparison_img, stored_img], axis=1)))[0][0]
        if similarity > closest_similarity:
            closest_similarity = similarity
            closest_image = file
    
    closest_image_name = os.path.splitext(os.path.basename(closest_image))[0] if closest_image else None
    return closest_image_name, closest_similarity

#%%

open_camera = cv2.VideoCapture(0)
while open_camera.isOpened():
    ret, frame = open_camera.read()
    frame_size = frame[150:250+150, 200:200+250, :]
    cv2.imshow('q', frame_size)
    
    if cv2.waitKey(1) & 0xFF == ord('v'):
        cv2.imwrite(os.path.join('image_storage', 'comparison_images', 'comparison_image.jpg'), frame_size)
        closest_image, closest_similarity = model_verification_function(model, 0.9, 0.7)
        print("Closest Image:", closest_image)
        print("Similarity:", closest_similarity)
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
open_camera.release()
cv2.destroyAllWindows()
#%%
