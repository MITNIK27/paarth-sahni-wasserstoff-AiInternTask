#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install opencv-python')
get_ipython().system('pip install matplotlib')


# In[4]:


import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def create_synthetic_data(num_images, img_size=(128, 128)):
    images = []
    masks = []

    for _ in range(num_images):
        # Create an empty image and mask
        img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)

        # Draw random geometric shapes
        num_shapes = np.random.randint(1, 4)
        for _ in range(num_shapes):
            shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])
            color = (255, 255, 255)

            if shape_type == 'circle':
                center = (np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1]))
                radius = np.random.randint(10, img_size[0] // 4)
                cv2.circle(img, center, radius, color, -1)
                cv2.circle(mask, center, radius, 255, -1)

            elif shape_type == 'rectangle':
                pt1 = (np.random.randint(0, img_size[0] // 2), np.random.randint(0, img_size[1] // 2))
                pt2 = (pt1[0] + np.random.randint(10, img_size[0] // 2), pt1[1] + np.random.randint(10, img_size[1] // 2))
                cv2.rectangle(img, pt1, pt2, color, -1)
                cv2.rectangle(mask, pt1, pt2, 255, -1)

            elif shape_type == 'triangle':
                pts = np.array([[np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1])],
                                [np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1])],
                                [np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1])]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(img, [pts], color)
                cv2.fillPoly(mask, [pts], 255)

        # Normalize the images and masks
        img = img / 255.0
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Create synthetic dataset
num_images = 1000
X, y = create_synthetic_data(num_images)


# In[ ]:





# In[8]:


import tensorflow as tf
from tensorflow.keras import layers, models

def unet_model(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Instantiate and compile the model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=20, batch_size=16, validation_split=0.1)


# In[2]:


# Function to visualize predictions
def visualize_predictions(model, images, num_images=3):
    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        img = images[i]
        img_expanded = np.expand_dims(img, axis=0)
        pred_mask = model.predict(img_expanded)[0]
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.title("Input Image")
        plt.imshow(img)
        plt.subplot(num_images, 3, i * 3 + 2)
        plt.title("Predicted Mask")
        plt.imshow(pred_mask[:, :, 0], cmap='gray')
        plt.subplot(num_images, 3, i * 3 + 3)
        plt.title("Overlay")
        plt.imshow(img)
        plt.imshow(pred_mask[:, :, 0], cmap='gray', alpha=0.5)
    plt.show()

# Visualize predictions
visualize_predictions(model, X)


# In[49]:


import numpy as np
import cv2
import os

def extract_and_save_objects(image, mask, master_id, output_dir):
    """
    Extract each segmented object from the mask and save them separately.
    
    Args:
        image (numpy array): The input image.
        mask (numpy array): The segmented mask.
        master_id (int): Master ID for the original image.
        output_dir (str): Directory to save the extracted objects.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert mask to binary
    mask_binary = (mask[:, :, 0] > 0.5).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    object_id = 0
    for contour in contours:
        # Create a mask for each object
        obj_mask = np.zeros_like(mask_binary)
        cv2.drawContours(obj_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Create an object image
        obj_image = cv2.bitwise_and(image, image, mask=obj_mask)
        
        # Increment object ID
        object_id += 1
        obj_id = f"{master_id}_{object_id}"
        
        # Save the object image
        obj_image_path = os.path.join(output_dir, f"{obj_id}.png")
        cv2.imwrite(obj_image_path, (obj_image * 255).astype(np.uint8))
        
        # Save metadata
        with open(os.path.join(output_dir, f"{obj_id}.txt"), 'w') as f:
            f.write(f"Object ID: {obj_id}\n")
            f.write(f"Master ID: {master_id}\n")
            f.write(f"Object Area: {cv2.contourArea(contour)}\n")

# Directory to save extracted objects
output_dir = 'extracted_objects'

# Loop through each image and mask
for idx in range(len(X)):
    img = X[idx]
    mask = y[idx]
    master_id = idx + 1
    extract_and_save_objects(img, mask, master_id, output_dir)


# In[ ]:





# In[1]:


model.save('my_model.h5')


# In[1]:


from tensorflow.keras.models import load_model


# In[2]:


model = load_model('my_model.h5')


# In[3]:


import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('my_model.h5')

# Recompile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[4]:


import warnings
import tensorflow as tf

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='absl')

# Load the saved model
model = tf.keras.models.load_model('my_model.h5')

# Continue with inference or other tasks


# In[5]:


import numpy as np
import cv2
import os
import tensorflow as tf
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='absl')

# Define the function to identify and describe objects
def identify_and_describe_objects(image, model):
    img_resized = cv2.resize(image, (128, 128)) / 255.0
    img_expanded = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_expanded)
    class_index = np.argmax(prediction)
    
    # Check the number of classes and update shape_dict accordingly
    num_classes = model.output_shape[-1]
    shape_dict = {i: f'class_{i}' for i in range(num_classes)}  # Example for class naming
    
    if class_index not in shape_dict:
        print(f"Unexpected class index: {class_index}. Valid indices are: {list(shape_dict.keys())}")
    
    predicted_shape = shape_dict.get(class_index, 'Unknown')
    return predicted_shape

# Directory containing extracted object images
extracted_objects_dir = 'extracted_objects'

# Load the saved model
model = tf.keras.models.load_model('my_model.h5')

# Recompile the model (optional but recommended for evaluation)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Document containing the identified objects and their descriptions
with open('identified_objects_descriptions.txt', 'w') as f:
    for obj_image_name in os.listdir(extracted_objects_dir):
        if obj_image_name.endswith('.png'):
            obj_image_path = os.path.join(extracted_objects_dir, obj_image_name)
            obj_image = cv2.imread(obj_image_path)
            if obj_image is not None:
                predicted_shape = identify_and_describe_objects(obj_image, model)
                f.write(f"{obj_image_name}: {predicted_shape}\n")
            else:
                print(f"Error loading image: {obj_image_path}")

print("Object identification and description complete. Results saved in 'identified_objects_descriptions.txt'.")


# In[10]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Model

def build_ocr_model(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # Convolutional layers for feature extraction
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Reshape to feed into RNN
    new_shape = ((input_shape[0] // 4), (input_shape[1] // 4) * 64)
    x = Reshape(target_shape=new_shape)(x)

    # RNN layers for sequence modeling
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Build and compile model
    model = Model(inputs, x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage
input_shape = (128, 128, 1)  # (height, width, channels)
num_classes = 10  # Number of unique characters or tokens
model = build_ocr_model(input_shape, num_classes)
model.summary()


# In[15]:


get_ipython().system('pip install labelImg')


# In[ ]:





# In[27]:


def load_dataset(image_dir, labels_file):
    # Load labels
    labels_df = pd.read_csv(labels_file, delimiter='\t', header=None, names=['filename', 'label'])
    
    # Load images and labels
    images = []
    labels = []
    for _, row in labels_df.iterrows():
        image_path = os.path.join(image_dir, row['filename'])
        image = parse_image(image_path)
        images.append(image)
        labels.append(row['label'])

    images = np.array(images)
    labels = np.array(labels)

    # Convert labels to numerical tokens
    label_encoder = tf.keras.preprocessing.text.Tokenizer()
    label_encoder.fit_on_texts(labels)
    labels = label_encoder.texts_to_sequences(labels)
    labels = np.array(labels).flatten()  # Flatten to 1D array if needed
    
    # Check for empty labels array
    if len(labels) == 0:
        raise ValueError("Labels array is empty. Check your labels and dataset.")
    
    # Find number of classes
    num_classes = len(label_encoder.word_index) + 1
    
    # Convert labels to one-hot encoding
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

    return images, labels


# In[30]:


def load_dataset(image_dir, labels_file):
    # Load labels
    labels_df = pd.read_csv(labels_file, delimiter='\t', header=None, names=['filename', 'label'])
    
    # Print out the first few rows of the DataFrame to check its content
    print("Labels DataFrame:")
    print(labels_df.head())
    
    # Check if DataFrame is empty
    if labels_df.empty:
        raise ValueError("Labels DataFrame is empty. Check the labels file.")
    
    # Load images and labels
    images = []
    labels = []
    for _, row in labels_df.iterrows():
        image_path = os.path.join(image_dir, row['filename'])
        image = parse_image(image_path)
        images.append(image)
        labels.append(row['label'])

    images = np.array(images)
    labels = np.array(labels)

    # Print shapes of arrays
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Convert labels to numerical tokens
    label_encoder = tf.keras.preprocessing.text.Tokenizer()
    label_encoder.fit_on_texts(labels)
    labels = label_encoder.texts_to_sequences(labels)
    labels = np.array(labels).flatten()  # Flatten to 1D array if needed

    # Check for empty labels array
    if len(labels) == 0:
        raise ValueError("Labels array is empty. Check your labels and dataset.")
    
    # Find number of classes
    num_classes = len(label_encoder.word_index) + 1
    
    # Convert labels to one-hot encoding
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

    return images, labels, num_classes


# In[31]:


def parse_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.resize(image, (128, 128))  # Adjust size as needed
        image = image / 255.0  # Normalize
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return np.zeros((128, 128, 3))  # Return a default image or handle as needed


# In[32]:


import os

# Check if labels file exists
if not os.path.isfile(labels_file):
    raise FileNotFoundError(f"Labels file does not exist: {labels_file}")

# Check if images directory exists
if not os.path.isdir(image_dir):
    raise FileNotFoundError(f"Images directory does not exist: {image_dir}")


# In[33]:


# Convert labels to one-hot encoding
labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
print(f"One-hot encoded labels shape: {labels.shape}")


# In[36]:





# In[37]:


from tensorflow.keras.utils import to_categorical

# Assuming labels are integers representing class indices
labels = to_categorical(labels, num_classes=num_classes)


# In[38]:


for batch_images, batch_labels in train_dataset.take(1):
    print(batch_images.shape)  # Should be (batch_size, height, width, channels)
    print(batch_labels.shape)  # Should be (batch_size, num_classes)


# In[39]:


import tensorflow as tf
import numpy as np

def create_tf_dataset(images, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Create datasets
train_dataset = create_tf_dataset(train_images, train_labels)
val_dataset = create_tf_dataset(val_images, val_labels)


# In[46]:


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

def build_ocr_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)  # Ensure this matches num_classes
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[47]:


input_shape = (128, 128, 3)  # Update based on your image size
num_classes = 8  # Ensure this matches your dataset

model = build_ocr_model(input_shape, num_classes)

# Train the model
model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset
)


# In[48]:


from tensorflow.keras.utils import to_categorical

labels = to_categorical(labels, num_classes=num_classes)


# In[49]:


for batch_images, batch_labels in train_dataset.take(1):
    print(f"Batch images shape: {batch_images.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")


# In[ ]:





# In[51]:


import numpy as np
import cv2

def predict_text(image, model):
    img_resized = cv2.resize(image, (128, 128)) / 255.0
    img_expanded = np.expand_dims(img_resized, axis=0)
    predictions = model.predict(img_expanded)
    predicted_class = np.argmax(predictions, axis=-1)
    # Map class index to text
    # Assuming you have a mapping from indices to text
    index_to_text = {i: f'Class_{i}' for i in range(num_classes)}
    return index_to_text.get(predicted_class[0], 'Unknown')


# In[52]:


import os

def process_images(image_dir, model, output_file):
    with open(output_file, 'w') as f:
        for image_name in os.listdir(image_dir):
            if image_name.endswith('.png'):
                image_path = os.path.join(image_dir, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    text = predict_text(image, model)
                    f.write(f"{image_name}: {text}\n")
                else:
                    print(f"Error loading image: {image_path}")

process_images('extracted_objects', model, 'extracted_text_data.txt')


# In[53]:


def save_annotated_images(image_dir, model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for image_name in os.listdir(image_dir):
        if image_name.endswith('.png'):
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                text = predict_text(image, model)
                # Annotate image
                annotated_image = cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                output_path = os.path.join(output_dir, image_name)
                cv2.imwrite(output_path, annotated_image)
            else:
                print(f"Error loading image: {image_path}")

save_annotated_images('extracted_objects', model, 'annotated_images')


# In[57]:


import os

def summarize_object_attributes(image_dir, model, output_file):
    with open(output_file, 'w') as f:
        for image_name in os.listdir(image_dir):
            if image_name.endswith('.png'):
                image_path = os.path.join(image_dir, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    # Predict the text
                    text = predict_text(image, model)
                    
                    # For this example, attributes include:
                    # - Image Name
                    # - Predicted Text
                    # - Additional attributes (e.g., object size, color, etc.)
                    height, width, _ = image.shape
                    attributes = {
                        'Image Name': image_name,
                        'Predicted Text': text,
                        'Image Height': height,
                        'Image Width': width
                    }
                    
                    # Write summary to file
                    f.write(f"Image Name: {attributes['Image Name']}\n")
                    f.write(f"Predicted Text: {attributes['Predicted Text']}\n")
                    f.write(f"Image Height: {attributes['Image Height']}\n")
                    f.write(f"Image Width: {attributes['Image Width']}\n")
                    f.write("\n" + "="*40 + "\n\n")
                else:
                    print(f"Error loading image: {image_path}")

# Call the function to generate the summary
summarize_object_attributes('extracted_objects', model, 'object_attributes_summary.txt')


# In[58]:


import json
import os
import cv2

def map_data_to_objects(image_dir, model, output_json):
    data_mapping = {}
    
    for image_name in os.listdir(image_dir):
        if image_name.endswith('.png'):
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            
            if image is not None:
                # Predict the text
                text = predict_text(image, model)
                
                # Generate unique ID for each object
                object_id = image_name.split('.')[0]
                
                # Gather object attributes
                height, width, _ = image.shape
                attributes = {
                    'Unique ID': object_id,
                    'Predicted Text': text,
                    'Image Height': height,
                    'Image Width': width
                }
                
                # Add attributes to the data mapping
                data_mapping[object_id] = attributes
            else:
                print(f"Error loading image: {image_path}")

    # Write data mapping to JSON file
    with open(output_json, 'w') as f:
        json.dump(data_mapping, f, indent=4)

# Call the function to map data and generate JSON
map_data_to_objects('extracted_objects', model, 'data_mapping.json')


# In[59]:


{
    "image_1": {
        "Unique ID": "image_1",
        "Predicted Text": "Class_0",
        "Image Height": 128,
        "Image Width": 128
    },
    "image_2": {
        "Unique ID": "image_2",
        "Predicted Text": "Class_1",
        "Image Height": 128,
        "Image Width": 128
    }
}


# In[60]:


import json

# Load the data mapping from JSON
with open('data_mapping.json', 'r') as f:
    data_mapping = json.load(f)

# Example query
for obj_id, attributes in data_mapping.items():
    print(f"Object ID: {obj_id}")
    print(f"Predicted Text: {attributes['Predicted Text']}")
    print(f"Image Height: {attributes['Image Height']}")
    print(f"Image Width: {attributes['Image Width']}")
    print()


# In[62]:


import os

print("Current Working Directory:", os.getcwd())


# In[73]:


from PIL import Image, ImageDraw, ImageFont
import cv2

def annotate_image(original_image_path, data_mapping, output_image_path):
    # Load the original image
    image = cv2.imread(original_image_path)
    
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {original_image_path}")
    
    height, width, _ = image.shape
    
    # Convert image to RGB (PIL needs RGB format)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    
    # Define a font (you may need to adjust the font path or size)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Annotate the image
    for obj_id, attributes in data_mapping.items():
        # Draw a rectangle and text
        text = attributes['Predicted Text']
        # Example bounding box (use actual coordinates)
        x1, y1, x2, y2 = attributes['Bounding Box']  # Use actual bounding box coordinates
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # Draw text
        draw.text((x1, y1 - 15), f"{text}", fill="red", font=font)

    # Save the annotated image
    image_pil.save(output_image_path)

# Example usage
data_mapping = {
    1: {'Predicted Text': 'Example Text', 'Bounding Box': (50, 50, 200, 100)}
}
annotate_image('C:/Users/Parth/Assessment/annotated_images/102_1.png', data_mapping, 'annotated_image.png')


# In[ ]:





# In[74]:


get_ipython().system('pip install streamlit')


# In[77]:


import streamlit as st
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
import os

# Load the trained model (update the path to your model file)
model_path = 'my_model.h5'  # Update this path
if not os.path.isfile(model_path):
    st.error(f"Model file not found at path: {model_path}")
else:
    model = tf.keras.models.load_model(model_path)

# Define a function to annotate the image
def annotate_image(image, data_mapping):
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for obj_id, attributes in data_mapping.items():
        text = attributes['Predicted Text']
        x1, y1, x2, y2 = attributes['bbox']  # Ensure your data_mapping has correct bbox

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 15), f"{text}", fill="red", font=font)

    return np.array(image_pil)

# Define a function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))  # Resize according to your model's input size
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("OCR Pipeline Testing")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess and predict
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    
    # Dummy data_mapping for demonstration; replace with actual processing
    data_mapping = {
        1: {
            'Predicted Text': 'Sample Text',
            'bbox': [50, 50, 150, 150]  # Dummy bounding box coordinates
        }
    }

    annotated_image = annotate_image(image, data_mapping)
    st.image(annotated_image, caption="Annotated Image", use_column_width=True)

    st.write("Done!")


# In[79]:


import streamlit as st
from PIL import Image

# Set up the Streamlit UI
st.title('Image Annotation Tool')

# File uploader widget to upload images
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Example: add a textbox for user input
    user_input = st.text_input("Enter text to annotate:")
    
    if st.button('Annotate Image'):
        # Annotation logic here (example only)
        st.write(f"Annotating image with: {user_input}")
        # Save or display annotated image here


# In[ ]:





# In[ ]:





# In[ ]:




