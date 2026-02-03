import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

def create_cnn_vit_model(input_shape=(128, 128, 1), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    
    # CNN Part: Extract local spatial features
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Reshape for ViT Part
    # After 3 poolings: 128/8 = 16. Shape: (16, 16, 128)
    c = x.shape[-1]
    x = layers.Reshape((-1, c))(x)
    
    # ViT Part: Global shape structure and attention
    num_heads = 4
    embed_dim = c
    ff_dim = 256
    num_transformer_blocks = 2
    
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    
    # Global Pooling and Classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# List of shapes
SHAPES = [
    "Circle", "Square", "Rectangle", "Triangle", "Pentagon",
    "Hexagon", "Heptagon", "Octagon", "Star", "Ellipse"
]

MODEL_PATH = 'shape_model.h5'

def generate_synthetic_data(samples_per_class=100):
    """Generates synthetic shape data for training."""
    X, y = [], []
    size = 128
    
    for idx, shape_name in enumerate(SHAPES):
        for _ in range(samples_per_class):
            img = np.zeros((size, size), dtype=np.uint8)
            center = (np.random.randint(40, 88), np.random.randint(40, 88))
            w = np.random.randint(30, 50)
            h = np.random.randint(30, 50)
            color = 255
            thickness = np.random.randint(2, 6)
            
            if shape_name == "Circle":
                cv2.circle(img, center, w, color, thickness)
            elif shape_name == "Square":
                cv2.rectangle(img, (center[0]-w, center[1]-w), (center[0]+w, center[1]+w), color, thickness)
            elif shape_name == "Rectangle":
                cv2.rectangle(img, (center[0]-w, center[1]-h), (center[0]+w, center[1]+h), color, thickness)
            elif shape_name == "Triangle":
                pts = np.array([[center[0], center[1]-h], [center[0]-w, center[1]+h], [center[0]+w, center[1]+h]])
                cv2.polylines(img, [pts], True, color, thickness)
            elif shape_name in ["Pentagon", "Hexagon", "Heptagon", "Octagon"]:
                sides = SHAPES.index(shape_name) + 1 if shape_name != "Triangle" else 3
                if shape_name == "Pentagon": sides = 5
                elif shape_name == "Hexagon": sides = 6
                elif shape_name == "Heptagon": sides = 7
                elif shape_name == "Octagon": sides = 8
                pts = []
                for i in range(sides):
                    angle = i * 2 * np.pi / sides
                    px = int(center[0] + w * np.cos(angle))
                    py = int(center[1] + w * np.sin(angle))
                    pts.append([px, py])
                cv2.polylines(img, [np.array(pts)], True, color, thickness)
            elif shape_name == "Star":
                pts = []
                for i in range(10):
                    angle = i * np.pi / 5 - np.pi / 2
                    r = w if i % 2 == 0 else w // 2
                    px = int(center[0] + r * np.cos(angle))
                    py = int(center[1] + r * np.sin(angle))
                    pts.append([px, py])
                cv2.polylines(img, [np.array(pts)], True, color, thickness)
            elif shape_name == "Ellipse":
                cv2.ellipse(img, center, (w, h), 0, 0, 360, color, thickness)
            
            # Add some noise
            noise = np.random.randint(0, 50, (size, size), dtype=np.uint8)
            img = cv2.add(img, noise)
            
            img = img.astype('float32') / 255.0
            X.append(np.expand_dims(img, axis=-1))
            y.append(idx)
            
    return np.array(X), np.array(y)

def train_model():
    print("Generating data...")
    X, y = generate_synthetic_data(200)
    print("Creating model...")
    model = create_cnn_vit_model()
    print("Training...")
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH, custom_objects={'TransformerBlock': TransformerBlock})
        except:
            return train_model()
    else:
        return train_model()
