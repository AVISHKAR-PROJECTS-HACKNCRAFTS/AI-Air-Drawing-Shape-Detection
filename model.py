import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import random

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
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


class PatchEmbedding(layers.Layer):
    """Converts CNN feature maps into patch embeddings with positional encoding."""
    def __init__(self, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.projection = layers.Dense(embed_dim)
    
    def build(self, input_shape):
        num_patches = input_shape[1]
        self.position_embedding = self.add_weight(
            name="pos_embed", shape=(1, num_patches, self.embed_dim), initializer="random_normal"
        )
        super().build(input_shape)
    
    def call(self, x):
        x = self.projection(x)
        x = x + self.position_embedding
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


def create_cnn_vit_model(input_shape=(128, 128, 1), num_classes=12):
    """
    Enhanced CNN + Vision Transformer hybrid model.
    - CNN backbone: 4 conv blocks for rich local feature extraction
    - Patch Embedding with learned positional encoding
    - 3 Transformer blocks for global attention over spatial features
    - Dense classification head with regularization
    """
    inputs = layers.Input(shape=input_shape)
    
    # ---- CNN Backbone: Extract local spatial features ----
    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.15)(x)
    
    # Block 4
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.15)(x)
    
    # ---- Reshape CNN features into sequence of patches ----
    # After 4 poolings: 128/16 = 8. Shape: (8, 8, 256) -> (64, 256)
    c = x.shape[-1]  # 256
    x = layers.Reshape((-1, c))(x)
    
    # ---- Patch Embedding with positional encoding ----
    embed_dim = 128
    x = PatchEmbedding(embed_dim)(x)
    
    # ---- Transformer Blocks: Global attention ----
    num_heads = 4
    ff_dim = 256
    num_transformer_blocks = 3
    
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, rate=0.15)(x)
    
    # ---- Classification Head ----
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Use learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3, decay_steps=1000, alpha=1e-5
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ═══════════════════════════════════════════════════════════════
# SHAPES LIST — Only creative/non-geometric shapes (12 classes)
# ═══════════════════════════════════════════════════════════════
SHAPES = [
    "Spiral", "Infinity", "Cloud", "Lightning bolt", "Flower",
    "Butterfly", "Crown", "Flame", "Fish", "Leaf",
    "Music note", "Smiley face"
]

MODEL_PATH = 'shape_model.h5'


# ═══════════════════════════════════════════════════════════════
# Data Augmentation Utilities
# ═══════════════════════════════════════════════════════════════

def augment_image(img):
    """Apply random augmentations to a single image."""
    size = img.shape[0]
    
    # Random rotation (-20 to +20 degrees)
    angle = np.random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (size, size))
    
    # Random scaling (0.85 to 1.15)
    scale = np.random.uniform(0.85, 1.15)
    M_scale = cv2.getRotationMatrix2D((size // 2, size // 2), 0, scale)
    img = cv2.warpAffine(img, M_scale, (size, size))
    
    # Random translation (-8 to +8 pixels)
    tx = np.random.randint(-8, 9)
    ty = np.random.randint(-8, 9)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M_trans, (size, size))
    
    # Random noise intensity
    noise_level = np.random.randint(0, 40)
    noise = np.random.randint(0, noise_level + 1, (size, size), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Random thickness variation via dilation/erosion
    if np.random.random() > 0.5:
        kernel = np.ones((2, 2), np.uint8)
        if np.random.random() > 0.5:
            img = cv2.dilate(img, kernel, iterations=1)
        else:
            img = cv2.erode(img, kernel, iterations=1)
    
    return img


def draw_shape_on_canvas(shape_name, size=128):
    """Draw a single shape on a blank canvas with random variations."""
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Randomized parameters for natural variation
    center = (np.random.randint(38, 90), np.random.randint(38, 90))
    w = np.random.randint(25, 50)
    h = np.random.randint(25, 50)
    color = 255
    thickness = np.random.randint(2, 6)
    
    if shape_name == "Spiral":
        pts = []
        turns = np.random.uniform(3, 5)  # vary spiral turns
        for theta in np.arange(0, turns * np.pi, 0.08):
            r = (theta / (turns * np.pi)) * w
            px = int(center[0] + r * np.cos(theta))
            py = int(center[1] + r * np.sin(theta))
            pts.append([px, py])
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], False, color, thickness)
    
    elif shape_name == "Infinity":
        pts = []
        for t in np.arange(0, 2 * np.pi, 0.08):
            den = 1 + np.sin(t)**2
            scale_x = np.random.uniform(1.3, 1.7) if t == 0 else scale_x
            px = int(center[0] + w * scale_x * np.cos(t) / den)
            py = int(center[1] + w * scale_x * np.sin(t) * np.cos(t) / den)
            pts.append([px, py])
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], True, color, thickness)
    
    elif shape_name == "Cloud":
        num_bumps = np.random.randint(3, 6)
        base_r = int(w * 0.7)
        # Main cloud body
        cv2.circle(img, center, base_r, color, thickness)
        for i in range(num_bumps):
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0.4, 0.7) * w
            cr = int(np.random.uniform(0.4, 0.8) * base_r)
            cx = int(center[0] + dist * np.cos(angle))
            cy = int(center[1] + dist * np.sin(angle))
            cv2.circle(img, (cx, cy), cr, color, thickness)
    
    elif shape_name == "Lightning bolt":
        # Zigzag bolt with some randomness
        num_zags = np.random.randint(3, 6)
        pts = [[center[0], center[1] - h]]
        for i in range(1, num_zags):
            direction = 1 if i % 2 == 0 else -1
            px = center[0] + direction * np.random.randint(w // 4, w // 2)
            py = center[1] - h + int((2 * h / num_zags) * i)
            pts.append([px, py])
        pts.append([center[0], center[1] + h])
        cv2.polylines(img, [np.array(pts)], False, color, thickness)
    
    elif shape_name == "Flower":
        # Center
        center_r = np.random.randint(w // 4, w // 3)
        cv2.circle(img, center, center_r, color, thickness)
        # Petals
        num_petals = np.random.randint(4, 7)
        for i in range(num_petals):
            angle = i * 2 * np.pi / num_petals + np.random.uniform(-0.15, 0.15)
            petal_dist = np.random.uniform(0.8, 1.1) * w
            px = int(center[0] + petal_dist * np.cos(angle))
            py = int(center[1] + petal_dist * np.sin(angle))
            petal_r = np.random.randint(center_r - 2, center_r + 6)
            cv2.circle(img, (px, py), petal_r, color, thickness)
    
    elif shape_name == "Butterfly":
        # Body
        body_w = np.random.randint(w // 6, w // 4)
        cv2.ellipse(img, center, (body_w, h), 0, 0, 360, color, thickness)
        # Upper wings
        wing_w = np.random.randint(w // 2 - 3, w // 2 + 5)
        wing_h = np.random.randint(h // 2 - 3, h // 2 + 5)
        cv2.ellipse(img, (center[0] - wing_w, center[1] - h // 3), (wing_w, wing_h), 30, 0, 360, color, thickness)
        cv2.ellipse(img, (center[0] + wing_w, center[1] - h // 3), (wing_w, wing_h), -30, 0, 360, color, thickness)
        # Lower wings (smaller)
        lw = int(wing_w * 0.65)
        lh = int(wing_h * 0.65)
        cv2.ellipse(img, (center[0] - wing_w, center[1] + h // 3), (lw, lh), -30, 0, 360, color, thickness)
        cv2.ellipse(img, (center[0] + wing_w, center[1] + h // 3), (lw, lh), 30, 0, 360, color, thickness)
    
    elif shape_name == "Crown":
        num_peaks = np.random.randint(3, 6)
        pts = [[center[0] - w, center[1] + h // 2]]
        for i in range(num_peaks):
            frac = (i + 0.5) / num_peaks
            tip_x = int(center[0] - w + 2 * w * frac)
            tip_y = center[1] - h // 2 + np.random.randint(-5, 5)
            valley_x = int(center[0] - w + 2 * w * (i + 1) / num_peaks)
            valley_y = center[1] + np.random.randint(-3, 5)
            pts.append([tip_x, tip_y])
            if i < num_peaks - 1:
                pts.append([valley_x, valley_y])
        pts.append([center[0] + w, center[1] + h // 2])
        cv2.polylines(img, [np.array(pts)], True, color, thickness)
    
    elif shape_name == "Flame":
        pts = []
        for t in np.arange(0, 2 * np.pi, 0.08):
            # Teardrop / flame shape
            fx = w * np.cos(t)
            mod = np.sin(t / 2) ** np.random.uniform(0.3, 0.7) if t < np.pi else 1.0
            fy = h * np.sin(t) * mod
            px = int(center[0] + fx)
            py = int(center[1] + fy)
            pts.append([px, py])
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], True, color, thickness)
    
    elif shape_name == "Fish":
        # Body
        body_w = np.random.randint(w - 5, w + 5)
        body_h = np.random.randint(h // 2 - 3, h // 2 + 5)
        cv2.ellipse(img, center, (body_w, body_h), 0, 0, 360, color, thickness)
        # Tail
        tail_w = np.random.randint(w // 3, w // 2 + 3)
        tail_pts = np.array([
            [center[0] - body_w, center[1]],
            [center[0] - body_w - tail_w, center[1] - body_h],
            [center[0] - body_w - tail_w, center[1] + body_h]
        ])
        cv2.polylines(img, [tail_pts], True, color, thickness)
        # Eye
        eye_r = max(2, body_h // 5)
        cv2.circle(img, (center[0] + body_w // 2, center[1] - body_h // 3), eye_r, color, -1)
    
    elif shape_name == "Leaf":
        pts = []
        leaf_w_var = np.random.uniform(0.8, 1.2)
        for t in np.linspace(0, np.pi, 25):
            pts.append([int(center[0] - w + 2 * w * t / np.pi), int(center[1] - h * leaf_w_var * np.sin(t))])
        for t in np.linspace(0, np.pi, 25):
            pts.append([int(center[0] + w - 2 * w * t / np.pi), int(center[1] + h * leaf_w_var * np.sin(t))])
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts)], True, color, thickness)
        # Midrib (center vein)
        cv2.line(img, (center[0] - w, center[1]), (center[0] + w, center[1]), color, max(1, thickness - 1))
    
    elif shape_name == "Music note":
        # Note Variants
        variant = np.random.choice(["single", "beamed"])
        head_r = np.random.randint(w // 5, w // 3)
        
        if variant == "single":
            # Head (filled oval)
            head_x = center[0] - w // 3
            head_y = center[1] + h // 3
            cv2.ellipse(img, (head_x, head_y), (head_r, int(head_r * 0.7)), -20, 0, 360, color, -1)
            # Stem
            stem_x = head_x + head_r
            cv2.line(img, (stem_x, head_y), (stem_x, center[1] - h), color, thickness)
            # Flag
            cv2.line(img, (stem_x, center[1] - h), (stem_x + w // 3, center[1] - h // 2), color, thickness)
        else:
            # Two beamed notes
            head_x1 = center[0] - w // 2
            head_x2 = center[0] + w // 4
            head_y = center[1] + h // 3
            cv2.ellipse(img, (head_x1, head_y), (head_r, int(head_r * 0.7)), -20, 0, 360, color, -1)
            cv2.ellipse(img, (head_x2, head_y), (head_r, int(head_r * 0.7)), -20, 0, 360, color, -1)
            # Stems
            stem_x1 = head_x1 + head_r
            stem_x2 = head_x2 + head_r
            cv2.line(img, (stem_x1, head_y), (stem_x1, center[1] - h), color, thickness)
            cv2.line(img, (stem_x2, head_y), (stem_x2, center[1] - h), color, thickness)
            # Beam connecting tops
            cv2.line(img, (stem_x1, center[1] - h), (stem_x2, center[1] - h), color, thickness + 1)
    
    elif shape_name == "Smiley face":
        face_r = np.random.randint(w - 5, w + 5)
        cv2.circle(img, center, face_r, color, thickness)
        # Eyes
        eye_r = max(2, face_r // 8)
        eye_y = center[1] - face_r // 3
        cv2.circle(img, (center[0] - face_r // 3, eye_y), eye_r, color, -1)
        cv2.circle(img, (center[0] + face_r // 3, eye_y), eye_r, color, -1)
        # Smile arc
        smile_w = face_r // 2
        smile_h = np.random.randint(face_r // 5, face_r // 3)
        cv2.ellipse(img, (center[0], center[1] + face_r // 5), (smile_w, smile_h), 0, 0, 180, color, thickness)
    
    return img


def generate_synthetic_data(samples_per_class=500):
    """
    Generates synthetic shape data with heavy augmentation for training.
    Each class gets `samples_per_class` samples, each with random variations
    in position, size, thickness, rotation, scaling, noise, and distortion.
    """
    X, y = [], []
    size = 128
    
    total = len(SHAPES) * samples_per_class
    print(f"Generating {total} samples ({samples_per_class} per class x {len(SHAPES)} classes)...")
    
    for idx, shape_name in enumerate(SHAPES):
        print(f"  [{idx+1}/{len(SHAPES)}] Generating '{shape_name}'...")
        for sample_i in range(samples_per_class):
            img = draw_shape_on_canvas(shape_name, size)
            
            # Apply augmentation to ~70% of samples
            if np.random.random() < 0.7:
                img = augment_image(img)
            else:
                # Still add some noise
                noise = np.random.randint(0, 30, (size, size), dtype=np.uint8)
                img = cv2.add(img, noise)
            
            img = img.astype('float32') / 255.0
            X.append(np.expand_dims(img, axis=-1))
            y.append(idx)
    
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    return X, y


def train_model():
    """Train the CNN + Vision Transformer hybrid model."""
    print("=" * 60)
    print("  TRAINING CNN + VISION TRANSFORMER MODEL")
    print("  Classes:", SHAPES)
    print("=" * 60)
    
    print("\n[1/3] Generating synthetic training data...")
    X, y = generate_synthetic_data(500)
    print(f"  Data shape: X={X.shape}, y={y.shape}")
    
    print("\n[2/3] Creating CNN+ViT model...")
    model = create_cnn_vit_model(num_classes=len(SHAPES))
    model.summary()
    
    print("\n[3/3] Training model...")
    
    # Callbacks for better training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True, min_delta=0.001
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
    ]
    
    history = model.fit(
        X, y,
        epochs=50,
        batch_size=32,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )
    
    # Print final metrics
    val_acc = max(history.history.get('val_accuracy', [0]))
    train_acc = max(history.history.get('accuracy', [0]))
    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best Train Accuracy: {train_acc:.4f}")
    print(f"  Best Val Accuracy:   {val_acc:.4f}")
    print(f"{'=' * 60}")
    
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


def load_model():
    """Load the trained model or train a new one if not found."""
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(
                MODEL_PATH, 
                custom_objects={
                    'TransformerBlock': TransformerBlock,
                    'PatchEmbedding': PatchEmbedding,
                }
            )
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Retraining...")
            return train_model()
    else:
        return train_model()
