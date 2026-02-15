# AI-Air-Drawing-Shape-Detection

 
model.py
 — Complete Rewrite
Removed all 10 geometrical shapes: Circle, Square, Rectangle, Triangle, Pentagon, Hexagon, Heptagon, Octagon, Star, Ellipse
Kept only 12 creative shapes: Spiral, Infinity, Cloud, Lightning bolt, Flower, Butterfly, Crown, Flame, Fish, Leaf, Music note, Smiley face
Enhanced CNN architecture: 4 convolutional blocks (was 3) with BatchNormalization for better training stability
Added 
PatchEmbedding
 layer: Learned positional encoding for the Transformer — boosts spatial awareness
3 Transformer blocks (was 2) for deeper global attention
Richer classification head: 256 → 128 → 12 with LayerNorm and Dropout
Cosine learning rate decay for smoother convergence
Heavy data augmentation: Random rotation (±20°), scaling (0.85–1.15x), translation (±8px), noise, dilation/erosion
500 samples per class (was 300) with diverse shape rendering (random turns, sizes, bump counts, etc.)
Early stopping + ReduceLROnPlateau callbacks for optimal training
2. 
utils.py
 — Updated
Removed all geometrical shape references from 
heuristic_classify
 and 
draw_perfect_shape
Heuristic classifier now analyzes circularity, solidity, aspect ratio, and path openness to classify the 12 creative shapes
Added padding to ROI preprocessing for better shape isolation
3. 
app.py
 — Updated
Title changed to "AI Air Canvas — Creative Shape Detector"
Added emoji mapping for all 12 shapes (🌀 Spiral, ♾️ Infinity, ☁️ Cloud, ⚡ Lightning bolt, etc.)
Added "Supported Shapes" sidebar section showing all 12 shapes
📊 Training Results
Metric	Value
Training Accuracy	~99.0%
Validation Accuracy	99.44%
Epochs trained	18 (early stopped at ~epoch 18)
Total parameters	1.68M
Model file	shape_model.h5 (19.5 MB)
You can run the app with streamlit run app.py and it will detect all 12 creative shapes!

Good
