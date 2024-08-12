import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Concatenate
from tensorflow.keras.models import Model

# Path to image and captions directories
IMAGE_DIR = 'images/'
CAPTIONS_FILE = 'Figure 1.txt'

# Load the InceptionV3 model
def build_feature_extractor():
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    return base_model

# Extract image features
def extract_features(image_path, model):
    img = Image.open(image_path).resize((299, 299))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    features = model.predict(img_array)
    return features

# Load captions from file
def load_captions(captions_file):
    captions = {}
    with open(captions_file, 'r') as f:
        for line in f:
            image, caption = line.strip().split('\t')
            if image not in captions:
                captions[image] = []
            captions[image].append(caption)
    return captions

# Prepare data for training
def prepare_data(images_dir, captions_file):
    model = build_feature_extractor()
    captions = load_captions(captions_file)
    all_captions = []
    image_features = []
    image_paths = []

    for img_file, img_captions in captions.items():
        img_path = os.path.join(images_dir, img_file)
        features = extract_features(img_path, model)
        image_features.append(features)
        image_paths.append(img_path)
        for caption in img_captions:
            all_captions.append(caption)
    
    return np.array(image_features), all_captions

# Tokenize and pad captions
def preprocess_captions(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    sequences = tokenizer.texts_to_sequences(captions)
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, padded_sequences, vocab_size, max_length

# Define the model
def build_captioning_model(vocab_size, max_length):
    image_features_input = Input(shape=(2048,))
    image_features_dense = Dense(256, activation='relu')(image_features_input)
    
    caption_input = Input(shape=(max_length,))
    caption_embedding = Embedding(vocab_size, 256, mask_zero=True)(caption_input)
    caption_lstm = LSTM(256)(caption_embedding)
    
    combined = Concatenate()([image_features_dense, caption_lstm])
    combined_dense = Dense(256, activation='relu')(combined)
    output = Dense(vocab_size, activation='softmax')(combined_dense)
    
    model = Model(inputs=[image_features_input, caption_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

# Train the model
def train_model(image_features, captions, tokenizer, model):
    captions_sequences = tokenizer.texts_to_sequences(captions)
    max_length = max(len(seq) for seq in captions_sequences)
    padded_sequences = pad_sequences(captions_sequences, maxlen=max_length, padding='post')
    vocab_size = len(tokenizer.word_index) + 1
    X1, X2, y = [], [], []

    for i, img_features in enumerate(image_features):
        img_features = np.repeat(img_features, len(captions_sequences[i]), axis=0)
        seqs = padded_sequences[i]
        for j in range(1, len(seqs)):
            in_seq, out_seq = seqs[:j], seqs[j]
            in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')
            out_seq = to_categorical([out_seq], num_classes=vocab_size)
            X1.append(img_features)
            X2.append(in_seq)
            y.append(out_seq)

    X1, X2, y = np.array(X1), np.array(X2), np.array(y)
    model.fit([X1, X2], y, epochs=10, batch_size=32, verbose=1)

# Generate captions for new images
def generate_caption(image_path, model, tokenizer, max_length):
    features = extract_features(image_path, build_feature_extractor())
    features = np.expand_dims(features, axis=0)
    
    caption = '<start>'
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = pad_sequences([seq], maxlen=max_length, padding='post')
        y_pred = model.predict([features, seq])
        word = tokenizer.index_word[np.argmax(y_pred[0])]
        if word == '<end>':
            break
        caption += ' ' + word
        if len(caption.split()) > max_length:
            break
    return caption

# Main function
if __name__ == "__main__":
    image_features, captions = prepare_data(IMAGE_DIR, CAPTIONS_FILE)
    tokenizer, padded_sequences, vocab_size, max_length = preprocess_captions(captions)
    model = build_captioning_model(vocab_size, max_length)
    train_model(image_features, captions, tokenizer, model)
    print(generate_caption('path/to/new/image.jpg', model, tokenizer, max_length))
