# -*- coding: utf-8 -*-
"""Sentiment Analysis for Customer Audio Calls: A Novel Approach"""

# Install necessary libraries
# !pip install SpeechRecognition
# !pip install transformers
# !pip install librosa
# !pip install soundfile
# !pip install scikit-learn
# !pip install numpy

import os
import torch
import numpy as np
import librosa
import soundfile as sf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import speech_recognition as sr

# Step 1: Preprocess Audio
def preprocess_audio(audio_file):
    """Preprocess audio to enhance quality and ensure compatibility."""
    try:
        audio, sr_rate = librosa.load(audio_file, sr=16000)  # Resample to 16 kHz
        audio = librosa.effects.preemphasis(audio)  # Apply pre-emphasis filter
        sf.write("temp.wav", audio, sr_rate)  # Use soundfile to write WAV file
        return "temp.wav"
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

# Step 2: Convert Audio to Text
def convert_audio_to_text(preprocessed_file):
    """Convert preprocessed audio to text using Google Speech Recognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(preprocessed_file) as source:
        audio_data = recognizer.record(source)

    try:
        conversation_text = recognizer.recognize_google(audio_data)
        return conversation_text
    except sr.UnknownValueError:
        print(f"Google Speech Recognition could not understand audio in {preprocessed_file}.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

# Step 3: Text Preprocessing
def preprocess_text(text):
    """Split text into sentences and clean empty entries."""
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences

# Step 4: Sentiment Analysis (Using Pre-trained Model)
def analyze_sentiment(sentences, model, tokenizer):
    """Perform sentiment analysis using a pre-trained BERT model."""
    positive, negative, neutral = 0, 0, 0
    predicted_labels = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=-1)
        sentiment_score = torch.argmax(probs).item()

        # Map 5-class sentiment to 3 classes
        if sentiment_score in [4, 5]:  # Positive and Very Positive
            positive += 1
            predicted_labels.append(2)  # 2 for positive
        elif sentiment_score in [1, 2]:  # Very Negative and Negative
            negative += 1
            predicted_labels.append(0)  # 0 for negative
        else:  # Neutral
            neutral += 1
            predicted_labels.append(1)  # 1 for neutral

    return positive, negative, neutral, predicted_labels

# Step 5: Evaluation Metrics
def evaluate_performance(true_labels, predicted_labels):
    """Evaluate model performance using accuracy, F1 score, and confusion matrix."""
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    return accuracy, f1, conf_matrix

# Step 6: Calculate Percentages
def calculate_percentage(positive, negative, neutral):
    """Calculate percentage distribution of sentiments."""
    total = positive + negative + neutral
    if total == 0:
        return 0, 0, 0
    positive_percentage = (positive / total) * 100
    negative_percentage = (negative / total) * 100
    neutral_percentage = (neutral / total) * 100
    return positive_percentage, negative_percentage, neutral_percentage

# Main function to process a single audio file
def process_audio_file(audio_file, model, tokenizer, true_label=None):
    print(f"Processing file: {audio_file}")

    preprocessed_file = preprocess_audio(audio_file)
    if not preprocessed_file:
        print(f"Error preprocessing {audio_file}. Skipping.")
        return None, None

    conversation_text = convert_audio_to_text(preprocessed_file)
    if not conversation_text:
        print(f"No text extracted from {audio_file}.")
        return None, None

    sentences = preprocess_text(conversation_text)

    positive, negative, neutral, predicted_labels = analyze_sentiment(sentences, model, tokenizer)

    positive_percentage, negative_percentage, neutral_percentage = calculate_percentage(positive, negative, neutral)

    print(f"File: {audio_file}")
    print(f"Positive Sentiment: {positive_percentage:.2f}%")
    print(f"Negative Sentiment: {negative_percentage:.2f}%")
    print(f"Neutral Sentiment: {neutral_percentage:.2f}%")

    if positive_percentage > negative_percentage and positive_percentage > neutral_percentage:
        print("The call is classified as Positive in terms of customer satisfaction.")
    elif negative_percentage > positive_percentage and negative_percentage > neutral_percentage:
        print("The call is classified as Negative in terms of customer satisfaction.")
    else:
        print("The call is classified as Neutral in terms of customer satisfaction.")
    print("\n")

    return predicted_labels, [true_label] * len(predicted_labels) if true_label is not None else None

# Main function to process multiple audio files
if __name__ == "__main__":
    # Specify the directory containing your audio files
    audio_directory = "/home/mohsn/Audio"

    # List all audio files in the directory
    audio_files = [os.path.join(audio_directory, f) for f in os.listdir(audio_directory) if f.endswith(('.mp3', '.wav'))]

    # Example true labels for evaluation (replace with actual labels)
    true_labels = [0, 2, 1, 2, 1, 0, 1]  # 0: Negative, 1: Neutral, 2: Positive

    # Load pre-trained BERT model and tokenizer
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    # Process each audio file
    all_predicted_labels = []
    all_true_labels = []
    for i, audio_file in enumerate(audio_files):
        predicted_labels, true_label = process_audio_file(audio_file, model, tokenizer, true_labels[i] if i < len(true_labels) else None)
        if predicted_labels and true_label:
            all_predicted_labels.extend(predicted_labels)
            all_true_labels.extend(true_label)

    # Evaluate overall performance
    if all_predicted_labels and all_true_labels:
        accuracy, f1, conf_matrix = evaluate_performance(all_true_labels, all_predicted_labels)
        print(f"Overall Accuracy: {accuracy:.2f}")
        print(f"Overall F1 Score: {f1:.2f}")
        print("Confusion Matrix:")
        print(conf_matrix)