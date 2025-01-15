# Sentiment Analysis for Customer Audio Calls
Introduction

This report provides a detailed explanation of the code designed to perform sentiment analysis on customer audio calls. The code leverages speech recognition, natural language processing (NLP), and machine learning techniques to analyze the sentiment of conversations in audio files. The goal is to classify customer satisfaction as positive, negative, or neutral based on the content of the audio calls.
Code Overview

### The code is structured into several key steps, each responsible for a specific part of the sentiment analysis pipeline:

Audio Preprocessing

Speech-to-Text Conversion

Text Preprocessing

Sentiment Analysis

Evaluation Metrics

Percentage Calculation

Main Processing Functions

### 1. Audio Preprocessing

The preprocess_audio function is responsible for preparing the audio file for further processing. It performs the following tasks:

Resampling: The audio is resampled to 16 kHz to ensure compatibility with the speech recognition system.

Pre-emphasis Filter: A pre-emphasis filter is applied to enhance the quality of the audio.

Temporary File Creation: The processed audio is saved as a temporary WAV file using the soundfile library.
```python
def preprocess_audio(audio_file):
    audio, sr_rate = librosa.load(audio_file, sr=16000)
    audio = librosa.effects.preemphasis(audio)
    sf.write("temp.wav", audio, sr_rate)
    return "temp.wav"
```
### 2. Speech-to-Text Conversion

The convert_audio_to_text function converts the preprocessed audio file into text using Google Speech Recognition. This step is crucial as it transforms the audio content into a format that can be analyzed by NLP models.
```python
def convert_audio_to_text(preprocessed_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(preprocessed_file) as source:
        audio_data = recognizer.record(source)
    conversation_text = recognizer.recognize_google(audio_data)
    return conversation_text
```
### 3. Text Preprocessing

The preprocess_text function splits the extracted text into individual sentences and removes any empty entries. This step ensures that the text is in a suitable format for sentiment analysis.
```python
def preprocess_text(text):
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences
```
### 4. Sentiment Analysis
The analyze_sentiment function uses a pre-trained BERT model ( nlptown/bert-base-multilingual-uncased-sentiment ) to classify the sentiment of each sentence. The model outputs a sentiment score, which is then mapped to one of three classes: positive, negative, or neutral.
```python
def analyze_sentiment(sentences, model, tokenizer):
    positive, negative, neutral = 0, 0, 0
    predicted_labels = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=-1)
        sentiment_score = torch.argmax(probs).item()
        if sentiment_score in [4, 5]:  # Positive
            positive += 1
            predicted_labels.append(2)
        elif sentiment_score in [1, 2]:  # Negative
            negative += 1
            predicted_labels.append(0)
        else:  # Neutral
            neutral += 1
            predicted_labels.append(1)
    return positive, negative, neutral, predicted_labels
```
### 5. Evaluation Metrics

The evaluate_performance function calculates the accuracy, F1 score, and confusion matrix to evaluate the performance of the sentiment analysis model.
```python
def evaluate_performance(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    return accuracy, f1, conf_matrix
```
### 6. Percentage Calculation

The calculate_percentage function computes the percentage distribution of positive, negative, and neutral sentiments in the conversation.
```python
def calculate_percentage(positive, negative, neutral):
    total = positive + negative + neutral
    positive_percentage = (positive / total) * 100
    negative_percentage = (negative / total) * 100
    neutral_percentage = (neutral / total) * 100
    return positive_percentage, negative_percentage, neutral_percentage
```
### 7. Main Processing Functions

The process_audio_file function orchestrates the entire pipeline for a single audio file, while the main script processes multiple audio files and evaluates the overall performance.
```python
def process_audio_file(audio_file, model, tokenizer, true_label=None):
    # Orchestrates the pipeline for a single audio file
    pass

if __name__ == "__main__":
    # Processes multiple audio files and evaluates overall performance
    pass
```
### Key Libraries and Tools

SpeechRecognition: Used for converting audio to text.

Librosa: Handles audio preprocessing tasks.

Transformers: Provides access to pre-trained BERT models for sentiment analysis.

Soundfile: Manages audio file I/O operations.

Scikit-learn: Used for evaluating model performance (accuracy, F1 score, confusion matrix).

### Conclusion

The code provides a robust pipeline for analyzing customer sentiment from audio calls. It combines audio processing, speech-to-text conversion, and NLP techniques to classify sentiment into positive, negative, or neutral categories. The use of a pre-trained BERT model ensures high accuracy in sentiment classification, while the evaluation metrics provide insights into the model's performance.

### Future Work

Improve Speech Recognition: Explore more accurate speech recognition models or services.

Fine-tune BERT Model: Fine-tune the BERT model on domain-specific data to improve sentiment classification.

Real-time Analysis: Extend the pipeline to support real-time sentiment analysis for live customer calls.

Enhanced Evaluation: Incorporate additional evaluation metrics and cross-validation techniques for a more comprehensive performance assessment.
    
