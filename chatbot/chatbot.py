import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents JSON file
intents = json.loads(open('H:\\python_projects\\intents.json').read())

# Load words and classes from pickle files
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pickle', 'rb'))

# Load pre-trained model
model = load_model("chatbot_By_Himanshu.h5")

# Function to clean up sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Function to create bag of words from sentence
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict class of sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

# Function to get response based on predicted intents
def get_response(intents_list, intents_json):
    list_of_intents = intents_json['intents']
    tag = intents_list[0]['intent']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    # Default response if no matching intents are found
    return "I'm sorry, I don't understand that."

print("Bot is running!!")

# Main loop for interactive chat
while True:
    message = input("")
    predicted_intents = predict_class(message)
    response = get_response(predicted_intents, intents)
    print(response)
