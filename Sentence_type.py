# Import Module
import os
import shutil
import joblib
import nltk
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

nltk.download('nps_chat')
posts = nltk.corpus.nps_chat.xml_posts()

posts_text = [post.text for post in posts]

# divide train and test in 80 20
train_text = posts_text[:int(len(posts_text) * 0.8)]
test_text = posts_text[int(len(posts_text) * 0.2):]

# Get TFIDF features
vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                             min_df=0.001,
                             max_df=0.7,
                             analyzer='word')

X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

y = [post.get('class') for post in posts]

y_train = y[:int(len(posts_text) * 0.8)]
y_test = y[int(len(posts_text) * 0.2):]

path_audio = "Log_PhD/wav_dataset"
new_path_audio = "Log_PhD/wav_dataset"
path_prompts = "Log_PhD/prompts_dataset"
old_prompts = "Log_PhD/prompts_dataset"
path_subject = "Log_PhD/data_subject"
CSV_SUBJECT = "C:/Users/ochie/PycharmProjects/OntologyLearningPipeline_PhD/Log_PhD/data_subject/subject.csv"
my_model = "C:/Users/ochie/PycharmProjects/OntologyLearningPipeline_PhD/Log_PhD/data_subject/gradient_boosting_model.pkl"
my_model2 = "C:/Users/ochie/PycharmProjects/OntologyLearningPipeline_PhD/Log_PhD/data_subject/question_classifier_model.pkl"

question_pattern = ["do i", "do you", "what", "who", "is it", "why", "would you", "how", "is there",
                    "are there", "is it so", "is this true", "to know", "is that true", "are we", "am i",
                    "question is", "tell me more", "can i", "can we", "tell me", "can you explain",
                    "question", "answer", "questions", "answers", "ask"]

helping_verbs = ["is", "am", "can", "are", "do", "does"]


def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features


question_types = ["whQuestion", "ynQuestion"]


def is_ques_using_nltk(ques):
    # Load the saved model from the file
    loaded_classifier = joblib.load(my_model2)
    question_type = loaded_classifier.classify(dialogue_act_features(ques))
    return question_type in question_types


# check with custom pipeline if still this is a question mark it as a question
def is_question(question):
    question = question.lower().strip()
    if not is_ques_using_nltk(question):
        is_ques = False
        # check if any of pattern exist in sentence
        for pattern in question_pattern:
            is_ques = pattern in question
            if is_ques:
                break

        # there could be multiple sentences so divide the sentence
        sentence_arr = question.split(".")
        for sentence in sentence_arr:
            if len(sentence.strip()):
                # if question ends with ? or start with any helping verb
                # word_tokenize will strip by default
                first_word = nltk.word_tokenize(sentence)[0]
                if sentence.endswith("?") or first_word in helping_verbs:
                    is_ques = True
                    break
        return is_ques
    else:
        return True


# Folder Path
path = path_prompts
path2 = path_subject


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        words = f.read()
        lines = words.split()
        number_of_words = len(lines)
        # print(words)
        # print(number_of_words)
        return words


def read_text_file_num(file_path):
    with open(file_path, 'r') as f:
        words = f.read()
        lines = words.split()
        number_of_words = len(lines)
        # print(words)
        # print(number_of_words)
        return number_of_words


import csv

# Change the directory
os.chdir(path_prompts)

# adding header to csv file
headerList = ["id", "paht", "subject"]
with open(CSV_SUBJECT, 'w', newline='') as f:
    writer = csv.writer(f)
    writerObj = csv.writer(f)
    writerObj.writerow(headerList)

loaded_gb = joblib.load(my_model)
# iterate through all file
count = 0
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{path_prompts}/{file}"
        full_filename = file_path.split('.')
        filename = full_filename[0] + ".wav"

        # print(filename)
        source_path = f"{new_path_audio}/{os.path.basename(filename)}"

        # Extract filename as id
        fname_id = (os.path.basename(source_path).split('.'))
        id = fname_id[0];

        try:
            # get prediction
            prediction = loaded_gb.predict(vectorizer.transform([read_text_file(file)]))
            prediction_text = prediction[0]
            # print(read_text_file(file_path))
            print(id, source_path, prediction_text)

            # write to csv
            with open(CSV_SUBJECT, 'a', newline='') as f:
                writer = csv.writer(f, dialect='excel')
                writer.writerow([id, source_path, prediction_text])

        except IOError:
            # print("File not found: {}".format(file_path))
            pass


def get_DialogueAct(text):
    # get prediction
    prediction = loaded_gb.predict(vectorizer.transform([read_text_file(file)]))
    prediction_text = prediction[0]
    # print(read_text_file(file_path))
    print(prediction_text)


if __name__ == '__main__':
    get_DialogueAct("Really! How amazing it is!")
