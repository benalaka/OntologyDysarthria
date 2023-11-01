# Import Module
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

my_model = "C:/Users/ochie/PycharmProjects/OntologyLearningPipeline_PhD/Log_PhD/data_subject/gradient_boosting_model.pkl"
my_model2 = "C:/Users/ochie/PycharmProjects/OntologyLearningPipeline_PhD/Log_PhD/data_subject/question_classifier_model.pkl"

loaded_gb = joblib.load(my_model)


# The code block below returns the dialogue act given a direct text
def get_DialogueAct_Text(text):
    # get prediction
    prediction = loaded_gb.predict(vectorizer.transform([text]))
    prediction_text = prediction[0]
    # print(read_text_file(file_path))
    print("Dialogue Act: " + prediction_text)
    return prediction_text


# In case we need to read this from a file (Which is the case) then use the code block below:
# 1: First we read the file
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        words = f.read()
        lines = words.split()
        number_of_words = len(lines)
        # print(words)
        # print(number_of_words)
        return words


# 2: Then we get the dialogue act
def get_DialogueAct_Audio(text):
    # get prediction
    prediction = loaded_gb.predict(vectorizer.transform([read_text_file(file)]))
    prediction_text = prediction[0]
    # print(read_text_file(file_path))
    print(prediction_text)


if __name__ == '__main__':
    get_DialogueAct_Text("Bye! See you later!")
