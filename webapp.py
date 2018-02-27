from flask import Flask, render_template, request
from topic import TopicAnalyzer



app = Flask(__name__)

@app.route("/training_size")
def training_size():
    a = TopicAnalyzer()
    precision_scores = []
    recall_scores = []
    f1_scores = []
    labels = []
    Title = "Number of tweets - Precision/Recall/F1"
    Description = 'Keeping topic consistant, chart displays cross-validation scores based on variance in size of training data.'
    for i in range(10,200, 20):
        precision, recall, f1 = ["%.2f" % (x * 100) for x in a.classification_scores("climbing", n_tweets = i)]
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        labels.append(i)
    return render_template("classification.html", precision_scores = precision_scores, graph_labels = labels, recall_scores = recall_scores, f1_scores = f1_scores, Title= Title, Description=Description)

@app.route("/tweet_topic")
def tweet_topic():
    a = TopicAnalyzer()
    precision_scores = []
    recall_scores = []
    f1_scores = []
    labels = ["Climbing", "Programming", "Politics", "Winter"]
    Title = "Tweet Searches - Precision/Recall/F1"
    Description = "Varitying topic, chart displays cross-validation scores"
    for i in labels:
        precision, recall, f1 = ["%.2f" % (x * 100) for x in a.classification_scores(i, n_tweets = 300)]
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    return render_template("classification.html", precision_scores = precision_scores, graph_labels = labels, recall_scores = recall_scores, f1_scores = f1_scores, Title=Title, Description=Description)

@app.route("/predict")
def predict_word():
    a = TopicAnalyzer()
    Title = "Interactive Perceptron Classifcation"
    Description = "Choose a topic.  Enter a string to test if it belongs in topic.  Training size is 300 tweets"
    if request.args.get("topic"):
        clf = a.binary_topic(request.args.get("topic"), request.args.get("topic2"),  n_tweets = 2000)
        clf.train()
        result = clf.predict([request.args.get("tweet")])
        if result[0] == True:
            result = "This is a tweet about %s" %(request.args.get("topic"))
        else:
            result = "This is a tweet about %s" %(request.args.get("topic2"))
    else:
        result = ""

    return render_template("predict.html", Title=Title, Description=Description, result = result)



if __name__ == "__main__":
    app.debug = True
    app.run()