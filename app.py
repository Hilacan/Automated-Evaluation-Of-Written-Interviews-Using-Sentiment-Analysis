from flask import Flask,render_template,request
from model import handler as h

app = Flask(__name__)



@app.route("/")
def home():
    return render_template("index.html")
 
@app.route("/predict",methods =["POST"])
def result():
    if request.method == "POST":
        ans = request.form['response']
        val = h.predict(ans)
        print(val)
        sentiment = h.label(h.predict(ans))
        print(sentiment)
    return render_template("index.html",eval = sentiment,response = ans,score = val)

if __name__ == '__main__':
    app.run(debug=True)




