from flask import Flask,render_template,request
import numpy as np
#import lm


app = Flask(__name__)


@app.route("/", methods =["GET","POST"])
def home():
    # if request.method == "POST":
    #     answer = request.form["answer"]
    #     result = lm.prediction(answer)
    #     print(result)
    return render_template("index.html")
 


# @app.route("/result",methods =["POST"])
# def result():
#     if request.method == "POST":
#         ans = request.form["answer"]
#     return render_template("result.html",result = ans)

if __name__ == '__main__':
    app.run(debug=True)




