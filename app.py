from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    beer = float(request.form["beer"])
    spirit = float(request.form["spirit"])
    wine = float(request.form["wine"])
    country = request.form["country"]
    continent = request.form["continent"]

    input_data = pd.DataFrame({
        "country": [country],
        "beer_servings": [beer],
        "spirit_servings": [spirit],
        "wine_servings": [wine],
        "continent": [continent]
    })

    prediction = model.predict(input_data)

    return render_template(
        "index.html",
        prediction=round(prediction[0], 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
    