from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    age = float(request.form["age"])
    is_female = int(request.form["is_female"])
    bmi = float(request.form["bmi"])
    children = int(request.form["children"])
    is_smoker = int(request.form["is_smoker"])
    region_southeast = int(request.form["region_southeast"])
    region_northwest = int(request.form["region_northwest"])
    bmi_category_obese = int(request.form["bmi_category_obese"])

    features = np.array([[
        age,
        is_female,
        bmi,
        children,
        is_smoker,
        region_southeast,
        region_northwest,
        bmi_category_obese
    ]])

    prediction = model.predict(features)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Insurance Charges: ₹ {prediction[0]:,.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)