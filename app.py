from flask import Flask, request, render_template_string
import joblib
import numpy as np

# Load the scaler and model using joblib
try:
    scaler = joblib.load('scaler (2).pkl')
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

try:
    model = joblib.load('best_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = Flask(__name__)

# Define mappings for categorical variables
WHO_REGION_MAP = {'Africa': 1, 'Europe': 2, 'Americas': 3, 'Eastern Mediterranean': 4, 'Western Pacific': 5, 'South-East Asia': 6}
COUNTRY_MAP = {'India': 1, 'USA': 2, 'France': 3}  # Add all required countries here
RESIDENCE_AREA_TYPE_MAP = {'Urban': 1, 'Rural': 2}

# HTML Template with feature names
template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction App</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
<h1> Basic and safely managed drinking water predictor </h1>
 <div class="container">
    <h2>Enter data for prediction</h2>
    <form action="/" method="post">
        <label for="feature0">Year:</label>
        <input type="text" name="feature0" id="feature0" required><br>

        <label for="feature1">WHO region:</label>
        <input type="text" name="feature1" id="feature1" required><br>

        <label for="feature2">Country:</label>
        <input type="text" name="feature2" id="feature2" required><br>

        <label for="feature3">Residence Area Type:</label>
        <input type="text" name="feature3" id="feature3" required><br>

        <input type="submit" value="Predict">
    </form>

    {% if prediction is not none %}
    <h3>Prediction: {{ prediction }}</h3>
    
    {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    # Check if the scaler and model were loaded correctly
    if not scaler or not model:
        return "Error: Model or scaler not loaded correctly. Please check the files."

    prediction = None

    if request.method == "POST":
        try:
            # Gather input features
            year = int(request.form["feature0"])  # Year as integer

            # Map categorical inputs to numbers using predefined mappings
            who_region = WHO_REGION_MAP.get(request.form["feature1"], 0)
            country = COUNTRY_MAP.get(request.form["feature2"], 0)
            residence_area_type = RESIDENCE_AREA_TYPE_MAP.get(request.form["feature3"], 0)

            # Create a feature array with all inputs
            features = np.array([[year, who_region, country, residence_area_type]])

            # Scale and predict
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)[0]
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template_string(template, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)