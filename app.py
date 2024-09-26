# from flask import Flask, request, render_template, jsonify
# from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

# # Initialize the Flask application
# application = Flask(__name__)
# app = application

# # Home route for the main page
# @app.route('/')
# def home_page():
#     return render_template('index.html')

# # Prediction route
# @app.route('/predict', methods=['GET', 'POST'])
# def predict_datapoint():
#     if request.method == 'GET':
#         return render_template('form.html')  # Show form for input
#     else:
#         # Collecting data from form inputs
#         data = CustomData(
#             limit_bal=float(request.form.get('LIMIT_BAL')),
#             sex=request.form.get('SEX'),
#             education=request.form.get('EDUCATION'),
#             marriage=request.form.get('MARRIAGE'),
#             age=int(request.form.get('AGE')),
#             pay_0=int(request.form.get('PAY_0')),
#             pay_2=int(request.form.get('PAY_2')),
#             pay_3=int(request.form.get('PAY_3')),
#             pay_4=int(request.form.get('PAY_4')),
#             pay_5=int(request.form.get('PAY_5')),
#             pay_6=int(request.form.get('PAY_6')),
#             bill_amt1=float(request.form.get('BILL_AMT1')),
#             bill_amt2=float(request.form.get('BILL_AMT2')),
#             bill_amt3=float(request.form.get('BILL_AMT3')),
#             bill_amt4=float(request.form.get('BILL_AMT4')),
#             bill_amt5=float(request.form.get('BILL_AMT5')),
#             bill_amt6=float(request.form.get('BILL_AMT6')),
#             pay_amt1=float(request.form.get('PAY_AMT1')),
#             pay_amt2=float(request.form.get('PAY_AMT2')),
#             pay_amt3=float(request.form.get('PAY_AMT3')),
#             pay_amt4=float(request.form.get('PAY_AMT4')),
#             pay_amt5=float(request.form.get('PAY_AMT5')),
#             pay_amt6=float(request.form.get('PAY_AMT6'))
#         )
        
#         # Convert collected data into a DataFrame
#         final_new_data = data.get_data_as_dataframe()

#         # Load the prediction pipeline
#         predict_pipeline = PredictPipeline()

#         # Make predictions
#         pred = predict_pipeline.predict(final_new_data)

#         # Round off the result for better presentation
#         results = round(pred[0], 2)

#         # Render the result on the results page
#         return render_template('results.html', final_result=results)

# # Start the Flask application
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', debug=True)

from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

# Initialize the Flask application
application = Flask(__name__)
app = application

# Home route for the main page
@app.route('/')
def home_page():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')  # Show form for input
    else:
        # Collecting data from form inputs
        data = CustomData(
            limit_bal=float(request.form.get('limit_bal', 0)),  # Provide default if missing
                sex=int(request.form.get('sex', 1)),  # Assuming '1' as a default for sex
                education=int(request.form.get('education', 1)),  # Default value if missing
                marriage=int(request.form.get('marriage', 1)),  # Default value if missing
                age=int(request.form.get('age', 0)),
                pay_0=int(request.form.get('pay_0', 0)),  # Default value
                pay_2=int(request.form.get('pay_2', 0)),
                pay_3=int(request.form.get('pay_3', 0)),
                pay_4=int(request.form.get('pay_4', 0)),
                pay_5=int(request.form.get('pay_5', 0)),
                pay_6=int(request.form.get('pay_6', 0)),
                bill_amt1=float(request.form.get('bill_amt1', 0)),
                bill_amt2=float(request.form.get('bill_amt2', 0)),
                bill_amt3=float(request.form.get('bill_amt3', 0)),
                bill_amt4=float(request.form.get('bill_amt4', 0)),
                bill_amt5=float(request.form.get('bill_amt5', 0)),
                bill_amt6=float(request.form.get('bill_amt6', 0)),
                pay_amt1=float(request.form.get('pay_amt1', 0)),
                pay_amt2=float(request.form.get('pay_amt2', 0)),
                pay_amt3=float(request.form.get('pay_amt3', 0)),
                pay_amt4=float(request.form.get('pay_amt4', 0)),
                pay_amt5=float(request.form.get('pay_amt5', 0)),
                pay_amt6=float(request.form.get('pay_amt6', 0))
            )
        
        # Convert collected data into a DataFrame
        final_new_data = data.get_data_as_dataframe()

        # Load the prediction pipeline
        predict_pipeline = PredictPipeline()

        # Make predictions
        pred = predict_pipeline.predict(final_new_data)

        # Round off the result for better presentation
        results = round(pred[0], 2)

        # Print prediction result for debugging purposes
        # print(f"Prediction result: {results}")

        # Render the result on the results page
        return render_template('results.html', final_result=results)

# Start the Flask application
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

    
