from flask import Flask, request,render_template,jsonify
from employee.pipeline.prediction_pipeline import CustomData,PredictPipeline


application = Flask(__name__)

app = application

@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])


def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')

    else:
        data = CustomData(
            department = request.form.get('department'),
            education = request.form.get('education'),
            gender = request.form.get('gender'),
            no_of_trainings = int(request.form.get('no_of_trainings')),
            age = int(request.form.get('age')),
            previous_year_rating = float(request.form.get('previous_year_rating')),
            length_of_service = int(request.form.get('length_of_service')),
            kpi_80 = int(request.form.get('kpi_80')),
            award_won = int(request.form.get('award_won')),
            avg_training_score = int(request.form.get('avg_training_score')),
            sum_metric = float(request.form.get('sum_metric')),
            total_score = int(request.form.get('total_score'))
       
        )

       
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)
        
        result = int(pred[0])
        
        return render_template('form.html',final_result=result)
        


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
