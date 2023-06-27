from flask import Flask, request,render_template,jsonify
from employee.pipeline.prediction_pipeline import CustomData,PredictPipeline

from flask import Flask, request, render_template, jsonify
from employee.pipeline.prediction_pipeline import CustomData, PredictPipeline
import os
from werkzeug.utils import secure_filename

from Prediction.batch import batch_prediction
from employee.logger import logging
from employee.components.data_transformation import DataTransformationConfig
from employee.config.configuration import MODEL_FILE_PATH, FEATURE_ENG_OBJ_PATH, PREPROCESSING_OBJ_PATH
from employee.pipeline.training_pipeline import Train

feature_engineering_file_path = FEATURE_ENG_OBJ_PATH
transformer_file_path = PREPROCESSING_OBJ_PATH
model_file_path = MODEL_FILE_PATH

UPLOAD_FOLDER = 'batch_prediction/Uploaded_CSV_FILE'



application = Flask(__name__,template_folder='templates')

app = application
ALLOWED_EXTENSIONS = {'csv'}

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
    
@app.route("/batch", methods=['GET','POST'])
def perform_batch_prediction():
    
    
    if request.method == 'GET':
        return render_template('batch.html')
    else:
        file = request.files['csv_file']  # Update the key to 'csv_file'
        # Directory path
        directory_path = UPLOAD_FOLDER
        # Create the directory
        os.makedirs(directory_path, exist_ok=True)

        # Check if the file has a valid extension
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            # Delete all files in the file path
            for filename in os.listdir(os.path.join(UPLOAD_FOLDER)):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # Save the new file to the uploads directory
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            print(file_path)

            logging.info("CSV received and Uploaded")

            # Perform batch prediction using the uploaded file
            batch = batch_prediction(file_path,
                                    model_file_path,
                                    transformer_file_path,
                                    feature_engineering_file_path)
            batch.start_batch_prediction()

            output = "Batch Prediction Done"
            return render_template("batch.html", prediction_result=output, prediction_type='batch')
        else:
            return render_template('batch.html', prediction_type='batch', error='Invalid file type')
        


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        return render_template('train.html')
    else:
        try:
            pipeline = Train()
            pipeline.main()

            return render_template('train.html', message="Training complete")

        except Exception as e:
            logging.error(f"{e}")
            error_message = str(e)
            return render_template('index.html', error=error_message)
        


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=8080)
