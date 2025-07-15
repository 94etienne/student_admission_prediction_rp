from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and artifacts
model = joblib.load('model/final_best_model.joblib')
label_encoders = joblib.load('model/label_encoders.joblib')
program_map = joblib.load('model/program_mapping.joblib')
subject_map = joblib.load('model/subject_mapping.joblib')

def check_admission_eligibility(combination, year, scores, fee_paid, is_tvet=False):
    if not fee_paid:
        return False, []
    
    if is_tvet:
        meets_req = all(score >= 50 for score in scores)
    else:
        if year < 2024:
            principal_passes = sum(score >= 50 for score in scores)
            meets_req = principal_passes >= 2
        else:
            meets_req = all(score >= 50 for score in scores)
    
    if not meets_req:
        return False, []
    
    return True, program_map.get(combination, [])

def prepare_prediction_data(student_data):
    student_df_data = {
        'combination': student_data['combination'],
        'completed_year': student_data['completed_year'],
        'has_trade_skills': student_data['has_trade_skills'],
        'application_fee_paid': student_data['application_fee_paid'],
        'program_choice': student_data['program_choice'],
        'is_tvet': student_data.get('is_tvet', 0)
    }
    
    max_subjects = 10
    for i in range(1, max_subjects + 1):
        student_df_data[f'subject{i}'] = 'None'
        student_df_data[f'subject{i}_score'] = 0
    
    for i, (subject, score) in enumerate(student_data['subject_scores']):
        student_df_data[f'subject{i+1}'] = subject
        student_df_data[f'subject{i+1}_score'] = score
    
    student_df = pd.DataFrame([student_df_data])
    
    for col in label_encoders:
        if col in student_df.columns:
            try:
                student_df[col] = label_encoders[col].transform(student_df[col])
            except ValueError:
                label_encoders[col].classes_ = np.append(
                    label_encoders[col].classes_, 'Unknown'
                )
                student_df[col] = label_encoders[col].transform(student_df[col])
    
    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
    if model_features is not None:
        for feature in model_features:
            if feature not in student_df.columns:
                if '_score' in feature:
                    student_df[feature] = 0
                else:
                    student_df[feature] = 'Unknown'
                    student_df[feature] = label_encoders[feature].transform(student_df[feature])
        student_df = student_df[model_features]
    
    return student_df

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    is_tvet = data.get('is_tvet', 0)
    combination = data['combination']
    subjects = [sub for sub, score in data['subject_scores']]
    scores = [score for sub, score in data['subject_scores']]
    
    # Check basic eligibility
    is_eligible, recommended = check_admission_eligibility(
        combination,
        data['completed_year'],
        scores,
        data['application_fee_paid'],
        is_tvet=is_tvet
    )
    
    if not is_eligible:
        return jsonify({
            'admission_status': 'Not Admitted',
            'recommended_programs': [],
            'subject_names': subjects,
            'scores': scores,
            'message': 'Does not meet minimum academic requirements'
        })
    
    # Prepare data for prediction
    student_df = prepare_prediction_data(data)
    
    # Get prediction
    pred = model.predict(student_df)[0]
    status = 'Admitted' if pred == 1 else 'Not Admitted'
    
    return jsonify({
        'admission_status': status,
        'recommended_programs': recommended,
        'subject_names': subjects,
        'scores': scores,
        'message': 'Meets academic requirements' if status == 'Admitted' 
                  else 'Model prediction: Not admitted'
    })

@app.route('/get_combinations', methods=['GET'])
def get_combinations():
    # REB combinations (all except RTB ones)
    reb_combinations = [comb for comb in program_map.keys() if comb not in [
        'ACCOUNTING','LSV', 'CET', 'EET', 'MET', 'CP','SoD','AH','MAS',
        'WOT','FOR','TOR','FOH','MMP','SPE','IND','MPA','NIT','PLT','ETL'
    ]]
    rtb_combinations = [
        'ACCOUNTING','LSV', 'CET', 'EET', 'MET', 'CP','SoD','AH','MAS',
        'WOT','FOR','TOR','FOH','MMP','SPE','IND','MPA','NIT','PLT','ETL'
    ]
    
    return jsonify({
        'reb_combinations': reb_combinations,
        'rtb_combinations': rtb_combinations
    })

@app.route('/get_subjects/<combination>', methods=['GET'])
def get_subjects(combination):
    subjects = subject_map.get(combination, [])
    return jsonify({'subjects': subjects})

@app.route('/get_programs/<combination>', methods=['GET'])
def get_programs(combination):
    programs = program_map.get(combination, [])
    return jsonify({'programs': programs})

if __name__ == '__main__':
    app.run(debug=True)