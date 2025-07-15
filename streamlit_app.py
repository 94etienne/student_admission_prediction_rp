import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import base64
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Student Admission Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
    }
    
    .step {
        flex: 1;
        text-align: center;
        padding: 10px;
        border-radius: 5px;
        margin: 0 5px;
    }
    
    .step.active {
        background: #6a11cb;
        color: white;
    }
    
    .step.completed {
        background: #2ecc71;
        color: white;
    }
    
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .admitted {
        background: rgba(46, 204, 113, 0.1);
        border-left: 5px solid #2ecc71;
    }
    
    .not-admitted {
        background: rgba(231, 76, 60, 0.1);
        border-left: 5px solid #e74c3c;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'form_type' not in st.session_state:
    st.session_state.form_type = 'REB'

# Load models and data
@st.cache_resource
def load_models():
    try:
        model = joblib.load('model/final_best_model.joblib')
        label_encoders = joblib.load('model/label_encoders.joblib')
        program_map = joblib.load('model/program_mapping.joblib')
        subject_map = joblib.load('model/subject_mapping.joblib')
        return model, label_encoders, program_map, subject_map
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

model, label_encoders, program_map, subject_map = load_models()

# Utility functions
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

def get_combinations():
    if program_map is None:
        return [], []
    
    rtb_combinations = [
        'ACCOUNTING','LSV', 'CET', 'EET', 'MET', 'CP','SoD','AH','MAS',
        'WOT','FOR','TOR','FOH','MMP','SPE','IND','MPA','NIT','PLT','ETL'
    ]
    
    reb_combinations = [comb for comb in program_map.keys() if comb not in rtb_combinations]
    
    return reb_combinations, rtb_combinations

def generate_pdf_report(student_data, prediction_result):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#6a11cb')
    )
    story.append(Paragraph("Student Admission Prediction Report", title_style))
    story.append(Spacer(1, 20))
    
    # Student Information
    story.append(Paragraph("Student Information", styles['Heading2']))
    student_info = [
        ['National ID:', student_data.get('nid', 'N/A')],
        ['Name:', f"{student_data.get('fname', '')} {student_data.get('lname', '')}"],
        ['Email:', student_data.get('email', 'N/A')],
        ['Phone:', student_data.get('phone', 'N/A')],
        ['Form Type:', student_data.get('form_type', 'N/A')]
    ]
    
    student_table = Table(student_info, colWidths=[2*inch, 3*inch])
    student_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(student_table)
    story.append(Spacer(1, 20))
    
    # Prediction Result
    story.append(Paragraph("Prediction Result", styles['Heading2']))
    result_color = colors.green if prediction_result['admission_status'] == 'Admitted' else colors.red
    status_style = ParagraphStyle(
        'StatusStyle',
        parent=styles['Normal'],
        fontSize=16,
        textColor=result_color,
        spaceAfter=10
    )
    story.append(Paragraph(f"Status: {prediction_result['admission_status']}", status_style))
    story.append(Paragraph(f"Message: {prediction_result['message']}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Subject Performance
    story.append(Paragraph("Subject Performance", styles['Heading2']))
    subject_data = [['Subject', 'Score']]
    for i, subject in enumerate(prediction_result['subject_names']):
        subject_data.append([subject, f"{prediction_result['scores'][i]}%"])
    
    subject_table = Table(subject_data, colWidths=[3*inch, 2*inch])
    subject_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(subject_table)
    story.append(Spacer(1, 20))
    
    # Recommended Programs
    if prediction_result['recommended_programs']:
        story.append(Paragraph("Recommended Programs", styles['Heading2']))
        for i, program in enumerate(prediction_result['recommended_programs'], 1):
            story.append(Paragraph(f"{i}. {program}", styles['Normal']))
    
    # Footer
    story.append(Spacer(1, 30))
    footer_style = ParagraphStyle(
        'FooterStyle',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.grey
    )
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))
    story.append(Paragraph("Student Admission Prediction System", footer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Student Admission Prediction System</h1>
        <p>Predict admission status based on academic performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if models are loaded
    if model is None:
        st.error("Failed to load prediction models. Please check if model files exist.")
        return
    
    # Form type selection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        form_type = st.selectbox(
            "Select Form Type",
            ["REB Prediction", "RTB Prediction"],
            key="form_type_select"
        )
        st.session_state.form_type = form_type.split()[0]
    
    # Step indicator
    steps = ["Personal Info", "Academic Info", "Additional Info", "Results"]
    current_step = st.session_state.current_step
    
    step_html = '<div class="step-indicator">'
    for i, step in enumerate(steps, 1):
        if i < current_step:
            step_html += f'<div class="step completed">{i}. {step}</div>'
        elif i == current_step:
            step_html += f'<div class="step active">{i}. {step}</div>'
        else:
            step_html += f'<div class="step">{i}. {step}</div>'
    step_html += '</div>'
    
    st.markdown(step_html, unsafe_allow_html=True)
    
    # Step 1: Personal Information
    if current_step == 1:
        st.subheader("📋 Personal Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            nid = st.text_input("National ID (NID)", key="nid")
            fname = st.text_input("First Name", key="fname")
            email = st.text_input("Email", key="email")
        
        with col2:
            lname = st.text_input("Last Name", key="lname")
            phone = st.text_input("Phone Number", key="phone")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Next Step", key="step1_next"):
                if all([nid, fname, lname, email, phone]):
                    st.session_state.form_data.update({
                        'nid': nid,
                        'fname': fname,
                        'lname': lname,
                        'email': email,
                        'phone': phone,
                        'form_type': st.session_state.form_type
                    })
                    st.session_state.current_step = 2
                    st.rerun()
                else:
                    st.error("Please fill in all required fields")
    
    # Step 2: Academic Information
    elif current_step == 2:
        st.subheader("📚 Academic Information")
        
        reb_combinations, rtb_combinations = get_combinations()
        combinations = reb_combinations if st.session_state.form_type == 'REB' else rtb_combinations
        
        combination = st.selectbox("Select Combination", combinations, key="combination")
        
        if combination and subject_map:
            subjects = subject_map.get(combination, [])
            
            if st.session_state.form_type == 'REB':
                st.write("**Principal Subjects (3 required):**")
                subject_scores = []
                
                for i, subject in enumerate(subjects[:3]):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**{subject}**")
                    with col2:
                        score = st.number_input(
                            f"Score for {subject}",
                            min_value=0,
                            max_value=100,
                            value=70,
                            key=f"score_{i}"
                        )
                        subject_scores.append((subject, score))
            
            else:  # RTB
                st.write("**Subjects (minimum 5 required):**")
                subject_scores = []
                
                num_subjects = st.slider("Number of subjects", 5, min(10, len(subjects)), 5)
                
                for i in range(num_subjects):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        selected_subject = st.selectbox(
                            f"Subject {i+1}",
                            subjects,
                            key=f"subject_{i}"
                        )
                    with col2:
                        score = st.number_input(
                            f"Score",
                            min_value=0,
                            max_value=100,
                            value=70,
                            key=f"score_{i}"
                        )
                    subject_scores.append((selected_subject, score))
            
            # Navigation buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("Previous", key="step2_prev"):
                    st.session_state.current_step = 1
                    st.rerun()
            
            with col3:
                if st.button("Next Step", key="step2_next"):
                    if combination and len(subject_scores) >= (3 if st.session_state.form_type == 'REB' else 5):
                        st.session_state.form_data.update({
                            'combination': combination,
                            'subject_scores': subject_scores
                        })
                        st.session_state.current_step = 3
                        st.rerun()
                    else:
                        st.error("Please select combination and enter all subject scores")
    
    # Step 3: Additional Information
    elif current_step == 3:
        st.subheader("ℹ️ Additional Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.number_input("Completion Year", min_value=2010, max_value=2025, value=2024)
            skills = st.selectbox("Has Trade Skills", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        
        with col2:
            fee_paid = st.selectbox("Application Fee Paid", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
            
            # Get programs for selected combination
            programs = program_map.get(st.session_state.form_data.get('combination', ''), [])
            program = st.selectbox("Program Choice", programs if programs else ["No programs available"])
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("Previous", key="step3_prev"):
                st.session_state.current_step = 2
                st.rerun()
        
        with col3:
            if st.button("Predict Admission", key="step3_predict"):
                if program and programs:
                    student_data = {
                        'combination': st.session_state.form_data['combination'],
                        'completed_year': year,
                        'has_trade_skills': skills[1],
                        'application_fee_paid': fee_paid[1],
                        'program_choice': program,
                        'is_tvet': 1 if st.session_state.form_type == 'RTB' else 0,
                        'subject_scores': st.session_state.form_data['subject_scores']
                    }
                    
                    # Make prediction
                    with st.spinner("Making prediction..."):
                        try:
                            scores = [score for _, score in student_data['subject_scores']]
                            is_eligible, recommended = check_admission_eligibility(
                                student_data['combination'],
                                student_data['completed_year'],
                                scores,
                                student_data['application_fee_paid'],
                                is_tvet=(student_data['is_tvet'] == 1)
                            )
                            
                            if not is_eligible:
                                result = {
                                    'admission_status': 'Not Admitted',
                                    'recommended_programs': [],
                                    'subject_names': [subj for subj, _ in student_data['subject_scores']],
                                    'scores': scores,
                                    'message': 'Does not meet minimum academic requirements'
                                }
                            else:
                                student_df = prepare_prediction_data(student_data)
                                pred = model.predict(student_df)[0]
                                status = 'Admitted' if pred == 1 else 'Not Admitted'
                                
                                result = {
                                    'admission_status': status,
                                    'recommended_programs': recommended,
                                    'subject_names': [subj for subj, _ in student_data['subject_scores']],
                                    'scores': scores,
                                    'message': 'Meets academic requirements' if status == 'Admitted' 
                                              else 'Model prediction: Not admitted'
                                }
                            
                            st.session_state.prediction_result = result
                            st.session_state.form_data.update(student_data)
                            st.session_state.current_step = 4
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error making prediction: {str(e)}")
                else:
                    st.error("Please select a program")
    
    # Step 4: Results
    elif current_step == 4:
        if st.session_state.prediction_result:
            result = st.session_state.prediction_result
            
            # Result header
            status_class = "admitted" if result['admission_status'] == 'Admitted' else "not-admitted"
            
            st.markdown(f"""
            <div class="result-card {status_class}">
                <h2>🎯 Prediction Result: {result['admission_status']}</h2>
                <p style="font-size: 16px; margin-bottom: 0;">{result['message']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create two columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Subject Performance")
                
                # Create performance chart
                subjects_df = pd.DataFrame({
                    'Subject': result['subject_names'],
                    'Score': result['scores']
                })
                
                fig = px.bar(
                    subjects_df,
                    x='Subject',
                    y='Score',
                    title='Subject Scores',
                    color='Score',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    range_color=[0, 100]
                )
                
                fig.update_layout(
                    xaxis_title="Subjects",
                    yaxis_title="Score (%)",
                    yaxis=dict(range=[0, 100])
                )
                
                # Add pass/fail line
                fig.add_hline(y=50, line_dash="dash", line_color="red", 
                             annotation_text="Pass Mark (50%)")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics
                avg_score = np.mean(result['scores'])
                pass_count = sum(1 for score in result['scores'] if score >= 50)
                
                met_col1, met_col2 = st.columns(2)
                with met_col1:
                    st.metric("Average Score", f"{avg_score:.1f}%")
                with met_col2:
                    st.metric("Subjects Passed", f"{pass_count}/{len(result['scores'])}")
            
            with col2:
                st.subheader("🎯 Recommended Programs")
                
                if result['recommended_programs']:
                    for i, program in enumerate(result['recommended_programs'], 1):
                        st.markdown(f"""
                        <div style="padding: 10px; margin: 5px 0; background-color: #e3f2fd; 
                                    border-left: 4px solid #2196f3; border-radius: 5px;">
                            <strong>{i}. {program}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No recommended programs available")
                
                # Student info summary
                st.subheader("👤 Student Information")
                student_info = st.session_state.form_data
                
                info_df = pd.DataFrame({
                    'Field': ['Name', 'Email', 'Phone', 'Form Type', 'Combination'],
                    'Value': [
                        f"{student_info.get('fname', '')} {student_info.get('lname', '')}",
                        student_info.get('email', ''),
                        student_info.get('phone', ''),
                        student_info.get('form_type', ''),
                        student_info.get('combination', '')
                    ]
                })
                
                st.dataframe(info_df, use_container_width=True, hide_index=True)
            
            # Action buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("🔄 New Prediction", key="new_prediction"):
                    st.session_state.current_step = 1
                    st.session_state.form_data = {}
                    st.session_state.prediction_result = None
                    st.rerun()
            
            with col2:
                if st.button("📝 Edit Information", key="edit_info"):
                    st.session_state.current_step = 1
                    st.rerun()
            
            with col3:
                # PDF Generation
                if st.button("📄 Generate PDF Report", key="generate_pdf"):
                    try:
                        pdf_buffer = generate_pdf_report(st.session_state.form_data, result)
                        
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"admission_prediction_{st.session_state.form_data.get('fname', 'student')}.pdf",
                            mime="application/pdf"
                        )
                        
                        st.success("PDF report generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")

# Sidebar with additional information
def sidebar():
    st.sidebar.title("📖 About This System")
    st.sidebar.markdown("""
    This Student Admission Prediction System helps students determine their likelihood 
    of admission to various academic programs based on their academic performance.
    
    **How it works:**
    1. Enter your personal information
    2. Select your academic combination and enter scores
    3. Provide additional information
    4. Get instant prediction results
    
    **Features:**
    - ✅ Real-time prediction
    - 📊 Visual performance analysis
    - 🎯 Program recommendations
    - 📄 PDF report generation
    - 📱 Mobile-friendly interface
    
    **Form Types:**
    - **REB**: Rwanda Education Board (A-Level)
    - **RTB**: Rwanda Technical Board (TVET)
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Status**")
    
    if model is not None:
        st.sidebar.success("✅ Model loaded successfully")
    else:
        st.sidebar.error("❌ Model not loaded")
    
    if st.sidebar.button("🔄 Reset Application"):
        st.session_state.current_step = 1
        st.session_state.form_data = {}
        st.session_state.prediction_result = None
        st.rerun()

if __name__ == "__main__":
    sidebar()
    main()