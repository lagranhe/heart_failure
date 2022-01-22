# Dataset [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
import sklearn
from datetime import datetime
startTime = datetime.now()

filename = "heart_failure_model.sv"
model = pickle.load(open(filename,'rb'))

sex_d = {0:"Female",1:"Male"}
diabetes_d = {0:"No",1:"Yes"}
anaemia_d = {0:"No",1:"Yes"}
high_blood_pressure_d = {0:"No", 1:"Yes"}
smoking_d = {0:"No", 1:"Yes"}

def main():

	st.set_page_config(page_title="Application for heart failure prediction")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://images.everydayhealth.com/images/heart-health/heart-failure/10-essential-facts-about-heart-failure-1440x810.jpg?w=1110")

	with overview:
		st.title("Application for heart failure prediction")

	with left:
		sex_radio = st.radio( "Sex", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
		diabetes_radio = st.radio( "Diabetes", list(sex_d.keys()), format_func=lambda x : diabetes_d[x] )
		anaemia_radio = st.radio( "Decrease of red blood cells or hemoglobin ", list(sex_d.keys()), format_func=lambda x : anaemia_d[x] )
		high_blood_pressure_radio = st.radio( "High blood pressure", list(sex_d.keys()), format_func=lambda x : high_blood_pressure_d[x] )
		smoking_radio = st.radio( "Smoking", list(sex_d.keys()), format_func=lambda x : smoking_d[x] )
        
	with right:
		age_slider = st.slider("Age", value=1, min_value=1, max_value=100)
		ejection_fraction_slider = st.slider("Percentage of blood leaving the heart at each contraction", min_value=0, max_value=100)
		creatinine_phosphokinase_number = st.number_input("Level of the CPK enzyme in the blood, mcg/L (0-8000) ", min_value=0, max_value=8000)
		platelets_number = st.number_input("Platelets in the blood, kiloplatelets/mL (0-1000000)", min_value=25000, max_value=800000, step=1000)
		serum_creatinine_number = st.number_input("Level of serum creatinine in the blood, mg/dL (0-20)", min_value=0, max_value=20)
		serum_sodium_number = st.number_input("Level of serum sodium in the blood, mEq/L (0-200)", min_value=0, max_value=200)

	data = [[age_slider,anaemia_radio,creatinine_phosphokinase_number,diabetes_radio,ejection_fraction_slider, high_blood_pressure_radio, platelets_number,serum_creatinine_number,serum_sodium_number,sex_radio, smoking_radio]]
	death_event = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Will the case end in death?")
		st.subheader(("Yes" if death_event[0] == 1 else "No"))
		st.write("Accuracy {0:.2f} %".format(s_confidence[0][death_event][0] * 100))

if __name__ == "__main__":
    main()