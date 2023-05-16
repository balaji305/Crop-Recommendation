import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings


st.beta_set_page_config(page_title="Crop Recommender", page_icon="🌿", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Crop Recommendation  🌱 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1,col2  = st.beta_columns([2,2])
    
    with col1: 
        with st.beta_expander(" ℹ️ Information", expanded=True):
            st.write("""
                Welcome to our innovative web application, dedicated to revolutionizing crop recommendation systems in 
                the field of agriculture. Our platform harnesses the power of precision agriculture to provide accurate and 
                data-driven recommendations for optimal crop selection. By analyzing a multitude of factors specific to each
                site, we ensure personalized and site-specific crop recommendations that can boost productivity and 
                efficiency. With our user-friendly interface and advanced algorithms, our web app empowers farmers and 
                agricultural professionals to make informed decisions, reducing risks and maximizing yields. Experience the
                future of crop recommendation systems with our cutting-edge web application, designed to elevate your 
                farming practices to new heights.
            """)
        '''
        ## How does it work ❓ 
        Complete all the parameters and the machine learning model will predict the most suitable crops to grow in a particular farm based on various parameters
        '''


    with col2:
        st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        N = st.number_input("Nitrogen", 1,100)
        P = st.number_input("Phosporus", 1,100)
        K = st.number_input("Potassium", 1,100)
        temp = st.number_input("Temperature",0.0,60.0)
        humidity = st.number_input("Humidity in %", 0.0,100.0)
        ph = st.number_input("Ph", 0.0,14.0)
        rainfall = st.number_input("Rainfall in mm",0.0,10000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):

            loaded_model = load_model('model.pkl')
            prediction = loaded_model.predict(single_pred)
            col1.write('''
		    ## Results 🔍 
		    ''')
            col1.success(f"{prediction.item().title()} are recommended by the A.I for your farm.")
      #code for html ☘️ 🌾 🌳 👨‍🌾  🍃

    st.warning("Note: This A.I application is for educational purposes only and cannot be relied upon. Check the source code [here](https://github.com/balaji305/Crop-Recommendation)")
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()