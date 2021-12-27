import pickle
import streamlit as st 


model_dict = {
    "Passive Aggressive Classifier": "models/pac.pkl",
    "Logistic Regression": "models/lr.pkl", 
}


# load all models
@st.cache(allow_output_mutation=True)
def load_models():
    models = {}
    for model_name in model_dict:
        model_dir = model_dict[model_name]
        with open(model_dir, "rb") as f:
            model = pickle.load(f)
            models[model_name] = model

    return models


# load text preprocessor
@st.cache(allow_output_mutation=True)
def load_preprocessor():
    dir = "models/preprocessor.pkl"
    with open(dir, "rb") as f:
        res = pickle.load(f)
        return res
    

def main():
    # load models 
    models = load_models()
    # load preprocessor
    preprocessor = load_preprocessor()

    # title
    title = """<h1 style="text-align:center;">Fake news Detection</h1>"""
    st.markdown(title, unsafe_allow_html=True)

    # model selection
    selection = st.selectbox('Choose a model', model_dict.keys())

    # news input
    news = st.text_area("Insert a piece of news here:")
    st.markdown("""<hr/>""", unsafe_allow_html=True)

    # submit button
    button = st.button('Check this news')
    if button:
        with st.spinner('Checking...'):
            model = models[selection]
            preprocessed_news = preprocessor.transform([news])

            res = model.predict(preprocessed_news)[0]
            if res == 1:
                st.success('This is a Real news')
            else:
                st.success('This is a Fake news')


if __name__ == '__main__':
    main()
