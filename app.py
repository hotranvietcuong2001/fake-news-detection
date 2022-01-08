import pickle
import streamlit as st 
from utils import text_preprocessor


model_dict = {
    "Multinomial Naive Bayes": "models/naive-bayes.pkl",
    "Logistic Regression": "models/logistic-regression.pkl",
    "Random Forest Classifier": "models/random-forest.pkl",
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


# load text vectorizer
@st.cache(allow_output_mutation=True)
def load_vectorizer():
    dir = "models/vectorizer.pkl"
    with open(dir, "rb") as f:
        res = pickle.load(f)
        return res


def main():
    st.set_page_config(
        page_title="Fake News Detection",
        page_icon="🔍",
        menu_items={
            'About': """
            ### Powered By:
            |    MSSV   |       Họ tên       |
            |:---------:|:------------------:|
            | 19120056  | Hồ Trần Việt Cường |
            | 19120069  | Nguyễn Thế Hải     |
            | 19120119  | Nguyễn Minh Phúc   |
            | 19120130  | Trần Đức Thắng     |
            | 19120508  | Nguyễn Đức Hiếu    |
            """,
        }
    )

    # load models 
    models = load_models()

    # load preprocessor
    vectorizer = load_vectorizer()

    # title
    title = """<h1 style="text-align:center;">Vietnamese Fake News Detection</h1>"""
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
            if len(news) == 0:
                st.warning('Please insert at least a piece of news')
            else:
                model = models[selection]

                # text preprocessing
                preprocessed_news = text_preprocessor([news])

                ## text vectorizing
                preprocessed_news = vectorizer.transform(preprocessed_news)

                ## predict
                res = model.predict(preprocessed_news)[0]
                if res == 1:
                    st.success('This is a Real news')
                else:
                    st.error('This is a Fake news')


if __name__ == '__main__':
    main()
