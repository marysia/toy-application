import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet import decode_predictions
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input

@st.cache(allow_output_mutation=True)
def load_model():
    model = MobileNet(weights="imagenet")  # this can take a bit
    return model


# Interface
st.title("Test your own image")

st.sidebar.write('## Upload your image')
file = st.sidebar.file_uploader("", type=['png', 'jpg'])
model = load_model()
if file is not None:
    image = Image.open(file)
    st.sidebar.write('### Your image')
    st.sidebar.image(image)

    image = image.resize((224, 224))
    x = np.array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    best_preds = decode_predictions(preds, top=5)
    preds = decode_predictions(model.predict(x), top=5)[0]

    prediction = preds[0][1]
    certainty = preds[0][2]

    st.write('### Prediction top 5')
    for i, prediction in enumerate(preds): 
        st.write(f'{i+1}. **{prediction[1].replace("_", " ").title()}** with {prediction[2]:.6f} certainty')



