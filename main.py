import os
import numpy as np
import tensorflow as tf
from keras.applications.convnext import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory

import io
import keras as keras
import streamlit as st
from PIL import Image


def load_image():
    """Создание формы для загрузки изображения"""
    # Форма для загрузки изображения средствами Streamlit
    uploaded_file = st.file_uploader(
        label='Выберите изображение для распознавания',
        type=['jpg', 'jpeg'])
    if uploaded_file is not None:
        # Получение загруженного изображения
        image_data = uploaded_file.getvalue()
        # Показ загруженного изображения на Web-странице средствами Streamlit
        st.image(image_data)
        # Возврат изображения в формате PIL
        return Image.open(io.BytesIO(image_data))
    else:
        return None


# @st.cache(allow_output_mutation=True)
@st.cache_resource()
def load_model():
    model = keras.models.load_model('communication_model_new.h5')
    return model


def preprocess_image(img):
    img = img.resize((100, 100))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def print_predictions(predictions):
    classes = ['АЦП', 'Вышка сотовой связи', 'Медиаконвертер', 'Спикерфон', 'Спутниковые антенны', 'Трансивер', 'Коммутатор',
               'Маршрутизатор', 'Рация', 'Цифровая антенна']
    st.write(classes[np.argmax(predictions[0])])

model = load_model()
# Выводим заголовок страницы
st.title('Классификация оборудования связи')
# Выводим форму загрузки изображения и получаем изображение
img = load_image()
# Показывам кнопку для запуска распознавания изображения
result = st.button('Распознать изображение')
 # Если кнопка нажата, то запускаем распознавание изображения
if result:
    # Предварительная обработка изображения
    x = preprocess_image(img)
    preds = model.predict(x)
    # Выводим заголовок результатов распознавания жирным шрифтом
    # используя форматирование Markdown
    st.write('**Результаты распознавания:**')
    # Выводим результаты распознавания
    print_predictions(preds)

st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.divider()
st.subheader('Проект выполнили:')
st.write('Овчинников Н.С. - РИ-120913')
st.write('Петрушина Н.Д. - РИ-120931')
st.write('Локтионов Н.В. - РИ-120931')
st.write('Довгань Е.А. - РИ-211111')
st.write('')
st.write('')
st.write('')
st.write('')
st.columns(3)[1].caption("Екатеринбург 2023")


