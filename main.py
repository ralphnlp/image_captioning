import streamlit as st
from PIL import Image
from utils import img2caption
import os
import time

st.title('Image caption')
file = st.file_uploader('')
if file != None:
    start_time = time.time()

    img = Image.open(file)
    print(img)
    st.image(img)
    img.save('./temp.jpg')

    caption = img2caption('./temp.jpg')
    st.write('({:.2f}s) Caption: {} '.format(time.time()-start_time, caption))
    os.system('rm temp.jpg')