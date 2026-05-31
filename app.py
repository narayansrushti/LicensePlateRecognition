import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="AI License Plate Recognition",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "license_plate_model.keras",
        compile=False
    )

model = load_model()

# ---------------------------
# LOAD CASCADE
# ---------------------------
plate_cascade = cv2.CascadeClassifier(
    "indian_license_plate.xml"
)

# ---------------------------
# DETECT PLATE
# ---------------------------
def detect_plate(img, text=''):

    plate_img = img.copy()
    roi = img.copy()

    plate_rect = plate_cascade.detectMultiScale(
        plate_img,
        scaleFactor=1.2,
        minNeighbors=7
    )

    plate = None

    for (x, y, w, h) in plate_rect:

        plate = roi[y:y+h, x:x+w]

        cv2.rectangle(
            plate_img,
            (x+2, y),
            (x+w-3, y+h-5),
            (51,181,155),
            3
        )

    return plate_img, plate

# ---------------------------
# FIND CONTOURS
# ---------------------------
def find_contours(dimensions, img):

    cntrs, _ = cv2.findContours(
        img.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    cntrs = sorted(
        cntrs,
        key=cv2.contourArea,
        reverse=True
    )[:15]

    x_cntr_list = []
    img_res = []

    for cntr in cntrs:

        x, y, w, h = cv2.boundingRect(cntr)

        if (
            w > lower_width and
            w < upper_width and
            h > lower_height and
            h < upper_height
        ):

            x_cntr_list.append(x)

            char_copy = np.zeros((44,24), dtype=np.uint8)

            char = img[y:y+h, x:x+w]

            char = cv2.resize(
                char,
                (20,40)
            )

            char = cv2.subtract(
                255,
                char
            )

            char_copy[2:42,2:22] = char

            img_res.append(char_copy)

    indices = sorted(
        range(len(x_cntr_list)),
        key=lambda k: x_cntr_list[k]
    )

    img_res_copy = []

    for idx in indices:
        img_res_copy.append(
            img_res[idx]
        )

    return np.array(img_res_copy)

# ---------------------------
# SEGMENT CHARACTERS
# ---------------------------
def segment_characters(image):

    img_lp = cv2.resize(
        image,
        (333,75)
    )

    img_gray_lp = cv2.cvtColor(
        img_lp,
        cv2.COLOR_BGR2GRAY
    )

    _, img_binary_lp = cv2.threshold(
        img_gray_lp,
        200,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    img_binary_lp = cv2.erode(
        img_binary_lp,
        (3,3)
    )

    img_binary_lp = cv2.dilate(
        img_binary_lp,
        (3,3)
    )

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    dimensions = [
        LP_WIDTH/6,
        LP_WIDTH/2,
        LP_HEIGHT/10,
        2*LP_HEIGHT/3
    ]

    char_list = find_contours(
        dimensions,
        img_binary_lp
    )

    return char_list

# ---------------------------
# FIX IMAGE DIMENSIONS
# ---------------------------
def fix_dimension(img):

    new_img = np.zeros(
        (28,28,3)
    )

    for i in range(3):
        new_img[:,:,i] = img

    return new_img

# ---------------------------
# OCR PREDICTION
# ---------------------------
def show_results(characters_list):

    characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    output = []

    for ch in characters_list:

        img_ = cv2.resize(
            ch,
            (28,28),
            interpolation=cv2.INTER_AREA
        )

        img = fix_dimension(img_)

        img = img.reshape(
            1,
            28,
            28,
            3
        )

        y_probs = model.predict(
            img,
            verbose=0
        )

        y_ = np.argmax(
            y_probs,
            axis=-1
        )[0]

        output.append(
            characters[y_]
        )

    return ''.join(output)

# ---------------------------
# UI
# ---------------------------
st.title("🚗 AI License Plate Recognition System")

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload Vehicle Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    image_np = np.array(image)

    st.image(
        image,
        caption="Uploaded Vehicle",
        use_container_width=True
    )

    if st.button("🔍 Detect Number Plate"):

        try:

            detected_img, plate = detect_plate(
                image_np
            )

            if plate is None:

                st.error(
                    "No Number Plate Detected"
                )

            else:

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader(
                        "Detected Vehicle"
                    )

                    st.image(
                        detected_img,
                        use_container_width=True
                    )

                with col2:

                    st.subheader(
                        "Extracted Plate"
                    )

                    st.image(
                        plate,
                        use_container_width=True
                    )

                chars = segment_characters(
                    plate
                )

                result = show_results(
                    chars
                )
                st.write("Detected Plate Number:", result)

                st.success(
                    f"Detected Number : {result}"
                )

                st.subheader(
                    "Segmented Characters"
                )

                cols = st.columns(
                    min(
                        len(chars),
                        6
                    )
                )

                for i, c in enumerate(chars):

                    c = c.astype(np.uint8)

                    cols[i % 6].image(
                    c,
                    width=80,
                    clamp=True
                )

        except Exception as e:

            st.error(
                f"Error : {e}"
            )

st.markdown("---")
st.markdown(
    "Built with CNN + OpenCV + Streamlit"
)