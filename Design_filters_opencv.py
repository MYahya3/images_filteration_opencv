import io
import base64
from PIL import Image
from filters import *
import cv2

# Generating a link to download a particular image file.
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format = 'JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Set title.
st.header('Image filtration with Opencv')

# Upload image.
uploaded_file = st.sidebar.file_uploader('Upload Images:', type=['png','jpeg','jpg'], accept_multiple_files=True)

if len(uploaded_file) > 0:
    images_files = [file.name for file in uploaded_file]
    img_list = list(dict.fromkeys(images_files))
    img_choose = st.selectbox("Choose the image ",img_list)
    image = images_files.index(img_choose)
    # Convert the file to an opencv image.
    raw_bytes = np.asarray(bytearray(uploaded_file[image].read()), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    input_col, output_col = st.columns(2)
    with input_col:
        st.write('Original')
        # Display uploaded image.
        st.image(img, channels='BGR', use_column_width=True)
    # Display a selection box for choosing the filter to apply.
    option = st.selectbox('Select a filter:',('None','Black White','Vintage','Edges Detection','Pencil Sketch','Stylization'
                                              ,'Sharpen'))

    # Colorspace of output image.
    color = 'BGR'

    output = None
     # Generate filtered image based on the selected option.
    if option == 'None':
        # Don't show output image.
        output = img
        st.warning("No Filter Apllied")
    elif option == 'Black White':
        output = bw_filter(img)
        color = 'GRAY'
    elif option == 'Vintage':
        output = BrownEffect(img)
    elif option == 'Edges Detection':
        thres_1 = st.slider('Threshold 1', 0, 300, 100, step=20)
        thres_2 = st.slider('Threshold 2', 0, 300, 200, step=20)
        output = CannyEdgeD(img, thres_1, thres_2)
    elif option == 'Pencil Sketch':
        ksize = st.slider('Blur kernel size', 1, 11, 5, step=2)
        output = pencil_sketch(img, ksize)
        color = 'GRAY'
    elif option == "Stylization":
        ksize = st.slider('kernel size', 1, 11, 5, step=2)
        output = stylization(img, ksize)
    elif option == "Sharpen":
        ksize = st.slider('kernel size', 0.0, 2.0, 1.0, step=0.2)
        output = sharpen(img, ksize)

    with output_col:
        st.write('Output')
        st.image(output, channels=color)
        # fromarray convert cv2 image into PIL format for saving it using download link.
        if color == 'BGR':
            result = Image.fromarray(output[:,:,::-1])
        else:
            result = Image.fromarray(output)
        # Display link.
        st.markdown(get_image_download_link(result,option + '_' + img_choose,'Download '+'Output'),
                    unsafe_allow_html=True)
else:
    st.warning("Upload Images from Sidebar")
    sample_img = 'sample.jpg'
    img_read = cv2.imread(sample_img, cv2.IMREAD_COLOR)
    input_col, output_col = st.columns(2)
    with input_col:
        st.write('Sample Input')
        # Display uploaded image.
        st.image(sample_img, channels='BGR', use_column_width=True)
    img_gray = cv2.cvtColor(img_read , cv2.COLOR_BGR2GRAY)
    with output_col:
        st.write('Sample Output')
        st.image(img_gray, channels="GRAY")
