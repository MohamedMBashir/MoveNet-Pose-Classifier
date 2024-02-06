import cv2
import numpy as np
import tensorflow as tf
from movenet import detect, draw_prediction_on_image
from pose_classification_model import model, class_names
import streamlit as st

cap = cv2.VideoCapture(0)

st.title('Squat Counter')
frame_placeholder = st.empty()
stop_button_pressed = st.button('Stop')

# Initialize session state for the counter and stage
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'stage' not in st.session_state:
    st.session_state.stage = None

def process_frame(frame):
    # Convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Make detection
    img = tf.convert_to_tensor(image)
    person = detect(img)
    
    # Overlay detections on the image
    image_overlayed = draw_prediction_on_image(
        img.numpy().astype(np.uint8), person, 
        close_figure=True, keep_input_size=True)
    
    # Convert back to BGR for displaying in Streamlit
    image_overlayed = cv2.cvtColor(image_overlayed, cv2.COLOR_RGB2BGR)

    # Prepare pose landmarks for classification
    pose_landmarks = np.array([[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                               for keypoint in person.keypoints], dtype=np.float32)
    coordinates = tf.expand_dims(tf.convert_to_tensor(pose_landmarks.flatten()), axis=0)
    
    # Predict pose
    prediction = model.predict(coordinates, verbose=0)
    pred_label = class_names[np.argmax(prediction)]
    
    # Update counter based on prediction
    if pred_label == 'Up':
        st.session_state.stage = 'Up'
    elif pred_label == 'Down' and st.session_state.stage == 'Up':
        st.session_state.stage = 'Down'
        st.session_state.counter += 1
    
    # Overlay counter and stage on the image
        
    # Set the size of the rectangle as a percentage of the image's size
    height, width, _ = image_overlayed.shape
    rect_width = int(width * 0.2)
    rect_height = int(height * 0.1)

    # Rectangle on the top left
    x1 = 0
    y1 = 0
    x2 = rect_width
    y2 = rect_height
    cv2.rectangle(image_overlayed, (x1, y1), (x2, y2), (51, 87, 255), -1)

    # Display the Stage and state
    text = 'STAGE'
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = x1 + (rect_width - text_size[0]) // 2
    text_y = y1 + (rect_height // 2 - text_size[1]) // 2
    cv2.putText(image_overlayed, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    text = pred_label
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    text_x = x1 + (rect_width - text_size[0]) // 2
    text_y = y1 + (rect_height // 2 + text_size[1] * 3) // 2
    cv2.putText(image_overlayed, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)  

    # Rectangle on the top right
    x1 = width - rect_width
    x2 = width
    cv2.rectangle(image_overlayed, (x1, y1), (x2, y2), (51, 87, 255), -1)

    # Display the Counter and count
    text = 'REPS'
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = x1 + (rect_width - text_size[0]) // 2
    text_y = y1 + (rect_height // 2 - text_size[1]) // 2
    cv2.putText(image_overlayed, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    text = str(st.session_state.counter)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0] 
    text_x = x1 + (rect_width - text_size[0]) // 2
    text_y = y1 + (rect_height // 2 + text_size[1] * 3) // 2
    cv2.putText(image_overlayed, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
    
    
    return image_overlayed

while cap.isOpened() and not stop_button_pressed:
    ret, frame = cap.read()

    if not ret:
        print("Video capture has ended.")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    processed_frame = process_frame(frame)

    frame_placeholder.image(processed_frame, channels='RGB')

    if cv2.waitKey(10) & 0xFF == ord('q') or stop_button_pressed:
        break

cap.release()
cv2.destroyAllWindows()
