import streamlit as st
import torch
from torchvision import transforms
from Predict import validation_dataset, predict, Model
import matplotlib.pyplot as plt

st.set_page_config(page_title="Deepfake Detection")
st.title("Deepfake Detector")
# Upload video
video = st.file_uploader(label="Upload a video file", type="mp4")

# Model and transformations
path_to_model = 'Final Outputs/Trained model/ResNet50_20.pt'

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

model = Model(2)
map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.load_state_dict(torch.load(path_to_model, map_location=map_location))
model.eval()

if video is not None:
    # Save the uploaded file temporarily
    with open("temp_video.mp4", "wb") as f:
        f.write(video.read())

    # Create the dataset
    video_data = validation_dataset(["temp_video.mp4"], sequence_length=20, transform=train_transforms)

    # Iterate over the dataset (for simplicity, assuming single video here)
    frames_tensor = video_data[0]  # Get the processed tensor for the video

    # Run prediction
    with plt.ioff():  # Turn off interactive mode for matplotlib
        prediction = predict(model, frames_tensor)

    # Display results
    if prediction[0] == 1:
        st.write("Prediction: Real")
    else:
        st.write("Prediction: Fake")
    st.write(f"Confidence: {prediction[1]:.2f}%")

    # Display the matplotlib plot
    st.write("Feature Visualization:")
    st.pyplot(plt.gcf())  # Display the current matplotlib figure
