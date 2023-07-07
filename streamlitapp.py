import streamlit as st
import os
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from main import load_model

st.set_page_config(layout='wide')
with st.sidebar:
    st.image(
        'https://imageio.forbes.com/specials-images/imageserve/5f51c38ba72e09805e578c53/3-Predictions-For-The-Role-Of'
        '-Artificial-Intelligence-In-Art-And-Design/960x0.jpg?format=jpg&width=960')
    st.title("Generative Adversarial Networks")
    st.info("With this model we are able to generate artificial anime faces.")

st.title('Deep Convolutional GAN')
epoch_array = [5,25,50]
models = []
for i in epoch_array:
    model = load_model(i)
    models.append([model, i])

image_size = 64
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]


def show_images(images, nmax=25):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xticks([])
    ax.set_yticks([])
    st.image(make_grid(denorm(images.detach()[:nmax]), nrow=5).permute(1, 2, 0).numpy())

images = []

def make_images(latent_size=128):
    xb = torch.randn(batch_size, latent_size, 1, 1)
    with tab1:
        with col1:
            generator,epoch = models[0]
            fake_images = generator(xb)
            images.append(fake_images)
            # st.write(f"Images generated by the model trained for {epoch} epochs:")
            # show_images(fake_images)
        with col2:
            generator,epoch = models[1]
            fake_images = generator(xb)
            # st.write(f"Images generated by the model trained for {epoch} epochs:")
            images.append(fake_images)
            # show_images(fake_images)
        with col3:
            generator,epoch = models[2]
            fake_images = generator(xb)
            images.append(fake_images)
            # st.write(f"Images generated by the model trained for {epoch} epochs:")
            # show_images(fake_images)


tab1,tab2 = st.tabs(["Demonstration","About"])
with tab1:
    col1,col2,col3 = st.columns(3)
    if images != []:
        with col1:
            st.write(f"Images generated by the model trained for {epoch} epochs:")
            show_images(images[0])
        with col2:
            st.write(f"Images generated by the model trained for {epoch} epochs:")
            show_images(images[1])
        with col3:
            st.write(f"Images generated by the model trained for {epoch} epochs:")
            show_images(images[2])
    with col1:
        st.button("Refresh",on_click=make_images)
            
