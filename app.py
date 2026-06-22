import os
import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_drawable_canvas import st_canvas

from model import MNISTCNN
from train import train_model

# Page Configuration & Styling
st.set_page_config(
    page_title="CNN Handwritten Digit Recognizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling using CSS
st.markdown("""
<style>
    /* Main body background and custom font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title Styling */
    .main-title {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8F8F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #888896;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Interactive Card designs */
    .metric-card {
        background-color: #1E222A;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #2D3139;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(255, 75, 75, 0.1);
        border-color: #FF4B4B;
    }
    
    /* Custom status pill */
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-loaded {
        background-color: rgba(46, 213, 115, 0.15);
        color: #2ed573;
        border: 1px solid #2ed573;
    }
    .status-missing {
        background-color: rgba(255, 71, 87, 0.15);
        color: #ff4757;
        border: 1px solid #ff4757;
    }
    
    /* Custom divider line */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, #FF4B4B 0%, rgba(255, 75, 75, 0) 100%);
        margin: 1.5rem 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load PyTorch model weights
@st.cache_resource
def load_trained_model(path='mnist_cnn.pth'):
    if not os.path.exists(path):
        return None
    model = MNISTCNN()
    try:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None

# Helper to plot activation maps in a beautiful grid
def plot_activation_grid(activations, title, num_cols=8):
    # activations: [1, C, H, W]
    act_data = activations.squeeze(0).detach().cpu().numpy()
    num_filters = act_data.shape[0]
    num_rows = (num_filters + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 1.3 * num_rows))
    fig.patch.set_facecolor('#0E1117') # Match Streamlit dark bg
    
    for i, ax in enumerate(axes.flatten()):
        ax.set_facecolor('#0E1117')
        if i < num_filters:
            # Show the activation map
            # We scale each filter map independently for visual contrast
            img = act_data[i]
            im = ax.imshow(img, cmap='magma', interpolation='nearest')
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.tight_layout()
    return fig

# Title area
st.markdown('<div class="main-title">Handwritten Digit Recognizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An interactive CNN playground: train, inspect, and test deep learning models in real-time.</div>', unsafe_allow_html=True)

# Load the model
model = load_trained_model()

# Sidebar: Model Status & Config
st.sidebar.markdown("### 🛠️ Model Engine Status")
if model is not None:
    st.sidebar.markdown('<span class="status-pill status-loaded">● Model Loaded</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="status-pill status-missing">● No Model Found</span>', unsafe_allow_html=True)
    st.sidebar.warning("Train a model using the 'Interactive Training Sandbox' tab to enable drawing recognition!")

st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.sidebar.markdown("### 📘 Project Info")
st.sidebar.info("""
This application implements a **Convolutional Neural Network (CNN)** in PyTorch to classify handwritten digits (0-9) using the classical **MNIST dataset**.
- **Input size**: 28x28 grayscale image
- **Architecture**: 2x Convolutional layers + 2x Max Pooling layers + 1x Dropout + 2x Dense layers
""")

# Setup Tabs
tab1, tab2, tab3 = st.tabs([
    "🎨 Interactive Drawing Board", 
    "⚙️ Interactive Training Sandbox", 
    "🧠 CNN Architecture Explainer"
])

# ================= TAB 1: DRAWING BOARD =================
with tab1:
    if model is None:
        st.info("👋 Welcome! Since you don't have a trained model yet, please head over to the **'Interactive Training Sandbox'** tab to train your first neural network. It takes less than a minute!")
    else:
        st.markdown("### Draw a digit (0-9) in the canvas below!")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("##### ✏️ Drawing Canvas")
            # Create drawing canvas
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 1)",  # Solid black fill if closed shapes are drawn
                stroke_width=18,
                stroke_color="#FFFFFF",        # White digit stroke
                background_color="#000000",    # Black background (standard MNIST format)
                update_streamlit=True,
                height=300,
                width=300,
                drawing_mode="freedraw",
                key="canvas",
                display_toolbar=True
            )
            st.caption("Use the toolbar options to clear drawing (trash icon) or undo mistakes.")
            
        with col2:
            st.markdown("##### 🔍 Model Prediction & Pre-processing")
            
            # Check if user has drawn anything
            has_drawing = (canvas_result.image_data is not None and 
                           np.any(canvas_result.image_data[:, :, :3] > 10)) # Check RGB channels
            
            if has_drawing:
                # 1. Image preprocessing
                img_data = canvas_result.image_data[:, :, :3] # Keep RGB
                # Convert to PIL Image, grayscale, and resize to 28x28
                img_pil = Image.fromarray(img_data.astype('uint8')).convert('L')
                img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Convert to numpy array, scale to [0, 1]
                img_arr = np.array(img_resized).astype(np.float32) / 255.0
                
                # Normalize using MNIST statistics (mean=0.1307, std=0.3081)
                img_normalized = (img_arr - 0.1307) / 0.3081
                
                # Convert to tensor: shape [1, 1, 28, 28] (batch_size=1, channels=1, H=28, W=28)
                tensor_input = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0).float()
                
                # Run inference
                with torch.no_grad():
                    logits = model(tensor_input)
                    probs = F.softmax(logits, dim=1).squeeze(0).numpy()
                    
                predicted_class = np.argmax(probs)
                confidence = probs[predicted_class] * 100
                
                # Display side-by-side: original/cropped drawing vs downsampled 28x28
                preview_col1, preview_col2 = st.columns([1, 1])
                with preview_col1:
                    st.image(img_pil, caption="Processed Image", width=120)
                with preview_col2:
                    st.image(img_resized, caption="28x28 MNIST Input", width=120)
                
                # Output box
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin-top:0; color:#FF4B4B;">Prediction: {predicted_class}</h3>
                    <p style="margin:0; font-size:1.1rem; color:#888896;">Confidence Score: <b>{confidence:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence distribution chart
                st.markdown("<br><b>Confidence Distribution</b>", unsafe_allow_html=True)
                chart_data = pd.DataFrame({
                    'Digit': [str(i) for i in range(10)],
                    'Probability': probs
                })
                st.bar_chart(chart_data.set_index('Digit'), height=180)
                
            else:
                st.info("🖊️ Draw inside the canvas box on the left to see the neural network predictions and image processing in real-time.")
                
        # Activation visualization section
        if has_drawing:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### 🔬 Visualizing the CNN Layers: Inside the Network's Mind")
            st.markdown("""
            A Convolutional Neural Network consists of layers of filters that apply mathematical convolutions to the image. 
            Below, you can inspect the actual activation maps of the first two convolutional layers for the digit you just drew!
            """)
            
            with torch.no_grad():
                act1, act2 = model.get_activations(tensor_input)
                
            col_act1, col_act2 = st.columns([1, 1])
            
            with col_act1:
                st.markdown("##### 🟥 Convolution Layer 1 (Conv1): Low-Level Feature Extraction")
                st.markdown("Produces 16 activation maps (28x28). These filters act as edge, corner, and orientation detectors.")
                fig1 = plot_activation_grid(act1, "Conv1 Activations", num_cols=4)
                st.pyplot(fig1, use_container_width=True)
                
            with col_act2:
                st.markdown("##### 🟦 Convolution Layer 2 (Conv2): Complex Shapes & Patterns")
                st.markdown("Produces 32 activation maps (14x14). The features are more abstract, representing combinations of curves, loops, and intersections.")
                fig2 = plot_activation_grid(act2, "Conv2 Activations", num_cols=4)
                st.pyplot(fig2, use_container_width=True)

# ================= TAB 2: TRAINING SANDBOX =================
with tab2:
    st.markdown("### ⚙️ Train your Convolutional Neural Network")
    st.markdown("""
    Customize the hyperparameters below and click **Start Training**. 
    The training loop will run in PyTorch, fetch the MNIST dataset, and plot loss and accuracy curves in real-time.
    """)
    
    col_params, col_log = st.columns([1, 2])
    
    with col_params:
        st.markdown("##### 🎛️ Training Hyperparameters")
        epochs = st.slider("Epochs (Training Cycles)", min_value=1, max_value=10, value=3, help="How many times the network goes through the entire dataset.")
        batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128, 256], value=64, help="Number of training samples processed before updating internal weights.")
        lr = st.select_slider("Learning Rate", options=[0.01, 0.005, 0.001, 0.0005, 0.0001], value=0.001, help="Controls how much we adjust the weights in response to the estimated error.")
        
        train_btn = st.button("🚀 Start Training Network", use_container_width=True)
        
    with col_log:
        st.markdown("##### 📊 Training Progress & Metrics")
        
        # Placeholders for live updates
        progress_bar = st.empty()
        status_text = st.empty()
        
        plot_col1, plot_col2 = st.columns([1, 1])
        with plot_col1:
            loss_chart_placeholder = st.empty()
        with plot_col2:
            acc_chart_placeholder = st.empty()
            
        metrics_table_placeholder = st.empty()

        # Handle button click
        if train_btn:
            # We initialize lists to hold the metrics per epoch
            epochs_list = []
            train_losses = []
            train_accs = []
            val_losses = []
            val_accs = []
            
            # Temporary storage to draw updating curves
            loss_df = pd.DataFrame(columns=["Train Loss", "Val Loss"])
            acc_df = pd.DataFrame(columns=["Train Acc", "Val Acc"])
            
            # Progress callbacks
            def update_progress(percent_done, loss_val):
                progress_bar.progress(percent_done)
                status_text.text(f"Batch progress: {percent_done*100:.1f}% | Current batch loss: {loss_val:.4f}")
                
            def update_epoch(epoch, train_loss, train_acc, val_loss, val_acc):
                epochs_list.append(epoch)
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                # Update charts
                new_loss_row = pd.DataFrame([[train_loss, val_loss]], columns=["Train Loss", "Val Loss"], index=[epoch])
                new_acc_row = pd.DataFrame([[train_acc, val_acc]], columns=["Train Acc", "Val Acc"], index=[epoch])
                
                loss_df.loc[epoch] = [train_loss, val_loss]
                acc_df.loc[epoch] = [train_acc, val_acc]
                
                loss_chart_placeholder.line_chart(loss_df)
                acc_chart_placeholder.line_chart(acc_df)
                
                # Update metrics dataframe table
                metrics_df = pd.DataFrame({
                    'Epoch': epochs_list,
                    'Train Loss': train_losses,
                    'Train Acc (%)': [x*100 for x in train_accs],
                    'Val Loss': val_losses,
                    'Val Acc (%)': [x*100 for x in val_accs]
                }).set_index('Epoch')
                metrics_table_placeholder.dataframe(metrics_df.style.format("{:.4f}"))
            
            st.toast("Downloading MNIST dataset & launching training script...", icon="📥")
            
            # Run the training
            with st.spinner("Training model in PyTorch..."):
                model_trained, _, _, _, _ = train_model(
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    progress_callback=update_progress,
                    epoch_callback=update_epoch
                )
                
            st.success("🎉 Training complete! The model weights have been updated successfully.", icon="✅")
            st.balloons()
            
            # Clear caches so the new model weights are reloaded on Tab 1
            st.cache_resource.clear()
            st.rerun()

# ================= TAB 3: CNN EXPLAINER =================
with tab3:
    st.markdown("### 🧠 How does a Convolutional Neural Network (CNN) Work?")
    st.markdown("""
    Convolutional Neural Networks are designed specifically for image recognition tasks because they preserve spatial relationships between pixels. Let's break down the layers in the model you just trained!
    """)
    
    st.markdown("""
    #### 1. Input Image
    The input is a **28x28 grayscale image**. Unlike color images that have 3 color channels (Red, Green, Blue), grayscale images have only **1 channel**. Each pixel is a value from `0` (pure black) to `255` (pure white).
    
    #### 2. The Convolution Operation (Conv Layer)
    In a convolution layer, a set of small learned weights called a **kernel** (or filter, typically size 3x3) slides across the image.
    * At each position, it multiplies the kernel weights with the overlapping pixel values and sums them up.
    * This allows the network to extract local patterns like **edges**, **corners**, and **shapes** regardless of where they appear in the image.
    * In our model:
      - **Conv1** applies **16 distinct filters** of size 3x3 to find basic edge patterns.
      - **Conv2** applies **32 distinct filters** of size 3x3 on top of Conv1 features to search for complex shape combinations (like loops or angles).
      
    #### 3. Activation Function (ReLU)
    After convolution, the outputs are passed through an activation function, usually **ReLU** (Rectified Linear Unit), defined as `f(x) = max(0, x)`.
    * This sets all negative pixel activations to zero.
    * ReLU introduces **non-linearity** into the network, enabling it to learn complex mathematical functions and decision boundaries.
    
    #### 4. Max Pooling (Downsampling)
    Pooling layers reduce the width and height of the image while keeping the most important feature activations.
    * **Max Pooling** with a 2x2 window looks at 4 pixels and outputs only the *maximum* value.
    * This halves the image width and height, reducing computing power requirements and helping the network become invariant to small shifts or rotations of the digit.
    
    #### 5. Dropout
    During training, **Dropout** randomly "switches off" a percentage of connections (25% in our model) at each step. 
    * This forces the network to learn robust, distributed representations of numbers, ensuring it doesn't over-rely on specific pixels (which prevents overfitting).
    
    #### 6. Fully Connected (Dense) Output
    Once all convolutions and poolings are done, the 2D feature maps are flattened into a 1D vector (in our case, of length `32 * 7 * 7 = 1568`).
    * This vector is fed into a **Fully Connected layer** which works like a standard neural network.
    * Finally, the last layer has **10 output nodes**, representing the logits (confidence score) for each digit `0` through `9`.
    * We apply a **Softmax** function to these logits to convert them into a probability distribution that sums to 100%!
    """)
