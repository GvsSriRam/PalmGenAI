# PalmGenAI - Palm Image Generation using Diffusion Models

This project explores generating palm images from lower-dimensional representations (weight templates) using diffusion models in PyTorch.

## Project Structure

- **Datasets:** Contains the PalmPrint dataset that's owned by Indian Institute of technology Delhi
    - `Left Hand`: Left Hand Palm Images
    - `Right Hand`: Right Hand Palm Images
    - `Segmented`: Extracted ROI images of both hands
    - `Preprocessed`: Eigenpalms and weight templates extracted from `pca.ipynb` and `pca_resized.ipynb`
- **Notebooks:** Contains the main code files:
    - `pca_resized.ipynb`: Resizes palm images, performs PCA, and extracts weight templates and eigenpalms.
    - `basic_diffusion_model_128.py`: Trains a diffusion model with a simple architecture.
    - `basic_unet_model_128.py`: Trains a diffusion model with a U-Net architecture.

- **requirements.txt:** Lists the project dependencies.

- **basic_wrapper.sh & basic.sub:** Scripts to run `basic_diffusion_model_128.py` on a GPU cluster.

- **unet_wrapper.sh & unet.sub:** Scripts to run `basic_unet_model_128.py` on a GPU cluster.


## Running the Code

This project is designed to be run on a GPU cluster due to the computational resources required.

### 1. Prepare the environment

- Install the required dependencies: `pip install -r requirements.txt`
- Ensure you have access to a GPU cluster with PyTorch installed.

### 2. Preprocess data and extract weight templates

- Run the `pca_resized.ipynb` notebook to:
    - Resize palm images to 128x128.
    - Perform Principal Component Analysis (PCA).
    - Extract weight templates and eigenpalms.

### 3. Train the diffusion models

#### Basic Model

- Modify `basic_wrapper.sh` to activate your PyTorch environment and specify the output and error log file paths (e.g., `output_basic.out`, `output_basic.err`).
- Submit the job to the cluster using `basic.sub`.

#### U-Net Model

- Modify `unet_wrapper.sh` similarly to `basic_wrapper.sh`.
- Submit the job to the cluster using `unet.sub`.

### 4. Outputs

- The trained models will generate palm images from the learned weight template representation.
- Training progress and logs will be saved in the specified output and error files (e.g., `output_basic.out`, `output_basic.err`, `output_unet.out`, `output_unet.err`).
- You can expect the U-Net model to potentially produce higher-quality images due to its more advanced architecture.