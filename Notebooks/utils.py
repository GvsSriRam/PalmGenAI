import matplotlib.pyplot as plt

def save_image_comparison(image, generated_image, filename):
    # Displaying the original and generated images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Generated image
    axes[1].imshow(generated_image, cmap='gray')
    axes[1].set_title("Generated Image")
    axes[1].axis('off')

    # Show the plot
    plt.savefig(filename)
    plt.close()
