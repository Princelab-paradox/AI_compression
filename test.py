import gradio as gr

def process_image(input_image):
    # Process the input_image and create the output_image
    output_image = "./result/dance.jpg"  # Replace this line with your image processing code
    return output_image,output_image,output_image

# Create the Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(),
    outputs=[gr.Image(), gr.File(extension=".png"), gr.File(label="Download Image", download=True, extension=".png")],
    title="Image Processing App",
    description="Upload an image, process it, and download the output image."
)

# Launch the Gradio app
iface.launch()