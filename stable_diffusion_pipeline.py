from diffusers import StableDiffusionPipeline

def generate_image(prompt, model_path="CompVis/stable-diffusion-v1-4"):
    pipeline = StableDiffusionPipeline.from_pretrained(model_path)
    pipeline = pipeline.to("cuda")

    image = pipeline(prompt).images[0]
    image.save("generated_image.png")

if __name__ == "__main__":
    prompt = "A futuristic cityscape with flying cars"
    generate_image(prompt)
