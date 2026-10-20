import torch
from diffusers import AutoPipelineForText2Image

print("Cargando Modelo...")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    dtype=torch.float32,      
    use_safetensors=True,
    low_cpu_mem_usage=True,
)
modelo = modelo.to("cpu")
modelo.enable_attention_slicing()

torch.set_num_threads(torch.get_num_threads())

prompt = input("Escribe el prompt de la imagen: ")

imagen = modelo(
    prompt=prompt,
    num_inference_steps=4,     
    guidance_scale=0.0,        
    height=512,
    width=512,
).images[0]

imagen.save("imagen.png")
print("Imagen Guardada en imagen.png")
