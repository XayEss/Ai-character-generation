# Video Game Character Generator Using Generative AI: Stable Diffusion and Dreambooth

## Abstract
This project focuses on leveraging the DreamBooth generative AI model to automate the creation of video game character skins. By responding to a wide array of descriptive prompts such as 'cyberpunk', 'mythological' and 'post-apocalyptic', our goal is to generate diverse, thematic character skins, enhancing both creativity and efficiency in game design.

## Introduction
### The Problem
Creating diverse and thematic character skins in video games is a resource-intensive task. This project aims to streamline this process by automating skin generation using AI, responsive to a variety of thematic prompts.

### Motivation and Interest
This project sits at the confluence of AI technology and creative game design, with the potential to revolutionize character skin creation. Automating this process will not only save time and resources but also allow for greater creative exploration in game design.

### Proposed Approach
Our approach utilizes the DreamBooth model, fine-tuned with a dataset of character images. The AI is trained to generate skins based on a wide range of prompts like 'cyberpunk', 'mythological' and 'post-apocalyptic', ensuring versatility and adaptability in design.

### Rationale
The choice of DreamBooth and a diverse range of prompts aims to surpass the limitations of traditional AI models in terms of creative freedom and thematic accuracy.

### Key Components and Limitations
The project hinges on the fine-tuned DreamBooth model and the comprehensiveness of the dataset. One limitation may be the AI's interpretation accuracy of complex or abstract themes.

## Setup
### Dataset
The dataset contains a total of 32,000 images with 50 images for each of the 646 characters. This was created by web scrapping using ```BeautifulSoup```

### Experimental Setup
For the model we used: ```runwayml/stable-diffusion-v1-5``` from huggingface.
For the variational auto encoder [vae] we used: ```stabilityai/sd-vae-ft-mse```
For finetuning we used ```DreamBooth``` from the diffusers library.

Training was done on a system with an RTX 4070 [12 GB of VRAM], 32 GB of RAM on a WSL environment.


### Problem Setup
StableDiffustion v1.5 takes in images of size 512x512 and for our training, we have used the parameters as:

precision - fp16

train_batch_size - 1

use_8bit_adam - TRUE

To keep the VRAM used just under 12GB




## Results
### Main Results
Initial results are promising, showing the model's capability to generate unique and thematic skins while adhereing to the prompt instructions.

### Supplementary Results
Parameters were selected so that the model could be trained on the system available.

## Discussion

## Conclusion
The fine-tuned model was able to make better video game characters following the prompt when compared to the baseline model.

## Future Scope
The next step is to try different models with this approach, like LoRA.

## References
- https://huggingface.co/runwayml/stable-diffusion-v1-5
- https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth
- https://huggingface.co/blog/stable_diffusion
