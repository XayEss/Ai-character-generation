# Video Game Character Generator Using Generative AI: Stable Diffusion, Dreambooth and LoRA

## Abstract
In the realm of video game development, the creation of unique and engaging characters plays a pivotal role in crafting immersive experiences. This project explores the innovative use of Stable Diffusion, an advanced deep learning model, for the generation of video game characters, enhanced through fine-tuning methodologies like DreamBooth and LoRA. The project focuses on leveraging the generative AI models to automate the creation of highly variative video game characters. By responding to a wide array of descriptive prompts such as 'cyberpunk', 'mythological' and 'post-apocalyptic', this project allows the generation of diverse, thematic character skins, enhancing both creativity and efficiency in game design.

## Introduction
### The Problem
Creating diverse and thematic character skins in video games is a resource-intensive task. It involves many iterations of creating designs and editing them. Sometimes samples can be discarded and new ones have to be created, which requires a lot of ideas and creativity. This project aims to boost the process of creatining drafts by streamlining this process with automated skin generation using AI, responsive to a variety of thematic prompts.

### Motivation and Interest
The motivation is deeply connected with evolving needs of the video game industry in the realm of character designing. As games become vast and complex in details more characters are required to populate the in-game world, even smaller companies might have difficulties to allocate resources for quality designs due to the lack of them. Making this resource intensive work of graphical designers more eficient can save a lot of time and efforts. This project sits at the confluence of AI technology and creative game design, with the potential to revolutionize character skin creation. Automating this process will not only save time and resources but also allow for greater creative exploration in game design.

### Proposed Approach
Our approach utilizes the DreamBooth and LoRA techniques, fine-tuned with a dataset of character images. The AI is trained to generate skins based on a wide range of prompts like 'cyberpunk', 'mythological' and 'post-apocalyptic', ensuring versatility and adaptability in design. 
DreamBooth - this approach changes all weights of the model associating it with some keywoard the model is trained on. Using this approach the model can learn how to generate many new and different characters and even change the generation style. For example, the base model was trained on realistic images, then generating fictional, art styled character will produce worse results, however with a lot of fine-tuning with the DreamBooth technique, the model will learn how to produce art styled images. The model produced is of the size of the original(4gb+).
LoRA(Low Rank Adaptation) - another model is trained which adds it's weights to the cross attention layer of the base diffusion model to produce results as close to the images it was trained on. This approach is good at changing some aspect of the generation. There can be character, style, pose, concept, clothing LoRAs that enhance some detail of the image. This is suitable to generate high-quality images of a certaing character. The LoRA models can weight from 20mb to 300mb which is much smaller than the base models.

### Rationale
Stable Diffusion represents a cutting-edge generative model, but its potential in specific domains like video game character design is yet to be fully explored. The model open-sourceness enables the possibility for customization and fine-tuning, making it a prefect choice for our goal.
The choice of DreamBooth and a diverse range of prompts aims to surpass the limitations of traditional AI models in terms of creative freedom and thematic accuracy. DreamBooth fine-tunes the model to understand previously unknown concepts. By combining those concepts the model is capable of creating diverse and unseen representations of the characters.
LoRa doesn't fine-tune the base model so it does learn only one aspect making it impossible to understand multiple characters. The base model might not produce quality images when applying known aspects but unseen with this this specific character. This is where LoRAs shine, they are able to introduce new notion to already known by learning it thoroughly. Character LoRAs can enhance the generation of certain characters, pose LoRAs - make the charater stand in a new pose, style LoRA - change the style of the image, clothing LoRAs - make the character dress a specific clothing.
Even with two good approaches the exceptional part is that they can be combined, producing the unbelievable results.

### Key Components and Limitations
The project hinges on the fine-tuned DreamBooth model and the comprehensiveness of the dataset. One limitation may be the AI's interpretation accuracy of complex or abstract themes. Another limitation is the resolution of our images which is lower than the standard for stable diffusion and leads to blurryness. Last limitation is the limited computational resources, with a dataset this size learning all characters requires a lot of training, which could not be achieved given our resources and time bounds.

## Setup
### Dataset
The dataset contains a total of 32,000 images with 50 images for each of the 646 characters. This was created by web scrapping using ```BeautifulSoup``` and ```Selenium```
The images sizes are approximately 200x200 pixels which is smaller than the 512x512 the stable diffusion model was trained, which will be upscaled and thus produces a more blurry picture after fine-tuning.

The ```WebScrapper.ipynb``` was used to collect images from bing for the dataset. This notebook can be run by executing all its cells. It requires a csv file with columns "Game" and "Character" along with specifying base_dir variable for the save directory.

The web scraping notebook ```web-scraping.ipynb``` can be run to download images for a desired character. The search_prompt variable describes what images to search on google images. Tha variables download_path specifies the image download location and charachter_name under which to save the images.

Use ```dataset_formater.ipynb``` to format the dataset in a specific structure. Given a csv file with character names and a directory with images extracts them into a new directory with this structure: "Game name"/"character name"/"images" format. Takes a list of Image_name:labels pairs and creates a directory with the same structure but for text files with image labels in the specified location.

### Experimental Setup
For the model we used: ```runwayml/stable-diffusion-v1-5``` from huggingface.
For the variational auto encoder [vae] we used: ```stabilityai/sd-vae-ft-mse```
For finetuning we used ```DreamBooth``` from the diffusers library.
For finetuning we used ```LoRA``` from the diffusers library.

Training was done on a system with an RTX 4070 [12 GB of VRAM], 32 GB of RAM on a WSL environment.


### Problem Setup
DreamBooth:
StableDiffustion v1.5 takes in images of size 512x512 and for our training, we have used the parameters as:

precision - fp16

train_batch_size - 1

use_8bit_adam - TRUE

To keep the VRAM used just under 12GB

LoRa:

precision - fp16

pretrained_model_name_or_path - runwayml/stable-diffusion-v1-5
dataloader_num_workers - 0 
resolution - 512 
center_crop 
random_flip 
train_batch_size - 2 
gradient_accumulation_steps - 4 
max_train_steps - 15000 
learning_rate - 1e-04 
max_grad_norm: 1 
lr_scheduler: "cosine" 
lr_warmup_steps - 0 
checkpointing_steps - 1000 
validation_epochs - 10

## Train process
The above variables can be customized inside the script file for DreamBooth or LoRA training.

DreamBooth:

Open ```DB-Finetuning.ipynb``` notebook and customize the variables as desired. This code utilizes the ```train_dreambooth.py``` script from the ```diffusers``` library. Customize MODEL_NAME to change the model and OUTPUT_DIR for the output directory. Edit the concepts list to add new concepts to train on. The concept list specifies a dictionaries with the folloeing parameters: instance_prompt - the prompt to associate images with unique words; class_prompt-general propt without any unique words; instance_data_dir - the directory with images we want to teach on;  class_data_dir - the directory of images with concepts we want to preserve. Run all the cells to train the model. Run the inference section to generate images using the trained model, th model_path and promp have to be specified.

LoRA:

A LoRA can be trained by using the ```lora-train.ipynb``` notebook, MODEL_NAME, OUTPUT_DIR, DATASET_NAME have to be specified to before running the notebook. DATASET_NAME specifies the name of the dataset as on huggingface hub. To create a custom dataset the 'dataset_path' and 'labels_paths' variables need to point to the images and labels, where labels is a folder of txt files where the name of the file matches the name of the corresponding image. Still, DATASET_NAME has to be correctly pointing to the newly uploaded dataset. Other training parameters can be specified as desired. This code utilizes the training script ```train_text_to_image_lora.py``` from the ```diffusers``` library. After training the model will be saved to th OUTPUT_DIR location.

## Results
### Main Results
Initial results are promising, showing the model's capability to generate unique and thematic skins while adhereing to the prompt instructions. Here is a comparison of generating the character Cypher from Valorant:
![alt text](https://images.contentstack.io/v3/assets/bltb6530b271fddd0b1/blt158572ec37653cf3/5eb7cdc19df5cf37047009d1/V_AGENTS_587x900_Cypher.png)

The following promt was used to generate the images:
3d, blur censor, blurry, blurry background, blurry foreground, bokeh, cellphone picture, chromatic aberration, concert, cosplay photo, depth of field, female pov, figure, film grain, focused, glowstick, meta, money, motion blur, multiple girls, photo \(medium\), photo \(object\), photo background, photo inset, photorealistic, poster \(object\), pov, pov hands, rainbow, reference inset, shopping, stadium, taking picture, unconventional media, cypher, valorant
The results produced by different models:

Base stable-diffusion-v1.5:
![alt text](https://github.com/XayEss/Ai-character-generation/blob/main/images/cypher-notune.png?raw=true)

DreamBooth fine-tuned:
![alt text](https://github.com/XayEss/Ai-character-generation/blob/main/images/cypher-db.png?raw=true)

LoRA:
![alt text](https://github.com/XayEss/Ai-character-generation/blob/main/images/cypher-lora.png?raw=true)

Here we can see the comparison on the generated images. Base stable diffusion generated incoherent noise and didn't know about the character at all. DreamBooth produced a recognizable picture of Cypher that is not precise, still elements of his character could be see, this is an impressive leap compared to the not fine-tuned model. The LoRA produced result image is the closes to the original character, which shows the high-quality approach of Low Rank Adaptations. However the image is all pixelated, mostly due to the lower resolution of our data. We can see that both approaches have different results, the DreamBooth fine-tuned model was able to learn the character to a limited extent which migh be because of insufficient training, nevertheless the picture stayed clean, as the model knows how to produce high-quality images. On the other hand, LoRA managed to capture the character in all details but also captured the low resolution of our data. Furher is a comparison of stable-diffusion-v1.5 to a couterfeitv3.0 model, which was already fully fine-tuned using the DreamBooth technique. This model was trained on different art, anime styled character and is better at producing them.
This is a picture of a character from League of Legends named Ahri:
![alt text](https://ddragon.leagueoflegends.com/cdn/img/champion/splash/Ahri_0.jpg)

This is a picture of Ahri created by stable-diffusion-v1.5 with Ahri LoRA applied:

This is a picture of Ahri created by couterfeitv3.0 with Ahri LoRA applied:


### Supplementary Results
Parameters were selected so that the model could be trained on the system available.

## Discussion

## Conclusion
The fine-tuned model was able to make better video game characters following the prompt when compared to the baseline model.

## Future Scope
- The next step is to try different models with this approach.
- Train the model on a high-resolution dataset
- Comibne LoRA and our DreamBooth trained model.
- Explore the use of LyCORIS for model fine-tuning

## References
- https://huggingface.co/runwayml/stable-diffusion-v1-5
- https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth
- https://huggingface.co/blog/stable_diffusion
- https://huggingface.co/docs/diffusers/training/lora?installation=PyTorch
- https://civitai.com/models/4468?modelVersionId=57618
- https://github.com/AUTOMATIC1111/stable-diffusion-webui
