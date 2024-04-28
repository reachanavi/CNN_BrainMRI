---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for CNN to Classify Brain Tumors Based on MRI Image

<!-- Provide a quick summary of what the model is/does. -->

This model aims to classify brain MRI images into four different categories of brain tumors using a convolutional neural network.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model aims to classify brain MRI images into four different categories of brain tumors using a convolutional neural network. It classifies MRIs as pituitary tumor, meningioma, glioma, or no tumor. The model uses two convolutional layers with max pooling and a nonlinearity.

- **Developed by:** Anavi Nayak, Dia Jain
<!-- - **Funded by [optional]:** {{ funded_by | default("[More Information Needed]", true)}} -->
<!-- - **Shared by [optional]:** {{ shared_by | default("[More Information Needed]", true)}} -->
- **Model type:** Convolutional Neural Network
- **Language(s) (NLP):** Python, PyTorch
<!-- - **License:** {{ license | default("[More Information Needed]", true)}} -->
<!-- - **Finetuned from model [optional]:** {{ base_model | default("[More Information Needed]", true)}} -->

<!-- ### Model Sources [optional] -->

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/reachanavi/CNN_BrainMRI
<!-- - **Paper [optional]:** {{ paper | default("[More Information Needed]", true)}} -->
<!-- - **Demo [optional]:** {{ demo | default("[More Information Needed]", true)}} -->

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
Our project focused on building a convolutional neural network (CNN) to classify brain tumors from MRI images into four types - pituitary, meningioma, glioma, and no tumor. There are several motivations behind building a model to accomplish this task. First, there is immense clinical importance surrounding this area. Brain tumors of different types each have different treatment pathways and prognoses, and may be best suited for different doctors. Timely and accurate classification of brain tumors using MRI imaging is crucial to create an effective treatment plan and improve overall patient outcomes. Automating this process using CNNs or other models can significantly enhance diagnostic accuracy and speed, thereby allowing for earlier intervention. In addition, it can be used as an additional check once a doctor has classified a tumor in order to improve accuracy. In recent years, CNNs have emerged as a powerful tool for image recognition and classification tasks due to their ability to capture spatial hierarchies in images. Leveraging these advancements and applying CNNs to medical imaging is an important next step for critical healthcare applications.

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

To test this model, put your MRI images in a "data/Testing" directory inside the project directory, then run the Jupyter notebook.

<!-- ### Downstream Use [optional] -->

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

<!-- {{ downstream_use | default("[More Information Needed]", true)}} -->

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->
This model should not be used with sensitive patient information that is not publicly available.
<!-- {{ out_of_scope_use | default("[More Information Needed]", true)}} -->

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->
This model is bias to adult brain MRI images in a clear resolution.
<!-- {{ bias_risks_limitations | default("[More Information Needed]", true)}} -->

<!-- ### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

<!-- {{ bias_recommendations | default("Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.", true)}} --> 

## How to Get Started with the Model

<!-- Use the code below to get started with the model.

{{ get_started_code | default("[More Information Needed]", true)}} -->

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The training data was obtained from this kaggle dataset: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

JPG images were preprocessed using torchvision and the PIL libraries into RGB tensors.


#### Training Hyperparameters

- Learning Rate = 0.001
- Criterion = Cross Entropy Loss
- Optimizer = Adam


## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

The testing data was obtained from the same kaggle dataset: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

The metrics of accuracy as well as loss over the course of training were measured. We plotted losses to ensure that they were consistently going down and our learning rate was effective. We measured accuracy to see if the model was generalizable to different test sets.

### Results

We were able to achieve a 77% accuracy with the addition of batch normalization regularization.


### Model Architecture and Objective

The model contained the following layers:
- 2D Convolutional Layer
- Batch Normalization
- Max Pooling
- 2D Convolutional Layer
- Batch Normalization
- Max Pooling
- ReLU for non-linearity


#### Hardware

The model was run on an Apple M1 chip.

#### Software

The project was created using Jupyter Notebook.

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

Badža MM, Barjaktarović MČ. Classification of Brain Tumors from MRI Images Using a Convolutional Neural Network. Applied Sciences. 2020; 10(6):1999. https://doi.org/10.3390/app10061999

## Model Card Contact

Anavi Nayak - anavinayak@utexas.edu
Dia Jain - diajain@utexas.edu
