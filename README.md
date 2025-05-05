# Layer-wise Relevance Propagation in PyTorch

This repository demonstrates a flexible framework for **layer-wise relevance propagation** (LRP) suitable for 
VGG- and ResNet-based (as well as any similar) image classification models. LRP is a method within the field of 
**explainable artificial intelligence** (XAI), that yields saliency maps highlighting important features in the input 
image to indicate the underlying decision criteria.

The code was developed as part of a student project in the context of biomedical image classification to aid 
**transparency and fairness** when deploying machine learning (ML)-based diagnostic methods.


## 1. What is XAI?

*“a set of processes and methods that allows human users to comprehend and trust the results and output created by machine learning algorithms”* 

- fundamentally AI models are **black boxes** (-> opaque decision-making)
- hence, XAI is crucial to **build trust** and ensure transparency (and accountability) when deploying AI models 
- often XAI is even **required to comply with regulation**
- XAI **facilitates model evaluation** (accuracy, fairness, biases, …)
- further XAI increases the information gain from AI models (-> more informed decision-making on user side) 

There is a variety of proposed **XAI methods** differing in
- scope:            local (single prediction) vs. global (working principle)
- applicability:    model-agnostic vs. model-specific
- method:           feature importance analysis, visualisation, concept-based explanation, ...

## 2. Saliency Maps with Layer-wise Relevance Propagation (LRP)
- **heatmaps** that act as visualisation tools for feature importance analysis (i.e. highlight the most impactful image regions for computer vision models)
- local and intuitive
- we utilize **LRP for that purpose based on the following desirable properties**:
  - theoretically sound (based on Deep Taylor Decomposition)
  - computationally efficient
  - model-specific
- alternative candidates replacing or complementing LRP are SHAP or Prediction Difference Analysis (PDA)

## 3. Repository Structure
- *src/*: source code directory containing classes and methods
- *notebooks/*: jupyter notebooks demonstrating the workflow separately on a VGG and ResNet based model
- *literature/*: a selection of papers explicating the theoretical underlinings
- *data/*: set of example input data and categories from ImageNet
- *images/*: illustrations for markup cells

## 4. How to Use
### 4.1. Required Modules
For sole source code usage **matplotlib**, **pytorch** and **torchvision** are required.
For the notebooks **ipython** and **ipykernel** are required.
The recommended way is to install both is calling 
`conda env create -f environment.yml`
in terminal in the project directory.

### 4.2. Recommendations
Usage is extensively demonstrated in the two notebooks, and it is advised to follow such procedure when implementing.
The utilized class is LRPEngine from *src/xai/lrp.py*. It conducts the calculation through the following steps:

1. **parsing a model to an equivalent ModuleList** while conducting necessary amendments (converting layers where recommended)
2. **initialising an appropriate LRP model** with each layer containing a suitable LRP-rule for the respective layer in the ModuleList
3. **computing and saving the relevance scores** for each layer
4. displaying or saving the resulting **saliency map**

## 5. Other Important Information
### 5.1. Authors and Acknowledgment
Paul Rüsing - pr@paulruesing.de - single and main contributor

I acknowledge the inspiration drawn from
- https://arxiv.org/pdf/2407.09115
- https://doi.org/10.1007/978-3-030-28954-6_10 
- https://github.com/keio-smilab24/LRP-for-ResNet/
- https://git.tu-berlin.de/gmontavon/lrp-tutorial
- https://github.com/kaifishr/PyTorchRelevancePropagation/blob/master/src/lrp_layers.py


### 5.2. License
The project is licensed under the MIT license. To view a copy of this license, see [LICENSE](https://github.com/paulruesing/lrp-xai-pytorch?tab=MIT-1-ov-file).
