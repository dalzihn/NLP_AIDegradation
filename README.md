# AI Degradation: A case study on Sentiment Analysis
 

## Project Description
The project devles into the topic of AI Degradation, where it experiemented on the case of Sentiment Analysis using a dataset consisting of the [UIT-VSFC](https://www.researchgate.net/publication/329645066_UIT-VSFC_Vietnamese_Students'_Feedback_Corpus_for_Sentiment_Analysis), the [UIT-ViSFD](https://github.com/LuongPhan/UIT-ViSFD) and the [Vietnamese Student Feedback](https://www.kaggle.com/datasets/linhlpv/vietnamese-sentiment-analyst/data).

The approach for simulating AI Degradation is illustrated in the following figure:

![The training-workflow](assets/training_workflow1.png)

![The training-workflow](assets/training_workflow2.png)

- First, splitting the data into three subsets: training set, test set and simulation set.

- Second, obtaining the first version of models by training on the training set. Afterwards, only the classification layer is allowed for further learning. 

- Third, further splitting the simulation into $n$ smaller subfolds, where each subfold is used to train a new version of models. The labels of each subfold will be removed and replaced by using the model from previous generation to predict the labels.

The structure of the project is as follows:

```
code
└── data_helper.py 
└── models.py 
└── training_helper.py 
└── experiment.py 
```

```data_helper.py``` consists of the preprocessing functions to process data.

```models.py``` consists of the model architecture conducted in the project.

```training_helper.py``` consists of the training functios of models deployed in PyTorch.

```experiment.ipynb``` responsible for running the experiment.

## Getting Started
To run the experiment, first using the ```requirement.txt``` file to install the required packages. Afterwards, the experiment can be conducted by running the ```experiment.ipynb```.

## Acknowledgement
I would like to express sincere gratitude to UEH and BIT for providing academic support, to Dr. Đặng Ngọc Hoàng Thành for constant support and guiding throughout the project, to my family members for their encouragement, to all my friends and DS001 members for always lending a helping hand. 