# csci335-635-movie-reccommender
## Abstract  

This project develops a personalized movie recommendation system using the MovieLens 100K dataset. The goal is to predict how a user would rate a movie they have not seen yet and compare multiple recommendation approaches. The project includes baseline, KNN, SVD / matrix factorization, linear regression, and neural network models. Model performance is evaluated using RMSE and MAE.

## Developers

- Evan Albrecht
- Stephen Moulton
- Michael Chang
- Connor Patterson 


## Dataset

This project uses the MovieLens 100K dataset.

Used files:
- `u.data`
- `u.item`
- `u.user`

These should be placed in:

```text
data/ml-100k/
```

## Requirements

This project uses the following Python packages:

- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib

## Setup

### 1. Create a virtual environment

From the project root folder, run:

```bash
py -3.12 -m venv .venv
```

### 2. Activate the virtual environment

```bash
.venv\Scripts\Activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


## Running the Project

Make sure you navigate to the `code/` folder.

### Run the traditional recommendation models

```bash
python recommender.py
```

This runs:
- Average baseline
- KNN
- SVD / Matrix Factorization
- Linear Regression

It prints:
- dataset information
- preprocessing information
- train/test split information
- RMSE and MAE for each traditional model

### Run the neural network model

```bash
python neural_model.py
```

This runs:
- neural network preprocessing
- neural network model creation
- neural network training
- neural network evaluation

It prints:
- train/test shapes
- encoded feature counts
- model summary
- training progress by epoch
- final RMSE and MAE

### Run the comparison / chart script

```bash
python main.py
```

main.py script:
- runs `recommender.py`
- runs `neural_model.py`
- collects RMSE and MAE values
- generates bar charts comparing model performance

## Models Used

- ##### Average Baseline


- ##### KNN


- ##### SVD / Matrix Factorization


- ##### Linear Regression


- ##### Neural Network


## Evaluation

Models are evaluated using:

- **RMSE (Root Mean Square Error)**
- **MAE (Mean Absolute Error)**
