# Olympics Medal Prediction Project

This project uses historical Olympic team data to predict how many medals each country team might win in the next Olympic Games.

It is designed as a beginner-friendly machine learning script written in Python.

## What This Program Does

The script in 'Olympics.py':

1. Loads Olympic team data from 'teams.csv'.
2. Trains and compares multiple machine learning models.
3. Measures model quality using common evaluation metrics.
4. Selects the best model automatically.
5. Predicts medal counts for the next Olympics (next year = latest year in data + 4).
6. Prints the top 20 teams with the highest predicted medal totals.

## How The Prediction Works (Simple Explanation)

Each team has information like:

- Number of events entered
- Number of athletes
- Average age, height, and weight
- Previous medal totals

The model learns patterns from past Olympics, then uses the latest available row for each team to estimate medals in the next Olympic cycle.

## Project Files

- Olympics.py: Main Python script (load data, train models, evaluate, predict).
- teams.csv: Input dataset used for training and prediction.
- README.md: Documentation file.

## Requirements

- Python 3.11 (recommended)
- Packages:
	- pandas
	- scikit-learn

## Setup Instructions

### 1. Open the project folder

Open this folder in VS Code:

e.g: c:\Users\Kamva\Olympics.py


### 2. Install dependencies


pip install pandas scikit-learn'

## How To Run

From the project root:

'python Olympics.py'

## Expected Output

You will see output similar to:

- Data shape loaded from CSV
- A model comparison table with:
	- MAE (Mean Absolute Error, lower is better)
	- RMSE (Root Mean Squared Error, lower is better)
	- R2 (coefficient of determination, higher is better)
- The best model name
- Top 20 predicted teams for the next Olympics

## Data Columns Used

The script uses the following columns from 'teams.csv':

- team
- country
- year
- events
- athletes
- age
- height
- weight
- prev_medals
- prev_3_medals
- medals (target value for training)

## Notes and Limitations

- Predictions are estimates, not guaranteed outcomes.
- Model quality depends on data quality and historical trends.
- External factors (injuries, politics, funding changes, rule changes) are not directly modeled.

## Troubleshooting

- If 'ModuleNotFoundError' appears, install missing packages with pip install <package_name>.


Then activate the virtual environment again.

## Future Improvements

- Save predictions to a CSV file.
- Add plots to visualize model performance.
- Use cross-validation for more reliable evaluation.
- Add feature engineering and hyperparameter tuning.