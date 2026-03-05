import pandas as pd
from sklearn.ensemble import *
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_data(path: str = "teams.csv") -> pd.DataFrame:
	teams = pd.read_csv(path)
	selected_columns = [
		"team",
		"country",
		"year",
		"events",
		"athletes",
		"age",
		"height",
		"weight",
		"prev_medals",
		"prev_3_medals",
		"medals",
	]
	teams = teams[selected_columns].copy()
	return teams


def build_models() -> dict:
	return {
		"LinearRegression": LinearRegression(),
		"RandomForest": RandomForestRegressor(
			n_estimators=300,
			random_state=42,
			min_samples_leaf=2,
			n_jobs=-1,
		),
		"GradientBoosting": GradientBoostingRegressor(random_state=42),
	}


def evaluate_models(df: pd.DataFrame):
	feature_columns = [
		"year",
		"events",
		"athletes",
		"age",
		"height",
		"weight",
		"prev_medals",
		"prev_3_medals",
	]
	X = df[feature_columns]
	y = df["medals"]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	results = []
	best_name = None
	best_model = None
	best_mae = float("inf")

	for name, model in build_models().items():
		pipeline = Pipeline(
			steps=[
				("imputer", SimpleImputer(strategy="median")),
				("model", model),
			]
		)
		pipeline.fit(X_train, y_train)
		preds = pipeline.predict(X_test)

		mae = mean_absolute_error(y_test, preds)
		rmse = mean_squared_error(y_test, preds) ** 0.5
		r2 = r2_score(y_test, preds)

		results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

		if mae < best_mae:
			best_mae = mae
			best_name = name
			best_model = pipeline

	metrics = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
	return best_name, best_model, metrics


def predict_next_games(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
	latest_year = int(df["year"].max())
	next_year = latest_year + 4

	latest_snapshot = (
		df.sort_values(["team", "year"]) 
		.groupby("team", as_index=False)
		.tail(1)
		.copy()
	)
	latest_snapshot["year"] = next_year

	feature_columns = [
		"year",
		"events",
		"athletes",
		"age",
		"height",
		"weight",
		"prev_medals",
		"prev_3_medals",
	]

	latest_snapshot["predicted_medals"] = model.predict(latest_snapshot[feature_columns])
	latest_snapshot["predicted_medals"] = (
		latest_snapshot["predicted_medals"].clip(lower=0).round(0).astype(int)
	)

	top_predictions = latest_snapshot[["team", "country", "year", "predicted_medals"]]
	top_predictions = top_predictions.sort_values("predicted_medals", ascending=False)
	return top_predictions


def main():
	df = load_data()
	print("Loaded data shape:", df.shape)

	best_name, best_model, metrics = evaluate_models(df)
	print("\nModel comparison:")
	print(metrics.to_string(index=False))
	print(f"\nBest model by MAE: {best_name}")

	predictions = predict_next_games(df, best_model)
	print("\nTop 20 predicted medal totals for next Olympics:")
	print(predictions.head(20).to_string(index=False))


if __name__ == "__main__":
	main()






