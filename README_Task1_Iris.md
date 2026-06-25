# Task 1: Exploring and Visualizing the Iris Dataset

**Intern:** Junaid

**Internship:** DevelopersHub Corporation — AI/ML Engineering Internship

## Objective

Load, inspect, and visualize the Iris dataset to understand data trends, distributions, and outliers using pandas, matplotlib, and seaborn.

## Dataset

- **Source:** Classic Iris dataset, loaded directly via `seaborn.load_dataset("iris")` (no manual download needed)
- **Records:** 150 samples, 4 numeric features, 1 target column (`species`)
- **Classes:** 3 perfectly balanced species — `setosa`, `versicolor`, `virginica` (50 each)

## Features

| Feature | Description |
|---|---|
| `sepal_length` | Sepal length (cm) |
| `sepal_width` | Sepal width (cm) |
| `petal_length` | Petal length (cm) |
| `petal_width` | Petal width (cm) |

**Target:** `species`

## Data Cleaning

No cleaning was required — the dataset has no missing values and consistent data types across all columns.

## Exploratory Data Analysis

- **Initial inspection:** Shape, column names, dtypes, and descriptive statistics (count, mean, std, min, max, quartiles) were reviewed via `.info()` and `.describe()`.
- **Class distribution:** Confirmed perfectly balanced classes (50 samples per species).
- **Pairplot:** Visualized all pairwise feature relationships colored by species.
- **Scatter plot:** Focused view of sepal length vs. petal length by species.
- **Histograms:** Distribution of each numeric feature, split by species.
- **Box plots:** Used to detect outliers within each feature, split by species.
- **Correlation heatmap:** Quantified linear relationships between all numeric features.

## Key Findings

1. **Dataset overview:** 150 samples, 4 numeric features, 1 target (species); 3 perfectly balanced classes; no missing values.
2. **Feature distributions:** `setosa` is clearly separable from `versicolor`/`virginica` in petal features. Sepal width is the least discriminative feature, while petal length and width show the most class separation.
3. **Outliers:** Sepal width in `setosa` has a few mild outliers in the upper range; petal features are tightly grouped within each class.
4. **Correlations:** Petal length and petal width are highly correlated (r ≈ 0.96). Sepal width has near-zero or negative correlation with petal features.
5. **Scatter plots:** `setosa` is linearly separable from the other two classes; `versicolor` and `virginica` overlap slightly in sepal features. Petal length vs. petal width gives the best 2D separation.

## Files

- `Task1_Iris_EDA.ipynb` — full notebook (inspection, statistics, and all visualizations)
- `README.md` — this file
- `task1_pairplot.png`, `task1_scatter.png`, `task1_histograms.png`, `task1_heatmap.png` — saved plot images (generated when the notebook is run)

## How to Run

1. Open the notebook in Jupyter, VS Code, or Kaggle Notebooks.
2. Run all cells top to bottom. The dataset downloads automatically via seaborn on first use — no manual file needed.
3. If internet access is restricted (e.g., a Kaggle kernel with internet off), load the data locally instead:
   ```python
   from sklearn.datasets import load_iris
   iris_obj = load_iris(as_frame=True)
   iris = iris_obj.frame
   iris["species"] = pd.Categorical.from_codes(iris_obj.target, iris_obj.target_names)
   ```

## Possible Improvements

- Add pairwise statistical tests (e.g., ANOVA) to formally confirm which features separate species best.
- Apply dimensionality reduction (PCA) to visualize all 4 features in 2D at once.
- Extend the EDA into a simple classification model (e.g., KNN or logistic regression) as a natural next step.
