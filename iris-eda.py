from sklearn.datasets import load_iris

iris_data = load_iris(as_frame=True)
print(iris_data.keys()) # feature_names is among the keys
df = iris_data.frame

# access the feature only
# slice the df using the column names
# or use df.columns[:-1]
print(df[iris_data.feature_names])

print(df["sepal length (cm)"].apply(lambda x: x+1))
print(df[iris_data.feature_names].apply(lambda x : (x - x.min()) / (x.max() - x.min())))
