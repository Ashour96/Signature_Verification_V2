import pandas as pd

new_data = pd.DataFrame({
    "Name": ["Ahmed", "Mohamed"],
    "Age": [25, 30],
    "Job": ["Engineer", "Analyst"]
})

new_data.to_csv(
    "employees.csv",
    mode="a",
    header=False,
    index=False
)