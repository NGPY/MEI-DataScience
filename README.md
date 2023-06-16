# MEI-DataScience
Codebase for the MEI Data Science course

# Key code snippets

## Pre-processing of data
### Get info about data types
``data_set.info()``
### Display a subset
``data_set[data_set['parameter'] == 'blank']``
### Take a slice of data
``new_data_set = data_set[data_set['parameter'] == 'blank'].copy()``
## Exploratory data analysis
### Get summary of numerical feature
``data_set['variable'].describe()``
### Draw a boxplot
``sns.catplot(kind='box', x='variable', data=data_set, aspect=2)``

![image](https://github.com/NGPY/MEI-DataScience/assets/78988335/e3d626e6-941e-4bf1-b4b1-4a60ac0bad88)
### Draw a boxplot based grouped by categorical features
``sns.catplot(kind='box', x='variable',y='category', data=data_set, aspect=2)``

![image](https://github.com/NGPY/MEI-DataScience/assets/78988335/b63651b4-e4ac-4fbb-82c8-c6a886de5bcd)

### Two way tables
``pd.crosstab(data_set['variable'], data_set['variable2'])``

### Two way table with mean of third variable
``pd.crosstab(data_set['variable'], data_set['variable2'], values=data_set['variable3], aggfunc='mean').round(2)``

### Violin plot
``sns.catplot(data=data_set, kind='violin', x='variable', y='variable', hue='variable', aspect=2``

![image](https://github.com/NGPY/MEI-DataScience/assets/78988335/e7a0f60a-f13a-4918-bbb7-55ac983b88e9)

### Histogram
``sns.displot(data=data_set, x='variable', col='Region');

![image](https://github.com/NGPY/MEI-DataScience/assets/78988335/e3f22be9-d3a0-44b6-8157-26364f159f5f)

### Create kernal density estimate to display distribution
``sns.displot(data=data_set, kind='kde', x='variable', hue='variable', aspect=2);``

![image](https://github.com/NGPY/MEI-DataScience/assets/78988335/b9984c2f-7f33-4a91-b076-91dbb3a44d1e)
## Scatter plot
``sns.relplot(data=data_set, x='variable', y='variable', hue='variable', style='variable', s=100)``

![image](https://github.com/NGPY/MEI-DataScience/assets/78988335/09dd63cb-f38f-4ef8-ace7-114086049dc2)
## Create all possible scatter plots for two numerical features
``sns.pairplot(data=data_set, x_vars=['var', 'var', 'var'], y_vars=['var']);``

![image](https://github.com/NGPY/MEI-DataScience/assets/78988335/8c6471f3-57b3-44d2-beb7-adc7b9aadc54)
### LINEAR REGRESSION
## The ultimate code block
```# Parameters
input_features = ['Mass', 'EngineSize']
target_features = ['CO2']
train_size = 0.8
target_set = diesel_16_data

# Data
input_data = target_set[input_features]
target_data = target_set[target_features]

# Data split
training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(input_data, target_data, train_size=train_size, random_state=1)

# Training
linear_model = LinearRegression().fit(training_inputs, training_targets)
testing_predictions = linear_model.predict(testing_inputs)

# Display parameters
print(f'Input features: {input_features}')
print(f'Target data: {target_features}')
print('Coefficients: ', linear_model.coef_)
print('Intercept: ', linear_model.intercept_)

# Display accuracy
print('RMSE: ',mean_squared_error(testing_targets, testing_predictions, squared=False))
print('R²: ',100*r2_score(testing_targets, testing_predictions))

# Display final equation
final_equation = 'y = '
coefs = linear_model.coef_[0]
for c in coefs:
    final_equation += f'{c.round(3)}x + '
final_equation += str(linear_model.intercept_[0].round(3))
print(f'Final equation: {final_equation}')
