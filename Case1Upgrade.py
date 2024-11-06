import pandas as pd
import numpy as np
import streamlit as st
import  plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
plt.style.use('seaborn-v0_8')

df1 = pd.read_csv('gender_submission.csv')
df2 = pd.read_csv('test.csv')
df2['Age'] = df2['Age'].fillna(df2['Age'].mean())
df2['Fare'] = df2['Fare'].fillna(df2['Fare'].mean())
df2_2 = df2.merge(df1)



df3 = pd.read_csv('train.csv')
df3['Age'] = df3['Age'].fillna(df3['Age'].mean())
df3['Embarked'] = df3['Embarked'].fillna('S')

nan_count = df2.isna().sum()

print(nan_count)

y = df3['Survived']

features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
X = pd.get_dummies(df3[features])
X_test = pd.get_dummies(df2[features])

model = RandomForestClassifier(n_estimators=95, max_depth=6, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': df2.PassengerId, 'Survived': predictions})
#print(output)



# Streamlit app layout
st.title("Titanic: Age vs Fare with Trendlines")

# Checkbox for showing Dood (not survived) scatterplot and trendline
show_dood = st.checkbox('Show Overleden Data')
show_levend = st.checkbox('Show Overleefd Data')

# Create subplots
fig, ax = plt.subplots(figsize=(10, 6))

if show_dood:
    df3_dood = df3[df3['Survived'] == 0]
    sns.scatterplot(data=df3_dood, x='Age', y='Fare', label='Dood', color='red', ax=ax)

    # Linear regression for Dood
    ages_reshaped = df3_dood['Age'].values.reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(ages_reshaped, df3_dood['Fare'])

    # Create trendline
    x_min, x_max = df3_dood['Age'].min(), df3_dood['Age'].max()
    x_values = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_values = regressor.predict(x_values)
    
    # Plot trendline for Dood
    ax.plot(x_values, y_values, color='red', label='Dood Trendline')

if show_levend:
    df3_levend = df3[df3['Survived'] == 1]
    sns.scatterplot(data=df3_levend, x='Age', y='Fare', label='Levend', color='blue', ax=ax)

    # Linear regression for Levend
    ages_reshaped = df3_levend['Age'].values.reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(ages_reshaped, df3_levend['Fare'])

    # Create trendline
    x_min, x_max = df3_levend['Age'].min(), df3_levend['Age'].max()
    x_values = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_values = regressor.predict(x_values)
    
    # Plot trendline for Levend
    ax.plot(x_values, y_values, color='blue', label='Levend Trendline')

# Set axis labels and title
ax.set_xlabel('Leeftijd [jaar]')
ax.set_ylabel('Fare prijs [£]')
ax.set_title('Age tegen Fare prijs')
ax.legend()

# Show plot in Streamlit
st.pyplot(fig)



df3_dood = df3[df3['Survived'] == 0][['Fare']]
df3_levend = df3[df3['Survived'] == 1][['Fare']]

fig, ax = plt.subplots(ncols=2, sharey=True)

sns.boxplot(data=df3_dood, y='Fare', ax=ax[0], color='red')
ax[0].set_xlabel('Dood')

sns.boxplot(data=df3_levend, y='Fare', ax=ax[1], color='blue')
ax[1].set_xlabel('Levend')

plt.ylim(0, 200)
fig.suptitle('Spreiding Fare van Dood / Levend')
plt.tight_layout()
plt.show()

aantal_vrouwen = df3[df3['Sex'] == 'female']['Survived']
percentage_levende_vrouwen = sum(aantal_vrouwen) / len(aantal_vrouwen) * 100
print(f'Percentage levende vrouwen is {round(percentage_levende_vrouwen, 2)} %', end='\n\n')

aantal_mannen = df3[df3['Sex'] == 'male']['Survived']
percentage_levende_mannen = sum(aantal_mannen) / len(aantal_mannen) * 100
print(f'Percentage levende vrouwen is {round(percentage_levende_mannen, 2)} %')

overleven = df3[df3['Survived'] == 1]
overleven_n = overleven.groupby('Sex').size().reset_index(name='Overleven_n')
total_n = df3.groupby('Sex').size().reset_index(name='Total_n')

merged_data = pd.merge(total_n, overleven_n, on='Sex', how='left')
merged_data = pd.melt(merged_data, id_vars='Sex', value_vars=['Total_n', 'Overleven_n'], 
                      var_name='Category', value_name='Aantal')

sns.barplot(data=merged_data, x='Sex', y='Aantal', hue='Category', palette='Set2')
plt.ylabel('Aantal [-]')
plt.xlabel('Geslacht')
plt.title('Aantal totale en levende personen per geslacht na de Titanic')

plt.tight_layout()
plt.show()

aantal_1e_klas = df3[df3['Pclass'] == 1]['Survived']
aantal_1e_klas_levend = sum(aantal_1e_klas) / len(aantal_1e_klas) * 100
print(f'Percentage personen uit de 1e klasse: {round(aantal_1e_klas_levend, 2)} %', end='\n\n')

aantal_2e_klas = df3[df3['Pclass'] == 2]['Survived']
aantal_2e_klas_levend = sum(aantal_2e_klas) / len(aantal_2e_klas) * 100
print(f'Percentage personen uit de 2e klasse: {round(aantal_2e_klas_levend, 2)} %', end='\n\n')

aantal_3e_klas = df3[df3['Pclass'] == 3]['Survived']
aantal_3e_klas_levend = sum(aantal_3e_klas) / len(aantal_3e_klas) * 100
print(f'Percentage personen uit de 3e klasse: {round(aantal_3e_klas_levend, 2)} %')

overleven = df3[df3['Survived'] == 1]
overleven_n = overleven.groupby('Pclass').size().reset_index(name='Overleven_n')
total_n = df3.groupby('Pclass').size().reset_index(name='Total_n')

merged_data = pd.merge(total_n, overleven_n, on='Pclass', how='left')
merged_data = pd.melt(merged_data, id_vars='Pclass', value_vars=['Total_n', 'Overleven_n'], 
                      var_name='Category', value_name='Aantal')

sns.barplot(data=merged_data, x='Pclass', y='Aantal', hue='Category', palette='Set2')
plt.ylabel('Aantal [-]')
plt.xlabel('Pclass')
plt.title('Aantal totale en levende personen per Pclass na de Titanic')

plt.tight_layout()
plt.show()

aantal_C = df3[df3['Embarked'] == 'C']['Survived']
aantal_C_levend = sum(aantal_C) / len(aantal_C) * 100
print(f'Percentage personen uit Cherbourg: {round(aantal_C_levend, 2)} %', end='\n\n')

aantal_Q = df3[df3['Embarked'] == 'Q']['Survived']
aantal_Q_levend = sum(aantal_Q) / len(aantal_Q) * 100
print(f'Percentage personen uit Queenstown: {round(aantal_Q_levend, 2)} %', end='\n\n')

aantal_S = df3[df3['Embarked'] == 'S']['Survived']
aantal_S_levend = sum(aantal_S) / len(aantal_S) * 100
print(f'Percentage personen uit Southampton: {round(aantal_S_levend, 2)} %', end='\n\n')

overleven = df3[df3['Survived'] == 1]
overleven_n = overleven.groupby('Embarked').size().reset_index(name='Overleven_n')
total_n = df3.groupby('Embarked').size().reset_index(name='Total_n')

merged_data = pd.merge(total_n, overleven_n, on='Embarked', how='left')
merged_data = pd.melt(merged_data, id_vars='Embarked', value_vars=['Total_n', 'Overleven_n'], 
                      var_name='Category', value_name='Aantal')

sns.barplot(data=merged_data, x='Embarked', y='Aantal', hue='Category', palette='Set2')
plt.ylabel('Aantal [-]')
plt.xlabel('Embarked')
plt.title('Aantal totale en levende personen per Embarked locatie na de Titanic')

plt.tight_layout()
plt.show()