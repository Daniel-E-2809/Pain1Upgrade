import pandas as pd
import plotly.express as px
import streamlit as st

df1 = pd.read_csv('gender_submission.csv')
df2 = pd.read_csv('test.csv')
df2_2 = df2.merge(df1)
df3 = pd.read_csv('train.csv')

df2['Age'] = df2['Age'].fillna(df2['Age'].mean())
df2['Fare'] = df2['Fare'].fillna(df2['Fare'].mean())

df3['Age'] = df3['Age'].fillna(df3['Age'].mean())
df3['Embarked'] = df3['Embarked'].fillna('S')


# Streamlit app
st.title("Analyse van de Titanic Dataset")
st.write("Dit dashboard is een uitbreiding op Casus 1 van het begin van de minor. Deze leek de meest logische keuzen gezien toen nog niet actief gewerkt werd met streamlit. Hieronder volgen enkele plotten waarmee een beeld kan worden verkregen van de gegeven data over de titanic. Ten eerste de verhouding van totale aantal passagiers, gesplitst op een gekozen kolom, in verhouding tot het aantal passagiers die overleefde.")

# Dropdown menu voor de keuze van de plot
plot_type = st.selectbox("Selecteer het type plot", ["Geslacht", "Pclass", "Embarked"])

# Voorbereiding van de gegevens voor de gekozen plot
if plot_type == "Geslacht":
    overleven = df3[df3['Survived'] == 1]
    overleven_n = overleven.groupby('Sex').size().reset_index(name='Overleven_n')
    total_n = df3.groupby('Sex').size().reset_index(name='Total_n')

    merged_data = pd.merge(total_n, overleven_n, on='Sex', how='left')
    merged_data = pd.melt(merged_data, id_vars='Sex', value_vars=['Total_n', 'Overleven_n'], 
                          var_name='Category', value_name='Aantal')

    fig = px.bar(merged_data, x='Sex', y='Aantal', color='Category',
                 labels={'Aantal': 'Aantal [-]', 'Sex': 'Geslacht'},
                 title='Aantal totale en levende personen per geslacht na de Titanic')

elif plot_type == "Pclass":
    overleven_n = df3[df3['Survived'] == 1].groupby('Pclass').size().reset_index(name='Overleven_n')
    total_n = df3.groupby('Pclass').size().reset_index(name='Total_n')

    merged_data = pd.merge(total_n, overleven_n, on='Pclass', how='left')
    merged_data = pd.melt(merged_data, id_vars='Pclass', value_vars=['Total_n', 'Overleven_n'], 
                          var_name='Category', value_name='Aantal')

    fig = px.bar(merged_data, x='Pclass', y='Aantal', color='Category',
                 labels={'Aantal': 'Aantal [-]', 'Pclass': 'Pclass'},
                 title='Aantal totale en levende personen per Pclass na de Titanic')

else:  # "Embarked"
    overleven_n = df3[df3['Survived'] == 1].groupby('Embarked').size().reset_index(name='Overleven_n')
    total_n = df3.groupby('Embarked').size().reset_index(name='Total_n')

    merged_data = pd.merge(total_n, overleven_n, on='Embarked', how='left')
    merged_data = pd.melt(merged_data, id_vars='Embarked', value_vars=['Total_n', 'Overleven_n'], 
                          var_name='Category', value_name='Aantal')

    fig = px.bar(merged_data, x='Embarked', y='Aantal', color='Category',
                 labels={'Aantal': 'Aantal [-]', 'Embarked': 'Embarked'},
                 title='Aantal totale en levende personen per Embarked locatie na de Titanic')

# Toon de plot in Streamlit
st.plotly_chart(fig)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# Voorbeeldgegevens - vervang dit met je werkelijke DataFrame 'df3'
# df3 = pd.read_csv('path_to_your_titanic_data.csv')

# Selecteer de 'Fare' data voor overleden en overlevende passagiers
df3_dood = df3[df3['Survived'] == 0][['Fare']]
df3_levend = df3[df3['Survived'] == 1][['Fare']]

# Maak de figuur aan
fig = go.Figure()

# Voeg de boxplots toe voor overleden en overlevende passagiers
fig.add_trace(go.Box(y=df3_dood['Fare'], name='Dood', marker_color='red'))
fig.add_trace(go.Box(y=df3_levend['Fare'], name='Levend', marker_color='blue'))

# Layout aanpassen
fig.update_layout(
    title='Spreiding Fare van Dood / Levend',
    yaxis_title='Fare prijs [ £ ]',
    xaxis_title='Status',
    yaxis=dict(range=[0, 200]),
    showlegend=False,
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=[
                dict(
                    args=[{"visible": [True, False]}],
                    label="Dood",
                    method="update"
                ),
                dict(
                    args=[{"visible": [False, True]}],
                    label="Levend",
                    method="update"
                ),
                dict(
                    args=[{"visible": [True, True]}],
                    label="Beide",
                    method="update"
                )
            ],
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.17,
            xanchor="left",
            y=1.1,
            yanchor="top"
        ),
    ]
)

# Streamlit app
st.title("Titanic Fare Analyse")
st.write('Hier volgt een boxplot dat de spreading van fares toont onder de overleden en overlevende passagiers, wat verder inzicht geeft in hoe de Fare waarde invloed had op overlevings kansen. Te zien is hoe voor de bovenste 50% van overlevende passagiers de Fare merkbaar hoger lag, met de bovenste 25% van deze group ruim hoger lag dan de limiet van de gestorven boxplot.')

# Toon de figuur
st.plotly_chart(fig)






import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Voorbeeldgegevens - vervang dit met je werkelijke datasetpaden
# df1 = pd.read_csv('path_to_gender_submission.csv')
# df2 = pd.read_csv('path_to_test.csv')


# Selecteer de numerieke kolommen
numeric_data = df2_2[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]

# Verwijder eventuele missende waarden
numeric_data = numeric_data.dropna()

# Bereken de correlatiematrix
correlation_matrix = numeric_data.corr()

# Streamlit app
st.title("Correlatiematrix van de Titanic Dataset")

# Plot de heatmap van de correlatiematrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='magma', fmt=".2f")
plt.title("Correlation Matrix of Titanic Dataset")

# Toon de heatmap in Streamlit
st.pyplot(plt)



import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import streamlit as st



# Definieer de doelvariabele en kenmerken
y = df3['Survived']
features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
X = pd.get_dummies(df3[features])
X_test = pd.get_dummies(df2[features])

# Train het model
model = RandomForestClassifier(n_estimators=95, max_depth=6, random_state=1)
model.fit(X, y)

# Maak voorspellingen
predictions = model.predict(X_test)

# Voeg de voorspellingen toe aan de testset voor analyse
df2['Survived'] = predictions

# Streamlit setup
st.title('Titanic Survival Prediction Analysis')

# Dropdown voor het selecteren van een feature
selected_feature = st.selectbox("Selecteer een kenmerk:", features)

# Maak een samenvatting van de overlevingspercentages op basis van het geselecteerde kenmerk
if selected_feature == "Sex":
    survival_summary = df2.groupby('Sex')['Survived'].mean().reset_index()
elif selected_feature == "Pclass":
    survival_summary = df2.groupby('Pclass')['Survived'].mean().reset_index()
elif selected_feature == "Embarked":
    survival_summary = df2.groupby('Embarked')['Survived'].mean().reset_index()
elif selected_feature == "Age":
    # Voor leeftijd moeten we mogelijk bins maken
    df2['Age_bins'] = pd.cut(df2['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    survival_summary = df2.groupby('Age_bins')['Survived'].mean().reset_index()
elif selected_feature == "Fare":
    # Voor fare moeten we mogelijk bins maken
    df2['Fare_bins'] = pd.cut(df2['Fare'], bins=[0, 10, 30, 50, 100, 200])
    survival_summary = df2.groupby('Fare_bins')['Survived'].mean().reset_index()

# Plot de overlevingspercentages op basis van de geselecteerde feature
fig = px.bar(survival_summary, x=survival_summary.columns[0], y='Survived',
             labels={survival_summary.columns[0]: selected_feature, 'Survived': 'Overlevingspercentage'},
             title=f'Overlevingspercentage op basis van {selected_feature}')

# Toon de plot in Streamlit
st.plotly_chart(fig)



import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import streamlit as st


df2 = pd.read_csv('test.csv')
df3 = pd.read_csv('train.csv')

df2['Age'] = df2['Age'].fillna(df2['Age'].mean())
df2['Fare'] = df2['Fare'].fillna(df2['Fare'].mean())
df3['Age'] = df3['Age'].fillna(df3['Age'].mean())
df3['Embarked'] = df3['Embarked'].fillna('S')

# Definieer de doelvariabele en kenmerken
y = df3['Survived']
features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
X = pd.get_dummies(df3[features])
X_test = pd.get_dummies(df2[features])

# Train het model
model = RandomForestClassifier(n_estimators=95, max_depth=6, random_state=1)
model.fit(X, y)

# Maak voorspellingen
predictions = model.predict(X_test)

# Voeg de voorspellingen toe aan de testset
df2['Predicted_Survived'] = predictions

# Streamlit setup
st.title('Vergelijking van Voorspellingen met Werkelijke Waarden')

# Dropdown voor het selecteren van een feature met unieke key
selected_feature = st.selectbox("Selecteer een kenmerk:", ['Sex', 'PClass', 'Embarked', 'Age_bins', 'Fare_bins'], key="unique_selectbox")

# Maak een samenvatting van de overlevingspercentages op basis van het geselecteerde kenmerk
if selected_feature == "Sex":
    survival_summary = df3.groupby('Sex').agg({'Survived': 'mean'}).reset_index()
    predicted_summary = df2.groupby('Sex').agg({'Predicted_Survived': 'mean'}).reset_index()

elif selected_feature == "Pclass":
    survival_summary = df3.groupby('Pclass').agg({'Survived': 'mean'}).reset_index()
    predicted_summary = df2.groupby('Pclass').agg({'Predicted_Survived': 'mean'}).reset_index()

elif selected_feature == "Embarked":
    survival_summary = df3.groupby('Embarked').agg({'Survived': 'mean'}).reset_index()
    predicted_summary = df2.groupby('Embarked').agg({'Predicted_Survived': 'mean'}).reset_index()

elif selected_feature == "Age_bins":
    # Ensure bins are consistent for both training and test datasets
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    df2['Age_bins'] = pd.cut(df2['Age'], bins=bins)
    df3['Age_bins'] = pd.cut(df3['Age'], bins=bins)
    
    # Convert Age_bins to strings to avoid serialization issues with Plotly
    df2['Age_bins'] = df2['Age_bins'].astype(str)
    df3['Age_bins'] = df3['Age_bins'].astype(str)
    
    # Group by Age_bins and reset the index for proper merging
    survival_summary = df3.groupby('Age_bins').agg({'Survived': 'mean'}).reset_index()
    predicted_summary = df2.groupby('Age_bins').agg({'Predicted_Survived': 'mean'}).reset_index()

elif selected_feature == "Fare_bins":
    #Ensure bins are consistent for both training and test datasets
    bins_fare = [0, 10, 30, 50, 100, 200]
    df2['Fare_bins'] = pd.cut(df2['Fare'], bins=bins_fare)
    df3['Fare_bins'] = pd.cut(df3['Fare'], bins=bins_fare)
    
    # Convert Fare_bins to strings to avoid serialization issues with Plotly
    df2['Fare_bins'] = df2['Fare_bins'].astype(str)
    df3['Fare_bins'] = df3['Fare_bins'].astype(str)
    
    # Group by Fare_bins and reset the index for proper merging
    survival_summary = df3.groupby('Fare_bins').agg({'Survived': 'mean'}).reset_index()
    predicted_summary = df2.groupby('Fare_bins').agg({'Predicted_Survived': 'mean'}).reset_index()

    


# Mergeer de samenvattingen
merged_summary = pd.merge(survival_summary, predicted_summary, how='outer', on=selected_feature)
merged_summary.columns = [selected_feature, 'Actual_Survival', 'Predicted_Survival']

# Plot de overlevingspercentages van de werkelijke en voorspelde waarden op basis van de geselecteerde feature
fig = px.bar(merged_summary, 
             x=selected_feature, 
             y=['Actual_Survival', 'Predicted_Survival'],
             labels={selected_feature: selected_feature, 
                     'value': 'Overlevingspercentage'},
             title=f'Overlevingspercentage op basis van {selected_feature}',
             barmode='group')

# Toon de plot in Streamlit
st.plotly_chart(fig)




import pandas as pd
import numpy as np
import  plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

plt.style.use('seaborn-v0_8')
# Streamlit app layout
st.title("Titanic: Age vs Fare with Trendlines")

# Checkbox for showing Dood (not survived) scatterplot and trendline
show_dood = st.checkbox('Show Dood (Not Survived) Data')
show_levend = st.checkbox('Show Levend (Survived) Data')

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
