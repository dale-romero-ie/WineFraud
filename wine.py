# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

# Load the saved models and encoders
rf_classifier = joblib.load('rf_wine_classifier.pkl')
le_type = joblib.load('le_type.pkl')
le_quality = joblib.load('le_quality.pkl')

# Load the dataset
wine_data = pd.read_csv('wine_fraud.csv')

# Split the dataset into two subsets: red wine and white wine
red_wine_data = wine_data[wine_data['type'] == 'red']
white_wine_data = wine_data[wine_data['type'] == 'white']

# Prediction function for Wine Fraud Detector
def predict_wine_fraud(values):
    """Predicts wine fraud using the trained model."""
    prediction = rf_classifier.predict([values])
    return le_quality.inverse_transform(prediction)[0]

# Distribution Plot
def distribution_plot_for_wine(wine_data_subset):
    """Creates and returns a distribution plot for wine quality."""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=wine_data_subset, x='quality')
    plt.title(f'Distribution of Wine Quality')
    return plt

# Correlation Plot
def correlation_plot_for_wine(wine_data_subset):
    """Creates and returns a correlation plot for features with wine quality."""
    wine_data_subset['quality_temp'] = wine_data_subset['quality'].map({'Legit': 0, 'Fraud': 1})
    correlations = wine_data_subset.corr(numeric_only=True)['quality_temp'].sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    correlations.drop('quality_temp').plot(kind='bar')
    plt.title(f'Feature Correlation with Quality')
    return plt

def plotly_sploom(wine_data_subset):
    """Creates and returns a scatter plot matrix for selected wine features."""
    features = ['alcohol', 'volatile acidity', 'quality']

    # Define custom color map for 'quality'
    color_map = {'Fraud': 'red', 'Legit': 'green'}

    subset = wine_data_subset[features]
    fig = px.scatter_matrix(subset, dimensions=features[:-1], color='quality',
                            labels={col: col.replace('_', ' ').capitalize() for col in features},
                            opacity=0.6, width=1200, height=1200, color_discrete_map=color_map)
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="black", plot_bgcolor='white', font=dict(color='white'))
    fig.update_traces(marker=dict(size=3, opacity=0.6))
    fig.update_traces(diagonal_visible=False)
    return fig




# Feature Importance Plot
def feature_importance_plot(wine_data_subset):
    """Creates and returns a plot showing the importance of each feature in predicting wine quality."""
    X = wine_data_subset.drop(columns=['type', 'quality'])
    y_encoded = wine_data_subset['quality'].map({'Legit': 0, 'Fraud': 1})
    rf_classifier_local = RandomForestClassifier(
        n_estimators=50,
        max_depth=None,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=42
    )
    rf_classifier_local.fit(X, y_encoded)
    feature_importances = rf_classifier_local.feature_importances_
    features_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=features_df, palette='viridis')
    plt.title('Feature Importance from Random Forest')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    return plt
    image_path = "winefraud.jpg"
    st.image(Image.open(image_path), caption="Wine Fraud", use_column_width=True)

def display_insights():
    st.title("Insights")
    st.write("Insights for both Red and White Wine")
  
    # Distribution
    st.markdown("## Distribution")
    st.write("- From the data (for both red and white wine), we can see there are many more legit wines than fraudulent wines.")
                
    # Correlation
    st.markdown("## Correlation")
    st.write("""
            **Red Wine**:
            - Alcohol has the most negative correlation with fraudulent wine, implying that as the alcohol content increases, the likelihood of the wine being legitimate also increases.
            - Features like volatile acidity, chlorides, and density have a positive correlation with fraudulent wines. This means that higher values of these features might indicate a higher chance of the wine being fraudulent.
                    
            **White Wine**:
            - Again, alcohol has the most negative correlation, suggesting a similar trend as with red wines.
            - Chlorides and density showcase a positive correlation, indicating that higher values for these features could suggest a higher likelihood of fraudulence in white wines.
            """)


    # Scatter Plots
    st.markdown("## Scatter Plots")
    st.write("""
            **Red Wine**:  
            - Most legitimate wines tend to have higher alcohol content and lower volatile acidity.
            - Fraudulent wines are somewhat scattered, but they predominantly occupy regions with lower alcohol content and higher volatile acidity.
                    
            **White Wine**:  
            - The legitimate wines are concentrated more towards higher alcohol content and lower volatile acidity.
            - Fraudulent wines, again, tend to have lower alcohol content, but their volatile acidity varies more than in red wines.

            The scatter plots reinforce our earlier observations from the correlation plots. It's evident that alcohol and volatile acidity are crucial features that distinguish between legitimate and fraudulent wines.
            """)

    # Feature Importance
    st.markdown("## Feature Importance")
    st.write("""
            **Red Wine**:  
            - Alcohol stands out as the most important feature in predicting wine fraud for red wines. This aligns with our earlier observations.
            - Other significant features include volatile acidity, sulphates, and density.
                    
            **White Wine**:  
            - Similar to red wines, alcohol is the most crucial feature in predicting wine fraud.
            - Density, chlorides, and volatile acidity also play vital roles in the prediction for white wines.
            """)

    # Summary
    st.markdown("## Summary")
    st.write("""
            - The dataset primarily contains legitimate wines, with a smaller portion of fraudulent wines.
            - Alcohol content plays a pivotal role in distinguishing between legitimate and fraudulent wines, with higher alcohol content generally indicating legitimate wines.
            - Features like volatile acidity, density, and chlorides also have notable importance in the prediction of wine fraud.
            - The Random Forest classifier gives importance to these features while predicting the legitimacy of a wine.
            """)

def display_notes():
    st.markdown("# Notes")
    
    st.markdown("## Objective")
    st.write("The application is designed to analyze a dataset containing information about wines, specifically focusing on determining wine fraud.")
    
    st.markdown("## Key Components")
    
    st.markdown("### Data Loading & Preparation")
    st.write("- The application loads pre-trained Random Forest models and label encoders from .pkl files (Using Pickle).")
    st.write("- The dataset 'wine_fraud.csv' is loaded and divided into two subsets: red wine and white wine.")
    
    st.markdown("### Functions")
    st.write("- `predict_wine_fraud`: Predicts if a given wine is fraudulent using the trained Random Forest model.")
    st.write("- `distribution_plot_for_wine`: Visualizes the distribution of wine quality for a given subset.")
    st.write("- `correlation_plot_for_wine`: Displays a bar chart of feature correlation with wine quality.")
    st.write("- `plotly_sploom`: Generates a scatter plot matrix for selected wine features.")
    st.write("- `feature_importance_plot`: Visualizes the importance of each feature in predicting wine quality using a local Random Forest model.")
    
    st.markdown("### Streamlit Application Interface")
    st.write("- The main interface includes a sidebar where users can select from five options: 'Red Wine', 'White Wine', 'Wine Fraud Detector', 'Insights', and 'Notes'.")
    st.write("- For both 'Red Wine' and 'White Wine' options, users can further choose to view the raw data, distribution, correlation, scatter plots, or feature importance.")
    st.write("- 'Wine Fraud Detector' allows users to input wine features through sliders, and upon submission, the app predicts if the wine is legitimate or fraudulent.")
    
    st.markdown("### Dataset Overview")
    st.write("- The dataset, 'wine_fraud.csv', contains columns for wine features such as acidity, sugar content, density, etc. It also includes a 'type' column indicating whether the wine is red or white and a 'quality' column that specifies if the wine is 'Legit' or a 'Fraud'.")
    
    st.markdown("### Visualization")
    st.write("- The application makes use of libraries like matplotlib, seaborn, and plotly for data visualization.")


# Main Streamlit application
def main():
    """Main function to run the Streamlit application."""
    image_path = "winefraud.jpg"
    #st.image(Image.open(image_path), caption="Wine Fraud", use_column_width=True)
    st.sidebar.image(Image.open(image_path), caption="Wine Fraud", use_column_width=True)


    st.sidebar.title("Wine Analysis Selection")
    wine_selection = st.sidebar.radio(
        "Choose Option:", 
        ["Red Wine", "White Wine", "Wine Fraud Detector", "Insights", "Notes"]
    )

    # Handling the different selections in the Streamlit app
    if wine_selection == "Red Wine":
        st.title("Analysis for Red Wine")
        analysis_option = st.radio("Choose an Analysis:", ["Data", "Distribution", "Correlation", "Scatter Plots", "Feature Importance"])
        
        if analysis_option == "Data":
            st.dataframe(red_wine_data.head(100))
        elif analysis_option == "Distribution":
            st.pyplot(distribution_plot_for_wine(red_wine_data))
        elif analysis_option == "Correlation":
            st.pyplot(correlation_plot_for_wine(red_wine_data))
        elif analysis_option == "Scatter Plots":
            st.plotly_chart(plotly_sploom(red_wine_data))
        elif analysis_option == "Feature Importance":
            st.pyplot(feature_importance_plot(red_wine_data))

    elif wine_selection == "White Wine":
        st.title("Analysis for White Wine")
        analysis_option = st.radio("Choose an Analysis:", ["Data", "Distribution", "Correlation", "Scatter Plots", "Feature Importance"])
        
        if analysis_option == "Data":
            st.dataframe(white_wine_data.head(100))
        elif analysis_option == "Distribution":
            st.pyplot(distribution_plot_for_wine(white_wine_data))
        elif analysis_option == "Correlation":
            st.pyplot(correlation_plot_for_wine(white_wine_data))
        elif analysis_option == "Scatter Plots":
            st.plotly_chart(plotly_sploom(white_wine_data))
        elif analysis_option == "Feature Importance":
            st.pyplot(feature_importance_plot(white_wine_data))

    elif wine_selection == "Wine Fraud Detector":
        st.title("Wine Fraud Detector")
        #st.markdown("<span style='color:red; font-size:20px;'>Select ALL metrics and then SUBMIT</span>", unsafe_allow_html=True)
        st.markdown("<span style='color:red; font-size:20px;'><strong>Select ALL metrics and then SUBMIT</strong></span>", unsafe_allow_html=True)

        wine_type = st.selectbox("Wine Type", ["red", "white"])
        
        # Slider definitions for wine features
        fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.0)
        volatile_acidity = st.slider("Volatile Acidity", 0.1, 2.0, 0.5)
        citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
        residual_sugar = st.slider("Residual Sugar", 0.0, 16.0, 2.0)
        chlorides = st.slider("Chlorides", 0.01, 0.6, 0.05)
        free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 72, 15)
        total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 289, 40)
        density = st.slider("Density", 0.99, 1.01, 1.0)
        pH = st.slider("pH", 2.7, 4.0, 3.2)
        sulphates = st.slider("Sulphates", 0.3, 2.0, 0.5)
        alcohol = st.slider("Alcohol", 8.0, 15.0, 10.0)

        if st.button("Submit"):
            values = [
                le_type.transform([wine_type])[0], fixed_acidity, volatile_acidity, 
                citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, 
                total_sulfur_dioxide, density, pH, sulphates, alcohol
            ]
            prediction = predict_wine_fraud(values)
            st.write(f"The wine is predicted to be: **{prediction}**")

    elif wine_selection == "Insights":
        display_insights()
        # st.title("Insights")
        # st.write("Insights for both Red and White Wine")
  
        # paragraph_2 = """
        #         Distribution:

        #             From the data (for both red and white wine, we can see there are many more legit wines, than fradulent wines)
                
        #         Correlation:

        #             Red Wine: Alcohol has the most negative correlation with fraudulent wine, implying that as the alcohol content increases, the likelihood of the wine being legitimate also increases.
        #             Features like volatile acidity, chlorides, and density have a positive correlation with fraudulent wines. This means that higher values of these features might indicate a higher chance of the wine being fraudulent.
                    
        #             White Wine: Again, alcohol has the most negative correlation, suggesting a similar trend as with red wines.
        #             Chlorides and density showcase a positive correlation, indicating that higher values for these features could suggest a higher likelihood of fraudulence in white wines.

        #         Scatter Plots:

        #             Red Wine:  Most legitimate wines tend to have higher alcohol content and lower volatile acidity.
        #             Fraudulent wines are somewhat scattered, but they predominantly occupy regions with lower alcohol content and higher volatile acidity.
                    
        #             White Wine:  The legitimate wines are concentrated more towards higher alcohol content and lower volatile acidity.
        #             Fraudulent wines, again, tend to have lower alcohol content, but their volatile acidity varies more than in red wines.

        #             The scatter plots reinforce our earlier observations from the correlation plots. It's evident that alcohol and volatile acidity are crucial features that distinguish between legitimate and fraudulent wines.

        #         Feature Importance:

        #             Red Wine:  Alcohol stands out as the most important feature in predicting wine fraud for red wines. This aligns with our earlier observations.
        #             Other significant features include volatile acidity, sulphates, and density.
                    
        #             White Wine:  Similar to red wines, alcohol is the most crucial feature in predicting wine fraud.
        #             Density, chlorides, and volatile acidity also play vital roles in the prediction for white wines.

        #         Summary:

        #             The dataset primarily contains legitimate wines, with a smaller portion of fraudulent wines.
        #             Alcohol content plays a pivotal role in distinguishing between legitimate and fraudulent wines, with higher alcohol content generally indicating legitimate wines.
        #             Features like volatile acidity, density, and chlorides also have notable importance in the prediction of wine fraud.
        #             The Random Forest classifier gives importance to these features while predicting the legitimacy of a wine.

        #             """
    
        # st.write(paragraph_2)


    elif wine_selection == "Notes":
        display_notes()
        # st.title("Notes")
        # paragraph_1 = """
        #         Objective: The application is designed to analyze a dataset containing information about wines, specifically focusing on determining wine fraud.

        #         Key Components:
        #         * Data Loading & Preparation:
        #             * The application loads pre-trained Random Forest models and label encoders from .pkl files. (Using Pickle)
        #             * The dataset: 'wine_fraud.csv' is loaded and divided into two subsets: red wine and white wine.
        #         * Functions:
        #             * predict_wine_fraud: Predicts if a given wine is fraudulent using the trained Random Forest model.
        #             * distribution_plot_for_wine: Visualizes the distribution of wine quality for a given subset.
        #             * correlation_plot_for_wine: Displays a bar chart of feature correlation with wine quality.
        #             * plotly_sploom: Generates a scatter plot matrix for selected wine features.
        #             * feature_importance_plot: Visualizes the importance of each feature in predicting wine quality using a local Random Forest model.
        #         * Streamlit Application Interface:
        #             * The main interface includes a sidebar where users can select from five options: "Red Wine", "White Wine", "Wine Fraud Detector", "Insights", and "Notes".
        #             * For both "Red Wine" and "White Wine" options, users can further choose to view the raw data, distribution, correlation, scatter plots, or feature importance.
        #         * "Wine Fraud Detector" allows users to input wine features through sliders, and upon submission, the app predicts if the wine is legitimate or fraudulent.

        #         Dataset Overview:
        #         - The dataset, 'wine_fraud.csv', contains columns for wine features such as acidity, sugar content, density, etc. It also includes a 'type' column indicating whether the wine is red or white and a 'quality' column that specifies if the wine is "Legit" or a "Fraud".
                
        #         Visualization:
        #         - The application makes use of libraries like matplotlib, seaborn, and plotly for data visualization.
        #             """
    
        # st.write(paragraph_1)


# Entry point of the program
if __name__ == "__main__":
    main()
