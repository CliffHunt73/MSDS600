import pandas as pd
from pycaret.classification import ClassificationExperiment


def load_data(filepath):
    """
    Loads Churn data into a DataFrame from a string filepath.
    """
    #df = pd.read_csv(filepath)
    df = pd.read_csv(filepath, index_col='customerID')

    # Add a column named 'Churn' to the DataFrame
    if 'Churn' not in df.columns:
        df['Churn'] = 0

    return df
    


def make_predictions(df):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    """
    classifier = ClassificationExperiment()
    model = classifier.load_model('Churn_data_pycaret_model')
    predictions = classifier.predict_model(model, data=df)
    predictions.rename({'Label': 'Churn'}, axis=1, inplace=True)
    predictions['Churn'].replace({1: 'Churn', 0: 'No Churn'}, inplace=True)
    return predictions['Churn']


if __name__ == "__main__":
    df = load_data('C://Users//cliff//Downloads//prepped_new_churn_data_unmodified.csv')

    
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)
