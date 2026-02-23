
"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""
from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """
    Creates a relational time-series plot of accidents over time.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    df['YearMonth'] = df['Accident Date'].dt.to_period('M')
    df.groupby('YearMonth').size().plot(ax=ax)

    ax.set_title('Accidents Over Time (Year-Month)')
    ax.set_xlabel('Year-Month')
    ax.set_ylabel('Number of Accidents')

    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.close(fig)

    return


def plot_categorical_plot(df):
    """
    Creates a categorical distribution plot for light conditions.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.countplot(
        data=df,
        x='Light_Conditions',
        ax=ax
    )

    ax.set_title('Distribution of Accidents by Light Condition')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    plt.close(fig)

    return


def plot_statistical_plot(df):
    """
    Creates a statistical correlation heatmap.
    """

    fig, ax = plt.subplots(figsize=(6, 5))

    corr_matrix = df[['Number_of_Casualties',
                      'Number_of_Vehicles',
                      'Severity_Num']].corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        ax=ax
    )

    ax.set_title('Correlation Heatmap')

    plt.tight_layout()
    plt.savefig('statistical_plot.png')
    plt.close(fig)

    return


def statistical_analysis(df, col: str):
    """
    Calculates statistical moments for a given column.
    """

    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])

    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Clean and prepare the dataset for analysis.
    """

    # Convert dates
    df['Accident Date'] = pd.to_datetime(
        df['Accident Date'],
        format='%d-%m-%Y',
        errors='coerce'
    )

    # Convert numeric columns
    df['Number_of_Casualties'] = pd.to_numeric(
        df['Number_of_Casualties'],
        errors='coerce'
    )

    df['Number_of_Vehicles'] = pd.to_numeric(
        df['Number_of_Vehicles'],
        errors='coerce'
    )

    # Drop rows with missing critical values
    df = df.dropna(subset=[
        'Number_of_Casualties',
        'Number_of_Vehicles',
        'Accident Date'
    ])

    # Encode severity
    df['Severity_Num'] = df['Accident_Severity'].map({
        'Slight': 1,
        'Serious': 2,
        'Fatal': 3
    })

    # Quick statistical overview
    print(df.describe())
    print(df.corr(numeric_only=True))

    return df


def writing(moments, col):
    """
    Prints statistical interpretation of calculated moments.
    """

    mean, stddev, skewness, excess_kurtosis = moments

    print(f'For the attribute {col}:')
    print(f'Mean = {mean:.2f}, '
          f'Standard Deviation = {stddev:.2f}, '
          f'Skewness = {skewness:.2f}, and '
          f'Excess Kurtosis = {excess_kurtosis:.2f}.')

    # Determine skew direction
    if skewness > 0.5:
        skew_type = 'right skewed'
    elif skewness < -0.5:
        skew_type = 'left skewed'
    else:
        skew_type = 'approximately symmetric'

    # Determine kurtosis type
    if excess_kurtosis > 0:
        kurtosis_type = 'leptokurtic'
    elif excess_kurtosis < 0:
        kurtosis_type = 'platykurtic'
    else:
        kurtosis_type = 'mesokurtic'

    print(f'The distribution is {skew_type} and {kurtosis_type}.')
    return

def main():
    """
    Main execution pipeline.
    """

    df = pd.read_csv('data.csv')

    df = preprocessing(df)

    col = 'Number_of_Casualties'

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)

    writing(moments, col)

    return


if __name__ == '__main__':
    main()
