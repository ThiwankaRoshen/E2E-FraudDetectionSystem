from pyspark.sql import functions as F

import mlflow

from pyspark.sql import functions as F

def build_frequency_encoding(train_df, col):
    """
    Returns frequency map as Spark DataFrame
    """

    freq_df = (
        train_df.groupBy(col)
        .count()
        .withColumnRenamed("count", f"{col}_freq")
    )

    return freq_df

def apply_frequency_encoding(df, freq_df, col):
    """
    Join frequency encoding back to dataset
    """

    return df.join(freq_df, on=col, how="left")

def log_frequency_mappings(source_freq, browser_freq):

    # convert to pandas for logging
    source_pdf = source_freq.toPandas()
    browser_pdf = browser_freq.toPandas()

    mlflow.log_dict(source_pdf.to_dict(), "source_frequency_map.json")
    mlflow.log_dict(browser_pdf.to_dict(), "browser_frequency_map.json")

def add_sex_ohe(df):
    """
    Adds one-hot encoding for sex column (F/M)
    """

    df = df.withColumn(
        "sex_F",
        F.when(F.col("sex") == "F", 1).otherwise(0)
    )

    df = df.withColumn(
        "sex_M",
        F.when(F.col("sex") == "M", 1).otherwise(0)
    )

    return df

def data_encoder(train_df, test_df):

    # --- frequency encoding ---
    source_freq = build_frequency_encoding(train_df, "source")
    browser_freq = build_frequency_encoding(train_df, "browser")
    log_frequency_mappings(source_freq, browser_freq)
    
    train_df = apply_frequency_encoding(train_df, source_freq, "source")
    test_df = apply_frequency_encoding(test_df, source_freq, "source")

    train_df = train_df.withColumnRenamed("source_freq", "source_fr_enc")
    test_df = test_df.withColumnRenamed("source_freq", "source_fr_enc")

    train_df = apply_frequency_encoding(train_df, browser_freq, "browser")
    test_df = apply_frequency_encoding(test_df, browser_freq, "browser")

    train_df = train_df.withColumnRenamed("browser_freq", "browser_fr_enc")
    test_df = test_df.withColumnRenamed("browser_freq", "browser_fr_enc")

    # ---  OHE  ---
    train_df = add_sex_ohe(train_df)
    test_df = add_sex_ohe(test_df)

    return train_df, test_df