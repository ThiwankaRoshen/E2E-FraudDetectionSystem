from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
import ipaddress


def transform_raw_data(df: DataFrame) -> DataFrame:
    """
    Transform raw fraud dataset.

    - signup_time: string -> timestamp
    - purchase_time: string -> timestamp
    - ip_address: numeric -> IPv4 string
    """

    # Convert timestamps
    df = (
        df.withColumn(
            "signup_time",
            F.to_timestamp("signup_time", "yyyy-MM-dd HH:mm:ss")
        )
        .withColumn(
            "purchase_time",
            F.to_timestamp("purchase_time", "yyyy-MM-dd HH:mm:ss")
        )
    )

    # Convert numeric IP to IPv4 string
    @F.udf(StringType())
    def int_to_ip(ip):
        if ip is None:
            return None
        return str(ipaddress.IPv4Address(int(ip)))

    df = df.withColumn(
        "ip_address",
        int_to_ip(F.col("ip_address"))
    )

    return df