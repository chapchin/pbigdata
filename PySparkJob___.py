import io
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff
from pyspark.sql import functions as F
from pathlib import Path


def process(spark, input_file, target_path):
    """Group input_file data, split the result to train/val/test and save them to folders inside target_path folder"""
    
    source = spark.read.parquet(input_file)
    
    grouped = source.groupby('ad_id').agg(
        F.first('target_audience_count').alias('target_audience_count'),
        F.first('has_video').alias('has_video'),
        (F.first('ad_cost_type') == 'CPM').astype('int').alias('is_cpm'),
        (F.first('ad_cost_type') == 'CPC').astype('int').alias('is_cpc'),
        F.first('ad_cost').alias('ad_cost'),
        F.countDistinct('date').alias('day_count'),
        (
            F.sum(F.when(F.col('event') == 'click', 1).otherwise(0)) / 
            F.sum(F.when(F.col('event') == 'view', 1).otherwise(0))
        ).alias('ctr')
    ).filter('ctr is not null')
    
    train, test, validate = grouped.randomSplit([0.5, 0.25, 0.25], seed=42)
    
    target_folder = Path(target_path)
    train.write.mode('overwrite').parquet(str(target_folder/'train'))
    test.write.mode('overwrite').parquet(str(target_folder/'test'))
    validate.write.mode('overwrite').parquet(str(target_folder/'validate'))


def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    target_path = argv[1]
    print("Target path: " + target_path)
    spark = _spark_session()
    process(spark, input_path, target_path)


def _spark_session():
    return SparkSession.builder.appName('PySparkJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are required.")
    else:
        main(arg)
