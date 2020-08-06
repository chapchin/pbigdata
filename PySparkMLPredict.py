import io
import sys

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession


GBT_MODEL = 'gbt_model'


def process(spark, input_file, output_file):
    #запуск python PySparkMLPredict.py test.parquet result
    #input_file - путь к файлу с данными для которых нужно предсказать ctr
    #output_file - путь по которому нужно сохранить файл с результатами [ads_id, prediction]
    #загружаю файл для предикта
    test_data = spark.read.parquet(input_file)
    #загружаю модель с диска, она должна находиться в директории с исполняемым айлом
    #model - модель на диске
    model = PipelineModel.load('model')
    #делаю предикт
    prediction = model.transform(test_data)
    #выовожу данные запроса в файл cvs
    prediction.select("ad_id", "prediction").write.csv(output_file)    
    return 1

def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    output_file = argv[1]
    print("Output path to file: " + output_file)
    spark = _spark_session()
    process(spark, input_path, output_file)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLPredict').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)
