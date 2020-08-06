import io
import sys

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


GBT_MODEL = 'gbt_model'


def process(spark, train_data, test_data):
    #train_data - путь к файлу с данными для обучения модели
    #test_data - путь к файлу с данными для оценки качества модели
    #сейчас использую только train_data
    #запуск python PySparkMLFit.py train.parquet validate.parquet

    #загружаю train_data
    train_data = spark.read.parquet(train_data)

    #обучаю модель
    #add feature
    feature = VectorAssembler(inputCols = train_data.columns[:7],outputCol="features")

    # Train a GBT model.
    gbt = GBTRegressor(labelCol="ctr", featuresCol="features", maxIter=10)

#pipeline
    pipeline = Pipeline(stages=[feature, gbt])

    paramGrid = ParamGridBuilder().addGrid(gbt.maxDepth, [2, 3, 4, 5,6,7,8,9]).addGrid(gbt.maxBins, [10, 16, 20, 24, 32, 36]).build()
    
# A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=RegressionEvaluator(labelCol="ctr", predictionCol="prediction", metricName="rmse"),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)

# Run TrainValidationSplit, and choose the best set of parameters.
    model = tvs.fit(train_data)

    #делаю выборку для тестирования
    (training_data1, test_data) = train_data.randomSplit([0.8, 0.2],seed = 42)

    #по тестовой выборке выделенной из train_data считаю rmse и вывожу его
    prediction = model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="ctr", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(prediction)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    #сохраняю модель
    model.bestModel.write().overwrite().save("model")
    


def main(argv):
    train_data = argv[0]
    print("Input path to train data: " + train_data)
    test_data = argv[1]
    print("Input path to test data: " + test_data)
    spark = _spark_session()
    process(spark, train_data, test_data)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Train and test data are require.")
    else:
        main(arg)
