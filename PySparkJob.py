import io
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff
from pyspark.sql import functions as F

def somefuncm(value):
  if   value == 'CPM': 
      return 1
  else:
      return 0
def somefuncc(value):
  if   value == 'CPC': 
      return 1
  else:
      return 0

def process(spark, input_file, target_path):
    # TODO Ваш код
    df=spark.read.parquet('clickstream.parquet')
    udfsomefuncm = F.udf(somefuncm, 'int')
    udfsomefuncc = F.udf(somefuncc, 'int')
    # добавляю колонку is_cpm
    ndf = df.withColumn('is_cpm',udfsomefuncm(col('ad_cost_type')))
    # добавляю колонку is_cpc
    ndf = ndf.withColumn('is_cpc',udfsomefuncc(col('ad_cost_type')))
    # подсчитываю число click с группировкой
    df2=ndf.where(ndf.event=="click").groupBy(['ad_id']).count().withColumnRenamed('count', 'countc')
    # подсчитываю число view с группировкой
    df3=ndf.where(ndf.event=="view").groupBy(['ad_id']).count().withColumnRenamed('count', 'countv')
    #число дней показа обьявлений
    df4=ndf.select('ad_id', 'date').distinct().groupBy(['ad_id']).count().withColumnRenamed('count', 'day_count')
    #собираю колонки из df2  df3
    df5=df2.join(df3, 'ad_id', 'outer').select('ad_id', 'countc','countv')
    #собираю колонки из df4 и df5
    df6=df5.join(df4, 'ad_id','outer').select('ad_id', 'countc','countv','day_count')
    #считаю CTR и добавляю к df6
    df7=df6.withColumn('CTR', col('countc') / col('countv'))
    # собираю в итоговый датафрейм
    nndf=ndf.join(df7, 'ad_id', 'outer').select('ad_id','target_audience_count','has_video','is_cpm','is_cpc','ad_cost','day_count','CTR')
    #К итоговому датафрейму применяю сплит.
    splits=nndf.randomSplit([0.5,0.25,0.25])
    #Пишу полученные выборки в соответствующие папки.
    splits[0].coalesce(1).write.parquet(target_path+'/train/persons_parquet')
    splits[1].coalesce(1).write.parquet(target_path+'/test/persons_parquet')
    splits[2].coalesce(1).write.parquet(target_path+'/validate/persons_parquet')
    

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
    #arg=['clickstream.parquet','result']    
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)
