import fire

from metagpt.roles.di.data_interpreter import DataInterpreter

WINE_REQ = "Run data analysis on sklearn Wine recognition dataset, include a plot, and train a model to predict wine class (20% as validation), and show validation accuracy."

DATA_DIR = "path/to/your/data"
# sales_forecast data from https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast/data
SALES_FORECAST_REQ = f"""Train a model to predict sales for each department in every store (split the last 40 weeks records as validation dataset, the others is train dataset), include plot total sales trends, print metric and plot scatter plots of
groud truth and predictions on validation data. Dataset is {DATA_DIR}/train.csv, the metric is weighted mean absolute error (WMAE) for test data. Notice: *print* key variables to get more information for next task step.
"""
# TTA数据分析
TTA_DATA_DIR = '/home/gee/repos/MetaGPT/Commonality.csv'
# TA_DATA_REQ = f"""根据给出的csv文件数据，分析出NG产品的关联Process的原因,每个步骤的图表都要显示出来且不能覆盖，最终结论用文字表达出来，NG关联原因的结论要包含数据。这是csv数据文件路径 {TTA_DATA_DIR}"""
TA_DATA_REQ = f"""根据给出的csv文件数据，分析出NG产品的关联Process的原因,每个步骤的图表都要显示出来，最终结论NG关联原因的结论要包含数据。这是csv数据文件路径 {TTA_DATA_DIR}"""

REQUIREMENTS = {"wine": WINE_REQ, "sales_forecast": SALES_FORECAST_REQ, "tta": TA_DATA_REQ}


async def main(use_case: str = "tta"):
    mi = DataInterpreter()
    requirement = REQUIREMENTS[use_case]
    await mi.run(requirement)


if __name__ == "__main__":
    fire.Fire(main)
