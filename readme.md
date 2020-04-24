## 运行流程

1、运行环境 win10、python3.6.4

2、运行之前需安装有如下python包

```python
re, pandas, numpy, sklearn, lightgbm, xgboost, catboost 
```

3、将复赛数据放入shandong_data文件中

4、运行run.py文件，提交submit.csv文件



##建模思路

目前大多企业得风险评估模型是基于KMV模型和Logit模型，但是这两种模型获得公司得标准化数据，对数据得质量要求较高。其难以快速得捕捉到内部得变化信息，市场变化等信息，且具有一定得滞后性。随着大数据技术得发展，如何利用大规模稀疏数据对企业建立信用机制，是一个值得思考得问题。

本次建模，主要从企业得基本信息、运营情况、区域/业务竞争力分析、信用历史5个维度对企业进行刻画。通过对数据进行清洗，值变换，删除无效变量等操作对原始数据进行清洗。最终通过KS值、单变量分析等方法选择了相对重要的 51个特征进行建模。

最终取得 复赛 A榜 第一，B 榜第五得成绩。



比赛链接 ： [http://sdac.qingdao.gov.cn/common/cmpt/%E8%AF%86%E5%88%AB%E5%A4%B1%E4%BF%A1%E4%BC%81%E4%B8%9A%E5%A4%A7%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html](http://sdac.qingdao.gov.cn/common/cmpt/识别失信企业大赛_竞赛信息.html)

