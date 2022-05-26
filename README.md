### 关于日本数据集
- 数据集目录在 D:\python_code\paper\data
- train2和val1是最原始的文件
- train3和val2是去除停用词后的文件
- train4和val3是在train3和val2基础上去除多余空格的文件
- train5和val4是去除停用词且数据增强的文件
- train6和val5是train3和val3进行数据清理后的文件
- test数据集是没有进行格式化的测试集
- test2数据集是进行了去除空格等格式化后的测试集
- total数据集是train5和val4组合起来的数据集
- total2数据集是train4和val3组合起来的数据集
- total3数据集是train2_del_split和val2_del_split组合起来的数据集

### 关于中国数据集
- 数据目录在 D:\python_code\paper\corpus\chinese
- train2和val2是最原始文件

### 效果
- 参数配置：textcnn过滤器大小(2, 3, 4, 5, 6, 7)，核大小768