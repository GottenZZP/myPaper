### 关于日本数据集
- 数据集目录在 D:\python_code\paper\data
- train3和val2是最原始的文件
- train4和val3是去除停用词后的文件
- train5和val4是去除停用词且数据增强的文件
- train6和val5是train3和val3进行数据清理后的文件

### 关于中国数据集
- 数据目录在 D:\python_code\paper\corpus\chinese
- train2和val2是最原始文件

### 效果
- 参数配置：textcnn过滤器大小(2, 3, 4, 5, 6, 7)，核大小768