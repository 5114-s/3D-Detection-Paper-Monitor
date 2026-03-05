基于Python与arXiv官方API构建的3D目标检测（3D Object Detection）领域学术情报自动化监测、清洗与多维可视化分析系统。

## 🚀 快速启动 (Quick Start)

### 1. 环境依赖
推荐使用 **Python 3.9 及以上版本**，执行以下命令安装依赖：
```bash
pip install arxiv pymysql requests pandas matplotlib seaborn wordcloud sqlalchemy
```

### 2. 数据库初始化
在本地 MySQL 8.0+ 中执行以下 SQL 语句，创建数据库并设置字符集：
```bash
CREATE DATABASE my_spider_db
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;
```
### 3. 配置与运行
1.修改脚本中的数据库连接凭证（DB_USER, DB_PASSWORD），确保与你的本地数据库配置一致。

2.运行爬虫监控引擎，抓取最新数据：
```bash
python monitor.py
```
3.运行数据分析与可视化引擎，生成图表：
```bash
python real_data_charts.py
```
