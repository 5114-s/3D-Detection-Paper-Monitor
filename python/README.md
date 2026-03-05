基于Python与arXiv官方API构建的3D目标检测（3D Object Detection）领域学术情报自动化监测、清洗与多维可视化分析系统。

## 🚀 快速启动 (Quick Start)

### 1. 环境依赖
```bash
# 推荐使用 Python 3.9 及以上版本
pip install arxiv pymysql requests pandas matplotlib seaborn wordcloud sqlalchemy
```
2. 数据库初始化
```bash
在本地 MySQL (8.0+) 中执行以下 SQL 初始化数据库与字符集：
CREATE DATABASE my_spider_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
3. 配置与运行
```bash
修改脚本中的数据库连接凭证 (DB_USER, DB_PASSWORD)。

运行爬虫监控引擎，抓取最新数据：
python monitor.py
运行数据分析与可视化引擎，生成图表：
python real_data_charts.py
