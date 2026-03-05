import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sqlalchemy import create_engine
import warnings
import itertools
import sys
import io

# 解决 Windows 终端可能出现的打印乱码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# ==========================================
# 1. 设置绘图风格与中文字体支持
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] # 支持中文
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 连接真实 MySQL 数据库并获取全量数据
# ==========================================
print("[INFO] 正在连接 MySQL 数据库...")
DB_USER = 'root'
DB_PASSWORD = '20031010' 
DB_HOST = 'localhost'
DB_NAME = 'my_spider_db'

# 创建 SQLAlchemy 引擎
engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}?charset=utf8mb4')

# 执行一次 SQL 查询，将画图所需的所有字段全部提取出来
query = "SELECT paper_id, title, published_date, summary, tags FROM detection_papers_v2"
df = pd.read_sql(query, engine)

print(f"[SUCCESS] 成功读取数据库！共获取 {len(df)} 条真实学术文献记录。")

# 数据初步清洗
df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')

# ==========================================
# 3. 图表一：核心技术栈交叉关联热力图
# ==========================================
print("[INFO] 正在分析技术栈关联度并生成热力图 (1/4)...")
# 获取所有非空的标签，并按 '|' 拆分
valid_tags = df['tags'].dropna().str.split('|')
unique_tags = sorted(list(set(valid_tags.explode())))

# 初始化一个全为 0 的共现矩阵
matrix = pd.DataFrame(0, index=unique_tags, columns=unique_tags)

for tags in valid_tags:
    for tag in tags:
        matrix.at[tag, tag] += 1
    for t1, t2 in itertools.combinations(sorted(tags), 2):
        matrix.at[t1, t2] += 1
        matrix.at[t2, t1] += 1

plt.figure(figsize=(10, 8))
sns.heatmap(matrix, annot=True, fmt="d", cmap="YlOrRd", linewidths=.5, cbar_kws={'label': 'Co-occurrence Count'})
plt.title('Technology Co-occurrence Heatmap in 3D Detection', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig('real_tech_heatmap.png', dpi=300)
print("  -> [SUCCESS] 已生成技术关联热力图: real_tech_heatmap.png")

# ==========================================
# 4. 图表二：真实技术路径标签分布柱状图
# ==========================================
print("[INFO] 正在统计技术标签分布 (2/4)...")
plt.figure(figsize=(10, 6))
all_tags = valid_tags.explode()
tag_counts = all_tags.value_counts().head(8) # 取 Top 8

sns.barplot(x=tag_counts.values, y=tag_counts.index, palette="viridis")
plt.title('Top Technology Tags in Your Database', fontsize=16, fontweight='bold')
plt.xlabel('Number of Papers', fontsize=12)
plt.ylabel('Technology Path', fontsize=12)
plt.tight_layout()
plt.savefig('real_tags_distribution.png', dpi=300)
print("  -> [SUCCESS] 已生成真实技术标签分布图: real_tags_distribution.png")

# ==========================================
# 5. 图表三：真实论文发表时间趋势折线图
# ==========================================
print("[INFO] 正在分析文献发表时间序列 (3/4)...")
plt.figure(figsize=(10, 6))
# 过滤掉日期为空的数据
df_time = df.dropna(subset=['published_date']).copy()
df_time['year_month'] = df_time['published_date'].dt.to_period('M')
monthly_counts = df_time.groupby('year_month').size().reset_index(name='counts')
monthly_counts['year_month'] = monthly_counts['year_month'].dt.to_timestamp()

sns.lineplot(data=monthly_counts, x='year_month', y='counts', marker="o", linewidth=2.5, color="#2c7fb8")
plt.fill_between(monthly_counts['year_month'], monthly_counts['counts'], alpha=0.2, color="#2c7fb8")
plt.title('Actual Publication Trend of Scraped Papers', fontsize=16, fontweight='bold')
plt.xlabel('Published Date', fontsize=12)
plt.ylabel('Number of Papers', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('real_publication_trend.png', dpi=300)
print("  -> [SUCCESS] 已生成真实时间趋势图: real_publication_trend.png")

# ==========================================
# 6. 图表四：学术摘要语义高频词云图
# ==========================================
print("[INFO] 正在进行文本清洗与生成词云 (4/4)...")
# 清洗摘要为空的数据
df_summary = df.dropna(subset=['summary'])
all_abstracts_text = " ".join(df_summary['summary'].tolist())

wordcloud = WordCloud(
    width=1000, 
    height=500, 
    background_color='white',
    colormap='Dark2',    
    max_words=150,       
    collocations=False
).generate(all_abstracts_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  
plt.title('Semantic Word Cloud of 3D Detection Abstracts', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('real_semantic_wordcloud.png', dpi=300)
print("  -> [SUCCESS] 已生成语义词云图: real_semantic_wordcloud.png")

print("\n[ALL DONE] 所有 4 张数据分析图表已成功生成，请在当前文件夹中查看！")