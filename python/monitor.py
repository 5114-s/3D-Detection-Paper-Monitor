import sys
import io
import arxiv
import pymysql
import logging
import time
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# 解决Windows终端乱码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(threadName)s: %(message)s'
)

class AcademicDBManager:
    """高级数据库管理模块：支持事务、去重及标签分类"""
    def __init__(self, host, user, pw, db):
        self.config = {'host': host, 'user': user, 'password': pw, 'database': db, 'charset': 'utf8mb4'}
        self.init_db()

    def init_db(self):
        conn = pymysql.connect(**self.config)
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_papers_v2 (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    paper_id VARCHAR(50) UNIQUE,
                    title VARCHAR(500) NOT NULL,
                    authors TEXT,
                    published_date DATE,
                    pdf_link VARCHAR(255),
                    local_path VARCHAR(255),
                    summary TEXT,
                    tags VARCHAR(100),
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)
            conn.commit()
            logging.info("Advanced Database Schema initialized.")
        finally:
            conn.close()

    def safe_insert(self, data):
        conn = pymysql.connect(**self.config)
        try:
            with conn.cursor() as cursor:
                sql = """
                INSERT IGNORE INTO detection_papers_v2 
                (paper_id, title, authors, published_date, pdf_link, summary, tags)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, data)
            conn.commit()
            return cursor.rowcount
        except Exception as e:
            logging.error(f"DB Insert Error: {e}")
            return 0
        finally:
            conn.close()

class IntelligentScraper:
    """智能爬虫：支持线性安全抓取、语义标签识别"""
    def __init__(self, db_manager):
        self.db = db_manager
        self.pdf_dir = "3d_detection_library"
        if not os.path.exists(self.pdf_dir):
            os.makedirs(self.pdf_dir)

    def classify_tags(self, text):
        """语义标签识别逻辑"""
        tags = []
        text = text.lower()
        if "lidar" in text or "point cloud" in text: tags.append("LiDAR")
        if "monocular" in text or "single image" in text: tags.append("Monocular")
        if "voxel" in text: tags.append("Voxel-based")
        if "transformer" in text or "attention" in text: tags.append("Transformer")
        return "|".join(tags) if tags else "General-3D"

    def process_single_paper(self, result):
        """处理单篇论文的入库逻辑"""
        try:
            p_id = result.entry_id.split('/')[-1]
            title = result.title
            authors = ", ".join([a.name for a in result.authors])
            pub_date = result.published.strftime('%Y-%m-%d')
            tags = self.classify_tags(title + " " + result.summary)
            
            data = (p_id, title, authors, pub_date, result.pdf_url, result.summary[:800], tags)
            
            is_new = self.db.safe_insert(data)
            if is_new:
                logging.info(f"New SOTA saved: {title[:60]}... [Tags: {tags}]")
            return is_new
        except Exception as e:
            logging.error(f"Processing error: {e}")
            return 0

    def run_concurrent_task(self, keywords, max_per_query=250):
        """
        优化方案：引入限速器与自适应休眠
        将并发降低为线性安全执行 (max_workers=1)，防止触发 HTTP 429
        """
        all_new_count = 0
        # 将线程池限制为单工位执行，确保请求序列化
        with ThreadPoolExecutor(max_workers=1) as executor:
            for kw in keywords:
                logging.info(f"Monitoring query: {kw}")
                try:
                    search = arxiv.Search(query=kw, max_results=max_per_query)
                    
                    # 获取结果迭代器
                    results = list(search.results())
                    logging.info(f"Query '{kw}' returned {len(results)} results.")
                    
                    futures = [executor.submit(self.process_single_paper, res) for res in results]
                    for f in futures:
                        all_new_count += f.result()
                    
                    # 自适应休眠策略：请求大块数据后强制休息 5-8 秒
                    wait_time = random.uniform(5, 8)
                    logging.info(f"Applying Rate Limiter: Cooling down for {wait_time:.2f}s...")
                    time.sleep(wait_time)
                    
                except arxiv.HTTPError as e:
                    logging.error(f"API Rate Limit Hit (429). Stopping for this session: {e}")
                    break
                except Exception as e:
                    logging.error(f"Unexpected error: {e}")
                    continue
        
        logging.info(f"--- Scan Finished. {all_new_count} papers integrated ---")

if __name__ == "__main__":
    # 数据库配置
    DB_PARAMS = ('localhost', 'root', 'your_password_here', 'my_spider_db')
    
    # 3D检测相关精准词
    TARGET_QUERIES = [
        'ti:"3D object detection" AND cat:cs.CV',
        'abs:"nuScenes" AND abs:"detection"',
        'ti:"point cloud" AND abs:"3D"',
        'all:"BEV detection"',
        'all:"autonomous driving" AND abs:"3D detection"', 
        'all:"LiDAR" AND abs:"transformer"',             
        'all:"Voxel" AND cat:cs.CV'
    ]

    manager = AcademicDBManager(*DB_PARAMS)
    scraper = IntelligentScraper(manager)
    
    # 启动任务
    scraper.run_concurrent_task(TARGET_QUERIES, max_per_query=250)