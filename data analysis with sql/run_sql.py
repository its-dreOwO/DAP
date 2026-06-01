import pandas as pd
import sqlite3

csv_path = "D:\DAp\DAP\Data\data\clean_data.csv"       
db_path = "analysis_database.db"
sql_path = "analysis.sql"

# 1. Đọc CSV và nạp vào SQLite
df = pd.read_csv(csv_path)
conn = sqlite3.connect(db_path)
df.to_sql("orders", conn, if_exists="replace", index=False)

print("Done: data imported to analysis_v2.db, table = orders")

# 2. Đọc file SQL
with open(sql_path, "r", encoding="utf-8") as f:
    sql_text = f.read()

# 3. Bỏ comment và dòng trống
clean_lines = []
for line in sql_text.splitlines():
    stripped = line.strip()
    if stripped.startswith("--") or stripped == "":
        continue
    clean_lines.append(line)

clean_sql = "\n".join(clean_lines)

# 4. Tách từng query theo dấu ;
queries = [q.strip() for q in clean_sql.split(";") if q.strip()]

# 5. Chạy từng query và in kết quả
for i, query in enumerate(queries, 1):
    print("\n" + "=" * 80)
    print(f"QUERY {i}")
    print("=" * 80)
    try:
        result = pd.read_sql_query(query, conn)
        print(result.head(20))   # in 20 dòng đầu
    except Exception as e:
        print(f"ERROR in query {i}: {e}")
        print("SQL:")
        print(query)

conn.close()
print("\nAll queries finished.")
