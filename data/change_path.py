import pandas as pd

# CSV 불러오기
csv_path = './all_real_with_split.csv'
df = pd.read_csv(csv_path)

# 바꿀 경로 prefix
old_prefix = '/mnt/aix23102/deepfake'
new_prefix = '/path/to'

# 경로 컬럼 수정
df['original_path'] = df['original_path'].str.replace(old_prefix, new_prefix, regex=False)
df['new_path'] = df['new_path'].str.replace(old_prefix, new_prefix, regex=False)

# 저장
df.to_csv('./all_real_with_split.csv', index=False)

print("CSV anonymized successfully.")
