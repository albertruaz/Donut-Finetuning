import pandas as pd
import json
from pathlib import Path

# 원본 엑셀 파일 경로
XLSX = Path("../data/raw_data/exported_raw_data.xlsx")

# 사용할 시트
TARGET_SHEETS = ["작업 시트_정민", "작업 시트_동환"]

# Donut finetuning용 JSONL 생성
def main():
    xl = pd.ExcelFile(XLSX)
    merged = []

    for sheet in TARGET_SHEETS:
        df = xl.parse(sheet).dropna(how="all")

        for _, row in df.iterrows():
            # Input: 이미지 + 회전 정보
            input_data = {
                "image_url": str(row.get("image_url(required)", None)),
                "rotate_to": row.get("rotate_to(시계방향 90도 몇번 돌아갔는지)", None),
            }

            # Output: 3개 속성을 JSON으로 묶음
            output_data = {
                "brand": row.get("brand(대소문자 구분하여 텍스트 원문 그대로)", None),
                "size": row.get("alphabet_size(대소문자 구분하여 텍스트 원문 그대로)", None),
                "material": row.get("material(원어 그대로/첫번째만)", None),
            }

            record = {
                "input": input_data,
                "output": output_data,
            }
            merged.append(record)

    # JSONL로 저장
    out_path = XLSX.parent / "donut_finetune_dataset.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] Saved {len(merged)} samples -> {out_path}")

if __name__ == "__main__":
    main()
