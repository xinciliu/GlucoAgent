import sys
import requests
import json
import pandas as pd
import argparse
import os

# Fixed API Configuration
API_URL = "https://api.chatanywhere.tech/v1/chat/completions"
API_KEY = "sk-6k6d7PIe27uMdZpbKVGWKJBFtRU1jPrBhAhOtWT51TmRlepn"
MODEL_NAME = "gpt-4o"
MAX_TOKENS = 2000
TEMPERATURE = 0.7

# 确保输出目录存在
os.makedirs("outputs", exist_ok=True)
SAVE_REC_PATH = "outputs/recommendation_result.json"

def run_gpt_api(prompt: str) -> str | None:
    """
    Send pure text prompt to LLM API, no image input
    Args:
        prompt: Full prompt containing patient info, glucose data and references
    Returns:
        Raw LLM response string, None if request fails
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    request_body = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "n": 1
    }
    try:
        response = requests.post(API_URL, headers=headers, json=request_body, timeout=30)
    except Exception as e:
        print(f"API connection error: {str(e)}")
        return None

    if response.status_code == 200:
        res_json = response.json()
        output_text = res_json["choices"][0]["message"]["content"]
        return output_text
    else:
        print(f"API request failed, status code: {response.status_code}")
        print(f"Response detail: {response.text}")
        return None

def extract_json_from_raw_text(raw_text: str) -> dict:
    """
    Extract valid JSON block from messy LLM output
    """
    start_idx = raw_text.find("{")
    end_idx = raw_text.rfind("}")
    if start_idx == -1 or end_idx == -1:
        print("No valid JSON found in LLM output")
        return {}
    json_str = raw_text[start_idx: end_idx + 1]
    try:
        parsed_dict = json.loads(json_str)
        return parsed_dict
    except json.JSONDecodeError as err:
        print(f"JSON parsing error: {str(err)}, raw json string: {json_str}")
        return {}

def load_patient_glucose(input_csv_path: str, output_csv_path: str):
    """
    Load last 96 historical OT values from input csv, predicted 24h glucose from output csv
    Auto drop Date column if exists
    """
    df_input = pd.read_csv(input_csv_path)
    if "Date" in df_input.columns:
        df_input = df_input.drop(columns=["Date"])
    history_glucose = df_input["OT"].tail(96).tolist()

    df_output = pd.read_csv(output_csv_path)
    pred_glucose = df_output["Predicted_Glucose"].tolist()
    return history_glucose, pred_glucose

def load_txt_file(file_path: str) -> str:
    """Read plain text file for patient info"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Failed to read {file_path}: {str(e)}")
        return "No patient information available"

def load_reference_json(json_path: str) -> list | str:
    """Load reference literature json file"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load {json_path}: {str(e)}")
        return "No reference papers available"

def build_full_prompt(patient_summary, history_glucose, pred_glucose, references):
    """Combine all input data into standard prompt template"""
    prompt_template = '''
### Task
You are a specialist in diabetes care. Based on the patient’s basic information, historical 96-hour CGM glucose data, predicted 24-hour future glucose values, and relevant diabetes research references, generate personalized, actionable daily lifestyle recommendations for glucose management.

### Strict Requirements
1. Each independent recommendation must start with the sentence "Since you + patient's personal situation".
2. Every suggestion must be logically linked to the patient’s actual conditions (diabetes type, age, complications, eating habits, historical blood glucose fluctuations, predicted high/low glucose risk).
3. All advice must only be supported by the provided reference papers; do not introduce external medical knowledge or arbitrary assumptions.
4. Only provide lifestyle guidance covering diet, exercise frequency/intensity and daily living routines. Do not mention insulin, hypoglycemic drugs or medical monitoring equipment.
5. Fully personalize advice according to demographics, diabetes classification, historical glycemic performance and predicted risk.
6. If historical and predicted glucose values stay within normal range, recognize the patient’s good habits and suggest maintaining current routines without major changes. If frequent hyper/hypoglycemia appears, clearly explain potential risks and give targeted improvement plans.
7. Use simple, friendly plain language, avoid complex medical jargon.
8. Do not add professional medical diagnosis conclusions to any output.
9. The historical glucose sequence includes 96 real measurement records; the prediction sequence contains 24 future glucose values to reflect upcoming glycemic trends.
10. All lifestyle advice must be clinically safe and will not bring physical or psychological harm to patients.
11. Output at most 6 independent recommendations in total.

### Input Data
1. Patient Profile Summary: {patient_text}
2. Historical 96-hour CGM Glucose Values: {history_data}
3. Predicted Next 24h Glucose Sequence: {predict_data}
4. Reference Research Literature: {reference_text}

### Mandatory Output Format
Return pure JSON object only, no extra text, markdown or explanation.
Keys are sequential numbers (1,2,3...), values are each complete recommendation sentence.
Example format:
{{
    "1": "Since you have type 1 diabetes..., your daily advice...",
    "2": "Since your predicted glucose will rise after meals, you can..."
}}
'''
    return prompt_template.format(
        patient_text=patient_summary,
        history_data=str(history_glucose),
        predict_data=str(pred_glucose),
        reference_text=str(references)
    )

def generate_recommendation(input_csv, output_csv, patient_txt, ref_json):
    """Main pipeline: load data → build prompt → call LLM → parse JSON result"""
    # Load all input data
    hist_glu, pred_glu = load_patient_glucose(input_csv, output_csv)
    patient_text = load_txt_file(patient_txt)
    ref_data = load_reference_json(ref_json)

    # Construct full prompt
    full_prompt = build_full_prompt(patient_text, hist_glu, pred_glu, ref_data)

    # Call LLM api
    raw_result = run_gpt_api(full_prompt)
    if raw_result is None:
        return {"error": "Failed to get response from LLM API"}

    # Extract json from response
    rec_dict = extract_json_from_raw_text(raw_result)
    return rec_dict

if __name__ == "__main__":
    """
    Command line usage unified with cgm forecasting script:
    python generate_llm.py --hist_cgm input.csv --pred_cgm outputs/prediction_result.csv --user_info patient_info.txt --reference reference.json
    --hist_cgm    raw CGM time series csv with OT column
    --pred_cgm    model predicted glucose csv file
    --user_info   patient personal info text file
    --reference   reference literature json file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--hist_cgm", type=str, required=True, help="Original CGM csv with OT column")
    parser.add_argument("--pred_cgm", type=str, required=True, help="Predicted glucose csv output from forecasting model")
    parser.add_argument("--user_info", type=str, required=True, help="Patient info text file path")
    parser.add_argument("--reference", type=str, required=True, help="Reference paper json file path")
    args = parser.parse_args()

    # Run recommendation generation
    final_rec = generate_recommendation(
        input_csv=args.hist_cgm,
        output_csv=args.pred_cgm,
        patient_txt=args.user_info,
        ref_json=args.reference
    )

    # Print final json result
    print(json.dumps(final_rec, indent=2, ensure_ascii=False))

    # 新增：保存结果到 outputs 文件夹
    with open(SAVE_REC_PATH, "w", encoding="utf-8") as f:
        json.dump(final_rec, f, indent=2, ensure_ascii=False)
    print(f"\nRecommendations saved to {SAVE_REC_PATH}")
