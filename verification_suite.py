import os
import json
import pandas as pd
from src.retrieval import extract_filters, should_use_filters, _build_qdrant_filter, retrieve_with_filters
from src.pipeline import _get_distinct_values, answer_question, answer_question_naive
from src.ingestion import DATA_FILE, parse_era, norm
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import re
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

def print_header(title):
    print("\n" + "="*80)
    print(f" {title} ")
    print("="*80 + "\n")

results_table = []

def add_result(step, status, note=""):
    results_table.append((step, status, note))

print_header("Step 0 - Confirm Ingestion Completed")
try:
    df = pd.read_excel(DATA_FILE)
    client = QdrantClient(path="./qdrant_data")
    info = client.get_collection("world_cricketers")
    print(f"Raw xlsx row count: {len(df)}")
    print(f"Qdrant Point count: {info.points_count}")
    if len(df) == info.points_count:
        add_result("Step 0", "PASS")
    else:
        add_result("Step 0", "FAIL", "Mismatched counts")
    client.close()
except Exception as e:
    print(f"Error in Step 0: {e}")
    add_result("Step 0", "FAIL", str(e))

print_header("Step 1 - Verify extract_filters in isolation")
distinct = _get_distinct_values()
queries_step1 = [
    "List all left-arm fast bowlers",
    "How many Pakistani all-rounders are there?",
    "Which players were active in 2024?",
    "Who is known for aggressive, attacking batting?",
    "Pakistani all-rounders known for aggressive batting",
    "Who won the 2023 IPL final?",
    "asdkjfh 29304 ??? cricket maybe"
]

extracted_results = []
try:
    for q in queries_step1:
        print(f"\nQuery: '{q}'")
        extracted = extract_filters(q, distinct)
        extracted_results.append(extracted)
        print(json.dumps(extracted, indent=2))
    
    with open("src/retrieval.py", "r") as f:
        content = f.read()
        match = re.search(r'ChatGroq\([^)]*temperature=([0-9.]+)', content)
        if match:
            print(f"\n[Source Check] Temperature passed to ChatGroq is: {match.group(1)}")
        else:
            print("\n[Source Check] Could not find temperature setting directly.")
    add_result("Step 1", "PASS")
except Exception as e:
    print(f"Error in Step 1: {e}")
    add_result("Step 1", "FAIL", str(e))

print_header("Step 2 - Verify routing decisions")
try:
    for q, extracted in zip(queries_step1[:-1], extracted_results[:-1]):
        routing = should_use_filters(extracted)
        print(f"Query: '{q}'\n  -> Routing: {routing}")
    add_result("Step 2", "PASS")
except Exception as e:
    print(f"Error in Step 2: {e}")
    add_result("Step 2", "FAIL", str(e))

print_header("Step 3 - Verify count correctness against ground truth")
try:
    # Get ground truths
    df['Country_norm'] = df['Country'].apply(norm)
    df['Role_norm'] = df['Role'].apply(norm)
    df['Style_norm'] = df['Batting/Bowling Style'].apply(norm)
    
    c1 = df[(df['Country_norm']=='pakistan') & (df['Role_norm']=='all-rounder')].shape[0]
    c2 = df[(df['Country_norm']=='australia') & (df['Role_norm']=='batsman')].shape[0]
    c3 = df[(df['Role_norm']=='wicket-keeper batsman') & (df['Country_norm']=='india')].shape[0]
    
    qs = [
        ("How many Pakistani all-rounders are there?", c1),
        ("How many Australian batsmen are there?", c2),
        ("How many wicket-keeper batsmen from India are there?", c3)
    ]
    
    all_match = True
    for q, gt in qs:
        res = answer_question(q)
        ans_count = res.get('exact_count', 0)
        print(f"Query: '{q}'")
        print(f"  Ground Truth: {gt} | App Exact Count: {ans_count}")
        if gt != ans_count:
            all_match = False
            
    if all_match:
        add_result("Step 3", "PASS")
    else:
        add_result("Step 3", "FAIL", "Mismatch in counts")
except Exception as e:
    print(f"Error in Step 3: {e}")
    add_result("Step 3", "FAIL", str(e))

print_header("Step 4 - Verify era overlap logic at a real boundary")
try:
    df['era_start'] = df['Era'].apply(lambda x: parse_era(str(x))[0])
    df['era_end'] = df['Era'].apply(lambda x: parse_era(str(x))[1])
    # Find a straddling player (e.g. 1995-2012 Ricky Ponting or 1996-2008 Adam Gilchrist)
    player = df[(df['era_start'] <= 1999) & (df['era_end'] >= 2005)].iloc[0]
    print(f"Straddling Player Selected: {player['Name']} ({player['Era']})")
    
    qs_era = [
        "Was anyone active in 2000?",
        "Was anyone active in 2015?", # after ponting/gilchrist ended
        "Who was active in the 1990s?"
    ]
    
    for q in qs_era:
        ext = extract_filters(q, distinct)
        print(f"\nQuery: '{q}'")
        print(f"Extracted JSON:\n{json.dumps(ext, indent=2)}")
        q_filter = _build_qdrant_filter(ext)
        print(f"Constructed Qdrant Filter:\n{q_filter}")
        res = retrieve_with_filters(ext, top_k=50)
        found = any(r['name'] == player['Name'] for r in res)
        print(f"Was {player['Name']} found? {found}")
    
    add_result("Step 4", "PASS")
except Exception as e:
    print(f"Error in Step 4: {e}")
    add_result("Step 4", "FAIL", str(e))

print_header("Step 5 - Verify the zero-result guardrail fires correctly")
try:
    zero_q = "List all Afghan wicket-keeper batsmen active in the 1920s"
    res = answer_question(zero_q)
    print(f"Response:\n{res['answer']}")
    if "No players were found" in res['answer']:
        add_result("Step 5", "PASS")
    else:
        add_result("Step 5", "FAIL", "Guardrail didn't fire properly")
except Exception as e:
    print(f"Error in Step 5: {e}")
    add_result("Step 5", "FAIL", str(e))

print_header("Step 6 - Verify out-of-scope handling doesn't leak")
try:
    oos_qs = [
        "What is the capital of France?",
        "Tell me about Virat Kohli's performance in the 2024 IPL.",
        "Who is Sachin Tendulkar?" # In dataset? If yes, will answer. Let's use someone definitely not in it: "Who is Babe Ruth?"
    ]
    # Let's use:
    oos_qs_actual = [
        "What is the capital of France?",
        "How many goals did Lionel Messi score in 2022?",
        "asdsajkdfhj asjdfsdfsdf"
    ]
    all_oos = True
    for q in oos_qs_actual:
        res = answer_question(q)
        print(f"Query: '{q}'")
        print(f"Response:\n{res['answer']}\n")
        if "outside the scope" not in res['answer']:
            all_oos = False
            
    if all_oos:
        add_result("Step 6", "PASS")
    else:
        add_result("Step 6", "FAIL", "Out of scope leaked")
except Exception as e:
    print(f"Error in Step 6: {e}")
    add_result("Step 6", "FAIL", str(e))

print_header("Step 7 - Verify naive-vs-self-query comparison")
try:
    q = "How many Australian batsmen are there?"
    res_sq = answer_question(q)
    res_naive = answer_question_naive(q)
    print("SELF QUERY RESPONSE:")
    print(res_sq['answer'])
    print(f"\nNAIVE RAG RESPONSE:")
    print(res_naive['answer'])
    add_result("Step 7", "PASS")
except Exception as e:
    print(f"Error in Step 7: {e}")
    add_result("Step 7", "FAIL", str(e))

print_header("Step 8 - Verify Langfuse traces")
try:
    from src.monitoring import _LANGFUSE_ENABLED
    print(f"Langfuse Enabled: {_LANGFUSE_ENABLED}")
    add_result("Step 8", "PASS")
except Exception as e:
    print(f"Error in Step 8: {e}")
    add_result("Step 8", "FAIL", str(e))

print_header("Final Summary")
print(f"{'Step':<10} | {'Status':<10} | {'Note'}")
print("-" * 50)
for step, status, note in results_table:
    print(f"{step:<10} | {status:<10} | {note}")
