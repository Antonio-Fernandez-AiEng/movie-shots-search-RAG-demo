import json
from pathlib import Path
from services.db_manager import db_manager
from services.video_processor import VideoProcessor

def run_evaluation():
    print("🚀 [EVAL] Start RAG evaluation...")
    db_manager.connect()
    processor = VideoProcessor()

    tests_dir = Path(__file__).resolve().parent
    dataset_path = tests_dir / "golden_dataset.json"

    if not dataset_path.exists():
        raise FileNotFoundError(f"File golden_dataset.json not found in: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    total_tests = len(golden_data)
    hits_top3 = 0
    hits_top1 = 0
    reciprocal_ranks = []

    evaluation_results = []

    print(f"🎯 [EVAL] Running {total_tests} test cases fom the Golden Dataset...\n")
    
    for i, item in enumerate(golden_data):
        query = item["query"]
        exp_movie = item["expected_movie"]
        exp_time = item["expected_timestamp"]
        tolerance = item["tolerance_range"]

        # 1. Generate query embeddings and search on ChromaDB
        query_vector = processor.encode_text_query(query)
        results = db_manager.search_similar(query_vector=query_vector, n_results=3)

        found_in_top = False
        found_position = None
        matched_movie = None
        matched_timestamp = None

        # 2. Evaluate the result position in the Top K
        for pos, r in enumerate(results, start=1):
            movie = r["metadata"].get("movie_title", "Unknown")
            timestamp = r["metadata"].get("timestamp", 0.0)

            # Test if the matching movie and timestamp are in the tolerance range
            if movie.lower() == exp_movie.lower() and abs(timestamp - exp_time) <= tolerance:
                found_in_top = True
                found_position = pos
                matched_movie = movie
                matched_timestamp = timestamp
                break

        # 3. Calculate recovery metrics (Hit Rate y MRR)
        if found_in_top:
            hits_top3 += 1
            if found_position == 1:
                hits_top1 += 1
            reciprocal_ranks.append(1.0 / found_position)
            status = f"✅ [HIT in Top {found_position}]"
        else:
            reciprocal_ranks.append(0.0)
            status = "❌ [MISS]"

        print(f"[{i+1}/{total_tests}] Query: '{query}' -> {status}")
        if found_in_top:
            print(f"   Expected: {exp_movie} ({exp_time}s) | Found: {matched_movie} ({matched_timestamp:.1f}s)\n")
        else:
            print(f"   Expected: {exp_movie} ({exp_time}s) | No valid result in the Top 3.\n")

        evaluation_results.append({
            "Query": query,
            "Expected_Movie": exp_movie,
            "Expected_Time": exp_time,
            "Status": "HIT" if found_in_top else "MISS",
            "Position": found_position if found_in_top else "N/A"
        })

    # Global Metrics
    hit_rate_top3 = (hits_top3 / total_tests) * 100
    hit_rate_top1 = (hits_top1 / total_tests) * 100
    mrr = (sum(reciprocal_ranks) / total_tests) * 100

    print("="*60)
    print("📈 PERFORMANCE REPORT FOR VIDEO RAG")
    print("="*60)
    print(f"🎯 Total tests: {total_tests}")
    print(f"🥇 Hit Rate @1 (Hit first position): {hit_rate_top1:.1f}%")
    print(f"🥉 Hit Rate @3 (Hit in top 3):       {hit_rate_top3:.1f}%")
    print(f"📊 MRR (Mean Reciprocal Rank):                {mrr:.1f}%")
    print("="*60)

    # Save report in csv
    report_path = tests_dir.parent / "video_rag_evaluation_report.csv"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Query,Expected_Movie,Expected_Time,Status,Position\n")
        for res in evaluation_results:
            f.write(f"\"{res['Query']}\",\"{res['Expected_Movie']}\",{res['Expected_Time']},{res['Status']},{res['Position']}\n")

    print(f"💾 Metrics report succesfully sabed in'{report_path}'.")
    db_manager.disconnect()

if __name__ == "__main__":
    run_evaluation()