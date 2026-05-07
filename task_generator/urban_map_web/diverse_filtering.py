import json
import argparse
import os
from collections import defaultdict

def get_action_signature(sample):
    """Extract consecutive action_names as a behavioral signature."""
    actions = sample.get("evaluation_criteria", {}).get("actions", [])
    return " -> ".join([a.get("name", "unknown") for a in actions])

def get_semantic_text(sample):
    """Concatenate Persona and Instructions to get a text string for semantic analysis."""
    persona = sample.get("user_scenario", {}).get("persona", "")
    reason = sample.get("user_scenario", {}).get("instructions", {}).get("reason_for_call", "")
    return f"Persona: {persona}\nFlow: {reason}"

def greedy_filter(samples, model, threshold):
    """Remove samples with Cosine similarity >= threshold compared to selected samples."""
    import torch
    from sentence_transformers import util
    
    if not samples: return []
    if len(samples) == 1: return samples
    
    texts = [get_semantic_text(s) for s in samples]
    embeddings = model.encode(texts, convert_to_tensor=True)
    
    kept_samples = [samples[0]]
    kept_embeddings = [embeddings[0]]
    
    for i in range(1, len(samples)):
        curr_emb = embeddings[i]
        similarities = util.cos_sim(curr_emb, torch.stack(kept_embeddings))[0]
        max_sim = torch.max(similarities).item()
        
        # If sample is different (similarities < threshold), keep it as a representative sample
        if max_sim < threshold:
            kept_samples.append(samples[i])
            kept_embeddings.append(curr_emb)
            
    return kept_samples

def filter_samples(input_file, output_file, use_action_seq=True, use_embedding=True, threshold=0.85, model_name="BAAI/bge-large-en-v1.5"):
    print(f"Reading data from: {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Total starting samples: {len(data)}")
    
    model = None
    if use_embedding:
        print(f"\nLoading embedding model '{model_name}'...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
        except ImportError:
            print("[!] ERROR: Missing library. Please run: pip install sentence-transformers torch")
            return
            
    # Group by sub_tasks
    grouped_by_subtask = defaultdict(list)
    for s in data:
        grouped_by_subtask[s.get("sub_tasks", "Unknown")].append(s)
        
    final_samples = []
    
    for sub_task, st_samples in grouped_by_subtask.items():
        print(f"\n> Processing Sub-task: '{sub_task}' ({len(st_samples)} samples)")
        filtered_st_samples = []
        
        if use_action_seq:
            # STEP 1: Group based on Action Sequence
            action_groups = defaultdict(list)
            for s in st_samples:
                action_groups[get_action_signature(s)].append(s)
                
            # STEP 2: Semantic filtering within each Action cluster
            for sig, group_samples in action_groups.items():
                if use_embedding:
                    filtered_st_samples.extend(greedy_filter(group_samples, model, threshold))
                else:
                    filtered_st_samples.append(group_samples[0])
        else:
            # If Step 1 is disabled, filter directly on the sub_task set using Step 2
            if use_embedding:
                filtered_st_samples = greedy_filter(st_samples, model, threshold)
            else:
                filtered_st_samples = st_samples
                
        print(f"  + Kept: {len(filtered_st_samples)}/{len(st_samples)} most diverse representative samples.")
        final_samples.extend(filtered_st_samples)
        
    # Write output file
    out_dir = os.path.dirname(output_file)
    if out_dir: 
        os.makedirs(out_dir, exist_ok=True)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_samples, f, indent=2, ensure_ascii=False)
        
    print(f"\n[COMPLETE] Successfully filtered out {len(data) - len(final_samples)} duplicate samples.")
    print(f"Saved {len(final_samples)} samples to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script chống trùng lặp mẫu Dataset (Diverse Filtering) phiên bản 2 lớp.")
    parser.add_argument("--input", default="generated/generated_all_static_validator.json", help="File đầu vào")
    parser.add_argument("--output", default="generated/generated_all_static_validator_diversity_filtered.json", help="File lưu trữ xuất ra")
    
    # Defaults are all True
    parser.add_argument("--disable_action_seq", action="store_false", dest="use_action_seq", help="Disable matching of API Call Sequences")
    parser.add_argument("--disable_embedding", action="store_false", dest="use_embedding", help="Disable semantic matching using Embeddings")
    
    parser.add_argument("--threshold", type=float, default=0.85, help="Deduplication threshold (values > threshold are considered duplicates). Default is 0.85")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5", help="Sentence-Transformers model name")
    
    args = parser.parse_args()
    
    filter_samples(
        input_file=args.input,
        output_file=args.output,
        use_action_seq=args.use_action_seq,
        use_embedding=args.use_embedding,
        threshold=args.threshold,
        model_name=args.model
    )
