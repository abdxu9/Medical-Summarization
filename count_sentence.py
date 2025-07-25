import json
import nltk
import pandas as pd
import os
import sys

# --- Installation Check for 'transformers' and 'sentencepiece' ---
# SentencePiece is required by modern tokenizers like Gemma's.
try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: The 'transformers' library is not installed.")
    print("Please install it to enable token counting by running: pip install transformers")
    sys.exit(1)

try:
    import sentencepiece
except ImportError:
    print("Error: The 'sentencepiece' library is not installed.")
    print("Modern tokenizers like Gemma require it. Please run: pip install sentencepiece")
    sys.exit(1)


def analyze_summary_metrics(directory_path, tokenizer_name):
    """
    Loads all model summary JSON files from a directory, calculates the average
    number of sentences and tokens per summary using a specified modern tokenizer.

    Args:
        directory_path (str): The path to the directory containing the JSON files.
        tokenizer_name (str): The name of the Hugging Face tokenizer to use.

    Returns:
        pd.DataFrame: A DataFrame with 'Model', 'Avg Sentences', and 'Avg Tokens' columns.
    """
    # --- 1. SETUP: Download NLTK data and load the tokenizer ---
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("First-time setup: Downloading NLTK's 'punkt' sentence tokenizer...")
        nltk.download('punkt', quiet=True)
        print("Download complete.")

    print(f"\nLoading modern tokenizer: '{tokenizer_name}'...")
    try:
        # Using a modern tokenizer from a state-of-the-art model family
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print("Tokenizer loaded successfully. ✅")
    except Exception as e:
        print(f"Error: Could not load the tokenizer '{tokenizer_name}'.")
        print(f"Please check the model name on Hugging Face Hub. Reason: {e}")
        return pd.DataFrame()

    if not os.path.isdir(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        return pd.DataFrame()

    # --- 2. ANALYSIS: Loop through files and calculate metrics ---
    analysis_results = []
    print(f"\nSearching for summary files in: '{directory_path}'...")
    
    files_found = 0
    for filename in os.listdir(directory_path):
        if filename.endswith('_summaries.json'):
            files_found += 1
            model_name = filename.replace('_summaries.json', '')
            file_path = os.path.join(directory_path, filename)
            
            print(f"  - Processing file: {filename}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                summaries = data.get("summaries", [])
                
                if not summaries:
                    print(f"    - Warning: '{filename}' is empty. Skipping.")
                    continue
                
                # Tokenize all summaries at once for efficiency
                tokens = tokenizer(summaries)
                total_tokens = sum(len(ids) for ids in tokens['input_ids'])
                
                # Calculate total sentences
                total_sentences = sum(len(nltk.sent_tokenize(summary_text)) for summary_text in summaries)
                
                # Calculate the averages
                num_summaries = len(summaries)
                avg_sentences = total_sentences / num_summaries
                avg_tokens = total_tokens / num_summaries
                
                analysis_results.append({
                    "Model": model_name,
                    "Avg Sentences": avg_sentences,
                    "Avg Tokens": avg_tokens
                })

            except Exception as e:
                print(f"    - An unexpected error occurred with file '{filename}': {e}")

    if files_found == 0:
        print("Warning: No files ending with '_summaries.json' were found.")
        return pd.DataFrame()
                
    # --- 3. FORMATTING: Create and sort the final DataFrame ---
    summary_df = pd.DataFrame(analysis_results)
    
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by="Avg Tokens", ascending=False)
        summary_df['Avg Sentences'] = summary_df['Avg Sentences'].round(2)
        summary_df['Avg Tokens'] = summary_df['Avg Tokens'].round(2)

    return summary_df


def main():
    """
    Main function to execute the script.
    """
    # --- ⚙️ SETTINGS ⚙️ ---
    
    # 1. Set the path to your folder containing the .json files.
    SUMMARY_FILES_DIRECTORY = './'  # Use './' for the current directory

    # 2. Set the modern tokenizer you want to use.
    #    'google/gemma-2b' is an excellent choice as it's a current and open model.
    TOKENIZER_TO_USE = 'google/gemma-2b'

    # --- Run Analysis ---
    results_df = analyze_summary_metrics(SUMMARY_FILES_DIRECTORY, TOKENIZER_TO_USE)

    if results_df.empty:
        print("\nAnalysis did not produce any data. The output file will not be created.")
        return

    # --- Save and Print Results ---
    output_filename = 'modern_counts_sentence_and_token.csv'
    try:
        results_df.to_csv(output_filename, index=False)
        
        print("\n--- ✅ Analysis Complete ---")
        print("Average sentence and token count per summary (using Gemma tokenizer):")
        print(results_df.to_string(index=False))
        print("\n" + "="*65)
        print(f"SUCCESS: Results have been saved to the file: '{output_filename}'")
        print("="*65)

    except Exception as e:
        print(f"\nError: Could not save the results to '{output_filename}'. Reason: {e}")


# --- Script execution starts here ---
if __name__ == "__main__":
    main()