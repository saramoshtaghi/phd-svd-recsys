import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ================== CONFIG ==================
ENHANCED_DIR = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/0905/enhanced_analysis"
PRIMARY_DIR  = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/0905/primary_analysis"
GENRE_COL = "genres_all"
USER_COL  = "user_id"

# If True, divide by ALL users in each file (users with zero matches count as 0).
# If False, divide by only users with ≥1 match.
INCLUDE_ZEROS = True

# Pretty girly color palette
GIRLY_COLORS = {
    'original': '#FF1493',      # Deep Pink
    'enhanced_1': '#7b1b4d',    # Light Pink
    'enhanced_2': '#DDA0DD',    # Plum
    'enhanced_3': '#9aa8b9',    # Orchid
    'enhanced_4': '#FF69B4',    # Hot Pink
    'primary_1': '#F0E68C',     # Khaki
    'primary_2': '#a7ae89',     # Pale Green
    'primary_3': '#87CEEB',     # Sky Blue
    'primary_4': '#DEB887'      # Burlywood
}

# ================== FILENAME PATTERNS ==================
RE_ORIGINAL = re.compile(r"^ORIGINAL_(\d+)recommendation\.csv$", re.IGNORECASE)
RE_ENHANCED = re.compile(r"^enhanced_([^_]+(?:_[^_]+)*)_([0-9]+)_(\d+)recommendation\.csv$", re.IGNORECASE)
RE_PRIMARY  = re.compile(r"^primary_p_([^_]+(?:_[^_]+)*)_([0-9]+)_(\d+)recommendation\.csv$", re.IGNORECASE)

def parse_file_meta(filename):
    """
    Return dict with:
      'source': 'enhanced'|'primary'|'original'
      'genre': str|None
      'size': int|None
      'k': int
    or None if not matching.
    """
    m = RE_ORIGINAL.match(filename)
    if m:
        return {"source": "original", "genre": None, "size": None, "k": int(m.group(1))}
    m = RE_ENHANCED.match(filename)
    if m:
        return {"source": "enhanced", "genre": m.group(1), "size": int(m.group(2)), "k": int(m.group(3))}
    m = RE_PRIMARY.match(filename)
    if m:
        return {"source": "primary", "genre": m.group(1), "size": int(m.group(2)), "k": int(m.group(3))}
    return None

# ================== GENRE TOKENIZATION ==================
def tokenize_genres(cell):
    """Split a genre cell into normalized tokens (case-insensitive), matching exact tokens (not substrings)."""
    if pd.isna(cell):
        return set()
    tokens = re.split(r"[;,|]", str(cell))
    return set(t.strip().lower() for t in tokens if t.strip())

def user_genre_counts(df, target_genre):
    """Series indexed by user_id: #rows containing the target_genre token in GENRE_COL."""
    target = str(target_genre).lower()
    match_mask = df[GENRE_COL].apply(lambda x: target in tokenize_genres(x))
    matched = df[match_mask]
    counts = matched.groupby(USER_COL).size()  # only users with ≥1 match
    if INCLUDE_ZEROS:
        all_users = df[USER_COL].unique()
        counts = counts.reindex(all_users, fill_value=0)
    return counts

def avg_genre_per_user_for_file(csv_path, target_genre):
    df = pd.read_csv(csv_path)
    if USER_COL not in df.columns or GENRE_COL not in df.columns:
        raise ValueError(f"{csv_path} must contain '{USER_COL}' and '{GENRE_COL}' columns.")
    counts = user_genre_counts(df, target_genre)
    return 0.0 if counts.empty else counts.mean()

def get_user_genre_counts_for_binning(csv_path, target_genre):
    """Return Series of genre counts per user for binning analysis."""
    df = pd.read_csv(csv_path)
    if USER_COL not in df.columns or GENRE_COL not in df.columns:
        raise ValueError(f"{csv_path} must contain '{USER_COL}' and '{GENRE_COL}' columns.")
    return user_genre_counts(df, target_genre)

def create_user_bins(all_users, n_bins=10):
    """Create n_bins of equal size from sorted users."""
    sorted_users = sorted(all_users)
    bin_size = len(sorted_users) // n_bins
    bins = []
    
    for i in range(n_bins):
        start_idx = i * bin_size
        if i == n_bins - 1:  # Last bin gets any remaining users
            end_idx = len(sorted_users)
        else:
            end_idx = (i + 1) * bin_size
        bins.append(sorted_users[start_idx:end_idx])
    
    return bins

def calculate_bin_averages(user_counts_dict, user_bins, target_genre):
    """Calculate average genre count per bin for each dataset."""
    bin_averages = {}
    
    for dataset_name, user_counts in user_counts_dict.items():
        bin_avgs = []
        for bin_users in user_bins:
            # Get counts for users in this bin
            bin_counts = [user_counts.get(user, 0) for user in bin_users]
            bin_avg = np.mean(bin_counts) if bin_counts else 0.0
            bin_avgs.append(bin_avg)
        bin_averages[dataset_name] = bin_avgs
    
    return bin_averages

class ComprehensiveTextAnalysis:
    """Class to manage comprehensive text analysis across all genres and K values."""
    
    def __init__(self, analysis_type, output_dir):
        self.analysis_type = analysis_type
        self.output_dir = output_dir
        self.text_filename = f"{analysis_type}_comprehensive_analysis.txt"
        self.text_path = os.path.join(output_dir, self.text_filename)
        self.analyses = []
        
    def add_analysis(self, genre, k, bin_averages, user_bins):
        """Add a genre-k analysis to the comprehensive report."""
        self.analyses.append({
            'genre': genre,
            'k': k,
            'bin_averages': bin_averages,
            'user_bins': user_bins
        })
    
    def save_comprehensive_report(self):
        """Save all analyses to one comprehensive text file."""
        with open(self.text_path, 'w') as f:
            # Write header
            f.write("🌸" * 80 + "\n")
            f.write(f"COMPREHENSIVE {self.analysis_type.upper()} ANALYSIS REPORT\n")
            f.write("🌸" * 80 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Type: {self.analysis_type.title()}\n")
            f.write(f"Total Genre-K Combinations Analyzed: {len(self.analyses)}\n")
            
            if self.analyses:
                total_users = sum(len(bin_users) for bin_users in self.analyses[0]['user_bins'])
                f.write(f"Total Users Analyzed: {total_users:,}\n")
                f.write(f"User Binning: 10 equal bins\n")
            
            f.write("\n")
            
            # Table of Contents
            f.write("💕 TABLE OF CONTENTS 💕\n")
            f.write("-" * 40 + "\n")
            for i, analysis in enumerate(self.analyses, 1):
                genre = analysis['genre']
                k = analysis['k']
                f.write(f"{i:2d}. {genre} Genre - Top-{k} Analysis\n")
            f.write("\n" + "="*80 + "\n\n")
            
            # Individual analyses
            for i, analysis in enumerate(self.analyses, 1):
                self._write_individual_analysis(f, i, analysis)
            
            # Summary section
            self._write_summary_section(f)
            
        return self.text_path
    
    def _write_individual_analysis(self, f, section_num, analysis):
        """Write individual genre-k analysis."""
        genre = analysis['genre']
        k = analysis['k']
        bin_averages = analysis['bin_averages']
        user_bins = analysis['user_bins']
        
        f.write(f"🌺 SECTION {section_num}: {genre.upper()} GENRE - TOP-{k} ANALYSIS 🌺\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Genre: {genre}\n")
        f.write(f"Top-K: {k}\n")
        f.write(f"Number of User Bins: {len(user_bins)}\n")
        f.write(f"Users per Analysis: {sum(len(bin_users) for bin_users in user_bins):,}\n\n")
        
        # Bin size information
        f.write("👥 USER BIN INFORMATION:\n")
        f.write("-" * 40 + "\n")
        for i, bin_users in enumerate(user_bins, 1):
            f.write(f"Bin {i:2d}: {len(bin_users):,} users (IDs: {min(bin_users):,} - {max(bin_users):,})\n")
        f.write("\n")
        
        # Dataset overview
        f.write("📊 DATASETS ANALYZED:\n")
        f.write("-" * 40 + "\n")
        for dataset_name in bin_averages.keys():
            overall_avg = np.mean(bin_averages[dataset_name])
            f.write(f"• {dataset_name}: Overall Average = {overall_avg:.4f}\n")
        f.write("\n")
        
        # Detailed bin-by-bin analysis
        f.write("🔍 DETAILED BIN-BY-BIN ANALYSIS:\n")
        f.write("="*60 + "\n\n")
        
        for bin_idx in range(len(user_bins)):
            bin_num = bin_idx + 1
            bin_users = user_bins[bin_idx]
            
            f.write(f"📍 BIN {bin_num} ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Users in bin: {len(bin_users):,}\n")
            f.write(f"User ID range: {min(bin_users):,} - {max(bin_users):,}\n\n")
            
            f.write(f"Average {genre} recommendations per user:\n")
            
            # Sort datasets for consistent ordering
            sorted_datasets = []
            if "ORIGINAL" in bin_averages:
                sorted_datasets.append("ORIGINAL")
            
            # Add other datasets sorted by size if possible
            other_datasets = [d for d in bin_averages.keys() if d != "ORIGINAL"]
            
            # Try to sort by size
            def extract_size(dataset_name):
                if '_' in dataset_name:
                    try:
                        return int(dataset_name.split('_')[-1])
                    except:
                        return 0
                return 0
            
            other_datasets.sort(key=extract_size)
            sorted_datasets.extend(other_datasets)
            
            # Show values for each dataset
            baseline_val = None
            for dataset_name in sorted_datasets:
                value = bin_averages[dataset_name][bin_idx]
                if dataset_name == "ORIGINAL":
                    baseline_val = value
                    f.write(f"  {dataset_name:<15}: {value:8.4f} (baseline) 💖\n")
                else:
                    if baseline_val is not None and baseline_val > 0:
                        change_pct = ((value - baseline_val) / baseline_val) * 100
                        if change_pct > 5:
                            emoji = " 🌟"
                        elif change_pct > 0:
                            emoji = " ✨"
                        elif change_pct < -5:
                            emoji = " 💔"
                        else:
                            emoji = " 🌸"
                        change_str = f"({change_pct:+6.1f}%){emoji}"
                    else:
                        change_str = "(N/A) 🤔"
                    f.write(f"  {dataset_name:<15}: {value:8.4f} {change_str}\n")
            f.write("\n")
        
        # Statistical analysis
        f.write("📈 STATISTICAL SUMMARY:\n")
        f.write("="*40 + "\n\n")
        
        for dataset_name in sorted_datasets:
            values = bin_averages[dataset_name]
            f.write(f"💎 {dataset_name} Statistics:\n")
            f.write(f"  Mean:     {np.mean(values):8.4f}\n")
            f.write(f"  Median:   {np.median(values):8.4f}\n")
            f.write(f"  Std Dev:  {np.std(values):8.4f}\n")
            f.write(f"  Min:      {np.min(values):8.4f} (Bin {np.argmin(values)+1})\n")
            f.write(f"  Max:      {np.max(values):8.4f} (Bin {np.argmax(values)+1})\n")
            f.write(f"  Range:    {np.max(values) - np.min(values):8.4f}\n\n")
        
        # Comparative analysis
        if "ORIGINAL" in bin_averages and len(sorted_datasets) > 1:
            f.write("⚖️  COMPARATIVE ANALYSIS VS BASELINE:\n")
            f.write("="*50 + "\n\n")
            
            baseline_values = bin_averages["ORIGINAL"]
            
            for dataset_name in sorted_datasets[1:]:  # Skip ORIGINAL
                values = bin_averages[dataset_name]
                f.write(f"🆚 {dataset_name} vs ORIGINAL:\n")
                
                improvements = 0
                degradations = 0
                no_change = 0
                
                total_change = 0
                valid_comparisons = 0
                
                for i, (val, baseline) in enumerate(zip(values, baseline_values)):
                    if baseline > 0:
                        change_pct = ((val - baseline) / baseline) * 100
                        total_change += change_pct
                        valid_comparisons += 1
                        
                        if change_pct > 0.1:  # > 0.1% improvement
                            improvements += 1
                        elif change_pct < -0.1:  # > 0.1% degradation
                            degradations += 1
                        else:
                            no_change += 1
                
                if valid_comparisons > 0:
                    avg_change = total_change / valid_comparisons
                    
                    if avg_change > 5:
                        performance_emoji = " 🎉"
                    elif avg_change > 0:
                        performance_emoji = " 😊"
                    elif avg_change < -5:
                        performance_emoji = " 😢"
                    else:
                        performance_emoji = " 😐"
                    
                    f.write(f"  Average change: {avg_change:+6.2f}%{performance_emoji}\n")
                    f.write(f"  Bins improved: {improvements}/10 ({improvements*10:.0f}%) 💚\n")
                    f.write(f"  Bins degraded: {degradations}/10 ({degradations*10:.0f}%) 💙\n")
                    f.write(f"  Bins unchanged: {no_change}/10 ({no_change*10:.0f}%) 💜\n")
                else:
                    f.write(f"  No valid comparisons possible (baseline all zeros) 🚫\n")
                f.write("\n")
        
        f.write("\n" + "🌸" * 80 + "\n\n")
    
    def _write_summary_section(self, f):
        """Write overall summary across all analyses."""
        f.write("🌈 COMPREHENSIVE SUMMARY ACROSS ALL ANALYSES 🌈\n")
        f.write("="*80 + "\n\n")
        
        if not self.analyses:
            f.write("No analyses to summarize.\n")
            return
        
        # Overall statistics
        f.write("📋 OVERALL STATISTICS:\n")
        f.write("-" * 30 + "\n")
        
        all_genres = [a['genre'] for a in self.analyses]
        all_ks = [a['k'] for a in self.analyses]
        unique_genres = sorted(set(all_genres))
        unique_ks = sorted(set(all_ks))
        
        f.write(f"• Total analyses: {len(self.analyses)}\n")
        f.write(f"• Genres analyzed: {len(unique_genres)} ({', '.join(unique_genres)})\n")
        f.write(f"• K values analyzed: {len(unique_ks)} ({', '.join(map(str, unique_ks))})\n\n")
        
        # Best and worst performers summary
        f.write("🏆 BEST AND WORST PERFORMERS:\n")
        f.write("-" * 35 + "\n")
        
        best_improvements = []
        worst_improvements = []
        
        for analysis in self.analyses:
            genre = analysis['genre']
            k = analysis['k']
            bin_averages = analysis['bin_averages']
            
            if "ORIGINAL" not in bin_averages:
                continue
                
            baseline_values = bin_averages["ORIGINAL"]
            
            for dataset_name, values in bin_averages.items():
                if dataset_name == "ORIGINAL":
                    continue
                    
                # Calculate average improvement
                total_change = 0
                valid_comparisons = 0
                
                for val, baseline in zip(values, baseline_values):
                    if baseline > 0:
                        change_pct = ((val - baseline) / baseline) * 100
                        total_change += change_pct
                        valid_comparisons += 1
                
                if valid_comparisons > 0:
                    avg_change = total_change / valid_comparisons
                    best_improvements.append({
                        'genre': genre,
                        'k': k,
                        'dataset': dataset_name,
                        'avg_change': avg_change
                    })
        
        if best_improvements:
            # Sort by average change
            best_improvements.sort(key=lambda x: x['avg_change'], reverse=True)
            
            f.write("🥇 TOP 5 BEST IMPROVEMENTS:\n")
            for i, item in enumerate(best_improvements[:5], 1):
                emoji = "🌟" if item['avg_change'] > 10 else "✨" if item['avg_change'] > 5 else "⭐"
                f.write(f"{i}. {item['genre']} Top-{item['k']} - {item['dataset']}: {item['avg_change']:+.2f}% {emoji}\n")
            
            f.write(f"\n🥉 TOP 5 WORST PERFORMANCES:\n")
            for i, item in enumerate(best_improvements[-5:], 1):
                emoji = "💔" if item['avg_change'] < -10 else "😢" if item['avg_change'] < -5 else "😐"
                f.write(f"{i}. {item['genre']} Top-{item['k']} - {item['dataset']}: {item['avg_change']:+.2f}% {emoji}\n")
        
        f.write("\n")
        
        # Conclusions
        f.write("💭 KEY INSIGHTS AND CONCLUSIONS:\n")
        f.write("-" * 40 + "\n")
        f.write("1. This comprehensive analysis covers all genre-recommendation combinations\n")
        f.write("2. User binning reveals how different user segments respond to synthetic bias\n")
        f.write("3. Percentage changes help identify effective vs ineffective bias injection strategies\n")
        f.write("4. Statistical summaries provide quantitative foundation for research conclusions\n")
        f.write("5. Comparative analysis highlights which approaches work best for specific genres\n\n")
        
        f.write("🌸" * 80 + "\n")
        f.write("END OF COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("🌸" * 80 + "\n")

# ================== ENHANCED FOLDER PROCESSOR ==================
def process_enhanced_folder(input_dir):
    """Process enhanced folder with user binning analysis and comprehensive text output"""
    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    meta = []
    for f in files:
        info = parse_file_meta(f)
        if info:
            info["filename"] = f
            meta.append(info)

    # Where to save
    out_plot_dir = os.path.join(input_dir, "genre_avg_plots")
    os.makedirs(out_plot_dir, exist_ok=True)
    out_summary_csv = os.path.join(input_dir, "genre_avg_summary.csv")

    # Initialize comprehensive text analysis
    text_analysis = ComprehensiveTextAnalysis("enhanced", input_dir)

    # Available genres from enhanced files only
    genres = sorted({m["genre"] for m in meta if m["source"] == "enhanced" and m["genre"] is not None})
    
    print(f"📁 Processing Enhanced folder: {len(genres)} genres")

    # Pre-index meta by type for speed
    originals_by_k = {}
    enhanced_by_genre_k = defaultdict(list)

    for m in meta:
        if m["source"] == "original":
            originals_by_k[m["k"]] = m
        elif m["source"] == "enhanced":
            enhanced_by_genre_k[(m["genre"], m["k"])].append(m)

    # Get all unique users from original file (using any K)
    if originals_by_k:
        sample_original = list(originals_by_k.values())[0]
        sample_df = pd.read_csv(os.path.join(input_dir, sample_original["filename"]))
        all_users = sorted(sample_df[USER_COL].unique())
        print(f"📊 Total unique users for binning: {len(all_users):,}")
        
        # Create 10 user bins
        user_bins = create_user_bins(all_users, n_bins=10)
        print(f"📊 Created {len(user_bins)} bins with sizes: {[len(bin) for bin in user_bins]}")
    else:
        print("⚠️ No original files found for user binning")
        return {}

    summary_rows = []
    plot_count = 0

    # Process each genre
    for genre in genres:
        Ks = sorted({k for (g,k) in enhanced_by_genre_k.keys() if g == genre} | set(originals_by_k.keys()))
        
        # Process each K value
        for k in Ks:
            print(f"🎯 Processing {genre} - Top-{k}")
            
            # Collect user counts for all datasets for this genre-k combination
            user_counts_dict = {}
            
            # ORIGINAL_K (if present)
            if k in originals_by_k:
                f = originals_by_k[k]["filename"]
                file_path = os.path.join(input_dir, f)
                user_counts = get_user_genre_counts_for_binning(file_path, genre)
                user_counts_dict["ORIGINAL"] = user_counts
                
                # Save summary data
                avg_o = user_counts.mean()
                summary_rows.append({
                    "folder": "enhanced",
                    "genre": genre,
                    "k": k,
                    "dataset": f.replace(".csv", ""),
                    "source": "original",
                    "size": None,
                    "avg_per_user": avg_o
                })

            # ENHANCED datasets for this genre and K
            enhanced_files = sorted(enhanced_by_genre_k.get((genre, k), []), key=lambda x: x["size"])
            sizes = []
            
            for m in enhanced_files:
                f = m["filename"]
                file_path = os.path.join(input_dir, f)
                user_counts = get_user_genre_counts_for_binning(file_path, genre)
                user_counts_dict[f"Enhanced_{m['size']}"] = user_counts
                sizes.append(m['size'])
                
                # Save summary data
                avg_e = user_counts.mean()
                summary_rows.append({
                    "folder": "enhanced",
                    "genre": genre,
                    "k": k,
                    "dataset": f.replace(".csv", ""),
                    "source": "enhanced",
                    "size": m["size"],
                    "avg_per_user": avg_e
                })

            # Calculate bin averages for all datasets
            if user_counts_dict:
                bin_averages = calculate_bin_averages(user_counts_dict, user_bins, genre)
                
                # Add to comprehensive text analysis
                text_analysis.add_analysis(genre, k, bin_averages, user_bins)
                
                # Create the binned comparison plot with girly colors
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Prepare data for plotting
                bin_numbers = list(range(1, 11))
                bar_width = 0.15
                
                datasets = []
                colors = []
                
                if "ORIGINAL" in bin_averages:
                    datasets.append(("ORIGINAL", bin_averages["ORIGINAL"]))
                    colors.append(GIRLY_COLORS['original'])
                
                # Add enhanced datasets in size order with pretty colors
                color_keys = ['enhanced_1', 'enhanced_2', 'enhanced_3', 'enhanced_4']
                for i, size in enumerate(sorted(sizes)):
                    key = f"Enhanced_{size}"
                    if key in bin_averages:
                        datasets.append((f"Enhanced_{size}", bin_averages[key]))
                        if i < len(color_keys):
                            colors.append(GIRLY_COLORS[color_keys[i]])
                        else:
                            colors.append('#FFB6C1')  # Default light pink
                
                # Plot bars
                for i, (label, values) in enumerate(datasets):
                    x_positions = [x + i * bar_width for x in bin_numbers]
                    bars = ax.bar(x_positions, values, bar_width, 
                                 label=label, color=colors[i], alpha=0.8, edgecolor='white', linewidth=0.8)
                    
                    # Add value labels on bars (only if not too many bars)
                    if len(datasets) <= 5:
                        for bar, value in zip(bars, values):
                            if value > 0:
                                ax.text(bar.get_x() + bar.get_width()/2., 
                                       bar.get_height() + 0.001,
                                       f'{value:.3f}', ha='center', va='bottom', 
                                       fontsize=8, rotation=0, fontweight='bold')
                
                # Customize plot with girly styling
                ax.set_xlabel('User Bins (1=Lowest User IDs, 10=Highest User IDs)', fontsize=12, fontweight='bold')
                ax.set_ylabel(f'Average #{genre} Recommendations per User', fontsize=12, fontweight='bold')
                ax.set_title(f'🌸 Enhanced Analysis: {genre} Genre - Top-{k} 🌸\nUser Binning Comparison (10 Equal Bins)', 
                           fontsize=14, fontweight='bold', color='#FF1493')
                ax.set_xticks([x + bar_width * (len(datasets)-1) / 2 for x in bin_numbers])
                ax.set_xticklabels([f'Bin {i}' for i in bin_numbers])
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
                ax.grid(True, alpha=0.3, axis='y', color='pink', linestyle='--')
                ax.set_facecolor('#FFFAFD')  # Very light pink background
                
                # Better y-axis scaling
                if any(any(values) for _, values in datasets):
                    all_values = [v for _, values in datasets for v in values if v > 0]
                    if all_values:
                        y_max = max(all_values)
                        y_min = min(all_values)
                        margin = (y_max - y_min) * 0.1 if y_max != y_min else y_max * 0.1
                        ax.set_ylim(max(0, y_min - margin), y_max + margin)
                
                plt.tight_layout()
                
                # Save plot
                out_png = os.path.join(out_plot_dir, f"enhanced_{genre}_top{k}_binned_comparison.png")
                plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                plot_count += 1
                print(f"📊 Saved enhanced binned figure: {out_png}")

    # Save comprehensive text analysis
    text_path = text_analysis.save_comprehensive_report()
    print(f"📝 Saved comprehensive enhanced text analysis: {text_path}")

    # Save summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(["genre", "k", "source", "size"], na_position="first")
        summary_df.to_csv(out_summary_csv, index=False)
        print(f"✅ Saved enhanced summary: {out_summary_csv}")

    print(f"🎉 Enhanced folder done: {plot_count} plots and 1 comprehensive text analysis generated")
    return {}

# ================== PRIMARY FOLDER PROCESSOR ==================
def process_primary_folder(input_dir, enhanced_dir):
    """Process primary folder with user binning analysis and comprehensive text output"""
    # Get primary files from primary folder
    primary_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    primary_meta = []
    for f in primary_files:
        info = parse_file_meta(f)
        if info and info["source"] == "primary":
            info["filename"] = f
            primary_meta.append(info)

    # Get original files from enhanced folder
    enhanced_files = [f for f in os.listdir(enhanced_dir) if f.endswith(".csv")]
    original_meta = []
    for f in enhanced_files:
        info = parse_file_meta(f)
        if info and info["source"] == "original":
            info["filename"] = f
            info["filepath"] = os.path.join(enhanced_dir, f)  # Full path to enhanced folder
            original_meta.append(info)

    # Where to save
    out_plot_dir = os.path.join(input_dir, "genre_avg_plots")
    os.makedirs(out_plot_dir, exist_ok=True)
    out_summary_csv = os.path.join(input_dir, "genre_avg_summary.csv")

    # Initialize comprehensive text analysis
    text_analysis = ComprehensiveTextAnalysis("primary", input_dir)

    # Available genres from primary files
    genres = sorted({m["genre"] for m in primary_meta if m["genre"] is not None})
    
    print(f"📁 Processing Primary folder: {len(genres)} genres, borrowing originals from enhanced")

    # Pre-index meta
    originals_by_k = {m["k"]: m for m in original_meta}
    primary_by_genre_k = defaultdict(list)
    for m in primary_meta:
        primary_by_genre_k[(m["genre"], m["k"])].append(m)

    # Get all unique users from original file (using any K)
    if originals_by_k:
        sample_original = list(originals_by_k.values())[0]
        sample_df = pd.read_csv(sample_original["filepath"])
        all_users = sorted(sample_df[USER_COL].unique())
        print(f"📊 Total unique users for binning: {len(all_users):,}")
        
        # Create 10 user bins
        user_bins = create_user_bins(all_users, n_bins=10)
        print(f"📊 Created {len(user_bins)} bins with sizes: {[len(bin) for bin in user_bins]}")
    else:
        print("⚠️ No original files found for user binning")
        return

    summary_rows = []
    plot_count = 0

    # Process each genre
    for genre in genres:
        Ks = sorted({k for (g,k) in primary_by_genre_k.keys() if g == genre} | set(originals_by_k.keys()))

        # Process each K value
        for k in Ks:
            print(f"🎯 Processing {genre} - Top-{k}")
            
            # Collect user counts for all datasets for this genre-k combination
            user_counts_dict = {}
            
            # ORIGINAL_K (borrowed from enhanced folder)
            if k in originals_by_k:
                orig_info = originals_by_k[k]
                user_counts = get_user_genre_counts_for_binning(orig_info["filepath"], genre)
                user_counts_dict["ORIGINAL"] = user_counts
                
                # Save summary data
                avg_o = user_counts.mean()
                summary_rows.append({
                    "folder": "primary",
                    "genre": genre,
                    "k": k,
                    "dataset": orig_info["filename"].replace(".csv", ""),
                    "source": "original_borrowed",
                    "size": None,
                    "avg_per_user": avg_o
                })

            # PRIMARY datasets for this genre and K
            primary_files = sorted(primary_by_genre_k.get((genre, k), []), key=lambda x: x["size"])
            sizes = []
            
            for m in primary_files:
                f = m["filename"]
                file_path = os.path.join(input_dir, f)
                user_counts = get_user_genre_counts_for_binning(file_path, genre)
                user_counts_dict[f"Primary_{m['size']}"] = user_counts
                sizes.append(m['size'])
                
                # Save summary data
                avg_p = user_counts.mean()
                summary_rows.append({
                    "folder": "primary",
                    "genre": genre,
                    "k": k,
                    "dataset": f.replace(".csv", ""),
                    "source": "primary",
                    "size": m["size"],
                    "avg_per_user": avg_p
                })

            # Calculate bin averages for all datasets
            if user_counts_dict:
                bin_averages = calculate_bin_averages(user_counts_dict, user_bins, genre)
                
                # Add to comprehensive text analysis
                text_analysis.add_analysis(genre, k, bin_averages, user_bins)
                
                # Create the binned comparison plot with girly colors
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Prepare data for plotting
                bin_numbers = list(range(1, 11))
                bar_width = 0.15
                
                datasets = []
                colors = []
                
                if "ORIGINAL" in bin_averages:
                    datasets.append(("ORIGINAL", bin_averages["ORIGINAL"]))
                    colors.append(GIRLY_COLORS['original'])
                
                # Add primary datasets in size order with pretty colors
                color_keys = ['primary_1', 'primary_2', 'primary_3', 'primary_4']
                for i, size in enumerate(sorted(sizes)):
                    key = f"Primary_{size}"
                    if key in bin_averages:
                        datasets.append((f"Primary_{size}", bin_averages[key]))
                        if i < len(color_keys):
                            colors.append(GIRLY_COLORS[color_keys[i]])
                        else:
                            colors.append('#98FB98')  # Default pale green
                
                # Plot bars
                for i, (label, values) in enumerate(datasets):
                    x_positions = [x + i * bar_width for x in bin_numbers]
                    bars = ax.bar(x_positions, values, bar_width, 
                                 label=label, color=colors[i], alpha=0.8, edgecolor='white', linewidth=0.8)
                    
                    # Add value labels on bars (only if not too many bars)
                    if len(datasets) <= 5:
                        for bar, value in zip(bars, values):
                            if value > 0:
                                ax.text(bar.get_x() + bar.get_width()/2., 
                                       bar.get_height() + 0.001,
                                       f'{value:.3f}', ha='center', va='bottom', 
                                       fontsize=8, rotation=0, fontweight='bold')
                
                # Customize plot with girly styling
                ax.set_xlabel('User Bins (1=Lowest User IDs, 10=Highest User IDs)', fontsize=12, fontweight='bold')
                ax.set_ylabel(f'Average #{genre} Recommendations per User', fontsize=12, fontweight='bold')
                ax.set_title(f'🌺 Primary Analysis: {genre} Genre - Top-{k} 🌺\nUser Binning Comparison (10 Equal Bins)', 
                           fontsize=14, fontweight='bold', color='#FF1493')
                ax.set_xticks([x + bar_width * (len(datasets)-1) / 2 for x in bin_numbers])
                ax.set_xticklabels([f'Bin {i}' for i in bin_numbers])
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
                ax.grid(True, alpha=0.3, axis='y', color='pink', linestyle='--')
                ax.set_facecolor('#FFFAFD')  # Very light pink background
                
                # Better y-axis scaling
                if any(any(values) for _, values in datasets):
                    all_values = [v for _, values in datasets for v in values if v > 0]
                    if all_values:
                        y_max = max(all_values)
                        y_min = min(all_values)
                        margin = (y_max - y_min) * 0.1 if y_max != y_min else y_max * 0.1
                        ax.set_ylim(max(0, y_min - margin), y_max + margin)
                
                plt.tight_layout()
                
                # Save plot
                out_png = os.path.join(out_plot_dir, f"primary_{genre}_top{k}_binned_comparison.png")
                plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                plot_count += 1
                print(f"📊 Saved primary binned figure: {out_png}")

    # Save comprehensive text analysis
    text_path = text_analysis.save_comprehensive_report()
    print(f"📝 Saved comprehensive primary text analysis: {text_path}")

    # Save summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(["genre", "k", "source", "size"], na_position="first")
        summary_df.to_csv(out_summary_csv, index=False)
        print(f"✅ Saved primary summary: {out_summary_csv}")

    print(f"🎉 Primary folder done: {plot_count} plots and 1 comprehensive text analysis generated")

# ================== RUN BOTH FOLDERS ==================
print("🚀 Starting user binning analysis with comprehensive text output...")
enhanced_results = process_enhanced_folder(ENHANCED_DIR)
process_primary_folder(PRIMARY_DIR, ENHANCED_DIR)
print("🎯 All analyses completed!")