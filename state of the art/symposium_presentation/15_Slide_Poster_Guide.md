# 15-Slide Poster Presentation Guide
**SVD-Enhanced Recommendation Systems: Addressing Long-tail Bias Through Synthetic User Injection**

## Presentation Structure (3×5 Grid Layout)

### **ROW 1: Problem & Context (Slides 1-5)**

**Slide 1: Title & Research Overview**
- Project title: "SVD-Enhanced Recommendation Systems: Addressing Long-tail Bias"
- Your name and institution
- Key research question focus

**Slide 2: Industry Problem Statement** 
- Long-tail bias in recommendation systems
- Business impact: 40-60% of catalog underutilized
- Revenue implications for e-commerce platforms

**Slide 3: Dataset & Domain Analysis**
- Book dataset characteristics
- 13 genres analyzed
- **Figures to use:** `plot1_total_books.png`, `plot2_dataset_pie.png`

**Slide 4: Current State Analysis**
- Baseline recommendation patterns
- Primary vs secondary genre distribution
- **Figures to use:** `plot3_primary_vs_secondary_stacked.png`

**Slide 5: Research Questions**
- RQ1: Can synthetic user injection improve long-tail coverage?
- RQ2: What are optimal parameters for enhancement?
- RQ3: Does approach generalize across domains?

### **ROW 2: Methodology & Implementation (Slides 6-10)**

**Slide 6: SVD Enhancement Approach**
- Synthetic user injection methodology
- System architecture overview
- **Figures to use:** `system_architecture_diagram.png`

**Slide 7: Experimental Design**
- Parameters tested: K=15,25,35; N=25,50,100,200
- Cross-validation setup
- **Figures to use:** `experimental_design_diagram.png`

**Slide 8: Implementation Details**
- SVD algorithm modifications
- Code snippets and technical approach
- **Figures to use:** `code_snippet_svd.png`, `code_snippet_eval.png`

**Slide 9: Cross-Domain Validation**
- Books vs Movies comparison setup
- Domain transfer methodology
- **Figures to use:** `venn_books_movies.png`, `domain_comparison_table.png`

**Slide 10: Evaluation Framework**
- Metrics: coverage, accuracy, diversity
- Measurement methodology
- **Figures to use:** `metrics_table.png`, `rmse_mae_bars.png`

### **ROW 3: Results & Impact (Slides 11-15)**

**Slide 11: RQ1 - Long-tail Improvement**
- Before/after comparison results
- Coverage improvement across genres
- **Figures to use:** `rq1_longtail_before_after.png`, `rq1_heatmap_after.png`

**Slide 12: RQ2 - Dosage Response Analysis**
- Optimal parameter identification
- Performance vs injection volume
- **Figures to use:** `rq2_dose_response.png`, `rq2_gain_heatmap.png`

**Slide 13: RQ3 - Cross-domain Validation**
- Books vs Movies comparison results
- Generalization evidence
- **Figures to use:** `rq3_scatter_parity_books_movies.png`, `rq3_box_books_vs_movies.png`

**Slide 14: Industry Applications**
- Real-world deployment scenarios
- Business impact dashboard
- **Figures to use:** `impact_dashboard.png`, `industry_icons.png`

**Slide 15: Key Contributions & Future Work**
- Main takeaways for practitioners
- Next research directions
- **Figures to use:** `key_contributions_infographic.png`, `future_roadmap.png`

## Content Guidelines for Mixed Audience

### **For 60% Industry Audience:**
- Emphasize **business impact** and **ROI** of long-tail recommendations
- Show **before/after performance metrics** prominently
- Include **implementation complexity** and **scalability** considerations
- Highlight **cross-domain applicability** (books → other e-commerce)

### **For 20% Students:**
- Include **technical methodology details**
- Show **experimental rigor** and **validation approaches**
- Explain **algorithmic innovations**

### **For 20% Faculty:**
- Emphasize **research contributions** and **novelty**
- Show **statistical significance** and **theoretical foundations**
- Include **limitations** and **future research directions**

## Key Messages to Emphasize

1. **Problem:** Current recommendation systems miss 40-60% of catalog (long-tail items)
2. **Solution:** SVD enhancement with synthetic users increases long-tail coverage
3. **Validation:** Approach works across domains (books AND movies)
4. **Implementation:** Scalable method, integrable into existing systems
5. **Impact:** Measurable improvement in recommendation diversity and coverage

## Figure File Locations

All figures are organized in your symposium presentation folders:

- **Main Results:** `/symposium_presentation/slide_figs/`
- **Methodology:** `/symposium_presentation/setup_figs/`  
- **Implementation:** `/symposium_presentation/impl_figs/`
- **Business Impact:** `/symposium_presentation/impact_figs/`
- **Cross-domain:** `/symposium_presentation/cross_domain_figs/`
- **Domain Analysis:** `/result/domain_analysis/`

## Implementation Steps

1. **Open Template:** `Poster_Template_36x48.pptx`
2. **Maintain Settings:** Do NOT modify template formatting
3. **Populate Slides:** Follow 3×5 grid layout above
4. **Insert Figures:** Use high-resolution versions from folders listed
5. **Review Content:** Ensure industry focus with technical rigor
6. **Save Files:** Both .pptx and .pdf versions
7. **Print Setup:** Contact UC Printing Service for 36×48 poster

## Additional Resources

- **Detailed Experimental Results:** Check `/result/rec/top_re/0909/` and `/result/rec/top_re/0902/`
- **Genre-specific Analysis:** Individual genre plots in `/result/rec/top_re/0918/figures/`
- **Cross-validation Data:** Side-by-side comparisons in `/result/rec/top_re/0918/figures_side_by_side/`

**Your research demonstrates exceptional experimental rigor with comprehensive cross-domain validation. This poster structure effectively communicates both technical innovation and practical business value.**
