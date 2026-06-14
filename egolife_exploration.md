# EgoLife Exploration: Chunk Size Ablation for Event-Rich Videos

## Motivation

Current evaluation of long-form egocentric video summarization (using ROUGE-L and BERTScore against Ego4D narration summaries) fails to capture the key claim we want to make:

> **Smaller chunks capture more fine-grained event details, especially in event-rich videos.**

The ground truth narration summaries in Ego4D are written at a high abstraction level (~5-minute sliding windows), so ROUGE-L rewards generic language and BERTScore stays nearly flat across chunk sizes (79–81%). Neither metric is sensitive to event density or temporal precision — exactly what we want to demonstrate.

---

## Why the Current Metric Falls Short

| Issue | Detail |
|---|---|
| Ground truth abstraction | Narration summaries describe activity at a 5-min granularity, not event level |
| Vocabulary mismatch | Annotators write "C hiked with group"; model writes "people on a trail" — ROUGE-L penalizes this even when semantically equivalent |
| BERTScore too flat | Only 2-point spread across chunk sizes — not sensitive enough to show ablation benefit |
| Small sample | Only 3 videos evaluated so far — insufficient for a paper figure |

---

## Proposed Evaluation Ideas

### Idea 1: QA-Based Event Recall (Recommended)

**Setup:**
1. For each video, use GPT-4 to generate event-specific QA pairs from the ground truth annotations.
   ```
   Q: Did the person chop vegetables?     A: Yes
   Q: What did they prepare first?        A: Onions
   Q: Did they use the microwave?         A: No
   ```
2. Feed each chunk-size's final summary to an LLM and ask it to answer those QA pairs.
3. **Score = % of questions answered correctly.**

**Why it works:** Smaller chunks → more granular event capture → higher QA accuracy. The ablation graph becomes clean: chunk size (2/5/10 min) on X-axis, QA accuracy on Y-axis, with a consistent downward trend as chunk size grows.

**EgoLife fit:** EgoLife has episodic memory QA tasks already annotated (e.g., "What did you have for lunch on Tuesday?") — these can be used directly without generating new QA pairs.

---

### Idea 2: LLM-as-Judge Pairwise Comparison

**Setup:** For each video, show GPT-4 two summaries (e.g., 2min vs 10min chunks) alongside the ground truth and ask:
> *"Which summary mentions more distinct events from the ground truth? Rate each on event coverage (1–10)."*

Run all pairs: 2min vs 5min, 5min vs 10min, 2min vs 10min. Report **win rate %**.

**Why it's compelling:** Interpretable, paraphrase-robust, cheap (API calls only). Pairwise design controls for video-level difficulty.

---

### Idea 3: LLM-Extracted Event List Recall

**Setup:**
1. Use GPT-4 to extract atomic events from ground truth:
   ```
   ["person hikes uphill", "group crosses wooden bridge", "person takes photo at summit", ...]
   ```
2. For each chunk-size summary, check which events are mentioned (via LLM).
3. **Event Recall = events captured / total events in ground truth.**

**Graph:** Event recall % vs chunk size, with error bars per dataset. Clean and intuitive for a paper figure.

---

### Idea 4: Temporal Precision Score (Novel Angle)

Not just *what* was captured but *when* it was captured matters for event-rich videos.

- Each chunk caption is naturally timestamped by its window (e.g., 0–2 min, 2–4 min).
- Ground truth narration timestamps give the true event time.
- **Metric: Mean Absolute Error (MAE) between predicted event time and true event time.**

This is a unique contribution — no current paper in long-form video captioning highlights temporal precision of chunk-level summarization. Smaller chunks give finer temporal resolution, which directly translates to lower MAE.

---

## EgoLife Dataset Opportunities

EgoLife is particularly well-suited for this ablation because:

| Property | Why It Helps |
|---|---|
| Multi-day recordings (up to hours) | Shows scaling behavior of chunk size effect beyond 30 min |
| Episodic memory QA already annotated | Directly usable for Idea 1 without extra annotation cost |
| Daily activity logs | Enables event list extraction (Idea 3) |
| Social interaction annotations | High event density — multiple people/events per minute — amplifies the chunk size difference |
| Longer videos (up to 60 min) | Effect of chunk size is more dramatic at longer durations |

**Key argument for EgoLife:** A 60-min EgoLife clip may have 50+ distinct events. A 10-min chunk compresses these into 6 summaries; a 2-min chunk gives 30 more precise windows. The difference in event recall will be much more dramatic than on Ego4D 30-min clips.

---

## Experimental Design for the Paper Figure

### Recommended Setup

| Parameter | Choice |
|---|---|
| Metric | QA-based event recall (Idea 1) |
| Datasets | ~10 Ego4D 30-min clips + ~10 EgoLife 60-min clips |
| Chunk sizes | 2 min, 5 min, 10 min |
| Design | Within-subject (same video tested at all chunk sizes) |
| Graph | Line plot: X = chunk size, Y = QA accuracy %, two lines (Ego4D / EgoLife), error bars |

### How Many Videos?

| Goal | # Videos | Reasoning |
|---|---|---|
| Motivating figure (trend only) | **10–15** | Consistent direction suffices; reviewers accept this for ablations |
| Statistically significant (p < 0.05) | **20–30** | Within-subject design needs fewer than between-subject |
| Strong claim with confidence intervals | **30–50** | Required if this is a primary result, not a supporting figure |

**Recommendation:** 15–20 event-rich videos for a motivating figure. Select videos by **event density** (number of narration summaries per minute) to ensure the videos are genuinely event-rich — otherwise chunk size differences won't be visible.

### Video Selection Criterion

Filter for event-rich videos using:
```
event_density = num_narration_summaries / video_duration_minutes
```
Select the top-N videos by this metric. For Ego4D 30-min clips, this means selecting videos with the most narration_pass_1 summaries within the 30-min window.

---

## Story for Reviewers

> *"We observe that smaller chunk sizes preserve event-level detail more faithfully. This effect grows with video length and event density, as demonstrated on both Ego4D (30-min) and EgoLife (60-min) datasets. Standard metrics like ROUGE-L and BERTScore are insensitive to this, motivating our use of QA-based event recall as the primary evaluation metric for event-rich long-form video understanding."*

---

## Next Steps

- [ ] Explore EgoLife annotation format (QA pairs, activity logs, episodic memory tasks)
- [ ] Implement event density filter to select top-N event-rich videos from Ego4D and EgoLife
- [ ] Generate QA pairs from ground truth using GPT-4 (or use existing EgoLife QA)
- [ ] Run chunk-size ablation (2/5/10 min) on selected videos using QA-based event recall
- [ ] Plot results as line graph with error bars for paper figure
