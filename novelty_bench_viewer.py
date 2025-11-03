import streamlit as st
import json
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="NoveltyBench Results Viewer", layout="wide")

def load_scores(file_path):
    """Load scores from JSONL file"""
    scores = []
    with open(file_path, 'r') as f:
        for line in f:
            scores.append(json.loads(line))
    return scores

def create_global_dataframe(scores):
    """Create a global view of all generations across all tasks"""
    rows = []
    for task in scores:
        task_id = task['id']
        prompt = task['prompt']
        generations = task['generations']
        rewards = task['generation_raw_rewards']
        gen_scores = task.get('generation_scores', [None] * len(generations))
        partitions = task.get('partition', [None] * len(generations))

        for i, (gen, reward, score, partition) in enumerate(zip(generations, rewards, gen_scores, partitions)):
            rows.append({
                'task_id': task_id,
                'prompt': prompt,
                'generation_idx': i,
                'generation': gen,
                'raw_reward': reward,
                'generation_score': score,
                'partition': partition
            })

    return pd.DataFrame(rows)

def main():
    st.title("NoveltyBench Results Viewer")

    # File path input
    default_path = "novelty-bench/results/curated/gpt4o/scores.jsonl"
    file_path = st.text_input("Path to scores.jsonl", value=default_path)

    if not Path(file_path).exists():
        st.error(f"File not found: {file_path}")
        return

    # Load data
    scores = load_scores(file_path)
    global_df = create_global_dataframe(scores)

    st.write(f"Loaded {len(scores)} tasks with {len(global_df)} total generations")

    # View mode selection
    view_mode = st.radio("View Mode", ["Task Level", "Global Level"], horizontal=True)

    # Search functionality
    st.subheader("Search and Filter")
    col1, col2 = st.columns(2)

    with col1:
        search_query = st.text_input("Search in prompts and generations", "")

    with col2:
        task_filter = st.multiselect(
            "Filter by Task ID",
            options=sorted(global_df['task_id'].unique()),
            default=None
        )

    # Apply filters
    filtered_df = global_df.copy()

    if search_query:
        mask = (
            filtered_df['prompt'].str.contains(search_query, case=False, na=False) |
            filtered_df['generation'].str.contains(search_query, case=False, na=False)
        )
        filtered_df = filtered_df[mask]

    if task_filter:
        filtered_df = filtered_df[filtered_df['task_id'].isin(task_filter)]

    if view_mode == "Global Level":
        # Global view - show all generations sorted by reward
        st.subheader("Global View - All Generations")

        sort_order = st.radio("Sort by reward", ["Highest first", "Lowest first"], horizontal=True)
        ascending = (sort_order == "Lowest first")

        sorted_df = filtered_df.sort_values('raw_reward', ascending=ascending).reset_index(drop=True)

        # Show statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Generations", len(sorted_df))
        with col2:
            st.metric("Mean Reward", f"{sorted_df['raw_reward'].mean():.4f}")
        with col3:
            st.metric("Max Reward", f"{sorted_df['raw_reward'].max():.4f}")
        with col4:
            st.metric("Min Reward", f"{sorted_df['raw_reward'].min():.4f}")

        # Display results
        st.write("---")
        for idx, row in sorted_df.iterrows():
            with st.expander(f"Rank {idx+1} | Task: {row['task_id']} | Reward: {row['raw_reward']:.4f} | Score: {row['generation_score']}"):
                st.write("**Prompt:**")
                st.write(row['prompt'])
                st.write("")
                st.write("**Generation:**")
                st.write(row['generation'])
                st.write("")
                st.write(f"**Metadata:** Generation #{row['generation_idx']}, Partition: {row['partition']}")

    else:
        # Task level view
        st.subheader("Task Level View")

        # Get unique tasks from filtered data
        available_tasks = filtered_df['task_id'].unique()

        if len(available_tasks) == 0:
            st.warning("No tasks match your search criteria")
            return

        # Task selector
        selected_task_id = st.selectbox("Select Task", sorted(available_tasks))

        # Get task data
        task_data = next((t for t in scores if t['id'] == selected_task_id), None)

        if not task_data:
            st.error("Task not found")
            return

        # Display task info
        st.write("**Prompt:**")
        st.info(task_data['prompt'])

        # Show task statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Generations", len(task_data['generations']))
        with col2:
            st.metric("Distinct Partitions", task_data['distinct'])
        with col3:
            st.metric("Utility", f"{task_data.get('utility', 0):.4f}")
        with col4:
            mean_reward = sum(task_data['generation_raw_rewards']) / len(task_data['generation_raw_rewards'])
            st.metric("Mean Reward", f"{mean_reward:.4f}")

        # Sort order for task level
        sort_order = st.radio("Sort generations by reward", ["Highest first", "Lowest first"], horizontal=True)
        ascending = (sort_order == "Lowest first")

        # Create task-level dataframe
        task_df = filtered_df[filtered_df['task_id'] == selected_task_id].copy()
        task_df = task_df.sort_values('raw_reward', ascending=ascending).reset_index(drop=True)

        # Display generations
        st.write("---")
        st.subheader("Generations")

        for idx, row in task_df.iterrows():
            reward = row['raw_reward']
            score = row['generation_score']
            partition = row['partition']
            gen_idx = row['generation_idx']

            # Color code by reward
            if reward > 0:
                badge = "🟢"
            elif reward > -2:
                badge = "🟡"
            else:
                badge = "🔴"

            with st.expander(f"{badge} Rank {idx+1} | Gen #{gen_idx} | Reward: {reward:.4f} | Score: {score} | Partition: {partition}"):
                st.write(row['generation'])

        # Show partition summary
        st.write("---")
        st.subheader("Partition Summary")

        partition_df = pd.DataFrame({
            'Partition': task_data.get('partition', []),
            'Score': task_data.get('generation_scores', []),
            'Reward': task_data['generation_raw_rewards']
        })

        partition_summary = partition_df.groupby('Partition').agg({
            'Score': 'first',
            'Reward': ['mean', 'min', 'max', 'count']
        }).round(4)

        st.dataframe(partition_summary, use_container_width=True)

if __name__ == "__main__":
    main()
