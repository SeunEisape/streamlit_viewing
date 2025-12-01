import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Model Results Graph Viewer", layout="wide")

# Title
st.title("Model Results Graph Viewer")

# Get the results directory
results_dir = Path("results")

# Get all model directories
models = sorted([d.name for d in results_dir.iterdir() if d.is_dir()])

if not models:
    st.error("No models found in the results directory!")
    st.stop()

# Sidebar for navigation
st.sidebar.header("Navigation")

# Model selection
selected_model = st.sidebar.selectbox(
    "Select Model",
    models,
    index=0
)

# Get all tasks for the selected model
model_dir = results_dir / selected_model
tasks = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])

if not tasks:
    st.warning(f"No tasks found for model: {selected_model}")
    st.stop()

# Task selection
selected_task = st.sidebar.selectbox(
    "Select Task",
    tasks,
    index=0
)

# Graph type selection
graph_types = []
task_dir = model_dir / selected_task

# Check what graph directories exist
if (task_dir / "graphs").exists():
    graph_types.append("Model Counts")
if (task_dir / "infinigram_graphs").exists():
    graph_types.append("Infinigram Counts")

if not graph_types:
    st.warning(f"No graph directories found for task: {selected_task}")
    st.stop()

# Allow multiple graph type selection
selected_graph_types = st.sidebar.multiselect(
    "Select Graph Types",
    graph_types,
    default=graph_types
)

# Display options
st.sidebar.header("Display Options")
cols_per_row = st.sidebar.slider("Graphs per row", 1, 3, 2)

# Main content area
st.header(f"Model: {selected_model}")
st.subheader(f"Task: {selected_task}")

# Display graphs
for graph_type in selected_graph_types:
    st.markdown(f"### {graph_type}")

    # Determine which directory to use
    if graph_type == "Model Counts":
        graph_dir = task_dir / "graphs"
    else:  # Infinigram Counts
        graph_dir = task_dir / "infinigram_graphs"

    # Get all images in this directory
    image_files = sorted(list(graph_dir.glob("*.png")) + list(graph_dir.glob("*.jpg")) + list(graph_dir.glob("*.jpeg")))

    if not image_files:
        st.info(f"No images found in {graph_dir}")
        continue

    # Display images in a grid
    for i in range(0, len(image_files), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(image_files):
                img_path = image_files[i + j]
                with col:
                    st.image(str(img_path), use_container_width=True, caption=img_path.name)

    st.markdown("---")

# Footer with statistics
st.sidebar.markdown("---")
st.sidebar.markdown("### Statistics")
st.sidebar.write(f"Total Models: {len(models)}")
st.sidebar.write(f"Tasks for {selected_model}: {len(tasks)}")

# Show task list
with st.sidebar.expander("All Tasks"):
    for task in tasks:
        st.write(f"- {task}")
