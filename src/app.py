import random
import gradio as gr
import pandas as pd

# Import necessary components from PyABSA for dataset listing
from pyabsa import AspectTermExtraction as ATEPC

# Import the initialized ATEPC model and utility functions from our local files
from src.models import aspect_extractor
from src.utils import load_atepc_examples


# Defining the specific datasets
DESIRED_ATEPC_DATASETS = [
    "Laptop14",
    "Restaurant14",
    "SemEval",
    "Twitter",
    "TShirt"
]

# Pre load Dataset Examples
# This dictionary will store example sentences for each desired dataset,
# which are used when the user leaves the input text box blank.

print("Loading ATEPC dataset examples for Gradio interface...")
atepc_dataset_examples = {}
# Iterate through available ATEPC datasets and load examples for each
# Filter to only include the desired datasets
for dataset_name in DESIRED_ATEPC_DATASETS:
    try:
        # Check if the dataset name is valid in ATEPC.ATEPCDatasetList
        if hasattr(ATEPC.ATEPCDatasetList(), dataset_name):
            atepc_dataset_examples[dataset_name] = load_atepc_examples(dataset_name)
        else:
            print(f"Warning: Dataset '{dataset_name}' not found in ATEPC.ATEPCDatasetList. Skipping.")
    except Exception as e:
        print(f"Error loading examples for ATEPC dataset '{dataset_name}': {e}")
print("ATEPC dataset examples loading complete.")


# Inference Function for Gradio

def run_atepc_inference(input_text: str, selected_dataset: str) -> tuple[pd.DataFrame, str]:

    # Check if the aspect_extractor model was successfully initialized
    if aspect_extractor is None:
        return pd.DataFrame({"Error": ["Model not initialized. Please check logs."]},
                            columns=["Error"]), "Model Unavailable"

    analyzed_text = input_text.strip() # Remove leading/trailing whitespace

    # If no text is provided, select a random example from the pre-loaded data
    if not analyzed_text:
        examples = atepc_dataset_examples.get(selected_dataset)
        if examples:
            analyzed_text = random.choice(examples)
        else:
            return pd.DataFrame({"Message": ["No examples available for this dataset or input text provided."]},
                                columns=["Message"]), "Please provide text or select a valid dataset."

    print(f"Performing ATEPC Inference on: '{analyzed_text}' (Dataset: {selected_dataset})")

    try:
        # Predict aspects and their sentiments
        prediction_result = aspect_extractor.predict(analyzed_text, pred_sentiment=True)

        # Check if any aspects were detected
        if not prediction_result or not prediction_result.get("aspect"):
            return pd.DataFrame({"Message": ["No aspects detected for the given text."]},
                                columns=["Message"]), analyzed_text

        # Create a DataFrame from the prediction results
        df_result = pd.DataFrame(
            {
                "Aspect": prediction_result["aspect"],
                "Sentiment": prediction_result["sentiment"],
                "Confidence": [round(c, 4) for c in prediction_result["confidence"]],
                "Position": prediction_result["position"],
            }
        )
        return df_result, analyzed_text
    except Exception as e:
        print(f"Error during ATEPC inference: {e}")
        return pd.DataFrame({"Error": [f"An error occurred: {e}"]},
                            columns=["Error"]), analyzed_text


# Gradio User Interface Definition

# Initialize the Gradio Blocks interface
with gr.Blocks(title="PyABSA Demonstration: Aspect-based Sentiment Analysis") as sentiment_analysis_app:
    # Main title for the entire application
    gr.Markdown("# <p align='center'>PyABSA: Multilingual Aspect-based Sentiment Analysis</p>")
    gr.Markdown("---") # Visual separator

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 📈 Analyze Aspects and Sentiments")
            gr.Markdown(
                "This tool identifies specific aspects (entities or attributes) in a sentence "
                "and determines the sentiment (positive, negative, neutral) associated with each. "
                "For example, in 'The laptop's battery life is excellent', 'battery life' would be "
                "identified with a 'positive' sentiment."
            )

            # Input area for ATEPC
            atepc_input_box = gr.Textbox(
                placeholder="Type a sentence here, or leave blank to load a random example from a dataset...",
                label="Input Sentence:",
                lines=3
            )


            # Dataset selection for ATEPC examples restricted to desired list
            atepc_dataset_selection = gr.Radio(
                choices=DESIRED_ATEPC_DATASETS, # Use the predefined list
                value=DESIRED_ATEPC_DATASETS[0] if DESIRED_ATEPC_DATASETS else None, # Set default to first or None
                label="Select Dataset (for random examples):",
                interactive=True
            )

            # Button to trigger ATEPC inference
            atepc_run_button = gr.Button("Analyze Aspects!", variant="primary")

            # Output areas for ATEPC
            atepc_output_sentence = gr.TextArea(label="Analyzed Sentence:", interactive=False)
            atepc_prediction_results_df = gr.DataFrame(label="Aspect Prediction Results:", interactive=False)

            # Define the interaction for the ATEPC button click
            atepc_run_button.click(
                fn=run_atepc_inference,
                inputs=[atepc_input_box, atepc_dataset_selection],
                outputs=[atepc_prediction_results_df, atepc_output_sentence],
                api_name="run_atepc_inference"
            )

    gr.Markdown("---") # Visual separator


# Launch the Gradio application
if __name__ == "__main__":
    if aspect_extractor is None:
        print("Warning: PyABSA ATEPC model failed to initialize. The application may not function correctly.")
    sentiment_analysis_app.launch(share=False, debug=True)

