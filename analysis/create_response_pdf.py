import pandas as pd
from fpdf import FPDF
from base_utils import read_model_configs, create_all_agent_response_dataframe, find_correct_mc, find_min_max_scaled_error, prepare_response_dataframe, prepare_original_raw_response_dataframe

def prepare_dataframe(test_type):
    model_configs = read_model_configs()
    model_responses = create_all_agent_response_dataframe(
        test_name=test_type, 
        model_configs=model_configs,
        print_stats=True
    )
    if test_type in ['ggr', 'vlat', 'calvi-trick', 'calvi-standard']:
        model_responses = find_correct_mc(model_responses_raw=model_responses)
        model_responses = model_responses.groupby(['agent_type', 'question', 'image_file',])['is_correct'].mean().reset_index()
    else:
        model_responses = find_min_max_scaled_error(
            model_responses_raw=model_responses, 
            test_name=test_type,
            units_of_measure="is_correct"
        )
        model_responses = model_responses.dropna(subset=['is_correct'])
        model_responses = model_responses.groupby(['agent_type', 'question', 'image_file',])['is_correct'].median().reset_index()

    # model_responses = model_responses[model_responses['is_correct'] < 1.6]
    # model_responses = model_responses.dropna(subset=['agent_response'])
    
    model_responses = model_responses.sort_values(by=['question', 'image_file'])
    
    # = model_responses.loc[model_responses.groupby(['agent_type', 'question', 'image_file',])['is_correct'].idxmin()]

    return model_responses
filtered_df = agent_responses[agent_responses['test_type'] == 'vlat'].drop_duplicates()
filtered_df = filtered_df.sort_values(by="Model",ascending=True)
pdf = FPDF()

# Loop through each row in the filtered DataFrame
for idx, row in filtered_df.iterrows():
    # Add a new page
    pdf.add_page()

    # Set the font for the question
    pdf.set_font("Arial", style="B", size=12)
    pdf.multi_cell(0, 10, f"Question: {row['question']} (Image File: {row['image_file']})")
    pdf.set_font("Arial", size=12) 

    selected_df = all_raw_responses_df[
        (all_raw_responses_df['question'] == row['question']) &
        (all_raw_responses_df['image_file'] == row['image_file']) &
        (all_raw_responses_df['agent_type'] == 'GPT-4V')
    ][['agent_response']].value_counts().reset_index()

    pdf.multi_cell(0, 10, f"Correct Answer: {row['correct_answer']}")
    pdf.multi_cell(0, 10, f"Avg. prop. correct for Humans: {round(row['Human'], 2)}; across all Models: {round(row['Model'], 2)}")
    pdf.multi_cell(0, 10, f"GPT-4V:")
    
    for i2, row2 in selected_df.iterrows():
        pdf.multi_cell(0, 10, f"---------------------------------")
        pdf.multi_cell(0, 10, f"{row2['agent_response']} ({row2['count']})")
        if pdf.get_y() > 200:
            pdf.add_page()

    # Get current y position after question and answer
    pdf.multi_cell(0, 10, f"---------------------------------")
    y_position = pdf.get_y()
    
    # Add the image at the next line after answer
    pdf.image(row['image_link'], x=10, y=y_position + 5, w=180)

# Save the PDF
pdf_output_path = "./output_vlat.pdf"
pdf.output(pdf_output_path)