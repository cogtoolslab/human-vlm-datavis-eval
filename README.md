# Human-Centered Evaluation of Data Visualization Understanding in Vision-Language Models

This repository contains code to reproduce the results in our XXXX paper, Human-Centered Evaluation of Data Visualization Understanding in Vision-Language Models.

To reproduce the results in the CogSci 2024 version of this project: [*Evaluating human and machine understanding of data visualizations*](/paper/vlm_datavis_benchmark_corrected_cogsci2024.pdf), please visit https://github.com/cogtoolslab/human-machine-datavis-cogsci/tree/master.

Directory Structure

```bash
├── admin
├── analysis
├── data
├── experiments
│   ├── api  
│   ├── model_inference
├── paper
├── results
│   ├── dataframe
│   ├── figures
├── stimuli
│   ├── ggr
│   ├── vlat
│   ├── holf
```

Each folder contains a README.md file that elaborates further on its contents. Below are the general descriptions of each folder:

`admin` contains contributions.md which describes author contributions

`analysis` contains all python scripts and notebooks used to  calculate statistics and generate figures reported in the paper.
- `human_machine_performance.ipynb`: Human and machine performance figure
- `feature_pairwise_performance.ipynb`: Pairwise comparison between human and machien eprformacne
- `human_machine_correlation.ipynb`: Human vs machine correlation figure
- `process_valid_model_responses.ipynb`: process and extract responses from GPT-4 processed responses

`data` contains instructions on how to download the data model and human responses to all items.

`experiments` contains the server api code used to save model responses during evaluations and the code to evaluate vision-language models. 

`paper` contains the pdfs for the orginial and corrected version our paper.

`results` contains the dataframes (csv files) and unedited figures for all plots in the paper.

`stimuli` contains the test items and instructions given to humans and machines.

BibTeX Citation for CogSci 2024 paper:
```
@inproceedings{verma2024evaluating,
  title={Evaluating human and machine understanding of data visualizations},
  author={Verma, Arnav and Mukherjee, Kushin and Potts, Christopher and Kreiss, Elisa and Fan, Judith E},
  booktitle={Proceedings of the Annual Meeting of the Cognitive Science Society},
  volume={46},
  year={2024}
}
```
