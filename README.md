# Auto-SPICE -- A Large Scale SPICE Netlist Dataset via Automated Netlist Extraction from Analog Circuit Diagrams

## Abstract
This work explores the use of Large Language Model (LLM) tools for automating netlist generation in analog circuits—one of the long-standing frontiers in circuit design automation. Analog circuit design requires careful attention to architecture, component selection, sizing, and connectivity, typically gained through years of experience. Automating this process could accelerate both design and verification, reduce dependence on specialized expertise, lower costs, and improve performance.

We identify key bottlenecks in automating analog circuit design and evaluate the multi-modal capabilities of state-of-the-art LLMs, particularly GPT-4o, to address these challenges. To overcome current limitations, we propose a workflow consisting of three steps: preprocessing, prompt tuning, and netlist verification. This workflow aims to create an end-to-end Simulation Program with Integrated Circuit Emphasis (SPICE) netlist generator from circuit schematic images.

Accurate netlist generation from schematics has been a major hurdle in automating analog design, and our approach demonstrates significant performance improvements. Tested on approximately ~1,000 schematic designs of varying complexity, our flow shows promising results. We plan to open-source the workflow to the community for further development.

## File and Folder Description

- `./hough/` : Folder containing scripts to use Hough Transform for net detection.
- `./models/` : Folder containing scripts for YOLOv8-based circuit component detection.
- `./sample-images/` : Folder containing sample images to run the Auto-SPICE netlist generator.
- `./trained_checkpoints/` : Contains checkpoint file for YOLOv8 model after training.
- `./utils/` : Supporting scripts for various components of the Auto-SPICE pipeline.
- `./Dataset/` : Folder containing dataset of the images with schematics.
     - This contains images from AMSNet repo as well.
     - Arranged across different data_* folder depending upon their sources.
- `main.py` : Main script that runs the entire pipeline.
- `run.py` : Script to be called for generating netlists for sample images.
- `environment.yml`: Requirements file for creating conda environment
- `visualize.ipynb`: Jupyter notebook for visualizing output of Autospice for a given circuit diagram

## Steps to Run the Framework

1. **Clone the repository and navigate into the repository:**

   ```bash
   git clone <repository_url>
   cd <repository_name>
2. **Create a Conda environment:**

   ```bash
   conda env create -f environment.yml
3. **Activate the Conda environment:**
	```bash
   conda activate autospice_env
4. **Add sample images:**
	Place your sample images in the `./sample-images/` folder.
5. **Run the pipeline:**
	```bash
	python run.py --src ./sample-images/ --tgt ./sample-output --api_key <openai_api_key>
	where - 
	- `--src` : Directory path to the sample images.
	- `--tgt` : Output directory path for the generated netlists.
	- `--api_key` : Your OpenAI API key for using GPT-4

## Output Files Description

For each sample circuit, the output consists of a number of files to help the user understand the output of various components in the pipeline:

1. **scanned_circuit.png**: Copy of the original circuit diagram.
2. **detected_components.png**, **component_removed_circuit.png**, **components_description.txt**: Output of the YOLOv8 component detection module:
   - `detected_components.png`: Components marked with bounding boxes.
   - `component_removed_circuit.png`: Components replaced with white spaces.
   - `components_description.txt`: Text file containing the description of the detected components.
3. **nodes_terminals.png**, **connections_descriptions.txt**, **nodes_description.txt**: Detected nodes in the circuit using Hough Transform:
   - `nodes_terminals.png`: Detected nodes in the circuit.
   - `connections_descriptions.txt`: Text file containing descriptions of various connections.
   - `nodes_description.txt`: Text file containing the description of various nodes in the circuit.
4. **text_and_comp_removed_circuit.png**: Original circuit diagram after removing all text content and detected circuit components.
5. **rebuilt_circuit.png**: Original circuit diagram overlaid with components and nodes.
6. **original_withComponentsAndLineLabels.png**, **original_withLineLabels.png**: Used for better visualization of the model output.
7. **sample_statistics.json**: Dictionary describing types of components in the circuit, along with node and net information.
8. **spice.txt**: Final generated SPICE netlist for the circuit diagram.

We also provide a helpful visualization of model output using a jupyer notebook: `visualize.ipynb`

## Citing Auto-SPICE

If you use Auto-SPICE in your research or project, please cite it using the following BibTeX entry:

```bibtex
@misc{autospice,
      title={Auto-SPICE: Leveraging LLMs for Dataset Creation via Automated SPICE Netlist Extraction from Analog Circuit Diagrams}, 
      author={Jitendra Bhandari and Vineet Bhat and Yuheng He and Siddharth Garg and Hamed Rahmani and Ramesh Karri},
      year={2024},
      eprint={2411.14299},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2411.14299}, 
}
