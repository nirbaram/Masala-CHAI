from PIL import Image
import os
import numpy as np
from main import main
import matplotlib.pyplot as plt
import argparse
from openai import OpenAI 
import anthropic
import base64
import re
import json
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU with Intel optimizations where possible")
    # Basic CPU optimizations
    try:
        torch.set_num_threads(os.cpu_count())
        print(f"PyTorch using {torch.get_num_threads()} CPU threads")
    except:
        print("Could not optimize thread count")

def run_gpt(this_messages, this_model="o3-mini", api_key="abc"):  # push to gpt
    client = OpenAI(
        api_key=api_key, 
    )
    chat_completion = client.chat.completions.create(messages=this_messages, model=this_model)
    return chat_completion

def run_claude(this_messages, this_model="claude-3-7-sonnet", api_key="abc"):  # push to claude
    client = anthropic.Anthropic(
        api_key=api_key,
    )
    
    # Convert OpenAI format messages to Anthropic format
    anthropic_messages = []
    for msg in this_messages:
        role = msg["role"]
        if role == "system":
            # For system messages, we'll add at the beginning of the first user message
            system_content = msg["content"]
            continue
        
        # For content that contains images, we need to handle differently
        if role == "user" and isinstance(msg["content"], list):
            text_parts = []
            image_parts = []
            
            for content_item in msg["content"]:
                if content_item["type"] == "text":
                    text_parts.append(content_item["text"])
                elif content_item["type"] == "image_url":
                    # Extract base64 data from data URL
                    img_url = content_item["image_url"]["url"]
                    if img_url.startswith("data:image/"):
                        # Extract base64 data
                        img_data = img_url.split(",", 1)[1]
                        image_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_data
                            }
                        })
            
            # Build content array with system message at the beginning if exists
            content = []
            if 'system_content' in locals():
                content.append({
                    "type": "text",
                    "text": system_content
                })
                del system_content  # Used it, now remove it
                
            if text_parts:
                content.append({
                    "type": "text",
                    "text": "\n".join(text_parts)
                })
                
            # Add image parts
            content.extend(image_parts)
            
            anthropic_messages.append({
                "role": role,
                "content": content
            })
        else:
            # Handle regular text messages
            if role == "user" and 'system_content' in locals():
                # Prepend system message to first user message
                anthropic_messages.append({
                    "role": role,
                    "content": [
                        {"type": "text", "text": system_content + "\n\n" + msg["content"]}
                    ]
                })
                del system_content  # Used it, now remove it
            else:
                anthropic_messages.append({
                    "role": role,
                    "content": [
                        {"type": "text", "text": msg["content"]}
                    ]
                })
    
    # Claude API call
    response = client.messages.create(
        model=this_model,
        messages=anthropic_messages,
        max_tokens=4096
    )
    
    # Create a structure similar to OpenAI response
    return type('obj', (object,), {
        'choices': [
            type('obj', (object,), {
                'message': type('obj', (object,), {
                    'content': response.content[0].text
                })
            })
        ]
    })

def process_with_ai(image_link1, image_link2, prompt, model_provider="openai", model_name=None, openai_api_key="abc", anthropic_api_key="abc", retry=2, verbose=True):
    """Generic function to process with either OpenAI or Anthropic"""
    
    # Set default model names based on provider
    if model_name is None:
        if model_provider == "openai":
            model_name = "gpt-4o"
        else:  # anthropic
            model_name = "claude-3-opus-20240229"
    
    # System role message
    if model_provider == "openai":
        role = "You are an expert in analog design. Your task is to provide the spice netlist for the image provided. You have to follow the instructions provided strictly."
    else:
        role = "You are an expert in analog circuit design with deep knowledge of SPICE netlists, component identification, and circuit analysis. Follow the instructions precisely."
    
    # Format content based on whether we're using text or images
    new_content = [{"type": "text", "text": prompt},
                  {"type": "image_url", "image_url": {"url": image_link1}},
                  {"type": "image_url", "image_url": {"url": image_link2}}]
    
    message_hist = [{"role": "system", "content": role},
                   {"role": "user", "content": new_content}]
    
    if verbose:
        print(f"Using {model_provider} with model {model_name}")
        print("Prompt:", prompt)

    # Try to get a response with retries
    attempts = 0
    response = ""
    while response == "" and attempts < retry:
        try:
            if model_provider == "openai":
                response = run_gpt(message_hist, model_name, openai_api_key).choices[0].message.content
            else:  # anthropic
                response = run_claude(message_hist, model_name, anthropic_api_key).choices[0].message.content
        except Exception as e:
            attempts += 1
            if verbose: 
                print(f"Attempt {attempts} failed with error:", e)
    
    if verbose: 
        print("\nResponse received, length:", len(response))
    
    return response

# helpers
def encode_image(image_path):  # Function to encode the image
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Prompts
prompt_netlist = "You are provided with 2 schematics, one that is the original image with no components detected and no net annotation. The second image has some component detected and some node annotated. There would be some text removed from the annotated image that is present in original image, use that also. Your task is to write spice netlist for the schematic treating the first image as golden and second one as helper to guide in the spice generation.the nets are highlighted in red to help identify the connections between the components. Your task is to identify each terminal of the components and map it to the correct net highlighted in red. Follow the below instructions: 1. The first task is to list all the transistor components, current sources, capacitors, inductors and voltage sources which you can observe from the figure. 2. NOTE MOSFET are 3 terminal device with (drain, gate, source) and body terminal connected to source, and Current/Voltage source are 2 terminal device. You have obtained the list of components from the previous step. 3. For NMOS, the arrow on the source terminal points outwards from the transistor. For PMOS, the arrow on the source terminal points inward toward the transistor."

# New prompt for generating a search-friendly description
prompt_description = """Analyze these circuit schematics and provide a detailed but concise description in the following format:

1. First line: A brief one-sentence summary of what type of circuit this is (e.g., "Two-stage CMOS operational amplifier with current mirror load").

2. Second paragraph: Describe the circuit's main functionality and purpose in 2-3 sentences. What does it do? What application is it used for?

3. Third paragraph: Describe the main components and their arrangement (e.g., "The circuit uses 6 MOSFETs, including a differential pair (M1-M2) at the input stage, followed by...").

4. Final paragraph: Mention any notable design features or characteristics (e.g., "Features a compensation capacitor for stability" or "Includes a temperature-independent biasing circuit").

Keep the description informative but concise, focusing on what someone searching for this type of circuit would want to know. Use terminology that would appear in search queries.
"""

def extract_code(image_path1, image_path2, model_provider="openai", model_name=None, verbose=True, openai_api_key="abc", anthropic_api_key="abc"):
    """ takes in the link of a local image and gets spice code from it """
    
    base64_image = encode_image(image_path1)
    image_link1 = f"data:image/png;base64,{base64_image}"

    base64_image = encode_image(image_path2)
    image_link2 = f"data:image/png;base64,{base64_image}"

    raw_code = process_with_ai(
        image_link1, 
        image_link2, 
        prompt_netlist, 
        model_provider=model_provider, 
        model_name=model_name, 
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        verbose=verbose
    )
    
    # Extract code between triple backticks if present
    code_match = re.search(r'```(.*?)```', raw_code, re.DOTALL)
    if code_match:
        extracted_text = code_match.group(1)
        # Remove language identifier if present (e.g., ```spice)
        if extracted_text.split('\n', 1)[0].strip() and not extracted_text.split('\n', 1)[0].strip()[0] in '*.$':
            extracted_text = extracted_text.split('\n', 1)[1]
        return extracted_text.strip()
    else:
        # If no code blocks found, return the raw text
        return raw_code.strip()

def generate_description(image_path1, image_path2, components_list, nodes_list, connections_list, stats_dict, model_provider="openai", model_name=None, verbose=True, openai_api_key="abc", anthropic_api_key="abc"):
    """Generate a search-friendly description of the circuit"""
    
    base64_image = encode_image(image_path1)
    image_link1 = f"data:image/png;base64,{base64_image}"

    base64_image = encode_image(image_path2)
    image_link2 = f"data:image/png;base64,{base64_image}"

    # Create a more specific prompt with the component information
    components_str = "\n".join(components_list)
    connections_str = "\n".join(connections_list)
    
    # Add component and connection information to the prompt
    enhanced_prompt = f"{prompt_description}\n\nHere's additional information about the circuit to help you:\n\nComponents:\n{components_str}\n\nConnections:\n{connections_str}"
    
    # If we have statistics, include them as well
    if stats_dict:
        stats_str = json.dumps(stats_dict, indent=2)
        enhanced_prompt += f"\n\nCircuit Statistics:\n{stats_str}"
    
    description = process_with_ai(
        image_link1, 
        image_link2, 
        enhanced_prompt, 
        model_provider=model_provider, 
        model_name=model_name, 
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        verbose=verbose
    )
    
    return description.strip()

def save_image_as_figure(image, caption, save_path):
    """
    Save the given image as a figure with the specified caption.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.title(caption)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def save_description(description, save_path):
    """
    Save the description to a text file.
    """
    with open(save_path, 'w') as file:
        for line in description:
            file.write(f"{line}\n")

def save_text(text, save_path):
    """
    Save text to a file.
    """
    with open(save_path, 'w') as file:
        file.write(text)

def process_image(image_path, output_dir, model_provider, model_name, openai_api_key, anthropic_api_key):
    """
    Process a single image and save the results in the output directory.
    """
    print(f"Processing {image_path} using {model_provider} API...")

    # Create an output directory for the image
    im = os.path.splitext(os.path.basename(image_path))
    image_name = im[0]+''.join(im[1:]).replace(".","")
    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    # Run the whole flow
    img = Image.open(image_path)
    save_image_as_figure(img, '', os.path.join(image_output_dir, 'scanned_circuit.png'))
    inp = np.array(img)
    
    rebuilt, comp, nodes, _, og_image_with_line_labels, comdetected_image_with_line_labels, comp_list, jns_list, conn_list, sample_dict = main(image_output_dir, inp, image_name)

    # Save images
    save_image_as_figure(comp, '', os.path.join(image_output_dir, 'detected_components.png'))
    save_image_as_figure(nodes, '', os.path.join(image_output_dir, 'nodes_terminals.png'))
    save_image_as_figure(rebuilt, '', os.path.join(image_output_dir, 'rebuilt_circuit.png'))

    img1 = Image.fromarray(og_image_with_line_labels)
    img2 = Image.fromarray(comdetected_image_with_line_labels)
    img1.save(os.path.join(image_output_dir, 'original_withLineLabels.png'))
    img2.save(os.path.join(image_output_dir, 'original_withComponentsAndLineLabels.png'))

    # Save descriptions
    save_description(comp_list, os.path.join(image_output_dir, 'components_description.txt'))
    save_description(jns_list, os.path.join(image_output_dir, 'nodes_description.txt'))
    save_description(conn_list, os.path.join(image_output_dir, 'connections_description.txt'))

    # Extract SPICE code
    code = extract_code(
        os.path.join(image_output_dir, 'scanned_circuit.png'), 
        os.path.join(image_output_dir, 'original_withComponentsAndLineLabels.png'),
        model_provider=model_provider,
        model_name=model_name,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key
    )

    save_text(code, os.path.join(image_output_dir, 'spice.txt'))

    # Generate circuit description for search
    search_description = generate_description(
        os.path.join(image_output_dir, 'scanned_circuit.png'), 
        os.path.join(image_output_dir, 'original_withComponentsAndLineLabels.png'),
        comp_list,
        jns_list,
        conn_list,
        sample_dict,
        model_provider=model_provider,
        model_name=model_name,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key
    )
    
    # Save the search-friendly description
    save_text(search_description, os.path.join(image_output_dir, 'circuit_description.txt'))
    
    # Generate an output summary for easier integration with the search app
    output_summary = f"Components in the circuit are: \n"
    for comp in comp_list:
        output_summary += f"{comp}\n"
    
    output_summary += f"\nJunctions in the circuit are: \n"
    for jn in jns_list:
        output_summary += f"{jn}\n"
        
    output_summary += f"\nConnections in the circuit are: \n"
    for conn in conn_list:
        output_summary += f"{conn}\n"
    
    save_text(output_summary, os.path.join(image_output_dir, 'output.txt'))

    # Save sample dict
    with open(os.path.join(image_output_dir, 'sample_statistics.json'), 'w') as json_file:
        json.dump(sample_dict, json_file, indent=4)  # 'indent' for pretty printing

    print(f"Results saved in {image_output_dir}")

def app(directory_path, target_path, model_provider, model_name, openai_api_key, anthropic_api_key):
    """
    Function to showcase the implementation and save outputs for each image in a directory.
    """
    print(f"Circuit recognition using {model_provider} API")

    if not os.path.exists(directory_path):
        print("Directory not found or invalid path provided.")
        return

    output_dir = target_path
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(directory_path, filename)
            try:
                process_image(image_path, output_dir, model_provider, model_name, openai_api_key, anthropic_api_key)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    print("All images have been processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process circuit images from a directory.")
    parser.add_argument('--src', type=str, required=True, help="Path to the directory containing images")
    parser.add_argument('--tgt', type=str, required=True, help="Path to the output directory")
    parser.add_argument('--provider', type=str, default="openai", choices=["openai", "anthropic"], help="AI provider to use (openai or anthropic)")
    parser.add_argument('--model', type=str, help="Model name to use (defaults to gpt-4o for OpenAI, claude-3-opus-20240229 for Anthropic)")
    parser.add_argument('--openai_api_key', type=str, required=True, help="API Key for the selected provider")
    parser.add_argument('--anthropic_api_key', type=str, required=True, help="API Key for the selected provider")
    
    args = parser.parse_args()
    directory_path = args.src
    target_path = args.tgt
    model_provider = args.provider
    model_name = args.model
    openai_api_key = args.openai_api_key
    anthropic_api_key = args.anthropic_api_key
    
    app(directory_path, target_path, model_provider, model_name, openai_api_key, anthropic_api_key)