# uom-dissertation
The repository for the third year project in UoM
### This folder contains code for ensemble learning
##### Created by w45242hy


## Note
- After generating a Python script from the Jupternotebook, remember to remove ```!pip install -r requirements.txt``` and add ```if __name__ == "__main__"``` to the ```.py``` file


## Run
- Make sure you run it at the **root folder**
    - ```sys.path``` for Python **import** is set up correctly
    - Because of the absence of Python package, you cannot use **relative import**
    - *Module names* will always be from **root folder** even if you are just importing modules in the same folder, i.e. format is always like ```<path_from_root>.<module_name>```
- Make sure you make the changes below to ```neat-python``` code
    - **IMPORTANT**: This is a bug from their code
    - ~~Change ```config.compatibility_disjoint_coefficient``` to ```config.genome_config.compatibility_disjoint_coefficient```~~
        - In ```genomes.py```
    - ~~Change ```config.compatibility_weight_coefficient``` to ```config.genome_config.compatibility_weight_coefficient```~~
        - In ```genes.py```
- Recommended to have a Python virtual environment
    - ### Linux/macOS
        ```sh
        # Create a virtual environment, which will be represented by a folder in the current directory
        python -m venv <virtual_environment_name>

        # Activate virtual environment
        # Linux/macOS
        source <virtual_environment_name>/bin/activate
        
        # Deactivate virtual environment to exit
        deactivate
        ```
    - ### Windows
        ```ps
        # Create a virtual environment, which will be represented by a folder in the current directory
        python -m venv <virtual_environment_name>

        # Activate virtual environment
        # Windows
        .\<virtual_environment_name>\Scripts\activate
        
        # Deactivate virtual environment to exit
        deactivate
        ```
- Error message
    > module 'torch' has no attribute 'accelerator'
    - ### Cause
        - Your PyTorch library is out-dated
    - ### Solution
        - Execute the commands below
            ```sh
            pip uninstall torch torchvision torchaudio

            # Then install PyTorch from the official website
            ```
        - Or just change the Python line of code that is not working
            ```python
            # Original
            device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

            # Updated
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    > ERROR: Could not find a version that satisfies the requirement pytorch-neat (from versions: none)
    > ERROR: No matching distribution found for pytorch-neat
    - ### Note
        - This is **deprecated** as we did not use the library at the end
        - Best to remove it to avoid name collisions when installing future packages
    - ### Cause
        - PyTorch-NEAT library is not supported by PyPI yet
    - ### Solution
        - ~~Install library manually~~
        - If you are using a virtual environment, you should use ```site-packages/``` located inside ```<virtual_environment_folder_path>/``` instead of your global ```site-packages/```
        - Execute the commands below
            ```sh
            git clone https://github.com/uber-research/PyTorch-NEAT.git
            cd PyTorch-NEAT
            cp -r pytorch_neat/ <path_to_python_site_packages>
            ```
        - Find path to Python site-packages
            ```sh
            python -m site
            ```
    > ERROR: Import "pytorch_neat.feedforward_net" could not be resolved
    - ### Note
        - This is **deprecated** as we did not use the library at the end
        - ~~Need to rename ```neat/``` to a new folder name as ```neat/``` is already reserved by the official ```neat/``` of *Python-NEAT*~~
            - This will mess up imports in the library
        - ~~Create a folder ```custom/``` that stores this library~~
            - This will not affect imports in the library while allowing you to address different NEAT libraries
        - I decide to use my own library because this library is deprecated
            - #### Example
                - It tries to reference the old variables in ```DefaultGenome```
            - Too much pain in the ass to resolve it
            - Besides, I have already created my own library for this project
    - ### Cause
        - ```from pytorch_neat.feedforward_net import FeedForwardNet```
        - Computer suggestions had a hallucination that a package called ```feedforward_net``` exists when in reality it does not
    - ### Solution
        - Install the correct library manually
        - If you are using a virtual environment, you should use ```site-packages/``` located inside ```<virtual_environment_folder_path>/``` instead of your global ```site-packages/```
        - Execute the commands below
            ```sh
            git clone https://github.com/ddehueck/pytorch-neat.git
            cd pytorch-neat/
            mv neat/ pytorch_neat/
            cp -r pytorch_neat/ <path_to_python_site_packages>
            ```
        - Find path to Python site-packages
            ```sh
            python -m site
            ```


## PyTorch
- ### ```nn.Linear```
    - Run the code below to understand how ```nn.Linear``` works
    ```python
    import torch
    from torch import nn

    m = nn.Linear(2, 3)

    with torch.no_grad():
        m.weight.copy_(torch.tensor([[1.0, 2.0], 
                                    [3.0, 4.0], 
                                    [5.0, 6.0]]))
        m.bias.copy_(torch.tensor([1.0, 0.0, 0.0]))


    print(f"m.weight: {m.weight}; size: {m.weight.shape}")
    print(f"m.bias: {m.bias}; size: {m.bias.shape}")

    input_data = torch.tensor([[1, 1], 
                            [10, 10], 
                            [100, 100], 
                            [1000, 1000]], dtype=torch.float)

    print(f"input_data: {input_data}; size: {input_data.shape}")

    output_data = m(input_data)

    print(f"output_data: {output_data}; size: {output_data.shape}")
    ```
