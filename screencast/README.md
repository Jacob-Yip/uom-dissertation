# Screencast using Manim
### This file contains information of the project of building the screencast video via Manim
##### Created by w45242hy


## Note
- 


## Set Up
1. Create folder ```screencast/``` for screencast using Manim
    - Do at root level
    ```sh
    uv init screencast/
    ```
2. Install dependencies
    - TODO: To be confirmed
    - Do it in ```screencast/```
    ```sh
    uv add manim
    ```
3. Activate environment for Manim
    ```sh
    .venv/Scripts/activate.ps1
    ```
4. Install Manim via pip
    - This allows code to run too
    - TODO: To be confirmed
    ```sh
    pip install manim
    ```
5. Check if Manim is installed correctly
    ```sh
    uv run manim checkhealth
    ```


## Run
- Low quality
    - ```480p15```
    ```sh
    manim -pql <file_path> <class_name>
    ```
- Medium quality
    - ```720p30```
    ```sh
    manim -pqm <file_path> <class_name>
    ```
- High quality
    - ```1080p60```
    ```sh
    manim -pqh <file_path> <class_name>
    ```
- 4K quality
    - ```2160p60```
    ```sh
    manim -pqk <file_path> <class_name>
    ```