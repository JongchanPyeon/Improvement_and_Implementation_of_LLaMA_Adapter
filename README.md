# This work is a replication of the work  [LLaMA-Adapter]([https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation](https://github.com/OpenGVLab/LLaMA-Adapter)).

## 1. Run the cells in "training_replicated.ipynb" step-by-step for the model training
- check [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter).
- After download "LLaMA-7B", make sure that the folder has this structure.
  ```
  /path/to/LLaMA-7B
  ├── 7B
  │   ├── checklist.chk
  │   ├── consolidated.00.pth
  │   └── params.json
  └── tokenizer.model
  ```


## 2. Evaluation
- check [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter).
- Run "extract_adapter_replicated.ipynb" to extract the trained adapter.
- Download MME datasets and `eval_tool` from the [MME repo](https://github.com/bradyfu/awesome-multimodal-large-language-models#our-mllm-works), and put them under `MME_Benchmark_release_version`. Now the folder structure will be:
    ```
    MME_Benchmark_release_version
        ├── artwork
        ├── celebrity
        ├── code_reasoning
        ├── color
        ├── commonsense_reasoning
        ├── count
        ├── eval_tool
        │   ├── calculation.py
        │   ├── LaVIN
        │   └── Your_Results
        ├── existence
        ├── landmark
        ├── numerical_calculation
        ├── OCR
        ├── position
        ├── posters
        ├── scene
        └── text_translation

- Run "evaluation_replicated.ipynb"


## 3. Results 
- check > [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter).

* **LLaMA-Adapter V2.1**

    ```
    =========== Perception ===========
    total score: 1326.0875953396435 

            existence  score: 185.0
            count  score: 133.33333333333331
            position  score: 56.666666666666664
            color  score: 118.33333333333334
            posters  score: 147.9591836734694
            celebrity  score: 134.70588235294116
            scene  score: 156.25
            landmark  score: 167.8391959798995
            artwork  score: 123.5
            OCR  score: 102.5


    =========== Cognition ===========
    total score: 356.42857142857144 

            commonsense_reasoning  score: 106.42857142857144
            numerical_calculation  score: 47.5
            text_translation  score: 112.5
            code_reasoning  score: 90.0
