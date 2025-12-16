#  Subject: Advanced AI PROJECT 2025

Prof: Hyung–Jeong Yang

## Task: Development of a Special-Language Translation AI Model Using Sovereign AI Open-Source LLM

Name: Kim Ngan Phan
⛔ ***Only used for demonstration of the subject***
### Project Structure
* Json data is collect from National Institute of Korean Language’s Language Resource Sharing Platform
* Create the following directories to store the original JSON files:

  * **Advanced AI Project/**

### Training the Model

1. Split the dataset into training, validation, and test sets with a ratio of 8:1:1, then save them in the Advanced AI Project/ directory.
2. Run the following command to train the model using the preprocessed data. The trained model will be saved in `skt-ax3.1light-kor-vie-qlora/` and `skt-ax3.1light-vie-kor-qlora/` directories:

   ```
   python train-kor2vie.py
   ```
   ```
   python train-vie2kor.py
   ```   

3. Run the following command to generate translation outputs using the trained models:

   ```
   python inference_kor2vie.py
   ```
   ```
   python inference_vie2kor.py
   ```
4. Run the following command to evaluate the performance of both models:

   ```
   python evaluate.py
   ```
4. To launch the application interface locally, run:

   ```
   streamlit run interface_main.py
   ```
