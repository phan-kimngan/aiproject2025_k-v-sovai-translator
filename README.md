# aiproject2025_k-v-sovai-translator

### Project Structure
* Json data is collect from National Institute of Korean Language’s Language Resource Sharing Platform
* Create the following directories to store the original JSON files:

  * **Advanced AI Project/**

### Training the Model

1. Slipt the data with ratio of 8:1:1 and save the results in the `Advanced AI Project/ folder
2. Run the following command to train the model using the preprocessed data. The trained model will be saved in `skt-ax3.1light-kor-vie-qlora/` and `skt-ax3.1light-vie-kor-qlora/` directories:

   ```
   python train-kor2vie.py
   ```
   ```
   python train-vie2kor.py
   ```   

3. Run the following command to get predictions of `skt-ax3.1light-kor-vie-qlora/` and `skt-ax3.1light-vie-kor-qlora/` model:

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
