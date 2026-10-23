# Bird species classifier (CUB-200)

COS30082 Applied Machine Learning, Assignment 1.

A Streamlit web app that identifies a bird photo as one of the 200 species of the Caltech-UCSD Birds 200 dataset. It uses a ConvNeXt-Tiny CNN fine-tuned in the training notebook (88.9% top-1 test accuracy) and shows a Grad-CAM heat map of where the model looked.

## Run locally
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files
* `streamlit_app.py`: the web page
* `model_def.py`: rebuilds the network (timm ConvNeXt-Tiny / custom CNN)
* `gradcam.py`: Grad-CAM explanation
* `assets/`: exported model weights (float16), class names, test predictions, reference and example photos
