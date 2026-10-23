"""Bird species classifier: Streamlit web app (deployed on Streamlit Community Cloud).

Uses the deployed model exported by Section 12 of the training notebook (ConvNeXt-Tiny,
384 px, fine-tuned on CUB-200) to name a bird photo as one of 200 species, and shows a
Grad-CAM heat map of where the model looked.

Run locally:  streamlit run streamlit_app.py
"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps

from gradcam import grad_cam, overlay_cam
from model_def import load_model

# ----------------------------- edit before publishing -----------------------------
AUTHOR = 'Daniel Lin Chen Xu (102780139)'
VIDEO_URL = ''           # YouTube presentation link
REPORT_URL = ''          # link to the PDF report (optional)
# ----------------------------------------------------------------------------------

ASSETS = Path(__file__).parent / 'assets'
CFG = json.loads((ASSETS / 'model_config.json').read_text())
KEY = CFG['default_model']                     # the deployed (best) model
MODEL_INFO = CFG['models'][KEY]
CLASS_NAMES = json.loads((ASSETS / 'class_names.json').read_text())
MEAN, STD, CROP = CFG['mean'], CFG['std'], CFG.get('crop_ratio', 0.875)
LOW_CONFIDENCE = 0.30                          # below this we warn that the photo may be out of scope

st.set_page_config(page_title='Bird Species Classifier', page_icon='🐦', layout='centered')


# ===================================== model =====================================
@st.cache_resource(show_spinner='Loading the model...')
def get_model():
    """Load the network once per server process (shared by all visitors)."""
    return load_model(MODEL_INFO['arch'], ASSETS / MODEL_INFO['weights'], CFG['num_classes'])


def eval_transform(size):
    """Exactly the notebook's evaluation pre-processing: resize, centre crop, ImageNet normalisation."""
    return T.Compose([T.Resize(round(size / CROP)), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])


def display_crop(img, size=384):
    """The region of the photo the model actually sees (for display and the Grad-CAM overlay)."""
    crop = T.Compose([T.Resize(round(size / CROP)), T.CenterCrop(size)])(img)
    return np.asarray(crop, dtype=np.float32) / 255


@st.cache_data(show_spinner='Identifying the bird...', max_entries=32)
def classify(image_bytes, tta):
    """-> (200 class probabilities, displayed crop, Grad-CAM overlay). Cached per photo."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert('RGB')
    model, size = get_model(), MODEL_INFO['img_size']
    cam, logits = grad_cam(model, eval_transform(size)(img)[None])
    probs = logits.softmax(1)[0]
    if tta:   # test-time augmentation: average over 2 scales x {original, mirrored}
        views = []
        for s in (size, int(round(size * 1.15 / 32) * 32)):
            x = eval_transform(s)(img)[None]
            views += [x, torch.flip(x, dims=[3])]
        with torch.no_grad():
            probs = torch.stack([model(v).softmax(1)[0] for v in views]).mean(0)
    shown = display_crop(img)
    return probs.numpy(), (shown * 255).astype(np.uint8), overlay_cam(shown, cam[0])


@st.cache_data
def species_record():
    """Test-set track record of the deployed model per species: (correct, total, most-confused-with)."""
    df = pd.read_csv(ASSETS / 'test_predictions.csv')
    col, record = f'{KEY}_pred', {}
    for label, g in df.groupby('label'):
        wrong = g.loc[g[col] != label, col]
        record[int(label)] = (int((g[col] == label).sum()), len(g), int(wrong.mode()[0]) if len(wrong) else None)
    return record


def about():
    with st.expander('How it works'):
        st.markdown(f"""
1. **Pre-processing.** The photo is resized and centre-cropped to {MODEL_INFO['img_size']} px and normalised exactly as in training.
2. **Classification.** A ConvNeXt-Tiny convolutional neural network, pretrained on ImageNet and fine-tuned on about 4,800 CUB-200 photos, outputs a probability for each of the 200 species. Overfitting was controlled with strong data augmentation, Mixup/CutMix, label smoothing, weight decay, stochastic depth, weight averaging (EMA) and early stopping.
3. **Explanation.** Grad-CAM highlights the regions that drove the prediction. A trustworthy prediction focuses on the bird, not the background.

**Test-set results** (1,204 unseen photos): top-1 accuracy **{MODEL_INFO['test_top1']:.1%}**, average per-class accuracy **{MODEL_INFO['test_avg_per_class']:.1%}**, top-5 accuracy **{MODEL_INFO['test_top5']:.1%}**.

**Limitations.** It only knows the 200 CUB species (mostly North American). Any other bird or object is still mapped to one of them, so check the confidence. It works best when the bird is large and in focus.

**Author:** {AUTHOR}""" + ''.join(f' · [{t}]({u})' for t, u in [('Video', VIDEO_URL), ('Report', REPORT_URL)] if u))


# ===================================== page =====================================
st.title('Bird species classifier')
st.caption(f'COS30082 Applied Machine Learning · Assignment 1 · {AUTHOR}')
st.write('Upload a photo of a bird and a convolutional neural network will name it among **200 species** '
         'from the Caltech-UCSD Birds dataset, show how confident it is and where it looked.')

m1, m2, m3 = st.columns(3)
m1.metric('Top-1 test accuracy', f"{MODEL_INFO['test_top1']:.1%}", border=True)
m2.metric('Avg per-class accuracy', f"{MODEL_INFO['test_avg_per_class']:.1%}", border=True)
m3.metric('Top-5 test accuracy', f"{MODEL_INFO['test_top5']:.1%}", border=True)

source = st.segmented_control('Photo', ['Upload a photo', 'Try an example'], default='Upload a photo',
                              label_visibility='collapsed')
image_bytes = None
if source == 'Try an example':
    examples = {p.stem.rsplit('_', 1)[0].replace('_', ' '): p for p in sorted((ASSETS / 'examples').glob('*.jpg'))}
    choice = st.selectbox('Example photo from the test set', list(examples))
    image_bytes = examples[choice].read_bytes()
else:
    upload = st.file_uploader('Upload a bird photo (JPG, PNG or WEBP)', type=['jpg', 'jpeg', 'png', 'webp'])
    if upload is not None:
        image_bytes = upload.getvalue()
tta = st.toggle('Higher-accuracy mode (averages 4 views of the photo, about 4x slower)')

if image_bytes is None:
    st.info('Upload a photo or choose **Try an example** to get a prediction.')
    about()
    st.stop()

try:
    probs, crop, cam_img = classify(image_bytes, tta)
except Exception:
    st.error('Could not read that file as an image. Please try another photo.')
    st.stop()

order = np.argsort(-probs)
top, conf = int(order[0]), float(probs[order[0]])
name = CLASS_NAMES[top]

st.divider()
left, right = st.columns(2, gap='large')
with left:
    show_cam = st.toggle('Show where the model looked (Grad-CAM)', value=True)
    st.image(cam_img if show_cam else crop, width='stretch',
             caption='Bright, orange areas influenced the prediction most' if show_cam else 'What the model sees')
with right:
    st.caption('PREDICTION')
    st.subheader(name)
    st.progress(conf, text=f'{conf:.1%} confidence')
    if conf < LOW_CONFIDENCE:
        st.warning('Low confidence: the bird may be small or partly hidden, or not one of the 200 species '
                   'this model knows. Check the other candidates below.')
    correct, total, confused = species_record().get(top, (None, None, None))
    if total:
        st.write(f'On the test set this model recognised **{correct} of {total}** photos of this species'
                 + (f' and most often confuses it with **{CLASS_NAMES[confused]}**.' if confused is not None else '.'))
    st.markdown(f"[Read about the {name} on Wikipedia ↗](https://en.wikipedia.org/wiki/Special:Search?search="
                f"{name.replace(' ', '+')}+bird)")

st.markdown('##### Top 5 candidates')
st.dataframe(pd.DataFrame({'Species': [CLASS_NAMES[i] for i in order[:5]], 'Confidence': probs[order[:5]]}),
             hide_index=True, width='stretch',
             column_config={'Confidence': st.column_config.ProgressColumn('Confidence', format='percent',
                                                                          min_value=0.0, max_value=1.0)})

refs = [p for p in (ASSETS / 'reference' / f'{top:03d}_{i}.jpg' for i in range(3)) if p.exists()]
if refs:
    st.markdown(f'##### Reference photos of the {name}')
    for col, p in zip(st.columns(3), refs):
        col.image(str(p), width='stretch')

about()
